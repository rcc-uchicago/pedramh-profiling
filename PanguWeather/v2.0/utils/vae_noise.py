"""Noise-fixing hook for the VAE reparameterization draw (DESIGN.md §4.0).

Why this exists
---------------
`PanguModel_Plasim.reparameterize` draws `eps = torch.randn_like(std)` from the GLOBAL
RNG. That makes the model's output stochastic, which is correct science and fatal for the
DESIGN §4 equivalence gate: `torch.compile` / FlexAttention can change RNG **kernel
selection and consumption order**, so a *correct* optimization can still produce different
ensemble outputs. Comparing them would report a failure that isn't one — and under
CLAUDE.md #1/#11 the response to a mismatch is to trace it, never to widen the tolerance.
So the mismatch must not be manufactured in the first place.

**A seed alone does not fix this.** Seeding the global RNG only makes the *stream* start at
a known place; it does not stop a compiled graph from consuming that stream in a different
order, or from selecting a different RNG kernel. That is why this module exists alongside
`--seed` (which lives in s2s/v2.0/utils/seeding.py and is NOT yet ported here — the two
trees are forks that share code by copy, DESIGN §2c).

**Never compare a bitwise hash of a stochastic output.** Fix the noise, or compare a
deterministic pre-sample quantity (`mu`/`sigma`), which is what `FIXED` below enables.

The three modes
---------------
    legacy    (default) `torch.randn_like(std)` — byte-for-byte the historical path.
                        Nothing changes unless a caller opts in.
    generator           A dedicated `torch.Generator` per device, seeded once. Fresh noise
                        every call (real ensemble spread, real training dynamics) and the
                        SEQUENCE is reproducible run-to-run, independent of how many draws
                        anything else made. Good for reproducible training; NOT sufficient
                        for the §4 gate, because an RNG kernel change under compile can
                        still move the values.
    fixed               **This is the mode for the §4 equivalence gate.** eps is drawn once
                        per (shape, dtype, device) on the CPU from a seeded generator, then
                        cached and reused. The graph contains no RNG kernel at all, so the
                        forward becomes a deterministic function of its input and eps is
                        bit-identical across eager/compiled/backends.

Why `fixed` does NOT collapse the ensemble
------------------------------------------
The ensemble dimension is folded into the batch (`to_ensemble_batch` repeats each sample
`num_ensemble_members` times), so `std.shape` already spans the members. One frozen eps
tensor therefore gives each member a DIFFERENT value — spread is preserved, and the CRPS
skill/spread decomposition stays meaningful. What is removed is step-to-step noise, which
is exactly the confound the gate needs gone. It does change training dynamics, so `fixed`
is a comparison harness, never a training setting.

Usage
-----
    from utils import vae_noise
    vae_noise.enable_fixed(seed=0)      # capture a baseline, then re-run the optimized code
    ...
    vae_noise.disable()                 # back to the legacy global-RNG draw

    with vae_noise.fixed(seed=0):       # or scoped
        ...
"""

import contextlib
import logging
import os
import zlib

import torch

LEGACY = "legacy"
GENERATOR = "generator"
FIXED = "fixed"
_MODES = (LEGACY, GENERATOR, FIXED)

_mode = LEGACY
_seed = None
_generators = {}   # device -> torch.Generator
_eps_cache = {}    # (shape, dtype, device) -> Tensor


def _resolve_from_env():
    """`$PANGU_VAE_NOISE=fixed:0` / `generator:1234` — opt-in, absent => legacy.

    Env-configurable so a bench/equivalence PBS script can flip it without editing code,
    matching how S2S_BENCH/S2S_NVTX are plumbed.
    """
    raw = os.environ.get("PANGU_VAE_NOISE", "").strip()
    if not raw:
        return
    mode, _, seed_str = raw.partition(":")
    if mode not in _MODES:
        raise ValueError(
            "PANGU_VAE_NOISE=%r: mode must be one of %s (optionally '<mode>:<seed>')"
            % (raw, ", ".join(_MODES)))
    if mode == LEGACY:
        disable()
        return
    if not seed_str:
        raise ValueError("PANGU_VAE_NOISE=%r: mode %r requires a seed, e.g. '%s:0'"
                         % (raw, mode, mode))
    _enable(mode, int(seed_str))


def _enable(mode, seed):
    global _mode, _seed
    if mode not in _MODES:
        raise ValueError("unknown vae_noise mode %r; expected one of %s" % (mode, _MODES))
    if seed is None:
        raise ValueError("vae_noise mode %r requires an explicit seed" % mode)
    _mode, _seed = mode, int(seed)
    reset()
    logging.info("vae_noise: mode=%s seed=%d — the VAE reparameterization draw is now "
                 "decoupled from the global RNG (DESIGN §4.0)", _mode, _seed)


def enable_fixed(seed):
    """Freeze eps per (shape, dtype, device). The §4 equivalence-gate mode."""
    _enable(FIXED, seed)


def enable_generator(seed):
    """Fresh-but-reproducible eps from a dedicated Generator."""
    _enable(GENERATOR, seed)


def disable():
    """Back to `torch.randn_like` on the global RNG (the historical path)."""
    global _mode, _seed
    _mode, _seed = LEGACY, None
    reset()


def reset():
    """Drop cached generators/eps. Call between baseline and comparison runs."""
    _generators.clear()
    _eps_cache.clear()


def mode():
    return _mode


def seed():
    return _seed


@contextlib.contextmanager
def fixed(seed):
    """Scoped `enable_fixed`, restoring whatever was set before."""
    prev_mode, prev_seed = _mode, _seed
    enable_fixed(seed)
    try:
        yield
    finally:
        if prev_mode == LEGACY:
            disable()
        else:
            _enable(prev_mode, prev_seed)


def _generator_for(device):
    key = str(device)
    gen = _generators.get(key)
    if gen is None:
        gen = torch.Generator(device=device)
        gen.manual_seed(_seed)
        _generators[key] = gen
    return gen


def _site_seed(site):
    """A stable per-site seed. crc32, NOT hash(): hash() of a str is randomized per
    process by PYTHONHASHSEED, so it would silently give a different eps on every run —
    a "fixed" noise that isn't fixed is worse than no hook at all."""
    return (_seed * 1_000_003 + zlib.crc32(site.encode())) % (2**63 - 1)


def _fixed_eps_for(std, site):
    key = (site, tuple(std.shape), std.dtype, str(std.device))
    eps = _eps_cache.get(key)
    if eps is None:
        # Drawn on the CPU from a seeded generator, then moved. Deliberate: this makes eps
        # depend ONLY on (seed, site, shape) — not on the CUDA RNG implementation, the
        # device count, or the rank — so a baseline captured on one backend is comparable
        # with a re-run on another. Drawing on-device would reintroduce the very coupling
        # to CUDA RNG kernel selection this module exists to remove.
        #
        # `site` is in the key because the two encoders can share a shape: keyed on shape
        # alone they would receive the IDENTICAL eps, correlating two draws that are
        # independent in the real model and quietly changing what the KL term measures.
        #
        # The shape spans the ensemble (to_ensemble_batch folds members into the batch),
        # so each member still gets its own value: freezing eps does NOT collapse spread.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(_site_seed(site))
        eps = torch.randn(std.shape, generator=gen, dtype=torch.float32)
        eps = eps.to(device=std.device, dtype=std.dtype)
        _eps_cache[key] = eps
    return eps


def draw_eps(std, site="default"):
    """The reparameterization noise for a tensor shaped like `std`.

    `site` names the call site (e.g. "encoder1"/"encoder2") so that independent draws stay
    independent under FIXED. Returns `torch.randn_like(std)` unless a mode was enabled — so
    the default path is the historical one, byte for byte.
    """
    if _mode == LEGACY:
        return torch.randn_like(std)
    if _mode == GENERATOR:
        return torch.randn(std.shape, generator=_generator_for(std.device),
                           dtype=std.dtype, device=std.device)
    if _mode == FIXED:
        return _fixed_eps_for(std, site)
    raise AssertionError("unreachable vae_noise mode %r" % _mode)


_resolve_from_env()
