#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The architecture gate for the SFNO apples-to-apples comparison.

The whole comparison rests on one claim, made only in a docstring:
``SfnoPlasim`` says it is *"faithful to PanguWeather v2.0
networks/modulus_sfno/sfnonet.py::SphericalFourierNeuralOperatorNet_v2"*. That
is a claim, not a proof. If it is false we are not comparing two harnesses
running the same model — we are comparing two different models, and every
throughput or loss number downstream is meaningless.

Three checks, cheapest first. Each is independently decisive:

1. ``--check-source`` — the vendored implementation vs PanguWeather's, file by
   file. Standard library only; runs anywhere in milliseconds. Differences are
   classified: an *allowed* difference is one that provably cannot change
   parameters or eager-mode numerics (a ``@torch.compiler.disable`` decorator, a
   ``return_latent`` branch, a print). Anything else fails.
2. ``--check-config`` — the two YAMLs' architecture blocks, key by key. Also
   standard library. Catches the case where the code is identical and the
   configs are not.
3. ``--check-params`` — build both models and compare
   ``sum(p.numel() for p in model.parameters())``. Needs torch. This is the
   check the handoff names, and the one that catches a wiring difference the
   first two cannot (e.g. a channel count reached by a different route).

``--check-weights`` adds the strong form: load one model's weights into the
other, run identical input through both, and compare outputs and gradient norms
with no optimizer step, so nothing compounds (the ``equivalence.py`` MODE=fixed
idea, across implementations rather than across a code change).

Where to run what
-----------------
1 and 2 are safe on a login node. 3 and 4 import torch and build a 1.18 B-param
model, so they belong in a job — ``physicsnemo_ai_rossby/polaris/polaris_sfno_parity_gate.pbs``
(CLAUDE.md #3: importing torch on a login node can hang or core-dump).

3 and 4 run both implementations in ONE process, which requires an environment
that satisfies both. The ai-rossby venv does (torch 2.10, torch_harmonics 0.9.1,
tltorch). Note that this proves *implementation* parity; the PanguWeather
production run executes under the base conda (torch_harmonics 0.7.4), so
``--check-params`` additionally asserts the count against
``EXPECTED_PARAMS``, the value measured in PanguWeather's own environment
(``polaris_bench_report.md``, job 7255410).

PASS = ``SFNO_PARITY_OK``. Any failure prints a single greppable ``ERROR`` line.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

PANGU_V20 = REPO / "PanguWeather/v2.0"
PANGU_SFNO_DIR = PANGU_V20 / "networks/modulus_sfno"
AR_ROOT = REPO / "physicsnemo_ai_rossby"
AR_SFNO_DIR = AR_ROOT / "physicsnemo/experimental/models/modulus_sfno"
AR_RECIPE = AR_ROOT / "examples/weather/ai_rossby"

PANGU_CONFIG = PANGU_V20 / "config/E3SM_SFNO_H5_POLARIS_ALLDATA.yaml"
AR_CONFIG = AR_RECIPE / "conf/model/sfno_e3sm_parity.yaml"

# Measured in PanguWeather's own environment: polaris_bench_report.md line 42,
# job 7255410, at embed_dim 512 / in_chans 105 / out_chans 101. An independent
# anchor for the count, not derived from either implementation here.
EXPECTED_PARAMS = 1_182_108_160

# The channel arithmetic both sides must reach. PanguWeather gets there via
# `variable_list_in` + `constant_boundary_variables` (sfnonet.py:762-765);
# ai-rossby sums its variable groups directly (sfno_plasim.py:208-210).
EXPECTED_IN_CHANS = 105
EXPECTED_OUT_CHANS = 101

# --- 1. Source parity -------------------------------------------------------
#
# Every file of the vendored SFNO must match PanguWeather's, except for
# differences that provably cannot change parameters or eager-mode numerics.
#
# The allowance is an EXACT-LINE list, not a set of regexes, and deliberately
# so. A regex broad enough to cover the `return_latent` refactor (`^\s*else:$`,
# `^\s*return ...`) is also broad enough to hide a real change on a line that
# happens to look like it. Every entry below is one literal line of one of the
# two files, with the reason it cannot matter. Waving a new difference through
# means adding its exact text here, in a diff, where a reviewer sees it.
#
# Only two families exist today:
#   * torch.compile hints — inert in eager, and eager is what runs (the §4 gate
#     measured torch.compile at 1.40x but FAILING equivalence, so it stays off);
#   * `return_latent` — an opt-in extra return value the training path never
#     requests. It adds no module, so it cannot add a parameter.
# Plus one config plumbing change: `data_grid`, whose VALUE --check-config
# asserts is 'equiangular' on both sides.
ALLOWED_DIFF_LINES = {
    # -- torch.compile hints (inert in eager) --
    "@torch.compiler.disable(recursive=True)",
    # -- return_latent: signatures --
    "def forward(self, x):",
    "def forward(self, x, return_latent=False):",
    "def forward(self, surface, c_boundary, v_boundary, upper_air, train=None):",
    "def forward(self, surface, c_boundary, v_boundary, upper_air, train=None, "
    "return_latent=False):",
    # -- return_latent: base-net body --
    "latent = x.detach().clone() if return_latent else None",
    "if return_latent:",
    "return x, latent",
    # -- return_latent: wrapper body. The super() call is unchanged; it only
    #    moved inside an if/else, and the else-branch is the original call.
    "x = super(SphericalFourierNeuralOperatorNet_v2, self).forward(x)",
    "x, latent = super(SphericalFourierNeuralOperatorNet_v2, self).forward(x, "
    "return_latent=True)",
    "else:",
    "latent = None",
    # -- return_latent: the return tuple, bound to a name before returning.
    #    Both spellings of both branches are listed, so a changed tuple fails.
    "return surface, upper_air, diagnostic, torch.tensor(0.0), torch.tensor(0.0), "
    "torch.tensor(0.0), torch.tensor(0.0)",
    "return surface, upper_air, torch.tensor(0.0), torch.tensor(0.0), "
    "torch.tensor(0.0), torch.tensor(0.0)",
    "result = (surface, upper_air, diagnostic, torch.tensor(0.0), torch.tensor(0.0), "
    "torch.tensor(0.0), torch.tensor(0.0))",
    "result = (surface, upper_air, torch.tensor(0.0), torch.tensor(0.0), "
    "torch.tensor(0.0), torch.tensor(0.0))",
    "return result + (latent,)",
    "return result",
    # -- data_grid made configurable (value asserted by --check-config) --
    "## GRID IS HARD CODED-- make sure it aligns with input data grid ##",
    "grid_type = Params('equiangular') # default for PlaSim: Params('legendre-gauss')",
    "if getattr(params_trainer, 'use_legacy_data_grid', False):",
    "grid_type = Params('legendre-gauss')",
    "grid_type = Params(getattr(params_trainer, 'data_grid', 'equiangular'))",
    # -- debug prints removed --
    "#print('data_grid',params.data_grid)",
    "print('params',params)",
}


def _is_allowed(line: str) -> bool:
    """Comments and blanks are always inert; everything else must be listed."""
    s = line.strip()
    return not s or s.startswith("#") or s in ALLOWED_DIFF_LINES

SFNO_FILES = [
    "sfnonet.py",
    "layers.py",
    "s2convolutions.py",
    "contractions.py",
    "factorizations.py",
    "activations.py",
    "initialization.py",
]


def _changed_lines(a: str, b: str) -> list[str]:
    """Lines that differ between two sources, without the diff bookkeeping."""
    return [
        line[1:]
        for line in difflib.unified_diff(
            a.splitlines(), b.splitlines(), lineterm="", n=0
        )
        if line[:1] in "+-" and not line.startswith(("---", "+++"))
    ]


def check_source() -> int:
    print("=== source parity: vendored modulus_sfno vs PanguWeather ===")
    failed = 0
    for name in SFNO_FILES:
        pangu, ar = PANGU_SFNO_DIR / name, AR_SFNO_DIR / name
        if not pangu.exists() or not ar.exists():
            print(f"[FAIL] {name}: missing on one side")
            failed += 1
            continue
        changed = _changed_lines(pangu.read_text(), ar.read_text())
        unexplained = [ln for ln in changed if not _is_allowed(ln)]
        if unexplained:
            failed += 1
            print(f"[FAIL] {name}: {len(unexplained)} unexplained changed line(s)")
            for ln in unexplained[:10]:
                print(f"         {ln.rstrip()}")
        else:
            note = f"{len(changed)} allowed" if changed else "identical"
            print(f"[PASS] {name} ({note})")
    if failed:
        print(f"ERROR SFNO_SOURCE_DIVERGED {failed}/{len(SFNO_FILES)} file(s)")
    return 1 if failed else 0


# --- 2. Config parity -------------------------------------------------------

ARCH_KEYS = [
    "spectral_transform", "filter_type", "operator_type", "scale_factor",
    "embed_dim", "num_layers", "use_mlp", "mlp_ratio", "activation_function",
    "encoder_layers", "pos_embed", "drop_rate", "drop_path_rate", "num_blocks",
    "sparsity_threshold", "normalization_layer", "hard_thresholding_fraction",
    "use_complex_kernels", "big_skip", "rank", "factorization", "separable",
    "complex_network", "complex_activation", "spectral_layers", "checkpointing",
]


def _scalar(raw: str):
    """A YAML scalar, normalized so the two spellings compare equal.

    Notably ``None`` -> ``None``: PanguWeather writes the bare word and
    ``YParams`` maps the resulting *string* to Python ``None``
    (``utils/YParams.py:19``), which is what selects the non-tensorly
    ComplexDense weight path (``sfnonet.py:115``). ai-rossby writes ``null``.
    Same value, and the difference is load-bearing enough to be worth naming.
    """
    raw = raw.split("#")[0].strip().strip("'\"")
    if raw in ("None", "null", "~", ""):
        return None
    if raw in ("True", "true"):
        return True
    if raw in ("False", "false"):
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _arch_block(text: str, *, section: str | None = None) -> dict:
    """The architecture keys of a config, as a normalized dict.

    ``section`` restricts the scan to a YAML top-level block (PanguWeather's
    file defines several; only ``SFNO:`` is the one that runs). Values are read
    with line regexes because the PanguWeather file uses merge keys and repeated
    anchors that a YAML round-trip mangles, and because this must run with only
    the standard library.
    """
    if section:
        m = re.search(rf"(?m)^{re.escape(section)}:.*$", text)
        if not m:
            raise KeyError(f"section {section!r} not found")
        rest = text[m.end():]
        nxt = re.search(r"(?m)^\S", rest)
        text = rest[: nxt.start()] if nxt else rest
    out = {}
    for key in ARCH_KEYS:
        hits = re.findall(rf"(?m)^\s*{re.escape(key)}:\s*(.*)$", text)
        if not hits:
            continue
        # Last wins, matching YAML's own duplicate-key semantics.
        out[key] = _scalar(hits[-1])
    return out


def check_config() -> int:
    print("=== config parity: SFNO architecture block ===")
    pangu = _arch_block(PANGU_CONFIG.read_text(), section="SFNO")
    ar = _arch_block(AR_CONFIG.read_text())
    failed = 0
    for key in ARCH_KEYS:
        p, a = pangu.get(key, "<absent>"), ar.get(key, "<absent>")
        if p != a:
            failed += 1
            print(f"[FAIL] {key}: pangu={p!r}  ai_rossby={a!r}")
    # data_grid: hardcoded 'equiangular' in PanguWeather, configurable here, so
    # it is not in ARCH_KEYS — check the ai-rossby value against the constant.
    m = re.search(r"(?m)^\s*data_grid:\s*(.*)$", AR_CONFIG.read_text())
    grid = _scalar(m.group(1)) if m else "<absent>"
    if grid != "equiangular":
        failed += 1
        print(f"[FAIL] data_grid: pangu hardcodes 'equiangular', ai_rossby={grid!r}")
    if failed:
        print(f"ERROR SFNO_CONFIG_DIVERGED {failed} key(s)")
        return 1
    print(f"[PASS] {len(ARCH_KEYS)} architecture keys + data_grid identical")
    print(f"       embed_dim={pangu['embed_dim']}  num_layers={pangu['num_layers']}  "
          f"num_blocks={pangu['num_blocks']}  checkpointing={pangu['checkpointing']}")
    return 0


# --- 3/4. Built models ------------------------------------------------------


class _Shim:
    """Stands in for ``YParams`` — attribute AND item access over a dict.

    ``SphericalFourierNeuralOperatorNet_v2`` reads ``params_trainer.embed_dim``
    but ``params_trainer['lat']``, so it needs both. Values come from the real
    PanguWeather YAML via :func:`_arch_block`, normalized the way ``YParams``
    normalizes them.
    """

    def __init__(self, d: dict):
        self.__dict__.update(d)

    def __getitem__(self, k):
        return self.__dict__[k]


class _DatasetShim:
    """The three attributes PanguWeather's SFNO reads off its dataset.

    Built from the variable contract rather than from a Zarr store or an H5
    archive, so the gate needs no data on disk. ``ai_rossby_variable_contract``
    is the single source those names come from, and it is separately asserted
    against the real store by ``--check-artifacts``.
    """

    def __init__(self, planned: dict, store_surface: list):
        upper = [
            f"{v}_{lev}" for v in planned["upper_air_variables"]
            for lev in planned["levels"]
        ]
        self.variable_list_out = upper + store_surface + planned["diagnostic_variables"]
        self.variable_list_in = upper + store_surface + planned["varying_boundary_variables"]
        self.constant_boundary_variables = planned["constant_boundary_variables"]


def _import_both():
    """Import both implementations, with the PYTHONPATH trap guarded.

    ``s2s/v2.0`` and ``PanguWeather/v2.0`` export the same top-level module
    names (``utils``, ``networks``, ``config``) and import unqualified, so with
    both on the path ``networks.*`` resolves to whichever is first — and the two
    differ by 106 lines. Only PanguWeather's goes on, and s2s's presence is a
    hard error rather than a silent wrong answer.
    """
    bad = [p for p in sys.path if p and Path(p).resolve() == (REPO / "s2s/v2.0").resolve()]
    if bad:
        print("ERROR PYTHONPATH_COLLISION: s2s/v2.0 is on sys.path alongside "
              "PanguWeather/v2.0; `networks.*` would resolve to the wrong tree.")
        raise SystemExit(3)
    sys.path.insert(0, str(PANGU_V20))
    from networks.modulus_sfno.sfnonet import (  # noqa: E402
        SphericalFourierNeuralOperatorNet_v2 as PanguSFNO,
    )
    from physicsnemo.experimental.models.sfno_plasim import SfnoPlasim  # noqa: E402

    return PanguSFNO, SfnoPlasim


def _dataset_shim():
    import ai_rossby_variable_contract as vc

    return _DatasetShim(vc.PLANNED, vc.STORE_SURFACE)


def _build_pangu():
    """PanguWeather's SFNO, from its own production YAML."""
    PanguSFNO, _ = _import_both()
    arch = _arch_block(PANGU_CONFIG.read_text(), section="SFNO")
    params = _Shim({
        **arch,
        # The class reads only len(lat)/len(lon) — the values never reach a
        # tensor, so a placeholder of the right length is the honest input here
        # and keeps the gate independent of any data on disk.
        "lat": [0.0] * 180,
        "lon": [0.0] * 360,
        "has_diagnostic": True,
    })
    dataset = _dataset_shim()
    return PanguSFNO(params, dataset), dataset


def _build_ai_rossby():
    """ai-rossby's SfnoPlasim, from the parity YAML, exactly as Hydra would.

    `build_model` (train.py:544) forwards every non-identity key of the model
    config as a constructor kwarg, so passing the parsed architecture block
    through as **kwargs reproduces the real construction path.
    """
    import ai_rossby_variable_contract as vc

    _, SfnoPlasim = _import_both()
    return SfnoPlasim(
        surface_variables=vc.STORE_SURFACE,
        upper_air_variables=vc.PLANNED["upper_air_variables"],
        constant_boundary_variables=vc.PLANNED["constant_boundary_variables"],
        varying_boundary_variables=vc.PLANNED["varying_boundary_variables"],
        diagnostic_variables=vc.PLANNED["diagnostic_variables"],
        levels=vc.PLANNED["levels"],
        horizontal_resolution=[180, 360],
        **_arch_block(AR_CONFIG.read_text()),
        data_grid=_scalar(
            re.search(r"(?m)^\s*data_grid:\s*(.*)$", AR_CONFIG.read_text()).group(1)
        ),
    )


def _n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def check_params() -> int:
    import torch

    print("=== parameter parity ===")
    torch.manual_seed(0)
    pangu, dataset = _build_pangu()
    ai_rossby = _build_ai_rossby()

    n_p, n_a = _n_params(pangu), _n_params(ai_rossby.sfno)
    in_chans = len(dataset.variable_list_in) + len(dataset.constant_boundary_variables)
    out_chans = len(dataset.variable_list_out)

    print(f"  channels          in={in_chans} out={out_chans} "
          f"(expected {EXPECTED_IN_CHANS}/{EXPECTED_OUT_CHANS})")
    print(f"  PanguWeather      {n_p:,}")
    print(f"  ai-rossby         {n_a:,}")
    print(f"  reference (measured, job 7255410)  {EXPECTED_PARAMS:,}")

    failed = 0
    if (in_chans, out_chans) != (EXPECTED_IN_CHANS, EXPECTED_OUT_CHANS):
        print("ERROR SFNO_CHANNELS_MISMATCH")
        failed += 1
    if n_p != n_a:
        print(f"ERROR SFNO_PARAM_MISMATCH pangu={n_p} ai_rossby={n_a} "
              f"delta={n_a - n_p:+,}")
        failed += 1
    if n_p != EXPECTED_PARAMS:
        # Not a parity failure between the two — a drift from the measured
        # reference, which means the config changed since the bench report.
        print(f"ERROR SFNO_PARAM_DRIFT built={n_p} reference={EXPECTED_PARAMS} "
              f"delta={n_p - EXPECTED_PARAMS:+,}")
        failed += 1
    # The wrapper must add nothing of its own — no head, no embedding.
    if _n_params(ai_rossby) != n_a:
        print(f"ERROR SFNO_WRAPPER_HAS_PARAMS extra="
              f"{_n_params(ai_rossby) - n_a:,}")
        failed += 1
    return 1 if failed else 0


def check_weights(device: str = "cuda") -> int:
    """Fixed weights, identical input: outputs and gradients must agree.

    The strong form of the gate. Parameter counts can match while the modules
    are wired differently; this catches that. No optimizer step is taken, so
    nothing compounds — a disagreement here is a real structural difference, not
    accumulated drift (the ``equivalence.py`` MODE=fixed argument).

    fp32 on purpose: the production runs are bf16, but bf16 noise would mask
    exactly the small structural differences this is looking for.
    """
    import torch

    import ai_rossby_variable_contract as vc

    print(f"=== fixed-weight equivalence (device={device}, fp32) ===")
    dev = torch.device(device)

    n_surf = len(vc.STORE_SURFACE)
    n_up, n_lev = len(vc.PLANNED["upper_air_variables"]), len(vc.PLANNED["levels"])
    n_const = len(vc.PLANNED["constant_boundary_variables"])
    n_vary = len(vc.PLANNED["varying_boundary_variables"])

    # One fixed input, generated on CPU from an explicit generator so it is
    # identical for both models regardless of what either constructor drew from
    # the global RNG.
    g = torch.Generator(device="cpu").manual_seed(1234)
    inputs = [
        torch.randn(1, n_surf, 180, 360, generator=g),
        torch.randn(1, n_const, 180, 360, generator=g),
        torch.randn(1, n_vary, 180, 360, generator=g),
        torch.randn(1, n_up, n_lev, 180, 360, generator=g),
    ]

    def run(model):
        """Forward + backward on `dev`, returning CPU results, then freeing.

        The two models are ~4.7 GB of fp32 parameters each and this config's
        measured training peak is ~25 GB, so they are never co-resident on a
        40 GB A100 — one runs, releases, then the other.
        """
        model = model.to(dev).eval()
        model.zero_grad(set_to_none=True)
        out = model(*[t.to(dev) for t in inputs])
        loss = sum(o.square().mean() for o in out[:3])
        loss.backward()
        grad_norm = torch.sqrt(
            sum(p.grad.double().square().sum()
                for p in model.parameters() if p.grad is not None)
        )
        result = ([o.detach().cpu() for o in out[:3]],
                  loss.detach().cpu(), grad_norm.cpu())
        del model, out, loss, grad_norm
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return result

    torch.manual_seed(0)
    pangu, _ = _build_pangu()
    weights = {k: v.clone() for k, v in pangu.state_dict().items()}
    out_p, loss_p, gn_p = run(pangu)
    del pangu

    ai_rossby = _build_ai_rossby()
    # ai-rossby nests the same net under `.sfno`; PanguWeather subclasses it. So
    # the state dicts must be key-for-key identical, and `strict=True` is the
    # assertion — a renamed or extra module fails loudly here instead of
    # silently leaving one side at its random init and "passing" a comparison of
    # two different models.
    ai_rossby.sfno.load_state_dict(weights, strict=True)
    del weights
    out_a, loss_a, gn_a = run(ai_rossby)

    failed = 0
    for tag, a, b in zip(("surface", "upper_air", "diagnostic"), out_p, out_a):
        denom = a.abs().max().clamp_min(1e-12)
        rel = ((a - b).abs().max() / denom).item()
        where = (a - b).abs().argmax().item()
        ok = rel <= 1e-6
        failed += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {tag:11s} max_rel_err={rel:.3e} "
              f"at flat idx {where}")

    for tag, a, b in (("loss", loss_p, loss_a), ("grad_norm", gn_p, gn_a)):
        rel = (abs(a - b) / abs(a).clamp_min(1e-12)).item()
        ok = rel <= 1e-6
        failed += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {tag:11s} rel_err={rel:.3e} "
              f"({a.item():.8e} vs {b.item():.8e})")

    if failed:
        print(f"ERROR SFNO_WEIGHTS_DIVERGED {failed} comparison(s)")
        return 1
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check-source", action="store_true")
    p.add_argument("--check-config", action="store_true")
    p.add_argument("--check-params", action="store_true")
    p.add_argument("--check-weights", action="store_true")
    p.add_argument("--all", action="store_true", help="all four (needs torch + a GPU)")
    p.add_argument("--static", action="store_true",
                   help="source + config only — safe on a login node")
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    if not any((a.check_source, a.check_config, a.check_params, a.check_weights,
                a.all, a.static)):
        p.error("pick at least one check (--static is the login-node-safe pair)")

    rc = 0
    if a.check_source or a.all or a.static:
        rc |= check_source()
        print()
    if a.check_config or a.all or a.static:
        rc |= check_config()
        print()
    if a.check_params or a.all:
        rc |= check_params()
        print()
    if a.check_weights or a.all:
        rc |= check_weights(a.device)
        print()

    print("ERROR SFNO_PARITY_FAILED" if rc else "SFNO_PARITY_OK")
    return rc


if __name__ == "__main__":
    sys.exit(main())
