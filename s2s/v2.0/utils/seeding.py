"""Seed control for reproducible baselines (DESIGN.md §4.0).

The equivalence gate in DESIGN §4 compares a post-optimization run against a captured
baseline. That comparison is meaningless unless the baseline can be re-run and land in the
same place, so a run must be reproducible from (config, seed) alone.

Before this module, `train.py` did only `torch.manual_seed(world_rank)`. Three gaps:
  * `numpy` was never seeded, yet `train.py` draws from it (`np.random.randint` picks the
    validation sample to log) — so two runs of the "same" config diverge.
  * `random` was never seeded.
  * `cudnn.benchmark = True` lets cuDNN pick algorithms by timing, which varies run to run.

Precedence (first wins):  --seed  >  $S2S_SEED  >  `seed:` in the YAML  >  None.

**None means the legacy path is preserved byte-for-byte** (`torch.manual_seed(world_rank)`,
`cudnn.benchmark = True`). Seeding is opt-in: passing no seed cannot change what an existing
run computes, so this module cannot perturb the greens that already exist. You opt in when
capturing or reproducing a baseline.

Shared code: `s2s/v2.0/` is imported by S2S *and* the Lightning port (CLAUDE.md #5), so this
module is additive and framework-agnostic — it touches only the process RNGs. The port keeps
using Lightning's `seed_everything`; `equivalent_to_seed_everything()` documents how the two
line up so a baseline captured under one is comparable to the other.
"""

import os
import random
import warnings

import numpy as np
import torch

__all__ = [
    "resolve_seed",
    "apply_seed",
    "enable_determinism",
    "make_generator",
    "seed_worker",
    "equivalent_to_seed_everything",
]

# Lightning's seed_everything accepts [0, 2**32-1]; match it so a seed is portable between
# the canonical trainer and the port.
_MIN_SEED = 0
_MAX_SEED = 2 ** 32 - 1


def resolve_seed(cli_seed=None, params=None, env=None):
    """Return the effective seed, or None to keep the legacy `manual_seed(world_rank)` path.

    Precedence: cli_seed > $S2S_SEED > params['seed'] > None.
    Raises ValueError on a malformed or out-of-range value rather than silently falling back —
    a run that was *asked* to be reproducible must never quietly become unreproducible.
    """
    env = os.environ if env is None else env

    if cli_seed is not None:
        return _validate(cli_seed, "--seed")

    raw = env.get("S2S_SEED")
    if raw is not None and raw != "":
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ValueError("S2S_SEED must be an integer, got %r" % (raw,))
        return _validate(value, "S2S_SEED")

    if params is not None:
        try:
            value = params["seed"]
        except (KeyError, TypeError):
            value = None
        # A YAML `seed:` may be absent (KeyError) or explicitly null (None) — both mean legacy.
        if value is not None:
            return _validate(value, "config 'seed'")

    return None


def _validate(value, origin):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("%s must be an integer, got %r" % (origin, value))
    value = int(value)
    if not _MIN_SEED <= value <= _MAX_SEED:
        raise ValueError(
            "%s must be in [%d, %d], got %d" % (origin, _MIN_SEED, _MAX_SEED, value))
    return value


def apply_seed(seed, rank=0):
    """Seed `random`, `numpy` and `torch` (CPU + all CUDA devices) for this process.

    Returns the per-rank seed actually applied: `seed + rank`.

    Why the rank offset: the legacy line was `torch.manual_seed(world_rank)`, i.e. every rank
    drew a *different* stream. That is deliberate — the loader adds per-sample noise
    (`data_loader_multifiles.py:474-481`) and identical noise on every rank would correlate
    the ranks' gradients. `seed + rank` keeps streams distinct while making each one a
    function of the seed, so the run is reproducible. Model weights are unaffected by the
    offset: DDP broadcasts rank 0's parameters at construction.

    For a §4.1 baseline (world_size 1) rank is 0, so the applied seed is exactly `seed`.
    """
    if seed is None:
        raise ValueError("apply_seed(None): use resolve_seed() and keep the legacy path")
    rank_seed = (int(seed) + int(rank)) % (_MAX_SEED + 1)
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        # Covers every visible device, not just the current one.
        torch.cuda.manual_seed_all(rank_seed)
    return rank_seed


def enable_determinism(warn_only=False):
    """Force deterministic kernels (DESIGN §4.1). Slower — for baseline capture, not throughput.

    Returns a list of human-readable warnings (empty when fully applied).

    CUBLAS_WORKSPACE_CONFIG must be set BEFORE the CUDA context is created; setting it here
    can be too late, so we report rather than pretend. Set it in the job script:
        export CUBLAS_WORKSPACE_CONFIG=:4096:8
    """
    notes = []
    torch.backends.cudnn.benchmark = False      # stop timing-based algo selection
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        notes.append(
            "CUBLAS_WORKSPACE_CONFIG is not set to ':4096:8'; deterministic cuBLAS reductions "
            "are not guaranteed. Export it in the job script BEFORE python starts.")

    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except Exception as exc:                      # noqa: BLE001 - report, never abort the run
        notes.append("torch.use_deterministic_algorithms(True) failed: %s" % (exc,))

    for note in notes:
        warnings.warn(note, RuntimeWarning, stacklevel=2)
    return notes


def make_generator(seed, rank=0):
    """A `torch.Generator` for a DataLoader's `generator=`, so shuffling is seed-derived."""
    g = torch.Generator()
    g.manual_seed((int(seed) + int(rank)) % (_MAX_SEED + 1))
    return g


def seed_worker(worker_id):
    """DataLoader `worker_init_fn`: give each worker a distinct, seed-derived numpy/random stream.

    PyTorch already seeds each worker's *torch* RNG from the parent's seed, which is what the
    loader's `torch.randn` noise uses — but it leaves `numpy` and `random` alone, so workers
    would share whatever state they inherited at fork. Derive both from the torch seed the
    parent handed this worker, keeping everything a function of the run seed.
    """
    worker_seed = torch.initial_seed() % (_MAX_SEED + 1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def equivalent_to_seed_everything(seed):
    """How this module lines up with Lightning's `seed_everything`, used by the port and SI.

    `seed_everything(s)` seeds random/numpy/torch with `s` on every rank and exports
    PL_GLOBAL_SEED. `apply_seed(s, rank)` seeds them with `s + rank`. They therefore agree
    exactly at rank 0 — which is the §4.1 baseline case (world_size 1) — and diverge by the
    intentional per-rank offset above.

    Practical consequence: a rank-0/world-size-1 baseline is comparable across the canonical
    trainer and the port. A multi-rank one is NOT; capture the 4-GPU baseline with the same
    launcher you compare against.
    """
    return {"rank0_equivalent": True, "multi_rank_equivalent": False, "seed": int(seed)}
