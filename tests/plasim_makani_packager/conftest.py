"""Pytest config — add repo src/ to sys.path + shared synthetic helpers.

The repo has no pyproject.toml / installed package, so tests import
``plasim_makani_packager.*`` by prepending {repo}/src to sys.path here.

Synthetic postproc helpers
--------------------------

Per docs/plasim_zg_plev_migration_plan.md (v7) §3.9, ``zg`` lives in the
postproc as ``zg_plev(time, lev_2, lat, lon)`` with ``lev_2`` carrying
hPa values [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925,
1000]. The packager reads it by *value*, not by slice index.

The audit gate ``stats._audit_zg500_inline`` hard-fails if the global
mean of ``zg500`` is outside [5400, 5700] m. So tests that build
synthetic postproc data and run ``compute_stats`` must use physically
plausible per-level zg means rather than zero or random values.
``_make_synthetic_zg_plev`` returns an array centred on standard-
atmosphere geopotential heights at each pressure, with ~50 m noise on
top — guaranteeing the audit passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
_TESTS = Path(__file__).resolve().parent

for entry in (_SRC, _TESTS):
    s = str(entry)
    if s not in sys.path:
        sys.path.insert(0, s)


# Standard-atmosphere reference geopotential height per hPa, used to give
# synthetic ``zg_plev`` arrays physically plausible per-level means so
# the v10 audit gate (zg500 in [5400, 5700] m) passes by construction.
_ZG_PLEV_REFERENCE_M: dict[int, float] = {
    50:   20500.0,
    100:  16100.0,
    150:  13500.0,
    200:  11700.0,
    250:  10300.0,
    300:   9100.0,
    400:   7100.0,
    500:   5550.0,  # centre of [5400, 5700] audit band
    600:   4200.0,
    700:   3000.0,
    850:   1450.0,
    925:    750.0,
    1000:   100.0,
}

# Canonical postproc lev_2 ordering (PRESSURE_LEVELS in
# src/plasim_postprocessor/plasim_postprocessor.py).
LEV_2_HPA: tuple[int, ...] = (
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000,
)


def _make_synthetic_zg_plev(
    T: int, H: int, W: int, *, rng: np.random.Generator,
    lev_2: tuple[int, ...] = LEV_2_HPA,
) -> np.ndarray:
    """(T, len(lev_2), H, W) float32 zg_plev with standard-atmosphere means.

    Per-level mean is the table value (so ``mean(zg500) ≈ 5550 m``
    independent of T/H/W); per-cell noise is ~50 m.
    """
    out = np.empty((T, len(lev_2), H, W), dtype=np.float32)
    for k, hpa in enumerate(lev_2):
        ref = _ZG_PLEV_REFERENCE_M[int(hpa)]
        out[:, k] = (
            ref + rng.normal(0.0, 50.0, size=(T, H, W))
        ).astype(np.float32)
    return out
