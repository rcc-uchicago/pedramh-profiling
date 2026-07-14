"""Synthetic data helpers shared between packager and sfno_training tests.

Living outside ``conftest.py`` so it can be imported by tests in
``tests/sfno_training/`` (where ``conftest`` resolves to a different
file). Both test directories add ``tests/plasim_makani_packager`` to
``sys.path``, so ``from synthetic_helpers import ...`` resolves the
same module regardless of which test directory is running.
"""

from __future__ import annotations

import numpy as np


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
    500:   5550.0,
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
    """(T, len(lev_2), H, W) float32 zg_plev with standard-atmosphere means."""
    out = np.empty((T, len(lev_2), H, W), dtype=np.float32)
    for k, hpa in enumerate(lev_2):
        ref = _ZG_PLEV_REFERENCE_M[int(hpa)]
        out[:, k] = (
            ref + rng.normal(0.0, 50.0, size=(T, H, W))
        ).astype(np.float32)
    return out
