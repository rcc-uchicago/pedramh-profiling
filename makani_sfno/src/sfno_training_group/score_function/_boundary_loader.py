"""Read varying-boundary forcing trajectories from converted group-format h5.

The score-function wrapper's ``rollout(steps=K)`` needs the (4, 64, 128)
per-step varying-boundary forcing for K + 1 timesteps (IC + K). This module
streams those from the flat ``<year>_<idx:04>.h5`` files emitted by
``convert_v10_to_group_h5.py``.

If a rollout crosses a year boundary, the loader transparently switches to
``<year+1>_0000.h5`` etc. (For Phase 1 smoke we typically stay within year
121 so this is a robustness measure.)
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import cftime
import h5py
import numpy as np

# Manifest source: convert_v10_to_group_h5 emits 1455 timesteps per year for
# our v10 source. We re-read the manifest per call so this file remains decoupled.


def _idx_for_dt(dt: cftime.datetime, *, data_timedelta_hours: int = 6) -> int:
    jan1 = cftime.DatetimeProlepticGregorian(dt.year, 1, 1, has_year_zero=True)
    return int((dt - jan1).total_seconds()) // (data_timedelta_hours * 3600)


def load_varying_boundary_trajectory(
    data_dir: Path,
    init_dt: cftime.datetime,
    steps: int,
    *,
    varying_boundary_variables: list[str],
    n_per_year: int = 1455,
) -> np.ndarray:
    """Load (steps + 1, n_vars, 64, 128) varying-boundary forcing.

    Order: ``varying_boundary_variables`` (e.g. ['z0', 'sst', 'rsdt', 'sic']).
    The leading dim covers ``init_dt, init_dt + 6h, ..., init_dt + steps * 6h``.
    """
    n_vars = len(varying_boundary_variables)
    out = np.empty((steps + 1, n_vars, 64, 128), dtype=np.float32)

    dt = init_dt
    for k in range(steps + 1):
        year = dt.year
        idx = _idx_for_dt(dt)
        if idx >= n_per_year:
            # roll into next year
            year += 1
            idx -= n_per_year
        path = data_dir / f"{year}_{idx:04d}.h5"
        if not path.is_file():
            raise FileNotFoundError(
                f"boundary trajectory step {k}: missing {path} (init={init_dt}, year={year}, idx={idx})"
            )
        with h5py.File(path, "r") as f:
            g = f["input"]
            for vi, var in enumerate(varying_boundary_variables):
                out[k, vi] = np.asarray(g[var], dtype=np.float32)
        dt = dt + timedelta(hours=6)
    return out
