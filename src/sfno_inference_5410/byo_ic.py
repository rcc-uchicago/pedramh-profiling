"""User-supplied (BYO) NetCDF initial condition reader for SFNO-5410 inference.

Used by ``scripts/infer_sfno5410_byo_ic.py``. The strict schema mirrors the
sim52 H5 layout that the production data loader expects internally; we
substitute the user's IC AFTER the data loader has yielded its (sim52-derived)
boundary forcing, so only the IC fields need to be supplied.

Required NetCDF schema
----------------------
Single-timestep dataset on the 64x128 lat/lon grid with all of:

| Variable | Dims                                  | Units / convention            |
|----------|---------------------------------------|-------------------------------|
| pl       | (lat, lon) or (time=1, lat, lon)      | dimensionless (ln(p_s))       |
| tas      | (lat, lon) or (time=1, lat, lon)      | K                             |
| pr_6h    | (lat, lon) or (time=1, lat, lon)      | rate * 6h (group convention)  |
| ta       | (lev, lat, lon) or (time=1, ...)      | K, sigma levels               |
| ua       | (lev, lat, lon) or (time=1, ...)      | m/s, sigma levels             |
| va       | (lev, lat, lon) or (time=1, ...)      | m/s, sigma levels             |
| hus      | (lev, lat, lon) or (time=1, ...)      | kg/kg, sigma levels           |
| zg       | (plev, lat, lon) or (time=1, ...)     | gpm (NOT m^2/s^2)             |

Coords required: ``lat`` (size 64), ``lon`` (size 128), ``lev`` (size 10) for
sigma vars, ``plev`` (size 10) for ``zg``. If a ``time`` dim is present it
must have length 1.

The function does NOT enforce that ``lev`` / ``plev`` *values* match the
sim52 sigma levels / pressure levels — the caller is responsible for ensuring
the user's vertical interpolation matches the model's training grid.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


SURFACE_VARS = ("pl", "tas")
UPPER_AIR_VARS = ("ta", "ua", "va", "hus")
DIAGNOSTIC_VARS = ("pr_6h",)
ZG_VAR = "zg"
ALL_REQUIRED = (*SURFACE_VARS, *UPPER_AIR_VARS, *DIAGNOSTIC_VARS, ZG_VAR)

EXPECTED_LAT = 64
EXPECTED_LON = 128
EXPECTED_LEV = 10
EXPECTED_PLEV = 10


def _squeeze_time(arr: xr.DataArray) -> np.ndarray:
    """Drop a length-1 time axis if present and return float32 np.ndarray."""
    if "time" in arr.dims:
        if arr.sizes["time"] != 1:
            raise ValueError(
                f"variable {arr.name!r}: time dim has size "
                f"{arr.sizes['time']}, expected 1 (single-timestep IC)"
            )
        arr = arr.isel(time=0)
    return np.ascontiguousarray(arr.values, dtype=np.float32)


def validate_byo_ic(nc_path: Path) -> dict[str, np.ndarray]:
    """Read user IC NetCDF, validate schema, return raw arrays in sim52 layout.

    Returns a dict keyed by variable name with shapes::

        pl, tas, pr_6h        -> (64, 128)
        ta, ua, va, hus, zg   -> (10, 64, 128)

    Raises
    ------
    FileNotFoundError
        If ``nc_path`` does not exist.
    ValueError
        For any schema violation (missing var, wrong grid, multi-timestep,
        wrong number of vertical levels).
    """
    p = Path(nc_path)
    if not p.is_file():
        raise FileNotFoundError(f"BYO IC NetCDF not found: {p}")

    out: dict[str, np.ndarray] = {}
    with xr.open_dataset(p) as ds:
        missing = [v for v in ALL_REQUIRED if v not in ds.data_vars]
        if missing:
            raise ValueError(
                f"{p}: missing required variables {missing}. "
                f"Required: {list(ALL_REQUIRED)}; got: {list(ds.data_vars)}"
            )

        for coord, expected in (("lat", EXPECTED_LAT), ("lon", EXPECTED_LON)):
            if coord not in ds.sizes:
                raise ValueError(f"{p}: missing required coord/dim {coord!r}")
            if ds.sizes[coord] != expected:
                raise ValueError(
                    f"{p}: dim {coord} has size {ds.sizes[coord]}, "
                    f"expected {expected} (sim52 grid)"
                )

        # Surface + diagnostic: 2D
        for var in (*SURFACE_VARS, *DIAGNOSTIC_VARS):
            arr = _squeeze_time(ds[var])
            if arr.shape != (EXPECTED_LAT, EXPECTED_LON):
                raise ValueError(
                    f"{p}: variable {var!r} has shape {arr.shape}, "
                    f"expected ({EXPECTED_LAT}, {EXPECTED_LON})"
                )
            out[var] = arr

        # Upper-air sigma vars: 3D
        if "lev" not in ds.sizes:
            raise ValueError(f"{p}: missing required dim 'lev' for sigma-level vars")
        if ds.sizes["lev"] != EXPECTED_LEV:
            raise ValueError(
                f"{p}: dim 'lev' has size {ds.sizes['lev']}, "
                f"expected {EXPECTED_LEV} (10 sigma levels)"
            )
        for var in UPPER_AIR_VARS:
            arr = _squeeze_time(ds[var])
            if arr.shape != (EXPECTED_LEV, EXPECTED_LAT, EXPECTED_LON):
                raise ValueError(
                    f"{p}: variable {var!r} has shape {arr.shape}, "
                    f"expected ({EXPECTED_LEV}, {EXPECTED_LAT}, {EXPECTED_LON})"
                )
            out[var] = arr

        # zg: pressure-level
        if "plev" not in ds.sizes:
            raise ValueError(f"{p}: missing required dim 'plev' for zg")
        if ds.sizes["plev"] != EXPECTED_PLEV:
            raise ValueError(
                f"{p}: dim 'plev' has size {ds.sizes['plev']}, "
                f"expected {EXPECTED_PLEV}"
            )
        zg = _squeeze_time(ds[ZG_VAR])
        if zg.shape != (EXPECTED_PLEV, EXPECTED_LAT, EXPECTED_LON):
            raise ValueError(
                f"{p}: variable 'zg' has shape {zg.shape}, "
                f"expected ({EXPECTED_PLEV}, {EXPECTED_LAT}, {EXPECTED_LON})"
            )
        out[ZG_VAR] = zg

    return out


def stack_for_model(
    raw: dict[str, np.ndarray],
    surface_variables: tuple[str, ...],
    upper_air_variables: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (surface_stack, upper_air_stack) in the order the model expects.

    surface_stack shape: (len(surface_variables), 64, 128)
    upper_air_stack shape: (len(upper_air_variables), 10, 64, 128)
    """
    surface = np.stack([raw[v] for v in surface_variables], axis=0).astype(np.float32)
    upper_air = np.stack([raw[v] for v in upper_air_variables], axis=0).astype(np.float32)
    return surface, upper_air


__all__ = (
    "validate_byo_ic",
    "stack_for_model",
    "SURFACE_VARS",
    "UPPER_AIR_VARS",
    "DIAGNOSTIC_VARS",
    "ZG_VAR",
    "ALL_REQUIRED",
    "EXPECTED_LAT",
    "EXPECTED_LON",
    "EXPECTED_LEV",
    "EXPECTED_PLEV",
)
