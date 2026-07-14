"""Build a group-format init NetCDF from one v10 test h5 IC.

Phase B.4 of the group-code training track plan v5.

Input: a v10 packaged h5 file (e.g. ``test/MOST.0121.h5``) plus a timestep index
and a synthetic init datetime to label the IC. Output: a NetCDF readable by
group's ``get_data_given_path_nc`` (``utils/data_loader_multifiles.py:83-160``)
and consumable by both ``long_inference.py --init_nc_filepaths`` (Phase F) and
the Phase 1 score-function wrapper.

Schema:

- ``time`` coord (1 entry): the synthetic init datetime, encoded with CF units
  ``"hours since {synthetic_init_dt}"`` and ``calendar="proleptic_gregorian"``.
- ``lat`` (64), ``lon`` (128) — 1-D coords.
- ``sigma`` (10) — 1-D coord with the 10 PlaSim native sigma floats; ALSO a
  variable in the file (``get_data_given_path_nc:130-150`` searches for the
  level coord by matching dim name to a variable name).
- ``lev`` (10) — 1-D coord with the 10 zg pressure values in Pa; ALSO a
  variable.
- 3D vars ``(time, sigma_or_lev, lat, lon)``:
  - ``ta(time, sigma, lat, lon)``, ``ua, va, hus`` similarly.
  - ``zg(time, lev, lat, lon)``.
- 2D vars ``(time, lat, lon)``: ``pl, tas``.
- Boundary forcing (sst, rsdt, sic) is NOT in the init NC — group's loader
  reads it separately from per-timestep h5 in ``boundary_data_dir``.

Audits before write:
- ``np.exp(pl).mean() ∈ [80000, 120000] Pa``.
- ``zg[..., lev=50000].mean() ∈ [5400, 5700] m``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cftime
import h5py
import numpy as np
import xarray as xr

from sfno_training_group.tools._h5_keys import PLEVS_PA, SIGMA_LEVELS

logger = logging.getLogger("build_init_nc_from_v10")

N_LEVELS = 10


def _parse_synthetic_dt(s: str) -> cftime.DatetimeProlepticGregorian:
    """Parse 'YYYY-MM-DD HH:MM:SS' as proleptic_gregorian with year-zero allowed."""
    return cftime.datetime.strptime(
        s, "%Y-%m-%d %H:%M:%S",
        calendar="proleptic_gregorian", has_year_zero=True,
    )


def _read_v10_ic(src_h5: Path, ic_idx: int) -> dict[str, np.ndarray]:
    """Extract per-variable arrays for one IC timestep from a v10 stacked h5."""
    with h5py.File(src_h5, "r") as f:
        state = np.asarray(f["fields_state"][ic_idx], dtype=np.float32)  # (52, 64, 128)
        lat = np.asarray(f["lat"][:], dtype=np.float64)
        lon = np.asarray(f["lon"][:], dtype=np.float64)

    # Slice per the v10 channel layout (verified in plan §B.1):
    #   state[0]=pl, state[1]=tas, state[2..11]=ta1..ta10, state[12..21]=ua,
    #   state[22..31]=va, state[32..41]=hus, state[42..51]=zg200..zg1000.
    return {
        "pl": state[0],            # (64, 128)
        "tas": state[1],
        "ta": state[2:12],         # (10, 64, 128) sigma TOA->surface
        "ua": state[12:22],
        "va": state[22:32],
        "hus": state[32:42],
        "zg": state[42:52],        # (10, 64, 128) plev ascending
        "lat": lat,
        "lon": lon,
    }


def build_init_dataset(
    ic: dict[str, np.ndarray],
    synthetic_init_dt: cftime.datetime,
) -> xr.Dataset:
    sigma_arr = np.array(SIGMA_LEVELS, dtype=np.float64)
    lev_arr = np.array(PLEVS_PA, dtype=np.float64)
    lat = ic["lat"]
    lon = ic["lon"]

    times = np.array([synthetic_init_dt], dtype=object)

    # 3D arrays: prepend time dim (length 1).
    def _3d(arr: np.ndarray) -> np.ndarray:
        return arr[np.newaxis, ...].astype(np.float32)  # (1, lev_or_sigma, H, W)

    def _2d(arr: np.ndarray) -> np.ndarray:
        return arr[np.newaxis, ...].astype(np.float32)  # (1, H, W)

    coords = {
        "time": (("time",), times),
        "sigma": (("sigma",), sigma_arr),
        "lev": (("lev",), lev_arr),
        "lat": (("lat",), lat),
        "lon": (("lon",), lon),
    }
    data_vars = {
        "ta": xr.DataArray(_3d(ic["ta"]), dims=("time", "sigma", "lat", "lon")),
        "ua": xr.DataArray(_3d(ic["ua"]), dims=("time", "sigma", "lat", "lon")),
        "va": xr.DataArray(_3d(ic["va"]), dims=("time", "sigma", "lat", "lon")),
        "hus": xr.DataArray(_3d(ic["hus"]), dims=("time", "sigma", "lat", "lon")),
        "zg": xr.DataArray(_3d(ic["zg"]), dims=("time", "lev", "lat", "lon")),
        "pl": xr.DataArray(_2d(ic["pl"]), dims=("time", "lat", "lon")),
        "tas": xr.DataArray(_2d(ic["tas"]), dims=("time", "lat", "lon")),
    }
    ds = xr.Dataset(data_vars, coords=coords)

    # Audits.
    pl_exp_mean = float(np.exp(ic["pl"]).mean())
    if not (80_000 <= pl_exp_mean <= 120_000):
        raise RuntimeError(
            f"pl audit failed: mean(exp(pl)) = {pl_exp_mean:.0f} Pa "
            f"(expected [80000, 120000])."
        )
    zg500_idx = list(PLEVS_PA).index(50_000)
    zg500_mean = float(ic["zg"][zg500_idx].mean())
    if not (5400 <= zg500_mean <= 5700):
        raise RuntimeError(
            f"zg500 audit failed: {zg500_mean:.1f} m (expected [5400, 5700])."
        )
    logger.info("Audits passed: exp(pl_mean) ≈ %.0f Pa, zg500 mean = %.1f m",
                pl_exp_mean, zg500_mean)
    return ds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-h5", required=True, type=Path,
                        help="v10 packaged h5 (e.g. test/MOST.0121.h5).")
    parser.add_argument("--ic-idx", type=int, default=0,
                        help="Timestep index in the source file (default 0).")
    parser.add_argument("--synthetic-init-dt", required=True, type=str,
                        help='"YYYY-MM-DD HH:MM:SS" group-calendar IC datetime.')
    parser.add_argument("--out", required=True, type=Path,
                        help="Output NetCDF path.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    init_dt = _parse_synthetic_dt(args.synthetic_init_dt)
    ic = _read_v10_ic(args.src_h5, args.ic_idx)
    ds = build_init_dataset(ic, init_dt)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"dtype": "float32", "_FillValue": None} for v in ds.data_vars}
    encoding["time"] = {
        "units": f"hours since {args.synthetic_init_dt}",
        "calendar": "proleptic_gregorian",
    }
    ds.to_netcdf(args.out, encoding=encoding)
    logger.info("Wrote %s (%d KB)", args.out, args.out.stat().st_size // 1024)
    return 0


if __name__ == "__main__":
    sys.exit(main())
