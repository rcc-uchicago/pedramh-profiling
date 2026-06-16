#!/usr/bin/env python3
"""Build per-IC single-timestep NetCDFs from per-timestep h5 (contingency C-B, §3 P-7).

Reads ``<data_root>/h5/sigma_data/<Y>_<s:04d>.h5`` and writes a 1-timestep
NetCDF at ``<run_root>/inference/ic_nc/<Y>_<s:04d>.nc`` with the schema
expected by upstream's ``get_data_given_path_nc``:

  - ``time(1)`` coord = ``cftime.DatetimeProlepticGregorian(Y, 1, 1, 0) + s × 6h``,
    with ``units = "hours since <Y>-01-01 00:00:00"``,
    ``calendar = "proleptic_gregorian"`` so ``nc.num2date`` round-trips
    cleanly back to the same cftime object.
  - ``sigma_lev(10)`` coord = yaml's ``sigma_levels``;
  - ``plev(10)`` coord = yaml's ``levels``;
  - ``ta, ua, va, hus(time=1, sigma_lev=10, lat=64, lon=128)`` from
    ``input/<var>_<sigma>`` keys in h5;
  - ``zg(time=1, plev=10, lat=64, lon=128)`` from ``input/zg_<plev>``
    keys (h5 has 13 plev levels; we subselect the 10 the yaml requests);
  - ``pl, tas(time=1, lat=64, lon=128)`` from ``input/pl`` and
    ``input/tas``.

This is intended as a Phase-1-only loader-compatibility shim; the full
production data path (boundary forcing, 6h step iteration) still uses
the per-timestep h5 files via the upstream loader's `_get_boundary_data`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import cftime
import h5py
import netCDF4 as nc
import numpy as np


_LOG = logging.getLogger("build_ic_nc_from_h5")


# Per the 5410 yaml (verified 2026-05-07).
SIGMA_LEVELS = (
    0.03830000013113022, 0.11910000443458557, 0.21085000783205032,
    0.3168500065803528, 0.4368000030517578, 0.5668000280857086,
    0.6993500888347626, 0.8233500719070435, 0.9240999817848206,
    0.983299970626831,
)
PLEV_LEVELS = (
    20000.0, 25000.0, 30000.0, 40000.0, 50000.0,
    60000.0, 70000.0, 85000.0, 92500.0, 100000.0,
)
SIGMA_VARS = ("ta", "ua", "va", "hus")
ZG_VAR = "zg"
SURFACE_VARS = ("pl", "tas")
LAT_SIZE = 64
LON_SIZE = 128
DATA_ROOT = Path("/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52")
H5_DIR = DATA_ROOT / "h5" / "sigma_data"


def _h5_path(Y: int, s: int) -> Path:
    return H5_DIR / f"{Y}_{s:04d}.h5"


def init_datetime_for(Y: int, s: int) -> cftime.DatetimeProlepticGregorian:
    base = cftime.DatetimeProlepticGregorian(Y, 1, 1, 0, has_year_zero=True)
    return base + dt.timedelta(hours=s * 6)


def _stack_sigma(h5: h5py.File, var: str) -> np.ndarray:
    """Stack `<var>_<sigma>` keys in yaml's sigma order → (10, 64, 128)."""
    arr = np.empty((len(SIGMA_LEVELS), LAT_SIZE, LON_SIZE), dtype=np.float32)
    for i, lev in enumerate(SIGMA_LEVELS):
        key = f"input/{var}_{lev}"
        arr[i] = h5[key][...]
    return arr


def _stack_plev(h5: h5py.File, var: str) -> np.ndarray:
    """Stack `<var>_<plev>` keys in yaml's plev order → (10, 64, 128).

    Note: h5 keys store the float as e.g. ``zg_5000.0``; we render with
    the same ``str(float)`` form to look them up.
    """
    arr = np.empty((len(PLEV_LEVELS), LAT_SIZE, LON_SIZE), dtype=np.float32)
    for i, lev in enumerate(PLEV_LEVELS):
        key = f"input/{var}_{lev}"
        arr[i] = h5[key][...]
    return arr


def build_one_ic_nc(Y: int, s: int, out_path: Path) -> Path:
    """Write ``<run_root>/inference/ic_nc/<Y>_<s:04d>.nc`` from h5 input."""
    h5_path = _h5_path(Y, s)
    if not h5_path.is_file():
        raise FileNotFoundError(f"h5 source missing: {h5_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    init_dt = init_datetime_for(Y, s)
    time_units = f"hours since {Y:04d}-01-01 00:00:00"

    with h5py.File(h5_path, "r") as h5, nc.Dataset(out_path, "w", format="NETCDF4") as ds:
        # Dimensions
        ds.createDimension("time", 1)
        ds.createDimension("sigma_lev", len(SIGMA_LEVELS))
        ds.createDimension("plev", len(PLEV_LEVELS))
        ds.createDimension("lat", LAT_SIZE)
        ds.createDimension("lon", LON_SIZE)

        # Coord vars
        t = ds.createVariable("time", "f8", ("time",))
        t.units = time_units
        t.calendar = "proleptic_gregorian"
        t[:] = nc.date2num([init_dt], units=time_units, calendar="proleptic_gregorian")

        sl = ds.createVariable("sigma_lev", "f8", ("sigma_lev",))
        sl[:] = np.array(SIGMA_LEVELS, dtype=np.float64)
        pl_dim = ds.createVariable("plev", "f8", ("plev",))
        pl_dim[:] = np.array(PLEV_LEVELS, dtype=np.float64)

        # Upper-air sigma vars
        for v in SIGMA_VARS:
            arr = _stack_sigma(h5, v)
            var = ds.createVariable(
                v, "f4", ("time", "sigma_lev", "lat", "lon"), zlib=False
            )
            var[0, :, :, :] = arr

        # zg on plev
        zg_arr = _stack_plev(h5, ZG_VAR)
        zg = ds.createVariable(
            "zg", "f4", ("time", "plev", "lat", "lon"), zlib=False
        )
        zg[0, :, :, :] = zg_arr

        # Surface vars
        for v in SURFACE_VARS:
            arr = h5[f"input/{v}"][...]
            var = ds.createVariable(v, "f4", ("time", "lat", "lon"), zlib=False)
            var[0, :, :] = arr

        ds.title = f"5410 IC NetCDF for (Y={Y}, s={s}) — built from h5"
        ds.source = str(h5_path)

    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--year", type=int, help="Single Y (omit with --all to do 121..128)")
    p.add_argument("--sample", type=int,
                   help="Single s (omit with --all to do 12 ICs/year)")
    p.add_argument("--all", action="store_true",
                   help="Build all 96 (Y, s) tuples in the run plan")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output dir (typically <run_root>/inference/ic_nc)")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = _parse_args()

    if args.all:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from sfno_inference_5410.ic_offsets import nwp_ic_offsets_5410
        targets: list[tuple[int, int]] = []
        for Y in range(121, 129):
            n_samples = 1464 if Y in (124, 128) else 1460
            for s in nwp_ic_offsets_5410(n_samples):
                targets.append((Y, s))
    else:
        if args.year is None or args.sample is None:
            raise SystemExit("either --all or both --year and --sample required")
        targets = [(args.year, args.sample)]

    for i, (Y, s) in enumerate(targets):
        out_path = args.out_dir / f"{Y}_{s:04d}.nc"
        build_one_ic_nc(Y, s, out_path)
        _LOG.info("[%2d/%2d] wrote %s", i + 1, len(targets), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
