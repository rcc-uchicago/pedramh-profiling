#!/usr/bin/env python3
"""Backfill truth_sic into existing own-track per-IC inference NetCDFs.

For each `inference/nwp/MOST.*_icXXX.nc` in `--in-root`:

  1. Read attrs (`ic_file`, `ic_sample_idx`) and the `lead_time` coord.
  2. For every lead h, locate the source sample step
     `step = ic_sample_idx + h // 6` and read
     `MOST.{YEAR}.h5[forcing][step, 5, :, :]` from the matching
     test_holdout file. If the step crosses into Y+1, wrap onto
     `MOST.{YEAR+1}.h5`.
  3. Stack into `truth_sic[K, H, W]`.
  4. Write a new NetCDF at `--out-root/inference/nwp/<same-basename>`
     with the original variables + `truth_sic`.
  5. Symlink the climatology from `--in-root/baselines/` so the
     downstream `score_nwp.py --clim-nc` invocation has it.

Only own-track NCs need this; 5410 NetCDFs are produced by the adapter,
which now writes `truth_sic` natively (so re-running `score_5410.py
--force` is the equivalent for that track).
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import xarray as xr


logger = logging.getLogger("backfill_truth_sic")

_SIC_FORCING_INDEX = 5  # forcing channel order: lsm, sg, z0, sst, rsdt, sic


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-root", type=Path, required=True,
                   help="Existing eval OUT_ROOT (must contain inference/nwp/*.nc)")
    p.add_argument("--out-root", type=Path, required=True,
                   help="New eval OUT_ROOT for augmented NCs (created if missing)")
    p.add_argument("--test-holdout", type=Path, required=True,
                   help="Dir with MOST.{YEAR}.h5 holding `forcing` and `time_plasim`")
    p.add_argument("--limit-ics", type=int, default=None,
                   help="Process at most N ICs (for smoke runs)")
    p.add_argument("--clim-name", type=str, default="climatology_proleptic.nc",
                   help="Climatology filename to symlink under baselines/")
    return p.parse_args()


def _year_of_ic_file(ic_file: str) -> int:
    """Parse '0121' from 'MOST.0121.h5'."""
    m = re.match(r"MOST\.(\d{4})\.h5", ic_file)
    if m is None:
        raise ValueError(f"unparseable ic_file attr: {ic_file!r}")
    return int(m.group(1))


def _holdout_path(holdout: Path, year: int) -> Path:
    return holdout / f"MOST.{year:04d}.h5"


def _samples_in_year(holdout: Path, year: int) -> int:
    with h5py.File(_holdout_path(holdout, year), "r") as f:
        return int(f["time_plasim"].shape[0])


def _read_sic_at_step(holdout: Path, year: int, step: int,
                      *, samples_cache: dict[int, int]) -> np.ndarray:
    """Read forcing[step, 5, :, :] from MOST.{year}.h5, wrapping to Y+1 if step OOB.

    Returns a (H, W) float32 array in raw units (0..1, NaN over land).
    """
    if year not in samples_cache:
        samples_cache[year] = _samples_in_year(holdout, year)
    T = samples_cache[year]
    while step >= T:
        step -= T
        year += 1
        if year not in samples_cache:
            samples_cache[year] = _samples_in_year(holdout, year)
        T = samples_cache[year]
    with h5py.File(_holdout_path(holdout, year), "r") as f:
        return np.asarray(f["forcing"][step, _SIC_FORCING_INDEX, :, :],
                          dtype=np.float32)


def _backfill_one_nc(nc_in: Path, nc_out: Path, holdout: Path,
                     samples_cache: dict[int, int]) -> None:
    """Add truth_sic to one NC and write to nc_out."""
    nc_out.parent.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(nc_in, decode_times=False) as ds:
        ic_file = str(ds.attrs["ic_file"])
        ic_sample_idx = int(ds.attrs["ic_sample_idx"])
        leads = np.asarray(ds["lead_time"].values, dtype=np.int64)  # hours
        H = int(ds.sizes["lat"])
        W = int(ds.sizes["lon"])
        # Load the full dataset into memory so we can write it back out.
        # Per-IC NCs are <200 MB; this is fine.
        ds_loaded = ds.load()

    if "truth_sic" in ds_loaded.variables:
        # Idempotency: don't double-write. Drop and re-attach below.
        ds_loaded = ds_loaded.drop_vars("truth_sic")

    year = _year_of_ic_file(ic_file)
    K = leads.shape[0]
    truth_sic = np.empty((K, H, W), dtype=np.float32)
    for k, h in enumerate(leads):
        step = ic_sample_idx + int(h) // 6
        truth_sic[k] = _read_sic_at_step(
            holdout, year, step, samples_cache=samples_cache,
        )

    ds_loaded["truth_sic"] = (
        ("init_time", "lead_time", "lat", "lon"),
        truth_sic[np.newaxis, ...],
    )
    ds_loaded["truth_sic"].attrs["units"] = "fraction"
    ds_loaded["truth_sic"].attrs["description"] = (
        "Truth sea-ice fraction at each lead; NaN over land. "
        "Downstream tas_no_ice mask uses sic >= 0.15 to drop sea-ice cells. "
        "Backfilled by scripts/backfill_truth_sic.py."
    )

    # Encoding mirrors src/sfno_inference/nc_writer.py.
    enc_vars = ("prediction", "truth", "init_state", "truth_sic")
    encoding = {v: {"zlib": True, "complevel": 4}
                for v in enc_vars if v in ds_loaded.variables}
    ds_loaded.to_netcdf(nc_out, encoding=encoding, format="NETCDF4")


def _maybe_symlink_climatology(in_root: Path, out_root: Path,
                               clim_name: str) -> None:
    src = in_root / "baselines" / clim_name
    if not src.is_file():
        logger.warning("no climatology at %s; downstream scoring will need it set explicitly", src)
        return
    dst_dir = out_root / "baselines"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / clim_name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())
    logger.info("symlinked %s -> %s", dst, src)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    nc_dir = args.in_root / "inference" / "nwp"
    out_nc_dir = args.out_root / "inference" / "nwp"
    if not nc_dir.is_dir():
        raise SystemExit(f"no inference/nwp dir under {args.in_root}")
    ncs = sorted(nc_dir.glob("MOST.*.nc"))
    if not ncs:
        raise SystemExit(f"no MOST.*.nc in {nc_dir}")
    if args.limit_ics is not None:
        ncs = ncs[: args.limit_ics]

    if not args.test_holdout.is_dir():
        raise SystemExit(f"--test-holdout dir not found: {args.test_holdout}")

    out_nc_dir.mkdir(parents=True, exist_ok=True)
    samples_cache: dict[int, int] = {}
    logger.info("backfilling truth_sic into %d NCs (%s -> %s)",
                len(ncs), nc_dir, out_nc_dir)
    for i, nc_in in enumerate(ncs):
        nc_out = out_nc_dir / nc_in.name
        _backfill_one_nc(nc_in, nc_out, args.test_holdout, samples_cache)
        if (i + 1) % 8 == 0 or (i + 1) == len(ncs):
            logger.info("  %d/%d done (latest: %s)", i + 1, len(ncs), nc_in.name)

    _maybe_symlink_climatology(args.in_root, args.out_root, args.clim_name)
    logger.info("done. wrote %d NCs to %s", len(ncs), out_nc_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
