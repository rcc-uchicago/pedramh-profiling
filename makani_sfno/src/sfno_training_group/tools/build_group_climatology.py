"""Build a group-format climatology.nc from converted train h5.

Phase B.3 of the group-code training track plan v5.

Group's ``Trainer.get_dataset`` (``train.py:459-466``) unconditionally opens
``params.climatology_file`` with ``xr.coders.CFDatetimeCoder(use_cftime=True)``
and renames ``time`` -> ``dayofyear``. Even though sigma-level training
auto-disables ACC diagnostics later (``train.py:3771-3776``), the open happens
during Trainer construction, so a valid file is required for any smoke train.

This builder produces a *daily-resolution* climatology (1 sample per day,
mean over the 4 timesteps in each day across all train years):

- Coord ``time`` (length 366): Jan 1..Dec 31 of a leap reference year (year 4,
  proleptic_gregorian) so all 366 day-of-year slots are present.
- Coord ``Z`` (10), ``Z_2`` (10), ``lat`` (64), ``lon`` (128).
- Variables ``ta(time, Z_2, lat, lon)``, ``ua, va, hus`` on ``Z_2``.
- Variable ``zg(time, Z, lat, lon)`` on ``Z``.
- Variables ``pl, tas, pr_6h, lsm, sg, z0, sst, rsdt, sic`` shaped
  ``(time, lat, lon)``.
- All float32.

For doys without data (PlaSim's 1455-step year stops a few timesteps before
the synthetic Dec 31), fill with the per-key annual mean over present doys.
ACC / GIF / spectra diagnostics that would consume this climatology are
auto-disabled by ``train.py:3771-3776`` whenever ``use_sigma_levels=True`` —
which holds for both Phase 1 smoke AND Phase F production. Daily resolution
is therefore sufficient; the missing-doy fill is a robustness measure only.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import cftime
import h5py
import numpy as np
import xarray as xr

from sfno_training_group.tools._h5_keys import (
    PLEVS_PA,
    SIGMA_LEVELS,
    h5_key,
)

logger = logging.getLogger("build_group_climatology")

UPPER_AIR_VARS_SIGMA = ("ta", "ua", "va", "hus")
SURFACE_VARS = ("pl", "tas")
DIAG_VARS = ("pr_6h",)
FORCING_VARS = ("lsm", "sg", "z0", "sst", "rsdt", "sic")
N_LEVELS = 10
H, W = 64, 128
N_DAYS = 366  # proleptic_gregorian leap reference year
DATA_TIMEDELTA_HOURS = 6


def _build_doy_index(year: int, n_timesteps: int) -> np.ndarray:
    """Map idx -> dayofyear (1..366) for a synthetic proleptic_gregorian year."""
    jan_1 = cftime.DatetimeProlepticGregorian(year, 1, 1, has_year_zero=True)
    doys = np.empty(n_timesteps, dtype=np.int32)
    for idx in range(n_timesteps):
        dt = jan_1 + timedelta(hours=DATA_TIMEDELTA_HOURS * idx)
        doys[idx] = (dt - cftime.DatetimeProlepticGregorian(year, 1, 1, has_year_zero=True)).days + 1
    return doys


def _accumulate(data_dir: Path, train_years: list[int]) -> tuple[
    dict[str, np.ndarray], np.ndarray
]:
    """Stream per-key (sum over (T, H, W)) into per-doy accumulators.

    Returns:
        sums: key -> array of shape (366, H, W), float64.
        counts: array of shape (366,), int64. (Uniform across keys.)
    """
    sums: dict[str, np.ndarray] = {}
    counts = np.zeros(N_DAYS, dtype=np.int64)

    keys: list[str] = []
    for var in UPPER_AIR_VARS_SIGMA:
        for i in range(N_LEVELS):
            keys.append(h5_key(var, i))
    for i in range(N_LEVELS):
        keys.append(h5_key("zg", i))
    keys.extend(SURFACE_VARS + DIAG_VARS + FORCING_VARS)

    for k in keys:
        sums[k] = np.zeros((N_DAYS, H, W), dtype=np.float64)

    # Phase F.0.b: skip padded indices via manifest if present.
    manifest_path = data_dir / "_v10_calendar_manifest.json"
    manifest: dict | None = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    by_year_meta = {y["year"]: y for y in (manifest or {}).get("years", [])}

    native_by_year: dict[int, int] = {}
    skipped_by_year: dict[int, int] = {}
    for year in train_years:
        files_full = sorted(data_dir.glob(f"{year}_*.h5"))
        if not files_full:
            raise RuntimeError(f"No files for year {year} under {data_dir}")
        if year in by_year_meta:
            n_native = int(by_year_meta[year].get("n_timesteps_native", len(files_full)))
        else:
            n_native = len(files_full)
        files = files_full[:n_native]
        native_by_year[year] = n_native
        skipped_by_year[year] = max(0, len(files_full) - n_native)
        if skipped_by_year[year] > 0:
            logger.info("Year %d: skipping %d padded files (idx >= %d)",
                        year, skipped_by_year[year], n_native)
        n_timesteps = len(files)
        doys = _build_doy_index(year, n_timesteps)
        logger.info("year %d: %d native files, doy range [%d, %d]",
                    year, n_timesteps, int(doys.min()), int(doys.max()))

        for idx, path in enumerate(files):
            doy = int(doys[idx])
            counts[doy - 1] += 1
            with h5py.File(path, "r") as f:
                grp = f["input"]
                for k in keys:
                    sums[k][doy - 1] += np.asarray(grp[k], dtype=np.float64)
            if (idx + 1) % 200 == 0 or idx == n_timesteps - 1:
                logger.info("  year %d: processed %d/%d", year, idx + 1, n_timesteps)
    has_padded_years = any("n_timesteps_padded" in y for y in (manifest or {}).get("years", []))
    provenance = {
        "native_timesteps_by_year": {str(y): native_by_year[y] for y in train_years},
        "native_timesteps_total": int(sum(native_by_year.values())),
        "manifest_skip_padded_used": any(v > 0 for v in skipped_by_year.values()),
        "built_before_padding": not has_padded_years,
    }
    return sums, counts, provenance


def _finalize(
    sums: dict[str, np.ndarray], counts: np.ndarray
) -> dict[str, np.ndarray]:
    """sums/counts -> per-doy means; fill missing doys with the per-key annual mean."""
    out: dict[str, np.ndarray] = {}
    has_data = counts > 0
    if not has_data.any():
        raise RuntimeError("counts all zero")
    for k, s in sums.items():
        # safe divide where counts>0
        means = np.zeros_like(s, dtype=np.float64)
        means[has_data] = s[has_data] / counts[has_data, None, None]
        # fill missing doys with annual mean
        if not has_data.all():
            annual_mean = means[has_data].mean(axis=0)  # (H, W)
            means[~has_data] = annual_mean
        out[k] = means.astype(np.float32)
    return out


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _attach_provenance(ds: xr.Dataset, provenance: dict) -> None:
    ds.attrs["source"] = "recomputed_from_converted_h5"
    ds.attrs["note"] = "daily resolution; reference year 4 (leap, proleptic_gregorian)"
    ds.attrs["native_timesteps_by_year"] = json.dumps(provenance["native_timesteps_by_year"])
    ds.attrs["native_timesteps_total"] = int(provenance["native_timesteps_total"])
    ds.attrs["manifest_skip_padded_used"] = int(bool(provenance["manifest_skip_padded_used"]))
    ds.attrs["built_before_padding"] = int(bool(provenance["built_before_padding"]))
    ds.attrs["git_sha"] = _git_sha()
    ds.attrs["build_timestamp_utc"] = _dt.datetime.now(_dt.UTC).isoformat()


def _build_dataset(
    means: dict[str, np.ndarray], lat: np.ndarray, lon: np.ndarray,
    provenance: dict,
) -> xr.Dataset:
    # Reference year 4 = leap year in proleptic_gregorian.
    ref_year = 4
    times = np.array(
        [cftime.DatetimeProlepticGregorian(ref_year, 1, 1, has_year_zero=True)
         + timedelta(days=d) for d in range(N_DAYS)],
        dtype=object,
    )

    coords = {
        "time": (("time",), times),
        "Z": (("Z",), np.array(PLEVS_PA, dtype=np.float64)),
        "Z_2": (("Z_2",), np.array(SIGMA_LEVELS, dtype=np.float64)),
        "lat": (("lat",), lat),
        "lon": (("lon",), lon),
    }

    data_vars: dict[str, xr.DataArray] = {}
    # Sigma upper-air: stack along new Z_2 axis.
    for var in UPPER_AIR_VARS_SIGMA:
        stack = np.stack(
            [means[h5_key(var, i)] for i in range(N_LEVELS)], axis=1
        )  # (time, Z_2, lat, lon)
        data_vars[var] = xr.DataArray(stack, dims=("time", "Z_2", "lat", "lon"))
    # zg on Z
    stack_zg = np.stack(
        [means[h5_key("zg", i)] for i in range(N_LEVELS)], axis=1
    )
    data_vars["zg"] = xr.DataArray(stack_zg, dims=("time", "Z", "lat", "lon"))
    # 2D surface/diag/forcing
    for v in SURFACE_VARS + DIAG_VARS + FORCING_VARS:
        data_vars[v] = xr.DataArray(means[v], dims=("time", "lat", "lon"))

    ds = xr.Dataset(data_vars, coords=coords)
    _attach_provenance(ds, provenance)
    return ds


def _read_lat_lon(data_dir: Path, sample_year: int) -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads((data_dir / "_v10_calendar_manifest.json").read_text())
    src_path = next(y["src_path"] for y in manifest["years"] if y["year"] == sample_year)
    with h5py.File(src_path, "r") as f:
        return np.asarray(f["lat"][:], dtype=np.float64), np.asarray(f["lon"][:], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--train-years", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=None,
                        help="Default: <data_dir>/climatology.nc")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    sums, counts, provenance = _accumulate(args.data_dir, args.train_years)
    logger.info("doys with data: %d/%d", int((counts > 0).sum()), N_DAYS)
    means = _finalize(sums, counts)

    lat, lon = _read_lat_lon(args.data_dir, args.train_years[0])
    ds = _build_dataset(means, lat, lon, provenance)

    out = args.out or (args.data_dir / "climatology.nc")
    encoding = {v: {"dtype": "float32", "_FillValue": None} for v in ds.data_vars}
    encoding["time"] = {"units": "days since 0001-01-01", "calendar": "proleptic_gregorian"}
    ds.to_netcdf(out, encoding=encoding)
    logger.info("Wrote %s (size: %d MB)", out, out.stat().st_size // 1024**2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
