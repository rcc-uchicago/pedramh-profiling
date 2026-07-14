"""Recompute mean/std NetCDF files from converted group-format h5.

Phase B.2 of the group-code training track plan v5.

Streams per-(var, level) sums and sum-of-squares over a *flat-dir* of
``<year>_<idx:04>.h5`` files for a chosen list of *train* years and emits two
NetCDF files at ``<data_dir>/data_train_mean.nc`` and ``data_train_std.nc``,
matching the schema expected by group's ``GetDataset.load_mean_std``
(``utils/data_loader_multifiles.py:751-780``):

- Coord ``Z`` (length 10): zg pressure levels in Pa.
- Coord ``Z_2`` (length 10): PlaSim native sigma values (TOA -> surface).
- Variables ``ta, ua, va, hus`` on ``Z_2``, shape (10,) — float32.
- Variable ``zg`` on ``Z``, shape (10,) — float32.
- Variables ``pl, tas, pr_6h, lsm, sg, z0, sst, rsdt, sic`` — scalar (0-D), float32.
- Coords ``lat, lon`` written for completeness.
- File-level attribute ``source = "recomputed_from_converted_h5"``.

CRITICAL: stats are recomputed from the converted h5 per the user's
"Verify external-loader contracts" guidance — we never mechanically transform
existing Makani .npy stats to NetCDF, since the conversion step could silently
shift values (and at minimum rebases per-channel positional indices into
per-(var, level) keys).

Audits before write:
- ``zg_50000.0`` mean ∈ [5400, 5700] m (zg500 sanity range from v10 packager).
- ``np.exp(pl_mean)`` ∈ [80000, 120000] Pa.
- All stds > 1e-12 (no exact-zero variance).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

from sfno_training_group.tools._h5_keys import (
    PLEVS_PA,
    SIGMA_LEVELS,
    h5_key,
)

logger = logging.getLogger("build_group_stats_netcdf")

UPPER_AIR_VARS_SIGMA = ("ta", "ua", "va", "hus")
UPPER_AIR_VARS_PLEV = ("zg",)
SURFACE_VARS = ("pl", "tas")
DIAG_VARS = ("pr_6h",)
FORCING_VARS = ("lsm", "sg", "z0", "sst", "rsdt", "sic")
N_LEVELS = 10


def _all_keys() -> list[str]:
    keys: list[str] = []
    for var in UPPER_AIR_VARS_SIGMA:
        for i in range(N_LEVELS):
            keys.append(h5_key(var, i))
    for i in range(N_LEVELS):
        keys.append(h5_key("zg", i))
    keys.extend(SURFACE_VARS + DIAG_VARS + FORCING_VARS)
    return keys


def _native_count_for_year(manifest: dict | None, year: int, files_full: list[Path]) -> int:
    """How many files of <year>_*.h5 to actually consume (skip padded indices).

    Reads manifest.years[*].n_timesteps_native if present (Phase F+); falls back
    to len(files_full) for legacy manifests / smoke runs without padding.
    """
    if manifest is None:
        return len(files_full)
    by_year = {y["year"]: y for y in manifest.get("years", [])}
    if year not in by_year:
        return len(files_full)
    return int(by_year[year].get("n_timesteps_native", len(files_full)))


def _accumulate(data_dir: Path, train_years: list[int]) -> tuple[dict[str, dict[str, float | int]], dict]:
    """Stream over all train files (NATIVE only); per-key accumulate sum, sum_sq, count.

    Returns (accs, provenance) where provenance describes which native frames
    were consumed per year. Skips files with idx >= n_timesteps_native if
    the manifest reports padding metadata (Phase F.0.b contract).
    """
    keys = _all_keys()
    accs: dict[str, dict[str, float | int]] = {
        k: {"sum": 0.0, "sum_sq": 0.0, "count": 0} for k in keys
    }
    manifest_path = data_dir / "_v10_calendar_manifest.json"
    manifest: dict | None = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())

    native_by_year: dict[int, int] = {}
    skipped_by_year: dict[int, int] = {}
    files: list[Path] = []
    for year in train_years:
        files_full = sorted(data_dir.glob(f"{year}_*.h5"))
        if not files_full:
            raise RuntimeError(f"No files found for year {year} under {data_dir}")
        n_native = _native_count_for_year(manifest, year, files_full)
        files_year = files_full[:n_native]
        native_by_year[year] = n_native
        skipped_by_year[year] = max(0, len(files_full) - n_native)
        files.extend(files_year)
        if skipped_by_year[year] > 0:
            logger.info("Year %d: skipping %d padded files (idx >= %d)",
                        year, skipped_by_year[year], n_native)
    if not files:
        raise RuntimeError(f"No native files found for years {train_years} under {data_dir}")
    logger.info("Streaming over %d native files (%d keys per file)", len(files), len(keys))

    for fi, path in enumerate(files):
        with h5py.File(path, "r") as f:
            grp = f["input"]
            for k in keys:
                arr = np.asarray(grp[k], dtype=np.float64)
                accs[k]["sum"] += float(arr.sum())
                accs[k]["sum_sq"] += float((arr * arr).sum())
                accs[k]["count"] += int(arr.size)
        if (fi + 1) % 200 == 0 or fi == len(files) - 1:
            logger.info("  processed %d/%d files", fi + 1, len(files))

    # Did the manifest already record any padded year? If so we still skipped the
    # padded indices via files[:n_native]; built_before_padding=False reflects
    # that the data_dir IS post-padded, but we used the manifest to skip cleanly.
    has_padded_years = any(
        manifest and any("n_timesteps_padded" in y for y in manifest.get("years", []))
        for _ in [None]
    ) if manifest else False
    provenance = {
        "native_timesteps_by_year": {str(y): native_by_year[y] for y in train_years},
        "native_timesteps_total": int(sum(native_by_year.values())),
        "manifest_skip_padded_used": any(v > 0 for v in skipped_by_year.values()),
        "built_before_padding": not has_padded_years,
    }
    return accs, provenance


def _finalize(accs: dict[str, dict[str, float | int]]) -> dict[str, tuple[float, float]]:
    """sum, sum_sq, count -> (mean, std). Std is population std (ddof=0)."""
    out: dict[str, tuple[float, float]] = {}
    for k, a in accs.items():
        n = a["count"]
        if n == 0:
            raise RuntimeError(f"key {k!r}: empty accumulator")
        mean = a["sum"] / n
        var = max(a["sum_sq"] / n - mean * mean, 0.0)  # guard tiny negatives from FP
        std = float(np.sqrt(var))
        out[k] = (float(mean), std)
        if std < 1e-12:
            raise RuntimeError(f"key {k!r}: std={std} too small — degenerate input?")
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
    """Stamp F.0.b provenance attrs onto a stats/climatology NetCDF."""
    ds.attrs["source"] = "recomputed_from_converted_h5"
    # JSON-encode dict because NetCDF attrs must be scalar-compatible.
    ds.attrs["native_timesteps_by_year"] = json.dumps(provenance["native_timesteps_by_year"])
    ds.attrs["native_timesteps_total"] = int(provenance["native_timesteps_total"])
    ds.attrs["manifest_skip_padded_used"] = int(bool(provenance["manifest_skip_padded_used"]))
    ds.attrs["built_before_padding"] = int(bool(provenance["built_before_padding"]))
    ds.attrs["git_sha"] = _git_sha()
    ds.attrs["build_timestamp_utc"] = _dt.datetime.now(_dt.UTC).isoformat()


def _build_xr_datasets(
    stats: dict[str, tuple[float, float]],
    lat: np.ndarray,
    lon: np.ndarray,
    provenance: dict,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Pack scalar / per-level stats into mean and std xr.Datasets."""

    def per_var(stats_idx: int) -> dict[str, xr.DataArray]:
        # stats_idx: 0=mean, 1=std
        d: dict[str, xr.DataArray] = {}
        # Sigma upper-air
        for var in UPPER_AIR_VARS_SIGMA:
            arr = np.array(
                [stats[h5_key(var, i)][stats_idx] for i in range(N_LEVELS)],
                dtype=np.float32,
            )
            d[var] = xr.DataArray(arr, dims=("Z_2",))
        # Pressure-level zg
        arr_zg = np.array(
            [stats[h5_key("zg", i)][stats_idx] for i in range(N_LEVELS)],
            dtype=np.float32,
        )
        d["zg"] = xr.DataArray(arr_zg, dims=("Z",))
        # Scalars
        for v in SURFACE_VARS + DIAG_VARS + FORCING_VARS:
            d[v] = xr.DataArray(np.float32(stats[v][stats_idx]))
        return d

    coords = {
        "Z": np.array(PLEVS_PA, dtype=np.float64),
        "Z_2": np.array(SIGMA_LEVELS, dtype=np.float64),
        "lat": (("lat",), lat),
        "lon": (("lon",), lon),
    }
    mean_ds = xr.Dataset(per_var(0), coords=coords)
    std_ds = xr.Dataset(per_var(1), coords=coords)
    _attach_provenance(mean_ds, provenance)
    _attach_provenance(std_ds, provenance)
    return mean_ds, std_ds


def _read_lat_lon(data_dir: Path, sample_year: int) -> tuple[np.ndarray, np.ndarray]:
    """The converted h5 doesn't carry lat/lon. Read them from the v10 source via manifest."""
    manifest = json.loads((data_dir / "_v10_calendar_manifest.json").read_text())
    src_path = next(y["src_path"] for y in manifest["years"] if y["year"] == sample_year)
    with h5py.File(src_path, "r") as f:
        return np.asarray(f["lat"][:], dtype=np.float64), np.asarray(f["lon"][:], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="Flat dir of converted <year>_<idx>.h5 + manifest.")
    parser.add_argument("--train-years", type=int, nargs="+", required=True,
                        help="Years to include in stats (typically train years).")
    parser.add_argument("--mean-out", type=Path, default=None,
                        help="Override mean output path (default: <data_dir>/data_train_mean.nc).")
    parser.add_argument("--std-out", type=Path, default=None,
                        help="Override std output path (default: <data_dir>/data_train_std.nc).")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    accs, provenance = _accumulate(args.data_dir, args.train_years)
    stats = _finalize(accs)

    # Audits.
    pl_mean = stats["pl"][0]
    if not (np.log(80_000) <= pl_mean <= np.log(120_000)):
        raise RuntimeError(f"pl mean audit failed: {pl_mean} (exp={np.exp(pl_mean):.0f} Pa)")
    zg500_mean = stats["zg_50000.0"][0]
    if not (5400 <= zg500_mean <= 5700):
        raise RuntimeError(f"zg500 mean audit failed: {zg500_mean} m (expected [5400, 5700])")
    logger.info("Audits passed: exp(pl_mean)=%.0f Pa, zg500_mean=%.1f m",
                np.exp(pl_mean), zg500_mean)

    lat, lon = _read_lat_lon(args.data_dir, args.train_years[0])
    mean_ds, std_ds = _build_xr_datasets(stats, lat, lon, provenance)

    mean_out = args.mean_out or (args.data_dir / "data_train_mean.nc")
    std_out = args.std_out or (args.data_dir / "data_train_std.nc")
    mean_ds.to_netcdf(mean_out)
    std_ds.to_netcdf(std_out)
    logger.info("Wrote %s and %s", mean_out, std_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
