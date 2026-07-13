"""Compute mean and std for variables missing from the normalize nc files
and patch them in-place.

Reads the missing variable from every HDF5 training file, computes a
running mean and std across all timesteps and spatial positions, then
writes the result into the existing normalize_mean.nc and normalize_std.nc
using the same variable shape as the other boundary variables in those files.

Usage (login node, no GPU needed):
    python compute_missing_stats.py --config configs/bench_midway.yaml

The script will:
  1. Inspect the normalize file to determine the expected variable shape.
  2. Iterate over all training-year HDF5 files and accumulate stats for
     any varying_boundary_variables that are missing from the normalize file.
  3. Write the new variables into the nc files.

Dry-run (inspect only, no writes):
    python compute_missing_stats.py --config configs/bench_midway.yaml --dry-run
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

from common.utils import get_yaml


def get_missing_vars(mean_path, varying_vars):
    with xr.open_dataset(mean_path, engine="h5netcdf") as ds:
        return [v for v in varying_vars if v not in ds]


def get_reference_shape(mean_path, varying_vars):
    """Return the shape used by existing boundary variables in the nc file."""
    with xr.open_dataset(mean_path, engine="h5netcdf") as ds:
        for v in varying_vars:
            if v in ds:
                return ds[v].shape, ds[v].dims
    return (), ()


def iter_h5_files(data_dir, year_start, year_end, variable):
    """Yield numpy arrays for `variable` from every HDF5 file in the year range."""
    data_dir = Path(data_dir)
    for year in range(year_start, year_end):
        year_files = sorted(data_dir.glob(f"{year}_*.h5"))
        for path in year_files:
            with h5py.File(path, "r") as f:
                if "input" not in f or variable not in f["input"]:
                    print(f"  SKIP {path.name} — '{variable}' not in input group")
                    continue
                yield np.array(f["input"][variable], dtype=np.float32)


def compute_mean_std(data_dir, year_start, year_end, variable, ref_shape):
    """Welford online algorithm for numerically stable mean and variance."""
    n = 0
    mean = 0.0
    M2 = 0.0

    files_seen = 0
    for arr in iter_h5_files(data_dir, year_start, year_end, variable):
        files_seen += 1
        for val in arr.ravel():
            n += 1
            delta = val - mean
            mean += delta / n
            M2 += delta * (val - mean)

    if files_seen == 0:
        raise RuntimeError(f"No HDF5 files contained '{variable}'")

    std = np.sqrt(M2 / n) if n > 1 else 0.0
    print(f"  {variable}: mean={mean:.6f}  std={std:.6f}  (n={n:,} from {files_seen} files)")
    return float(mean), float(std)


def patch_nc(path, variable, value, ref_shape, ref_dims, dry_run):
    """Add `variable` with scalar `value` to the nc file, matching ref shape."""
    if dry_run:
        print(f"  [dry-run] would write {variable}={value:.6f} into {path}")
        return

    ds = xr.open_dataset(path, engine="h5netcdf")

    if ref_shape == () or ref_shape == (1,):
        new_var = xr.DataArray(np.array([value], dtype=np.float32), dims=ref_dims or ["scalar"])
    else:
        # Broadcast scalar to the same shape as existing boundary vars.
        data = np.full(ref_shape, value, dtype=np.float32)
        new_var = xr.DataArray(data, dims=ref_dims)

    ds[variable] = new_var
    # Write to a temp file then replace (xarray can't overwrite in-place).
    tmp = path + ".tmp"
    ds.to_netcdf(tmp, engine="h5netcdf")
    ds.close()
    os.replace(tmp, path)
    print(f"  Written {variable} → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   required=True)
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print what would be done without writing anything")
    args = parser.parse_args()

    cfg  = get_yaml(args.config)
    data = cfg["data"]

    mean_path    = data["mean_path"]
    std_path     = data["std_path"]
    data_dir     = data["data_dir"]
    year_start   = data["train_year_start"]
    year_end     = data["train_year_end"]
    varying_vars = data.get("varying_boundary_variables", [])

    print(f"\nMean file : {mean_path}")
    print(f"Std file  : {std_path}")
    print(f"Data dir  : {data_dir}")
    print(f"Years     : {year_start}–{year_end}\n")

    missing = get_missing_vars(mean_path, varying_vars)
    if not missing:
        print("Nothing missing — normalize files are complete.")
        return

    print(f"Missing variables: {missing}\n")

    ref_shape, ref_dims = get_reference_shape(mean_path, varying_vars)
    print(f"Reference shape from existing boundary vars: {ref_shape}  dims={ref_dims}\n")

    for var in missing:
        print(f"Computing stats for '{var}' ...")
        mean_val, std_val = compute_mean_std(data_dir, year_start, year_end, var, ref_shape)
        patch_nc(mean_path, var, mean_val, ref_shape, ref_dims, args.dry_run)
        patch_nc(std_path,  var, std_val,  ref_shape, ref_dims, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
