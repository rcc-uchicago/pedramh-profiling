"""Check that all variables listed in a config exist in the normalize nc files
and in a sample HDF5 data file.

Usage (login node, no GPU needed):
    python check_normalize_vars.py --config configs/bench_midway.yaml

Prints OK / MISSING for every variable in surface, upper_air, diagnostic,
varying_boundary, and constant_boundary groups against both the mean/std
nc files and one HDF5 data file from the first training year.
"""

import argparse
import os
import sys

import h5py
import xarray as xr

from common.utils import get_yaml


def check_nc(path, variables, label, upper_air=False, levels=None):
    print(f"\n  [{label}]  {path}")
    if not os.path.exists(path):
        print(f"    FILE NOT FOUND: {path}")
        return []
    missing = []
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        available = list(ds.data_vars)
        for var in variables:
            if var in ds:
                if upper_air and levels is not None:
                    ds_levels = list(ds["level"].values) if "level" in ds.dims else []
                    missing_lvls = [l for l in levels if l not in ds_levels]
                    if missing_lvls:
                        print(f"    MISSING LEVELS  {var}: {missing_lvls}")
                        missing.append(var)
                    else:
                        print(f"    OK              {var}")
                else:
                    print(f"    OK              {var}")
            else:
                print(f"    MISSING         {var}   (available: {available})")
                missing.append(var)
    return missing


def check_h5(path, variables, label):
    print(f"\n  [{label}]  {path}")
    if not os.path.exists(path):
        print(f"    FILE NOT FOUND: {path}")
        return []
    missing = []
    with h5py.File(path, "r") as f:
        if "input" not in f:
            print(f"    No 'input' group found. Top-level keys: {list(f.keys())}")
            return variables
        available = list(f["input"].keys())
        for var in variables:
            if var in f["input"]:
                print(f"    OK              {var}")
            else:
                print(f"    MISSING         {var}   (available: {available})")
                missing.append(var)
    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = get_yaml(args.config)
    data = cfg["data"]

    mean_path = data["mean_path"]
    std_path  = data["std_path"]
    data_dir  = data["data_dir"]
    year_start = data["train_year_start"]
    levels    = data.get("levels", [])

    upper_air_vars   = data.get("upper_air_variables", [])
    surface_vars     = data.get("surface_variables", [])
    diagnostic_vars  = data.get("diagnostic_variables", [])
    varying_bnd_vars = data.get("varying_boundary_variables", [])
    constant_bnd_vars= data.get("constant_boundary_variables", [])

    all_nc_vars = upper_air_vars + surface_vars + diagnostic_vars + varying_bnd_vars + constant_bnd_vars
    all_h5_vars = upper_air_vars + surface_vars + diagnostic_vars + varying_bnd_vars + constant_bnd_vars

    sample_h5 = os.path.join(data_dir, f"{year_start}_0000.h5")

    print("=" * 64)
    print(f"  Config : {args.config}")
    print(f"  Mean   : {mean_path}")
    print(f"  Std    : {std_path}")
    print(f"  Sample : {sample_h5}")
    print("=" * 64)

    all_missing = []

    print("\n--- normalize_mean.nc ---")
    all_missing += check_nc(mean_path, surface_vars,     "surface")
    all_missing += check_nc(mean_path, diagnostic_vars,  "diagnostic")
    all_missing += check_nc(mean_path, varying_bnd_vars, "varying_boundary")
    all_missing += check_nc(mean_path, upper_air_vars,   "upper_air", upper_air=True, levels=levels)

    print("\n--- normalize_std.nc ---")
    check_nc(std_path, surface_vars,     "surface")
    check_nc(std_path, diagnostic_vars,  "diagnostic")
    check_nc(std_path, varying_bnd_vars, "varying_boundary")
    check_nc(std_path, upper_air_vars,   "upper_air", upper_air=True, levels=levels)

    print("\n--- sample HDF5 data file ---")
    check_h5(sample_h5, all_h5_vars, "input group")

    print("\n" + "=" * 64)
    if all_missing:
        missing_unique = list(dict.fromkeys(all_missing))
        print(f"  MISSING from normalize file(s): {missing_unique}")
        print("  Update mean_path/std_path or remove these variables before training.")
        sys.exit(1)
    else:
        print("  All variables present in normalize files.")


if __name__ == "__main__":
    main()
