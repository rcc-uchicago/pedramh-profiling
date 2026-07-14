"""Prepare Polaris-local copies of the E3SM SFNO auxiliary files.

The staged E3SM tree on Polaris
(/eagle/lighthouse-uchicago/members/jesswan/AI4SRM/data/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101)
ships two files that need a one-time fix before train.py can use them:

1. h5/plev_data/data_2015-2050_mean.nc and data_2015-2050_std_corr.nc carry
   the 18 sigma levels on a dim named 'Z'.  With use_sigma_levels=True,
   utils/data_loader_multifiles.load_mean_std selects every non-zg upper-air
   variable on a dim named 'Z_2' (the Derecho sigma_data copies had that
   name), so 'Z' must be renamed to 'Z_2'.  Values are already the sigma
   levels the SFNO config expects — only the dim name is wrong.

2. h5/plev_data/climatology.nc is CDF-5 (netCDF3-64bit-data).  xarray's
   default engine resolution hands CDF magic to scipy, which fails with
   'Unexpected header' (train.py opens it with no explicit engine).
   Re-saving as NETCDF4 (HDF5 magic) makes the default open work.

Run once on a login node (CPU, a few seconds):
    bash -lc 'module use /soft/modulefiles; module load conda; conda activate base; \
        python polaris_prepare_e3sm_stats.py'

Outputs go to $PANGU_AUX (set by the PBS script to a shared read-only dir OUTSIDE the
repo, so the ~17 GB is not re-encoded into every clone); if PANGU_AUX is unset they fall
back to PanguWeather/v2.0/polaris_data/ (gitignored via *.nc).
The Polaris config (config/E3SM_SFNO_H5_POLARIS.yaml) points at them with
absolute paths — os.path.join(data_dir, <abs path>) returns the abs path,
so no loader code changes are needed.
"""
import os
import xarray as xr

SRC = ("/eagle/lighthouse-uchicago/members/jesswan/AI4SRM/data/"
       "E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/h5/plev_data")
# Default OUT is passed in by the PBS script (PANGU_AUX, outside the repo) so the
# ~17 GB is shared read-only instead of re-encoded into every user's clone.
OUT = os.environ.get("PANGU_AUX") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "polaris_data")
os.makedirs(OUT, exist_ok=True)

# 1) stats: rename the level dim Z -> Z_2 (values already sigma levels)
for fname in ("data_2015-2050_mean.nc", "data_2015-2050_std_corr.nc"):
    with xr.open_dataset(os.path.join(SRC, fname)) as ds:
        ds.rename({"Z": "Z_2"}).to_netcdf(os.path.join(OUT, fname))
    print(f"OK wrote {OUT}/{fname} (Z -> Z_2)")

# 2) climatology: CDF-5 -> NETCDF4 so xarray's default engine opens it
time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
with xr.open_dataset(os.path.join(SRC, "climatology.nc"), engine="netcdf4",
                     decode_times=time_coder) as ds:
    ds.to_netcdf(os.path.join(OUT, "climatology.nc"), format="NETCDF4")
print(f"OK wrote {OUT}/climatology.nc (NETCDF4)")

# sanity: the exact open train.py performs
with xr.open_dataset(os.path.join(OUT, "climatology.nc"),
                     decode_times=time_coder) as ds:
    assert "time" in ds.sizes and ds.sizes["time"] == 365
print("PREP_OK")
