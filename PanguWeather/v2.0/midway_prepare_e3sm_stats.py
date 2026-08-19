"""Prepare the E3SM sigma stats for the Midway loader path.

Counterpart to polaris_prepare_e3sm_stats.py, which exists because the staged
stats never quite match what utils/data_loader_multifiles.load_mean_std wants.
The Midway mismatch is different from the Polaris one:

  Polaris: climatology.nc was CDF-5 and xarray could not open it.
  Midway:  climatology opens fine, but the mean/std files name their level
           dimension `Z`, while load_mean_std(use_sigma_levels=True) indexes
           non-geopotential variables on `Z_2`:

               coordinates.append("Z_2"); levels_for_var.append(self.sigma_levels)

           Only `zg`/`geopotential_height` take the `Z` branch, and this dataset
           calls that variable `Z3`, so EVERY variable wants `Z_2`. Job 53539745
           died with KeyError: 'Z_2'.

The values under `Z` are already the sigma levels -- verified equal to the
config's `sigma_levels` list to 3 dp -- so this is purely a naming fix, not a
recomputation. `where(..., dims=['Z_2'])` broadcasts rather than selects unless
the DIMENSION is renamed, so a coordinate alias is not enough.

NOTE these stats are not the correct normalisation for this data (per the data
owner). Runs using them are for PROFILING ONLY; ignore their loss values.
"""

import os

import xarray as xr

SRC = "/project/pedramh/E3SM/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/h5/sigma_data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "midway_data")

os.makedirs(OUT, exist_ok=True)
for fname in ("data_2015-2050_mean.nc", "data_2015-2050_std_corr.nc"):
    with xr.open_dataset(os.path.join(SRC, fname)) as ds:
        if "Z" not in ds.sizes:
            raise SystemExit(f"ERROR {fname}: expected a 'Z' level dim, got {dict(ds.sizes)}")
        out = ds.rename({"Z": "Z_2"})
        out.to_netcdf(os.path.join(OUT, fname))
    print(f"  {fname}: Z -> Z_2 ({ds.sizes['Z']} levels)")
print(f"STATS_PREP_OK -> {OUT}")
