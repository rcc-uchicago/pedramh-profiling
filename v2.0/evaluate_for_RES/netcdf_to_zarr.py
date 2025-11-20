import numpy as np
import pandas as pd
import xarray as xr
import os
import re
import cftime
import numcodecs
from dask.diagnostics import ProgressBar
import time
import argparse

def main(args):
    # number of years by which to shift in order to use weatherbench2
    time_shift = 2000 

    # Regex to extract initialization date from filename
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})(?=_.*\.nc$)")

    # Helper function to convert cftime to datetime64
    def cftime_to_datetime64(cftime_array):
        return pd.to_datetime([t.isoformat() for t in cftime_array]).to_numpy("datetime64[ns]")

    start_time = time.perf_counter()

    all_datasets = []

    for fname in sorted(os.listdir(args.input_dir)):
        match = pattern.search(fname)
        if not match:
            continue

        ds = xr.open_dataset(os.path.join(args.input_dir, fname), use_cftime=True, chunks={})

        # Rename dimensions for weatherbench2
        ds = ds.rename({"ensemble_idx": "realization", "lat": "latitude", "lon": "longitude", "plev": "level"})
        
        # Decrease size of ensemble, if desired
        if args.ens_size < len(ds.realization):
            ds = ds.isel(realization = (ds.realization < args.ens_size))
        elif args.ens_size > len(ds.realization):
            raise ValueError(
                f"Requested {args.ens_size} ensemble members, but only {len(ds.realization)} are available."
                )

        # Take the ensemble mean, if desired
        if args.mean:
            ds = ds.mean(dim="realization")

        # Extract init_time from file name
        init_time_str = match.group(1)  # e.g. "0011-01-01_00:00:00"
        init_time_str = init_time_str.replace("_", " ")

        # Convert init_time to datetime64 and add offset
        init_time_dt = np.datetime64(init_time_str)
        offset = np.timedelta64(time_shift,'Y').astype('timedelta64[s]') # add 2000 year offset for compatibility
        init_time = init_time_dt + offset

        # Shift and convert times to datetime64
        shifted_times = [cftime.DatetimeProlepticGregorian(t.year + time_shift, t.month, t.day,
                    t.hour, t.minute, t.second) for t in ds.time.values]
        np_times = cftime_to_datetime64(shifted_times)

        # Calculate lead times
        lead_times = (np_times - init_time).astype('timedelta64[ns]')

        # Add prediction_timedelta
        ds_shift = ds.assign_coords(prediction_timedelta=("time", lead_times))

        # Reformulate time dimension
        ds_shift = ds_shift.swap_dims({"time": "prediction_timedelta"})
        ds_shift = ds_shift.drop_vars("time")
        ds_shift = ds_shift.expand_dims(time=[init_time])

        all_datasets.append(ds_shift)

    # Concatenate all initialization times
    print(f"Concatenating {len(all_datasets)} datasets...")
    combined_ds = xr.concat(all_datasets, dim="time")
    print(f"Concatenated {len(all_datasets)} datasets.")

    # Chunk for faster saving
    combined_ds = combined_ds.chunk({
        "time": 1,
        "realization": 10,
        "prediction_timedelta": 30,
        "level": 1,
        "latitude": 16,
        "longitude": 16
    })

    print("Final dataset shape:", combined_ds.sizes)
    print("Final dataset chunks:", combined_ds.chunks)

    # Compress for faster writing to zarr
    encoding = {
        var: {"compressor": numcodecs.Blosc(cname="zstd", clevel=3)}
        for var in combined_ds.data_vars
    }

    print(f"Starting conversion")

    with ProgressBar():
        combined_ds.to_zarr(args.output_zarr, mode="w", encoding=encoding, consolidated=True, compute=True)

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--output_zarr", required=True, type = str)
    parser.add_argument("--ens_size", default=100, type = int)
    parser.add_argument("--mean", action='store_true', help="Save ensemble mean only")

    ####### for UCAR
    parser.add_argument("--local-rank", type=int)
    #######

    args = parser.parse_args()
    main(args)