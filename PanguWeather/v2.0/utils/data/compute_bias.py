#!/usr/bin/env python -u
#SBATCH -p pvc
#SBATCH -t 8:00:00
#SBATCH -N 1
#SBATCH -n 96
#SBATCH -A TG-ATM170020
#SBATCH -J normalize_compute
#SBATCH -o /work2/09979/awikner/stampede3/PLASIM/log_files/%x-%j.out
#SBATCH -e /work2/09979/awikner/stampede3/PLASIM/log_files/%x-%j.err
#SBATCH --mail-user=awikner@uchicago.edu
#SBATCH --mail-type=ALL

import xarray as xr
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm
import warnings


def compute_year_stats(args):
    """
    Load one year file and return per-slice statistics without keeping the full
    time series in memory.

    Parameters
    ----------
    args : (file, var, slice_offsets)
        file          : path to the NetCDF year file
        var           : variable name to load
        slice_offsets : list of (step, start) tuples; timesteps are selected
                        as data[start::step, ...]

    Returns
    -------
    list of (sum_array, count) for each slice_offset, or None if var is absent.
    sum_array is float64 to avoid precision loss when accumulating across years.
    """
    file, var, slice_offsets = args
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^.*Unable to decode time axis into full numpy\.datetime64 objects.*$")
        with xr.open_dataset(file) as ds:
            if var not in ds:
                return None
            data = ds[var].values.astype(np.float32)

    results = []
    for step, start in slice_offsets:
        sliced = data[start::step]
        results.append((sliced.sum(axis=0, dtype=np.float64), sliced.shape[0]))
    return results


data_dir = '/glade/derecho/scratch/awikner/PLASIM/data/2100_year_sims_rerun/sim52'
bias_dir = os.path.join(data_dir, 'bias2')
os.makedirs(bias_dir, exist_ok=True)
years = list(range(12, 132))
atm_vars = ['ta', 'ua', 'va', 'hus', 'zg']

# Each tuple is (step, start): selects data[start::step, ...].
# Original code had a bug: data[offset[0]::offset[0]] used [0] for both start
# and step, so slice_offset[1] was never used and all four synoptic-hour slices
# produced the same result.  Fixed: data[start::step] = data[offset[1]::offset[0]].
slice_offsets = [(1, 0), (4, 0), (4, 1), (4, 2), (4, 3)]
slice_names   = ['',    '_0z', '_6z', '_12z', '_18z']


def accumulate(pool, args_in, desc):
    """
    Run compute_year_stats in parallel and accumulate weighted sums across years.

    Returns (total_sums, total_counts) where total_sums[i] is the float64 sum
    of all selected timesteps for slice i across all years, and total_counts[i]
    is the total number of those timesteps.
    """
    total_sums   = None
    total_counts = None

    for year_stats in tqdm(pool.imap(compute_year_stats, args_in),
                           total=len(args_in), desc=desc):
        if year_stats is None:
            continue
        if total_sums is None:
            total_sums   = [np.zeros_like(s, dtype=np.float64) for s, _ in year_stats]
            total_counts = np.zeros(len(year_stats), dtype=np.int64)
        for i, (s, n) in enumerate(year_stats):
            total_sums[i]   += s
            total_counts[i] += n

    return total_sums, total_counts


if __name__ == '__main__':
    plev_files  = [os.path.join(data_dir, 'plev_data',  f'{y}_gaussian.nc') for y in years]
    sigma_files = [os.path.join(data_dir, 'sigma_data', f'{y}_gaussian.nc') for y in years]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^.*Unable to decode time axis into full numpy\.datetime64 objects.*$")
        with xr.open_dataset(plev_files[0]) as ds:
            data_vars = list(ds.data_vars)
        with xr.open_dataset(sigma_files[0]) as ds:
            lev  = ds.lev
            plev = ds.plev
            data_vars = list(set(data_vars + list(ds.data_vars)) - {'time_bnds'})

    print('Variables:', data_vars)

    pool = Pool(16)
    for var in data_vars:
        if var in atm_vars:
            for levels, files in zip([plev, lev], [plev_files, sigma_files]):
                if var == 'zg' and levels.name == 'lev':
                    continue  # zg not on sigma levels

                args_in = [(f, var, slice_offsets) for f in files]
                total_sums, total_counts = accumulate(
                    pool, args_in, desc=f'{var}/{levels.name}')

                if total_sums is None:
                    print(f'WARNING: no data found for {var}/{levels.name}, skipping.')
                    continue

                for i, (slice_name, count) in enumerate(zip(slice_names, total_counts)):
                    bias = (total_sums[i] / count).astype(np.float32)
                    for j, level in enumerate(levels.values):
                        np.save(
                            os.path.join(bias_dir, f'{var}_{float(level)}_bias{slice_name}.npy'),
                            bias[j])
        else:
            args_in = [(f, var, slice_offsets) for f in plev_files]
            total_sums, total_counts = accumulate(pool, args_in, desc=var)

            if total_sums is None:
                print(f'WARNING: no data found for {var}, skipping.')
                continue

            for i, (slice_name, count) in enumerate(zip(slice_names, total_counts)):
                bias = (total_sums[i] / count).astype(np.float32)
                np.save(os.path.join(bias_dir, f'{var}_bias{slice_name}.npy'), bias)

    pool.close()
    pool.join()
