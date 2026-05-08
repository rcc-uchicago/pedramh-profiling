#!/work2/09979/awikner/stampede3/miniconda3/envs/emul4climate_nompi/bin/python -u
#SBATCH -p spr
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH -n 112
#SBATCH -A TG-ATM170020
#SBATCH -J normalize_compute
#SBATCH -o /work2/09979/awikner/stampede3/PLASIM/log_files/%x-%j.out
#SBATCH -e /work2/09979/awikner/stampede3/PLASIM/log_files/%x-%j.err
#SBATCH --mail-user=awikner@uchicago.edu
#SBATCH --mail-type=ALL

import os
import argparse
import numpy as np
import xarray as xr
from tqdm import tqdm
import warnings


CONSTANTS = [
    "glac",
    "lsm",
    "mrfc",
    "sg",
    "vegf",
    "z0t",
    "z0v",
    "z0"
]

VARYING_BOUNDARY = [
    "alb",
    "dlai",
    "mrso_climatology",
    "rsdt",
    "sic",
    "sst",
    "vegc",
]

SINGLE_LEVEL_VARS = [
    #Diagostic
    "evap",
    "mrro",
    "pr_6h",
    "pr_12h",
    "pr_24h",
    "snm",
    "sndc",
    "rss",
    "rls",
    "rst",
    "rlut",
    "rsut",
    "ssru",
    "stru",

    #Prognostic
    "mrso",
    "pl",
    "tas",
    "ts"
]

SIGMA_LEVEL_VARS = [
    "hus", "ta", "ua", "va"
]

PRESSURE_LEVEL_VARS = [
    "hus", "ta", "ua", "va", "zg"
]

#1000, 925, 850, 700, 600, 500, 400, 300, 250, 150, 100, 70, 50, 30, 20, 10, 5
DEFAULT_PRESSURE_LEVELS = [50, 100, 150, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
DEFAULT_SIGMA_LEVELS = [0.0383000001311302, 0.119100004434586, 0.21085000783205, 
    0.316850006580353, 0.436800003051758, 0.566800028085709, 
    0.699350088834763, 0.823350071907043, 0.924099981784821, 0.983299970626831]

def open_dataset(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_dataset(path)
    return ds


def parse_args():
    parser = argparse.ArgumentParser(description='Regridding NetCDF files.')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing input data.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save regridded files.')
    parser.add_argument('--start_year', type=int, default=12, help='Start year for the data range.')
    parser.add_argument('--end_year', type=int, default=111, help='End year for the data range.')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size for reading datasets (default=100).')
    parser.add_argument('--lead_time', type=int, default=None, help='Lead time for difference normalization.')  # None means we're computing normalization for the original data
    parser.add_argument('--data_frequency', type=int, default=6, help='Data frequency in hours (default=6).')  # this depends on the dataset
    parser.add_argument('--lev_type', type=str, default='plev', help='Type of level axis. Must be either "lev" or "plev".')  # this depends on the dataset
    
    return parser.parse_args()

def main():
    args = parse_args()

    
    root_dir = args.root_dir
    save_dir = args.save_dir
    start_year = args.start_year
    end_year = args.end_year
    chunk_size = args.chunk_size
    lead_time = args.lead_time
    data_freq = args.data_frequency
    lev_type = args.lev_type
    
    #print(save_dir)
    #print(lead_time)
    #print(end_year)
    years = list(range(start_year, end_year + 1))
    #print(years)
    os.makedirs(save_dir, exist_ok=True)

    list_constant_vars = CONSTANTS
    list_single_vars = SINGLE_LEVEL_VARS
    list_varying_boundary = VARYING_BOUNDARY
    if lev_type == 'plev':
        list_pressure_vars = PRESSURE_LEVEL_VARS
    else:
        list_pressure_vars = SIGMA_LEVEL_VARS + ['zg']
    
    mask_fill = dict(zip(list_constant_vars + list_single_vars + list_varying_boundary,
                         [0. for var in list_constant_vars + list_single_vars + list_varying_boundary]))
    mask_fill['sst'] = 270.
    mask_fill['ts'] = 270.
    
    if lev_type == 'plev':
        mean_file_name = f"normalize_diff_mean_{lead_time}.npz" if lead_time is not None else "normalize_mean.npz"
        std_file_name = f"normalize_diff_std_{lead_time}.npz" if lead_time is not None else "normalize_std.npz"
        test_ds = open_dataset(os.path.join(root_dir, 'plev_data', f'{start_year}_gaussian.nc'))
        levels = test_ds[lev_type].values
        plevels = levels
    else:
        mean_file_name = f"normalize_diff_mean_{lead_time}_sigma.npz" if lead_time is not None else "normalize_mean_sigma.npz"
        std_file_name = f"normalize_diff_std_{lead_time}_sigma.npz" if lead_time is not None else "normalize_std_sigma.npz"
        test_ds = open_dataset(os.path.join(root_dir, 'sigma_data', f'{start_year}_gaussian.nc'))
        levels = test_ds[lev_type].values
        test_plev_ds = open_dataset(os.path.join(root_dir, 'plev_data', f'{start_year}_gaussian.nc'))
        plevels = test_plev_ds['plev'].values

    # initialize normalization values if not exist, else load them
    if not os.path.exists(os.path.join(save_dir, mean_file_name)):
        normalize_mean = {}
        normalize_std = {}

        for var in list_single_vars:
            normalize_mean[var] = []
            normalize_std[var] = []
        for var in list_pressure_vars:
            if var == 'zg':
                for level in plevels:
                    #print(level)
                    normalize_mean[f'{var}_{float(level)}'] = []
                    normalize_std[f'{var}_{float(level)}'] = []
            else:
                for level in levels:
                    #print(level)
                    normalize_mean[f'{var}_{float(level)}'] = []
                    normalize_std[f'{var}_{float(level)}'] = []
    else:
        normalize_mean = np.load(os.path.join(save_dir, mean_file_name))
        normalize_std = np.load(os.path.join(save_dir, std_file_name))
        normalize_mean = {k: list(v) for k, v in normalize_mean.items()}
        normalize_std = {k: list(v) for k, v in normalize_std.items()}

    steps = lead_time // data_freq if lead_time else None

    for var in tqdm(list_single_vars + list_pressure_vars, desc='variables', position=0):
        #print(var)
        for year in tqdm(years, desc='years', position=1, leave=False):
            #print(year)
            if var in SINGLE_LEVEL_VARS:
                path = os.path.join(root_dir, 'plev_data', f'{year}_gaussian.nc')
            
            else:
                if lev_type == 'plev' or var == 'zg':
                    path = os.path.join(root_dir, 'plev_data', f'{year}_gaussian.nc')
                else:
                    path = os.path.join(root_dir, 'sigma_data', f'{year}_gaussian.nc')
                
            ds = open_dataset(path)
            #print(ds.level.values)
            
            # chunk to smaller sizes
            if chunk_size is not None:
                n_chunks = len(ds.time) // chunk_size + 1
            else:
                n_chunks = 1
                chunk_size = len(ds.time)
            
            for chunk_id in tqdm(range(n_chunks), desc='chunks', position=2, leave=False):
                ds_small = ds.isel(time=slice(chunk_id*chunk_size, (chunk_id+1)*chunk_size))
                if var in SINGLE_LEVEL_VARS:
                    ds_np = ds_small[var].values# N, H, W
                    ds_np = np.where(np.isnan(ds_np), mask_fill[var], ds_np)
                    if steps is not None:
                        ds_np = ds_np[steps:] - ds_np[:-steps]
                    normalize_mean[var].append(np.nanmean(ds_np))
                    normalize_std[var].append(np.nanstd(ds_np))
                else:
                    ds_np = ds_small[var].values # N, Levels, H, W
                    if var == 'zg':
                        levels_in_ds = ds['plev'].values
                    else:
                        levels_in_ds = ds[lev_type].values
                    #assert np.sum(np.array(DEFAULT_PRESSURE_LEVELS) - levels_in_ds) == 0 # ensure the same order of pressure levels
                    for i, level in enumerate(levels_in_ds):
                        ds_np_lev = ds_np[:, i]
                        #print(np.nanmean(ds_np_lev))
                        if steps is not None:
                            ds_np_lev = ds_np_lev[steps:] - ds_np_lev[:-steps]
                        normalize_mean[f'{var}_{float(level)}'].append(np.nanmean(ds_np_lev))
                        normalize_std[f'{var}_{float(level)}'].append(np.nanstd(ds_np_lev))
            
        if var in SINGLE_LEVEL_VARS:
            mean_over_files, std_over_files = np.array(normalize_mean[var]), np.array(normalize_std[var])
            # var(X) = E[var(X|Y)] + var(E[X|Y])
            variance = (std_over_files**2).mean() + (mean_over_files**2).mean() - mean_over_files.mean()**2
            std = np.sqrt(variance)
            # E[X] = E[E[X|Y]]
            mean = mean_over_files.mean()
            normalize_mean[var] = mean.reshape([1])
            normalize_std[var] = std.reshape([1])
        
            np.savez(os.path.join(save_dir, mean_file_name), **normalize_mean)
            np.savez(os.path.join(save_dir, std_file_name), **normalize_std)
        else:
            if var == 'zg':
                for l in plevels:
                    var_lev = f'{var}_{l}'
                    mean_over_files, std_over_files = np.array(normalize_mean[var_lev]), np.array(normalize_std[var_lev])
                    # var(X) = E[var(X|Y)] + var(E[X|Y])
                    variance = (std_over_files**2).mean() + (mean_over_files**2).mean() - mean_over_files.mean()**2
                    std = np.sqrt(variance)
                    # E[X] = E[E[X|Y]]
                    mean = mean_over_files.mean()
                    normalize_mean[var_lev] = mean.reshape([1])
                    normalize_std[var_lev] = std.reshape([1])
            else:
                for l in levels:
                    var_lev = f'{var}_{l}'
                    mean_over_files, std_over_files = np.array(normalize_mean[var_lev]), np.array(normalize_std[var_lev])
                    # var(X) = E[var(X|Y)] + var(E[X|Y])
                    variance = (std_over_files**2).mean() + (mean_over_files**2).mean() - mean_over_files.mean()**2
                    std = np.sqrt(variance)
                    # E[X] = E[E[X|Y]]
                    mean = mean_over_files.mean()
                    normalize_mean[var_lev] = mean.reshape([1])
                    normalize_std[var_lev] = std.reshape([1])
        
            np.savez(os.path.join(save_dir, mean_file_name), **normalize_mean)
            np.savez(os.path.join(save_dir, std_file_name), **normalize_std)
        
    for var in list_constant_vars + list_varying_boundary:
        if steps is not None:
            normalize_mean[var] = [0.0]
            normalize_std[var] = [0.0]
        else:
            if var in list_varying_boundary:
                path = os.path.join(root_dir, 'boundary_data', f'{var}_masked_6h.nc')
            else:
                path = os.path.join(root_dir, 'boundary_data', f'{var}_masked.nc')
            ds = open_dataset(path)
            ds_np = ds[var].values
            ds_np = np.where(np.isnan(ds_np), mask_fill[var], ds_np)
            normalize_mean[var] = ds_np.mean().reshape([1])
            normalize_std[var] = ds_np.std().reshape([1])

    np.savez(os.path.join(save_dir, mean_file_name), **normalize_mean)
    np.savez(os.path.join(save_dir, std_file_name), **normalize_std)


if __name__ == "__main__":
    main()
