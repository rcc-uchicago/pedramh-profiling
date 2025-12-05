#!/work2/09979/awikner/stampede3/miniconda3/envs/emul4climate/bin/python -u
#SBATCH -p skx
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=48
#SBATCH -A TG-ATM170020
#SBATCH -J netcdf-to-h5
#SBATCH -o /work2/09979/awikner/stampede3/PLASIM/log_files/%x-%j.out
#SBATCH -e /work2/09979/awikner/stampede3/PLASIM/log_files/%x-%j.err
#SBATCH --mail-user=awikner@uchicago.edu
#SBATCH --mail-type=ALL

# In[1]:


import os
import argparse
import numpy as np
import xarray as xr
import h5py
from tqdm import tqdm
import cftime
import warnings



# In[7]:

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
    "snd",
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
DEFAULT_PRESSURE_LEVELS = [5, 10, 20, 30, 50, 70, 100, 150, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
DEFAULT_SIGMA_LEVELS = [0.0383000001311302, 0.119100004434586, 0.21085000783205, 
    0.316850006580353, 0.436800003051758, 0.566800028085709, 
    0.699350088834763, 0.823350071907043, 0.924099981784821, 0.983299970626831]

def open_dataset(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_dataset(path)
    return ds


def create_one_step_dataset(root_dir, save_dir, years, chunk_size=None, run_idx=0):
    save_dir_sigma = os.path.join(save_dir, 'sigma_data')
    save_dir_pl = os.path.join(save_dir, 'plev_data')
    os.makedirs(save_dir_sigma, exist_ok=True)
    os.makedirs(save_dir_pl, exist_ok=True)
    
    list_constant_vars = CONSTANTS
    list_varying_boundary_vars = VARYING_BOUNDARY
    list_single_vars = SINGLE_LEVEL_VARS
    list_sigma_vars = SIGMA_LEVEL_VARS
    list_pressure_vars = PRESSURE_LEVEL_VARS
    
    # load a constant variable to save lat and lon arrays
    """
    ds_constant = open_dataset(os.path.join(root_dir, str(run_idx), f'{list_constant_vars[0]}_masked.nc'))
    lat = ds_constant.lat.to_numpy()
    lat.sort()
    lon = ds_constant.lon.to_numpy()
    lon.sort()
    np.save(os.path.join(save_dir, 'lat.npy'), lat)
    np.save(os.path.join(save_dir, 'lon.npy'), lon)
    """
    
    for year in tqdm(years, desc='years', position=0):
        print(f'Saving year {year}')
        ds_sample = open_dataset(os.path.join(root_dir, 'sigma_data', f'{year}_gaussian.nc'))
        is_leap_year = cftime.is_leap_year(year, 'proleptic_gregorian', has_year_zero=True)
        
        if chunk_size is not None:
            n_chunks = len(ds_sample.time) // chunk_size + 1
        else:
            n_chunks = 1
            chunk_size = len(ds_sample.time)
        
        idx_in_year = 0
        
        ds_dict_boundary = {}
        for var in (list_varying_boundary_vars):
            if is_leap_year:
                ds_dict_boundary[var] = open_dataset(os.path.join(root_dir, str(run_idx), f'{var}_masked_6h_leap.nc'))
            else:
                ds_dict_boundary[var] = open_dataset(os.path.join(root_dir, str(run_idx), f'{var}_masked_6h.nc'))
        ds_plev  = open_dataset(os.path.join(root_dir, 'plev_data', f'{year}_gaussian.nc'))
        ds_sigma = open_dataset(os.path.join(root_dir, 'sigma_data', f'{year}_gaussian.nc'))
        #for var in (list_single_vars):
        #    ds_dict[var] = open_dataset(os.path.join(root_dir, var, f'{year}_gaussian.nc'))
        #for var in (list_sigma_vars):
        #    ds_dict_sigma[var] = open_dataset(os.path.join(root_dir, f'{var}_sigma', f'{year}_gaussian.nc'))
        #for var in (list_pressure_vars):
        #    ds_dict[var] = open_dataset(os.path.join(root_dir, var, f'{year}_gaussian.nc'))

        for chunk_id in tqdm(range(n_chunks), desc='chunks', position=1, leave=False):
            dict_np = {}
            dict_np_sigma = {}
            list_time_stamps = None
            ### convert ds to numpy
            for var in list_varying_boundary_vars:
                ds = ds_dict_boundary[var].isel(time=slice(chunk_id*chunk_size, (chunk_id+1)*chunk_size))
                if list_time_stamps is None:
                    list_time_stamps = ds.time.values
                dict_np[var] = ds[var].values.astype(np.float32)

            for var in list_single_vars + list_pressure_vars:
                ds = ds_plev[var].isel(time=slice(chunk_id*chunk_size, (chunk_id+1)*chunk_size))
                if list_time_stamps is None:
                    list_time_stamps = ds.time.values
                if var in list_single_vars + list_varying_boundary_vars:
                    dict_np[var] = ds.values.astype(np.float32)
                else:
                    available_levels = ds.plev.values
                    ds_np = ds.values.astype(np.float32)
                    for i, level in enumerate(available_levels):
                        dict_np[f'{var}_{level}'] = ds_np[:, i]
                            
            for var in (list_sigma_vars):
                ds = ds_sigma[var].isel(time=slice(chunk_id*chunk_size, (chunk_id+1)*chunk_size))
                if list_time_stamps is None:
                    list_time_stamps = ds.time.values
                if var in list_single_vars:
                    dict_np_sigma[var] = ds.values.astype(np.float32)
                else:
                    available_levels = ds.lev.values
                    ds_np = ds.values.astype(np.float32)
                    for i, level in enumerate(available_levels):
                        dict_np_sigma[f'{var}_{level}'] = ds_np[:, i]
                    
            for i in tqdm(range(len(list_time_stamps)), desc='time stamps', position=2, leave=False):
                data_dict = {
                    'input': {'time': str(list_time_stamps[i])}
                }
                data_dict_sigma = {
                    'input': {'time': str(list_time_stamps[i])}
                }
                for var in dict_np.keys():
                    data_dict['input'][var] = dict_np[var][i]
                    if var in list_varying_boundary_vars + list_single_vars:
                        data_dict_sigma['input'][var] = dict_np[var][i]
                for var in dict_np_sigma.keys():
                    data_dict_sigma['input'][var] = dict_np_sigma[var][i]
                for var in list_constant_vars:
                    constant_path = os.path.join(root_dir, str(run_idx), f'{var}_masked.nc')
                    constant_field = open_dataset(constant_path)[var].values
                    constant_field = constant_field.reshape(constant_field.shape[-2:])
                    data_dict['input'][var] = constant_field
                    data_dict_sigma['input'][var] = constant_field
                    
                with h5py.File(os.path.join(save_dir_pl, f'{year}_{idx_in_year:04}.h5'), 'w', libver='latest') as f:
                    for main_key, sub_dict in data_dict.items():
                        # Create a group for the main key (e.g., 'input' or 'output')
                        group = f.create_group(main_key)
                        
                        # Now, save each array in the sub-dictionary to this group
                        for sub_key, array in sub_dict.items():
                            if sub_key != 'time':
                                group.create_dataset(sub_key, data=array, compression=None, dtype=np.float32)
                            else:
                                group.create_dataset(sub_key, data=array, compression=None)
                                
                with h5py.File(os.path.join(save_dir_sigma, f'{year}_{idx_in_year:04}.h5'), 'w', libver='latest') as f:
                    for main_key, sub_dict in data_dict_sigma.items():
                        # Create a group for the main key (e.g., 'input' or 'output')
                        group = f.create_group(main_key)
                        
                        # Now, save each array in the sub-dictionary to this group
                        for sub_key, array in sub_dict.items():
                            if sub_key != 'time':
                                group.create_dataset(sub_key, data=array, compression=None, dtype=np.float32)
                            else:
                                group.create_dataset(sub_key, data=array, compression=None)
                
                idx_in_year += 1


# In[10]:


def parse_args():
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--root_dir', type=str, default='/scratch/09979/awikner/PLASIM/data/2100_year_sims_rerun/sim52/', help='Root directory containing input data.')
    parser.add_argument('--save_dir', type=str, default='/scratch/09979/awikner/PLASIM/data/2100_year_sims_rerun/sim52/h5/', help='Directory to save regridded files.')
    parser.add_argument('--start_year', type=int, default=7, help='Start year for the data range.')
    parser.add_argument('--end_year', type=int, default=132, help='End year for the data range.')
    parser.add_argument('--run_idx', type=int, default=23, help='Index for the run.')
    parser.add_argument("--chunk_size", type=int, default=100, help="Chunk size for reading datasets (default=10).")
    
    return parser.parse_args()


# In[11]:


def main():
    args = parse_args()

    create_one_step_dataset(
        root_dir=args.root_dir,
        save_dir=args.save_dir,
        years=list(range(args.start_year, args.end_year)),
        chunk_size=args.chunk_size,
        run_idx=args.run_idx
    )


if __name__ == "__main__":
    main()

