#!/work2/09979/awikner/stampede3/miniconda3/envs/emul4climate_nompi/bin/python -u
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
import os, glob
from multiprocessing import Pool
from tqdm import tqdm
import warnings

def load_var(args):
    file, var = args
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='^.*Unable to decode time axis into full numpy.datetime64 objects.*$')
        with xr.open_dataset(file) as ds:
            data = ds[var].values.astype(np.float32)
    return data

data_dir = '/scratch/09979/awikner/PLASIM/data/2100_year_sims_rerun/sim52/'
bias_dir = os.path.join(data_dir, 'bias')
os.makedirs(bias_dir, exist_ok=True)
years = list(range(12, 132))
atm_vars = ['ta', 'ua', 'va', 'hus', 'zg']
slice_offsets = [(1,0), (4,0), (4,1), (4,2), (4,3)]
slice_names = ['', '_0z','_6z','_12z','_18z']

if __name__ == '__main__':
    plev_files = [os.path.join(data_dir, 'plev_data', f'{year}_gaussian.nc') for year in years]
    sigma_files = [os.path.join(data_dir, 'sigma_data', f'{year}_gaussian.nc') for year in years]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='^.*Unable to decode time axis into full numpy.datetime64 objects.*$')
        with xr.open_dataset(plev_files[0]) as ds:
            data_vars = list(ds.data_vars)
        with xr.open_dataset(sigma_files[0]) as ds:
            lev = ds.lev
            plev = ds.plev
            data_vars = list(set(list(ds.data_vars) + data_vars) - set(['time_bnds']))
    print(data_vars)
    pool = Pool()
    for var in data_vars:
        if var in atm_vars:
            for levels, files in zip([plev, lev], [plev_files, sigma_files]):
                if var != 'zg' or levels.name != 'lev':
                    args_in = [(file, var) for file in files]
                    data = np.concat(list(tqdm(pool.imap(load_var, args_in),
                                            total = len(files),
                                            desc = f'Loading {var} {levels.name}')), axis = 0)
                    for slice_offset, slice_name in tqdm(zip(slice_offsets, slice_names),
                                                         total = len(slice_names),
                                                         desc = f'Saving slice {slice_offset}'):
                        bias = data[slice_offset[0]::slice_offset[0]].mean(axis = 0)
                        for i, level in enumerate(levels.values):
                            np.save(os.path.join(bias_dir, f'{var}_{float(level)}_bias{slice_name}.npy'),
                                    bias[i])
        else:
            args_in = [(file, var) for file in plev_files]
            data = np.concat(list(tqdm(pool.imap(load_var, args_in),
                                            total = len(plev_files),
                                            desc = f'Loading {var}')), axis = 0)
            for slice_offset, slice_name in tqdm(zip(slice_offsets, slice_names),
                                                         total = len(slice_names),
                                                         desc = f'Saving slices'):
                bias = data[slice_offset[0]::slice_offset[0]].mean(axis = 0)
                np.save(os.path.join(bias_dir, f'{var}_bias{slice_name}.npy'), bias)
            
            
                


    
