import numpy as np
import xarray as xr
import sys, os, glob
import warnings

# --- input/output ---
data_dir = '/scratch/09979/awikner/PLASIM/data/2100_year_sims_rerun/sim52'
npz_files = glob.glob(os.path.join(data_dir, '*.npz'))
npz_filestrs = ['_'.join(os.path.basename(file).split('_')[1:]).split('.')[0] for file in npz_files]
nc_files = [os.path.join(data_dir, 'h5', 'sigma_data', f'data_12-132_{filestr}.nc') if 'sigma' in filestr \
            else os.path.join(data_dir, 'h5', 'plev_data', f'data_12-132_{filestr}.nc') \
            for filestr in npz_filestrs]

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message='^.*Unable to decode time axis into full numpy.datetime64 objects.*$')
    test_ds = xr.open_dataset(os.path.join(data_dir, 'plev_data', '100_gaussian.nc'))
    plev = test_ds.plev
    with xr.open_dataset(os.path.join(data_dir, 'sigma_data', '100_gaussian.nc')) as sigma_ds:
        lev = sigma_ds.lev

atm_vars = ['ta', 'ua', 'va', 'hus']

for npz_file, nc_file in zip(npz_files, nc_files):
    print(f'Converting {os.path.basename(npz_file)}')
    is_sigma = 'sigma' in os.path.basename(npz_file)
    if is_sigma:
        new_ds = xr.Dataset(coords = {'Z': plev.values,
                                      'Z_2': lev.values})
    else:
        new_ds = xr.Dataset(coords = {'Z': plev.values})
    data = np.load(npz_file)
    keys = list(data.keys())

    for atm_var in atm_vars:
        if is_sigma:
            da = xr.DataArray(data = np.array([data[f'{atm_var}_{float(level)}'][0] for level in lev],
                                                     dtype = np.float32),
                              dims = ['Z_2'],
                              coords = {'Z_2': lev.values},
                              attrs = test_ds[atm_var].attrs)
        else:
            da = xr.DataArray(data = np.array([data[f'{atm_var}_{float(level)}'][0] for level in plev],
                                                     dtype = np.float32),
                              dims = ['Z'],
                              coords = {'Z': plev.values},
                              attrs = test_ds[atm_var].attrs)
        new_ds[atm_var] = da
    da = xr.DataArray(data = np.array([data[f'zg_{float(level)}'][0] for level in plev],
                                                     dtype = np.float32),
                              dims = ['Z'],
                              coords = {'Z': plev.values},
                              attrs = test_ds[atm_var].attrs)
    new_ds['zg'] = da

    non_atm_keys = [key for key in keys if '_' not in key or not any([atm_var in key for atm_var in ['zg'] + atm_vars])]
    print(non_atm_keys)

    for var in non_atm_keys:
        if var in test_ds:
            da = xr.DataArray(data[var][0], attrs = test_ds[var].attrs)
        else:
            da = xr.DataArray(data[var][0])
        new_ds[var] = da
    new_ds.to_netcdf(nc_file)

    print(f"Converted {os.path.basename(npz_file)} -> {os.path.basename(nc_file)}")