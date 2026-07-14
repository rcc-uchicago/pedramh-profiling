import os
import xarray as xr


years = ['1979']
for year in years:
    ds = xr.open_mfdataset('data/raw/' + year + '/' + '*_pl.nc')
    file_name = 'data/raw/' + year + '/' + year + '_pl.nc'
    os.remove(file_name) if os.path.exists(file_name) else None
    ds.to_netcdf(file_name)
    print('Combining nc files for {} is done!'.format(year))

print('Done!')