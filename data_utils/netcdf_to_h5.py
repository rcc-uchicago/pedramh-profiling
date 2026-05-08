import os
import xarray
import h5py
from netCDF4 import Dataset as DS

from parallel_copy import writetofile, writetofile_pl



sfc_vars = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']
pl_vars = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']
pressure_level = ['1000', '925', '850', '700', '600', '500', '400', '300', '250', '200', '150', '100','50']

path = '/anvil/scratch/x-mzand/ERA5_polaris/6h_721_1440_with_poles/'
var_dirs = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
var_dirs.sort() 
paths_pl = {}


for v_no, var_dir in enumerate(var_dirs):
    if (var_dir not in sfc_vars) and (var_dir not in pl_vars):
        continue
    
    file_paths = [f for f in os.listdir(path+var_dir) if os.path.isfile(os.path.join(path+var_dir, f))]
    file_paths.sort()
    for f_no, file_path in enumerate(file_paths):
        
        year = file_path.split('.')[0]
        dest_sfc = 'data/h5/train/' + year + '_sfc.h5'
        full_file_path = os.path.join(path+var_dir, file_path)

        time_steps = 1460 #days*4 if dt=6h 
        with h5py.File(dest_sfc, 'w') as f:
            f.create_dataset('fields', shape = (time_steps, 4, 721, 1440), dtype='f')
        
        if var_dir in sfc_vars:
            #mslp u10 v10 t2m
            if var_dir == 'mean_sea_level_pressure':
                writetofile(full_file_path, dest_sfc, 0, [var_dir], Nimgtot=time_steps)
            elif var_dir == '10m_u_component_of_wind':
                writetofile(full_file_path, dest_sfc, 1, [var_dir], Nimgtot=time_steps)
            elif var_dir == '10m_v_component_of_wind':    
                writetofile(full_file_path, dest_sfc, 2, [var_dir], Nimgtot=time_steps)
            elif var_dir == '2m_temperature':
                writetofile(full_file_path, dest_sfc, 3, [var_dir], Nimgtot=time_steps)
            
        elif var_dir in pl_vars:
            #z, q, t, u, v
            if year in paths_pl.keys():
                paths_pl[year].append(full_file_path) 
            else:
                paths_pl[year] = [full_file_path]
                

out_pl_vars = ['z', 'q', 't', 'u', 'v']
for year in paths_pl.keys():
    dest_pl = 'data/h5/train/' + year + '_pl.h5'
    with h5py.File(dest_pl, 'w') as f:
        f.create_dataset('fields', shape = (time_steps, 5, 13, 721, 1440), dtype='f')
    
    paths = paths_pl[year]
    
    for path in paths:
        var_name = path.split(year)[0].split('/')[-2]
        writetofile_pl(path, dest_pl, [pl_vars.index(var_name)], [var_name], Nimgtot=time_steps)
    