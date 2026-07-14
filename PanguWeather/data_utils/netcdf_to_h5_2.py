#!/eagle/MDClimSim/hyadav/Pangu_env_2
#PBS -N random
##PBS -l select=2:system=polaris
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -q debug 
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle                          
#PBS -A MDClimSim
#PBS -e logs/
#PBS -o logs/
import os
import xarray
import h5py
from netCDF4 import Dataset as DS
import time
from mpi4py import MPI

from parallel_copy import writetofile, writetofile_pl



sfc_vars = ['2m_temperature','10m_u_component_of_wind', '10m_v_component_of_wind', '10m_wind_speed', 'surface_pressure']
sfc_vars_run = []
precip = 'total_precipitation_6hr'
pl_vars = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']
pressure_level = ['50', '100', '150', '200', '250', '300', '400', '500', '600', '700', '850', '925', '1000']

path = '/eagle/MDClimSim/tungnd/data/wb2/6h_721_1440_with_poles'
dest_path = '/eagle/MDClimSim/awikner/PanguWeather-UC/data/h5'
#train_years = list(range(1979,2018))
train_years = list(range(2010,2018))
val_years = [2019]
test_years = [2018, 2020, 2021]
var_dirs = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
var_dirs.sort() 
paths_pl = {}

rank = MPI.COMM_WORLD.rank
comm = MPI.COMM_WORLD

for v_no, var_dir in enumerate(var_dirs):
    if (var_dir not in sfc_vars) and (var_dir not in pl_vars) and var_dir != precip:
        continue
    
    file_paths = [f for f in os.listdir(os.path.join(path, var_dir)) if os.path.isfile(os.path.join(path, var_dir, f))]
    file_paths.sort()
    for f_no, file_path in enumerate(file_paths):
        
        year = file_path.split('.')[0]
        if int(year) in train_years:
            dest_sfc = os.path.join(dest_path, 'train', year + '_sfc.h5')
            dest_sfc_wprecip = os.path.join(dest_path, 'train', year + '_sfc_wprecip.h5')
        elif int(year) in val_years:
            dest_sfc = os.path.join(dest_path, 'val', year + '_sfc.h5')
            dest_sfc_wprecip = os.path.join(dest_path, 'val', year + '_sfc_wprecip.h5')
        elif int(year) in test_years:
            dest_sfc = os.path.join(dest_path, 'test', year + '_sfc.h5')
            dest_sfc_wprecip = os.path.join(dest_path, 'test', year + '_sfc_wprecip.h5')
        else:
            continue
        full_file_path = os.path.join(path, var_dir, file_path)

        time_steps = 1460 #days*4 if dt=6h 
        if not os.path.exists(dest_sfc) and rank == 0:
            with h5py.File(dest_sfc, 'w') as f:
                f.create_dataset('fields', shape = (time_steps, len(sfc_vars), 721, 1440), dtype='f')
            with h5py.File(dest_sfc_wprecip, 'w') as f:
                f.create_dataset('fields', shape = (time_steps, len(sfc_vars)+1, 721, 1440), dtype='f')
        comm.Barrier()

        
        if var_dir in sfc_vars_run:
            #mslp u10 v10 t2m
            writetofile(full_file_path, dest_sfc, sfc_vars.index(var_dir), [var_dir], Nimgtot=time_steps) 
            writetofile(full_file_path, dest_sfc_wprecip, sfc_vars.index(var_dir), [var_dir], Nimgtot=time_steps)
            """
            if var_dir == 'mean_sea_level_pressure':
                writetofile(full_file_path, dest_sfc, 0, [var_dir], Nimgtot=time_steps)
            elif var_dir == '10m_u_component_of_wind':
                writetofile(full_file_path, dest_sfc, 1, [var_dir], Nimgtot=time_steps)
            elif var_dir == '10m_v_component_of_wind':    
                writetofile(full_file_path, dest_sfc, 2, [var_dir], Nimgtot=time_steps)
            elif var_dir == '2m_temperature':
                writetofile(full_file_path, dest_sfc, 3, [var_dir], Nimgtot=time_steps)
            """
            #elif var_dir == precip:
            #writetofile(full_file_path, dest_sfc_wprecip, len(sfc_vars), [var_dir], Nimgtot=time_steps)
        elif var_dir in pl_vars:
            #z, q, t, u, v
            if year in paths_pl.keys():
                paths_pl[year].append(full_file_path) 
            else:
                paths_pl[year] = [full_file_path]
                

out_pl_vars = ['z', 'q', 't', 'u', 'v']
for year in paths_pl.keys():
    if int(year) in train_years:
        dest_pl = os.path.join(dest_path, 'train', year + '_pl.h5')
    elif int(year) in val_years:
        dest_pl = os.path.join(dest_path, 'val', year + '_pl.h5')
    elif int(year) in test_years:
        dest_pl = os.path.join(dest_path, 'test', year + '_pl.h5')
    else:
        continue
    if not os.path.exists(dest_pl) and rank == 0:
        with h5py.File(dest_pl, 'w') as f:
            f.create_dataset('fields', shape = (time_steps, 5, 13, 721, 1440), dtype='f')
    comm.Barrier()
    
    paths = paths_pl[year]
    
    for path in paths:
        var_name = path.split(year)[0].split('/')[-2]
        writetofile_pl(path, dest_pl, [pl_vars.index(var_name)], [var_name], Nimgtot=time_steps)
    
