import h5py as h5
from tqdm.auto import tqdm

import numpy as np
import pickle

def get_mean_std(data):
    # data in shape nt nlat nlon c
    return np.nanmean(data, axis=(0, 1, 2)), np.nanstd(data, axis=(0, 1, 2))

def reduce_list(list_):
    return np.mean(np.array(list_), axis=0)

def calculate_mean_std(f):
    surface = f['surface'] # nt nlat nlon c
    multi = f['multilevel'] # nt nlevel nlat nlon c
    diag = f['diagnostic'] # nt nlat nlon c
    forcing = f['forcing'] # nt nlat nlon c
    years = f['year']
    
    nt = surface.shape[0]
    nlat, nlon, nsurface = surface.shape[1], surface.shape[2], surface.shape[3]
    nlevel = multi.shape[1]
    nmulti = multi.shape[-1]
    ndiag = diag.shape[-1]
    nforcing = forcing.shape[-1]

    surface_mean = np.zeros(nsurface)
    surface_mean_list = []
    surface_std = np.zeros(nsurface)
    surface_std_list = []

    multi_mean = np.zeros((nlevel, nmulti))
    multi_mean_list = []
    multi_std = np.zeros((nlevel, nmulti))
    multi_std_list = []

    diag_mean = np.zeros(ndiag)
    diag_std = np.zeros(ndiag)
    diag_mean_list = []
    diag_std_list = []

    forcing_mean = np.zeros(nforcing)
    forcing_std = np.zeros(nforcing)
    forcing_mean_list = []
    forcing_std_list = []

    t_start = 16072 #1990
    t_final = 49676 #2012

    #batch_size = 1084 
    batch_size = 1
    
    for t_start in tqdm(range(t_start, t_final, batch_size)):
        t_end = t_start + batch_size
        
        surface_data = np.array(surface[t_start:t_end]) # nt nlat nlon
        sm, ss = get_mean_std(surface_data)
        surface_mean_list.append(sm)
        surface_std_list.append(ss)

        diag_data = np.array(diag[t_start:t_end]) # nt nlat nlon
        dm, ds = get_mean_std(diag_data)
        diag_mean_list.append(dm)
        diag_std_list.append(ds)

        forcing_data = np.array(forcing[t_start:t_end]) # nt nlat nlon
        fm, fs = get_mean_std(forcing_data)
        forcing_mean_list.append(fm)
        forcing_std_list.append(fs)

        temp_mean = np.zeros((nlevel, nmulti))
        temp_std = np.zeros((nlevel, nmulti))
        for l in range(nlevel):
            multi_data = np.array(multi[t_start:t_end, l]) # nt nlat nlon
            mm, ms = get_mean_std(multi_data)
            temp_mean[l] = mm
            temp_std[l] = ms

        multi_mean_list.append(temp_mean)
        multi_std_list.append(temp_std)

    surface_mean = reduce_list(surface_mean_list)
    surface_std = reduce_list(surface_std_list)

    multi_mean = reduce_list(multi_mean_list)
    multi_std = reduce_list(multi_std_list)
    
    forcing_mean = reduce_list(forcing_mean_list)
    forcing_std = reduce_list(forcing_std_list)
    
    diag_mean = reduce_list(diag_mean_list)
    diag_std = reduce_list(diag_std_list)
    
    return {'surface_mean': surface_mean, 'surface_std': surface_std, 'multi_mean': multi_mean, 'multi_std': multi_std, 'diag_mean': diag_mean, 'diag_std': diag_std, 'forcing_mean': forcing_mean, 'forcing_std': forcing_std}


file = "/glade/derecho/scratch/ayz/AMIP_train.h5"
data = h5.File(file, 'r', rdcc_nbytes=1024**3, rdcc_nslots=521, rdcc_w0=1)

norm_stats_dict = calculate_mean_std(data['train']) 

with open("/glade/derecho/scratch/ayz/norm_stats_dict.pkl", 'wb') as f:
    # Serialize and write the data to the file
    pickle.dump(norm_stats_dict, f)