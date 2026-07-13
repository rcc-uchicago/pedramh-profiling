import h5py as h5
from tqdm.auto import tqdm

import numpy as np
import pickle

surface_variables = ["skin_temperature",
                     "surface_pressure",
                     "2m_temperature",
                     "2m_specific_humidity", 
                     "10m_u_component_of_wind", 
                     "10m_v_component_of_wind"]

multi_variables = ["temperature",
                  "u_component_of_wind",
                  "v_component_of_wind",
                  "geopotential",
                  "specific_humidity",
                  "specific_cloud_liquid_water_content",
                  "specific_cloud_ice_water_content",
                  "fraction_of_cloud_cover"]

forcing_variables = ["DSWRFtoa", "sea_surface_temperature", "sea_ice_cover"] # in the diagnostic dir
invariant_variables = ["geopotential_at_surface", "land_sea_mask"]   
diagnostic_variables = ["USWRFtoa",
                        "ULWRFtoa", 
                       "USWRFsfc",
                       "ULWRFsfc",
                       "DSWRFsfc",
                       "DLWRFsfc",
                       "PRATEsfc",
                       "LHTFLsfc",
                       "SHTFLsfc"]

def calculate_climatologies(f, year_start = 1990, year_end = 1995):
    surface = f['surface'] # nt nlat nlon c
    multi = f['multilevel'] # nt nlevel nlat nlon c
    diag = f['diagnostic']
    years = f['year']
    
    nt = surface.shape[0]
    nlat, nlon, nsurface = surface.shape[1], surface.shape[2], surface.shape[3]
    nlevel = multi.shape[1]
    nmulti = multi.shape[-1]
    ndiag = diag.shape[-1]
    
    surface_mean = np.zeros((nlat, nlon, nsurface))
    multi_mean = np.zeros((nlevel, nlat, nlon, nmulti))
    diag_mean = np.zeros((nlat, nlon, ndiag))
    n = 0 
    for t in tqdm(range(nt)):
        year = years[t]

        if year < year_start:
            continue
        elif year > year_end: 
            break 
        else:
            n += 1 
            delta_surface = (surface[t] - surface_mean) / n 
            surface_mean += delta_surface 
            
            delta_multi = (multi[t] - multi_mean) / n
            multi_mean += delta_multi 
            
            delta_diag = (diag[t] - diag_mean) / n
            diag_mean += delta_diag 

    clim_dict = {}

    for i, var in enumerate(surface_variables):
        clim_dict[var] = surface_mean[..., i] 

    for i, var in enumerate(multi_variables):
        clim_dict[var] = multi_mean[..., i] 

    for i, var in enumerate(diagnostic_variables):
        clim_dict[var] = diag_mean[..., i] 

    return clim_dict 

file = "/glade/derecho/scratch/ayz/AMIP_train.h5"
data = h5.File(file, 'r', rdcc_nbytes=1024**3, rdcc_nslots=521, rdcc_w0=1)

clim_dict = calculate_climatologies(data['train']) 

with open("/glade/derecho/scratch/ayz/climatology_1990_1996.pkl", 'wb') as f:
    # Serialize and write the data to the file
    pickle.dump(clim_dict, f)