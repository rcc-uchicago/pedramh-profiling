
from YParams import YParams
import os
import numpy as np
import argparse
import xarray as xr

def get_variable_list(params):
    variable_list_surface = params.surface_variables.copy()
    if hasattr(params, 'land_variables'):
        variable_list_surface.extend(params.land_variables)
    if hasattr(params, 'ocean_variables'):
        variable_list_surface.extend(params.ocean_variables)
    if hasattr(params, 'diagnostic_variables'):
        variable_list_surface.extend(params.diagnostic_variables)
    if hasattr(params, 'varying_boundary_variables'):
        variable_list_surface.extend(params.varying_boundary_variables)
    return variable_list_surface
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='v2.0/config/PANGU_S2S.yaml', type=str)
    parser.add_argument("--config", default='S2S', type=str)
    parser.add_argument("--data_name", default='pangu_s2s_1979-2018', type=str)
    parser.add_argument("--mean_npz", default='/eagle/MDClimSim/awikner/pangu_s2s/normalize_mean.npz', type=str)
    parser.add_argument("--std_npz", default='/eagle/MDClimSim/awikner/pangu_s2s/normalize_std.npz', type=str)

    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    mean_data = np.load(args.mean_npz)
    std_data = np.load(args.std_npz)
    level_units = '.0'

    surface_variables = get_variable_list(params)
    upper_air_mean_dict = dict(zip(params.upper_air_variables, [(("Z",), np.array([mean_data[f'{variable}_{int(level)}{level_units}'][0] for level in params.levels])) for variable in params.upper_air_variables]))
    upper_air_std_dict = dict(zip(params.upper_air_variables, [(("Z",), np.array([std_data[f'{variable}_{int(level)}{level_units}'][0] for level in params.levels])) for variable in params.upper_air_variables]))
    surface_mean_dict = dict(zip(surface_variables, [mean_data[variable][0] for variable in surface_variables]))
    surface_std_dict = dict(zip(surface_variables, [std_data[variable][0] for variable in surface_variables]))
    upper_air_mean_ds = xr.Dataset(
        upper_air_mean_dict,
        coords = {
            "Z": np.array(params.levels, dtype = np.float64)
        }
    )
    upper_air_std_ds = xr.Dataset(
        upper_air_std_dict,
        coords = {
            "Z": np.array(params.levels, dtype = np.float64)
        }
    )
    surface_mean_ds = xr.Dataset(surface_mean_dict)
    surface_std_ds = xr.Dataset(surface_std_dict)

    upper_air_mean_ds.to_netcdf(os.path.join(params.data_dir, f'{args.data_name}_mean.nc'))
    upper_air_std_ds.to_netcdf(os.path.join(params.data_dir, f'{args.data_name}_std.nc'))
    surface_mean_ds.to_netcdf(os.path.join(params.data_dir, f'{args.data_name}_surface_mean.nc'))
    surface_std_ds.to_netcdf(os.path.join(params.data_dir, f'{args.data_name}_surface_std.nc'))

