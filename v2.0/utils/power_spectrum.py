
import sys
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from collections import OrderedDict
import torch
import matplotlib
import os
import matplotlib.colors as mcolors

import pandas as pd
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
from collections import OrderedDict
from itertools import product
# sns.set_style('darkgrid')
# sns.set_context('notebook')


matplotlib.use('Agg')  # Set the backend to 'Agg'


# Assume field is your xarray DataArray with dimensions (lat, lon)
# Load or define your field DataArray here

def zonal_averaged_power_spectrum(field, time_avg=True):
    """
    This function calculates the zonal averaged power spectrum of a given field. It is designed to work with xarray DataArrays or Datasets that have 'lat', 'lon', and optionally 'time' dimensions. The function first transposes the dimensions to ensure 'lat' and 'lon' are the first two dimensions, then performs a Fast Fourier Transform (FFT) along the 'lon' axis to compute the power spectrum. The power spectrum is then averaged over 'lat' and 'time' (if present) to produce the zonal averaged power spectrum.

    Parameters:
    - field (xarray.DataArray or xarray.Dataset): The input field for which to calculate the zonal averaged power spectrum.

    Returns:
    - power_spectrum_avg (xarray.DataArray): The zonal averaged power spectrum of the input field.
    """
    # but work if i have several variables on one plev or lead_time 
    initial_field = field.copy()
   
    # if field is a xr.Dataset:
    if isinstance(field, xr.Dataset):
        vars = list(initial_field.data_vars)
        field = field.to_array(dim='var')
        print("Dataset detected. Converting to a DataArray with the first dimension being the variable.")
        print("Shape:", field.shape)
        print("Variables detected:", vars)
    else: 
        vars=None
        print("DataArray detected.")
    
    field = field.transpose('lon', 'lat', ...)
    if 'time' in field.coords:
        field = field.transpose('lon', 'lat', 'time', ...)
    dims = list(field.dims)
    print("Dimensions detected:", dims)
    if not 'lat' in dims or not 'lon' in dims:
        raise ValueError("Latitude and longitude coordinates must be present in the field.")
    # create a dict to store the coordinates
    coords = OrderedDict()
    for dim in dims:
        coords[dim] = np.array(field[dim])
    n_lon = len(coords['lon'])
    

    ###########################################################################################
    field_fft = np.fft.rfft(field, axis=0, norm='forward') # Convention used: the first Fourier coefficient is the mean of the field

    # Compute the power spectrum (squared magnitude of Fourier coefficients)
    power_spectrum = np.abs(field_fft)**2

    # Define the zonal wavenumbers
    nx = n_lon
    print("n_x =" , nx)
    k_x = np.fft.fftfreq(nx, d=1/nx)

    # Only take the positive frequencies (or the first half if using real FFT)
    k_x = k_x[:nx//2]
    power_spectrum = power_spectrum[:nx//2]
    # count the positive frequencies twice except for the first one (zero frequency), because the FFT of a real function is symmetric
    power_spectrum[1:] *= 2
    # multiply by a factor cos(pi latitude[i] / 180) in axis 1
    # C0 = 40.075*10**6 # Earth's circumference in meters
    weights = np.cos(np.pi * coords['lat']/180) # * C0
    weights = weights.reshape(1, -1, *([1] * (power_spectrum.ndim - 2)))
    power_spectrum *= weights 

    # Average the power spectrum over latitudes and time (axis 1 and 2)
    if 'time' not in coords or time_avg==False:
        power_spectrum_avg = power_spectrum.mean(axis=1)
    else:
        print("Warning: 'time' dimension detected. Averaging over the time dimension.")
        power_spectrum_avg = power_spectrum.mean(axis=(1, 2))

       # print(len(initial_field.coords))
    print("Shape after averaged FFT: ", power_spectrum_avg.shape)
    ################################################################################################

    # convert to xarray Dataset or DataArray
    new_coords = coords.copy()
    # replace lon by k_x (at same position in the ord dict)
    new_coords['lon'] = k_x
    # drop the lat dimension
    new_coords.pop('lat')
    if 'time' in new_coords and time_avg:
        new_coords.pop('time')
    # print("New coordinates:", new_coords)
    power_spectrum_avg = xr.DataArray(power_spectrum_avg, coords=new_coords.values(), dims=new_coords.keys())
    if vars is not None:
        power_spectrum_avg = power_spectrum_avg.to_dataset(dim='var')
        # rename 'lon' to 'k_x'
    power_spectrum_avg = power_spectrum_avg.rename({'lon':'k_x'})

    return k_x, power_spectrum_avg

def plot_power_spectrum(power_spectrum_avg_preds, power_spectrum_avg_gt, preds_times, filename, lead_times,
                             use_sigma_levels=False, var_dict=None):
    """ Plot the power spectrum of the forecast and ground truth
    :param power_spectrum_avg_preds: xarray dataset, power spectrum of the forecast
    :param power_spectrum_avg_gt: xarray dataset, power spectrum of the ground truth
    :param preds_times: array, time values of the forecast
    :param filename: str, path to save the plot
    :param lead_times: list, lead times in hours
    :param use_sigma_levels: bool, whether to use sigma levels or pressure levels for non-zg variables
    :param var_dict: dict, variables to levels mapping {'var': [levels]}
    """
    import matplotlib.pyplot as plt
    
    if var_dict is None:
        # Default variables and levels
        var_dict = {
            "tas": [],
            "zg": [50000],
            "ua": [0.4368]
        }
    
    # Get available variables
    vars = list(var_dict.keys())
    vars = [var for var in vars if var in power_spectrum_avg_preds.data_vars]
    
    # Print available dimensions and variables
    print(f"Available variables in power spectrum: {list(power_spectrum_avg_preds.data_vars)}")
    for var in vars:
        print(f"Dimensions for {var}: {power_spectrum_avg_preds[var].dims}")
        print(f"Shape for {var}: {power_spectrum_avg_preds[var].shape}")
    
    # Get available lead times
    available_lead_times = power_spectrum_avg_preds.lead_time.values
    print(f"Available lead times: {available_lead_times}")
    plot_lead_times = [lt for lt in lead_times if lt in available_lead_times]
    
    # Create figure
    fig, axs = plt.subplots(len(plot_lead_times), len(vars), figsize=(18, 20), squeeze=False)
    
    k_x_preds = power_spectrum_avg_preds.k_x.values
    k_x_gt = power_spectrum_avg_gt.k_x.values
    
    for i, lead_time in enumerate(plot_lead_times):
        for j, var in enumerate(vars):
            # Determine if this is a surface variable
            levels = var_dict[var]
            is_surface = len(levels) == 0
            
            # Determine which coordinate system to use (plev or lev)
            is_zg = var == 'zg' or var == 'geopotential_height'
            level_coord_name = 'plev' if is_zg or not use_sigma_levels else 'lev'
            
            if is_surface:
                # For surface variables (like tas)
                try:
                    # Select lead time first
                    preds_data = power_spectrum_avg_preds[var].sel(lead_time=lead_time)
                    gt_data = power_spectrum_avg_gt[var].sel(lead_time=lead_time)
                    
                    print(f"Surface var {var} shape after lead_time selection: {preds_data.shape}")
                    
                    # For surface variables, average over all dimensions except k_x
                    dims_to_reduce = [dim for dim in preds_data.dims if dim != 'k_x']
                    if dims_to_reduce:
                        preds_data = preds_data.mean(dim=dims_to_reduce)
                        gt_data = gt_data.mean(dim=dims_to_reduce)
                        print(f"Averaged over dimensions {dims_to_reduce} for surface var {var}")
                    
                    print(f"Final shape for plotting {var}: {preds_data.shape}")
                    
                    axs[i,j].plot(k_x_preds, preds_data.values, label='Forecast')
                    axs[i,j].plot(k_x_gt, gt_data.values, linestyle='--', label='Ground Truth')
                    
                    title = f"var = '{var}' (surface), lead time = {lead_time} hours"
                except Exception as e:
                    print(f"Error plotting surface variable {var}: {e}")
                    import traceback
                    traceback.print_exc()
                    title = f"var = '{var}' (ERROR)"
            else:
                # For variables with levels
                for level_idx, level in enumerate(levels):
                    try:
                        # Select lead time first
                        preds_data = power_spectrum_avg_preds[var].sel(lead_time=lead_time)
                        gt_data = power_spectrum_avg_gt[var].sel(lead_time=lead_time)
                        
                        print(f"Variable {var} shape after lead_time selection: {preds_data.shape}")
                        
                        # Select the specific vertical level
                        if level_coord_name in preds_data.dims:
                            preds_data = preds_data.sel({level_coord_name: level}, method='nearest')
                            gt_data = gt_data.sel({level_coord_name: level}, method='nearest')
                            
                            # Format level label based on coordinate system
                            actual_level = float(preds_data[level_coord_name].values) if level_coord_name in preds_data.coords else level
                            if level_coord_name == 'plev':
                                level_label = f"{actual_level/100:.0f} hPa"
                            else:
                                level_label = f"{actual_level:.4f} σ"
                                
                            print(f"Variable {var} shape after selecting {level_coord_name}={level}: {preds_data.shape}")
                        else:
                            level_label = f"level {level}"
                            print(f"Warning: {level_coord_name} dimension not found for {var}")
                        
                        # Now reduce any remaining dimensions except k_x
                        if len(preds_data.shape) > 1:
                            print(f"Warning: {var} at {level_label} still has multiple dimensions after level selection")
                            print(f"Dimensions: {preds_data.dims}")
                            dims_to_reduce = [dim for dim in preds_data.dims if dim != 'k_x']
                            if dims_to_reduce:
                                preds_data = preds_data.mean(dim=dims_to_reduce)
                                gt_data = gt_data.mean(dim=dims_to_reduce)
                                print(f"Reduced dimensions for {var} at {level_label}")
                        
                        axs[i,j].plot(k_x_preds, preds_data.values, label=f'Forecast {level_label}')
                        axs[i,j].plot(k_x_gt, gt_data.values, linestyle='--', label=f'Ground Truth {level_label}')
                        
                        title = f"var = '{var}' at {level_label}, lead time = {lead_time} hours"
                    except Exception as e:
                        print(f"Error plotting variable {var} at level {level}: {e}")
                        import traceback
                        traceback.print_exc()
                        title = f"var = '{var}' at level {level} (ERROR)"
            
            # Format the plot
            axs[i,j].set_yscale('log')
            axs[i,j].set_xscale('log')
            axs[i,j].set_xlabel(r'Zonal Wavenumber $k_x$')
            axs[i,j].set_ylabel('Energy Spectrum')
            axs[i,j].set_title(title)
            axs[i,j].grid(True)
            axs[i,j].legend()
    
    plt.suptitle(f"Latitude-averaged Instantaneous Fourier Spectrum", y=1.01)
    plt.tight_layout()
    plt.savefig(filename, pad_inches=0.1, bbox_inches='tight')
    plt.close(fig)
    
# def plot_bias(bias_pred, bias_gt, filename,
#               vars=["tas", "zg", "ua", "hus"], 
#               plevs=[None, 50000, 25000, 85000]):
#     """ Plot the power spectrum of the forecast and ground truth
#     :param bias_pred: xarray dataset, power spectrum of the forecast
#     :param bias_gt: xarray dataset, power spectrum of the ground truth
#     :param preds_times: array, time values of the forecast
#     :param filename: str, path to save the plot
#     :param lead_times: list, lead times in hours to plot
#     :param vars: list, variables to plot (default: ["ta", "zg", "ua"])
#     :param plevs: list, pressure levels in Pa to plot (default: [850*100, 500*100, 250*100])
#     """
#     # Filter variables that are present in the data
#     available_vars = [var for var in vars if var in bias_pred.data_vars]
#     if not available_vars:
#         raise ValueError(f"None of the specified variables {vars} are present in the data. Available variables: {list(bias_pred.data_vars)}")

#     # Filter pressure levels that are present in the data
#     available_plevs = [plev for plev in plevs if plev in bias_pred.plev.values and plev != None]
#     if not available_plevs:
#         raise ValueError(f"None of the specified pressure levels {plevs} are present in the data. Available levels: {bias_pred.plev.values}")

#     # Create subplots
#     # fig, axs = plt.subplots(len(plot_lead_times), len(available_vars), figsize=(18, 20), squeeze=False)
#     plot_dims =  (len(available_vars) // 2, len(available_vars) // 2 + len(available_vars) % 2)
#     fig, axs = plt.subplots(plot_dims[0], plot_dims[1], figsize=(6*plot_dims[1], 13), squeeze=False)#, subplot_kw={"projection": ccrs.PlateCarree()})

#     for i, j in product(range(plot_dims[0]), range(plot_dims[1])):
#         if j+i*plot_dims[1] < len(vars):
#             var, plev = vars[j + i*plot_dims[1]], plevs[j + i*plot_dims[1]]
#             if plev:
                
#                 var_bias_pred = bias_pred[var].sel(plev = plev)
#                 var_bias_gt = bias_gt[var].sel(plev = plev)
#             else:
#                 var_bias_pred = bias_pred[var]
#                 var_bias_gt = bias_gt[var]
#             var_bias_pred_aligned = var_bias_pred.squeeze().transpose(*var_bias_gt.dims)
#             var_bias_pred_aligned['lat'] = var_bias_gt.lat
#             diff = var_bias_pred_aligned - var_bias_gt
            
#             pcm = axs[i,j].pcolormesh(diff.lon, diff.lat, diff, cmap="RdBu_r")#, transform=ccrs.PlateCarree())
#             contours = axs[i,j].contour(var_bias_gt.lon, var_bias_gt.lat, var_bias_gt, colors="black", linewidths=1)#, transform=ccrs.PlateCarree())
            
#             # Add continent outlines
#             #axs[i,j].add_feature(cfeature.COASTLINE, linewidth=1)
#             #axs[i,j].add_feature(cfeature.BORDERS, linestyle=":")
            
#             # Add colorbar
#             cbar = plt.colorbar(pcm, ax=axs[i,j], orientation="horizontal", fraction=0.046, pad=0.04)
#             if plev:
#                 cbar.set_label(f"{var}")
#             else:
#                 cbar.set_label(f"{var} {plev}")
#             axs[i,j].clabel(contours, inline=True, fontsize=8)
    
#     plt.suptitle(f"Prediction Bias (Pred. - Truth)", y=1.01)
#     plt.tight_layout() 
#     plt.savefig(filename, pad_inches=0.1, bbox_inches='tight')
#     plt.close(fig)

#     return fig, axs

def plot_bias(bias_pred, bias_gt, filename, var_level_dict):
    """
    """
    # Filter variables based on presence in both datasets
    print(f"Bias pred: {bias_pred.data_vars}")
    print(f"Bias gt: {bias_gt.data_vars}")
    print(f"Var level dict: {var_level_dict}")
    available_vars_dict = {
        var: levels for var, levels in var_level_dict.items()
        if var in bias_pred.data_vars and var in bias_gt.data_vars
    }

    if not available_vars_dict:
        print(f"Warning: No valid variables found to plot based on var_level_dict and data availability. Skipping bias plot.")
        return

    num_vars = len(available_vars_dict)
    var_list = list(available_vars_dict.keys())
    print(f"Available variables: {var_list}")

    # Determine subplot layout
    cols = 2 if num_vars > 1 else 1
    rows = (num_vars + cols - 1) // cols
    plot_dims = (rows, cols)

    fig, axs = plt.subplots(
        plot_dims[0], plot_dims[1],
        figsize=(7 * plot_dims[1], 6 * plot_dims[0]),
        squeeze=False,
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    axs_flat = axs.flatten()

    plot_idx = 0
    for var in var_list:
        levels = available_vars_dict[var]
        ax = axs_flat[plot_idx]

        is_surface = not levels
        level_coord_name = None
        level_value = None
        level_label = "Surface"

        try:
            if not is_surface:
                # one level per panel? 
                level_value = levels[0] 

                # Consistent with plot_acc_over_lead_time heuristic
                threshold = 10.0 
                if 0 < level_value < threshold: # Assumed sigma
                     level_coord_name = 'lev'
                     level_label = f"{level_value:.4f} σ"
                elif level_value >= threshold: # Assumed pressure (Pa)
                     level_coord_name = 'plev'
                     level_label = f"{level_value/100:.0f} hPa"
                else:
                    raise ValueError(f"Invalid level value {level_value} for non-surface variable {var}")

                # Check if the determined coordinate exists before selecting
                if level_coord_name not in bias_pred.coords or level_coord_name not in bias_gt.coords:
                     print(f"Warning: Determined coordinate '{level_coord_name}' not found for variable '{var}'. Skipping plot.")
                     ax.set_title(f"{var} ({level_label})\nCoord Missing")
                     ax.set_axis_off()
                     plot_idx += 1
                     continue

                # Select data using the determined coordinate
                var_bias_pred = bias_pred[var].sel(**{level_coord_name: level_value}, method='nearest')
                var_bias_gt = bias_gt[var].sel(**{level_coord_name: level_value}, method='nearest')

            else: # Surface variable
                var_bias_pred = bias_pred[var]
                var_bias_gt = bias_gt[var]

            var_bias_pred_aligned = var_bias_pred.squeeze().transpose(*var_bias_gt.dims)
            var_bias_pred_aligned = var_bias_pred_aligned.assign_coords(lat=var_bias_gt.lat, lon=var_bias_gt.lon)
            diff = var_bias_pred_aligned - var_bias_gt

            max_abs_diff = np.max(np.abs(diff.fillna(0).values))
            cmap_limit = max_abs_diff if max_abs_diff > 1e-9 else 1.0

            pcm = ax.pcolormesh(diff.lon, diff.lat, diff.values, cmap="RdBu_r",
                                vmin=-cmap_limit, vmax=cmap_limit, transform=ccrs.PlateCarree())
            contours = ax.contour(var_bias_gt.lon, var_bias_gt.lat, var_bias_gt.values,
                                  colors="black", linewidths=0.8, transform=ccrs.PlateCarree())

            ax.coastlines(linewidth=0.5)
            ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            cbar = plt.colorbar(pcm, ax=ax, orientation="horizontal", fraction=0.046, pad=0.1, extend='both')
            cbar.set_label(f"Bias")
            ax.clabel(contours, inline=True, fontsize=8, fmt='%g')
            ax.set_title(f"Bias: {var} ({level_label})")
            # --- End Plotting Logic ---

            plot_idx += 1 # Increment successfully plotted index

        except Exception as e:
            print(f"ERROR plotting {var} at level '{level_label}': {e}")
            traceback.print_exc()
            ax.set_title(f"{var} ({level_label})\nPlotting Error")
            ax.set_axis_off()
            plot_idx += 1

    # Hide unused axes
    for k in range(plot_idx, len(axs_flat)):
        fig.delaxes(axs_flat[k])

    plt.suptitle(f"Prediction Bias (Forecast Mean - Climatology Mean)", y=1.02, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"Saved bias plot (level value heuristic) to: {filename}")
    except Exception as e:
        print(f"ERROR saving bias plot '{filename}': {e}")
    finally:
        plt.close(fig)

# def plot_power_spectrum(power_spectrum_avg_preds, power_spectrum_avg_gt, preds_times, filename, lead_times,
#                              var_dict=None):
#     """ Plot the power spectrum of the forecast and ground truth
#     :param power_spectrum_avg_preds: xarray dataset, power spectrum of the forecast
#     :param power_spectrum_avg_gt: xarray dataset, power spectrum of the ground truth
#     :param preds_times: array, time values of the forecast
#     :param filename: str, path to save the plot
#     :param lead_times: list, lead times in hours
#     :param var_dict: dict, variables to levels mapping {'var': [levels]}
#     """
#     import matplotlib.pyplot as plt
    
#     if var_dict is None:
#         # Default variables and levels
#         var_dict = {
#             "tas": [],
#             "zg": [50000],
#             "ua": [0.4368]
#         }
    
#     # Get available variables
#     vars = list(var_dict.keys())
#     vars = [var for var in vars if var in power_spectrum_avg_preds.data_vars]
    
#     # Print available dimensions and variables
#     print(f"Available variables in power spectrum: {list(power_spectrum_avg_preds.data_vars)}")
#     for var in vars:
#         print(f"Dimensions for {var}: {power_spectrum_avg_preds[var].dims}")
    
#     # Get available lead times
#     available_lead_times = power_spectrum_avg_preds.lead_time.values
#     print(f"Available lead times: {available_lead_times}")
#     plot_lead_times = [lt for lt in lead_times if lt in available_lead_times]
    
#     # Create figure
#     fig, axs = plt.subplots(len(plot_lead_times), len(vars), figsize=(18, 20), squeeze=False)
    
#     k_x_preds = power_spectrum_avg_preds.k_x.values
#     k_x_gt = power_spectrum_avg_gt.k_x.values
    
#     for i, lead_time in enumerate(plot_lead_times):
#         for j, var in enumerate(vars):
#             # Determine if this is a surface variable
#             levels = var_dict[var]
#             is_surface = len(levels) == 0
            
#             # Determine which coordinate system to use (plev or lev)
#             is_zg = var == 'zg' or var == 'geopotential_height'
#             level_coord_name = 'plev' if is_zg else 'lev'
            
#             if is_surface:
#                 # For surface variables (like tas)
#                 try:
#                     preds_data = power_spectrum_avg_preds[var].sel(lead_time=lead_time)
#                     gt_data = power_spectrum_avg_gt[var].sel(lead_time=lead_time)
                    
#                     axs[i,j].plot(k_x_preds, preds_data.values, label='Forecast')
#                     axs[i,j].plot(k_x_gt, gt_data.values, linestyle='--', label='Ground Truth')
                    
#                     title = f"var = '{var}' (surface), lead time = {lead_time} hours"
#                 except Exception as e:
#                     print(f"Error plotting surface variable {var}: {e}")
#                     title = f"var = '{var}' (ERROR)"
#             else:
#                 # For variables with levels
#                 for level in levels:
#                     try:
#                         # Select the data with the appropriate level
#                         preds_data = power_spectrum_avg_preds[var].sel(lead_time=lead_time, **{level_coord_name: level}, method='nearest')
#                         gt_data = power_spectrum_avg_gt[var].sel(lead_time=lead_time, **{level_coord_name: level}, method='nearest')
                        
#                         # Format level label based on coordinate system
#                         if level_coord_name == 'plev':
#                             level_label = f"{level/100:.0f} hPa"
#                         else:
#                             level_label = f"{level:.4f} σ"
                        
#                         axs[i,j].plot(k_x_preds, preds_data.values, label=f'Forecast {level_label}')
#                         axs[i,j].plot(k_x_gt, gt_data.values, linestyle='--', label=f'Ground Truth {level_label}')
                        
#                         title = f"var = '{var}' at {level_label}, lead time = {lead_time} hours"
#                     except Exception as e:
#                         print(f"Error plotting variable {var} at level {level}: {e}")
#                         title = f"var = '{var}' at level {level} (ERROR)"
            
#             # Format the plot
#             axs[i,j].set_yscale('log')
#             axs[i,j].set_xscale('log')
#             axs[i,j].set_xlabel(r'Zonal Wavenumber $k_x$')
#             axs[i,j].set_ylabel('Energy Spectrum')
#             axs[i,j].set_title(title)
#             axs[i,j].grid(True)
#             axs[i,j].legend()
    
#     plt.suptitle(f"Latitude-averaged Instantaneous Fourier Spectrum", y=1.01)
#     plt.tight_layout()
#     plt.savefig(filename, pad_inches=0.1, bbox_inches='tight')
#     plt.close(fig)
    
#     return fig, axs

# # Amaury's code
# def plot_power_spectrum(power_spectrum_avg_preds, preds_times, vars=["temperatire", "geopotential", "u_component_of_wind"],
#                          plevs = [850, 500, 250], lead_times=[6, 48, 120]):
#     """ Plot the power spectrum of the ground truth and the forecast
#     :param power_spectrum_avg: xarray dataset, power spectrum of the ground truth
#     :param power_spectrum_avg_preds: xarray dataset, power spectrum of the forecast
#     :param preds_times: array, time values of the forecast
#     :param name_fc: str, name of the forecast
#     """
#     # Check that len(vars) == len(plevs)
#     assert len(vars) == len(plevs), 'vars and plevs must have the same length'

#     # Loop through variables and pressure levels to plot
#     fig, axs = plt.subplots(len(lead_times), len(vars), figsize=(18, 20))
#     # k_x_gt = power_spectrum_avg_gt.k_x.values
#     k_x_preds = power_spectrum_avg_preds.k_x.values
#     for i, lead_time in enumerate(lead_times):
#         for j, (var, plev) in enumerate(zip(vars, plevs)):
#             power_spectrum_avg_preds2 = power_spectrum_avg_preds[var].sel(lead_time=lead_time).sel(lev=plev)
#             # power_spectrum_avg2 = power_spectrum_avg_gt[var].sel(time=preds_times + timedelta(hours=lead_time)).mean('time').sel(plev=plev)

#             # axs[i,j].plot(k_x_gt, power_spectrum_avg2, label='Ground Truth')
#             axs[i,j].plot(k_x_preds, power_spectrum_avg_preds2)
#             axs[i,j].legend()
#             axs[i,j].set_yscale('log')
#             axs[i,j].set_xscale('log')
#             axs[i,j].set_xlabel(r'Zonal Wavenumber $k_x$')
#             axs[i,j].set_ylabel('Energy Spectrum')
#             if i==0:
#                 axs[i,j].set_title(f"var = '{var}' at {int(plev)} hPa, lead time = {lead_time} hours")
#             else:
#                 axs[i,j].set_title(f"var = '{var}' at {int(plev)} hPa, lead time = {lead_time//24} days")
#             axs[i,j].grid(True)

#     plt.suptitle(f"Latitude-averaged Instantaneous Fourier Spectrum", y = 1.01)
#     plt.tight_layout() 
#     plt.savefig(f"spectrum_results.png", pad_inches=0.1, bbox_inches='tight')
#     return fig, axs

# def plot_power_spectrum_test(power_spectrum_avg_preds, power_spectrum_avg_gt, preds_times, filename, lead_times,
#                              var_dict=None):
#     """ Plot the power spectrum of the forecast and ground truth
#     :param power_spectrum_avg_preds: xarray dataset, power spectrum of the forecast
#     :param power_spectrum_avg_gt: xarray dataset, power spectrum of the ground truth
#     :param preds_times: array, time values of the forecast
#     :param filename: str, path to save the plot
#     :param lead_times: list, lead times in hours
#     :param var_dict: dict, variables to levels mapping {'var': [levels]}
#     """
#     if var_dict is None:
#         # some default args for testing if not specified
#         var_dict = {
#             "tas": [],
#             "zg": [50000],
#             "ua": [0.4368]
#         }
    
#     # Get available variables
#     available_vars = [var for var in var_dict.keys() if var in power_spectrum_avg_preds.data_vars]
#     if not available_vars:
#         raise ValueError(f"None of the specified variables {list(var_dict.keys())} are present in the data. Available variables: {list(power_spectrum_avg_preds.data_vars)}")

#     # Handle lead times
#     available_lead_times = power_spectrum_avg_preds.lead_time.values
#     print(f"Available lead times: {available_lead_times}")
#     plot_lead_times = [lt for lt in lead_times if lt in available_lead_times]
#     if not plot_lead_times:
#         raise ValueError(f"None of the specified lead times {lead_times} are present in the data. Available lead times: {available_lead_times}")

#     # Create subplots
#     fig, axs = plt.subplots(len(available_vars), len(plot_lead_times), figsize=(18, 20), squeeze=False)

#     k_x_preds = power_spectrum_avg_preds.k_x.values
#     k_x_gt = power_spectrum_avg_gt.k_x.values

#     for i, var in enumerate(available_vars):
#         # Determine coordinate system to use based on variable
#         level_coord_name = 'plev' if var == 'zg' or var == 'geopotential_height' else 'lev'
#         levels = var_dict[var]
        
#         # For surface variables, just plot the variable without level selection
#         is_surface = len(levels) == 0
        
#         for j, lead_time in enumerate(plot_lead_times):
#             if is_surface:
#                 # Handle surface variables (no level selection)
#                 try:
#                     power_spectrum_avg_preds2 = power_spectrum_avg_preds[var].sel(lead_time=lead_time)
#                     axs[i,j].plot(k_x_preds, power_spectrum_avg_preds2.values, label=f'Forecast')
                    
#                     power_spectrum_avg_gt2 = power_spectrum_avg_gt[var].sel(lead_time=lead_time)
#                     axs[i,j].plot(k_x_gt, power_spectrum_avg_gt2.values, linestyle='--', label=f'Ground Truth')
#                 except KeyError as e:
#                     print(f"Warning: Could not select data for var={var}, lead_time={lead_time}. Error: {e}")
#                     continue
                
#                 title = f"var = '{var}', lead time = {lead_time} hours"
#             else:
#                 # Handle variables with levels
#                 for level in levels:
#                     try:
#                         # Format level label based on coordinate system
#                         if level_coord_name == 'plev':
#                             level_label = f"{level/100:.0f} hPa"
#                         else:
#                             level_label = f"{level:.4f} σ"
                            
#                         power_spectrum_avg_preds2 = power_spectrum_avg_preds[var].sel(lead_time=lead_time, **{level_coord_name: level}, method='nearest')
#                         axs[i,j].plot(k_x_preds, power_spectrum_avg_preds2.values, label=f'Forecast {level_label}')
                        
#                         power_spectrum_avg_gt2 = power_spectrum_avg_gt[var].sel(lead_time=lead_time, **{level_coord_name: level}, method='nearest')
#                         axs[i,j].plot(k_x_gt, power_spectrum_avg_gt2.values, linestyle='--', label=f'Ground Truth {level_label}')
#                     except KeyError as e:
#                         print(f"Warning: Could not select data for var={var}, lead_time={lead_time}, {level_coord_name}={level}. Error: {e}")
#                         continue
                
#                 title = f"var = '{var}', lead time = {lead_time} hours"

#             # Format plot
#             axs[i,j].set_yscale('log')
#             axs[i,j].set_xscale('log')
#             axs[i,j].set_xlabel(r'Zonal Wavenumber $k_x$')
#             axs[i,j].set_ylabel('Energy Spectrum')
#             axs[i,j].set_title(title)
#             axs[i,j].grid(True)
#             axs[i,j].legend()
    
#     plt.suptitle(f"Latitude-averaged Instantaneous Fourier Spectrum", y=1.01)
#     plt.tight_layout() 
#     plt.savefig(filename, pad_inches=0.1, bbox_inches='tight')
#     plt.close(fig)

#     return fig, axs


# def plot_acc_over_lead_time(acc, lead_times_hours, vars=["tas", "ta", "zg", "ua"], plevs=[None, 85000, 50000, 25000], 
#                             colors=None, fontsize_title=14):
#     """
#     Plot the ACC over lead time for each variable and pressure level
#     :param acc: OrderedDict or xr.Dataset, ACC scores
#     :param lead_times_hours: list, lead times in hours
#     :param vars: list, variables to plot
#     :param plevs: list, pressure levels to plot (use None for surface variables)
#     :param colors: dict, colors for each model
#     :param fontsize_title: int, font size for the title
#     """
#     if isinstance(acc, xr.Dataset):
#         acc = {'Model': acc}

#     if colors is None:
#         colors = {'Pangu': 'blue'}

#     fig, axs = plt.subplots(len(vars), 1, figsize=(12, 5*len(vars)), squeeze=False)

#     for i, (var, plev) in enumerate(zip(vars, plevs)):
#         ax = axs[i, 0]
        
#         if plev is None:
#             title = f'ACC for {var}'
#         else:
#             title = f'ACC for {var} at {plev:.0f} hPa'
        
#         for model, ds in acc.items():
#             if var in ds:
#                 if 'plev' in ds[var].dims and plev is not None:
#                     data = ds[var].sel(plev=plev, method='nearest')
#                 elif 'plev' not in ds[var].dims:
#                     data = ds[var]
#                 else:
#                     print(f"Warning: {var} has unexpected dimensions. Skipping.")
#                     continue
                
#                 ax.plot(lead_times_hours, data.values, label=model, color=colors[model], marker='o')

#         ax.set_ylabel(f'{var} ACC')
#         ax.set_title(title, fontsize=fontsize_title)
#         ax.set_ylim(-0.3, 1.1)
#         ax.axhline(0, ls='--', c='0.', lw=1)
#         ax.axhline(0.6, ls='--', c='r', lw=1, label='ACC = 0.6')  # Add horizontal line at ACC = 0.6
#         ax.set_xlabel('Lead time [days]')
#         lead_times_ticks = np.arange(0, max(lead_times_hours)+1, 24)
#         ax.set_xticks(lead_times_ticks)
#         ax.set_xticklabels(['%d' % i for i in range(len(lead_times_ticks))])
#         ax.legend(loc='lower left')

#     plt.tight_layout()
#     return fig, axs

# Update the plot_acc_over_lead_time function
def plot_acc_over_lead_time(acc, lead_times_hours, var_dict=None, colors=None, fontsize_title=14):
    """
    Plot the ACC over lead time for each variable and level specified in var_dict.
    Supports both pressure levels (Pa) and sigma levels.
    
    Args:
        acc: OrderedDict or xr.Dataset, ACC scores
        lead_times_hours: list, lead times in hours
        var_dict: dict, variables and levels to plot {'var': [levels]}
        colors: dict, colors for each model
        fontsize_title: int, font size for the title
    """
    if isinstance(acc, xr.Dataset):
        acc = {'Pangu': acc}

    if colors is None:
        colors = {'Pangu': 'blue'}
        
    # Get variables and levels from var_dict if provided
    if var_dict:
        plot_vars = list(var_dict.keys())
    else:
        # Default variables if not specified
        plot_vars = ["tas", "zg", "ua"]
        var_dict = {
            "tas": [],
            "zg": [50000],
            "ua": [0.4368]
        }
    
    # Create a figure with subplots for each variable
    fig, axs = plt.subplots(len(plot_vars), 1, figsize=(12, 5*len(plot_vars)), squeeze=False)
    
    # Convert hours to days for x-axis
    lead_times_days = np.array(lead_times_hours) / 24
    lead_times_ticks = np.arange(0, max(lead_times_days)+1, 1)
    
    for i, var in enumerate(plot_vars):
        ax = axs[i, 0]
        levels = var_dict[var]
        
        # Determine if this is a surface variable
        is_surface = len(levels) == 0
        
        if not is_surface:
            for level in levels:
                for model, ds in acc.items():
                    if var in ds:
                        # Determine coordinate system based on level value
                        # set threshold for sigma levels
                        if level < 1.0:  
                            coord = 'lev'
                            level_label = f"{level:.4f} σ"
                        else: 
                            coord = 'plev'
                            level_label = f"{level/100:.0f} hPa"

                        if coord in ds[var].dims:
                            data = ds[var].sel({coord: level}, method='nearest')
                            ax.plot(lead_times_days, data.values, 
                                    label=f"{model} at {level_label}", 
                                    color=colors[model], marker='o')
        else:
            # Surface variable
            for model, ds in acc.items():
                if var in ds:
                    data = ds[var]
                    ax.plot(lead_times_days, data.values, 
                            label=model, color=colors[model], marker='o')
        
        # Set title based on variable type
        if is_surface:
            ax.set_title(f'ACC for {var}', fontsize=fontsize_title)
        else:
            ax.set_title(f'ACC for {var} at multiple levels', fontsize=fontsize_title)
        
        # Format the plot
        ax.set_ylabel(f'{var} ACC')
        ax.set_xlabel('Lead time [days]')
        ax.set_ylim(-0.3, 1.1)
        ax.axhline(0, ls='--', c='0.', lw=1)
        ax.axhline(0.6, ls='--', c='r', lw=1, label='ACC = 0.6')
        ax.set_xticks(lead_times_ticks)
        ax.set_xticklabels(['%d' % i for i in range(len(lead_times_ticks))])
        ax.legend(loc='lower left')
        ax.grid(True)

    plt.tight_layout()
    return fig, axs





import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import numpy as np
import time




# THIS IS THE GIF OF ANOMOLIES

from datetime import timedelta

def make_gif(combined_dataset, gt_combined_dataset, name_fc, var, output_filename, climatology = None, sample_index=0, level_coord_name = 'plev', plev=None):
    """
    Create a gif of the forecast anomalies for a single sample, evolving over all time steps up the maximum lead time,
    without using coastlines.
    """
    start_time = time.time()
    if climatology:
        print(f"Starting GIF creation for {var} anomalies")
    else:
        print(f"Starting GIF creation for {var}")

    # Data selection and setup
    if plev is not None:
        if level_coord_name == 'lev' and var != 'zg' and var != 'geopotential_height':
            data_inference = combined_dataset[var].isel(time=sample_index).sel(lev=plev, method='nearest')
            data_gt = gt_combined_dataset[var].isel(time=sample_index).sel(lev=plev, method='nearest')
            if climatology:
                climatology_data = climatology[var].sel(lev=plev, method='nearest')
        else:
            data_inference = combined_dataset[var].isel(time=sample_index).sel(plev=plev, method='nearest')
            data_gt = gt_combined_dataset[var].isel(time=sample_index).sel(plev=plev, method='nearest')
            if climatology:
                climatology_data = climatology[var].sel(plev=plev, method='nearest')
    else:
        data_inference = combined_dataset[var].isel(time=sample_index)
        data_gt = gt_combined_dataset[var].isel(time=sample_index)
        if climatology:
            climatology_data = climatology[var]

    print(f"Data shape - Inference: {data_inference.shape}, Ground Truth: {data_gt.shape}")
    print(f"Inference dimensions: {data_inference.dims}")
    print(f"Ground Truth dimensions: {data_gt.dims}")
    if climatology:
        print(f"Climatology dimensions: {climatology_data.dims}")
    print(f"Lead times: {data_inference.lead_time.values}")
    
    # Get the start time for this sample from the original dataset
    start_datetime = combined_dataset.time.values[sample_index]
    print(f"Start datetime for sample {sample_index}: {start_datetime}")

    # Calculate time range for all lead times
    time_range = [start_datetime + timedelta(hours=int(lt)) for lt in data_inference.lead_time.values]

    if climatology:
        # Prepare climatology
        if 'zsfc' in climatology_data:
            climatology_data = climatology_data.drop_vars('zsfc')
        
        # Ensure climatology has the correct spatial dimensions
        climatology_data = climatology_data.transpose('dayofyear', 'lat', 'lon')

        
        # print_info(climatology_aligned, "Aligned Climatology")
        climatology_data = climatology_data.assign_coords(lat=data_inference.lat)

        # Initialize anomalies arrays
        anomalies_inference = xr.zeros_like(data_inference)
        anomalies_gt = xr.zeros_like(data_gt)

        # Calculate anomalies for each lead time
        for i, forecast_datetime in enumerate(time_range):
            # Get the day of year for this forecast time
            # forecast_day = forecast_datetime.dayofyr
            

            # # Select the corresponding climatology
            # clim_for_leadtime = climatology_data.sel(dayofyear=np.array([date.dayofyr for date in climatology_data.dayofyear.values]) == forecast_day).squeeze(axis=0)
            clim_index = get_climatology_index(forecast_datetime)
            clim_for_leadtime = climatology_data.isel(dayofyear=clim_index)


            # Calculate anomalies for this lead time
            anomalies_inference[i] = data_inference[i] - clim_for_leadtime
            anomalies_gt[i] = data_gt[i] - clim_for_leadtime

            print(f"Processed lead time {data_inference.lead_time.values[i]} hours, forecast date: {forecast_datetime}")
    else:
        anomalies_inference = data_inference
        anomalies_gt = data_gt



    # vmin = float(anomalies_gt.min())
    # vmax = float(anomalies_gt.max())
    # print(f"Anomaly value range: {vmin} to {vmax}")

    max_abs_anomaly = abs(anomalies_gt).max()
    vmin = -max_abs_anomaly
    vmax = max_abs_anomaly
    print(f"Value range: {vmin} to {vmax}")

    

    # # Calculate symmetric range for anomalies
    # max_abs_anomaly = abs(anomalies_gt).max()
    # vmin = -max_abs_anomaly
    # vmax = max_abs_anomaly
    # print(f"Anomaly value range: {vmin} to {vmax}")

    # Figure setup
    fig_gif, axs = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    print("Figure created")

    # Create initial plots and colorbars
    im1 = axs[0].pcolormesh(anomalies_inference.lon, anomalies_inference.lat, 
                            anomalies_inference.isel(lead_time=0),
                            transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, cmap='RdBu_r')
    im2 = axs[1].pcolormesh(anomalies_gt.lon, anomalies_gt.lat, 
                            anomalies_gt.isel(lead_time=0),
                            transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, cmap='RdBu_r')
    
    # Add colorbars
    if climatology:
        plt.colorbar(im1, ax=axs[0], orientation='horizontal', pad=0.05, label='Anomaly')
        plt.colorbar(im2, ax=axs[1], orientation='horizontal', pad=0.05, label='Anomaly')
    else:
        plt.colorbar(im1, ax=axs[0], orientation='horizontal', pad=0.05, label=var)
        plt.colorbar(im2, ax=axs[1], orientation='horizontal', pad=0.05, label=var)

    axs[0].set_global()
    axs[1].set_global()
    if climatology:
        axs[0].set_title(f'{name_fc} Anomaly')
        axs[1].set_title('Ground Truth Anomaly')
    else:
        axs[0].set_title(f'{name_fc}')
        axs[1].set_title('Ground Truth')

    frame_times = []

    def plot(i):
        frame_start = time.time()

        lead_time = data_inference.lead_time.values[i]
        current_time = time_range[i]

        forecast = anomalies_inference.isel(lead_time=i)
        truth = anomalies_gt.isel(lead_time=i)

        # Update plot data
        im1.set_array(forecast.values.ravel())
        im2.set_array(truth.values.ravel())

        if level_coord_name == 'lev' and var != 'zg' and var != 'geopotential_height':
            var_up = f'{var}_{plev:.0f}' if plev is not None else var
        else:
            var_up = f'{var}_{plev:.0f}hPa' if plev is not None else var
        if climatology:
            title = f'{var_up} Anomaly at {current_time} (Lead time: {lead_time} hours, Sample {sample_index})'
        else:
            title = f'{var_up} at {current_time} (Lead time: {lead_time} hours, Sample {sample_index})'
        plt.suptitle(title, y=0.95)

        frame_end = time.time()
        frame_times.append(frame_end - frame_start)
        print(f"Frame {i} completed in {frame_end - frame_start:.2f} seconds")

    print("Starting animation creation")
    ani = animation.FuncAnimation(fig_gif, plot, frames=len(anomalies_inference.lead_time), repeat=False)
    
    print("Saving animation")
    ani.save(output_filename, writer='pillow', fps=1)
    plt.close(fig_gif)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"GIF saved as {output_filename}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average frame time: {np.mean(frame_times):.2f} seconds")
    print(f"Max frame time: {np.max(frame_times):.2f} seconds")

    return ani



def get_climatology_index(date):
    """
    Maps a date to the correct index in a 366-day climatology,
    adjusting for leap years vs. non-leap years.
    
    Args:
        date: A cftime.datetime or datetime object
        
    Returns:
        int: The index in the climatology (0-based)
    """
    # Get day of year (1-based)
    if hasattr(date, 'dayofyr'):
        day_of_year = date.dayofyr
    elif hasattr(date, 'dayofyear'):
        day_of_year = date.dayofyear
    else:
        day_of_year = date.timetuple().tm_yday
    
    # Check if it's a leap year
    if hasattr(date, 'calendar'):
        # cftime handling
        year = date.year
        if date.calendar in ['standard', 'gregorian', 'proleptic_gregorian']:
            is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        else:
            is_leap_year = False
    else:
        # Standard datetime handling
        year = date.year
        is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    
    # For non-leap years, if date is after Feb 28, adjust the index
    if not is_leap_year and day_of_year > 59:  # Feb 28 is the 59th day
        # Skip over the leap day in the climatology (Feb 29)
        clim_index = day_of_year  # This effectively adds 1 to the index
    else:
        clim_index = day_of_year - 1  # -1 for 0-based indexing
    
    return clim_index


# Makes GIF of th full zg field

# def make_gif_zg(combined_dataset, gt_combined_dataset, name_fc, var, output_filename, sample_index=0, plev=None):
#     """
#     Create a gif of the forecast for a single sample, evolving over lead times, without using coastlines.
#     """
#     start_time = time.time()
#     print(f"Starting GIF creation for {var}")

#     # Data selection and setup
#     if plev is not None:
#         data_inference = combined_dataset[var].isel(time=sample_index).sel(plev=plev, method='nearest')
#         data_gt = gt_combined_dataset[var].isel(time=sample_index).sel(plev=plev, method='nearest')
#     else:
#         data_inference = combined_dataset[var].isel(time=sample_index)
#         data_gt = gt_combined_dataset[var].isel(time=sample_index)

#     print(f"Data shape - Inference: {data_inference.shape}, Ground Truth: {data_gt.shape}")
#     print(f"Lead times: {data_inference.lead_time.values}")

#     vmin = float(min(data_inference.min(), data_gt.min()))
#     vmax = float(max(data_inference.max(), data_gt.max()))
#     print(f"Value range: {vmin} to {vmax}")

#     # Figure setup
#     fig_gif, axs = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={'projection': ccrs.PlateCarree()})
#     print("Figure created")

#     # Create initial plots and colorbars
#     im1 = axs[0].pcolormesh(data_inference.lon, data_inference.lat, 
#                             data_inference.isel(lead_time=0),
#                             transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, cmap='RdBu_r')
#     im2 = axs[1].pcolormesh(data_gt.lon, data_gt.lat, 
#                             data_gt.isel(lead_time=0),
#                             transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, cmap='RdBu_r')
    
#     # Add colorbars
#     plt.colorbar(im1, ax=axs[0], orientation='horizontal', pad=0.05)
#     plt.colorbar(im2, ax=axs[1], orientation='horizontal', pad=0.05)

#     axs[0].set_global()
#     axs[1].set_global()
#     axs[0].set_title(f'{name_fc}')
#     axs[1].set_title('Ground Truth')

#     frame_times = []

#     def plot(i):
#         frame_start = time.time()

#         lead_time = data_inference.lead_time.values[i]

#         forecast = data_inference.isel(lead_time=i)
#         truth = data_gt.isel(lead_time=i)

#         # Update plot data
#         im1.set_array(forecast.values.ravel())
#         im2.set_array(truth.values.ravel())

#         var_up = f'{var}_{plev/100:.0f}hPa' if plev is not None else var
#         title = f'{var_up} at lead time {lead_time} hours (Sample {sample_index})'
#         plt.suptitle(title, y=0.95)

#         frame_end = time.time()
#         frame_times.append(frame_end - frame_start)
#         print(f"Frame {i} completed in {frame_end - frame_start:.2f} seconds")

#     print("Starting animation creation")
#     ani = animation.FuncAnimation(fig_gif, plot, frames=len(data_inference.lead_time), repeat=False)
    
#     print("Saving animation")
#     ani.save(output_filename, writer='pillow', fps=1)
#     plt.close(fig_gif)
    
#     end_time = time.time()
#     total_time = end_time - start_time
#     print(f"GIF saved as {output_filename}")
#     print(f"Total time: {total_time:.2f} seconds")
#     print(f"Average frame time: {np.mean(frame_times):.2f} seconds")
#     print(f"Max frame time: {np.max(frame_times):.2f} seconds")

#     return ani
