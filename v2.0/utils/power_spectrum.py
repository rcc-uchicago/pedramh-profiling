
import sys
import xarray as xr
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import torch
import matplotlib

import os
import matplotlib.colors as mcolors

from dask.diagnostics import ProgressBar
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import seaborn as sns
from collections import OrderedDict
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


# Amaury's code
def plot_power_spectrum(power_spectrum_avg_preds, preds_times, vars = ["ta", "zg", "ua"], plevs = [850*100, 500*100, 250*100], lead_times=[6, 48, 120]):
    """ Plot the power spectrum of the ground truth and the forecast
    :param power_spectrum_avg: xarray dataset, power spectrum of the ground truth
    :param power_spectrum_avg_preds: xarray dataset, power spectrum of the forecast
    :param preds_times: array, time values of the forecast
    :param name_fc: str, name of the forecast
    """
    # Check that len(vars) == len(plevs)
    assert len(vars) == len(plevs), 'vars and plevs must have the same length'

    # Loop through variables and pressure levels to plot
    fig, axs = plt.subplots(len(lead_times), len(vars), figsize=(18, 20))
    # k_x_gt = power_spectrum_avg_gt.k_x.values
    k_x_preds = power_spectrum_avg_preds.k_x.values
    for i, lead_time in enumerate(lead_times):
        for j, (var, plev) in enumerate(zip(vars, plevs)):
            power_spectrum_avg_preds2 = power_spectrum_avg_preds[var].sel(lead_time=lead_time).sel(lev=plev)
            # power_spectrum_avg2 = power_spectrum_avg_gt[var].sel(time=preds_times + timedelta(hours=lead_time)).mean('time').sel(plev=plev)

            # axs[i,j].plot(k_x_gt, power_spectrum_avg2, label='Ground Truth')
            axs[i,j].plot(k_x_preds, power_spectrum_avg_preds2)
            axs[i,j].legend()
            axs[i,j].set_yscale('log')
            axs[i,j].set_xscale('log')
            axs[i,j].set_xlabel(r'Zonal Wavenumber $k_x$')
            axs[i,j].set_ylabel('Energy Spectrum')
            if i==0:
                axs[i,j].set_title(f"var = '{var}' at {int(plev/100)} hPa, lead time = {lead_time} hours")
            else:
                axs[i,j].set_title(f"var = '{var}' at {int(plev/100)} hPa, lead time = {lead_time//24} days")
            axs[i,j].grid(True)

    plt.suptitle(f"Latitude-averaged Instantaneous Fourier Spectrum", y = 1.01)
    plt.tight_layout() 
    plt.savefig(f"spectrum_results.png", pad_inches=0.1, bbox_inches='tight')
    return fig, axs

# Debugging Testing
def minimal_plot_function(power_spectrum_avg_preds, preds_times):
    print("Entering minimal plot function")
    # Access some data to simulate the data access in the real function
    _ = list(power_spectrum_avg_preds.data_vars)
    _ = power_spectrum_avg_preds.lead_time.values
    _ = power_spectrum_avg_preds.lev.values
    print("Exiting minimal plot function")



# def plot_power_spectrum_test(power_spectrum_avg_preds, power_spectrum_avg_gt, preds_times, filename, lead_times):
#     """ Plot the power spectrum of the forecast and ground truth
#     :param power_spectrum_avg_preds: xarray dataset, power spectrum of the forecast
#     :param power_spectrum_avg_gt: xarray dataset, power spectrum of the ground truth
#     :param preds_times: array, time values of the forecast
#     :param filename: str, path to save the plot
#     :param lead_times: list, lead times in hours to plot (default: [6, 18])
#     """
#     # Get available variables and sigma levels from the data
#     available_vars = list(power_spectrum_avg_preds.data_vars)
#     sigma_levels = power_spectrum_avg_preds.plev.values

#     # Filter lead times that are present in the data
#     available_lead_times = power_spectrum_avg_preds.lead_time.values
#     plot_lead_times = [lt for lt in lead_times if lt in available_lead_times]
#     if not plot_lead_times:
#         raise ValueError(f"None of the specified lead times {lead_times} are present in the data. Available lead times: {available_lead_times}")

#     # Select a subset of variables and sigma levels if there are too many
#     if len(available_vars) > 3:
#         available_vars = available_vars[:3]
#     if len(sigma_levels) > 3:
#         sigma_levels = [sigma_levels[0], sigma_levels[len(sigma_levels)//2], sigma_levels[-1]]

#     # Create subplots
#     fig, axs = plt.subplots(len(plot_lead_times), len(available_vars), figsize=(18, 20), squeeze=False)
#     k_x_preds = power_spectrum_avg_preds.k_x.values
#     k_x_gt = power_spectrum_avg_gt.k_x.values

#     for i, lead_time in enumerate(plot_lead_times):
#         for j, var in enumerate(available_vars):
#             for sigma in sigma_levels:
#                 try:
#                     power_spectrum_avg_preds2 = power_spectrum_avg_preds[var].sel(lead_time=lead_time, lev=sigma, method='nearest')
#                     axs[i,j].plot(k_x_preds, power_spectrum_avg_preds2.values, label=f'Forecast')
                    
#                     # Add ground truth plot
#                     power_spectrum_avg_gt2 = power_spectrum_avg_gt[var].sel(lead_time=lead_time, lev=sigma, method='nearest')
#                     axs[i,j].plot(k_x_gt, power_spectrum_avg_gt2.values, linestyle='--', label=f'Ground Truth')
#                 except KeyError as e:
#                     print(f"Warning: Could not select data for var={var}, lead_time={lead_time}, sigma={sigma}. Error: {e}")
#                     continue

#             axs[i,j].set_yscale('log')
#             axs[i,j].set_xscale('log')
#             axs[i,j].set_xlabel(r'Zonal Wavenumber $k_x$')
#             axs[i,j].set_ylabel('Energy Spectrum')
#             axs[i,j].set_title(f"var = '{var}', lead time = {lead_time} hours")
#             axs[i,j].grid(True)
#             axs[i,j].legend()
    
#     plt.suptitle(f"Latitude-averaged Instantaneous Fourier Spectrum (Sigma Levels)", y=1.01)
#     plt.tight_layout() 
#     plt.savefig(filename, pad_inches=0.1, bbox_inches='tight')
#     plt.close(fig)

#     return fig, axs


# def plot_power_spectrum_test(power_spectrum_avg_preds, power_spectrum_avg_gt, preds_times, filename, lead_times):
#     """ Plot the power spectrum of the forecast and ground truth
#     :param power_spectrum_avg_preds: xarray dataset, power spectrum of the forecast
#     :param power_spectrum_avg_gt: xarray dataset, power spectrum of the ground truth
#     :param preds_times: array, time values of the forecast
#     :param filename: str, path to save the plot
#     :param lead_times: list, lead times in hours to plot (default: [6, 18])
#     """
#     # Get available variables and pressure levels from the data
#     available_vars = list(power_spectrum_avg_preds.data_vars)
#     pressure_levels = power_spectrum_avg_preds.plev.values

#     # Filter lead times that are present in the data
#     available_lead_times = power_spectrum_avg_preds.lead_time.values
#     plot_lead_times = [lt for lt in lead_times if lt in available_lead_times]
#     if not plot_lead_times:
#         raise ValueError(f"None of the specified lead times {lead_times} are present in the data. Available lead times: {available_lead_times}")

#     # Select a subset of variables and pressure levels if there are too many
#     if len(available_vars) > 3:
#         available_vars = available_vars[:3]
#     if len(pressure_levels) > 3:
#         pressure_levels = [pressure_levels[0], pressure_levels[len(pressure_levels)//2], pressure_levels[-1]]

#     # Create subplots
#     fig, axs = plt.subplots(len(plot_lead_times), len(available_vars), figsize=(18, 20), squeeze=False)
#     k_x_preds = power_spectrum_avg_preds.k_x.values
#     k_x_gt = power_spectrum_avg_gt.k_x.values

#     for i, lead_time in enumerate(plot_lead_times):
#         for j, var in enumerate(available_vars):
#             for plev in pressure_levels:
#                 try:
#                     power_spectrum_avg_preds2 = power_spectrum_avg_preds[var].sel(lead_time=lead_time, plev=plev, method='nearest')
#                     axs[i,j].plot(k_x_preds, power_spectrum_avg_preds2.values, label=f'Forecast')
                    
#                     # Add ground truth plot
#                     power_spectrum_avg_gt2 = power_spectrum_avg_gt[var].sel(lead_time=lead_time, plev=plev, method='nearest')
#                     axs[i,j].plot(k_x_gt, power_spectrum_avg_gt2.values, linestyle='--', label=f'Ground Truth')
#                 except KeyError as e:
#                     print(f"Warning: Could not select data for var={var}, lead_time={lead_time}, pressure level={plev}. Error: {e}")
#                     continue

#             axs[i,j].set_yscale('log')
#             axs[i,j].set_xscale('log')
#             axs[i,j].set_xlabel(r'Zonal Wavenumber $k_x$')
#             axs[i,j].set_ylabel('Energy Spectrum')
#             axs[i,j].set_title(f"var = '{var}', lead time = {lead_time} hours")
#             axs[i,j].grid(True)
#             axs[i,j].legend()
    
#     plt.suptitle(f"Latitude-averaged Instantaneous Fourier Spectrum (Pressure Levels)", y=1.01)
#     plt.tight_layout() 
#     plt.savefig(filename, pad_inches=0.1, bbox_inches='tight')
#     plt.close(fig)

#     return fig, axs

def plot_power_spectrum_test(power_spectrum_avg_preds, power_spectrum_avg_gt, preds_times, filename, lead_times,
                             vars=["ta", "zg", "ua"], 
                             plevs=[850*100, 500*100, 250*100]):
    """ Plot the power spectrum of the forecast and ground truth
    :param power_spectrum_avg_preds: xarray dataset, power spectrum of the forecast
    :param power_spectrum_avg_gt: xarray dataset, power spectrum of the ground truth
    :param preds_times: array, time values of the forecast
    :param filename: str, path to save the plot
    :param lead_times: list, lead times in hours to plot
    :param vars: list, variables to plot (default: ["ta", "zg", "ua"])
    :param plevs: list, pressure levels in Pa to plot (default: [850*100, 500*100, 250*100])
    """
    # Filter variables that are present in the data
    available_vars = [var for var in vars if var in power_spectrum_avg_preds.data_vars]
    if not available_vars:
        raise ValueError(f"None of the specified variables {vars} are present in the data. Available variables: {list(power_spectrum_avg_preds.data_vars)}")

    # Filter pressure levels that are present in the data
    available_plevs = [plev for plev in plevs if plev in power_spectrum_avg_preds.plev.values]
    if not available_plevs:
        raise ValueError(f"None of the specified pressure levels {plevs} are present in the data. Available levels: {power_spectrum_avg_preds.plev.values}")

    # Handle lead times as specified by the user
    available_lead_times = power_spectrum_avg_preds.lead_time.values
    print(f" Available lead times: {available_lead_times}")
    plot_lead_times = [lt for lt in lead_times if lt in available_lead_times]
    if not plot_lead_times:
        raise ValueError(f"None of the specified lead times {lead_times} are present in the data. Available lead times: {available_lead_times}")

    # Create subplots
    fig, axs = plt.subplots(len(plot_lead_times), len(available_vars), figsize=(18, 20), squeeze=False)
    k_x_preds = power_spectrum_avg_preds.k_x.values
    k_x_gt = power_spectrum_avg_gt.k_x.values

    for i, lead_time in enumerate(plot_lead_times):
        for j, var in enumerate(available_vars):
            for plev in available_plevs:
                try:
                    power_spectrum_avg_preds2 = power_spectrum_avg_preds[var].sel(lead_time=lead_time, plev=plev, method='nearest')
                    axs[i,j].plot(k_x_preds, power_spectrum_avg_preds2.values, label=f'Forecast')
                    
                    # Add ground truth plot
                    power_spectrum_avg_gt2 = power_spectrum_avg_gt[var].sel(lead_time=lead_time, plev=plev, method='nearest')
                    axs[i,j].plot(k_x_gt, power_spectrum_avg_gt2.values, linestyle='--', label=f'Ground Truth')
                except KeyError as e:
                    print(f"Warning: Could not select data for var={var}, lead_time={lead_time}, pressure level={plev}. Error: {e}")
                    continue

            axs[i,j].set_yscale('log')
            axs[i,j].set_xscale('log')
            axs[i,j].set_xlabel(r'Zonal Wavenumber $k_x$')
            axs[i,j].set_ylabel('Energy Spectrum')
            axs[i,j].set_title(f"var = '{var}', lead time = {lead_time} hours")
            axs[i,j].grid(True)
            axs[i,j].legend()
    
    plt.suptitle(f"Latitude-averaged Instantaneous Fourier Spectrum (Pressure Levels)", y=1.01)
    plt.tight_layout() 
    plt.savefig(filename, pad_inches=0.1, bbox_inches='tight')
    plt.close(fig)

    return fig, axs


def plot_acc_over_lead_time(acc, lead_times_hours, vars=["tas", "ta", "zg", "ua"], plevs=[None, 850*100, 500*100, 250*100], 
                            colors=None, fontsize_title=14):
    """
    Plot the ACC over lead time for each variable and pressure level
    :param acc: OrderedDict or xr.Dataset, ACC scores
    :param lead_times_hours: list, lead times in hours
    :param vars: list, variables to plot
    :param plevs: list, pressure levels to plot (use None for surface variables)
    :param colors: dict, colors for each model
    :param fontsize_title: int, font size for the title
    """
    if isinstance(acc, xr.Dataset):
        acc = {'Model': acc}

    if colors is None:
        colors = {'Pangu': 'blue'}

    fig, axs = plt.subplots(len(vars), 1, figsize=(12, 5*len(vars)), squeeze=False)

    for i, (var, plev) in enumerate(zip(vars, plevs)):
        ax = axs[i, 0]
        
        if plev is None:
            title = f'ACC for {var}'
        else:
            title = f'ACC for {var} at {plev/100:.0f} hPa'
        
        for model, ds in acc.items():
            if var in ds:
                if 'plev' in ds[var].dims and plev is not None:
                    data = ds[var].sel(plev=plev, method='nearest')
                elif 'plev' not in ds[var].dims:
                    data = ds[var]
                else:
                    print(f"Warning: {var} has unexpected dimensions. Skipping.")
                    continue
                
                ax.plot(lead_times_hours, data.values, label=model, color=colors[model], marker='o')

        ax.set_ylabel(f'{var} ACC')
        ax.set_title(title, fontsize=fontsize_title)
        ax.set_ylim(-0.3, 1.1)
        ax.axhline(0, ls='--', c='0.', lw=1)
        ax.set_xlabel('Lead time [hours]')
        ax.set_xticks(lead_times_hours)
        ax.set_xticklabels([f'{h}h\n(Step {i+1})' for i, h in enumerate(lead_times_hours)])
        ax.legend(loc='lower left')

    plt.tight_layout()
    return fig, axs

# Example usage:
# fig, axs = plot_acc_over_lead_time(acc)
# plt.show()


# def plot_acc(acc, colors=None, vars = ["tas", "ta", "zg", "ua"], plevs = [None, 850*100, 500*100, 250*100], 
#              units = ['K', 'K', 'm', 'm/s'], fontsize_title=14):
#     """
#     Plot the ACC for each variable and pressure level
#     :param acc: dict, ACC scores
#     :param colors: dict, colors for each model
#     """
#     # if score is not a OrderedDict, but a xr.Dataset with 'model' names as variables, convert it to an OrderedDict
#     if isinstance(acc, xr.Dataset):
#         print('Converting xr.Dataset to OrderedDict for ACC')
#         acc = OrderedDict({key: acc.sel(model=key) for key in acc.model.values})

#     # Check that len(vars) == len(plevs) == len(units)
#     assert len(vars) == len(plevs) == len(units), 'vars, plevs, and units must have the same length'

#     fig, axs = plt.subplots(len(plevs), 1, figsize=(12, 20))

#     for i, (var, plev) in enumerate(zip(vars, plevs)):
#         if plev is None:
#             title = f'{var}'
#         else:
#             title = f'{var} at {plev/100:.0f} hPa'
        
#         create_plot(acc, var, lev=plev, ax=axs[i], ylabel=f'{var} ACC', title=title, colors=colors)
#         axs[i].axhline(0, ls='--', c='0.', lw=2)
#         axs[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
#         axs[i].set_ylim(-0.3, 1.1)
#         axs[i].title.set_fontsize(fontsize_title)

#     for i in range(len(plevs)):
#         axs[i].legend(loc='upper left')

#     plt.tight_layout()
#     return fig, axs

# def create_plot(score, var, lev=None, colors=None, save_fn=None, ax=None, legend=False, ylabel=None, title=None, ylim=None, mult_tp=1.):
#     """ Create a plot of a particular score
#     :param score: dict, scores
#     :param var: str, variable to plot
#     :param lev: int, pressure level to plot
#     :param colors: dict, colors for each model
#     :param save_fn: str, path to save the plot
#     :param ax: axis object
#     :param legend: bool, whether to plot the legend
#     """

#     # if score is not a OrderedDict, but a xr.Dataset with 'model' names as variables, convert it to an OrderedDict
#     if isinstance(score, xr.Dataset):
#         print('Converting xr.Dataset to OrderedDict')
#         score = OrderedDict({key: score.sel(model=key) for key in score.model.values})
#         # print(score.keys())

#     if colors is None: 
#         colors = standard_color_dict = list(mcolors.CSS4_COLORS.values())[43:43+len(score.keys())]
#         colors = {exp: colors[i] for i, exp in enumerate(score.keys())}

#     if ax is None: 
#         fig, ax = plt.subplots(1, 1, figsize=(5, 4)) 
#     for exp, ds in score.items():
#         s = ds.copy(deep=True)
#         # # convert s to dataset
#         # if isinstance(s, xr.DataArray):
#         if var in s.variables:
#             if var == 'tp': s[var] *= mult_tp
#             if exp in ['Climatology', 'Weekly clim.']:
#                 ax.axhline(s[var], ls='--', c=colors[exp], label=exp, lw=3)
#             elif 'direct' in exp:
#                 ax.scatter(s['lead_time'], s[var], c=colors[exp], s=100, label=exp, lw=2, edgecolors='k', zorder=10)
#             else:
#                 if 'plev' in s[var].dims and lev is not None:
#                     s[var].sel(plev=lev, method='nearest').plot(c=colors[exp], label=exp, lw=3, ax=ax)
#                 elif 'plev' in s[var].dims and lev is None:
#                     print('lev is None. Please provide a level')
#                 else:
#                     s[var].plot(c=colors[exp], label=exp, lw=3, ax=ax)
            
#     ax.set_ylabel(ylabel)
#     ax.set_title(title)
#     ax.set_ylim(ylim)
#     # ax.set_xlim(0, 122)
#     # ax.set_xticks([0, 24, 48, 72, 96, 120])
#     # ax.set_xticklabels([0, 1, 2, 3, 4, 5])
#     # Calculating the number of days dynamically

#     num_days = s.lead_time[-1].values // 24 + 1

#     # Setting x-ticks dynamically based on the data range
#     days = np.arange(0, num_days)
#     hours_ticks = days * 24

#     ax.set_xticks(hours_ticks)
#     ax.set_xticklabels(days)


#     ax.set_xlabel('Lead time [days]')
    
#     if not save_fn is None: 
#         plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)
#         fig.savefig(save_fn)