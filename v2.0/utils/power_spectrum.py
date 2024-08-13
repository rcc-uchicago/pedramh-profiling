
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



def plot_power_spectrum_test(power_spectrum_avg_preds, power_spectrum_avg_gt, preds_times, filename, lead_times):
    """ Plot the power spectrum of the forecast and ground truth
    :param power_spectrum_avg_preds: xarray dataset, power spectrum of the forecast
    :param power_spectrum_avg_gt: xarray dataset, power spectrum of the ground truth
    :param preds_times: array, time values of the forecast
    :param filename: str, path to save the plot
    :param lead_times: list, lead times in hours to plot (default: [6, 18])
    """
    # Get available variables and sigma levels from the data
    available_vars = list(power_spectrum_avg_preds.data_vars)
    sigma_levels = power_spectrum_avg_preds.lev.values

    # Filter lead times that are present in the data
    available_lead_times = power_spectrum_avg_preds.lead_time.values
    plot_lead_times = [lt for lt in lead_times if lt in available_lead_times]
    if not plot_lead_times:
        raise ValueError(f"None of the specified lead times {lead_times} are present in the data. Available lead times: {available_lead_times}")

    # Select a subset of variables and sigma levels if there are too many
    if len(available_vars) > 3:
        available_vars = available_vars[:3]
    if len(sigma_levels) > 3:
        sigma_levels = [sigma_levels[0], sigma_levels[len(sigma_levels)//2], sigma_levels[-1]]

    # Create subplots
    fig, axs = plt.subplots(len(plot_lead_times), len(available_vars), figsize=(18, 20), squeeze=False)
    k_x_preds = power_spectrum_avg_preds.k_x.values
    k_x_gt = power_spectrum_avg_gt.k_x.values

    for i, lead_time in enumerate(plot_lead_times):
        for j, var in enumerate(available_vars):
            for sigma in sigma_levels:
                try:
                    power_spectrum_avg_preds2 = power_spectrum_avg_preds[var].sel(lead_time=lead_time, lev=sigma, method='nearest')
                    axs[i,j].plot(k_x_preds, power_spectrum_avg_preds2.values, label=f'Forecast')
                    
                    # Add ground truth plot
                    power_spectrum_avg_gt2 = power_spectrum_avg_gt[var].sel(lead_time=lead_time, lev=sigma, method='nearest')
                    axs[i,j].plot(k_x_gt, power_spectrum_avg_gt2.values, linestyle='--', label=f'Ground Truth')
                except KeyError as e:
                    print(f"Warning: Could not select data for var={var}, lead_time={lead_time}, sigma={sigma}. Error: {e}")
                    continue

            axs[i,j].set_yscale('log')
            axs[i,j].set_xscale('log')
            axs[i,j].set_xlabel(r'Zonal Wavenumber $k_x$')
            axs[i,j].set_ylabel('Energy Spectrum')
            axs[i,j].set_title(f"var = '{var}', lead time = {lead_time} hours")
            axs[i,j].grid(True)
            axs[i,j].legend()
    
    plt.suptitle(f"Latitude-averaged Instantaneous Fourier Spectrum (Sigma Levels)", y=1.01)
    plt.tight_layout() 
    plt.savefig(filename, pad_inches=0.1, bbox_inches='tight')
    plt.close(fig)

    return fig, axs

# def plot_power_spectrum_test(power_spectrum_avg_preds, preds_times, filename):
#     """ Plot the power spectrum of the forecast
#     :param power_spectrum_avg_preds: xarray dataset, power spectrum of the forecast
#     :param preds_times: array, time values of the forecast
#     """
#     # Ensure all data is on CPU
#     # Ensure all data is on CPU
#     # if isinstance(power_spectrum_avg_preds, xr.Dataset):
#     #     power_spectrum_avg_preds = power_spectrum_avg_preds.compute()
#     # elif torch.is_tensor(power_spectrum_avg_preds):
#     #     power_spectrum_avg_preds = power_spectrum_avg_preds.detach().cpu().numpy()

#     # Get available variables, lead times, and sigma levels from the data
#     available_vars = list(power_spectrum_avg_preds.data_vars)
#     lead_times = power_spectrum_avg_preds.lead_time.values
#     sigma_levels = power_spectrum_avg_preds.lev.values

#     # Select a subset of variables, lead times, and sigma levels if there are too many
#     if len(available_vars) > 3:
#         available_vars = available_vars[:3]
#     if len(lead_times) > 3:
#         lead_times = [lead_times[0], lead_times[len(lead_times)//2], lead_times[-1]]
#     if len(sigma_levels) > 3:
#         sigma_levels = [sigma_levels[0], sigma_levels[len(sigma_levels)//2], sigma_levels[-1]]

#     # Create subplots
#     fig, axs = plt.subplots(len(lead_times), len(available_vars), figsize=(18, 20), squeeze=False)
#     k_x_preds = power_spectrum_avg_preds.k_x.values

#     for i, lead_time in enumerate(lead_times):
#         for j, var in enumerate(available_vars):
#             for sigma in sigma_levels:
#                 try:
#                     power_spectrum_avg_preds2 = power_spectrum_avg_preds[var].sel(lead_time=lead_time, lev=sigma, method='nearest')
#                     axs[i,j].plot(k_x_preds, power_spectrum_avg_preds2.values, label=f'σ={sigma:.4f}')
#                 except KeyError as e:
#                     print(f"Warning: Could not select data for var={var}, lead_time={lead_time}, sigma={sigma}. Error: {e}")
#                     continue

#             axs[i,j].set_yscale('log')
#             axs[i,j].set_xscale('log')
#             axs[i,j].set_xlabel(r'Zonal Wavenumber $k_x$')
#             axs[i,j].set_ylabel('Energy Spectrum')
#             axs[i,j].set_title(f"var = '{var}', lead time = {lead_time:.2f}")
#             axs[i,j].grid(True)
#             axs[i,j].legend()
    
#     plt.suptitle(f"Latitude-averaged Instantaneous Fourier Spectrum (Sigma Levels)", y=1.01)
#     plt.tight_layout() 
#     plt.savefig(filename, pad_inches=0.1, bbox_inches='tight')
#     plt.close(fig)

#     return fig, axs

