import argparse
import torch
import os
import yaml
from einops import rearrange
from matplotlib import pyplot as plt
import numpy as np
from common.loss import _weight_for_latitude_vector_without_poles

def plot_loss(losses, filename, key=None):
    # losses in shape [t]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(losses)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Loss')
    ax.set_title(f'{key} loss over time')
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_reconstruction(y_pred, y, filename=None, cmap='twilight_shifted'):
    # y in shape [h w], y_pred in shape [h w]

    fig, axs = plt.subplots(2, 1, figsize=(6, 6))

    vmin = y.min()
    vmax = y.max()

    im0 = axs[0].imshow(y.numpy(), vmin=vmin, vmax=vmax,cmap=cmap, origin='lower')
    im1 = axs[1].imshow(y_pred.numpy(), vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')

    # set the title
    axs[0].set_title(f"True")
    axs[1].set_title(f"Pred")

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im0, cax=cbar_ax)
    # save the figure
    if filename is not None:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_result(y_pred, y, filename=None, num_t=5, cmap='twilight_shifted'):
    # y in shape [t h w], y_pred in shape [t h w]

    t_total, h, w = y_pred.shape

    dt = 0
    if num_t != 1:
        dt = t_total // num_t
        if dt == 0:
            num_t = t_total # since t_total < num_t
            dt = 1
        y_pred = y_pred[::dt]
        y = y[::dt]

    fig, axs = plt.subplots(2, num_t, figsize=(num_t*6, 6))

    vmin = y.min()
    vmax = y.max()

    for i in range(num_t):
        if num_t == 1:
            im0 = axs[0].imshow(y[i].numpy(), vmin=vmin, vmax=vmax,cmap=cmap, origin='lower')
            im1 = axs[1].imshow(y_pred[i].numpy(), vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')

            # set the title
            axs[0].set_title(f"True t={(i+1)*dt}")
            axs[1].set_title(f"Pred t={(i+1)*dt}")
        else:
            im0 = axs[0][i].imshow(y[i].numpy(), vmin=vmin, vmax=vmax,cmap=cmap, origin='lower')
            im1 = axs[1][i].imshow(y_pred[i].numpy(), vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')

            # set the title
            axs[0][i].set_title(f"True t={(i+1)*dt}")
            axs[1][i].set_title(f"Pred t={(i+1)*dt}")

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im0, cax=cbar_ax)
    # save the figure
    if filename is not None:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_bias(pred, target, save_path=None, title=""):
    # pred, target in shape nlat nlon
    bias = pred - target
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    vmin = target.min()
    vmax = target.max()
    
    bias_min = bias.min()
    bias_max = bias.max()

    bias_scale = max(abs(bias_min), abs(bias_max))

    im0 = axs[0].imshow(pred, cmap='twilight_shifted', vmin=vmin, vmax=vmax, origin='lower')
    axs[0].set_title('Predicted')
    fig.colorbar(im0, ax=axs[0], orientation='horizontal')

    im1 = axs[1].imshow(target, cmap='twilight_shifted', vmin=vmin, vmax=vmax, origin='lower')
    axs[1].set_title('Target')
    fig.colorbar(im1, ax=axs[1], orientation='horizontal')

    im2 = axs[2].imshow(bias, cmap='bwr', vmin=-bias_scale, vmax=bias_scale, origin='lower')
    axs[2].set_title('Bias (Predicted - Target)')
    fig.colorbar(im2, ax=axs[2], orientation='horizontal')

    fig.suptitle(title)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

def plot_crps(crps, title, t=120, save_path=None):
    plt.figure()
    plt.plot(crps)
    plt.xticks(np.arange(0, t+1, 24), np.arange(0, t//4+1, 6))
    plt.xlabel('Forecast lead time (days)')
    plt.ylabel('CRPS')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_ssr(ssr, title, t=120, save_path=None):
    plt.figure()
    plt.plot(ssr)
    plt.xticks(np.arange(0, t+1, 24), np.arange(0, t//4+1, 6))
    plt.xlabel('Forecast lead time (days)')
    plt.ylabel('SSR')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_spectrum(pred, target, path=None, num_t = 4):
    # pred and target in shape (t, nlat, nlon)
    if pred.shape[0] == 1: # assume t is trivial
        pred = pred.squeeze()
        target = target.squeeze()
        nlat = pred.shape[0]
        nlon = pred.shape[1]

        k_x_pred, power_spectrum_pred = zonal_averaged_power_spectrum(pred, nlon=nlon, nlat=nlat)
        k_x_target, power_spectrum_target = zonal_averaged_power_spectrum(target, nlon=nlon, nlat=nlat)

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(k_x_pred.numpy(), power_spectrum_pred.numpy(), label='Predicted', color='blue')
        ax.plot(k_x_target.numpy(), power_spectrum_target.numpy(), label='Target', color='orange')
        ax.set_xlabel('Zonal wavenumber')
        ax.set_ylabel('Power Spectrum')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Zonal Averaged Power Spectrum')
        ax.legend()
    else:
        t_total = pred.shape[0]

        dt = 0
        if num_t != 1:
            dt = t_total // num_t
            if dt == 0:
                num_t = t_total # since t_total < num_t
                dt = 1
            pred = pred[::dt]
            target = target[::dt]

        nlat = pred.shape[1]
        nlon = pred.shape[2]

        fig, axs = plt.subplots(1, num_t, figsize=(num_t*6, 6))

        for i in range(num_t):
            k_x_pred, power_spectrum_pred = zonal_averaged_power_spectrum(pred[i], nlon=nlon, nlat=nlat)
            k_x_target, power_spectrum_target = zonal_averaged_power_spectrum(target[i], nlon=nlon, nlat=nlat)

            axs[i].plot(k_x_pred.numpy(), power_spectrum_pred.numpy(), label='Predicted', color='blue')
            axs[i].plot(k_x_target.numpy(), power_spectrum_target.numpy(), label='Target', color='orange')
            axs[i].set_xlabel('Zonal wavenumber')
            axs[i].set_ylabel('Power Spectrum')
            axs[i].set_xscale('log')
            axs[i].set_yscale('log')
            axs[i].set_title(f'Zonal Averaged Power Spectrum at t={(i+1)*dt}')
            axs[i].legend()

    if path is not None:
        plt.savefig(path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_spectrum_by_latitude(pred, target, latitudes=None, filename=None):
    # pred and target in shape (nlat, nlon)
    nlat, nlon = pred.shape

    if latitudes is None:
        latitudes = [-85, -60, -30, 0, 30, 60, 85]

    # Reconstruct latitude grid (same as zonal_averaged_power_spectrum)
    lat_end = (nlat - 1) * (360 / nlon) / 2
    latitude_grid = np.linspace(-lat_end, lat_end, nlat)

    # Map requested latitudes to nearest grid indices
    lat_indices = []
    actual_lats = []
    for lat in latitudes:
        idx = np.argmin(np.abs(latitude_grid - lat))
        lat_indices.append(idx)
        actual_lats.append(latitude_grid[idx])

    # Wavenumber axis
    k_x = torch.fft.fftfreq(nlon, d=1/nlon)[:nlon//2]

    # Subplot layout
    n = len(latitudes)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axs = np.atleast_2d(axs)

    for i, (idx, actual_lat) in enumerate(zip(lat_indices, actual_lats)):
        row_idx = i // ncols
        col_idx = i % ncols
        ax = axs[row_idx, col_idx]

        # Per-row spectral computation
        fft_pred = torch.fft.rfft(pred[idx, :], norm='forward')
        fft_target = torch.fft.rfft(target[idx, :], norm='forward')

        power_pred = torch.abs(fft_pred) ** 2
        power_target = torch.abs(fft_target) ** 2

        power_pred = power_pred[:nlon//2]
        power_target = power_target[:nlon//2]
        power_pred[1:] *= 2
        power_target[1:] *= 2

        ax.plot(k_x.numpy(), power_pred.numpy(), label='Predicted', color='blue')
        ax.plot(k_x.numpy(), power_target.numpy(), label='Target', color='orange')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Zonal wavenumber')
        ax.set_ylabel('Power Spectrum')
        ax.set_title(f'Lat = {actual_lat:.1f}\u00b0')
        ax.legend()

    # Hide unused subplots
    for i in range(n, nrows * ncols):
        axs[i // ncols, i % ncols].set_visible(False)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def zonal_averaged_power_spectrum(field,
                                  nlon=360,
                                  nlat=180):
    """
    This function calculates the zonal averaged power spectrum of a given field. It is designed to work with xarray DataArrays or Datasets that have 'lat', 'lon', and optionally 'time' dimensions. The function first transposes the dimensions to ensure 'lat' and 'lon' are the first two dimensions, then performs a Fast Fourier Transform (FFT) along the 'lon' axis to compute the power spectrum. The power spectrum is then averaged over 'lat' and 'time' (if present) to produce the zonal averaged power spectrum.

    Parameters:
    - field, tensor of shape (nlat, nlon)
    """

    field = torch.transpose(field, 0, 1)  # nlon, nlat

    ###########################################################################################
    field_fft = torch.fft.rfft(field, dim=0, norm='forward') # Convention used: the first Fourier coefficient is the mean of the field

    # Compute the power spectrum (squared magnitude of Fourier coefficients)
    power_spectrum = torch.abs(field_fft)**2

    # Define the zonal wavenumbers
    nx = nlon
    k_x = torch.fft.fftfreq(nx, d=1/nx)

    # Only take the positive frequencies (or the first half if using real FFT)
    k_x = k_x[:nx//2]
    power_spectrum = power_spectrum[:nx//2]
    # count the positive frequencies twice except for the first one (zero frequency), because the FFT of a real function is symmetric
    power_spectrum[1:] *= 2
    # multiply by a factor cos(pi latitude[i] / 180) in axis 1
    lat_end = (nlat-1)*(360/nlon) / 2
    latitude = np.linspace(-lat_end, lat_end, nlat)
    weights = _weight_for_latitude_vector_without_poles(latitude)
    weights = torch.from_numpy(weights)
    weights = weights.view(1, -1)
    power_spectrum *= weights 

    power_spectrum_avg = power_spectrum.mean(axis=1)

    return k_x, power_spectrum_avg