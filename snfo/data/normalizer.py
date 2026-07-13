import torch
import numpy as np
from einops import rearrange
import pickle 

class Normalizer:
    def __init__(self, stat_path,
                 downsample_levels=1,
                 clouds = True):
        
        with open(stat_path, 'rb') as file:
            self.stat_dict = pickle.load(file)

        # load stats
        self.surface_means = torch.tensor(self.stat_dict['surface_mean'], dtype=torch.float32)  # shape (surface_channels,)
        self.surface_stds = torch.tensor(self.stat_dict['surface_std'], dtype=torch.float32)    # shape (surface_channels,)
        
        self.multilevel_means = torch.tensor(self.stat_dict['multi_mean'], dtype=torch.float32)  # shape (nlevels, multi_level_channels)
        self.multilevel_stds = torch.tensor(self.stat_dict['multi_std'], dtype=torch.float32)    # shape (nlevels, multi_level_channels)

        if downsample_levels > 1:
            self.multilevel_means = self.multilevel_means[::downsample_levels]
            self.multilevel_stds = self.multilevel_stds[::downsample_levels]

        # Some multilevel variables have zero mean/std, since they are constantly zero (upper atmosphere cloud cover) 
        eps = 1e-7
        zero_std_mask = (self.multilevel_stds < eps)
        zero_mean_mask = (self.multilevel_means.abs() < eps)
        self.multilevel_stds[zero_std_mask] = 1.0
        self.multilevel_means[zero_mean_mask] = 0.0

        if not clouds:
            # cloud variables are the last 5 channels in the multilevel data
            self.multi_idx = torch.tensor([0, 1, 2, 3, 4])
            self.multilevel_means = self.multilevel_means[..., self.multi_idx]
            self.multilevel_stds = self.multilevel_stds[..., self.multi_idx]

        self.forcing_means = torch.tensor(self.stat_dict['forcing_mean'], dtype=torch.float32)  # shape (forcing_channels,)
        self.forcing_stds = torch.tensor(self.stat_dict['forcing_std'], dtype=torch.float32)    # shape (forcing_channels,)
        # SST and SIC have nans in the raw data
        self.forcing_nans = [1, 2]

        self.invariant_means = torch.tensor(self.stat_dict['invariant_mean'], dtype=torch.float32)  # shape (invariant_channels,)
        self.invariant_stds = torch.tensor(self.stat_dict['invariant_std'], dtype=torch.float32)    # shape (invariant_channels,)

        self.diagnostic_means = torch.tensor(self.stat_dict['diag_mean'], dtype=torch.float32)  # shape (diagnostic_channels,)
        self.diagnostic_stds = torch.tensor(self.stat_dict['diag_std'], dtype=torch.float32)    # shape (diagnostic_channels,)

        # reshape stats
        self.surface_means = rearrange(self.surface_means, 'c -> 1 1 1 c') # nt nlat nlon c
        self.surface_stds = rearrange(self.surface_stds, 'c -> 1 1 1 c') 

        self.multilevel_means = rearrange(self.multilevel_means, 'n c -> 1 n 1 1 c')
        self.multilevel_stds = rearrange(self.multilevel_stds, 'n c -> 1 n 1 1 c')

        self.forcing_means = rearrange(self.forcing_means, 'c -> 1 1 1 c')
        self.forcing_stds = rearrange(self.forcing_stds, 'c -> 1 1 1 c')

        self.invariant_means = rearrange(self.invariant_means, 'c -> 1 1 c')
        self.invariant_stds = rearrange(self.invariant_stds, 'c -> 1 1 c')

        self.diagnostic_means = rearrange(self.diagnostic_means, 'c -> 1 1 1 c')
        self.diagnostic_stds = rearrange(self.diagnostic_stds, 'c -> 1 1 1 c')

    def normalize_surface(self, x):
        # x in shape (nt, nlat, nlon, surface_channels) or (b, nt, nlat, nlon, surface_channels)
        if len(x.shape) == 5:
            x = (x - self.surface_means.unsqueeze(0).to(x.device)) / self.surface_stds.unsqueeze(0).to(x.device)
        else:
            x = (x - self.surface_means.to(x.device)) / self.surface_stds.to(x.device)
        return x
    
    def normalize_multilevel(self, x):
        # x in shape (nt, nlevels, nlat, nlon, multi_level_channels) or (b, nt, nlevels, nlat, nlon, multi_level_channels)
        if len(x.shape) == 6:
            x = (x - self.multilevel_means.unsqueeze(0).to(x.device)) / self.multilevel_stds.unsqueeze(0).to(x.device)
        else:
            x = (x - self.multilevel_means.to(x.device)) / self.multilevel_stds.to(x.device)
        return x
    
    def normalize_forcing(self, x):
        # needs logic to handle NaNs in forcing data (SST, SIC)
        for nan_idx in self.forcing_nans:
            # replace nan w/ mean of the feature
            x[..., nan_idx] = torch.nan_to_num(x[..., nan_idx], nan=self.forcing_means[..., nan_idx].item())

        # x in shape (nt, nlat, nlon, forcing_channels) or (b, nt, nlat, nlon, forcing_channels)
        if len(x.shape) == 5:
            x = (x - self.forcing_means.unsqueeze(0).to(x.device)) / self.forcing_stds.unsqueeze(0).to(x.device)
        else:
            x = (x - self.forcing_means.to(x.device)) / self.forcing_stds.to(x.device)
        return x
    
    def normalize_invariant(self, x):
        # x in shape (nlat, nlon, invariant_channels) or (b, nlat, nlon, invariant_channels)
        if len(x.shape) == 4:
            x = (x - self.invariant_means.unsqueeze(0).to(x.device)) / self.invariant_stds.unsqueeze(0).to(x.device)
        else:
            x = (x - self.invariant_means.to(x.device)) / self.invariant_stds.to(x.device)
        return x
    
    def normalize_diagnostic(self, x):
        # x in shape (nt, nlat, nlon, diagnostic_channels) or (b, nt, nlat, nlon, diagnostic_channels)
        if len(x.shape) == 5:
            x = (x - self.diagnostic_means.unsqueeze(0).to(x.device)) / self.diagnostic_stds.unsqueeze(0).to(x.device)
        else:
            x = (x - self.diagnostic_means.to(x.device)) / self.diagnostic_stds.to(x.device)
        return x
    
    def denormalize_surface(self, x):
        # x in shape (nt, nlat, nlon, surface_channels) or (b, nt, nlat, nlon, surface_channels)
        if len(x.shape) == 5:
            x = x * self.surface_stds.unsqueeze(0).to(x.device) + self.surface_means.unsqueeze(0).to(x.device)
        else:
            x = x * self.surface_stds.to(x.device) + self.surface_means.to(x.device)
        return x
    
    def denormalize_multilevel(self, x):
        # x in shape (nt, nlat, nlon, nlevels, multi_level_channels) or (b, nt, nlat, nlon, nlevels, multi_level_channels)
        if len(x.shape) == 6:
            x = x * self.multilevel_stds.unsqueeze(0).to(x.device) + self.multilevel_means.unsqueeze(0).to(x.device)
        else:
            x = x * self.multilevel_stds.to(x.device) + self.multilevel_means.to(x.device)
        return x

    def denormalize_forcing(self, x):
        # x in shape (nt, nlat, nlon, forcing_channels) or (b, nt, nlat, nlon, forcing_channels)
        if len(x.shape) == 5:
            x = x * self.forcing_stds.unsqueeze(0).to(x.device) + self.forcing_means.unsqueeze(0).to(x.device)
        else:
            x = x * self.forcing_stds.to(x.device) + self.forcing_means.to(x.device)
        return x

    def denormalize_invariant(self, x):
        # x in shape (nlat, nlon, invariant_channels) or (b, nlat, nlon, invariant_channels)
        if len(x.shape) == 4:
            x = x * self.invariant_stds.unsqueeze(0).to(x.device) + self.invariant_means.unsqueeze(0).to(x.device)
        else:
            x = x * self.invariant_stds.to(x.device) + self.invariant_means.to(x.device)
        return x
    
    def denormalize_diagnostic(self, x):
        # x in shape (nt, nlat, nlon, diagnostic_channels) or (b, nt, nlat, nlon, diagnostic_channels)
        if len(x.shape) == 5:
            x = x * self.diagnostic_stds.unsqueeze(0).to(x.device) + self.diagnostic_means.unsqueeze(0).to(x.device)
        else:
            x = x * self.diagnostic_stds.to(x.device) + self.diagnostic_means.to(x.device)
        return x
