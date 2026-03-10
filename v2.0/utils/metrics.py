"""
Unified metrics computation for weather forecasting validation.
Computes latitude-weighted RMSE and ACC in PyTorch (GPU-accelerated).

This replaces the xarray-based ACC computation with an equivalent PyTorch
implementation that avoids GPU→CPU→xarray conversion overhead.

Author: Code cleanup collaboration
Date: 2024
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple
import numpy as np
import xarray as xr


class MetricsAggregator:
    """
    Accumulates predictions and targets during validation,
    then computes RMSE and ACC metrics.
    
    Handles:
    - Latitude-weighted RMSE (accumulates raw sums, divides at end)
    - Latitude-weighted ACC (standard meteorological formula)
    - DDP all-reduce synchronization
    - Formatting for wandb logging
    
    RMSE Formula:
        w_rmse = N_lat * cos(lat) / sum(cos(lat))
        RMSE = sqrt( sum(w * (pred - target)²) / count ) * std
        
    ACC Formula (Pearson correlation of anomalies):
        w_acc = cos(lat) / mean(cos(lat))
        fa = pred - climatology[day_of_year]
        a = target - climatology[day_of_year]
        fa' = fa - mean(fa)   # mean over entire validation set
        a' = a - mean(a)
        ACC = sum(w * fa' * a') / sqrt(sum(w * fa'²) * sum(w * a'²))
    """
    
    def __init__(
        self,
        surface_variables: List[str],
        upper_air_variables: List[str],
        diagnostic_variables: List[str],
        levels: np.ndarray,
        lead_time_steps: List[int],
        latitudes: torch.Tensor,
        climatology: xr.Dataset,  # xarray Dataset with 'dayofyear' dimension
        surface_std: torch.Tensor,
        upper_air_std: torch.Tensor,
        diagnostic_std: Optional[torch.Tensor],
        device: torch.device,
        use_sigma_levels: bool = False,
        sigma_levels: Optional[np.ndarray] = None,
    ):
        """
        Args:
            surface_variables: List of surface variable names (e.g., ['tas', 'psl'])
            upper_air_variables: List of upper air variable names (e.g., ['ta', 'ua', 'zg'])
            diagnostic_variables: List of diagnostic variable names
            levels: Array of pressure/sigma levels
            lead_time_steps: List of lead time steps to track (e.g., [1, 2, 4, 8])
            latitudes: 1D tensor of latitude values in degrees
            climatology: xarray Dataset with 'dayofyear' dim (366 days) containing all variables
            surface_std: Tensor of shape [num_surface_vars] for denormalization
            upper_air_std: Tensor of shape [num_upper_vars, num_levels] for denormalization
            diagnostic_std: Tensor of shape [num_diagnostic_vars] or None
            device: torch device
            use_sigma_levels: Whether to use sigma levels for non-zg variables
            sigma_levels: Array of sigma level values (used for logging labels)
        """
        self.surface_variables = surface_variables
        self.upper_air_variables = upper_air_variables
        self.diagnostic_variables = diagnostic_variables
        self.levels = levels
        self.lead_time_steps = lead_time_steps
        self.device = device
        self.use_sigma_levels = use_sigma_levels
        self.sigma_levels = sigma_levels if sigma_levels is not None else levels
        
        # Store stds for denormalization
        self.surface_std = surface_std.to(device)
        self.upper_air_std = upper_air_std.to(device)
        self.diagnostic_std = diagnostic_std.to(device) if diagnostic_std is not None else None
        
        # Precompute latitude weights
        lat_rad = latitudes * (np.pi / 180.0)
        cos_lat = torch.cos(lat_rad).to(device)
        
        # RMSE weights: w = N_lat * cos(lat) / sum(cos(lat)) - use float32
        self.rmse_weights = (len(latitudes) * cos_lat / cos_lat.sum())
        
        # ACC weights: w = cos(lat) / mean(cos(lat)) - use float64 for precision
        self.acc_weights = (cos_lat / cos_lat.mean()).double()
        
        # Reshape for broadcasting
        # Surface/diagnostic: [batch, vars, lat, lon] -> weight shape [1, 1, lat, 1]
        # Upper air: [batch, vars, levels, lat, lon] -> weight shape [1, 1, 1, lat, 1]
        self.rmse_weights_2d = self.rmse_weights.view(1, 1, -1, 1)
        self.rmse_weights_3d = self.rmse_weights.view(1, 1, 1, -1, 1)
        self.acc_weights_2d = self.acc_weights.view(1, 1, -1, 1)
        self.acc_weights_3d = self.acc_weights.view(1, 1, 1, -1, 1)
        
        # Convert climatology to torch tensors
        # Climatology is xarray Dataset with dims: (dayofyear, [lev/plev], lat, lon)
        self._prepare_climatology(climatology)
        
        self.has_diagnostic = len(diagnostic_variables) > 0
        self.num_steps = len(lead_time_steps)
        self.num_sfc = len(surface_variables)
        self.num_ua = len(upper_air_variables)
        self.num_levels = len(levels)
        self.num_diag = len(diagnostic_variables)
        
        # Initialize accumulators
        self.reset()
    
    def _prepare_climatology(self, climatology: xr.Dataset):
        """
        Convert xarray climatology to torch tensors.
        
        Climatology structure:
        - Dimension 'dayofyear' with 366 values
        - Surface variables: (dayofyear, lat, lon)
        - Upper air variables: (dayofyear, lev/plev, lat, lon)
        """
        # Surface climatology: [366, num_vars, lat, lon]
        clim_surface_list = []
        for var in self.surface_variables:
            if var in climatology:
                data = climatology[var].values  # (dayofyear, lat, lon)
                clim_surface_list.append(torch.from_numpy(data).float())
            else:
                # If variable not in climatology, use zeros (e.g., for precipitation)
                shape = (366, climatology.dims['lat'], climatology.dims['lon'])
                clim_surface_list.append(torch.zeros(shape, dtype=torch.float32))
        
        if clim_surface_list:
            # Stack to [366, num_vars, lat, lon]
            self.clim_surface = torch.stack(clim_surface_list, dim=1).to(self.device)
        else:
            self.clim_surface = None
        
        # Upper air climatology: [366, num_vars, num_levels, lat, lon]
        clim_upper_air_list = []
        # Determine level coordinate name
        level_coord = 'plev' if 'plev' in climatology.dims else 'lev'
        
        for var in self.upper_air_variables:
            if var in climatology:
                data = climatology[var].values  # (dayofyear, lev, lat, lon)
                # Select only the levels we need
                if level_coord in climatology[var].dims:
                    clim_levels = climatology[level_coord].values
                    level_indices = [np.argmin(np.abs(clim_levels - lev)) for lev in self.levels]
                    data = data[:, level_indices, :, :]
                clim_upper_air_list.append(torch.from_numpy(data).float())
            else:
                shape = (366, len(self.levels), climatology.dims['lat'], climatology.dims['lon'])
                clim_upper_air_list.append(torch.zeros(shape, dtype=torch.float32))
        
        if clim_upper_air_list:
            # Stack to [366, num_vars, num_levels, lat, lon]
            self.clim_upper_air = torch.stack(clim_upper_air_list, dim=1).to(self.device)
        else:
            self.clim_upper_air = None
        
        # Diagnostic climatology: [366, num_vars, lat, lon]
        if self.diagnostic_variables:
            clim_diag_list = []
            for var in self.diagnostic_variables:
                if var in climatology:
                    data = climatology[var].values
                    clim_diag_list.append(torch.from_numpy(data).float())
                else:
                    shape = (366, climatology.dims['lat'], climatology.dims['lon'])
                    clim_diag_list.append(torch.zeros(shape, dtype=torch.float32))
            self.clim_diagnostic = torch.stack(clim_diag_list, dim=1).to(self.device)
        else:
            self.clim_diagnostic = None
    
    def reset(self):
        """Reset all accumulators for a new validation epoch."""
        # ===== RMSE accumulators =====
        # Accumulate raw weighted squared errors (sum, not mean)
        self.rmse_sfc_sum = torch.zeros(self.num_steps, self.num_sfc, device=self.device)
        self.rmse_ua_sum = torch.zeros(self.num_steps, self.num_ua, self.num_levels, device=self.device)
        if self.has_diagnostic:
            self.rmse_diag_sum = torch.zeros(self.num_steps, self.num_diag, device=self.device)
        
        # Count total samples (grid points) for proper averaging
        self.rmse_sfc_count = torch.zeros(self.num_steps, device=self.device)
        self.rmse_ua_count = torch.zeros(self.num_steps, device=self.device)
        if self.has_diagnostic:
            self.rmse_diag_count = torch.zeros(self.num_steps, device=self.device)
        
        # ===== ACC accumulators =====
        # For Pearson correlation, we need to accumulate:
        # sum(w), sum(w*fa), sum(w*a), sum(w*fa*a), sum(w*fa²), sum(w*a²)
        # Then: mean_fa = sum(w*fa)/sum(w), etc.
        # cov(fa,a) = E[fa*a] - E[fa]*E[a]
        # var(fa) = E[fa²] - E[fa]²
        # ACC = cov / sqrt(var_fa * var_a)
        
        self.acc_sfc = self._init_acc_accumulators(self.num_sfc)
        self.acc_ua = self._init_acc_accumulators((self.num_ua, self.num_levels))
        if self.has_diagnostic:
            self.acc_diag = self._init_acc_accumulators(self.num_diag)
    
    def _init_acc_accumulators(self, shape) -> Dict[str, torch.Tensor]:
        """Initialize accumulator tensors for ACC computation.

        Uses float64 throughout for numerical stability.  Raw (unshifted) weighted
        sums are stored so that DDP all_reduce is exact — the previous shifted
        approach required each rank to use the *same* shift constant, which was not
        guaranteed after averaging shifted values across ranks.
        """
        if isinstance(shape, int):
            full_shape = (self.num_steps, shape)
        else:
            full_shape = (self.num_steps,) + tuple(shape)

        return {
            # Weighted sums for correlation
            'w_sum':   torch.zeros(full_shape, device=self.device, dtype=torch.float64),
            'wfa_sum': torch.zeros(full_shape, device=self.device, dtype=torch.float64),
            'wa_sum':  torch.zeros(full_shape, device=self.device, dtype=torch.float64),
            'wfaa_sum': torch.zeros(full_shape, device=self.device, dtype=torch.float64),
            'wfa2_sum': torch.zeros(full_shape, device=self.device, dtype=torch.float64),
            'wa2_sum':  torch.zeros(full_shape, device=self.device, dtype=torch.float64),
            # Unweighted sums for the unweighted mean used in centering
            'fa_sum': torch.zeros(full_shape, device=self.device, dtype=torch.float64),
            'a_sum':  torch.zeros(full_shape, device=self.device, dtype=torch.float64),
            'count':  torch.zeros(full_shape, device=self.device, dtype=torch.float64),
        }
    
    def _get_climatology_indices(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Convert timestamps [batch, 4] (year, month, day, hour) to climatology indices (0-365).
        
        Handles leap year adjustment: for non-leap years after Feb 28, skip Feb 29 in climatology.
        This matches the logic in get_climatology_index() in train.py.
        """
        year = timestamps[:, 0].long()
        month = timestamps[:, 1].long()
        day = timestamps[:, 2].long()
        
        # Days before each month (non-leap year)
        days_before_month = torch.tensor(
            [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334],
            device=self.device, dtype=torch.long
        )
        
        # Day of year (1-indexed)
        day_of_year = days_before_month[month - 1] + day
        
        # Leap year check
        is_leap = ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)
        after_feb = month > 2
        
        # Add 1 for leap years after Feb
        day_of_year = day_of_year + (is_leap & after_feb).long()
        
        # Convert to 0-indexed climatology index
        clim_idx = day_of_year - 1
        
        # For non-leap years after Feb 28, skip Feb 29 (index 59) in climatology
        non_leap_after_feb28 = (~is_leap) & (day_of_year > 59)
        clim_idx = clim_idx + non_leap_after_feb28.long()
        
        return torch.clamp(clim_idx, 0, 365)
    
    def update(
        self,
        pred_surface: torch.Tensor,       # [batch, num_vars, lat, lon]
        pred_upper_air: torch.Tensor,     # [batch, num_vars, levels, lat, lon]
        target_surface: torch.Tensor,
        target_upper_air: torch.Tensor,
        timestamps: torch.Tensor,         # [batch, 4] - year, month, day, hour
        step_idx: int,                    # Index into lead_time_steps (0-indexed)
        pred_diagnostic: Optional[torch.Tensor] = None,
        target_diagnostic: Optional[torch.Tensor] = None,
    ):
        """
        Update accumulators with a batch of predictions/targets.
        
        Call this inside the autoregressive validation loop for each lead time step.
        
        Args:
            pred_surface: Normalized predictions [batch, vars, lat, lon]
            pred_upper_air: Normalized predictions [batch, vars, levels, lat, lon]
            target_surface: Normalized targets [batch, vars, lat, lon]
            target_upper_air: Normalized targets [batch, vars, levels, lat, lon]
            timestamps: Start times [batch, 4] as (year, month, day, hour)
            step_idx: Which lead time step this is (index into self.lead_time_steps)
            pred_diagnostic: Optional diagnostic predictions
            target_diagnostic: Optional diagnostic targets
        """
        batch_size = pred_surface.shape[0]
        lat_size = pred_surface.shape[2]
        lon_size = pred_surface.shape[3]
        
        # ===== RMSE Update =====
        # Surface: accumulate sum of weighted squared errors
        sq_err_sfc = (pred_surface - target_surface) ** 2  # [batch, vars, lat, lon]
        weighted_sq_err_sfc = self.rmse_weights_2d * sq_err_sfc
        # Sum over batch, lat, lon; result is [vars]
        self.rmse_sfc_sum[step_idx] += weighted_sq_err_sfc.sum(dim=(0, 2, 3))
        self.rmse_sfc_count[step_idx] += batch_size * lat_size * lon_size
        
        # Upper air: [batch, vars, levels, lat, lon]
        sq_err_ua = (pred_upper_air - target_upper_air) ** 2
        weighted_sq_err_ua = self.rmse_weights_3d * sq_err_ua
        # Sum over batch, lat, lon; result is [vars, levels]
        self.rmse_ua_sum[step_idx] += weighted_sq_err_ua.sum(dim=(0, 3, 4))
        self.rmse_ua_count[step_idx] += batch_size * lat_size * lon_size
        
        if self.has_diagnostic and pred_diagnostic is not None:
            sq_err_diag = (pred_diagnostic - target_diagnostic) ** 2
            weighted_sq_err_diag = self.rmse_weights_2d * sq_err_diag
            self.rmse_diag_sum[step_idx] += weighted_sq_err_diag.sum(dim=(0, 2, 3))
            self.rmse_diag_count[step_idx] += batch_size * lat_size * lon_size
        
        # ===== ACC Update =====
        # Get climatology for this batch (need to denormalize first, then compute anomaly)
        # But wait - the original code computes ACC on DENORMALIZED data (after convert_to_xarray)
        # So we need to denormalize here too
        
        # Denormalize predictions and targets for ACC
        pred_sfc_denorm = pred_surface * self.surface_std.view(1, -1, 1, 1)
        target_sfc_denorm = target_surface * self.surface_std.view(1, -1, 1, 1)
        pred_ua_denorm = pred_upper_air * self.upper_air_std.view(1, -1, self.num_levels, 1, 1)
        target_ua_denorm = target_upper_air * self.upper_air_std.view(1, -1, self.num_levels, 1, 1)
        
        # Get climatology for these timestamps
        clim_idx = self._get_climatology_indices(timestamps)  # [batch]
        
        # Surface anomalies
        if self.clim_surface is not None:
            clim_sfc = self.clim_surface[clim_idx]  # [batch, vars, lat, lon]
            fa_sfc = pred_sfc_denorm - clim_sfc
            a_sfc = target_sfc_denorm - clim_sfc
            self._update_acc_accumulators(self.acc_sfc, fa_sfc, a_sfc, 
                                          self.acc_weights_2d, step_idx, reduce_dims=(0, 2, 3))
        
        # Upper air anomalies
        if self.clim_upper_air is not None:
            clim_ua = self.clim_upper_air[clim_idx]  # [batch, vars, levels, lat, lon]
            fa_ua = pred_ua_denorm - clim_ua
            a_ua = target_ua_denorm - clim_ua
            self._update_acc_accumulators(self.acc_ua, fa_ua, a_ua,
                                          self.acc_weights_3d, step_idx, reduce_dims=(0, 3, 4))
        
        # Diagnostic anomalies
        if self.has_diagnostic and pred_diagnostic is not None:
            pred_diag_denorm = pred_diagnostic * self.diagnostic_std.view(1, -1, 1, 1)
            target_diag_denorm = target_diagnostic * self.diagnostic_std.view(1, -1, 1, 1)
            if self.clim_diagnostic is not None:
                clim_diag = self.clim_diagnostic[clim_idx]
                fa_diag = pred_diag_denorm - clim_diag
                a_diag = target_diag_denorm - clim_diag
                self._update_acc_accumulators(self.acc_diag, fa_diag, a_diag,
                                              self.acc_weights_2d, step_idx, reduce_dims=(0, 2, 3))
    
    def _update_acc_accumulators(
        self,
        acc_dict: Dict[str, torch.Tensor],
        fa: torch.Tensor,  # forecast anomaly (from climatology)
        a: torch.Tensor,   # truth anomaly (from climatology)
        w: torch.Tensor,   # latitude weights
        step_idx: int,
        reduce_dims: Tuple[int, ...],
    ):
        """Accumulate raw float64 weighted sums for Pearson correlation."""
        fa = fa.double()
        a = a.double()
        w = w.double()

        w_expanded = w.expand_as(fa)

        acc_dict['w_sum'][step_idx] += w_expanded.sum(dim=reduce_dims)
        acc_dict['wfa_sum'][step_idx] += (w_expanded * fa).sum(dim=reduce_dims)
        acc_dict['wa_sum'][step_idx] += (w_expanded * a).sum(dim=reduce_dims)
        acc_dict['wfaa_sum'][step_idx] += (w_expanded * fa * a).sum(dim=reduce_dims)
        acc_dict['wfa2_sum'][step_idx] += (w_expanded * fa * fa).sum(dim=reduce_dims)
        acc_dict['wa2_sum'][step_idx] += (w_expanded * a * a).sum(dim=reduce_dims)
        acc_dict['fa_sum'][step_idx] += fa.sum(dim=reduce_dims)
        acc_dict['a_sum'][step_idx] += a.sum(dim=reduce_dims)

        num_elements = 1
        for d in reduce_dims:
            num_elements *= fa.shape[d]
        acc_dict['count'][step_idx] += num_elements
    
    def all_reduce(self):
        """Synchronize accumulators across DDP ranks."""
        if not dist.is_initialized():
            return
        
        # RMSE accumulators
        dist.all_reduce(self.rmse_sfc_sum)
        dist.all_reduce(self.rmse_ua_sum)
        dist.all_reduce(self.rmse_sfc_count)
        dist.all_reduce(self.rmse_ua_count)
        if self.has_diagnostic:
            dist.all_reduce(self.rmse_diag_sum)
            dist.all_reduce(self.rmse_diag_count)
        
        # ACC accumulators - only reduce tensor values, not metadata
        tensor_keys = ['w_sum', 'wfa_sum', 'wa_sum', 'wfaa_sum', 'wfa2_sum', 'wa2_sum',
                       'fa_sum', 'a_sum', 'count']
        for key in tensor_keys:
            dist.all_reduce(self.acc_sfc[key])
            dist.all_reduce(self.acc_ua[key])
            if self.has_diagnostic:
                dist.all_reduce(self.acc_diag[key])
        
    
    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Compute final RMSE and ACC metrics.
        
        Returns:
            Dictionary with:
            - 'rmse_surface': [num_steps, num_vars] - denormalized RMSE
            - 'rmse_upper_air': [num_steps, num_vars, num_levels]
            - 'rmse_diagnostic': [num_steps, num_vars] (if has_diagnostic)
            - 'acc_surface': [num_steps, num_vars]
            - 'acc_upper_air': [num_steps, num_vars, num_levels]
            - 'acc_diagnostic': [num_steps, num_vars] (if has_diagnostic)
        """
        results = {}
        
        # ===== RMSE =====
        # RMSE = sqrt(sum / count) * std
        # The weights are already applied, so this gives latitude-weighted RMSE
        rmse_sfc = torch.sqrt(self.rmse_sfc_sum / self.rmse_sfc_count.unsqueeze(-1))
        rmse_sfc = rmse_sfc * self.surface_std.unsqueeze(0)  # denormalize
        results['rmse_surface'] = rmse_sfc
        
        rmse_ua = torch.sqrt(self.rmse_ua_sum / self.rmse_ua_count.unsqueeze(-1).unsqueeze(-1))
        rmse_ua = rmse_ua * self.upper_air_std.unsqueeze(0)  # denormalize
        results['rmse_upper_air'] = rmse_ua
        
        if self.has_diagnostic:
            rmse_diag = torch.sqrt(self.rmse_diag_sum / self.rmse_diag_count.unsqueeze(-1))
            rmse_diag = rmse_diag * self.diagnostic_std.unsqueeze(0)
            results['rmse_diagnostic'] = rmse_diag
        
        # ===== ACC =====
        results['acc_surface'] = self._compute_acc(self.acc_sfc)
        results['acc_upper_air'] = self._compute_acc(self.acc_ua)
        if self.has_diagnostic:
            results['acc_diagnostic'] = self._compute_acc(self.acc_diag)
        
        return results
    
    def _compute_acc(self, acc_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute ACC from raw accumulated float64 sums.

        ACC uses unweighted mean for centering (matching xarray convention):
            fa' = fa - mean(fa),  a' = a - mean(a)   (unweighted means)
            ACC = E_w[fa' * a'] / sqrt(E_w[fa'²] * E_w[a'²])

        Using raw moments:
            E_w[fa' * a'] = E_w[fa*a] - μ_fa*E_w[a] - μ_a*E_w[fa] + μ_fa*μ_a
            E_w[fa'²]     = E_w[fa²]  - 2*μ_fa*E_w[fa] + μ_fa²
        """
        w_sum = acc_dict['w_sum']
        count = acc_dict['count']

        # Unweighted means for centering (matches xarray fa.mean())
        mean_fa = acc_dict['fa_sum'] / count
        mean_a = acc_dict['a_sum'] / count

        # Weighted expectations
        E_w_fa = acc_dict['wfa_sum'] / w_sum
        E_w_a = acc_dict['wa_sum'] / w_sum
        E_w_fa2 = acc_dict['wfa2_sum'] / w_sum
        E_w_a2 = acc_dict['wa2_sum'] / w_sum
        E_w_faa = acc_dict['wfaa_sum'] / w_sum

        # Covariance and variance with unweighted-mean centering
        cov = E_w_faa - mean_fa * E_w_a - mean_a * E_w_fa + mean_fa * mean_a
        var_fa = E_w_fa2 - 2 * mean_fa * E_w_fa + mean_fa ** 2
        var_a = E_w_a2 - 2 * mean_a * E_w_a + mean_a ** 2

        var_product = var_fa * var_a
        valid_mask = var_product > 1e-60

        acc = torch.zeros_like(cov)
        acc[valid_mask] = cov[valid_mask] / torch.sqrt(var_product[valid_mask])
        acc = torch.clamp(acc, -1.0, 1.0)
        return acc.float()
    
    def format_logs(self, epoch: int) -> Dict[str, float]:
        """
        Format metrics as a flat dictionary for wandb logging.
        
        Returns dict with keys like:
        - 'valid_tas_1step_lwrmse'
        - 'valid_zg_level50000.000_2step_lwrmse'
        - 'valid_tas_1step_acc'
        - 'valid_zg_level50000.000_2step_acc'
        """
        results = self.compute()
        logs = {'epoch': epoch}
        
        for step_i, step in enumerate(self.lead_time_steps):
            # Surface variables
            for var_i, var in enumerate(self.surface_variables):
                logs[f'valid_{var}_{step}step_lwrmse'] = results['rmse_surface'][step_i, var_i].item()
                logs[f'valid_{var}_{step}step_acc'] = results['acc_surface'][step_i, var_i].item()
            
            # Upper air variables
            for var_i, var in enumerate(self.upper_air_variables):
                # Use sigma levels for non-zg variables if enabled
                if var not in ['zg', 'geopotential_height'] and self.use_sigma_levels:
                    level_list = self.sigma_levels
                else:
                    level_list = self.levels
                
                for lev_i, level in enumerate(level_list):
                    logs[f'valid_{var}_level{level:.3f}_{step}step_lwrmse'] = \
                        results['rmse_upper_air'][step_i, var_i, lev_i].item()
                    logs[f'valid_{var}_level{level:.3f}_{step}step_acc'] = \
                        results['acc_upper_air'][step_i, var_i, lev_i].item()
            
            # Diagnostic variables
            if self.has_diagnostic:
                for var_i, var in enumerate(self.diagnostic_variables):
                    logs[f'valid_{var}_{step}step_lwrmse'] = results['rmse_diagnostic'][step_i, var_i].item()
                    logs[f'valid_{var}_{step}step_acc'] = results['acc_diagnostic'][step_i, var_i].item()
        
        # Aggregate metrics
        logs['valid_mean_lwrmse_surface'] = results['rmse_surface'].mean().item()
        logs['valid_mean_lwrmse_upper_air'] = results['rmse_upper_air'].mean().item()
        logs['valid_mean_acc_surface'] = results['acc_surface'].mean().item()
        logs['valid_mean_acc_upper_air'] = results['acc_upper_air'].mean().item()
        
        if self.has_diagnostic:
            logs['valid_mean_lwrmse_diagnostic'] = results['rmse_diagnostic'].mean().item()
            logs['valid_mean_acc_diagnostic'] = results['acc_diagnostic'].mean().item()
        
        return logs


def create_metrics_aggregator(trainer) -> MetricsAggregator:
    """
    Factory function to create MetricsAggregator from Trainer instance.
    
    This extracts all necessary configuration from the trainer to initialize
    the MetricsAggregator.
    
    Args:
        trainer: Trainer instance with valid_dataset and params
        
    Returns:
        Configured MetricsAggregator instance
    """
    dataset = trainer.valid_dataset
    params = trainer.params
    
    return MetricsAggregator(
        surface_variables=list(dataset.surface_variables),
        upper_air_variables=list(dataset.upper_air_variables),
        diagnostic_variables=list(dataset.diagnostic_variables) if hasattr(dataset, 'diagnostic_variables') else [],
        levels=dataset.levels,
        lead_time_steps=params.forecast_lead_times,
        latitudes=torch.from_numpy(np.array(params.lat)),
        climatology=trainer.climatology,
        surface_std=dataset.surface_std,
        upper_air_std=dataset.upper_air_std,
        diagnostic_std=dataset.diagnostic_std if hasattr(dataset, 'diagnostic_std') else None,
        device=trainer.device,
        use_sigma_levels=dataset.use_sigma_levels if hasattr(dataset, 'use_sigma_levels') else False,
        sigma_levels=dataset.sigma_levels if hasattr(dataset, 'sigma_levels') else None,
    )

def create_metrics_aggregator_new(trainer, lead_time_steps: list = None) -> MetricsAggregator:
    """
    Factory function to create MetricsAggregator from Trainer instance.
    
    This extracts all necessary configuration from the trainer to initialize
    the MetricsAggregator.
    
    Args:
        trainer: Trainer instance with valid_dataset and params
        lead_time_steps: Optional list of lead time steps to track. 
                         If None, uses params.forecast_lead_times.
                         For just_validate, pass range(1, max_lead_time+1).
        
    Returns:
        Configured MetricsAggregator instance
    """
    dataset = trainer.valid_dataset
    params = trainer.params
    
    # Use provided lead_time_steps or default to forecast_lead_times
    if lead_time_steps is None:
        lead_time_steps = params.forecast_lead_times
    
    return MetricsAggregator(
        surface_variables=list(dataset.surface_variables),
        upper_air_variables=list(dataset.upper_air_variables),
        diagnostic_variables=list(dataset.diagnostic_variables) if hasattr(dataset, 'diagnostic_variables') else [],
        levels=dataset.levels,
        lead_time_steps=list(lead_time_steps),  # Ensure it's a list
        latitudes=torch.from_numpy(np.array(params.lat)),
        climatology=trainer.climatology,
        surface_std=dataset.surface_std,
        upper_air_std=dataset.upper_air_std,
        diagnostic_std=dataset.diagnostic_std if hasattr(dataset, 'diagnostic_std') else None,
        device=trainer.device,
        use_sigma_levels=dataset.use_sigma_levels if hasattr(dataset, 'use_sigma_levels') else False,
        sigma_levels=dataset.sigma_levels if hasattr(dataset, 'sigma_levels') else None,
    )