import torch
import numpy as np
import torch.nn as nn
from einops import repeat, rearrange
import math 
from common.utils import assemble_input

try:
    import torch_harmonics as th
except ImportError:
    print("Warning: torch harmonics could not be imported.")

# base on the code from graphcast
def _check_uniform_spacing_and_get_delta(vector):
    diff = np.diff(vector)
    if not np.all(np.isclose(diff[0], diff)):
        raise ValueError(f'Vector {diff} is not uniformly spaced.')
    return diff[0]


def _weight_for_latitude_vector_without_poles(latitude):
    """Weights for uniform latitudes of the form [+-90-+d/2, ..., -+90+-d/2]."""
    delta_latitude = np.abs(_check_uniform_spacing_and_get_delta(latitude))
    if (not np.isclose(np.max(latitude), 90 - delta_latitude/2) or
        not np.isclose(np.min(latitude), -90 + delta_latitude/2)):
        raise ValueError(
            f'Latitude vector {latitude} does not start/end at '
            '+- (90 - delta_latitude/2) degrees.')
    return np.cos(np.deg2rad(latitude))


def _weight_for_latitude_vector_with_poles(latitude):
    """Weights for uniform latitudes of the form [+- 90, ..., -+90]."""
    delta_latitude = np.abs(_check_uniform_spacing_and_get_delta(latitude))
    if (not np.isclose(np.max(latitude), 90.) or
        not np.isclose(np.min(latitude), -90.)):
        raise ValueError(
            f'Latitude vector {latitude} does not start/end at +- 90 degrees.')
    weights = np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(delta_latitude/2))
    # The two checks above enough to guarantee that latitudes are sorted, so
    # the extremes are the poles
    weights[[0, -1]] = np.sin(np.deg2rad(delta_latitude/4)) ** 2
    return weights


class WeightedLoss(nn.Module):
    def __init__(self,
                 latitude_resolution = 180,
                 longitude_resolution = 360,
                 with_poles=False,
                 latitude_weight='equal',
                 level_weight='equal',
                 multi_level_variable_weight=None,
                 surface_variable_weight=None,
                 diag_variable_weight=None,
                 nlevels=26,
                 nsurface=6,
                 nmulti=5,
                 ndiag = 15,
                 normalize = True,
                 eps = 1e-3,
                 channel_first=True,
                 use_diagnostic=True,
                 ):
        super().__init__()
        self.use_diagnostic = use_diagnostic
        self.loss_fn = nn.MSELoss(reduction='none')
        if latitude_weight == 'cosine':
            if with_poles:
                latitude = np.linspace(-90, 90, latitude_resolution)
                weights = _weight_for_latitude_vector_with_poles(latitude)
            else:
                # assume equiangular grid
                lat_end = (latitude_resolution-1)*(360/longitude_resolution) / 2
                latitude = np.linspace(-lat_end, lat_end, latitude_resolution)
                weights = _weight_for_latitude_vector_without_poles(latitude)
            weights = torch.tensor(weights)
            latitude_weight = weights / weights.mean()
        else:
            weights = torch.ones(latitude_resolution)   # all latitudes weight the same
            latitude_weight = weights / weights.mean() # shape (nlat, )
        self.register_buffer('latitude_weight', latitude_weight)

        if level_weight == 'linear':     # weighs the lower levels
            level_weight = torch.linspace(0.065, 0.05, nlevels)
        elif level_weight == 'exp':
            level_weight = torch.exp(torch.linspace(0, -3, nlevels))
            level_weight = level_weight / level_weight.sum()
        else:
            level_weight = torch.ones(nlevels)
            level_weight = level_weight / level_weight.sum()
        self.register_buffer('level_weight', level_weight)

        if surface_variable_weight is None:
            surface_variable_weight = torch.ones(nsurface) # default equal weight
        else:
            surface_variable_weight = torch.tensor(surface_variable_weight, dtype=torch.float32)

        if multi_level_variable_weight is None:
            multi_level_variable_weight = torch.ones(nmulti) # default equal weight
        else:
            multi_level_variable_weight = torch.tensor(multi_level_variable_weight, dtype=torch.float32)

        if diag_variable_weight is None:
            diag_variable_weight = torch.ones(ndiag)
        else:
            diag_variable_weight = torch.tensor(diag_variable_weight, dtype=torch.float32)
        
        self.register_buffer('diag_variable_weight', diag_variable_weight)
        self.register_buffer('surface_variable_weight', surface_variable_weight)
        self.register_buffer('multi_level_variable_weight', multi_level_variable_weight)

        self.normalize = normalize
        self.eps = eps
        self.channel_first= channel_first

    def forward(self,
                surface_pred, surface_target,
                multilevel_pred, multilevel_target,
                diagnostic_pred=None, diagnostic_target=None
                ):

        if self.channel_first:
            # (b, c, nlat, nlon) -> (b, nlat, nlon, c)
            surface_pred = surface_pred.permute(0, 2, 3, 1)
            surface_target = surface_target.permute(0, 2, 3, 1)
            # (b, c, nlevel, nlat, nlon) -> (b, nlevel, nlat, nlon, c)
            multilevel_pred = multilevel_pred.permute(0, 2, 3, 4, 1)
            multilevel_target = multilevel_target.permute(0, 2, 3, 4, 1)

        surface_loss = self.loss_fn(surface_pred, surface_target) # b nlat nlon nsurface
        surface_loss = surface_loss * self.surface_variable_weight.view(1, 1, 1, -1) # b nlat nlon nsurface
        surface_loss = surface_loss.sum(dim=-1) # b nlat nlon

        multi_level_loss = self.loss_fn(multilevel_pred, multilevel_target) # b nlevel nlat nlon nmulti
        multi_level_loss = multi_level_loss * self.level_weight.view(1, -1, 1, 1, 1) # b nlevel nlat nlon nmulti
        multi_level_loss = multi_level_loss.sum(dim=1) # b nlat nlon nmulti
        multi_level_loss = multi_level_loss * self.multi_level_variable_weight.view(1, 1, 1, -1) # b nlat nlon nmulti
        multi_level_loss = multi_level_loss.sum(dim=-1) # b nlat nlon

        if self.normalize:
            surface_loss = surface_loss / (torch.norm(surface_target, p=2, keepdim=True) + self.eps)
            multi_level_loss = multi_level_loss / (torch.norm(multilevel_target, p=2, keepdim=True) + self.eps)

        loss = surface_loss + multi_level_loss # b nlat nlon

        if self.use_diagnostic and diagnostic_pred is not None and diagnostic_target is not None:
            if self.channel_first:
                diagnostic_pred = diagnostic_pred.permute(0, 2, 3, 1)
                diagnostic_target = diagnostic_target.permute(0, 2, 3, 1)

            diag_loss = self.loss_fn(diagnostic_pred, diagnostic_target) # b nlat nlon ndiag
            diag_loss = diag_loss * self.diag_variable_weight.view(1, 1, 1, -1) # b nlat nlon ndiag
            diag_loss = diag_loss.sum(dim=-1) # b nlat nlon

            if self.normalize:
                diag_loss = diag_loss / (torch.norm(diagnostic_target, p=2, keepdim=True) + self.eps)

            loss = loss + diag_loss

        latitude_weight = self.latitude_weight.view(1, -1, 1) # b nlat nlon
        loss = loss * latitude_weight

        return loss.mean()   # reduce over batch/lat/lon


class LatitudeWeightedMSE(nn.Module):
    def __init__(self, nlat, nlon, loss_module=nn.MSELoss(), with_poles=False):
        super().__init__()
        self.loss_module = loss_module
        self.with_poles = with_poles
        # print(nlat, nlon)

        if not with_poles:
            longitude_resolution = nlon
            lat_end = (nlat - 1) * (360 / longitude_resolution) / 2
            lat_weight = _weight_for_latitude_vector_without_poles(np.linspace(-lat_end, lat_end, nlat))
        else:
            lat_weight = _weight_for_latitude_vector_with_poles(np.linspace(-90, 90, nlat))

        lat_weight = torch.tensor(lat_weight)
        lat_weight = lat_weight / lat_weight.mean()
        self.register_buffer('lat_weight', lat_weight)

    def forward(self, pred, target):
        # pred, target in shape [b, nlat, nlon, c]
        lat_weight = repeat(self.lat_weight, 'nlat -> b nlat nlon', b=pred.shape[0], nlon=pred.shape[2])
        return (self.loss_module(pred, target).mean(-1) * lat_weight).mean()


def latitude_weighted_rmse(pred, 
                           target,
                           with_poles=False, 
                           nlon=None,
                           nlat=None,
                           with_time=True):
    # if with_time, pred/target in shape: b t nlat nlon or b t l nlat nlon
    # else, pred/target in shape: b nlat nlon or b l nlat nlon

    if nlat is None:
        nlat = target.shape[2]
    if not with_poles:
        lat_end = (nlat-1)*(360/nlon) / 2
        lat_weight = _weight_for_latitude_vector_without_poles(np.linspace(-lat_end, lat_end, nlat))
    else:
        lat_weight = _weight_for_latitude_vector_with_poles(np.linspace(-90, 90, nlat))

    lat_weight = torch.tensor(lat_weight).to(target.device)
    lat_weight = lat_weight / lat_weight.mean()
    if with_time:
        if len(pred.shape) == 5:
            lat_weight = lat_weight.view(1, 1, nlat, 1, 1)
            pred = rearrange(pred, 'b t l nlat nlon -> b t nlat nlon l')
            target = rearrange(target, 'b t l nlat nlon -> b t nlat nlon l')
        else:
            lat_weight = lat_weight.view(1, 1, nlat, 1)
        return torch.sqrt((((pred - target)**2) * lat_weight).mean(dim=(2, 3)))   # spatial averaging
    else:
        if len(pred.shape) == 4:
            lat_weight = lat_weight.view(1, nlat, 1, 1)
            pred = rearrange(pred, 'b l nlat nlon -> b nlat nlon l')
            target = rearrange(target, 'b l nlat nlon -> b nlat nlon l')
        else:
            lat_weight = lat_weight.view(1, nlat, 1)
        return torch.sqrt((((pred - target)**2) * lat_weight).mean(dim=(1, 2)))   # spatial averaging

def rmse(pred, target):
    # directly infer latitude from target: b t nface nside nside or b t nface nside nside l
    return torch.sqrt(((pred - target)**2).mean(dim=(2, 3, 4)))   # spatial averaging

def latitude_weighted_l1(pred, target):
    # directly infer latitude from target: b t nlat nlon or b t nlat nlon l
    nlat = target.shape[2]
    lat_weight = _weight_for_latitude_vector_with_poles(np.linspace(-90, 90, nlat))
    lat_weight = torch.tensor(lat_weight).to(target.device)
    lat_weight = lat_weight / lat_weight.mean()
    if len(pred.shape) == 5:
        lat_weight = lat_weight.view(1, 1, nlat, 1, 1)
    else:
        lat_weight = lat_weight.view(1, 1, nlat, 1)

    return ((pred - target).abs() * lat_weight).mean(dim=(2, 3))   # spatial averaging


class FairCRPSLoss(nn.Module):
    """
    Almost-fair CRPS estimator for N=2 ensemble members (ACE2S, Appendix A):

    afCRPS_{α,M}(F,y) = E[|X-y|] - (1 - (1-α)/M) * (1/2) * E[|X-X'|]

    When alpha=1.0, this reduces to the standard fair CRPS (FGN Eq. 5).
    When alpha<1.0 (e.g. 0.95), the spread penalty is slightly reduced,
    encouraging more ensemble diversity.

    Supports latitude weighting, level weighting, and per-variable weighting
    matching the WeightedLoss interface.
    """
    def __init__(self,
                 latitude_resolution=180,
                 longitude_resolution=360,
                 with_poles=False,
                 latitude_weight='equal',
                 level_weight='equal',
                 multi_level_variable_weight=None,
                 surface_variable_weight=None,
                 diag_variable_weight=None,
                 nlevels=26,
                 nsurface=6,
                 nmulti=5,
                 ndiag=15,
                 alpha=1.0,
                 n_ensemble=2,
                 ):
        super().__init__()
        # Latitude weighting (same as WeightedLoss)
        if latitude_weight == 'cosine':
            if with_poles:
                latitude = np.linspace(-90, 90, latitude_resolution)
                weights = _weight_for_latitude_vector_with_poles(latitude)
            else:
                lat_end = (latitude_resolution - 1) * (360 / longitude_resolution) / 2
                latitude = np.linspace(-lat_end, lat_end, latitude_resolution)
                weights = _weight_for_latitude_vector_without_poles(latitude)
            weights = torch.tensor(weights)
            latitude_weight = weights / weights.mean()
        else:
            weights = torch.ones(latitude_resolution)
            latitude_weight = weights / weights.mean()
        self.register_buffer('latitude_weight', latitude_weight)

        # Level weighting
        if level_weight == 'linear':
            level_weight = torch.linspace(0.065, 0.05, nlevels)
        elif level_weight == 'exp':
            level_weight = torch.exp(torch.linspace(0, -3, nlevels))
            level_weight = level_weight / level_weight.sum()
        elif level_weight == 'cosine':
            level_weight = torch.cos(torch.linspace(0, math.pi / 2, nlevels))
            level_weight = level_weight / level_weight.sum()
        else:
            level_weight = torch.ones(nlevels)
            level_weight = level_weight / level_weight.sum()
        self.register_buffer('level_weight', level_weight)

        # Variable weighting
        if surface_variable_weight is None:
            surface_variable_weight = torch.ones(nsurface)
        else:
            surface_variable_weight = torch.tensor(surface_variable_weight, dtype=torch.float32)

        if multi_level_variable_weight is None:
            multi_level_variable_weight = torch.ones(nmulti)
        else:
            multi_level_variable_weight = torch.tensor(multi_level_variable_weight, dtype=torch.float32)

        if diag_variable_weight is None:
            diag_variable_weight = torch.ones(ndiag)
        else:
            diag_variable_weight = torch.tensor(diag_variable_weight, dtype=torch.float32)

        self.register_buffer('diag_variable_weight', diag_variable_weight)
        self.register_buffer('surface_variable_weight', surface_variable_weight)
        self.register_buffer('multi_level_variable_weight', multi_level_variable_weight)

        # Almost-fair CRPS spread factor: (1 - (1-alpha)/M)
        # alpha=1.0 gives standard fair CRPS, alpha=0.95 gives almost-fair
        self.spread_factor = 1.0 - (1.0 - alpha) / n_ensemble
        self.N = n_ensemble

    def _fair_crps_gridpoint(self, pred1, pred2, target):
        """Compute per-gridpoint almost-fair CRPS for N=2 ensemble members."""

        return (1 / self.N) * (torch.abs(pred1 - target) + torch.abs(pred2 - target)) \
               - self.spread_factor * 0.5 * (1 / self.N) * torch.abs(pred1 - pred2)

    def forward(self,
                surface_pred1, surface_pred2, surface_target,
                multilevel_pred1, multilevel_pred2, multilevel_target,
                diagnostic_pred1, diagnostic_pred2, diagnostic_target):
        """
        All surface/diagnostic tensors: (B, nlat, nlon, C)
        All multilevel tensors: (B, nlevel, nlat, nlon, C)
        """
        # Surface CRPS: (B, nlat, nlon, nsurface)
        surface_crps = self._fair_crps_gridpoint(surface_pred1, surface_pred2, surface_target)
        surface_crps = surface_crps * self.surface_variable_weight.view(1, 1, 1, -1)
        surface_crps = surface_crps.sum(dim=-1)  # (B, nlat, nlon)

        # Diagnostic CRPS: (B, nlat, nlon, ndiag)
        diag_crps = self._fair_crps_gridpoint(diagnostic_pred1, diagnostic_pred2, diagnostic_target)
        diag_crps = diag_crps * self.diag_variable_weight.view(1, 1, 1, -1)
        diag_crps = diag_crps.sum(dim=-1)  # (B, nlat, nlon)

        # Multilevel CRPS: (B, nlevel, nlat, nlon, nmulti)
        multi_crps = self._fair_crps_gridpoint(multilevel_pred1, multilevel_pred2, multilevel_target)
        multi_crps = multi_crps * self.level_weight.view(1, -1, 1, 1, 1)
        multi_crps = multi_crps.sum(dim=1)  # (B, nlat, nlon, nmulti)
        multi_crps = multi_crps * self.multi_level_variable_weight.view(1, 1, 1, -1)
        multi_crps = multi_crps.sum(dim=-1)  # (B, nlat, nlon)

        loss = surface_crps + multi_crps + diag_crps  # (B, nlat, nlon)
        latitude_weight = self.latitude_weight.view(1, -1, 1)
        loss = loss * latitude_weight

        return loss.mean()


class SpectralBaseLoss(nn.Module):
    """
    Geometric base loss class used by all geometric losses
    """

    def __init__(
        self,
        img_shape=(180, 360),
        grid_type='equiangular',
        eps=1e-3,
        absolute=False,
        surface_uv_idx=(4, 5),
        not_surface_uv_idx = (0, 1, 2, 3),
        multilevel_uv_idx=(1, 2),
        not_multilevel_uv_idx=(0, 3, 4),
        vector_loss_weight=0.25,
        z500_weight=1.0,
        channel_first=True,
        use_diagnostic=True,
        K = -1,
    ):
        super().__init__()
        self.eps = eps
        self.absolute = absolute
        self.vector_loss_weight = vector_loss_weight
        self.channel_first = channel_first
        self.use_diagnostic = use_diagnostic

        if K > 0:
            self.sht = th.RealSHT(*img_shape, grid=grid_type, lmax=K, mmax=K).float()
            self.vsht = th.RealVectorSHT(*img_shape, grid=grid_type, lmax=K, mmax=K).float()
        else:
            self.sht  = th.RealSHT(*img_shape, grid=grid_type).float()
            self.vsht = th.RealVectorSHT(*img_shape, grid=grid_type).float()

        self.surface_uv_idx = surface_uv_idx
        self.multilevel_uv_idx = multilevel_uv_idx
        self.not_surface_uv_idx = not_surface_uv_idx
        self.not_multilevel_uv_idx = not_multilevel_uv_idx
        self.z500_weight = z500_weight

        self.register_buffer('not_surface_uv_idx_tensor', torch.tensor(not_surface_uv_idx), persistent=False)
        self.register_buffer('not_multilevel_uv_idx_tensor', torch.tensor(not_multilevel_uv_idx), persistent=False)
        self.register_buffer('surface_uv_idx_tensor', torch.tensor(surface_uv_idx), persistent=False)
        self.register_buffer('multilevel_uv_idx_tensor', torch.tensor(multilevel_uv_idx), persistent=False)

        # Spectral weights for scalar SHT: uniform in l, double-weight m>0
        lmax, mmax = self.sht.lmax, self.sht.mmax
        l_w = torch.ones(lmax)
        m_w = 2 * torch.ones(mmax)
        m_w[0] = 1.0
        l_w, m_w = torch.meshgrid(l_w, m_w, indexing='ij')
        self.register_buffer('lm_weights', l_w * m_w, persistent=False)

        # Spectral weights for vector SHT (same convention)
        lmax_v, mmax_v = self.vsht.lmax, self.vsht.mmax
        l_w_v = torch.ones(lmax_v)
        m_w_v = 2 * torch.ones(mmax_v)
        m_w_v[0] = 1.0
        l_w_v, m_w_v = torch.meshgrid(l_w_v, m_w_v, indexing='ij')
        self.register_buffer('lm_weights_v', l_w_v * m_w_v, persistent=False)

    def forward(self, surface_pred, surface_target,
                    multilevel_pred, multilevel_target,
                    diagnostic_pred=None, diagnostic_target=None) -> torch.Tensor:

        if self.channel_first:
            # Permute to channels-last for variable indexing
            # (b, c, nlat, nlon) -> (b, nlat, nlon, c)
            surface_pred = surface_pred.permute(0, 2, 3, 1)
            surface_target = surface_target.permute(0, 2, 3, 1)
            # (b, c, nlevel, nlat, nlon) -> (b, nlevel, nlat, nlon, c)
            multilevel_pred = multilevel_pred.permute(0, 2, 3, 4, 1)
            multilevel_target = multilevel_target.permute(0, 2, 3, 4, 1)

        use_diag = self.use_diagnostic and diagnostic_pred is not None and diagnostic_target is not None

        if use_diag and self.channel_first:
            diagnostic_pred = diagnostic_pred.permute(0, 2, 3, 1)
            diagnostic_target = diagnostic_target.permute(0, 2, 3, 1)

        # surface: (B, nlat, nlon, nsurface) — variables in last dim
        surface_pred_uv = surface_pred[..., self.surface_uv_idx_tensor]         # B nlat nlon 2
        surface_target_uv = surface_target[..., self.surface_uv_idx_tensor]     # B nlat nlon 2
        surface_pred_scalar = surface_pred[..., self.not_surface_uv_idx_tensor] # B nlat nlon C_s
        surface_target_scalar = surface_target[..., self.not_surface_uv_idx_tensor]

        # multilevel: (B, nlevel, nlat, nlon, nmulti) — variables in last dim
        multilevel_pred_uv = multilevel_pred[..., self.multilevel_uv_idx_tensor]         # B nlevel nlat nlon 2
        multilevel_target_uv = multilevel_target[..., self.multilevel_uv_idx_tensor]     # B nlevel nlat nlon 2
        multilevel_pred_scalar = multilevel_pred[..., self.not_multilevel_uv_idx_tensor] # B nlevel nlat nlon C_m
        multilevel_target_scalar = multilevel_target[..., self.not_multilevel_uv_idx_tensor]

        # --- Scalar SHT loss (all non-wind channels) ---
        # Permute back to channel-first for assemble_input and SHT
        # surface scalar: (B, nlat, nlon, C_s) -> (B, C_s, nlat, nlon)
        # multilevel scalar: (B, nlevel, nlat, nlon, C_m) -> (B, C_m, nlevel, nlat, nlon)
        diag_pred_input = diagnostic_pred.permute(0, 3, 1, 2) if use_diag else None
        diag_target_input = diagnostic_target.permute(0, 3, 1, 2) if use_diag else None
        forecasts = assemble_input(surface_pred_scalar.permute(0, 3, 1, 2),
                                   multilevel_pred_scalar.permute(0, 4, 1, 2, 3),
                                   diag_pred_input)
        observations = assemble_input(surface_target_scalar.permute(0, 3, 1, 2),
                                      multilevel_target_scalar.permute(0, 4, 1, 2, 3),
                                      diag_target_input)
        forecasts = self.sht(forecasts) / 4.0 / math.pi
        observations = self.sht(observations) / 4.0 / math.pi

        if self.absolute:
            forecasts = torch.abs(forecasts)
            observations = torch.abs(observations)
        else:
            forecasts = torch.view_as_real(forecasts)
            observations = torch.view_as_real(observations)
            # (B, C, lmax, mmax, 2) -> (B, C, 2, lmax, mmax) -> (B, 2C, lmax, mmax)
            forecasts = torch.movedim(forecasts, 4, 2).flatten(1, 2)
            observations = torch.movedim(observations, 4, 2).flatten(1, 2)

        B, C, H, W = forecasts.shape
        spectral_weights_split = self.lm_weights.reshape(1, 1, H * W)

        crps = torch.abs(observations - forecasts).reshape(B, C, H * W)
        norm = torch.abs(observations).reshape(B, C, H * W)
        scalar_loss = torch.sum(crps * spectral_weights_split, dim=-1).mean() / \
                      (torch.sum(norm * spectral_weights_split, dim=-1).mean() + self.eps)

        # --- Vector SHT loss (u/v wind pairs) ---
        # Surface: (B, nlat, nlon, 2) -> (B, 2, nlat, nlon)
        f_surf_uv = surface_pred_uv.permute(0, 3, 1, 2)
        o_surf_uv = surface_target_uv.permute(0, 3, 1, 2)

        # Multilevel: (B, nlevel, nlat, nlon, 2) -> (B*nlevel, 2, nlat, nlon)
        nlevel, nlat, nlon = multilevel_pred_uv.shape[1], multilevel_pred_uv.shape[2], multilevel_pred_uv.shape[3]
        f_multi_uv = multilevel_pred_uv.permute(0, 1, 4, 2, 3).reshape(B * nlevel, 2, nlat, nlon)
        o_multi_uv = multilevel_target_uv.permute(0, 1, 4, 2, 3).reshape(B * nlevel, 2, nlat, nlon)

        # Combine surface and multilevel pairs into a single batch for one vsht call
        f_all_uv = torch.cat([f_surf_uv, f_multi_uv], dim=0)  # (B + B*nlevel, 2, nlat, nlon)
        o_all_uv = torch.cat([o_surf_uv, o_multi_uv], dim=0)

        # Apply vector SHT -> (N, 2, lmax_v, mmax_v) complex; dim 1: [spheroidal, toroidal]
        f_v = self.vsht(f_all_uv) / 4.0 / math.pi
        o_v = self.vsht(o_all_uv) / 4.0 / math.pi

        if self.absolute:
            f_v = torch.abs(f_v)
            o_v = torch.abs(o_v)
        else:
            # (N, 2, lmax_v, mmax_v, 2) -> (N, 2, 2, lmax_v, mmax_v) -> (N, 4, lmax_v, mmax_v)
            f_v = torch.movedim(torch.view_as_real(f_v), 4, 2).flatten(1, 2)
            o_v = torch.movedim(torch.view_as_real(o_v), 4, 2).flatten(1, 2)

        N, C_v, H_v, W_v = f_v.shape
        swv = self.lm_weights_v.reshape(1, 1, H_v * W_v)
        v_crps = torch.abs(o_v - f_v).reshape(N, C_v, H_v * W_v)
        v_norm = torch.abs(o_v).reshape(N, C_v, H_v * W_v)
        vector_loss = torch.sum(v_crps * swv, dim=-1).mean() / \
                      (torch.sum(v_norm * swv, dim=-1).mean() + self.eps)


        if self.z500_weight != 1.0:
            # z500 is variable index 3 in channels-last multilevel: (B, nlevel, nlat, nlon)
            z500_pred = multilevel_pred[..., 3]
            z500_target = multilevel_target[..., 3]

            z500_forecasts = self.sht(z500_pred) / 4.0 / math.pi
            z500_observations = self.sht(z500_target) / 4.0 / math.pi

            z500_forecasts = torch.view_as_real(z500_forecasts)
            z500_observations = torch.view_as_real(z500_observations)
            # (B, C, lmax, mmax, 2) -> (B, C, 2, lmax, mmax) -> (B, 2C, lmax, mmax)
            z500_forecasts = torch.movedim(z500_forecasts, 4, 2).flatten(1, 2)
            z500_observations = torch.movedim(z500_observations, 4, 2).flatten(1, 2)
            
            B, C, H, W = z500_forecasts.shape
            spectral_weights_split = self.lm_weights.reshape(1, 1, H * W)
            z500_norm = torch.abs(z500_observations).reshape(B, C, H * W)

            crps_z500 = torch.abs(z500_observations - z500_forecasts).reshape(B, C, H * W)
            z500_loss = torch.sum(crps_z500 * spectral_weights_split, dim=-1).mean() / \
                        (torch.sum(z500_norm * spectral_weights_split, dim=-1).mean() + self.eps)

            return scalar_loss + self.vector_loss_weight * vector_loss + self.z500_weight * z500_loss

        return scalar_loss + self.vector_loss_weight * vector_loss
    


class SpectralScalarLoss(nn.Module):
    """
    Geometric base loss class used by all geometric losses
    """

    def __init__(
        self,
        img_shape=(180, 360),
        grid_type='equiangular',
        absolute=False,
    ):
        super().__init__()
        self.absolute = absolute

        self.sht  = th.RealSHT(*img_shape, grid=grid_type).float()
        self.vsht = th.RealVectorSHT(*img_shape, grid=grid_type).float()

        # Spectral weights for scalar SHT: uniform in l, double-weight m>0
        lmax, mmax = self.sht.lmax, self.sht.mmax
        l_w = torch.ones(lmax)
        m_w = 2 * torch.ones(mmax)
        m_w[0] = 1.0
        l_w, m_w = torch.meshgrid(l_w, m_w, indexing='ij')
        self.register_buffer('lm_weights', l_w * m_w, persistent=False)

        # Spectral weights for vector SHT (same convention)
        lmax_v, mmax_v = self.vsht.lmax, self.vsht.mmax
        l_w_v = torch.ones(lmax_v)
        m_w_v = 2 * torch.ones(mmax_v)
        m_w_v[0] = 1.0
        l_w_v, m_w_v = torch.meshgrid(l_w_v, m_w_v, indexing='ij')
        self.register_buffer('lm_weights_v', l_w_v * m_w_v, persistent=False)

    def forward(self, x_pred, x_true) -> torch.Tensor:

        forecasts = self.sht(x_pred) / 4.0 / math.pi
        observations = self.sht(x_true) / 4.0 / math.pi

        if self.absolute:
            forecasts = torch.abs(forecasts)
            observations = torch.abs(observations)
        else:
            forecasts = torch.view_as_real(forecasts)
            observations = torch.view_as_real(observations)
            # (B, C, lmax, mmax, 2) -> (B, C, 2, lmax, mmax) -> (B, 2C, lmax, mmax)
            forecasts = torch.movedim(forecasts, 4, 2).flatten(1, 2)
            observations = torch.movedim(observations, 4, 2).flatten(1, 2)

        B, C, H, W = forecasts.shape
        spectral_weights_split = self.lm_weights.reshape(1, 1, H * W)

        diff = torch.abs(observations - forecasts).reshape(B, C, H * W)
        scalar_loss = torch.sum(diff * spectral_weights_split, dim=-1).mean()

        return scalar_loss 
    

def rankdata(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    ordinal ranking along dimension dim
    """
    ndim = x.dim()
    perm = torch.argsort(x, dim=dim, descending=False, stable=True)

    idx = torch.arange(x.shape[dim], device=x.device).reshape([-1 if i == dim else 1 for i in range(ndim)])
    rank = torch.empty_like(x, dtype=torch.long).scatter_(dim=dim, index=perm, src=idx.expand_as(perm)) + 1
    return rank

def _crps_skillspread_kernel(observation: torch.Tensor, forecasts: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    alternative CRPS variant that uses spread and skill
    """

    observation = observation.unsqueeze(0)

    # get the ranks for the spread computation
    rank = rankdata(forecasts, dim=0)

    #  ensemble size
    num_ensemble = forecasts.shape[0]

    # get the ensemble spread (total_weight is ensemble size here)
    espread = 2 * torch.mean((2 * rank - num_ensemble - 1) * forecasts, dim=0) * (float(num_ensemble) - 1.0 + alpha) / float(num_ensemble * (num_ensemble - 1))
    eskill = (observation - forecasts).abs().mean(dim=0)

    # crps = torch.where(nanmasks.sum(dim=0) != 0, torch.nan, eskill - 0.5 * espread)
    crps = eskill - 0.5 * espread

    return crps

class SpectralCRPSLoss(nn.Module):
    """
    Geometric base loss class used by all geometric losses
    """

    def __init__(
        self,
        img_shape = (180, 360),
        grid_type = 'equiangular',
        eps = 1e-3,
        absolute = False,
        alpha=0.95,
    ):
        super().__init__()

        self.img_shape = img_shape

        self.sht = th.RealSHT(*img_shape, grid=grid_type).float()

        # get the local l weights
        lmax = self.sht.lmax
        # l_weights = 1 / (2*ls+1)
        l_weights = torch.ones(lmax)

        # get the local m weights
        mmax = self.sht.mmax
        m_weights = 2 * torch.ones(mmax)#.reshape(1, -1)
        m_weights[0] = 1.0

        # get meshgrid of weights:
        l_weights, m_weights = torch.meshgrid(l_weights, m_weights, indexing="ij")

        # use the product weights
        lm_weights = l_weights * m_weights

        self.eps = eps
        self.absolute = absolute
        self.alpha = alpha

        # register
        self.register_buffer("lm_weights", lm_weights, persistent=False)

    def forward(self, forecasts: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:

        # get the data type before stripping amp types
        dtype = forecasts.dtype

        forecasts = self.sht(forecasts) / 4.0 / math.pi
        observations = self.sht(observations) / 4.0 / math.pi

        if self.absolute:
            forecasts = torch.abs(forecasts).to(dtype)
            observations = torch.abs(observations).to(dtype)
        else:
            forecasts = torch.view_as_real(forecasts).to(dtype)
            observations = torch.view_as_real(observations).to(dtype)

            # merge complex dimension after channel dimension and flatten
            # this needs to be undone at the end
            forecasts = torch.movedim(forecasts, 5, 3).flatten(2, 3)
            observations = torch.movedim(observations, 4, 2).flatten(1, 2)

        # we assume the following shapes:
        # forecasts: batch, ensemble, channels, mmax, lmax
        # observations: batch, channels, mmax, lmax
        B, E, C, H, W = forecasts.shape

        spectral_weights = self.lm_weights

        # transpose forecasts: ensemble, batch, channels, lat, lon
        forecasts = torch.movedim(forecasts, 1, 0)

        # now we need to transpose the forecasts into ensemble direction.
        # ideally we split spatial dims
        forecasts = forecasts.reshape(E, B, C, H * W)

        # observations does not need a transpose, but just a split
        observations = observations.reshape(B, C, H * W)

        # tile in complex dim, then flatten last 3 dims
        spectral_weights_split = spectral_weights.reshape(1, 1, H * W)


        crps = _crps_skillspread_kernel(observations, forecasts, self.alpha)

        # perform spatial average of crps score
        crps = torch.sum(crps * spectral_weights_split, dim=-1)

        # finally undo the folding of the complex dimension into the channel dimension
        if not self.absolute:
            crps = crps.reshape(B, -1, 2).sum(dim=-1)

        # the resulting tensor should have dimension B, C, which is what we return
        return torch.mean(crps)