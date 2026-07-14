"""
Dataset for weather/climate forecasting models.

Loads atmospheric state data from per-timestep HDF5 files, applies normalization,
and returns (input, target) pairs for training or multi-step rollout sequences
for autoregressive inference/validation.

Data layout per file: ``{year}_{index:04d}.h5`` containing an ``input`` group
with one dataset per variable (plus ``time``).  Variables are split into:

- **Upper-air** (3-D): variables on pressure levels (e.g. temperature, wind)
- **Surface** (2-D): single-level fields (e.g. 2m temperature, surface pressure)
- **Diagnostic** (2-D, output only): radiation fluxes, precipitation, etc.
- **Varying boundary** (2-D, input only): SST, sea-ice, TOA solar forcing
- **Constant boundary**: land-sea mask, surface geopotential (loaded once)

Calendar-aware date handling is provided via ``cftime`` so that non-standard
calendars (no-leap, 360-day, etc.) used by different climate models are supported.
"""

import sys

import cftime
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from datetime import timedelta
from itertools import product
from os.path import join
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_data_given_path(path, variables):
    """Read selected variables from an HDF5 file and return as a stacked array.

    Parameters
    ----------
    path : str
        Path to an HDF5 file with an ``input`` group.
    variables : list[str]
        Variable names to extract from the ``input`` group.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_variables, ...)``.
    """
    with h5py.File(path, 'r') as f:
        data = {
            sub_key: np.array(value)
            for sub_key, value in f['input'].items()
            if sub_key in variables + ['time']
        }
    return np.stack([data[v] for v in variables], axis=0)


def get_out_path(root_dir, year, file_idx):
    """Build the HDF5 file path for a given year and timestep index."""
    return join(root_dir, f'{year}_{file_idx:04}.h5')


def _gaussian_kernel_2d(k, sigma, device, dtype=torch.float32):
    ax = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
    g = torch.exp(-ax**2 / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def smooth_masked_boundary(
    data: torch.Tensor,
    mask: torch.Tensor,
    sigma: float = 1.5,
    kernel_size: int = 9,
    n_iters: int = 10,
    lon_circular: bool = True,
) -> torch.Tensor:
    """Preserve interior values exactly and produce a smooth, curved fade to 0
    outside the mask using iterative Dirichlet diffusion.

    Each iteration:
        state <- Gaussian blur of state
        state <- original data inside mask, blurred state outside

    The interior is reset every step so it stays exact. Outside, the field
    spreads roughly sigma * sqrt(2 * n_iters) pixels from the boundary, with
    smooth iso-contours (no blocky 3x3 staircasing).

    Parameters
    ----------
    data : Tensor (..., H, W). Must be 0 wherever mask == 0.
    mask : Tensor (..., H, W), binary.
    sigma : Std-dev of the per-step Gaussian, in pixels. Controls smoothness
        of the contours.
    kernel_size : Width of the Gaussian, ideally >= 6 * sigma + 1.
    n_iters : Number of diffusion steps. Larger -> wider, softer transition.
    lon_circular : Periodic last axis (global longitude).
    """
    out_dtype = data.dtype
    *batch, H, W = data.shape
    d = data.reshape(-1, 1, H, W).to(torch.float32)
    m = mask.reshape(-1, 1, H, W).to(torch.float32)

    kernel = _gaussian_kernel_2d(kernel_size, sigma, device=d.device)
    p = kernel_size // 2
    inv_m = 1.0 - m
    d_in = d * m

    state = d.clone()
    for _ in range(n_iters):
        lon_mode = "circular" if lon_circular else "replicate"
        s = F.pad(state, (p, p, 0, 0), mode=lon_mode)
        s = F.pad(s,     (0, 0, p, p), mode="replicate")
        blurred = F.conv2d(s, kernel)
        state = d_in + blurred * inv_m

    return state.reshape(*batch, H, W).to(out_dtype)


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------

CALENDAR_TO_DATETIME = {
    'standard': cftime.DatetimeGregorian,
    'Gregorian': cftime.DatetimeGregorian,
    'noleap': cftime.DatetimeNoLeap,
    '365_day': cftime.DatetimeNoLeap,
    'proleptic_gregorian': cftime.DatetimeProlepticGregorian,
    'all_leap': cftime.DatetimeAllLeap,
    '366_day': cftime.DatetimeAllLeap,
    '360_day': cftime.Datetime360Day,
    'julian': cftime.DatetimeJulian,
}


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def get_data_loader(params, 
                    year_start: int, 
                    year_end: int, 
                    num_inferences: int = 0,
                    train: bool = True, 
                    validate: bool = False,
                    shuffle: bool = True):
    """Create a DataLoader (and sampler) for training or evaluation.

    Parameters
    ----------
    params : dict-like
        Full dataset/config parameters forwarded to :class:`GetDataset`.
    distributed : bool
        Whether to use a ``DistributedSampler``.
    year_start, year_end : int
        Date range for the dataset (end-exclusive).
    num_inferences : int
        Number of evenly spaced inference samples to draw (0 = all).
    train : bool
        Training mode flag — controls shuffling and return values.
    validate : bool
        If *True* (and ``train`` is *False*), load multi-step targets.

    Returns
    -------
    tuple
        ``(dataloader, dataset)``.
    """
    dataset = GetDataset(params, 
                         year_start=year_start,
                         year_end=year_end,
                         num_inferences=num_inferences,
                         train=train, 
                         validate=validate)

    num_workers = int(params["num_data_workers"])
    dataloader = DataLoader(
        dataset,
        batch_size=int(params["batch_size"]),
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return dataloader, dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GetDataset(Dataset):
    """PyTorch Dataset for atmospheric reanalysis data.

    Each sample consists of an input atmospheric state at time *t* and
    (during training) the target state at time *t + dt*.  For validation /
    inference the dataset can return multi-step target sequences for
    autoregressive rollout evaluation.

    Parameters
    ----------
    params : dict-like
        Configuration object (typically loaded from YAML) containing at least:

        - ``data_dir``: root directory with per-timestep HDF5 files
        - ``year_start``, ``year_end``: date range (end-exclusive)
        - ``train``: bool — training vs. inference mode
        - ``calendar``: calendar type string (e.g. ``'noleap'``)
        - ``timedelta_hours``: forecast step in hours
        - ``data_timedelta_hours``: temporal resolution of files in hours
        - ``has_year_zero``: bool for cftime year-zero support
        - ``surface_variables``, ``upper_air_variables``, ``diagnostic_variables``: list of variable names
        - ``constant_boundary_variables``, ``varying_boundary_variables``: list of forcing field names
        - ``forecast_lead_times``: list of lead-time steps for evaluation
        - ``levels``: pressure levels to use
        - ``horizontal_resolution``: ``(nlat, nlon)``
        - ``num_inferences``: number of evenly-spaced inference starts (0 = all)
        - ``epsilon_factor``: input noise scale (0 disables noise)
        - ``predict_delta``: if True, targets are state increments
        - ``mean_path``, ``std_path``: paths to NetCDF files with normalization stats
    train : bool
        If True, return (input, target) pairs for training. 
    validate : bool
        If True and not training, load full target sequences.
    """

    def __init__(self, params: dict, 
                 year_start: int, 
                 year_end: int, 
                 num_inferences: int = 0,
                 train: bool = True, 
                 validate: bool = False):
        self.params = params
        self.data_dir = params['data_dir']
        self.train = train
        self.num_inferences = num_inferences
        self.has_year_zero = params['has_year_zero']
        self.epsilon_factor = params['epsilon_factor']
        self.diagnostic_input = params.get('diagnostic_input', False) # whether to use diagnostic as prognostic
        self.validate = validate if not self.train else False
        self.autoencoder = params.get('autoencoder', False)
        self.return_calendar = params.get('return_calendar', False)
        self.multistep_rollout = int(params.get('multistep_rollout', 1))
        self.smooth_nan_boundaries = params.get('smooth_nan_boundaries', False)
        self.smooth_sigma = float(params.get('smooth_sigma', 1.5))
        self.smooth_kernel_size = int(params.get('smooth_kernel_size', 9))
        self.smooth_n_iters = int(params.get('smooth_n_iters', 10))

        if not self.train and not self.params['forecast_lead_times']:
            self.params['forecast_lead_times'] = [1]

        self.mask_fill = params.get('mask_fill', {
            'land_sea_mask': 0.,
            'sea_surface_temperature': 270.,
            'sea_ice_cover': 0.,
        })

        # Calendar / time setup
        self.year_start = year_start
        self.year_end = year_end
        self.calendar = params["calendar"]
        self.timedelta_hours = params["timedelta_hours"]
        self.data_timedelta_hours = params["data_timedelta_hours"]
        self.datetime_class = CALENDAR_TO_DATETIME[self.calendar]

        days, hours = divmod(self.timedelta_hours, 24)
        self.timedelta = (
            self.datetime_class(1, 1, 1 + days, hour=hours)
            - self.datetime_class(1, 1, 1, hour=0)
        )

        # Variable lists
        self.surface_variables = params["surface_variables"]

        # disabled
        self.land_variables = []
        self.ocean_variables = []

        if self.land_variables:
            if any(v in self.surface_variables for v in self.land_variables):
                raise ValueError('land variables cannot be in surface variables.')
            self.surface_variables = self.surface_variables + self.land_variables

        if self.ocean_variables:
            if any(v in self.surface_variables for v in self.ocean_variables):
                raise ValueError('ocean variables cannot be in surface variables.')
            self.surface_variables = self.surface_variables + self.ocean_variables

        self.upper_air_variables = params["upper_air_variables"]
        self.constant_boundary_variables = params["constant_boundary_variables"]
        self.varying_boundary_variables = params["varying_boundary_variables"]
        self.diagnostic_variables = params['diagnostic_variables']

        # Optional: compute delta (current minus N hours prior) for a subset of
        # surface/forcing fields and append them as additional varying-boundary
        # channels. Used to give the model a "recent change" signal for slowly
        # varying forcings like SST, SIC, DSWRFtoa.
        self.delta_boundary_variables = params.get('delta_boundary_variables', [])
        self.delta_boundary_hours = params.get('delta_boundary_hours', 720)

        # Date range
        self.dates, self.start_date, self.end_date = self._get_dates(
            hour_step=params["data_timedelta_hours"]
        )

        # Constant boundary fields (e.g. land-sea mask, orography)
        if len(self.constant_boundary_variables) > 0:
            self.constant_boundary_data, self.land_mask = self._load_constant_boundary_data()
            if torch.any(torch.isnan(self.constant_boundary_data)):
                raise ValueError('Constant boundary data contains NaN values.')
            self.use_boundary = True
        else:
            self.use_boundary = False

        # Inference index selection
        max_step_horizon = max(max(self.params['forecast_lead_times']), self.multistep_rollout)
        max_inference_idx = (
            len(self.dates)
            - max_step_horizon * self.timedelta_hours // self.data_timedelta_hours
        )
        if self.num_inferences > 0:
            self.inference_idxs = np.linspace(0, max_inference_idx, num=self.num_inferences + 1, dtype=int)
        else:
            self.inference_idxs = np.arange(0, max_inference_idx)

        # Optional solstice biasing: build an oversampled index pool that
        # repeats timesteps near Jun 21 / Dec 21. Only applied in training mode
        # — validation/inference always use ``self.inference_idxs`` as-is.
        self._sample_order = None
        if self.train and params.get('solstice_bias', False):
            self._sample_order = self._build_solstice_sample_order()

        # Pressure levels
        if len(params['levels']) > 0:
            self.levels = np.array(params['levels'])
        else:
            raise ValueError('levels must be explicitly specified in config file.')

        # Load normalization statistics
        mean_path = params["mean_path"]
        std_path = params["std_path"]
        self.surface_mean, self.surface_std = self._load_mean_std(
            mean_path,
            std_path,
            self.surface_variables, upper_air=False,
        )
        self.upper_air_mean, self.upper_air_std = self._load_mean_std(
            mean_path,
            std_path,
            self.upper_air_variables,
        )

        if self.params['predict_delta']:
            _, self.surface_delta_std = self._load_mean_std(
                mean_path,
                std_path,
                self.surface_variables, upper_air=False,
            )
            _, self.upper_air_delta_std = self._load_mean_std(
                mean_path,
                std_path,
                self.upper_air_variables,
            )

        if self.use_boundary:
            self.varying_boundary_mean, self.varying_boundary_std = self._load_mean_std(
                mean_path,
                std_path,
                self.varying_boundary_variables, upper_air=False,
            )

        if self.diagnostic_variables:
            self.diagnostic_mean, self.diagnostic_std = self._load_mean_std(
                mean_path,
                std_path,
                self.diagnostic_variables, upper_air=False,
            )

        if self.delta_boundary_variables:
            # Use each variable's own std to scale the raw difference; delta is
            # approximately zero-mean so no mean subtraction is applied.
            _, self.delta_boundary_std = self._load_mean_std(
                mean_path,
                std_path,
                self.delta_boundary_variables, upper_air=False,
            )

        self._build_variable_lists()

        if self.epsilon_factor > 0.:
            torch.manual_seed(0)

        self.print_info()

    def print_info(self):
        print(f"Dataset info:")
        print(f"  Date range: {self.start_date} to {self.end_date} ({len(self.dates)} total hours)")
        print(f"  Number of inference samples: {len(self.inference_idxs)}")
        print(f"  Upper-air variables: {self.upper_air_variables}")
        print(f"  Surface variables: {self.surface_variables}")
        print(f"  Diagnostic variables: {self.diagnostic_variables}")
        print(f"  Varying boundary variables: {self.varying_boundary_variables}")
        if self.delta_boundary_variables:
            print(f"  Delta boundary variables: {self.delta_boundary_variables} "
                  f"(appended to varying boundary; {self.delta_boundary_hours}h prior)")
        print(f"  Constant boundary variables: {self.constant_boundary_variables}")
        print(f"  Pressure levels: {self.levels}")
        print(f"  Horizontal resolution: {self.params['horizontal_resolution']}")
        print(f"  Forecast lead times (hours): {self.params['forecast_lead_times']}")
        print(f"  Diagnostic input: {self.diagnostic_input}")

    # ------------------------------------------------------------------
    # Variable list bookkeeping
    # ------------------------------------------------------------------

    def _build_variable_lists(self, level_units='.0'):
        """Build ordered variable name lists for input and output tensors.

        Sets ``self.variable_list_in`` and ``self.variable_list_out`` as well
        as ``self.upper_air_len`` (the number of upper-air channels after
        flattening variables x levels).
        """
        self.variable_list_out = []
        for variable, level in product(self.upper_air_variables, self.levels):
            self.variable_list_out.append(f'{variable}_{int(level)}{level_units}')
        self.upper_air_len = len(self.variable_list_out)
        self.variable_list_out.extend(self.surface_variables)
        self.variable_list_in = self.variable_list_out.copy()
        self.variable_list_out.extend(self.diagnostic_variables)

        if self.use_boundary:
            self.variable_list_in.extend(self.varying_boundary_variables)

        if self.diagnostic_input: # add diagnostic variables to input list if configured
            self.variable_list_in.extend(self.diagnostic_variables)

    # ------------------------------------------------------------------
    # Reshaping / masking
    # ------------------------------------------------------------------

    def _reshape_and_mask_variables(self, data_array, out=False):
        """Reshape a flat channel array into (upper_air, surface, [extra]) tensors.

        For **input** (``out=False``), the extra tensor is ``varying_boundary``.
        For **output** (``out=True``), the extra tensor is ``diagnostic``.

        NaN values in surface / boundary / diagnostic fields are filled using
        ``self.mask_fill``.

        Parameters
        ----------
        data_array : np.ndarray
            Shape ``(n_channels, nlat, nlon)`` in the order defined by
            ``variable_list_in`` (input) or ``variable_list_out`` (output).
        out : bool
            Whether this is an output (target) array.

        Returns
        -------
        tuple of torch.Tensor
            ``(upper_air, surface)`` or ``(upper_air, surface, extra)``
        """
        nlat, nlon = self.params['horizontal_resolution']
        n_ua = len(self.upper_air_variables)
        n_lev = len(self.levels)
        n_sfc = len(self.surface_variables)

        upper_air = torch.tensor(
            data_array[:self.upper_air_len].reshape(n_ua, n_lev, nlat, nlon)
        ).to(torch.float32)

        surface = torch.tensor(
            data_array[self.upper_air_len:self.upper_air_len + n_sfc].reshape(n_sfc, nlat, nlon)
        ).to(torch.float32)
        surface = self._fill_mask(surface, self.surface_variables,
                                  self.land_variables + self.ocean_variables)

        offset = self.upper_air_len + n_sfc

        if out:
            if self.diagnostic_variables:
                n_diag = len(self.diagnostic_variables)
                diagnostic = torch.tensor(
                    data_array[offset:offset + n_diag].reshape(n_diag, nlat, nlon)
                ).to(torch.float32)
                diagnostic = self._fill_mask(diagnostic, self.diagnostic_variables)
                return upper_air, surface, diagnostic
            return upper_air, surface
        else:
            if self.varying_boundary_variables:
                n_bnd = len(self.varying_boundary_variables)
                varying_boundary = torch.tensor(
                    data_array[offset:offset + n_bnd].reshape(n_bnd, nlat, nlon)
                ).to(torch.float32)
                varying_boundary = self._fill_mask(varying_boundary, self.varying_boundary_variables)

                if self.diagnostic_input:
                    offset += n_bnd
                    n_diag = len(self.diagnostic_variables)
                    diagnostic = torch.tensor(
                        data_array[offset:offset + n_diag].reshape(n_diag, nlat, nlon)
                    ).to(torch.float32)
                    diagnostic = self._fill_mask(diagnostic, self.diagnostic_variables)
                    return upper_air, surface, diagnostic, varying_boundary
                else:
                    return upper_air, surface, varying_boundary
            else:
                if self.diagnostic_input:
                    n_diag = len(self.diagnostic_variables)
                    diagnostic = torch.tensor(
                        data_array[offset:offset + n_diag].reshape(n_diag, nlat, nlon)
                    ).to(torch.float32)
                    diagnostic = self._fill_mask(diagnostic, self.diagnostic_variables)
                    return upper_air, surface, diagnostic
                else:
                    return upper_air, surface

    def _fill_mask(self, data, variables, optional_variables=None):
        """Replace NaN values with predefined fill values from ``self.mask_fill``.

        Parameters
        ----------
        data : torch.Tensor
            Shape ``(n_vars, nlat, nlon)``.
        variables : list[str]
            Variable names corresponding to the first dimension.
        optional_variables : list[str] or None
            If provided, only fill NaNs for variables in this subset.
        """
        for i, var in enumerate(variables):
            if optional_variables and var not in optional_variables:
                continue
            nans = torch.isnan(data[i])
            if torch.any(nans):
                fill_val = self.mask_fill[var]

                if self.smooth_nan_boundaries:
                    # smooth_masked_boundary requires data == 0 outside the mask
                    # and fades extended values toward 0 with distance. Center
                    # on fill_val so the far field equals fill_val and the
                    # blend runs from the true boundary value down to fill_val.
                    mask = (~nans).to(torch.float32)
                    centered = torch.where(
                        nans, torch.zeros_like(data[i]), data[i] - fill_val,
                    )
                    smoothed = smooth_masked_boundary(
                        centered, mask,
                        sigma=self.smooth_sigma,
                        kernel_size=self.smooth_kernel_size,
                        n_iters=self.smooth_n_iters,
                        lon_circular=True,
                    )
                    data[i] = smoothed + fill_val
                else:
                    data[i] = data[i].masked_fill(nans, fill_val)

        return data

    # ------------------------------------------------------------------
    # Date handling
    # ------------------------------------------------------------------

    def _get_dates(self, hour_step=6.):
        """Generate an array of hour-offsets from ``year_start`` to ``year_end``.

        Returns
        -------
        tuple
            ``(date_offsets, start_date, end_date)`` where *date_offsets* is a
            1-D numpy array of hours since *start_date*.
        """
        start_date = self.datetime_class(self.year_start, 1, 1)
        end_date = self.datetime_class(self.year_end, 1, 1)
        hours = (end_date - start_date).days * 24.
        date_range = np.arange(0., hours, hour_step)
        return date_range, start_date, end_date

    def _date_offset(self, index):
        """Return the hour-offset into ``self.dates`` for dataloader position *index*.

        When solstice biasing is active, indirects through ``self._sample_order``
        so that repeated (oversampled) positions map to the correct timestep.
        """
        if self._sample_order is not None:
            return self.dates[self._sample_order[index]]
        return self.dates[index]

    def _build_solstice_sample_order(self):
        """Build an oversampled index pool biased toward Jun 21 / Dec 21.

        For each index in ``self.inference_idxs``, compute a Gaussian weight
        based on the circular day-of-year distance to the nearest solstice and
        repeat that index ``round(weight)`` times (minimum 1). The resulting
        array is a valid index list into ``self.dates`` whose empirical
        distribution is concentrated around the solstices.
        """
        sigma_days = float(self.params.get('solstice_bias_sigma_days', 60.0))
        peak = float(self.params.get('solstice_bias_peak_multiplier', 2.0))
        if peak < 1.0:
            raise ValueError('solstice_bias_peak_multiplier must be >= 1.0')

        ref_year = self.year_start
        yr_start = self.datetime_class(ref_year, 1, 1, has_year_zero=self.has_year_zero)
        jun_doy = (self.datetime_class(ref_year, 6, 21, has_year_zero=self.has_year_zero) - yr_start).days
        dec_doy = (self.datetime_class(ref_year, 12, 21, has_year_zero=self.has_year_zero) - yr_start).days
        year_len = (self.datetime_class(ref_year + 1, 1, 1, has_year_zero=self.has_year_zero) - yr_start).days

        order = []
        near_solstice = 0
        for idx in self.inference_idxs:
            sample_time = self.start_date + timedelta(hours=float(self.dates[idx]))
            sample_yr_start = self.datetime_class(
                sample_time.year, 1, 1, has_year_zero=self.has_year_zero,
            )
            doy = (sample_time - sample_yr_start).total_seconds() / 86400.0

            d_jun = abs(doy - jun_doy); d_jun = min(d_jun, year_len - d_jun)
            d_dec = abs(doy - dec_doy); d_dec = min(d_dec, year_len - d_dec)
            d = min(d_jun, d_dec)

            weight = 1.0 + (peak - 1.0) * np.exp(-0.5 * (d / sigma_days) ** 2)
            repeats = max(1, int(round(weight)))
            order.extend([int(idx)] * repeats)
            if d <= sigma_days:
                near_solstice += 1

        order = np.array(order, dtype=np.int64)
        print(
            f"  Solstice biasing enabled: sigma={sigma_days}d, peak={peak}x, "
            f"effective epoch size {len(order)} vs base {len(self.inference_idxs)} "
            f"({near_solstice} base samples within 1 sigma of a solstice)"
        )
        return order

    # ------------------------------------------------------------------
    # Data I/O
    # ------------------------------------------------------------------

    def _get_data(self, data_datetime, out=False, variable_list=None):
        """Load raw data for a single datetime from disk.

        Parameters
        ----------
        data_datetime : cftime datetime
            Timestamp to load.
        out : bool
            If True and *variable_list* is None, use output variable list.
        variable_list : list[str] or None
            Explicit variable list override.

        Returns
        -------
        np.ndarray
            Shape ``(n_channels, ...)``.
        """
        data_year = data_datetime.year
        seconds_into_year = int(
            (data_datetime - self.datetime_class(data_year, 1, 1, hour=0,
                                                  has_year_zero=self.has_year_zero)).total_seconds()
        )
        data_idx = seconds_into_year // 3600 // self.data_timedelta_hours
        data_file_path = get_out_path(self.data_dir, data_year, data_idx)

        if variable_list:
            return get_data_given_path(data_file_path, variable_list)
        if out:
            return get_data_given_path(data_file_path, self.variable_list_out)
        return get_data_given_path(data_file_path, self.variable_list_in)

    def _compute_calendar(self, time):
        # Returns (second_of_day, day_of_year). DOY is 1-indexed.
        data_year = time.year
        seconds_into_year = int(
            (time - self.datetime_class(data_year, 1, 1, hour=0,
                                        has_year_zero=self.has_year_zero)).total_seconds()
        )
        doy = (seconds_into_year // 86400) + 1
        sod = seconds_into_year % 86400
        return sod, doy

    def _get_delta_prior_datetime(self, current_time):
        # Wrap forward by a full year when the prior timestep would fall before
        # start_date so the first 30 days of the dataset use the last 30 days
        # of year_start as a reference.
        prior_time = current_time - timedelta(hours=self.delta_boundary_hours)
        if prior_time < self.start_date:
            days_in_year = (
                self.datetime_class(self.year_start + 1, 1, 1, has_year_zero=self.has_year_zero)
                - self.datetime_class(self.year_start, 1, 1, has_year_zero=self.has_year_zero)
            ).days
            prior_time = prior_time + timedelta(days=days_in_year)
        return prior_time

    def _compute_delta_boundary(self, current_time):
        current_raw = torch.tensor(
            self._get_data(current_time, variable_list=self.delta_boundary_variables)
        ).to(torch.float32)
        current_raw = self._fill_mask(current_raw, self.delta_boundary_variables)

        prior_time = self._get_delta_prior_datetime(current_time)
        prior_raw = torch.tensor(
            self._get_data(prior_time, variable_list=self.delta_boundary_variables)
        ).to(torch.float32)
        prior_raw = self._fill_mask(prior_raw, self.delta_boundary_variables)

        delta = current_raw - prior_raw
        return delta / self.delta_boundary_std.reshape(-1, 1, 1)

    def _load_constant_boundary_data(self):
        """Load and normalize constant boundary fields (e.g. land-sea mask).

        Returns
        -------
        tuple
            ``(constant_boundary_data, land_mask)`` both as float32 tensors.
        """
        raw = torch.tensor(
            self._get_data(self.start_date, variable_list=self.constant_boundary_variables)
        ).to(torch.float32)
        raw = self._fill_mask(raw, self.constant_boundary_variables)

        if self.smooth_nan_boundaries and 'land_sea_mask' in self.constant_boundary_variables:
            # Soft coastline: keep land = 1 exactly, fade 1 -> 0 over the sea
            # using the same Dirichlet diffusion as the masked-field smoothing.
            lsm_idx = self.constant_boundary_variables.index('land_sea_mask')
            lsm = raw[lsm_idx]
            raw[lsm_idx] = smooth_masked_boundary(
                lsm, (lsm > 0.5).to(torch.float32),
                sigma=self.smooth_sigma,
                kernel_size=3,
                n_iters=self.smooth_n_iters,
                lon_circular=True,
            )

        land_mask = raw[(np.array(self.constant_boundary_variables) == 'land_sea_mask').tolist()].clone().detach()
        mean = torch.mean(raw, dim=(1, 2))
        std = torch.std(raw, dim=(1, 2))
        normalized = (raw - mean.reshape(-1, 1, 1)) / std.reshape(-1, 1, 1)
        return normalized, land_mask

    # ------------------------------------------------------------------
    # Normalization statistics
    # ------------------------------------------------------------------

    def _load_mean_std(self, mean_file, std_file, datavars, upper_air=True):
        """Load mean and standard deviation tensors from NetCDF files.

        Parameters
        ----------
        mean_file, std_file : str
            Paths to NetCDF files containing per-variable statistics.
        datavars : list[str]
            Variable names to extract.
        upper_air : bool
            If True, select only the configured pressure levels along the
            vertical (``level``) dimension.

        Returns
        -------
        tuple
            ``(mean, std)`` tensors.
        """
        if upper_air:
            with xr.open_dataset(mean_file, engine = "h5netcdf") as ds:
                level_mask = xr.DataArray(
                    data=[lev in self.levels for lev in ds['level'].values], dims=['level']
                )
                mean = torch.stack([
                    torch.tensor(ds[var].where(level_mask, drop=True).values).to(torch.float32)
                    for var in datavars
                ], dim=0)
            with xr.open_dataset(std_file, engine = "h5netcdf") as ds:
                level_mask = xr.DataArray(
                    data=[lev in self.levels for lev in ds['level'].values], dims=['level']
                )
                std = torch.stack([
                    torch.tensor(ds[var].where(level_mask, drop=True).values).to(torch.float32)
                    for var in datavars
                ], dim=0)
        else:
            with xr.open_dataset(mean_file, engine = "h5netcdf") as ds:
                mean = torch.stack([
                    torch.tensor(ds[var].values).to(torch.float32) for var in datavars
                ], dim=0)
            with xr.open_dataset(std_file, engine = "h5netcdf") as ds:
                std = torch.stack([
                    torch.tensor(ds[var].values).to(torch.float32) for var in datavars
                ], dim=0)
        return mean, std

    # ------------------------------------------------------------------
    # Transforms (normalize / denormalize)
    # ------------------------------------------------------------------

    def surface_transform(self, data):
        """Normalize surface fields: ``(x - mean) / std``."""
        device=data.device
        return (data - self.surface_mean.reshape(-1, 1, 1).to(device)) / self.surface_std.reshape(-1, 1, 1).to(device)

    def diagnostic_transform(self, data):
        """Normalize diagnostic fields."""
        device=data.device
        return (data - self.diagnostic_mean.reshape(-1, 1, 1).to(device)) / self.diagnostic_std.reshape(-1, 1, 1).to(device)

    def boundary_transform(self, data):
        """Normalize varying boundary fields."""
        device=data.device
        return (data - self.varying_boundary_mean.reshape(-1, 1, 1).to(device)) / self.varying_boundary_std.reshape(-1, 1, 1).to(device)

    def upper_air_transform(self, data):
        """Normalize upper-air fields (shape: ``(n_vars, n_levels, nlat, nlon)``)."""
        n = len(self.upper_air_variables)
        device=data.device
        return (data - self.upper_air_mean.reshape(n, -1, 1, 1).to(device)) / self.upper_air_std.reshape(n, -1, 1, 1).to(device)

    def surface_inv_transform(self, data):
        """Denormalize surface fields (expects leading batch dim)."""
        device=data.device
        return data * self.surface_std.reshape(1, -1, 1, 1).to(device) + self.surface_mean.reshape(1, -1, 1, 1).to(device)

    def upper_air_inv_transform(self, data):
        """Denormalize upper-air fields (expects leading batch dim)."""
        n = len(self.upper_air_variables)
        device=data.device
        return data * self.upper_air_std.reshape(1, n, -1, 1, 1).to(device) + self.upper_air_mean.reshape(1, n, -1, 1, 1).to(device)

    def diagnostic_inv_transform(self, data):
        """Denormalize diagnostic fields (expects leading batch dim)."""
        device=data.device
        return data * self.diagnostic_std.reshape(1, -1, 1, 1).to(device) + self.diagnostic_mean.reshape(1, -1, 1, 1).to(device)

    def surface_delta_transform(self, data):
        """Normalize surface increments (zero-mean assumed)."""
        device=data.device
        return data / self.surface_delta_std.reshape(-1, 1, 1).to(device)

    def upper_air_delta_transform(self, data):
        """Normalize upper-air increments (zero-mean assumed)."""
        n = len(self.upper_air_variables)
        device=data.device
        return data / self.upper_air_delta_std.reshape(n, -1, 1, 1).to(device)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        if self._sample_order is not None:
            return len(self._sample_order)
        return len(self.inference_idxs)

    def __getitem__(self, index):
        """Return a sample for training, validation, or inference.

        Returns
        -------
        tuple of torch.Tensor
            The exact contents depend on mode:

            **Training** (``self.train``):
              ``(surface_t, upper_air_t, surface_t1, upper_air_t1,
              [diagnostic_t1,] varying_boundary)``

            **Validation** (``self.validate`` and ``forecast_lead_times``):
              ``(surface_t, upper_air_t, targets_surface, targets_upper_air,
              [targets_diagnostic,] [targets_delta_surface, targets_delta_upper_air,]
              varying_boundary, start_time_tensor)``

            **Inference** (``forecast_lead_times`` without validate):
              ``(surface_t, upper_air_t, varying_boundary)``

            **Single-step eval** (no ``forecast_lead_times``):
              Same as training format.
        """
        lead_times = self.params['forecast_lead_times']
        has_boundary = len(self.varying_boundary_variables) > 0
        has_diagnostic = len(self.diagnostic_variables) > 0

        # ---- Training ----
        if self.train:
            if self.multistep_rollout > 1 and not self.autoencoder:
                return self._getitem_train_multistep(index, has_boundary, has_diagnostic)
            return self._getitem_train(index, has_boundary, has_diagnostic)

        # ---- Autoregressive inference / validation ----
        if lead_times:
            return self._getitem_autoregressive(index, lead_times, has_boundary, has_diagnostic)

        # ---- Single-step evaluation ----
        return self._getitem_single_step(index, has_boundary)

    def _getitem_train(self, index, has_boundary, has_diagnostic):
        """Build a single training sample (input at t, target at t+dt)."""
        hour_offset = self._date_offset(index)
        start_time = self.start_date + timedelta(hours=hour_offset)
        end_time = self.start_date + timedelta(hours=hour_offset + self.timedelta_hours)

        data_in = self._get_data(start_time, out=False)

        if has_boundary:
            if self.diagnostic_input:
                upper_air_t, surface_t, diagnostic_t, varying_boundary_data = self._reshape_and_mask_variables(data_in, out=False)
            else:
                upper_air_t, surface_t, varying_boundary_data = self._reshape_and_mask_variables(data_in, out=False)
        else:
            if self.diagnostic_input:
                upper_air_t, surface_t, diagnostic_t = self._reshape_and_mask_variables(data_in, out=False)
            else:
                upper_air_t, surface_t = self._reshape_and_mask_variables(data_in, out=False)

        if self.autoencoder: # assume using diagnostics
            
            surface_t = self.surface_transform(surface_t)
            upper_air_t = self.upper_air_transform(upper_air_t)
            diagnostic_t = self.diagnostic_transform(diagnostic_t)
            return surface_t, upper_air_t, diagnostic_t

        data_out = self._get_data(end_time, out=True)

        if has_diagnostic:
            upper_air_t1, surface_t1, diagnostic_t1 = self._reshape_and_mask_variables(data_out, out=True)
        else:
            upper_air_t1, surface_t1 = self._reshape_and_mask_variables(data_out, out=True)

        # Normalize
        if self.params['predict_delta']:
            surface_t1 = self.surface_delta_transform(surface_t1 - surface_t)
            upper_air_t1 = self.upper_air_delta_transform(upper_air_t1 - upper_air_t)
            surface_t = self.surface_transform(surface_t)
            upper_air_t = self.upper_air_transform(upper_air_t)
        else:
            surface_t = self.surface_transform(surface_t)
            surface_t1 = self.surface_transform(surface_t1)
            upper_air_t = self.upper_air_transform(upper_air_t)
            upper_air_t1 = self.upper_air_transform(upper_air_t1)

        if has_diagnostic:
            diagnostic_t1 = self.diagnostic_transform(diagnostic_t1)
        if has_boundary:
            varying_boundary_data = self.boundary_transform(varying_boundary_data)
            if self.delta_boundary_variables:
                varying_boundary_data = torch.cat(
                    [varying_boundary_data, self._compute_delta_boundary(start_time)], dim=0,
                )
        if self.diagnostic_input:
            diagnostic_t = self.diagnostic_transform(diagnostic_t)

        # Optional input noise
        if self.epsilon_factor > 0.:
            surface_t = self._add_input_noise(surface_t)
            upper_air_t = self._add_input_noise(upper_air_t)

        self._check_nans(surface_t=surface_t, upper_air_t=upper_air_t,
                         varying_boundary_data=varying_boundary_data if has_boundary else None,
                         surface_t1=surface_t1, upper_air_t1=upper_air_t1,
                         diagnostic_t1=diagnostic_t1 if has_diagnostic else None,
                         diagnostic_t=diagnostic_t if self.diagnostic_input else None)


        if self.return_calendar:
            sod, doy = self._compute_calendar(start_time)

            co2 = varying_boundary_data[0, 0, 0] # c nlat nlon -> 1
            varying_boundary_data = varying_boundary_data[1:, :, :] # remove co2 from boundary data and return separately
            calendar = torch.tensor([sod, doy, co2], dtype=torch.float32)

            # assume full diagnostic input for now if return_calendar is True
            return surface_t, upper_air_t, diagnostic_t, surface_t1, upper_air_t1, diagnostic_t1, varying_boundary_data, calendar
        
        if self.diagnostic_input:
            return surface_t, upper_air_t, diagnostic_t, surface_t1, upper_air_t1, diagnostic_t1, varying_boundary_data
        if has_diagnostic:
            return surface_t, upper_air_t, surface_t1, upper_air_t1, diagnostic_t1, varying_boundary_data
        return surface_t, upper_air_t, surface_t1, upper_air_t1, varying_boundary_data

    def _getitem_train_multistep(self, index, has_boundary, has_diagnostic):
        """Build a multi-step training sample.

        Returns the initial state, a stack of varying-boundary forcings for each
        rollout step, and a single final target at ``t + rollout * dt``.
        Only the final timestep is used as a supervision label — intermediate
        states are produced autoregressively by the model.
        """
        rollout = self.multistep_rollout
        start_time = self.start_date + timedelta(hours=self._date_offset(index))

        data_in = self._get_data(start_time, out=False)
        if has_boundary:
            if self.diagnostic_input:
                upper_air_t, surface_t, diagnostic_t, varying_boundary_t = self._reshape_and_mask_variables(data_in, out=False)
            else:
                upper_air_t, surface_t, varying_boundary_t = self._reshape_and_mask_variables(data_in, out=False)
        else:
            if self.diagnostic_input:
                upper_air_t, surface_t, diagnostic_t = self._reshape_and_mask_variables(data_in, out=False)
            else:
                upper_air_t, surface_t = self._reshape_and_mask_variables(data_in, out=False)
            varying_boundary_t = None

        # Stack per-step boundary forcings for each of the `rollout` steps the
        # model will roll out over (first one already loaded with the inputs).
        if has_boundary:
            boundary_list = [varying_boundary_t]
            step_times = [start_time]
            for step in range(1, rollout):
                bnd_time = start_time + timedelta(hours=self.timedelta_hours * step)
                step_times.append(bnd_time)
                bnd_raw = torch.tensor(
                    self._get_data(bnd_time, variable_list=self.varying_boundary_variables)
                ).to(torch.float32)
                boundary_list.append(self._fill_mask(bnd_raw, self.varying_boundary_variables))
            varying_boundary_data = torch.stack(
                [self.boundary_transform(b) for b in boundary_list], dim=0
            )  # (rollout, c, nlat, nlon)
            if self.delta_boundary_variables:
                delta_stack = torch.stack(
                    [self._compute_delta_boundary(t) for t in step_times], dim=0
                )
                varying_boundary_data = torch.cat([varying_boundary_data, delta_stack], dim=1)
        else:
            varying_boundary_data = None

        # Final target only — at t + rollout * dt
        end_time = start_time + timedelta(hours=self.timedelta_hours * rollout)
        data_out = self._get_data(end_time, out=True)
        if has_diagnostic:
            upper_air_t1, surface_t1, diagnostic_t1 = self._reshape_and_mask_variables(data_out, out=True)
        else:
            upper_air_t1, surface_t1 = self._reshape_and_mask_variables(data_out, out=True)

        if self.params['predict_delta']:
            # delta is relative to the immediately preceding (rollout-1) target; we
            # don't know it without loading it, so disallow delta + multistep.
            raise NotImplementedError("predict_delta is not supported with multistep_rollout > 1")

        surface_t = self.surface_transform(surface_t)
        surface_t1 = self.surface_transform(surface_t1)
        upper_air_t = self.upper_air_transform(upper_air_t)
        upper_air_t1 = self.upper_air_transform(upper_air_t1)

        if has_diagnostic:
            diagnostic_t1 = self.diagnostic_transform(diagnostic_t1)
        if self.diagnostic_input:
            diagnostic_t = self.diagnostic_transform(diagnostic_t)

        if self.epsilon_factor > 0.:
            surface_t = self._add_input_noise(surface_t)
            upper_air_t = self._add_input_noise(upper_air_t)

        self._check_nans(surface_t=surface_t, upper_air_t=upper_air_t,
                         varying_boundary_data=varying_boundary_data if has_boundary else None,
                         surface_t1=surface_t1, upper_air_t1=upper_air_t1,
                         diagnostic_t1=diagnostic_t1 if has_diagnostic else None,
                         diagnostic_t=diagnostic_t if self.diagnostic_input else None)

        if self.return_calendar:
            # Per-step calendar (T, 3): [sod, doy, co2]. Channel 0 is assumed
            # to be global_mean_co2 (must be first in varying_boundary_variables).
            co2_per_step = varying_boundary_data[:, 0, 0, 0].clone()
            varying_boundary_data = varying_boundary_data[:, 1:, :, :]
            cal = torch.empty(len(step_times), 3, dtype=torch.float32)
            for i, t in enumerate(step_times):
                sod, doy = self._compute_calendar(t)
                cal[i, 0] = sod
                cal[i, 1] = doy
            cal[:, 2] = co2_per_step
            return surface_t, upper_air_t, diagnostic_t, surface_t1, upper_air_t1, diagnostic_t1, varying_boundary_data, cal

        if self.diagnostic_input:
            return surface_t, upper_air_t, diagnostic_t, surface_t1, upper_air_t1, diagnostic_t1, varying_boundary_data
        if has_diagnostic:
            return surface_t, upper_air_t, surface_t1, upper_air_t1, diagnostic_t1, varying_boundary_data
        return surface_t, upper_air_t, surface_t1, upper_air_t1, varying_boundary_data

    def _getitem_autoregressive(self, index, lead_times, has_boundary, has_diagnostic):
        """Build an autoregressive sample with multi-step boundary forcing."""
        start_time = self.start_date + timedelta(hours=self._date_offset(index))
        data_in = self._get_data(start_time, out=False)

        if has_boundary:
            if self.diagnostic_input:
                upper_air_t, surface_t, diagnostic_t, varying_boundary_t = self._reshape_and_mask_variables(data_in, out=False)
            else:
                upper_air_t, surface_t, varying_boundary_t = self._reshape_and_mask_variables(data_in, out=False)
                diagnostic_t = None
        else:
            upper_air_t, surface_t = self._reshape_and_mask_variables(data_in, out=False)

        # Load boundary forcing for all lead times
        max_lead_time = lead_times[-1]
        start_time_tensor = torch.tensor([start_time.year, start_time.month, start_time.day, start_time.hour])

        varying_boundary_data = [varying_boundary_t]
        step_times = [start_time]
        for step in range(max_lead_time):
            bnd_time = start_time + timedelta(hours=self.timedelta_hours * step)
            step_times.append(bnd_time)
            bnd_raw = torch.tensor(
                self._get_data(bnd_time, variable_list=self.varying_boundary_variables)
            ).to(torch.float32)
            varying_boundary_data.append(self._fill_mask(bnd_raw, self.varying_boundary_variables))
        varying_boundary_data = torch.stack(
            [self.boundary_transform(b) for b in varying_boundary_data], dim=0
        )
        if self.delta_boundary_variables:
            delta_stack = torch.stack(
                [self._compute_delta_boundary(t) for t in step_times], dim=0
            )
            varying_boundary_data = torch.cat([varying_boundary_data, delta_stack], dim=1)

        calendar = None
        if self.return_calendar:
            co2_per_step = varying_boundary_data[:, 0, 0, 0].clone()
            varying_boundary_data = varying_boundary_data[:, 1:, :, :]
            calendar = torch.empty(len(step_times), 3, dtype=torch.float32)
            for i, t in enumerate(step_times):
                sod, doy = self._compute_calendar(t)
                calendar[i, 0] = sod
                calendar[i, 1] = doy
            calendar[:, 2] = co2_per_step

        if self.validate:
            return self._getitem_validate(
                start_time, max_lead_time, surface_t, upper_air_t, diagnostic_t,
                varying_boundary_data, start_time_tensor, has_diagnostic, calendar,
            )

        # Inference only — return input + boundary
        surface_t = self.surface_transform(surface_t)
        upper_air_t = self.upper_air_transform(upper_air_t)
        if self.diagnostic_input:
            diagnostic_t = self.diagnostic_transform(diagnostic_t)

        self._check_nans(surface_t=surface_t, upper_air_t=upper_air_t,
                         varying_boundary_data=varying_boundary_data)

        if self.return_calendar:
            if self.diagnostic_input:
                return surface_t, upper_air_t, diagnostic_t, varying_boundary_data, calendar
            return surface_t, upper_air_t, varying_boundary_data, calendar

        if self.diagnostic_input:
            return surface_t, upper_air_t, diagnostic_t, varying_boundary_data

        return surface_t, upper_air_t, varying_boundary_data

    def _getitem_validate(self, start_time, max_lead_time, surface_t, upper_air_t, diagnostic_t,
                          varying_boundary_data, start_time_tensor, has_diagnostic, calendar=None):
        """Load multi-step targets for validation scoring."""
        targets_surface = []
        targets_upper_air = []
        targets_diagnostic = [] if has_diagnostic else None
        targets_delta_surface = [] if self.params['predict_delta'] else None
        targets_delta_upper_air = [] if self.params['predict_delta'] else None

        for step in range(1, max_lead_time + 1):
            target_time = start_time + timedelta(hours=self.timedelta_hours * step)
            raw_target = self._get_data(target_time, out=True)

            if has_diagnostic:
                ua_target, sfc_target, diag_target = self._reshape_and_mask_variables(raw_target, out=True)
                targets_diagnostic.append(diag_target)
            else:
                ua_target, sfc_target = self._reshape_and_mask_variables(raw_target, out=True)

            targets_surface.append(sfc_target)
            targets_upper_air.append(ua_target)

            if self.params['predict_delta']:
                if step == 1:
                    sfc_delta = targets_surface[-1] - surface_t
                    ua_delta = targets_upper_air[-1] - upper_air_t
                else:
                    sfc_delta = targets_surface[-1] - targets_surface[-2]
                    ua_delta = targets_upper_air[-1] - targets_upper_air[-2]
                targets_delta_surface.append(self.surface_delta_transform(sfc_delta))
                targets_delta_upper_air.append(self.upper_air_delta_transform(ua_delta))

        # Normalize all targets
        targets_surface = torch.stack([self.surface_transform(s) for s in targets_surface], dim=0)
        targets_upper_air = torch.stack([self.upper_air_transform(u) for u in targets_upper_air], dim=0)
        if has_diagnostic:
            targets_diagnostic = torch.stack([self.diagnostic_transform(d) for d in targets_diagnostic], dim=0)

        surface_t = self.surface_transform(surface_t)
        upper_air_t = self.upper_air_transform(upper_air_t)
        diagnostic_t = self.diagnostic_transform(diagnostic_t) if self.diagnostic_input else None

        self._check_nans(surface_t=surface_t, upper_air_t=upper_air_t,
                         varying_boundary_data=varying_boundary_data)

        # Build return tuple
        if diagnostic_t is not None:
            result = [surface_t, upper_air_t, diagnostic_t, targets_surface, targets_upper_air]
        else:
            result = [surface_t, upper_air_t, targets_surface, targets_upper_air]
        if has_diagnostic:
            result.append(targets_diagnostic)
        if self.params['predict_delta']:
            targets_delta_surface = torch.stack(targets_delta_surface, dim=0)
            targets_delta_upper_air = torch.stack(targets_delta_upper_air, dim=0)
            result.extend([targets_delta_surface, targets_delta_upper_air])
        result.extend([varying_boundary_data, start_time_tensor])
        if calendar is not None:
            result.append(calendar)
        return tuple(result)

    def _getitem_single_step(self, index, has_boundary):
        """Single-step evaluation without lead times."""
        start_time = self.start_date + timedelta(hours=self._date_offset(index))
        data_in = self._get_data(start_time, out=False)

        if has_boundary:
            if self.diagnostic_input:
                upper_air_t, surface_t, diagnostic_t, varying_boundary_data = self._reshape_and_mask_variables(data_in, out=False)
                diagnostic_t = self.diagnostic_transform(diagnostic_t)
            else:
                upper_air_t, surface_t, varying_boundary_data = self._reshape_and_mask_variables(data_in, out=False)
            varying_boundary_data = self.boundary_transform(varying_boundary_data)
            if self.delta_boundary_variables:
                varying_boundary_data = torch.cat(
                    [varying_boundary_data, self._compute_delta_boundary(start_time)], dim=0,
                )
            varying_boundary_data = varying_boundary_data.unsqueeze(0)
        else:
            upper_air_t, surface_t = self._reshape_and_mask_variables(data_in, out=False)

        surface_t = self.surface_transform(surface_t)
        upper_air_t = self.upper_air_transform(upper_air_t)

        self._check_nans(surface_t=surface_t, upper_air_t=upper_air_t,
                         varying_boundary_data=varying_boundary_data if has_boundary else None)

        if self.return_calendar:
            # varying_boundary_data shape here is (1, c, nlat, nlon) after the unsqueeze.
            co2 = varying_boundary_data[0, 0, 0, 0].clone()
            varying_boundary_data = varying_boundary_data[:, 1:, :, :]
            sod, doy = self._compute_calendar(start_time)
            calendar = torch.tensor([sod, doy, co2], dtype=torch.float32)
            if self.diagnostic_input:
                return surface_t, upper_air_t, diagnostic_t, varying_boundary_data, calendar
            return surface_t, upper_air_t, varying_boundary_data, calendar

        if self.diagnostic_input:
            return surface_t, upper_air_t, diagnostic_t, varying_boundary_data
        return surface_t, upper_air_t, upper_air_t, varying_boundary_data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_input_noise(self, data):
        """Add scaled Gaussian noise to input for regularization.

        Parameters
        ----------
        data : torch.Tensor
        field_type : str
            ``'surface'`` or ``'upper_air'``.
        """
        scale = self.epsilon_factor
        return data + torch.randn_like(data) * scale

    @staticmethod
    def _check_nans(**tensors):
        """Raise ValueError if any provided tensor contains NaN."""
        for name, tensor in tensors.items():
            if tensor is not None and torch.any(torch.isnan(tensor)):
                raise ValueError(f'{name} contains NaN values.')
