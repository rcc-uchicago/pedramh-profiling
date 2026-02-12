"""
Observation functions for ensemble forecast validation.

All observation functions follow a standard signature:
- First 5 arguments (required):
  1. datasets: list of xarray.Dataset objects containing forecast for each ensemble member
  2. particle_idxs: list of particle indices corresponding to initial condition indices
  3. ensemble_start: index of the first ensemble member in the forecast
  4. ensemble_end: index of the last ensemble member in the forecast (exclusive)
  5. event_type: event type identifier (str) used for file organization and metric grouping
  
- Additional arguments (function-specific) can follow
- All observation functions require save_basename as an additional argument (single file path)
  Note: Lead time is now specified in the directory path (e.g., {base_dir}/observations/{lead_time}h/),
  so it should already be included in save_basename and is not passed as a separate argument.

All arguments are passed as a single tuple for multiprocessing compatibility.
"""

import numpy as np
import xarray as xr
import json
import os
import glob
import re
from typing import List, Tuple, Union, Dict, Optional
from natsort import natsorted


def get_observation_filepath(save_basename: str, obs_function_name: str, particle_idx: int, 
                             event_type: str,
                             ensemble_start: Optional[int] = None, ensemble_end: Optional[int] = None,
                             function_specific_string: str = "") -> str:
    """
    Construct the file path for saving observation data.
    
    Note: Lead time is now specified in the directory path (e.g., {base_dir}/observations/{lead_time}h/),
    so it is no longer included in the filename.
    
    Args:
        save_basename: Base file path to append information to (should already include lead time in directory path)
        obs_function_name: Name of the observation function
        particle_idx: Particle index (initial condition index)
        event_type: Event type identifier (str) used for file organization
        ensemble_start: Index of first ensemble member (optional; omitted for truth files)
        ensemble_end: Index of last ensemble member (exclusive; optional; omitted for truth files)
        function_specific_string: Additional string specific to the observation function (e.g., region name, variable name)
    
    Returns:
        Complete file path for saving the observation data
    """
    # Extract directory and base filename from save_basename
    dir_path = os.path.dirname(save_basename) if os.path.dirname(save_basename) else ""
    base_name = os.path.basename(save_basename) if os.path.basename(save_basename) else save_basename
    
    # Construct the filename components
    # event_type is always first after base_name for easy grouping
    components = [base_name, event_type, f"particle_{particle_idx:04d}"]
    if ensemble_start is not None and ensemble_end is not None:
        components.append(f"ens_{int(ensemble_start):04d}-{int(ensemble_end):04d}")
    components.append(obs_function_name)
    
    # Add function-specific string if provided
    if function_specific_string:
        components.append(function_specific_string)
    
    # Join components with underscores
    filename = "_".join(components) + ".nc"
    
    # Combine with directory path if it exists
    if dir_path:
        filepath = os.path.join(dir_path, filename)
    else:
        filepath = filename
    
    return filepath


def _select_region_from_dataset(ds: xr.Dataset, lon_region: list, lat_region: list) -> xr.Dataset:
    """
    Select a spatial region from an xarray dataset, handling various coordinate edge cases.
    
    This function handles:
    - Longitude ranges from -180 to 180 or 0 to 360
    - Longitude coordinates in descending order (highest to lowest)
    - Latitude ranges spanning negative to positive values
    - Latitude coordinates in descending order (highest to lowest)
    
    Args:
        ds: xarray Dataset with 'lon' and 'lat' coordinates
        lon_region: List of two longitude values [min, max] or [max, min] defining the region
        lat_region: List of two latitude values [min, max] or [max, min] defining the region
    
    Returns:
        xarray Dataset with selected region
    
    Raises:
        ValueError: If required coordinates are not found in the dataset
    """
    if 'lon' not in ds.coords and 'lon' not in ds.dims:
        raise ValueError("Dataset must have 'lon' coordinate")
    if 'lat' not in ds.coords and 'lat' not in ds.dims:
        raise ValueError("Dataset must have 'lat' coordinate")
    
    # Get coordinate arrays
    lon_coords = ds.coords['lon'].values if 'lon' in ds.coords else ds['lon'].values
    lat_coords = ds.coords['lat'].values if 'lat' in ds.coords else ds['lat'].values
    
    # Determine coordinate ordering
    lon_is_descending = len(lon_coords) > 1 and lon_coords[0] > lon_coords[-1]
    lat_is_descending = len(lat_coords) > 1 and lat_coords[0] > lat_coords[-1]
    
    # Get min and max for region
    lon_min_region = min(lon_region)
    lon_max_region = max(lon_region)
    lat_min_region = min(lat_region)
    lat_max_region = max(lat_region)
    
    # Handle longitude: check if we need to handle wrap-around (0-360 vs -180 to 180)
    lon_data_min = lon_coords.min()
    lon_data_max = lon_coords.max()
    
    # Normalize longitude region to match data coordinate system
    # If data is 0-360 and region is -180 to 180, convert region to 0-360
    if lon_data_min >= 0 and lon_data_max <= 360:
        # Data is in 0-360 range
        if lon_min_region < 0:
            lon_min_region += 360
        if lon_max_region < 0:
            lon_max_region += 360
    elif lon_data_min >= -180 and lon_data_max <= 180:
        # Data is in -180 to 180 range
        if lon_min_region > 180:
            lon_min_region -= 360
        if lon_max_region > 180:
            lon_max_region -= 360
    
    # Handle longitude wrap-around case (e.g., region spans 350 to 10 degrees)
    if lon_min_region > lon_max_region:
        # Region wraps around (e.g., 350 to 10 degrees)
        # We need to select two ranges: [lon_min_region, max] and [min, lon_max_region]
        # Selection is value-based regardless of coordinate ordering
        mask_lon = (lon_coords >= lon_min_region) | (lon_coords <= lon_max_region)
    else:
        # Normal case: region is within a continuous range
        # Selection is always value-based: we want lon_min_region <= lon <= lon_max_region
        # This works regardless of whether coordinates are ascending or descending
        mask_lon = (lon_coords >= lon_min_region) & (lon_coords <= lon_max_region)
    
    # Handle latitude: selection is always value-based regardless of coordinate ordering
    # We want all values where lat_min_region <= lat <= lat_max_region
    mask_lat = (lat_coords >= lat_min_region) & (lat_coords <= lat_max_region)
    
    # Apply selection using isel with boolean indexing (more robust than slice for complex cases)
    # Get indices where masks are True
    lat_indices = np.where(mask_lat)[0]
    lon_indices = np.where(mask_lon)[0]
    
    if len(lat_indices) == 0 or len(lon_indices) == 0:
        raise ValueError(f"No data points found in the specified region (lon: {lon_region}, lat: {lat_region})")
    
    # Use isel to select by indices
    ds_selected = ds.isel(lat=lat_indices, lon=lon_indices)
    
    return ds_selected


def _get_variable_with_level(ds_region: xr.Dataset, var_name: str) -> xr.DataArray:
    """
    Get a variable from a dataset, handling level selection if specified in the variable name.
    
    Args:
        ds_region: xarray Dataset with region selected
        var_name: Variable name, optionally with level specification (e.g., "ta_50000" or "ta")
    
    Returns:
        xarray DataArray with the variable data (and level selected if specified)
    
    Raises:
        ValueError: If variable not found or level coordinate not found
    """
    # Check if variable name contains "_" and first part is not "pr"
    if "_" in var_name:
        var_parts = var_name.split("_", 1)
        base_var = var_parts[0]
        level_str = var_parts[1]
        
        # If first part is "pr", treat as single variable name
        if base_var == "pr":
            if var_name not in ds_region.data_vars:
                raise ValueError(f"Variable '{var_name}' not found in dataset")
            return ds_region[var_name]
        
        # Otherwise, treat second part as level coordinate
        try:
            level_value = float(level_str)
        except ValueError:
            raise ValueError(f"Invalid level specification '{level_str}' in variable name '{var_name}'. Expected numeric value.")
        
        # Check if base variable exists
        if base_var not in ds_region.data_vars:
            raise ValueError(f"Variable '{base_var}' not found in dataset")
        
        var_data = ds_region[base_var]
        
        # Try to select level using "plev" coordinate first
        if 'plev' in var_data.coords or 'plev' in var_data.dims:
            try:
                var_data = var_data.sel(plev=level_value, method='nearest')
                return var_data
            except (KeyError, ValueError):
                pass
        
        # Then try "lev" coordinate
        if 'lev' in var_data.coords or 'lev' in var_data.dims:
            try:
                var_data = var_data.sel(lev=level_value, method='nearest')
                return var_data
            except (KeyError, ValueError):
                pass
        
        # If neither coordinate worked, raise error
        raise ValueError(f"Could not find level coordinate ('plev' or 'lev') for variable '{base_var}' with level '{level_value}'")
    else:
        # No level specification, just return the variable
        if var_name not in ds_region.data_vars:
            raise ValueError(f"Variable '{var_name}' not found in dataset")
        return ds_region[var_name]


def unweighted_nday_mean(args: Tuple):
    """
    Compute unweighted N-day mean observable for ensemble forecasts.
    
    This function computes the mean value of a specified variable over a target duration
    at the end of the forecast, averaged over specified regions.
    
    Arguments (passed as a tuple):
    1. datasets: List of xarray.Dataset objects (Stepper output: typically one dataset per particle)
    2. particle_idxs: List of particle indices (initial condition indices)
    3. ensemble_start: Index of first ensemble member
    4. ensemble_end: Index of last ensemble member (exclusive)
    5. event_type: Event type identifier (str) used for file organization
    6. target_duration: Number of days to average over (int)
    7. var: Variable name(s) to compute observable for (str or list of str). 
           If variable name contains "_" and first part is not "pr", 
           the second part is treated as a level coordinate (e.g., "ta_50000" selects ta at 50000 Pa).
    8. regions: List of region names to compute observable for (list of str)
    9. region_file_path: Path to JSON file containing region boundaries (str)
    10. save_basename: Base file path for saving results (str, required)
           Note: Lead time is now specified in the directory path (e.g., {base_dir}/observations/{lead_time}h/),
           so it should already be included in save_basename.
    
    Returns:
        None (saves results to files)
    """
    # Unpack required arguments
    datasets = args[0]
    particle_idxs = args[1]
    ensemble_start = args[2]
    ensemble_end = args[3]
    event_type = args[4]  # Required: event type identifier

    # Accept both:
    # - truth mode: [datasets, particle_idxs, ensemble_start, ensemble_end, event_type, target_duration, var, regions, region_file_path, save_basename] (10 args)
    # - forecast mode: [datasets, particle_idxs, ensemble_start, ensemble_end, event_type, lead_time_hours, target_duration, var, regions, region_file_path, save_basename] (11 args)
    #   Note: lead_time_hours is still accepted for backward compatibility but is ignored (lead time is in directory path)
    if len(args) == 10:
        # Truth mode: no lead_time_hours
        target_duration = args[5]
        var = args[6]
        regions = args[7]
        region_file_path = args[8]
        save_basename = args[9]
    elif len(args) >= 11:
        # Forecast mode: lead_time_hours may be present for backward compatibility but is ignored
        # Lead time is now extracted from directory path if needed
        target_duration = args[6]
        var = args[7]
        regions = args[8]
        region_file_path = args[9]
        save_basename = args[10]
    else:
        raise ValueError(f"unweighted_nday_mean requires at least 10 arguments (truth mode) or 11 (forecast mode), got {len(args)}")
    
    if save_basename is None or len(save_basename) == 0:
        raise ValueError("save_basename is required for unweighted_nday_mean")
    
    # Convert var to list if it's a string
    if isinstance(var, str):
        var_list = [var]
    elif isinstance(var, list):
        var_list = var
    else:
        raise ValueError(f"var must be a string or list of strings, got {type(var)}")
    
    # Load region boundaries from JSON file
    with open(region_file_path, 'r') as f:
        all_regions = json.load(f)
    
    # Process each dataset and particle index
    # Note: Stepper returns `datasets` as a list of per-particle datasets, and each dataset
    # contains an `ensemble_idx` dimension for members.
    
    # Handle particle_idxs - it might be a single value or a list
    if isinstance(particle_idxs, (list, np.ndarray)):
        if len(particle_idxs) > 0:
            particle_idx = int(particle_idxs[0])  # Use first particle index
        else:
            raise ValueError("particle_idxs list is empty")
    else:
        particle_idx = int(particle_idxs)
    
    # Get the dataset for this particle (typically first element)
    if len(datasets) == 0:
        raise ValueError("datasets list is empty")
    ds = datasets[0]  # Use first dataset (Stepper obs functions assume batch_size=1)
    
    for region in regions:
        if region not in all_regions:
            raise ValueError(f"Region '{region}' not found in region file {region_file_path}")
        
        lon_region = all_regions[region]['xvals']
        lat_region = all_regions[region]['yvals']
        
        # Select all lat and lon values within the region range using the robust selection function
        ds_region = _select_region_from_dataset(ds, lon_region, lat_region)
        
        # Process each variable
        for var_name in var_list:
            # Get variable data, handling level selection if needed
            var_data = _get_variable_with_level(ds_region, var_name)
            
            # Get additional dimensions info (beyond ensemble_idx, time, lat, lon)
            additional_dims_info = {}
            for dim in var_data.dims:
                if dim not in ['ensemble_idx', 'time', 'lat', 'lon']:
                    additional_dims_info[dim] = {
                        'values': var_data[dim].values,
                        'coords': var_data.coords[dim]
                    }
            
            # Compute the mean over spatial dimensions (lon, lat)
            A = var_data.mean(dim=['lon', 'lat'])
            
            # Convert temperature from Kelvin to Celsius if needed
            # Check base variable name (before any level specification)
            base_var = var_name.split("_")[0] if "_" in var_name and var_name.split("_")[0] != "pr" else var_name
            if base_var in ['tas', 'ta']:
                A = A - 273.15
            
            # Resample to daily mean
            A = A.resample(time='1D').mean()

            # Compute at the end of the forecast: average the last `target_duration` days.
            if A.sizes.get('time', 0) < int(target_duration):
                raise ValueError(
                    f"Not enough daily samples to compute {target_duration}-day mean at end of forecast "
                    f"(have {A.sizes.get('time', 0)} daily samples)."
                )
            A = A.isel(time=slice(-int(target_duration), None)).mean(dim='time')
            
            # Now A has dimensions: ensemble_idx, [additional_dims]
            # Filter ensemble_idx to only include members in the range [ensemble_start, ensemble_end)
            ensemble_indices = A['ensemble_idx'].values
            valid_mask = (ensemble_indices >= ensemble_start) & (ensemble_indices < ensemble_end)
            A_filtered = A.isel(ensemble_idx=valid_mask)
            
            # Extract values - A_filtered should have ensemble_idx and possibly additional dims
            ensemble_values = A_filtered.values  # Shape: (num_ensemble_members, [additional_dims])
            
            # Get ensemble member indices from filtered data
            ensemble_member_indices = A_filtered['ensemble_idx'].values.tolist()
            
            # Construct filepath using the helper function
            # Function-specific string includes variable and region
            function_specific = f"{target_duration}day_{var_name}_{region}"
            # Forecast files include ensemble info; truth files omit them.
            # Lead time is now in the directory path, not the filename
            filepath = get_observation_filepath(
                save_basename=save_basename,
                obs_function_name=f"unweighted_nday_mean",
                particle_idx=particle_idx,
                event_type=event_type,
                ensemble_start=ensemble_start,
                ensemble_end=ensemble_end,
                function_specific_string=function_specific
            )
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
            
            # Determine structure based on whether we have additional dimensions
            if additional_dims_info and len(additional_dims_info) > 0:
                # Multi-dimensional observable
                # ensemble_values already has shape (num_ensemble_members, [additional_dims])
                
                # Build coordinates
                coords = {
                    'ensemble_member': ensemble_member_indices,
                    **{dim: info['values'] for dim, info in additional_dims_info.items()}
                }
                
                # Build dimension list
                dims = ['ensemble_member'] + list(additional_dims_info.keys())
                
                # Build attributes
                attrs = {
                    'obs_function_name': 'unweighted_nday_mean',
                    'description': f'Unweighted {target_duration}-day mean of {var_name} over {region}',
                    'particle_idx': particle_idx,
                    'variable': var_name,
                    'region': region,
                    'target_duration_days': target_duration
                }
                
                observation_data = xr.DataArray(
                    data=ensemble_values,
                    dims=dims,
                    coords=coords,
                    attrs=attrs
                )
            else:
                # Scalar observable - ensemble_values is 1D array
                # Build attributes
                attrs = {
                    'obs_function_name': 'unweighted_nday_mean',
                    'description': f'Unweighted {target_duration}-day mean of {var_name} over {region}',
                    'particle_idx': particle_idx,
                    'variable': var_name,
                    'region': region,
                    'target_duration_days': target_duration
                }
                
                observation_data = xr.DataArray(
                    data=ensemble_values,
                    dims=['ensemble_member'],
                    coords={
                        'ensemble_member': ensemble_member_indices
                    },
                    attrs=attrs
                )
            
            # Create Dataset and save as netCDF
            observation_ds = xr.Dataset({'observation': observation_data})
            observation_ds.to_netcdf(filepath)


def combine_observations(save_basename: str, obs_function_names: List[str], 
                        output_dir: str = None, data_dict: Dict = None) -> None:
    """
    Load and combine observation .nc files into xarray datasets.
    
    This function:
    1. Finds all .nc files matching the pattern for specified observation functions
    2. Groups files by their function-specific strings
    3. Combines each group into an xarray dataset with dimensions:
       - lead_time: Lead time in hours
       - particle: Particle index (initial condition index)
       - ensemble_member: Ensemble member index
       - Additional dimensions as stored in the .nc files
    4. Saves each dataset as a netCDF file
    5. Deletes the original .nc files
    
    Args:
        save_basename: Base file path used when creating observation files
        obs_function_names: List of observation function names to combine
        output_dir: Directory to save combined datasets (defaults to directory of save_basename)
        data_dict: Dictionary containing event data (event_type -> data_path, start/end datetimes)
                   Used for reference but not required for file matching
    
    Returns:
        None
    """
    # Extract directory and base filename from save_basename
    dir_path = os.path.dirname(save_basename) if os.path.dirname(save_basename) else "."
    base_name = os.path.basename(save_basename) if os.path.basename(save_basename) else save_basename
    
    if output_dir is None:
        output_dir = dir_path
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Pattern to match observation files
    # Actual format: obs_{event_type}_particle{particle_idx_in_event:03d}_epoch{epoch:04d}_{event_type}_particle_{particle_idx:04d}_ens_{ensemble_start:04d}-{ensemble_end:04d}_{obs_function_name}_{function_specific_string}.nc
    # Note: Lead time is now in the directory path (e.g., {base_dir}/observations/{lead_time}h/), not in the filename
    # Files are saved directly in the lead-time directory, not in subdirectories
    # The base_name passed to combine_observations is like "obs_epoch{epoch:04d}", so we need to match files that contain this pattern
    # Extract epoch from base_name if it follows the pattern "obs_epoch{epoch:04d}"
    epoch_match = re.search(r'obs_epoch(\d{4})', base_name)
    if epoch_match:
        epoch_str = epoch_match.group(1)
        # Match files that contain _epoch{epoch}_ anywhere in the filename (before the function name)
        # Pattern: obs_*_epoch{epoch}_*_particle_*_ens_*_*.nc
        pattern_base = os.path.join(dir_path, f"obs_*_epoch{epoch_str}_*_particle_*_ens_*_*.nc")
    else:
        # Fallback: use the original pattern
        pattern_base = os.path.join(dir_path, f"{base_name}_*_particle_*_ens_*_*.nc")
    
    # Find all matching files (files are directly in dir_path, not in subdirectories)
    all_files = glob.glob(pattern_base)
    
    if len(all_files) == 0:
        print(f"No observation files found matching pattern: {pattern_base}")
        return
    
    # Parse filenames and group by event_type, observation function, function-specific string, and lead_time
    file_groups = {}
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        file_dir = os.path.dirname(filepath)
        
        # Extract lead_time from directory path (e.g., "observations/24h" -> 24)
        lead_time_hours = None
        # Look for pattern like "{number}h" in the directory path
        lead_time_match = re.search(r'(\d+)h', file_dir)
        if lead_time_match:
            lead_time_hours = int(lead_time_match.group(1))
        else:
            print(f"Warning: Could not extract lead_time from directory path {file_dir}, skipping {filename}")
            continue
        
        # Parse filename components
        # Actual format: obs_{event_type}_particle{particle_idx_in_event:03d}_epoch{epoch:04d}_{event_type}_particle_{particle_idx:04d}_ens_{ensemble_start:04d}-{ensemble_end:04d}_{obs_function_name}_{function_specific_string}.nc
        # Extract epoch from base_name if available
        epoch_match = re.search(r'obs_epoch(\d{4})', base_name)
        if epoch_match:
            epoch_str = epoch_match.group(1)
            # Pattern: obs_{event_type}_particle{particle_idx_in_event:03d}_epoch{epoch:04d}_{event_type}_particle_{particle_idx:04d}_ens_{ensemble_start:04d}-{ensemble_end:04d}_{obs_function_name}_{function_specific_string}.nc
            # Match: obs_(.+?)_particle\d{3}_epoch{epoch}_[^_]+_particle_(\d+)_ens_(\d+)-(\d+)_(.+)\.nc
            # Note: The second event_type is redundant, so we skip it with [^_]+
            pattern = rf"obs_(.+?)_particle\d{{3}}_epoch{epoch_str}_[^_]+_particle_(\d+)_ens_(\d+)-(\d+)_(.+)\.nc"
        else:
            # Fallback: original pattern
            pattern = rf"{re.escape(base_name)}_(.+?)_particle_(\d+)_ens_(\d+)-(\d+)_(.+)\.nc"
        match = re.match(pattern, filename)
        
        if not match:
            print(f"Warning: Could not parse filename {filename}, skipping")
            continue
        
        event_type = match.group(1)
        particle_idx = int(match.group(2))
        ensemble_start = int(match.group(3))
        ensemble_end = int(match.group(4))
        remaining = match.group(5)
        
        # Split remaining into obs_function_name and function_specific_string
        # The obs_function_name should be one of the specified function names
        obs_function_name = None
        function_specific_string = ""
        
        for func_name in obs_function_names:
            if remaining.startswith(func_name):
                obs_function_name = func_name
                # Extract function-specific string (everything after obs_function_name + "_")
                if len(remaining) > len(func_name):
                    function_specific_string = remaining[len(func_name) + 1:]  # +1 for underscore
                break
        
        if obs_function_name is None:
            print(f"Warning: Could not identify observation function in {filename}, skipping")
            continue
        
        # Create group key: (event_type, obs_function_name, function_specific_string)
        # Only combine files with the same event_type
        group_key = (event_type, obs_function_name, function_specific_string)
        
        if group_key not in file_groups:
            file_groups[group_key] = []
        
        file_groups[group_key].append({
            'filepath': filepath,
            'particle_idx': particle_idx,
            'ensemble_start': ensemble_start,
            'ensemble_end': ensemble_end,
            'lead_time_hours': lead_time_hours
        })
    
    # Process each group (grouped by event_type, obs_function_name, function_specific_string)
    # Files from different lead_time directories are combined into a single dataset with lead_time dimension
    for (event_type, obs_function_name, function_specific_string), files in file_groups.items():
        print(f"Processing event_type={event_type}, {obs_function_name} with function-specific string: {function_specific_string}")
        print(f"  Found {len(files)} files")
        
        # Collect unique values for dimensions (including lead_times from different directories)
        lead_times = sorted(set(f['lead_time_hours'] for f in files))
        particles = sorted(set(f['particle_idx'] for f in files))
        
        # Determine ensemble member range
        all_ensemble_starts = [f['ensemble_start'] for f in files]
        all_ensemble_ends = [f['ensemble_end'] for f in files]
        min_ensemble_start = min(all_ensemble_starts)
        max_ensemble_end = max(all_ensemble_ends)
        ensemble_members = list(range(min_ensemble_start, max_ensemble_end))
        
        # Load one file to determine structure and additional dimensions
        sample_file = files[0]
        try:
            sample_ds = xr.open_dataset(sample_file['filepath'])
            if 'observation' not in sample_ds.data_vars:
                print(f"Warning: 'observation' variable not found in {os.path.basename(sample_file['filepath'])}")
                sample_ds.close()
                continue
            
            sample_obs = sample_ds['observation']
            sample_ds.close()
            
            # Get additional dimensions (beyond ensemble_member)
            additional_dims = [dim for dim in sample_obs.dims if dim != 'ensemble_member']
            additional_coords = {dim: sample_obs.coords[dim].values for dim in additional_dims}
            additional_shape = tuple(sample_obs.sizes[dim] for dim in additional_dims)
        except Exception as e:
            print(f"Warning: Error loading sample file {os.path.basename(sample_file['filepath'])}: {e}")
            continue
        
        # Initialize data array
        all_dims = ['lead_time', 'particle', 'ensemble_member'] + additional_dims
        data_shape_full = (len(lead_times), len(particles), len(ensemble_members)) + additional_shape
        combined_data = np.full(data_shape_full, np.nan, dtype=np.float32)
        
        # Load and combine data
        for file_info in files:
            try:
                file_ds = xr.open_dataset(file_info['filepath'])
                if 'observation' not in file_ds.data_vars:
                    print(f"Warning: 'observation' variable not found in {os.path.basename(file_info['filepath'])}")
                    file_ds.close()
                    continue
                
                obs_data = file_ds['observation']
                
                # Find indices
                lead_time_idx = lead_times.index(file_info['lead_time_hours'])
                particle_idx = particles.index(file_info['particle_idx'])
                ensemble_start_idx = ensemble_members.index(file_info['ensemble_start'])
                # Note: ensemble_end is exclusive (like Python slicing), so it's not in ensemble_members list
                # We don't need ensemble_end_idx since we iterate through file_ensemble_members directly
                
                # Expected number of ensemble members in this file
                expected_ensemble_count = file_info['ensemble_end'] - file_info['ensemble_start']
                
                # Get ensemble member indices from the file
                file_ensemble_members = obs_data['ensemble_member'].values
                
                # Verify ensemble member count
                if len(file_ensemble_members) != expected_ensemble_count:
                    print(f"Warning: Ensemble member count mismatch in {os.path.basename(file_info['filepath'])}")
                    print(f"  Expected {expected_ensemble_count} members, got {len(file_ensemble_members)}")
                    file_ds.close()
                    continue
                
                # Verify additional dimensions match
                file_additional_dims = [dim for dim in obs_data.dims if dim != 'ensemble_member']
                if file_additional_dims != additional_dims:
                    print(f"Warning: Additional dimensions mismatch in {os.path.basename(file_info['filepath'])}")
                    print(f"  Expected {additional_dims}, got {file_additional_dims}")
                    file_ds.close()
                    continue
                
                # Extract data values
                data_values = obs_data.values
                
                # Assign data to combined array
                for i, ens_member in enumerate(file_ensemble_members):
                    if ens_member in ensemble_members:
                        ens_idx = ensemble_members.index(ens_member)
                        if len(additional_shape) == 0:
                            combined_data[lead_time_idx, particle_idx, ens_idx] = data_values[i]
                        else:
                            # Use ellipsis to assign all additional dimensions
                            combined_data[(lead_time_idx, particle_idx, ens_idx) + (...,)] = data_values[i, ...]
                
                file_ds.close()
            except Exception as e:
                print(f"Warning: Error processing file {os.path.basename(file_info['filepath'])}: {e}")
                continue
        
        # Create coordinates (use stored coordinates from sample file if available)
        coords = {
            'lead_time': lead_times,
            'particle': particles,
            'ensemble_member': ensemble_members
        }
        # Add additional dimension coordinates
        for dim in additional_dims:
            if dim in additional_coords:
                coords[dim] = additional_coords[dim]
        
        # Create DataArray
        data_array = xr.DataArray(
            data=combined_data,
            dims=all_dims,
            coords=coords,
            name='observation'
        )
        
        # Create Dataset
        dataset = xr.Dataset({'observation': data_array})
        
        # Add event_type to dataset attributes
        dataset.attrs['event_type'] = event_type
        
        # Add data_dict as JSON string in attributes if provided
        if data_dict is not None:
            try:
                dataset.attrs['data_dict'] = json.dumps(data_dict)
                dataset.attrs['data_dict_description'] = 'Event data dictionary (event_type -> data_path, start/end datetimes) in JSON format'
            except (TypeError, ValueError) as e:
                print(f"  Warning: Could not serialize data_dict to JSON: {e}")
        
        # Construct output filename (include event_type)
        if function_specific_string:
            output_filename = f"{base_name}_{event_type}_{obs_function_name}_{function_specific_string}_combined.nc"
        else:
            output_filename = f"{base_name}_{event_type}_{obs_function_name}_combined.nc"
        
        output_path = os.path.join(output_dir, output_filename)
        
        # Save dataset
        print(f"  Saving combined dataset to {output_path}")
        dataset.to_netcdf(output_path)
        
        # Delete original .nc files
        print(f"  Deleting {len(files)} original .nc files")
        for file_info in files:
            try:
                os.remove(file_info['filepath'])
            except OSError as e:
                print(f"  Warning: Could not delete {file_info['filepath']}: {e}")
        
        print(f"  Completed processing {obs_function_name} with function-specific string: {function_specific_string}")
    
    print("Finished combining all observations")


def combine_observation_truth(save_basename: str, obs_function_names: List[str], 
                              output_dir: str = None, data_dict: Dict = None) -> None:
    """
    Load and combine truth observation .nc files into xarray datasets.
    
    This function:
    1. Finds all .nc files matching the pattern for specified observation functions (truth data)
    2. Groups files by their function-specific strings
    3. Combines each group into an xarray dataset with dimensions:
       - particle: Particle index (initial condition index)
       - Additional dimensions as stored in the .nc files
       Note: No lead_time or ensemble_member dimensions for truth data
    4. Saves each dataset as a netCDF file
    5. Deletes the original .nc files
    
    Args:
        save_basename: Base file path used when creating observation files
        obs_function_names: List of observation function names to combine
        output_dir: Directory to save combined datasets (defaults to directory of save_basename)
        data_dict: Dictionary containing event data (event_type -> data_path, start/end datetimes)
                   Used for reference but not required for file matching
    
    Returns:
        None
    """
    # Extract directory and base filename from save_basename
    dir_path = os.path.dirname(save_basename) if os.path.dirname(save_basename) else "."
    base_name = os.path.basename(save_basename) if os.path.basename(save_basename) else save_basename
    
    if output_dir is None:
        output_dir = dir_path
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Pattern to match truth observation files
    # Actual format: {base_name}_{event_type}_particle_{particle_idx:04d}_ens_{ensemble_start:04d}-{ensemble_end:04d}_{obs_function_name}_{function_specific_string}.nc
    # Note: Truth files may have ensemble info (ens_0000-0001) but are still truth files
    # Truth files are saved directly in the observations directory (not in lead-time subdirectories)
    # Match files with or without ensemble info
    pattern_base = os.path.join(dir_path, f"{base_name}_*_particle_*.nc")
    
    # Find all matching files (search in dir_path only, not recursively, since truth files are in the base directory)
    all_files = glob.glob(pattern_base)
    
    # Filter to exclude files in lead-time subdirectories (e.g., "24h/", "48h/")
    # Truth files should be directly in dir_path, not in subdirectories
    # Note: Truth files may have _ens_ in the filename, so we don't filter by that
    filtered_files = []
    for f in all_files:
        filename = os.path.basename(f)
        file_dir = os.path.dirname(f)
        # Check if file is in a lead-time subdirectory (e.g., "24h/", "48h/")
        # Truth files should be directly in dir_path, not in subdirectories
        rel_path = os.path.relpath(f, dir_path) if os.path.commonpath([f, dir_path]) == dir_path else f
        if not re.search(r'/\d+h/', rel_path) and not re.search(r'\\\d+h\\', rel_path):
            # Exclude files with _lead in filename (these are forecast files, not truth)
            if '_lead' not in filename:
                filtered_files.append(f)
    all_files = filtered_files
    
    if len(all_files) == 0:
        print(f"No truth observation files found matching pattern: {pattern_base}")
        return
    
    # Parse filenames and group by event_type, observation function and function-specific string
    file_groups = {}
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        
        # Parse filename components
        # Actual format: {base_name}_{event_type}_particle_{particle_idx:04d}_ens_{ensemble_start:04d}-{ensemble_end:04d}_{obs_function_name}_{function_specific_string}.nc
        # Note: Truth files may have ensemble info (ens_0000-0001) but are still truth files
        # Pattern matches with or without ensemble info
        # Note: event_type can contain underscores, so we match everything until _particle_
        # Pattern with ensemble: {base_name}_{event_type}_particle_{particle_idx}_ens_{ensemble_start}-{ensemble_end}_{obs_function_name}_{function_specific_string}.nc
        # Pattern without ensemble: {base_name}_{event_type}_particle_{particle_idx}_{obs_function_name}_{function_specific_string}.nc
        pattern_with_ens = rf"{re.escape(base_name)}_(.+?)_particle_(\d+)_ens_\d+-\d+_(.+)\.nc"
        pattern_without_ens = rf"{re.escape(base_name)}_(.+?)_particle_(\d+)_(.+)\.nc"
        
        match = re.match(pattern_with_ens, filename)
        if not match:
            match = re.match(pattern_without_ens, filename)
        
        if not match:
            print(f"Warning: Could not parse filename {filename}, skipping")
            continue
        
        event_type = match.group(1)
        particle_idx = int(match.group(2))
        remaining = match.group(3)
        
        # Split remaining into obs_function_name and function_specific_string
        # The obs_function_name should be one of the specified function names
        obs_function_name = None
        function_specific_string = ""
        
        for func_name in obs_function_names:
            if remaining.startswith(func_name):
                obs_function_name = func_name
                # Extract function-specific string (everything after obs_function_name + "_")
                if len(remaining) > len(func_name):
                    function_specific_string = remaining[len(func_name) + 1:]  # +1 for underscore
                break
        
        if obs_function_name is None:
            print(f"Warning: Could not identify observation function in {filename}, skipping")
            continue
        
        # Create group key: (event_type, obs_function_name, function_specific_string)
        # Only combine files with the same event_type
        group_key = (event_type, obs_function_name, function_specific_string)
        
        if group_key not in file_groups:
            file_groups[group_key] = []
        
        file_groups[group_key].append({
            'filepath': filepath,
            'particle_idx': particle_idx
        })
    
    # Process each group (grouped by event_type, obs_function_name, function_specific_string)
    for (event_type, obs_function_name, function_specific_string), files in file_groups.items():
        print(f"Processing truth event_type={event_type}, {obs_function_name} with function-specific string: {function_specific_string}")
        print(f"  Found {len(files)} files")
        
        # Collect unique values for dimensions
        particles = sorted(set(f['particle_idx'] for f in files))
        
        # Load one file to determine structure and additional dimensions
        sample_file = files[0]
        try:
            sample_ds = xr.open_dataset(sample_file['filepath'])
            if 'observation' not in sample_ds.data_vars:
                print(f"Warning: 'observation' variable not found in {os.path.basename(sample_file['filepath'])}")
                sample_ds.close()
                continue
            
            sample_obs = sample_ds['observation']
            sample_ds.close()
            
            # Get additional dimensions (truth data has no ensemble_member dimension)
            additional_dims = list(sample_obs.dims)
            additional_coords = {dim: sample_obs.coords[dim].values for dim in additional_dims}
            additional_shape = tuple(sample_obs.sizes[dim] for dim in additional_dims)
        except Exception as e:
            print(f"Warning: Error loading sample file {os.path.basename(sample_file['filepath'])}: {e}")
            continue
        
        # Initialize data array (no lead_time or ensemble_member dimensions)
        all_dims = ['particle'] + additional_dims
        data_shape_full = (len(particles),) + additional_shape
        combined_data = np.full(data_shape_full, np.nan, dtype=np.float32)
        
        # Load and combine data
        for file_info in files:
            try:
                file_ds = xr.open_dataset(file_info['filepath'])
                if 'observation' not in file_ds.data_vars:
                    print(f"Warning: 'observation' variable not found in {os.path.basename(file_info['filepath'])}")
                    file_ds.close()
                    continue
                
                obs_data = file_ds['observation']
                
                # Find particle index
                particle_idx = particles.index(file_info['particle_idx'])
                
                # Verify dimensions match
                file_dims = list(obs_data.dims)
                if file_dims != additional_dims:
                    print(f"Warning: Dimensions mismatch in {os.path.basename(file_info['filepath'])}")
                    print(f"  Expected {additional_dims}, got {file_dims}")
                    file_ds.close()
                    continue
                
                # Extract data values
                data_values = obs_data.values
                
                # Assign data (no ensemble dimension for truth data)
                if len(additional_shape) == 0:
                    # Scalar output
                    combined_data[particle_idx] = data_values.item() if hasattr(data_values, 'item') else float(data_values)
                else:
                    # Multi-dimensional output
                    if data_values.shape != additional_shape:
                        print(f"Warning: Shape mismatch in {os.path.basename(file_info['filepath'])}")
                        print(f"  Expected {additional_shape}, got {data_values.shape}")
                        file_ds.close()
                        continue
                    # Assign data
                    combined_data[(particle_idx,) + (...,)] = data_values[...]
                
                file_ds.close()
            except Exception as e:
                print(f"Warning: Error processing file {os.path.basename(file_info['filepath'])}: {e}")
                continue
        
        # Create coordinates (use stored coordinates from sample file if available)
        coords = {
            'particle': particles
        }
        # Add additional dimension coordinates
        for dim in additional_dims:
            if dim in additional_coords:
                coords[dim] = additional_coords[dim]
        
        # Create DataArray
        data_array = xr.DataArray(
            data=combined_data,
            dims=all_dims,
            coords=coords,
            name='observation_truth'
        )
        
        # Create Dataset
        dataset = xr.Dataset({'observation_truth': data_array})
        
        # Add event_type to dataset attributes
        dataset.attrs['event_type'] = event_type
        
        # Add data_dict as JSON string in attributes if provided
        if data_dict is not None:
            try:
                dataset.attrs['data_dict'] = json.dumps(data_dict)
                dataset.attrs['data_dict_description'] = 'Event data dictionary (event_type -> data_path, start/end datetimes) in JSON format'
            except (TypeError, ValueError) as e:
                print(f"  Warning: Could not serialize data_dict to JSON: {e}")
        
        # Construct output filename (include event_type)
        if function_specific_string:
            output_filename = f"{base_name}_{event_type}_{obs_function_name}_{function_specific_string}_truth_combined.nc"
        else:
            output_filename = f"{base_name}_{event_type}_{obs_function_name}_truth_combined.nc"
        
        output_path = os.path.join(output_dir, output_filename)
        
        # Save dataset
        print(f"  Saving combined truth dataset to {output_path}")
        dataset.to_netcdf(output_path)
        
        # Delete original .nc files
        print(f"  Deleting {len(files)} original .nc files")
        for file_info in files:
            try:
                os.remove(file_info['filepath'])
            except OSError as e:
                print(f"  Warning: Could not delete {file_info['filepath']}: {e}")
        
        print(f"  Completed processing truth {obs_function_name} with function-specific string: {function_specific_string}")
    
    print("Finished combining all truth observations")


def _latitude_weighting_factor(latitudes: np.ndarray) -> np.ndarray:
    """
    Compute latitude weighting factors for area-weighted metrics.
    
    Args:
        latitudes: Array of latitude values in degrees
    
    Returns:
        Array of weighting factors (normalized so sum equals number of latitudes)
    """
    lat_weights_unweighted = np.cos(np.pi / 180.0 * latitudes)
    n_lat = len(latitudes)
    return n_lat * lat_weights_unweighted / np.sum(lat_weights_unweighted)


def _compute_metric(forecast: xr.DataArray, truth: xr.DataArray, metric: str, 
                   forecast_full: xr.DataArray = None) -> Tuple[float, Optional[List[float]]]:
    """
    Compute a single error metric between forecast and truth DataArrays.
    
    Args:
        forecast: Forecast values (typically ensemble mean, xarray DataArray)
        truth: Truth values (xarray DataArray)
        metric: Name of the metric to compute. Supported metrics:
            - 'mse': Mean squared error
            - 'mae': Mean absolute error
            - 'rmse': Root mean squared error
            - 'bias': Mean bias
            - 'correlation': Pearson correlation
            - 'lat_weighted_rmse': Latitude-weighted RMSE
            - 'lat_weighted_mae': Latitude-weighted MAE
            - 'lat_weighted_fair_crps': Latitude-weighted fair CRPS (requires forecast_full with ensemble_member)
            - 'fair_crps': Unweighted fair CRPS (requires forecast_full with ensemble_member)
            - 'pearson_correlation': Pearson correlation over particle dimension
            - 'spearman_correlation': Spearman correlation over particle dimension
            - 'kendall_tau': Kendall's tau over particle dimension
        forecast_full: Full forecast DataArray with ensemble_member dimension (required for CRPS metrics)
    
    Returns:
        Tuple of (mean_error, per_particle_errors):
        - mean_error: Mean error over all particles (float)
        - per_particle_errors: List of per-particle errors (List[float] or None)
          For correlation metrics (pearson_correlation, spearman_correlation, kendall_tau),
          per_particle_errors is None since these metrics operate over the particle dimension.
    """
    from scipy.stats import spearmanr, kendalltau
    
    metric_lower = metric.lower()
    
    # Handle correlation metrics that operate over particle dimension
    if metric_lower in ['pearson_correlation', 'spearman_correlation', 'kendall_tau']:
        # These metrics compute correlation over the particle dimension
        # Both forecast and truth should have 'particle' dimension
        if 'particle' not in forecast.dims or 'particle' not in truth.dims:
            return (np.nan, None)
        
        # Align particles
        common_particles = sorted(set(forecast['particle'].values) & set(truth['particle'].values))
        if len(common_particles) < 2:
            return (np.nan, None)
        
        forecast_particle = forecast.sel(particle=common_particles)
        truth_particle = truth.sel(particle=common_particles)
        
        # Aggregate over all non-particle dimensions to get one value per particle
        # This gives us a scalar value for each particle
        forecast_agg = forecast_particle.mean(dim=[d for d in forecast_particle.dims if d != 'particle'])
        truth_agg = truth_particle.mean(dim=[d for d in truth_particle.dims if d != 'particle'])
        
        # Get values as arrays (one per particle)
        forecast_vals = forecast_agg.values
        truth_vals = truth_agg.values
        
        # Remove NaN values
        valid_mask = ~(np.isnan(forecast_vals) | np.isnan(truth_vals))
        if not np.any(valid_mask) or np.sum(valid_mask) < 2:
            return (np.nan, None)
        
        forecast_valid = forecast_vals[valid_mask]
        truth_valid = truth_vals[valid_mask]
        
        if metric_lower == 'pearson_correlation':
            return (float(np.corrcoef(forecast_valid, truth_valid)[0, 1]), None)
        elif metric_lower == 'spearman_correlation':
            corr, _ = spearmanr(forecast_valid, truth_valid)
            return (float(corr), None)
        elif metric_lower == 'kendall_tau':
            tau, _ = kendalltau(forecast_valid, truth_valid)
            return (float(tau), None)
    
    # Handle CRPS metrics (require ensemble dimension)
    if metric_lower in ['lat_weighted_fair_crps', 'fair_crps']:
        if forecast_full is None or 'ensemble_member' not in forecast_full.dims:
            return (np.nan, None)
        
        # Align particles and other dimensions
        common_particles = sorted(set(forecast_full['particle'].values) & set(truth['particle'].values))
        if len(common_particles) == 0:
            return (np.nan, None)
        
        forecast_ens = forecast_full.sel(particle=common_particles)
        truth_ens = truth.sel(particle=common_particles)
        
        # Get ensemble member count
        ensemble_members = forecast_ens['ensemble_member'].values
        M = len(ensemble_members)
        if M < 2:
            return (np.nan, None)
        
        # Compute CRPS components
        # CRPS = mean(|x_i - y|) - 0.5 * mean(|x_i - x_j|) for all i, j
        # where x_i are ensemble members, y is truth
        
        # Compute CRPSSkill: mean(|x_i - y|) over ensemble members
        # Compute CRPSSpread: mean(|x_i - x_j|) over all i, j pairs
        
        # Flatten spatial dimensions for computation
        forecast_flat = forecast_ens.values  # Shape: (particles, ensemble_members, [spatial_dims])
        truth_flat = truth_ens.values  # Shape: (particles, [spatial_dims])
        
        # Reshape to (particles, ensemble_members, n_spatial)
        original_shape = forecast_flat.shape
        n_particles = original_shape[0]
        n_spatial = int(np.prod(original_shape[2:])) if len(original_shape) > 2 else 1
        
        forecast_reshaped = forecast_flat.reshape(n_particles, M, n_spatial)
        truth_reshaped = truth_flat.reshape(n_particles, n_spatial)
        
        # Compute CRPSSkill: mean(|x_i - y|) over ensemble members
        crps_skill = np.mean(np.abs(forecast_reshaped - truth_reshaped[:, np.newaxis, :]), axis=1)  # (particles, n_spatial)
        
        # Compute CRPSSpread: mean(|x_i - x_j|) over all i, j pairs
        crps_spread = np.zeros((n_particles, n_spatial))
        for i in range(M):
            for j in range(i + 1, M):
                crps_spread += np.abs(forecast_reshaped[:, i, :] - forecast_reshaped[:, j, :])
        crps_spread = crps_spread * 2 / (M * (M - 1))  # Normalize by number of pairs
        
        # Compute CRPS = CRPSSkill - 0.5 * CRPSSpread
        crps = crps_skill - 0.5 * crps_spread  # (particles, n_spatial)
        
        # Apply latitude weighting if requested
        if metric_lower == 'lat_weighted_fair_crps':
            if 'lat' in forecast_ens.dims:
                # Get latitude coordinates
                lat_coords = forecast_ens['lat'].values
                lat_weights = _latitude_weighting_factor(lat_coords)
                
                # Find lat dimension in spatial dims
                spatial_dims = list(forecast_ens.dims[2:])  # Skip particle and ensemble_member
                if 'lat' in spatial_dims:
                    lat_dim_idx = spatial_dims.index('lat')
                    # Reshape crps to have lat as a dimension
                    crps_reshaped = crps.reshape(n_particles, *[original_shape[i] for i in range(2, len(original_shape))])
                    # Apply weights along lat dimension
                    # Create weight array with shape matching lat dimension
                    weight_shape = [1] * len(spatial_dims)
                    weight_shape[lat_dim_idx] = len(lat_weights)
                    lat_weights_reshaped = lat_weights.reshape(tuple(weight_shape))
                    # Apply weights and average over all spatial dimensions
                    crps_weighted = np.mean(crps_reshaped * lat_weights_reshaped, axis=tuple(range(1, len(crps_reshaped.shape))))
                    # Return mean and per-particle errors
                    per_particle_errors = [float(x) for x in crps_weighted]
                    return (float(np.nanmean(crps_weighted)), per_particle_errors)
            
            # If no lat dimension, return unweighted
            # Average over spatial dimensions to get per-particle CRPS
            crps_per_particle = np.nanmean(crps, axis=1)  # (particles,)
            per_particle_errors = [float(x) for x in crps_per_particle]
            return (float(np.nanmean(crps)), per_particle_errors)
        else:  # fair_crps
            # Average over spatial dimensions to get per-particle CRPS
            crps_per_particle = np.nanmean(crps, axis=1)  # (particles,)
            per_particle_errors = [float(x) for x in crps_per_particle]
            return (float(np.nanmean(crps)), per_particle_errors)
    
    # Handle latitude-weighted metrics
    if metric_lower in ['lat_weighted_rmse', 'lat_weighted_mae']:
        # Check if lat dimension exists
        if 'lat' not in forecast.dims or 'lat' not in truth.dims:
            # Fall back to unweighted version
            if metric_lower == 'lat_weighted_rmse':
                metric_lower = 'rmse'
            else:
                metric_lower = 'mae'
        else:
            # Get latitude coordinates
            lat_coords = forecast['lat'].values
            lat_weights = _latitude_weighting_factor(lat_coords)
            
            # Align dimensions
            common_dims = set(forecast.dims) & set(truth.dims)
            for dim in common_dims:
                if dim != 'lat':
                    common_vals = sorted(set(forecast[dim].values) & set(truth[dim].values))
                    if len(common_vals) > 0:
                        forecast = forecast.sel({dim: common_vals})
                        truth = truth.sel({dim: common_vals})
            
            # Compute weighted metric per particle if particle dimension exists
            if 'particle' in forecast.dims and 'particle' in truth.dims:
                # Align particles
                common_particles = sorted(set(forecast['particle'].values) & set(truth['particle'].values))
                if len(common_particles) == 0:
                    return (np.nan, None)
                
                forecast_particle = forecast.sel(particle=common_particles)
                truth_particle = truth.sel(particle=common_particles)
                
                # Compute metric per particle
                diff = forecast_particle - truth_particle
                
                # Apply latitude weights
                weight_shape = [1] * len(diff.dims)
                if 'lat' in diff.dims:
                    lat_idx = diff.dims.index('lat')
                    weight_shape[lat_idx] = len(lat_weights)
                    lat_weights_reshaped = lat_weights.reshape(tuple(weight_shape))
                    
                    if metric_lower == 'lat_weighted_rmse':
                        weighted_squared_diff = lat_weights_reshaped * (diff ** 2)
                        # Average over non-particle dimensions to get per-particle RMSE
                        per_particle_errors = np.sqrt(np.nanmean(weighted_squared_diff.values, axis=tuple(range(1, len(weighted_squared_diff.dims)))))
                        per_particle_errors = [float(x) for x in per_particle_errors]
                        return (float(np.nanmean(per_particle_errors)), per_particle_errors)
                    else:  # lat_weighted_mae
                        weighted_abs_diff = lat_weights_reshaped * np.abs(diff)
                        # Average over non-particle dimensions to get per-particle MAE
                        per_particle_errors = np.nanmean(weighted_abs_diff.values, axis=tuple(range(1, len(weighted_abs_diff.dims))))
                        per_particle_errors = [float(x) for x in per_particle_errors]
                        return (float(np.nanmean(per_particle_errors)), per_particle_errors)
            
            # No particle dimension - compute over all data
            diff = forecast - truth
            
            # Apply latitude weights
            weight_shape = [1] * len(diff.dims)
            if 'lat' in diff.dims:
                lat_idx = diff.dims.index('lat')
                weight_shape[lat_idx] = len(lat_weights)
                lat_weights_reshaped = lat_weights.reshape(tuple(weight_shape))
                
                if metric_lower == 'lat_weighted_rmse':
                    weighted_squared_diff = lat_weights_reshaped * (diff ** 2)
                    return (float(np.sqrt(np.nanmean(weighted_squared_diff.values))), None)
                else:  # lat_weighted_mae
                    weighted_abs_diff = lat_weights_reshaped * np.abs(diff)
                    return (float(np.nanmean(weighted_abs_diff.values)), None)
    
    # Standard metrics (compute per-particle if particle dimension exists)
    if 'particle' in forecast.dims and 'particle' in truth.dims:
        # Align particles
        common_particles = sorted(set(forecast['particle'].values) & set(truth['particle'].values))
        if len(common_particles) == 0:
            return (np.nan, None)
        
        forecast_particle = forecast.sel(particle=common_particles)
        truth_particle = truth.sel(particle=common_particles)
        
        # Compute metric per particle
        per_particle_errors = []
        for particle in common_particles:
            forecast_p = forecast_particle.sel(particle=particle)
            truth_p = truth_particle.sel(particle=particle)
            
            # Flatten spatial dimensions
            forecast_flat = forecast_p.values.flatten()
            truth_flat = truth_p.values.flatten()
            
            # Remove NaN values
            valid_mask = ~(np.isnan(forecast_flat) | np.isnan(truth_flat))
            if not np.any(valid_mask):
                per_particle_errors.append(np.nan)
                continue
            
            forecast_valid = forecast_flat[valid_mask]
            truth_valid = truth_flat[valid_mask]
            
            if metric_lower == 'mse':
                per_particle_errors.append(float(np.mean((forecast_valid - truth_valid) ** 2)))
            elif metric_lower == 'mae':
                per_particle_errors.append(float(np.mean(np.abs(forecast_valid - truth_valid))))
            elif metric_lower == 'rmse':
                per_particle_errors.append(float(np.sqrt(np.mean((forecast_valid - truth_valid) ** 2))))
            elif metric_lower == 'bias':
                per_particle_errors.append(float(np.mean(forecast_valid - truth_valid)))
            elif metric_lower == 'correlation':
                if len(forecast_valid) < 2:
                    per_particle_errors.append(np.nan)
                else:
                    per_particle_errors.append(float(np.corrcoef(forecast_valid, truth_valid)[0, 1]))
            else:
                supported = ['mse', 'mae', 'rmse', 'bias', 'correlation', 'lat_weighted_rmse', 
                             'lat_weighted_mae', 'lat_weighted_fair_crps', 'fair_crps',
                             'pearson_correlation', 'spearman_correlation', 'kendall_tau']
                raise ValueError(f"Unknown error metric: {metric}. Supported metrics: {', '.join(supported)}")
        
        # Compute mean error
        mean_error = float(np.nanmean(per_particle_errors))
        return (mean_error, per_particle_errors)
    
    # No particle dimension - compute over all data
    forecast_flat = forecast.values.flatten()
    truth_flat = truth.values.flatten()
    
    # Remove NaN values
    valid_mask = ~(np.isnan(forecast_flat) | np.isnan(truth_flat))
    if not np.any(valid_mask):
        return (np.nan, None)
    
    forecast_valid = forecast_flat[valid_mask]
    truth_valid = truth_flat[valid_mask]
    
    if metric_lower == 'mse':
        return (float(np.mean((forecast_valid - truth_valid) ** 2)), None)
    elif metric_lower == 'mae':
        return (float(np.mean(np.abs(forecast_valid - truth_valid))), None)
    elif metric_lower == 'rmse':
        return (float(np.sqrt(np.mean((forecast_valid - truth_valid) ** 2))), None)
    elif metric_lower == 'bias':
        return (float(np.mean(forecast_valid - truth_valid)), None)
    elif metric_lower == 'correlation':
        if len(forecast_valid) < 2:
            return (np.nan, None)
        return (float(np.corrcoef(forecast_valid, truth_valid)[0, 1]), None)
    else:
        supported = ['mse', 'mae', 'rmse', 'bias', 'correlation', 'lat_weighted_rmse', 
                     'lat_weighted_mae', 'lat_weighted_fair_crps', 'fair_crps',
                     'pearson_correlation', 'spearman_correlation', 'kendall_tau']
        raise ValueError(f"Unknown error metric: {metric}. Supported metrics: {', '.join(supported)}")


def compute_error_metrics(save_basename: str, save_basename_truth: str, 
                          obs_function_names: List[str], error_metrics: List[str],
                          output_dir: str = None) -> Dict:
    """
    Compute error metrics between ensemble observable forecasts and their true values.
    
    This function:
    1. Loads combined forecast and truth datasets
    2. Matches them by observable function and function-specific string
    3. Computes error metrics as a function of lead time
    4. Organizes results in a nested dictionary structure
    5. Saves results to a JSON file
    
    Args:
        save_basename: Base file path used for forecast observation files
        save_basename_truth: Base file path used for truth observation files
        obs_function_names: List of observation function names to process
        error_metrics: List of error metric names to compute (e.g., 'mse', 'mae', 'rmse', 'bias', 'correlation')
        output_dir: Directory to save error metrics JSON file (defaults to directory of save_basename)
    
    Returns:
        Dictionary organized as: {event_type: {observable_function: {function_specific_string: {error_metric: [errors_by_lead_time]}}}}
    """
    # Extract directory and base filename
    dir_path = os.path.dirname(save_basename) if os.path.dirname(save_basename) else "."
    base_name = os.path.basename(save_basename) if os.path.basename(save_basename) else save_basename
    
    dir_path_truth = os.path.dirname(save_basename_truth) if os.path.dirname(save_basename_truth) else "."
    base_name_truth = os.path.basename(save_basename_truth) if os.path.basename(save_basename_truth) else save_basename_truth
    
    if output_dir is None:
        output_dir = dir_path
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find combined forecast datasets
    # Note: compute_error_metrics is called per lead-time directory
    # Forecast files should be in dir_path (which is already a lead-time subdirectory like {obs_base_dir}/24h/)
    # Pattern: {base_name}_{event_type}_{obs_function_name}_{function_specific_string}_combined.nc
    forecast_pattern = os.path.join(dir_path, f"{base_name}_*_*_combined.nc")
    forecast_files = glob.glob(forecast_pattern)
    
    # Find combined truth datasets
    # Truth files are saved directly in the observations directory (not in lead-time subdirectories)
    # Pattern: {base_name_truth}_{event_type}_{obs_function_name}_{function_specific_string}_truth_combined.nc
    # Note: Truth files should be in dir_path_truth (observations directory), not in subdirectories
    truth_pattern = os.path.join(dir_path_truth, f"{base_name_truth}_*_*_truth_combined.nc")
    truth_files = glob.glob(truth_pattern)
    
    # Also search for truth files that might be in the same directory as forecast files (for backward compatibility)
    # But exclude lead-time subdirectories
    if len(truth_files) == 0:
        # Try searching in parent directory if dir_path_truth is a lead-time subdirectory
        parent_dir = os.path.dirname(dir_path_truth)
        if parent_dir and parent_dir != dir_path_truth:
            truth_pattern_parent = os.path.join(parent_dir, f"{base_name_truth}_*_*_truth_combined.nc")
            truth_files = glob.glob(truth_pattern_parent)
    
    if len(forecast_files) == 0:
        print(f"No forecast observation files found matching pattern: {forecast_pattern}")
        return {}
    
    if len(truth_files) == 0:
        print(f"No truth observation files found matching pattern: {truth_pattern}")
        return {}
    
    # Parse filenames to extract event_type, observable function and function-specific string
    forecast_info = {}
    for filepath in forecast_files:
        filename = os.path.basename(filepath)
        # Pattern: {base_name}_{event_type}_{obs_function_name}_{function_specific_string}_combined.nc
        # Note: event_type can contain underscores, so we match everything until we find the obs_function_name
        # We need to try matching with each possible obs_function_name to find where event_type ends
        matched = False
        for func_name in obs_function_names:
            # Try pattern: {base_name}_{event_type}_{func_name}_{function_specific_string}_combined.nc
            # or: {base_name}_{event_type}_{func_name}_combined.nc
            pattern1 = rf"{re.escape(base_name)}_(.+?)_{re.escape(func_name)}_(.+?)_combined\.nc"
            pattern2 = rf"{re.escape(base_name)}_(.+?)_{re.escape(func_name)}_combined\.nc"
            
            match1 = re.match(pattern1, filename)
            match2 = re.match(pattern2, filename)
            
            if match1:
                event_type = match1.group(1)
                function_specific_string = match1.group(2)
                obs_function_name = func_name
                key = (event_type, obs_function_name, function_specific_string)
                forecast_info[key] = filepath
                matched = True
                break
            elif match2:
                event_type = match2.group(1)
                function_specific_string = ""
                obs_function_name = func_name
                key = (event_type, obs_function_name, function_specific_string)
                forecast_info[key] = filepath
                matched = True
                break
        
        if not matched:
            print(f"Warning: Could not parse forecast filename {filename}, skipping. Expected pattern: {base_name}_<event_type>_<obs_function>_<function_specific>_combined.nc")
    
    truth_info = {}
    for filepath in truth_files:
        filename = os.path.basename(filepath)
        # Pattern: {base_name_truth}_{event_type}_{obs_function_name}_{function_specific_string}_truth_combined.nc
        # Note: event_type can contain underscores, so we match everything until we find the obs_function_name
        # We need to try matching with each possible obs_function_name to find where event_type ends
        matched = False
        for func_name in obs_function_names:
            # Try pattern: {base_name_truth}_{event_type}_{func_name}_{function_specific_string}_truth_combined.nc
            # or: {base_name_truth}_{event_type}_{func_name}_truth_combined.nc
            pattern1 = rf"{re.escape(base_name_truth)}_(.+?)_{re.escape(func_name)}_(.+?)_truth_combined\.nc"
            pattern2 = rf"{re.escape(base_name_truth)}_(.+?)_{re.escape(func_name)}_truth_combined\.nc"
            
            match1 = re.match(pattern1, filename)
            match2 = re.match(pattern2, filename)
            
            if match1:
                event_type = match1.group(1)
                function_specific_string = match1.group(2)
                obs_function_name = func_name
                key = (event_type, obs_function_name, function_specific_string)
                truth_info[key] = filepath
                matched = True
                break
            elif match2:
                event_type = match2.group(1)
                function_specific_string = ""
                obs_function_name = func_name
                key = (event_type, obs_function_name, function_specific_string)
                truth_info[key] = filepath
                matched = True
                break
        
        if not matched:
            print(f"Warning: Could not parse truth filename {filename}, skipping. Expected pattern: {base_name_truth}_<event_type>_<obs_function>_<function_specific>_truth_combined.nc")
    
    # Diagnostic output
    print(f"Found {len(forecast_info)} forecast file(s) and {len(truth_info)} truth file(s) after parsing")
    if len(forecast_info) == 0:
        print(f"Error: No forecast files could be parsed. Found {len(forecast_files)} files matching pattern, but none matched expected format.")
        print(f"  Pattern searched: {forecast_pattern}")
        print(f"  Sample filenames found:")
        for f in forecast_files[:5]:  # Show first 5
            print(f"    {os.path.basename(f)}")
        return {}
    
    if len(truth_info) == 0:
        print(f"Error: No truth files could be parsed. Found {len(truth_files)} files matching pattern, but none matched expected format.")
        print(f"  Pattern searched: {truth_pattern}")
        print(f"  Sample filenames found:")
        for f in truth_files[:5]:  # Show first 5
            print(f"    {os.path.basename(f)}")
        return {}
    
    # Show what was successfully parsed
    print(f"Successfully parsed forecast files:")
    for key, path in forecast_info.items():
        print(f"  {key}: {os.path.basename(path)}")
    print(f"Successfully parsed truth files:")
    for key, path in truth_info.items():
        print(f"  {key}: {os.path.basename(path)}")
    
    # Initialize results dictionary (event_type is top-level key)
    results = {}
    
    # Check if we have any matching pairs
    matching_pairs = []
    for key in forecast_info.keys():
        if key in truth_info:
            matching_pairs.append(key)
    
    if len(matching_pairs) == 0:
        print(f"Error: No matching forecast-truth pairs found!")
        print(f"  Forecast keys: {list(forecast_info.keys())}")
        print(f"  Truth keys: {list(truth_info.keys())}")
        return {}
    
    print(f"Found {len(matching_pairs)} matching forecast-truth pair(s) to process")
    
    # Process each matched pair of forecast and truth datasets
    for (event_type, obs_function_name, function_specific_string), forecast_path in forecast_info.items():
        if (event_type, obs_function_name, function_specific_string) not in truth_info:
            print(f"Warning: No truth dataset found for event_type={event_type}, {obs_function_name} with function-specific string: {function_specific_string}")
            continue
        
        truth_path = truth_info[(event_type, obs_function_name, function_specific_string)]
        
        print(f"Processing event_type={event_type}, {obs_function_name} with function-specific string: {function_specific_string}")
        print(f"  Forecast: {os.path.basename(forecast_path)}")
        print(f"  Truth: {os.path.basename(truth_path)}")
        
        # Load datasets
        try:
            forecast_ds = xr.open_dataset(forecast_path)
            truth_ds = xr.open_dataset(truth_path)
        except Exception as e:
            print(f"  Error loading datasets: {e}")
            import traceback
            print(traceback.format_exc())
            continue
        
        # Get observation data
        if 'observation' not in forecast_ds.data_vars:
            print(f"  Warning: 'observation' variable not found in forecast dataset")
            continue
        
        if 'observation_truth' not in truth_ds.data_vars:
            print(f"  Warning: 'observation_truth' variable not found in truth dataset")
            continue
        
        forecast_data = forecast_ds['observation']
        truth_data = truth_ds['observation_truth']
        
        # Check dimensions
        # Forecast should have: lead_time, particle, ensemble_member, [additional_dims]
        # Truth should have: particle, [additional_dims]
        
        # Get lead times from forecast
        if 'lead_time' not in forecast_data.dims:
            print(f"  Warning: 'lead_time' dimension not found in forecast dataset")
            continue
        
        lead_times = sorted(forecast_data['lead_time'].values)
        
        # Get particles (should match between forecast and truth)
        if 'particle' not in forecast_data.dims or 'particle' not in truth_data.dims:
            print(f"  Warning: 'particle' dimension not found in datasets")
            continue
        
        forecast_particles = sorted(forecast_data['particle'].values)
        truth_particles = sorted(truth_data['particle'].values)
        
        # Debug output
        print(f"  Forecast particles: {forecast_particles}")
        print(f"  Truth particles: {truth_particles}")
        
        # Find common particles
        common_particles = sorted(set(forecast_particles) & set(truth_particles))
        if len(common_particles) == 0:
            print(f"  Warning: No common particles between forecast and truth datasets")
            print(f"    Forecast has particles: {forecast_particles}")
            print(f"    Truth has particles: {truth_particles}")
            continue
        
        print(f"  Common particles: {common_particles}")
        
        # Initialize results for this event_type and observable
        # Structure: results[event_type][obs_function_name][function_specific_string]
        if event_type not in results:
            results[event_type] = {}
        
        if obs_function_name not in results[event_type]:
            results[event_type][obs_function_name] = {}
        
        if function_specific_string not in results[event_type][obs_function_name]:
            results[event_type][obs_function_name][function_specific_string] = {
                'lead_times': [float(lt) for lt in lead_times]  # Store lead times for reference
            }
        
        # Compute ensemble mean forecast (average over ensemble_member dimension)
        if 'ensemble_member' in forecast_data.dims:
            forecast_mean = forecast_data.mean(dim='ensemble_member')
        else:
            forecast_mean = forecast_data
        
        # Compute error metrics for each lead time
        for metric in error_metrics:
            if metric not in results[event_type][obs_function_name][function_specific_string]:
                results[event_type][obs_function_name][function_specific_string][metric] = []
            
            # Store per-particle errors separately
            metric_per_particle_key = f"{metric}_per_particle"
            if metric_per_particle_key not in results[event_type][obs_function_name][function_specific_string]:
                results[event_type][obs_function_name][function_specific_string][metric_per_particle_key] = []
            
            errors_by_lead_time = []
            per_particle_errors_by_lead_time = []
            
            # Check if metric requires full ensemble data
            metric_lower = metric.lower()
            requires_full_ensemble = metric_lower in ['lat_weighted_fair_crps', 'fair_crps']
            
            for lead_time in lead_times:
                # Select data for this lead time
                if requires_full_ensemble:
                    # Use full forecast data (with ensemble_member dimension)
                    forecast_lt = forecast_data.sel(lead_time=lead_time)
                else:
                    # Use ensemble mean
                    forecast_lt = forecast_mean.sel(lead_time=lead_time)
                
                # Align particles between forecast and truth
                forecast_lt_aligned = forecast_lt.sel(particle=common_particles)
                truth_aligned = truth_data.sel(particle=common_particles)
                
                # Debug: Check for NaN values and data shapes
                forecast_nan_count = np.isnan(forecast_lt_aligned.values).sum()
                truth_nan_count = np.isnan(truth_aligned.values).sum()
                forecast_total = forecast_lt_aligned.size
                truth_total = truth_aligned.size
                print(f"    Lead time {lead_time}: forecast shape={forecast_lt_aligned.shape}, NaNs={forecast_nan_count}/{forecast_total}, truth shape={truth_aligned.shape}, NaNs={truth_nan_count}/{truth_total}")
                
                # Compute metric
                try:
                    if requires_full_ensemble:
                        error_value, per_particle_errors = _compute_metric(
                            forecast_mean.sel(lead_time=lead_time).sel(particle=common_particles),
                            truth_aligned, metric, 
                            forecast_full=forecast_lt_aligned)
                    else:
                        error_value, per_particle_errors = _compute_metric(forecast_lt_aligned, truth_aligned, metric)
                    
                    # Store mean error
                    errors_by_lead_time.append(float(error_value))
                    
                    # Store per-particle errors (as list of lists: [lead_time][particle])
                    if per_particle_errors is not None:
                        per_particle_errors_by_lead_time.append([float(x) for x in per_particle_errors])
                    else:
                        per_particle_errors_by_lead_time.append(None)
                except Exception as e:
                    print(f"  Warning: Error computing {metric} at lead_time {lead_time}: {e}")
                    import traceback
                    print(traceback.format_exc())
                    errors_by_lead_time.append(np.nan)
                    per_particle_errors_by_lead_time.append(None)
            
            # Store mean errors (for backward compatibility and return value)
            results[event_type][obs_function_name][function_specific_string][metric] = errors_by_lead_time
            
            # Store per-particle errors
            results[event_type][obs_function_name][function_specific_string][metric_per_particle_key] = per_particle_errors_by_lead_time
        
        # Close datasets
        forecast_ds.close()
        truth_ds.close()
        
        print(f"  Completed processing event_type={event_type}, {obs_function_name} with function-specific string: {function_specific_string}")
    
    # Check if we have any results
    if len(results) == 0:
        print(f"Error: No error metrics were computed! Results dictionary is empty.")
        print(f"  This could be due to:")
        print(f"    - No matching forecast-truth pairs found")
        print(f"    - Errors during dataset loading or processing")
        print(f"    - Missing required dimensions or variables in datasets")
        return {}
    
    print(f"Successfully computed error metrics for {len(results)} event type(s)")
    
    # Save results to JSON file
    output_filename = f"{base_name}_error_metrics.json"
    output_path = os.path.join(output_dir, output_filename)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if np.isnan(obj) or np.isfinite(obj) else None
        elif isinstance(obj, np.ndarray):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (int, float)):
            return float(obj) if np.isnan(obj) or np.isfinite(obj) else None
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Saved error metrics to {output_path}")
    
    return results
