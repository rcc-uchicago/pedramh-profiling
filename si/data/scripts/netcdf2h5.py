import h5py as h5
from tqdm.auto import tqdm

import numpy as np
import xarray as xr
import os 
import datetime

train_years = [str(year) for year in range(1979, 2013)]
valid_years = [str(year) for year in range(2013, 2015)]
test_years = [str(year) for year in range(2015, 2025)]

num_train_steps = 49676
num_valid_steps = 2920
num_test_steps = 14612
surface_variables = ["skin_temperature",
                     "surface_pressure",
                     "2m_temperature",
                     "2m_specific_humidity", 
                     "10m_u_component_of_wind", 
                     "10m_v_component_of_wind"]

multi_variables = ["temperature",
                  "u_component_of_wind",
                  "v_component_of_wind",
                  "geopotential",
                  "specific_humidity",
                  "specific_cloud_liquid_water_content",
                  "specific_cloud_ice_water_content",
                  "fraction_of_cloud_cover"]

forcing_variables = ["DSWRFtoa", "sea_surface_temperature", "sea_ice_cover"] # in the diagnostic dir
invariant_variables = ["geopotential_at_surface", "land_sea_mask"]   
diagnostic_variables = ["USWRFtoa",
                        "ULWRFtoa", 
                       "USWRFsfc",
                       "ULWRFsfc",
                       "DSWRFsfc",
                       "DLWRFsfc",
                       "PRATEsfc",
                       "LHTFLsfc",
                       "SHTFLsfc"]


def dt2cal(dt):
    """
    Convert array of datetime64 to a calendar array of year, month, day, hour,
    minute, seconds, microsecond with these quantites indexed on the last axis.

    Parameters
    ----------
    dt : datetime64 array (...)
        numpy.ndarray of datetimes of arbitrary shape

    Returns
    -------
    cal : uint32 array (..., 7)
        calendar array with last axis representing year, month, day, hour,
        minute, second, microsecond
    """

    # allocate output 
    out = np.empty(dt.shape + (4,), dtype="u4")
    # decompose calendar floors
    Y, M, D, h, m, s = [dt.astype(f"M8[{x}]") for x in "YMDhms"]
    out[..., 0] = Y + 1970 # Gregorian Year
    out[..., 1] = (M - Y) + 1 # month
    out[..., 2] = (D - M) + 1 # day
    out[..., 3] = (dt - D).astype("m8[h]") # hour
    return out

def get_year_fractions(days, months, years):
    """
    Converts arrays of days, months, and years into an array of 
    year fractions (0 to ~1).
    
    Formula: (DayOfYear - 1) / TotalDaysInYear
    """
    fractions = []
    
    # Zip combines the three arrays so we can iterate through them simultaneously
    for d, m, y in zip(days, months, years):
        # Convert year string to int if necessary
        y = int(y)
        
        # Create a date object
        current_date = datetime.date(y, m, d)
        
        # Get the day of the year (1 for Jan 1st, 365/366 for Dec 31st)
        day_of_year = current_date.timetuple().tm_yday
        
        # Determine if it's a leap year to set the divisor (365 or 366)
        # A year is a leap year if divisible by 4, unless divisible by 100 but not 400
        is_leap = (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)
        days_in_year = 366 if is_leap else 365
        
        # Calculate fraction: Jan 1 becomes 0.0
        frac = (day_of_year - 1) / days_in_year
        fractions.append(frac)
        
    return fractions

def add_buffer(h5f, variables, year, root, offset, current_time, nchannels, batch_size=100, multi=False):
    nlat = 180
    nlon = 360
    nlevels = 26
    batch_size = 100
    for t_start in tqdm(range(0, offset, batch_size)):
        t_end = min(t_start + batch_size, offset)
        
        # Pre-allocate a much smaller buffer
        # Shape: (batch, lat, lon, channels)
        if multi:
            batch_buffer = np.zeros((t_end - t_start, nlevels, nlat, nlon, nchannels), dtype='f4')
        else:
            batch_buffer = np.zeros((t_end - t_start, nlat, nlon, nchannels), dtype='f4')
        
        for i, key in enumerate(variables):
            file = f"{root}/{key}/{year}_180x360.nc"
            with xr.open_dataset(file) as ds:
                key = list(ds.keys())[0]
                # Only load the specific time slice from the NetCDF
                if multi:
                    batch_buffer[:, :, :, :, i] = ds[key].isel(time=slice(t_start, t_end)).values
                else:
                    batch_buffer[:, :, :, i] = ds[key].isel(time=slice(t_start, t_end)).values
                
        # Write the batch to H5
        h5f[current_time + t_start : current_time + t_end, ...] = batch_buffer

def add_data(f, split = "train", num_samples=49676, years=train_years):
    nlat = 180
    nlon = 360
    nlevels = 26
    nsurface_channels = 6
    nmulti_channels = 8
    nforcing_channels = 3
    ninvariant_channels = 2
    ndiagnostic_channels = 9
    root_dir = "/glade/derecho/scratch/pahlavan/ai-models/AMIP/ERA5"
    
    dataset = f.create_group(split)
    h5f_surface = dataset.create_dataset('surface', (num_samples, nlat, nlon, nsurface_channels), dtype='f4', chunks=(1, nlat, nlon, nsurface_channels))
    h5f_multi = dataset.create_dataset('multilevel', (num_samples, nlevels, nlat, nlon, nmulti_channels), dtype='f4', chunks=(1, nlevels, nlat, nlon, nmulti_channels))
    h5f_forcing = dataset.create_dataset('forcing', (num_samples, nlat, nlon, nforcing_channels), dtype='f4', chunks=(1, nlat, nlon, nforcing_channels))
    h5f_invariant = dataset.create_dataset('invariant', (nlat, nlon, ninvariant_channels), dtype='f4', chunks=(nlat, nlon, ninvariant_channels))
    h5f_diagnostic = dataset.create_dataset('diagnostic', (num_samples, nlat, nlon, ndiagnostic_channels), dtype='f4', chunks=(1, nlat, nlon, ndiagnostic_channels))

    hourofdaycoord = dataset.create_dataset('hour', (num_samples), dtype='f4')
    dayofyearcoord = dataset.create_dataset('day', (num_samples), dtype='f4')
    monthofyearcoord = dataset.create_dataset('month', (num_samples), dtype='i4')
    yearcoord = dataset.create_dataset('year', (num_samples), dtype='i4')
    latcoord = dataset.create_dataset('lat', (nlat), dtype='f4')
    loncoord = dataset.create_dataset('lon', (nlon), dtype='f4')

    # load constants
    for i, key in enumerate(invariant_variables):
        file = f"{root_dir}/forcing/{key}/180x360.nc"
        ds = xr.open_dataset(file, engine="netcdf4")
        key = list(ds.keys())[0]
        data = np.array(ds[key])[0] # shape nlat nlon
        h5f_invariant[:, :, i] = data # shape nlat nlon
        ds.close()

    latcoord[:] = np.array(ds['lat'])
    loncoord[:] = np.array(ds['lon'])

    current_time = 0
    for year in tqdm(years):

        # get offset
        file = f"/glade/derecho/scratch/pahlavan/ai-models/AMIP/ERA5/prognostic/10m_u_component_of_wind/{year}_180x360.nc"
        ds = xr.open_dataset(file, engine="netcdf4")
        offset = ds['time'].shape[0]
        ds.close()
        
        # do multilevel variables
        add_buffer(h5f_multi, multi_variables, year, f"{root_dir}/prognostic/3D_PL", offset, current_time, nmulti_channels, batch_size=100, multi=True)
        
        # do surface variables
        add_buffer(h5f_surface, surface_variables, year, f"{root_dir}/prognostic", offset, current_time, nsurface_channels, batch_size=offset, multi=False)

        # do forcing variables
        add_buffer(h5f_forcing, forcing_variables, year, f"{root_dir}/forcing", offset, current_time, nforcing_channels, batch_size=offset, multi=False)

        # do diagnostic variables
        add_buffer(h5f_diagnostic, diagnostic_variables, year, f"{root_dir}/diagnostic", offset, current_time, ndiagnostic_channels, batch_size=offset, multi=False)

        # save times
        times = np.array(ds['time'])
        time_data = dt2cal(times) # year, month, day, hour
        years = time_data[:, 0]
        months = time_data[:, 1]
        days = time_data[:, 2]
        hours = time_data[:, 3]
        hourofday = hours/24
        dayofyear = get_year_fractions(days, months, years)
        
        hourofdaycoord[current_time:current_time+offset] = hourofday
        dayofyearcoord[current_time:current_time+offset] = dayofyear
        monthofyearcoord[current_time:current_time+offset] = months
        yearcoord[current_time:current_time+offset] = years

        # increment offset
        current_time += offset

fname = "/glade/derecho/scratch/ayz/AMIP_train.h5"

if os.path.exists(fname):
    print(f"'{fname}' exists. Deleting it")
    os.remove(fname)

f = h5.File(fname, 'a')
add_data(f, 'train', num_train_steps, train_years)
f.close()