import os
from os import listdir
from os.path import join
import pickle
import cftime
import warnings
from typing import Literal

from torch.utils.data import Dataset
from torchvision.transforms import Normalize, Compose
import torch

import xarray as xr
import numpy as np

from datetime import datetime
from typing import Literal

from dateutil.relativedelta import relativedelta

def load_mean_std(mean_file, std_file, datavars):
   with xr.open_dataset(mean_file) as ds:
       mean = torch.stack([torch.from_numpy(ds[var].values).to(torch.float32) for var in datavars], dim=0)
   with xr.open_dataset(std_file) as ds:
       std = torch.stack([torch.from_numpy(ds[var].values).to(torch.float32) for var in datavars], dim=0)
   return mean, std

def datetime_class_from_calendar(calendar):
    datetime_class_dict = {'standard': cftime.DatetimeGregorian,
                           'Gregorian:': cftime.DatetimeGregorian,
                           'noleap': cftime.DatetimeNoLeap,
                           '365_day': cftime.DatetimeNoLeap,
                           'proleptic_gregorian': cftime.DatetimeProlepticGregorian,
                           'all_leap': cftime.DatetimeAllLeap,
                           '366_day': cftime.DatetimeAllLeap,
                           '360_day': cftime.Datetime360Day,
                           'julian': cftime.DatetimeJulian}
    return datetime_class_dict[calendar]

class DatasetFromFolder(Dataset):
    def __init__(self, data_dir, year_start, year_end, flag, surface_variables=None,
                 upper_air_variables=None, 
                 constant_boundary_variables=None, varying_boundary_variables=None,
                 boundary_dir="boundary_variables",
                 surface_mean_file = "surface_mean.nc", surface_std_file = "surface_std.nc",
                 upper_air_mean_file = "upper_air_mean.nc", upper_air_std_file = "upper_air_std.nc",
                 calendar = 'proleptic_gregorian', timedelta_hours = 6):
        super().__init__()
        self.data_dir = data_dir
        self.year_start = year_start
        self.year_end = year_end
        self.flag = flag
        self.calendar = calendar
        self.timedelta_hours = timedelta_hours
        self.datetime_class = datetime_class_from_calendar(self.calendar)
        self.timedelta = self.datetime_class(1, 1, 1, hour = self.timedelta_hours) - \
                         self.datetime_class(1, 1, 1, hour = 0)
        self.surface_variables = surface_variables or []
        self.upper_air_variables = upper_air_variables or []

        self.constant_boundary_variables = constant_boundary_variables or []
        self.varying_boundary_variables = varying_boundary_variables or []
        self.boundary_dir = boundary_dir
        self.constant_boundary_data = self._load_constant_boundary_data()


        self.surface_mean, self.surface_std = load_mean_std(join(data_dir, surface_mean_file),
                                                            join(data_dir, surface_std_file),
                                                            self.surface_variables)
        self.upper_air_mean, self.upper_air_std = load_mean_std(join(data_dir, upper_air_mean_file),
                                                                join(data_dir, upper_air_std_file),
                                                                self.upper_air_variables)
        self.num_levels = self.upper_air_mean.size(-1)
        self.surface_transform = self._create_surface_transform()
        self.upper_air_transform = self._create_upper_air_transform()
        self.surface_inv_transform = self._create_surface_inv_transform()
        self.upper_air_inv_transform = self._create_upper_air_inv_transform()
        #self.channel_seq = self.surface_variables + self.upper_air_variables

        self.boundary_ds = self._load_boundary_data()
        self.dates = self._get_dates()
        self.data_ds = self._load_data()

    def __getitem__(self, index):
        start_time = self.dates[index]
        end_time = self.dates[index + 1]
        constant_boundary_data = self.constant_boundary_data
        varying_boundary_data = self._get_boundary_data(start_time)
        surface_t, upper_air_t = self._get_data(start_time)
        surface_t_1, upper_air_t_1 = self._get_data(end_time)

        if self.flag == "train":
            return surface_t, upper_air_t, surface_t_1, upper_air_t_1, varying_boundary_data
        return surface_t, upper_air_t, surface_t_1, upper_air_t_1, varying_boundary_data, torch.tensor([start_time, end_time])


    def _load_constant_boundary_data(self):
        constant_boundary_files = [join(self.data_dir, self.boundary_dir, f) for f in \
                                   os.listdir(join(self.data_dir, self.boundary_dir)) \
                                   if any(var in f for var in self.constant_boundary_variables)]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message='^.*Unable to decode time axis into full numpy.datetime64 objects.*$')
            constant_boundary_ds = xr.open_mfdataset(constant_boundary_files, engine = 'netcdf4', parallel=False)
        constant_boundary_data = torch.tensor(np.stack([constant_boundary_ds[var].values for var in \
                                                        self.constant_boundary_variables], axis=0), dtype=torch.float32)
        return constant_boundary_data

    def _load_data(self):
        data_files = [join(self.data_dir, f'data_{year}.nc') for year in range(self.year_start, self.year_end)]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message='^.*Unable to decode time axis into full numpy.datetime64 objects.*$')
            data_ds = xr.open_mfdataset(data_files, chunks={'time': 1, 'lev': self.num_levels}, engine = 'netcdf4',
                                        parallel=False)
        return data_ds

    def _get_data(self, date):
        surface_data = torch.stack([torch.from_numpy(
            self.data_ds[var].sel(time=date).values).to(torch.float32) for var in self.surface_variables], dim = 0)
        surface_data = self.surface_transform(surface_data)

        upper_air_data = torch.stack([
            torch.from_numpy(self.data_ds[var].sel(time=date).values).to(torch.float32)
            for var in self.upper_air_variables], dim = 0)
        upper_air_data = self.upper_air_transform(upper_air_data)

        return surface_data, upper_air_data

    def _get_boundary_data(self, start_time):
        start_time_boundary = self._get_boundary_date(start_time)
        varying_boundary_data = torch.stack([torch.from_numpy(\
            self.boundary_ds[var].sel(time=start_time_boundary).values).to(torch.float32)
            for var in self.varying_boundary_variables], dim = 0)
        return varying_boundary_data

    def _get_boundary_date(self, date):
        if self._check_leap_year(date):
            boundary_date = self.datetime_class(self.boundary_ds.leap_year, date.month, date.day, hour=date.hour)
        else:
            boundary_date = self.datetime_class(self.boundary_ds.noleap_year, date.month, date.day, hour=date.hour)
        return boundary_date

    def __len__(self):
        return len(self.dates) - 1

    def _create_surface_transform(self):
        return lambda data: (data - self.surface_mean.reshape(-1, 1, 1))/self.surface_std.reshape(-1, 1, 1)

    def _create_upper_air_transform(self):
        return lambda data: (data - self.upper_air_mean.reshape(len(self.upper_air_variables), -1, 1, 1))/ \
            self.upper_air_std.reshape(len(self.upper_air_variables), -1, 1, 1)

    def _create_surface_inv_transform(self):
        return lambda data: data * self.surface_std.reshape(-1, 1, 1) + self.surface_mean.reshape(-1, 1, 1)

    def _create_upper_air_inv_transform(self):
        return lambda data: data * self.upper_air_std.reshape(len(self.upper_air_variables), -1, 1, 1) + \
            self.upper_air_std.reshape(len(self.upper_air_variables), -1, 1, 1)

# If the order of the variables in the mean and standard deviation dictionaries is assumed to match the order in the dataset follow above or else
# below implementation :
    
    # def _create_surface_transform(self):
    #     mean_seq = [self.surface_mean[var] for var in self.channel_seq if var in self.surface_variables]
    #     std_seq = [self.surface_std[var] for var in self.channel_seq if var in self.surface_variables]
    #     return Normalize(mean_seq, std_seq)

    # def _create_upper_air_transform(self):
    #     normalize = {}
    #     for pl in self.upper_air_std:
    #         mean_seq = [self.upper_air_mean[var] for var in self.channel_seq if var in self.upper_air_variables]
    #         std_seq = [self.upper_air_std[pl] for _ in range(len(mean_seq))]
    #         normalize[pl] = Normalize(mean_seq, std_seq)
    #     return normalize

    #def surface_inv_transform(self):
    #    mean_seq = [self.surface_mean[var] for var in self.surface_variables]
    #    std_seq = [self.surface_std[var] for var in self.surface_variables]
    #    invTrans = Compose([
    #        Normalize([0.] * len(mean_seq), [1 / x for x in std_seq]),
    #        Normalize([-x for x in mean_seq], [1.] * len(std_seq))
    #    ])
    #    return invTrans

    #def upper_air_inv_transform(self):
    #    normalize = {}
    #    for pl in self.upper_air_std:
    #        mean_seq = [self.upper_air_mean[var] for var in self.upper_air_variables]
    #        std_seq = [self.upper_air_std[pl] for _ in self.upper_air_variables]
    #        invTrans = Compose([
    #            Normalize([0.] * len(mean_seq), [1 / x for x in std_seq]),
    #            Normalize([-x for x in mean_seq], [1.] * len(std_seq))
    #        ])
    #        normalize[pl] = invTrans
    #
    #    return normalize

    # def _load_boundary_data(self):
    #     # Check which variables have time axes and, if calendar has leap years, which variables are for leap years.
    #     boundary_files = [join(self.data_dir, self.boundary_dir, f) for f in os.listdir(join(self.data_dir, self.boundary_dir))]
    #     return xr.open_dataset(boundary_files, combine='nested', concat_dim='boundary')
    def _load_boundary_data(self):
        print('Loading varying boundary from %s' % join(self.data_dir, self.boundary_dir))
        boundary_files = [join(self.data_dir, self.boundary_dir, f) for f in \
                                 os.listdir(join(self.data_dir, self.boundary_dir)) \
                                 if any(var in f for var in self.varying_boundary_variables)]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message='^.*Unable to decode time axis into full numpy.datetime64 objects.*$')
            boundary_ds = xr.open_mfdataset(boundary_files, chunks={'time': 1}, engine = 'netcdf4', parallel=False)
        has_year_zero = boundary_ds.time[0].item().has_year_zero
        years = np.unique([date.year for date in boundary_ds.time.values])
        isleap = [self._check_leap_year(year, has_year_zero) for year in years]
        boundary_ds = boundary_ds.assign_attrs(noleap_year = years[np.logical_not(isleap)][0],
                                               leap_year = years[isleap][0])
        return boundary_ds

    def _get_dates(self):
        start_date = self.datetime_class(self.year_start, 1, 1)
        end_date = self.datetime_class(self.year_end, 1, 1)
        return xr.cftime_range(start_date, end_date, freq = '%dh' % self.timedelta_hours, calendar=self.calendar,
                               inclusive='left')

    def _check_leap_year(self, date, has_year_zero=None):
        if has_year_zero is None:
            return cftime.is_leap_year(date.year, calendar = self.calendar, has_year_zero=date.has_year_zero)
        else:
            return cftime.is_leap_year(date, calendar=self.calendar, has_year_zero=has_year_zero)

    def get_lat_lon(self):
        example_file = join(self.data_dir, f"data_{self.year_start}.nc")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message='^.*Unable to decode time axis into full numpy.datetime64 objects.*$')
            ds = xr.open_mfdataset(example_file, engine = 'netcdf4', parallel=False)
        return ds["lat"].values, ds["lon"].values

