# From FourCastNet repo


# BSD 3-Clause License
#
# Copyright (c) 2022, FourCastNet authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The code was authored by the following people:
#
# Jaideep Pathak - NVIDIA Corporation
# Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
# Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
# Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory
# Ashesh Chattopadhyay - Rice University
# Morteza Mardani - NVIDIA Corporation
# Thorsten Kurth - NVIDIA Corporation
# David Hall - NVIDIA Corporation
# Zongyi Li - California Institute of Technology, NVIDIA Corporation
# Kamyar Azizzadenesheli - Purdue University
# Pedram Hassanzadeh - Rice University
# Karthik Kashinath - NVIDIA Corporation
# Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation

import os
import logging
import glob
import torch
#import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
#from torch import Tensor
#import h5py
#import math
# import cv2
#from utils.img_utils import reshape_fields

from os.path import join
import cftime
import xarray as xr
import warnings


def get_data_loader(params, world_rank, files_pattern, distributed, year_start, year_end, train):

    dataset = GetDataset(params, files_pattern, year_start, year_end, train)
    # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    sampler = DistributedSampler(dataset, num_replicas=params['world_size'], rank=world_rank, shuffle=train) if distributed else None

    dataloader = DataLoader(dataset,
                            batch_size=int(params.batch_size),
                            num_workers=params.num_data_workers,
                            shuffle=False,  # (sampler is None),
                            sampler=sampler if train else None,
                            drop_last=True,
                            pin_memory=torch.cuda.is_available())

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset


class GetDataset(Dataset):
    def __init__(self, params, data_dir, year_start, year_end, train):
        self.params = params
        self.data_dir = data_dir
        self.train = train

        #self._get_files_stats()

        self.has_year_zero = params.has_year_zero
        self.mask_fill = {'lsm': 0., 'sst': 270., 'sic': 0., }

        self.year_start = year_start
        self.year_end = year_end
        self.calendar = params.calendar
        self.timedelta_hours = params.timedelta_hours
        self.datetime_class = self.datetime_class_from_calendar(self.calendar)
        self.timedelta = self.datetime_class(1, 1, 1, hour=self.timedelta_hours) - self.datetime_class(1, 1, 1, hour=0)

        self.surface_variables = params.surface_variables or []
        self.upper_air_variables = params.upper_air_variables or []

        self.constant_boundary_variables = params.constant_boundary_variables or []
        self.varying_boundary_variables = params.varying_boundary_variables or []
        self.boundary_dir = params.boundary_dir
        self.constant_boundary_data = self._load_constant_boundary_data()

        self.surface_mean, self.surface_std = self.load_mean_std(join(
            data_dir, params.surface_mean), join(data_dir, params.surface_std), self.surface_variables)

        self.upper_air_mean, self.upper_air_std = self.load_mean_std(join(
            data_dir, params.upper_air_mean), join(data_dir, params.upper_air_std), self.upper_air_variables)

        self.varying_boundary_mean, self.varying_boundary_std = self.load_mean_std(join(data_dir, params.boundary_dir, params.boundary_mean),
                                                                                   join(data_dir, params.boundary_dir, params.boundary_std),
                                                                                   self.varying_boundary_variables)
        self.num_levels = self.upper_air_mean.size(-1)
        self.surface_transform = self._create_surface_transform()
        self.boundary_transform = self._create_boundary_transform()
        self.upper_air_transform = self._create_upper_air_transform()
        self.surface_inv_transform = self._create_surface_inv_transform()
        self.upper_air_inv_transform = self._create_upper_air_inv_transform()
        # self.channel_seq = self.surface_variables + self.upper_air_variables
        self.boundary_dss = self._load_boundary_data()
        self.dates = self._get_dates(hour_step=params.timedelta_hours)
        self.data_dss = self._load_data()


    def _get_files_stats(self):
        self.files_paths_sfc = glob.glob(self.data_dir + "/*_sfc.h5")
        self.files_paths_pl = glob.glob(self.data_dir + "/*_pl.h5")
        self.files_paths_sfc.sort()
        self.files_paths_pl.sort()
        assert len(self.files_paths_sfc) == len(self.files_paths_pl), 'number of surface and upper_air files must be equal'
        self.n_years = len(self.files_paths_sfc)
        with h5py.File(self.files_paths_sfc[0], 'r') as _f:
            logging.info("Getting file stats from {}".format(self.files_paths_sfc[0]))
            self.n_samples_per_year = _f['fields'].shape[0]
            self.N_channel = _f['fields'].shape[1]
            # original image shape (before padding)
            # -1#just get rid of one of the pixels
            self.img_shape_x = _f['fields'].shape[2]
            self.img_shape_y = _f['fields'].shape[3]

        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files_sfc = [None for _ in range(self.n_years)]
        self.files_pl = [None for _ in range(self.n_years)]
        logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
        logging.info("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(
            self.data_dir, self.n_samples_total, self.N_channel, self.img_shape_x, self.img_shape_y))
        logging.info("Delta t: {} hours".format(6*self.dt))
        logging.info("Including {} hours of past history in training at a frequency of {} hours".format(6*self.dt*self.n_history, 6*self.dt))


    def datetime_class_from_calendar(self, calendar):
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

    def _load_constant_boundary_data(self):
        constant_boundary_files = [join(self.data_dir, self.boundary_dir, f) for f in
                                   os.listdir(join(self.data_dir, self.boundary_dir))
                                   if any(var in f for var in self.constant_boundary_variables)]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message='^.*Unable to decode time axis into full numpy.datetime64 objects.*$')
            constant_boundary_ds = xr.open_mfdataset(constant_boundary_files, engine='netcdf4', parallel=True)
        constant_boundary_masked = []
        for var in self.constant_boundary_variables:
            constant_boundary_tensor = torch.from_numpy(constant_boundary_ds[var].values).to(torch.float32)
            nans = torch.isnan(constant_boundary_tensor)
            if torch.any(nans):
                constant_boundary_tensor = constant_boundary_tensor.masked_fill(nans, self.mask_fill[var])
            constant_boundary_masked.append(constant_boundary_tensor)
        constant_boundary_data = torch.stack(constant_boundary_masked, dim=0)
        return constant_boundary_data

    def load_mean_std(self, mean_file, std_file, datavars):
        with xr.open_dataset(mean_file) as ds:
            mean = torch.stack([torch.from_numpy(ds[var].values).to(torch.float32) for var in datavars], dim=0)
        with xr.open_dataset(std_file) as ds:
            std = torch.stack([torch.from_numpy(ds[var].values).to(torch.float32) for var in datavars], dim=0)
        return mean, std
    
    def _create_surface_transform(self):
        return lambda data: (data - self.surface_mean.reshape(-1, 1, 1))/self.surface_std.reshape(-1, 1, 1)

    def _create_boundary_transform(self):
        return lambda data: (data - self.varying_boundary_mean.reshape(-1, 1, 1))/self.varying_boundary_std.reshape(-1, 1, 1)

    def _create_upper_air_transform(self):
        return lambda data: (data - self.upper_air_mean.reshape(len(self.upper_air_variables), -1, 1, 1))/ \
            self.upper_air_std.reshape(len(self.upper_air_variables), -1, 1, 1)

    def _create_surface_inv_transform(self):
        return lambda data: data * self.surface_std.reshape(-1, 1, 1) + self.surface_mean.reshape(-1, 1, 1)

    def _create_upper_air_inv_transform(self):
        return lambda data: data * self.upper_air_std.reshape(len(self.upper_air_variables), -1, 1, 1) + \
            self.upper_air_std.reshape(len(self.upper_air_variables), -1, 1, 1)
    
    def _load_boundary_data(self):
        print('Loading varying boundary from %s' % join(self.data_dir, self.boundary_dir))
        boundary_files = [join(self.data_dir, self.boundary_dir, f) for f in os.listdir(join(self.data_dir, self.boundary_dir)) \
                                 if any(var in f for var in self.varying_boundary_variables)]
        boundary_leap_files = [file for file in boundary_files if '_leap' in os.path.basename(file)]
        boundary_noleap_files = [file for file in boundary_files if '_leap' not in os.path.basename(file)]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message='^.*Unable to decode time axis into full numpy.datetime64 objects.*$')
            boundary_ds_leap = xr.open_mfdataset(boundary_leap_files, chunks={'time': 1}, engine='netcdf4', parallel=True, decode_cf=False)
            boundary_ds_noleap = xr.open_mfdataset(boundary_noleap_files, chunks={'time': 1}, engine='netcdf4', parallel=True, decode_cf=False)
        return [boundary_ds_noleap, boundary_ds_leap]
    
    def _get_dates(self, hour_step = 6.):
        start_date = self.datetime_class(self.year_start, 1, 1)
        end_date = self.datetime_class(self.year_end, 1, 1)
        hours = (end_date - start_date).days * 24.
        date_range = np.arange(0., hours, hour_step)
        return date_range
    
    def _check_leap_year(self, date, has_year_zero=None):
        if has_year_zero is None:
            return cftime.is_leap_year(date.year, calendar = self.calendar, has_year_zero=date.has_year_zero)
        else:
            return cftime.is_leap_year(date, calendar=self.calendar, has_year_zero=has_year_zero)
    
    def _load_data(self):
        data_files = [join(self.data_dir, f'data_{year}.nc') for year in range(self.year_start, self.year_end)]
        self.year_start_hours = [(self.datetime_class(year, 1, 1) - self.datetime_class(self.year_start, 1, 1)).days*24.
                                 for year in range(self.year_start, self.year_end)]
        self.is_leap_year = [self._check_leap_year(year, self.has_year_zero) for year in
                             range(self.year_start, self.year_end)]
        data_dss = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message='^.*Unable to decode time axis into full numpy.datetime64 objects.*$')
            for file in data_files:
                data_ds = xr.open_mfdataset(file, chunks={'time': 1, 'lev': self.num_levels}, engine='netcdf4', parallel=True, decode_cf=False)
                data_dss.append(data_ds)
        return data_dss
    
    def _get_data(self, year, hour):

        surface_data = torch.stack([torch.from_numpy(
            self.data_dss[year][var].sel(time=hour).values).to(torch.float32) for var in self.surface_variables], dim = 0)
        surface_data = self.surface_transform(surface_data)

        upper_air_data = torch.stack([
            torch.from_numpy(self.data_dss[year][var].sel(time=hour).values).to(torch.float32)
            for var in self.upper_air_variables], dim = 0)
        upper_air_data = self.upper_air_transform(upper_air_data)
        
        return surface_data, upper_air_data



    def _get_boundary_data(self, start_time_boundary, leap_idx):
        varying_boundary_masked = []
        for var in self.varying_boundary_variables:
            varying_boundary_tensor = torch.from_numpy(
                self.boundary_dss[leap_idx][var].sel(time=start_time_boundary).values).to(torch.float32)
            nans = torch.isnan(varying_boundary_tensor)
            if torch.any(nans):
                varying_boundary_tensor = varying_boundary_tensor.masked_fill(nans, self.mask_fill[var])
            varying_boundary_masked.append(varying_boundary_tensor)
        varying_boundary_data = torch.stack(varying_boundary_masked, dim = 0)
        return varying_boundary_data
    


    def __len__(self):
        return len(self.dates) - 1


    def __getitem__(self, index):
        start_time = self.dates[index]
        end_time = self.dates[index + 1]
        start_hour_diff = start_time - self.year_start_hours
        start_idx = np.where(start_hour_diff >= 0)[0][-1]
        start_leap_idx = 1 if self.is_leap_year[start_idx] else 0
        end_hour_diff = end_time - self.year_start_hours
        end_idx = np.where(end_hour_diff >= 0)[0][-1]
        varying_boundary_data = self._get_boundary_data(start_hour_diff[start_idx], start_leap_idx)
        varying_boundary_data = self.boundary_transform(varying_boundary_data)
        surface_t, upper_air_t = self._get_data(start_idx, start_hour_diff[start_idx])
        surface_t_1, upper_air_t_1 = self._get_data(end_idx, end_hour_diff[end_idx])

        if self.train:
            return surface_t, upper_air_t, surface_t_1, upper_air_t_1, varying_boundary_data
        return surface_t, upper_air_t, surface_t_1, upper_air_t_1, varying_boundary_data, torch.tensor([start_time, end_time])
