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

import os, sys, gc, shutil
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
from datetime import timedelta
import xarray as xr
import warnings


def get_data_loader(params, files_pattern, distributed, year_start, year_end, train, num_inferences = 0, validate = False):

    dataset = GetDataset(params, files_pattern, year_start, year_end, train, num_inferences, validate)
    sampler = DistributedSampler(dataset, shuffle=train) if distributed else None
    if train and not distributed:
        sampler = torch.utils.data.RandomSampler(dataset)


    dataloader = DataLoader(dataset,
                            batch_size=int(params.batch_size),
                            num_workers=params.num_data_workers,
                            shuffle=False,  # (sampler is None),
                            sampler=sampler,# if train else None,
                            drop_last=True,
                            pin_memory=torch.cuda.is_available())

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset


class GetDataset(Dataset):
    def __init__(self, params, data_dir, year_start, year_end, train, num_inferences = 0, validate = False):
        self.params = params
        self.data_dir = data_dir
        self.train = train
        if not self.train:
            self.validate = validate
        else:
            self.validate = False
        if not self.train and not self.params.forecast_lead_times:
            self.params['forecast_lead_times'] = [1]
        self.epsilon_factor = self.params.epsilon_factor
        self.parallel = False #True if params['num_data_workers'] > 1 else False
        self.num_inferences = num_inferences

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
        if torch.any(torch.isnan(self.constant_boundary_data)):
            print('Constant boundary has nan')
            sys.exit(2)

        self.surface_mean, self.surface_std = self.load_mean_std(join(
            data_dir, params.surface_mean), join(data_dir, params.surface_std), self.surface_variables)

        self.upper_air_mean, self.upper_air_std = self.load_mean_std(join(
            data_dir, params.upper_air_mean), join(data_dir, params.upper_air_std), self.upper_air_variables)

        if 'surface_ff_std' in self.params:
            _, self.surface_ff_std = self.load_mean_std(join(
                data_dir, params.surface_mean), join(data_dir, params.surface_ff_std), self.surface_variables)
        if 'upper_air_ff_std' in self.params:
            _, self.upper_air_ff_std = self.load_mean_std(join(
                data_dir, params.upper_air_mean), join(data_dir, params.upper_air_ff_std), self.upper_air_variables)

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
        if self.num_inferences > 0:
            self.inference_idxs = np.linspace(0, len(self.dates), num = num_inferences + 1, dtype = int)
        else:
            self.inference_idxs = np.arange(0, len(self.dates))
        self.data_dss = self._load_data()
        self.lat = torch.from_numpy(self.data_dss[0].lat.values)
        self.lev = torch.from_numpy(self.data_dss[0].lev.values)
        if self.epsilon_factor > 0.:
            torch.manual_seed(0)
        for ds in self.data_dss:
            ds.close()
        for ds in self.boundary_dss:
            ds.close()
        gc.collect()


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
        print(constant_boundary_files)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message='^.*Unable to decode time axis into full numpy.datetime64 objects.*$')
            constant_boundary_ds = xr.open_mfdataset(constant_boundary_files, engine='netcdf4', parallel=self.parallel)
        print('Loaded Constant Boundary')
        constant_boundary_masked = []
        for var in self.constant_boundary_variables:
            constant_boundary_tensor = torch.from_numpy(constant_boundary_ds[var].values).to(torch.float32)
            nans = torch.isnan(constant_boundary_tensor)
            if torch.any(nans):
                constant_boundary_tensor = constant_boundary_tensor.masked_fill(nans, self.mask_fill[var])
            constant_boundary_masked.append(constant_boundary_tensor)
        constant_boundary_ds.close()
        constant_boundary_data = torch.stack(constant_boundary_masked, dim=0)
        constant_boundary_mean = torch.mean(constant_boundary_data, dim=(1,2))
        constant_bounadry_std = torch.std(constant_boundary_data, dim = (1,2))
        constant_boundary_data = (constant_boundary_data - constant_boundary_mean.reshape(-1, 1, 1)) / constant_bounadry_std.reshape(-1, 1, 1)
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
        return lambda data: data * self.surface_std.reshape(1, -1, 1, 1) + self.surface_mean.reshape(1, -1, 1, 1)

    def _create_upper_air_inv_transform(self):
        return lambda data: data * self.upper_air_std.reshape(1, len(self.upper_air_variables), -1, 1, 1) + \
            self.upper_air_mean.reshape(1, len(self.upper_air_variables), -1, 1, 1)
    
    def _load_boundary_data(self, initial = True):
        if initial:
            print('Loading varying boundary from %s' % join(self.data_dir, self.boundary_dir))
            self.boundary_files = [join(self.data_dir, self.boundary_dir, f) for f in os.listdir(join(self.data_dir, self.boundary_dir)) \
                                    if any(var in f for var in self.varying_boundary_variables)]
            self.boundary_leap_files = [file for file in self.boundary_files if '_leap' in os.path.basename(file)]
            self.boundary_noleap_files = [file for file in self.boundary_files if '_leap' not in os.path.basename(file)]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message='^.*Unable to decode time axis into full numpy.datetime64 objects.*$')
            boundary_ds_leap = xr.open_mfdataset(self.boundary_leap_files, chunks={'time': 1}, engine='netcdf4', parallel=self.parallel, decode_cf=False)
            boundary_ds_noleap = xr.open_mfdataset(self.boundary_noleap_files, chunks={'time': 1}, engine='netcdf4', parallel=self.parallel, decode_cf=False)
        return [boundary_ds_noleap, boundary_ds_leap]
    
    # def _get_dates(self, hour_step = 6.):
    #     start_date = self.datetime_class(self.year_start, 1, 1)
    #     end_date = self.datetime_class(self.year_end, 1, 1)
    #     if not self.train and self.params['inference_steps'] > 0:
    #         hours = (end_date - start_date).days * 24. - (self.params['inference_steps'])
    #     else:
    #         hours = (end_date - start_date).days * 24.
    #     date_range = np.arange(0., hours, hour_step)
    #     return date_range

    # Modification for the autoregressive parameter
    def _get_dates(self, hour_step=6.):
        start_date = self.datetime_class(self.year_start, 1, 1)
        end_date = self.datetime_class(self.year_end, 1, 1)
        
        if not self.train:
            # POTENTIAL BUG: WILL PRIORITIZE INFERENCE STEPS FIRST
            #if self.params['inference_steps'] > 0:
            #    # Original inference mode
            #    hours = (end_date - start_date).days * 24. - (self.params['inference_steps'])
            #elif self.params['autoreg_steps'] > 0:
            #    # New autoregressive inference mode
            #    hours = (end_date - start_date).days * 24. - (self.params['autoreg_steps'])
            #else:
            #    # Default case (no steps specified)
            #    hours = (end_date - start_date).days * 24.
            hours = (end_date - start_date).days * 24. - (max(self.params.forecast_lead_times) - 1) * hour_step
        else:
            # Training mode
            hours = (end_date - start_date).days * 24.
        
        date_range = np.arange(0., hours, hour_step)
        return date_range

    
    def _check_leap_year(self, date, has_year_zero=None):
        if has_year_zero is None:
            return cftime.is_leap_year(date.year, calendar = self.calendar, has_year_zero=date.has_year_zero)
        else:
            return cftime.is_leap_year(date, calendar=self.calendar, has_year_zero=has_year_zero)
    
    def _load_data(self, initial = True):
        if initial:
            self.data_files = [join(self.data_dir, f'data_{year}.nc') for year in range(self.year_start, self.year_end)]
            self.year_start_hours = np.array([(self.datetime_class(year, 1, 1) - self.datetime_class(self.year_start, 1, 1)).days*24.
                                    for year in range(self.year_start, self.year_end)])
            self.is_leap_year = np.array([self._check_leap_year(year, self.has_year_zero) for year in
                                range(self.year_start, self.year_end)])
        data_dss = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message='^.*Unable to decode time axis into full numpy.datetime64 objects.*$')
            for file in self.data_files:
                data_ds = xr.open_mfdataset(file, chunks={'time': 1, 'lev': self.num_levels}, engine='netcdf4', parallel=self.parallel, decode_cf=False)
                data_dss.append(data_ds)
        return data_dss

    def _load_year_data(self, year_idx):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message='^.*Unable to decode time axis into full numpy.datetime64 objects.*$')
            data_ds = xr.open_mfdataset(self.data_files[year_idx], chunks={'time': 1, 'lev': self.num_levels},
                engine='netcdf4', parallel=self.parallel, decode_cf=False)
        return data_ds
    
    def _get_data(self, data_ds, year, hour):

        surface_da_list = [data_ds[var].sel(time=hour) for var in self.surface_variables]
        surface_data = torch.stack([torch.from_numpy(da.values).to(torch.float32) for da in surface_da_list], dim = 0)
        surface_data = self.surface_transform(surface_data)
        #for da in surface_da_list:
        #    da[:] = np.nan

        upper_air_da_list = [data_ds[var].sel(time=hour) for var in self.upper_air_variables]
        upper_air_data = torch.stack([torch.from_numpy(da.values).to(torch.float32) for da in upper_air_da_list], dim = 0)
        upper_air_data = self.upper_air_transform(upper_air_data)
        #for da in upper_air_da_list:
        #    da[:] = np.nan

        #gc.collect()
        
        return surface_data, upper_air_data



    def _get_boundary_data(self, start_time_boundary, leap_idx):
        # Added fix for boundary data
        # load boundary data based on if multi step or single step. 
        if not self.train and max(self.params.forecast_lead_times) > 1: 
            varying_boundary_data_all = []
            # print(start_time_boundary)
            for start_time_boundary_i, leap_idx_i in zip(start_time_boundary, leap_idx):
                varying_boundary_masked = []
                for var in self.varying_boundary_variables:
                    varying_boundary_tensor = torch.from_numpy(
                        self.boundary_dss[leap_idx_i][var].sel(time=start_time_boundary_i).values).to(torch.float32)
                    nans = torch.isnan(varying_boundary_tensor)
                    if torch.any(nans):
                        varying_boundary_tensor = varying_boundary_tensor.masked_fill(nans, self.mask_fill[var])
                    varying_boundary_masked.append(varying_boundary_tensor)
                varying_boundary_data = torch.stack(varying_boundary_masked, dim = 0)
                varying_boundary_data_all.append(varying_boundary_data)
            varying_boundary_data_out = torch.stack(varying_boundary_data_all, dim = 0)
            # print(varying_boundary_data_out.shape)
            return varying_boundary_data_out
        else:
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
        if self.num_inferences > 0:
            return len(self.inference_idxs) - 1
        else:
            return len(self.dates) - 1


    def __getitem__(self, index):
        self.boundary_dss = self._load_boundary_data(initial = False)
        #self.dates = self._get_dates(hour_step=params.timedelta_hours)
        #self.data_dss = self._load_data(initial=False)
        #self.lat = torch.from_numpy(self.data_dss[0].lat.values)
        #self.lev = torch.from_numpy(self.data_dss[0].lev.values)
        lead_times = self.params.forecast_lead_times

        # Condition 1: Training
        if self.train:
            start_time = self.dates[index]
            start_hour_diff = start_time - self.year_start_hours
            start_idx = np.where(start_hour_diff >= 0)[0][-1]
            start_leap_idx = 1 if self.is_leap_year[start_idx] else 0
            end_time = self.dates[index + 1]
            end_hour_diff = end_time - self.year_start_hours
            end_idx = np.where(end_hour_diff >= 0)[0][-1]
            if start_idx == end_idx:
                data_ds = self._load_year_data(start_idx)
                surface_t, upper_air_t = self._get_data(data_ds, start_idx, start_hour_diff[start_idx])
                surface_t_1, upper_air_t_1 = self._get_data(data_ds, end_idx, end_hour_diff[end_idx])
                data_ds.close()
            else:
                data_ds_start = self._load_year_data(start_idx)
                data_ds_end = self._load_year_data(end_idx)
                surface_t, upper_air_t = self._get_data(data_ds_start, start_idx, start_hour_diff[start_idx])
                surface_t_1, upper_air_t_1 = self._get_data(data_ds_end, end_idx, end_hour_diff[end_idx])
                data_ds_start.close()
                data_ds_end.close()
            varying_boundary_data = self._get_boundary_data(start_hour_diff[start_idx], start_leap_idx)
            varying_boundary_data = self.boundary_transform(varying_boundary_data)
            if self.epsilon_factor > 0.:
                if 'surface_ff_std' in self.params:
                    surface_t_noise = torch.randn(*surface_t.shape) * (self.epsilon_factor * self.surface_ff_std / self.surface_std).reshape(len(self.surface_variables), 1, 1)
                else:
                    surface_t_noise = torch.randn(*surface_t.shape) * self.epsilon_factor
                surface_t = surface_t + surface_t_noise
                if 'upper_air_ff_std' in self.params:
                    upper_air_t_noise = torch.randn(*upper_air_t.shape) * (self.epsilon_factor * self.upper_air_ff_std / self.upper_air_std).reshape(len(self.upper_air_variables), self.num_levels, 1, 1)
                else:
                    upper_air_t_noise = torch.randn(*upper_air_t.shape) * self.epsilon_factor
                upper_air_t = upper_air_t + upper_air_t_noise
        
        # Condition for autoregression
        elif lead_times:

            start_time = self.dates[self.inference_idxs[index]]
            start_hour_diff = start_time - self.year_start_hours
            start_idx = np.where(start_hour_diff >= 0)[0][-1]
            start_leap_idx = 1 if self.is_leap_year[start_idx] else 0

            # Load initial conditions
            data_ds_start = self._load_year_data(start_idx)
            surface_t, upper_air_t = self._get_data(data_ds_start, start_idx, start_hour_diff[start_idx])

            # # The final target is now the last element in the lists
            # surface_t_target, upper_air_t_target = targets_surface[-1], targets_upper_air[-1]

            # # Load target conditions
            # if start_idx == end_idx:
            #     surface_t_target, upper_air_t_target = self._get_data(data_ds_start, end_idx, end_hour_diff[end_idx])
            #     data_ds_start.close()
            # else:
            #     data_ds_end = self._load_year_data(end_idx)
            #     surface_t_target, upper_air_t_target = self._get_data(data_ds_end, end_idx, end_hour_diff[end_idx])
            #     data_ds_start.close()
            #     data_ds_end.close()
            max_lead_time = lead_times[-1]
            boundary_times = self.dates[self.inference_idxs[index]:self.inference_idxs[index] + max_lead_time]
            boundary_hour_diffs = np.array(boundary_times).reshape(-1,1) - self.year_start_hours.reshape(1,-1)
            boundary_idxs = np.array([np.where(diff >= 0)[0][-1] for diff in boundary_hour_diffs])
            boundary_leap_idxs = np.array([1 if self.is_leap_year[idx] else 0 for idx in boundary_idxs])

            varying_boundary_data = self._get_boundary_data(boundary_hour_diffs[np.arange(len(boundary_times)), boundary_idxs], boundary_leap_idxs)
            varying_boundary_data = torch.stack([self.boundary_transform(varying_boundary_data[i]) for i in range(varying_boundary_data.shape[0])], dim=0)

            if self.validate:
                # Load targets for each lead time
                targets_surface = []
                targets_upper_air = []
                current_ds = data_ds_start
                current_idx = start_idx

                for step in lead_times:
                    target_time = self.dates[self.inference_idxs[index] + step]
                    target_hour_diff = target_time - self.year_start_hours
                    target_idx = np.where(target_hour_diff >= 0)[0][-1]
                    
                    if target_idx != current_idx:
                        current_ds.close()
                        current_ds = self._load_year_data(target_idx)
                        current_idx = target_idx
                    
                    surface_target, upper_air_target = self._get_data(current_ds, target_idx, target_hour_diff[target_idx])
                    
                    targets_surface.append(surface_target)
                    targets_upper_air.append(upper_air_target)
                current_ds.close()
                targets_surface = torch.stack(targets_surface, dim = 0)
                targets_upper_air = torch.stack(targets_upper_air, dim = 0)
            else:
                current_ds.close()


        # inference steps > 1
        #elif not self.train and self.params['inference_steps'] > 1:
        #    start_time = np.array(self.dates[self.inference_idxs[index]:self.inference_idxs[index] + self.params['inference_steps']])
        #    start_hour_diff = start_time.reshape(-1,1) - self.year_start_hours.reshape(1,-1)
        #    start_idx = np.array([np.where(start_hour_diff[i] >= 0)[0][-1] for i in range(start_hour_diff.shape[0])])
        #    start_leap_idx = np.array([1 if self.is_leap_year[start_idx_i] else 0 for start_idx_i in start_idx])
        #    data_ds = self._load_year_data(start_idx[0])
        #    surface_t, upper_air_t = self._get_data(data_ds, start_idx[0], start_hour_diff[0][start_idx[0]])
        #    data_ds.close()
        #    varying_boundary_data = self._get_boundary_data(start_hour_diff[np.arange(len(start_time)), start_idx], start_leap_idx)
        #    varying_boundary_data = torch.stack([self.boundary_transform(varying_boundary_data[i]) for i in range(varying_boundary_data.shape[0])], dim = 0)
        #    #start_time_cf = self.datetime_class(start_idx[0] + self.params.val_year_start, 1, 1, hour = 0) + timedelta(start_hour_diff[0][start_idx[0]])
        #    #end_time = self.dates[index + 1:index + 1 + self.params['inference_steps']]
        
        # when inference steps is 1
        else:
            start_time = self.dates[self.inference_idxs[index]]
            start_hour_diff = start_time - self.year_start_hours
            start_idx = np.where(start_hour_diff >= 0)[0][-1]
            start_leap_idx = 1 if self.is_leap_year[start_idx] else 0
            data_ds = self._load_year_data(start_idx)
            surface_t, upper_air_t = self._get_data(data_ds, start_idx, start_hour_diff[start_idx])
            data_ds.close()
            varying_boundary_data = self._get_boundary_data(start_hour_diff[start_idx], start_leap_idx)
            varying_boundary_data = self.boundary_transform(varying_boundary_data).unsqueeze(0)
            #start_time_cf = self.datetime_class(start_idx[0] + self.params.val_year_start, 1, 1, hour = 0) + timedelta(start_hour_diff[0][start_idx[0]])
            #end_time = [self.dates[index + 1]]
        #varying_boundary_data = self._get_boundary_data(start_hour_diff[start_idx], start_leap_idx)
        #varying_boundary_data = self.boundary_transform(varying_boundary_data)
        if torch.any(torch.isnan(varying_boundary_data)):
            print('Boundary data has nan')
            sys.exit(2)
        if torch.any(torch.isnan(surface_t)):
            print('Surface has nan')
            sys.exit(2)
        if torch.any(torch.isnan(upper_air_t)):
            print('Upper air has nan')
            sys.exit(2)
        for ds in self.boundary_dss:
            ds.close()
        gc.collect()

        if self.train:
            return surface_t, upper_air_t, surface_t_1, upper_air_t_1, varying_boundary_data, torch.tensor([index, start_time, start_idx, start_leap_idx, start_hour_diff[start_idx], end_time, end_idx, end_hour_diff[end_idx]])
        elif self.validate and lead_times:
            # return surface_t, upper_air_t, surface_t_target, upper_air_t_target, varying_boundary_data, torch.tensor([index, start_time, start_idx, start_leap_idx, start_hour_diff[start_idx], end_time, end_idx, end_hour_diff[end_idx], self.params['autoreg_steps']])
            return surface_t, upper_air_t, targets_surface, targets_upper_air, varying_boundary_data, torch.tensor([index, start_time, start_idx, start_leap_idx, start_hour_diff[start_idx]])
        elif lead_times:
            return surface_t, upper_air_t, varying_boundary_data, torch.tensor([start_idx[0], start_hour_diff[0][start_idx[0]]])
        else:
            return surface_t, upper_air_t, surface_t_1, upper_air_t_1, varying_boundary_data, torch.tensor([start_time, end_time])
