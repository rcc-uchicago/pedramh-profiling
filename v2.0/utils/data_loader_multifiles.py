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
import h5py
#import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
#from torch import Tensor
#import h5py
#import math
# import cv2
#from utils.img_utils import reshape_fields
from itertools import product
from os.path import join
import cftime
from datetime import timedelta
import xarray as xr
import warnings
from tqdm import tqdm
import netCDF4 as nc

def get_data_given_path(path, variables):
    with h5py.File(path, 'r') as f:
        data = {
            main_key: {
                sub_key: np.array(value) for sub_key, value in group.items() if sub_key in variables + ['time']
        } for main_key, group in f.items() if main_key in ['input']}

    x = [data['input'][v] for v in variables]
    return np.stack(x, axis=0)

def get_data_given_path_nc(path, variables_3D, variables_2D, timestep_offset = -1):
    x = []
    with nc.Dataset(path, 'r') as f:
        if variables_3D:
            for variable in variables_3D:
                if len(f.variables[variable].shape) == 4:
                    x.append(f.variables[variable][timestep_offset,:])
                else:
                    x.append(f.variables[variable][:])
        if variables_2D:
            for variable in variables_2D:
                if len(f.variables[variable].shape) == 3:
                    x_in = f.variables[variable][timestep_offset,:]
                else:
                    x_in = f.variables[variable][:]
                x.append(x_in.reshape(1, x_in.shape[0], x_in.shape[1]))
    return np.concatenate(x, axis=0)

def get_out_path(root_dir, year, inp_file_idx):
    # year: current year
    # inp_file_idx: file index of the input in the current year
    # steps: number of steps forward
    out_path = os.path.join(root_dir, f'{year}_{inp_file_idx:04}.h5')
    return out_path



def get_data_loader(params, files_pattern, distributed, year_start, year_end, train, num_inferences = 0, validate = False, single_ic = False,
                    ensemble = False, init_from_nc = False, load_all_bcs = True):
    if train:
        try:
            assert not single_ic
        except:
            raise ValueError('single_ic cannot be True when train is True.')
    if validate:
        try:
            assert not single_ic
        except:
            raise ValueError('Set validate to False when using single_ic = True.')
    dataset = GetDataset(params, files_pattern, year_start, year_end, train, num_inferences, validate, single_ic, ensemble, init_from_nc, load_all_bcs)
    if single_ic:
        dataloader = DataLoader(dataset, batch_size = 1, num_workers = 1, shuffle = False, pin_memory=torch.cuda.is_available())
    else:
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
    def __init__(self, params, data_dir, year_start, year_end, train, num_inferences = 0, validate = False, single_ic = False,
                 ensemble = False, init_from_nc = False, load_all_bcs = True, ):
        self.params = params
        self.data_dir = data_dir
        self.train = train
        self.single_ic = single_ic
        if not self.train and not self.single_ic:
            self.validate = validate
        else:
            self.validate = False
        self.ensemble = ensemble
        if self.ensemble:
            self.load_all_bcs = load_all_bcs
        self.init_from_nc = init_from_nc
        if self.ensemble:
            assert self.init_from_nc
            assert hasattr(self.params, "init_nc_filepaths")
            self.init_nc_timestep_offset = -1
            if hasattr(self.params, "init_nc_timestep_offset"):
                self.init_nc_timestep_offset = self.params.init_nc_timestep_offset
        if not self.train and not self.params.forecast_lead_times:
            self.params['forecast_lead_times'] = [1]
        if self.single_ic or self.ensemble:
            self.no_leap_year       = 51
            self.leap_year          = 52
        if self.single_ic:
            self.single_ic_offset   = 0
            self.long_rollout_years = 1
            self.nc_bc_offset       = 0
            if hasattr(params, 'long_rollout_years'):
                self.long_rollout_years = self.params.long_rollout_years
            if hasattr(params, 'no_leap_year'):
                self.no_leap_year = self.params.no_leap_year
            if hasattr(params, 'leap_year'):
                self.leap_year = self.params.leap_year
            if hasattr(params, 'single_ic_offset'):
                self.single_ic_offset = self.params.single_ic_offset
            if hasattr(params, 'nc_bc_offset'):
                self.nc_bc_offset = self.params.nc_bc_offset
            
            
        self.epsilon_factor = self.params.epsilon_factor
        self.parallel = False #True if params['num_data_workers'] > 1 else False
        self.num_inferences = num_inferences

        #self._get_files_stats()

        self.has_year_zero = params.has_year_zero
        self.land_sea_mask_name = 'lsm'
        if hasattr(params, 'mask_fill'):
            self.mask_fill = self.params.mask_fill
        else:
            self.mask_fill = {'alb': 0.069, 'dlai': 0., 'glac': 0., 'lsm': 0., 'mrfc': 0., 'mrso': 0., 
                              'sic': 0., 'sst': 270., 'ts': 270, 'vegc': 0., 'vegf': 0.}
            #self.mask_fill = {'lsm': 0., 'sst': 270., 'sea_ice_cover': 0., 'volumetric_soil_water_layer_1': 0.}

        self.year_start = year_start
        self.year_end = year_end
        self.calendar = params.calendar
        self.timedelta_hours = params.timedelta_hours
        self.data_timedelta_hours = params.data_timedelta_hours
        self.datetime_class  = self.datetime_class_from_calendar(self.calendar)
        # for timedelta_hours > 24
        days, hours = divmod(self.timedelta_hours, 24)
        self.timedelta = self.datetime_class(1, 1, 1 + days, hour=hours) - self.datetime_class(1, 1, 1, hour=0)
        # self.timedelta = self.datetime_class(1, 1, 1, hour=self.timedelta_hours) - self.datetime_class(1, 1, 1, hour=0)

        self.surface_variables = params.surface_variables or []
        if hasattr(params, 'land_variables'):
            if len(params.land_variables) > 0:
                if any([land_variable in self.surface_variables for land_variable in params.land_variables]):
                    raise ValueError('land variables cannot be in surface variables.')
                self.surface_variables = self.surface_variables + params.land_variables
            else:
                self.land_variables = []
            self.land_variables = params.land_variables
        else:
            self.land_variables = []
            

        if hasattr(params, 'ocean_variables'):
            if len(params.ocean_variables) > 0:
                if any([ocean_variable in self.surface_variables for ocean_variable in params.ocean_variables]):
                    raise ValueError('ocean variables cannot be in surface variables.')
                self.ocean_variables = params.ocean_variables
            else:
                self.ocean_variables = []
            self.surface_variables = self.surface_variables + params.ocean_variables
        else:
            self.ocean_variables = []
        self.upper_air_variables = params.upper_air_variables or []

        self.constant_boundary_variables = params.constant_boundary_variables or []
        self.varying_boundary_variables = params.varying_boundary_variables or []



        # self.channel_seq = self.surface_variables + self.upper_air_variables

        # self.boundary_dss = self._load_boundary_data()
        if self.single_ic:
            self.dates, self.start_date, self.end_date = self._get_dates(hour_step=params.timedelta_hours)
        elif self.ensemble:
            self.ensemble_inference_steps = self.params.ensemble_inference_hours // self.params.timedelta_hours
            self.dates = np.zeros(len(self.params.init_nc_filepaths))
            print(f'Num dates: {len(self.dates)}')
            self.start_date = self.params.init_datetime
        else:
            self.dates, self.start_date, self.end_date = self._get_dates(hour_step=params.data_timedelta_hours) #(hour_step=params.timedelta_hours)

        self.constant_boundary_data, self.land_mask = self._load_constant_boundary_data()
        if torch.any(torch.isnan(self.constant_boundary_data)):
            print('Constant boundary has nan')
            sys.exit(2)
        
        if self.single_ic or self.ensemble:
            self.inference_idxs = np.arange(0, len(self.dates))
        else:
            max_inference_idx = len(self.dates) - max(self.params.forecast_lead_times) * self.timedelta_hours // self.data_timedelta_hours
            if self.num_inferences > 0:
                self.inference_idxs = np.linspace(0, max_inference_idx, num = num_inferences + 1, dtype = int)
            else:
                self.inference_idxs = np.arange(0, max_inference_idx)
        print(f'Len(inference_idxs): {len(self.inference_idxs)}')
        #print('Inference idxs:')
        #print(self.inference_idxs)
        #self.data_dss = self._load_data()
        #self.lat = torch.from_numpy(self.data_dss[0].lat.values)
        if len(params['levels']) > 0:
            self.levels = np.array(params['levels'])
        else:
            #self.levels = self.data_dss[0][self.params.lev].values
            raise ValueError('levels must now be explicitly specified in config file.')
        
        self.use_sigma_levels = False
        if hasattr(params, 'use_sigma_levels'):
            self.use_sigma_levels = params.use_sigma_levels
            if self.use_sigma_levels:
                try:
                    self.sigma_levels = np.array(params.sigma_levels)
                except:
                    raise ValueError('use_sigma_levels is True, but no sigma_levels are specified.')
                try:
                    assert len(self.sigma_levels) == len(self.levels)
                except:
                    raise ValueError('sigma_levels and levels must be the same length and ordered from top to bottom.')
                
        
        
        self.surface_mean, self.surface_std = self.load_mean_std(join(data_dir, params.surface_mean),
                                                                 join(data_dir, params.surface_std), self.surface_variables,
                                                                 upper_air = False,
                                                                 use_sigma_levels=self.use_sigma_levels)

        self.upper_air_mean, self.upper_air_std = self.load_mean_std(join(
            data_dir, params.upper_air_mean), join(data_dir, params.upper_air_std), self.upper_air_variables, use_sigma_levels=self.use_sigma_levels)

        if 'surface_ff_std' in self.params:
            _, self.surface_ff_std = self.load_mean_std(join(
                data_dir, params.surface_mean), join(data_dir, params.surface_ff_std), self.surface_variables, upper_air = False)
        if 'upper_air_ff_std' in self.params:
            _, self.upper_air_ff_std = self.load_mean_std(join(
                data_dir, params.upper_air_mean), join(data_dir, params.upper_air_ff_std), self.upper_air_variables, use_sigma_levels=self.use_sigma_levels)

        if self.params.predict_delta:
            _, self.surface_delta_std = self.load_mean_std(join(
                data_dir, params.surface_mean), join(data_dir, params.surface_delta_std), self.surface_variables, upper_air = False)
            _, self.upper_air_delta_std = self.load_mean_std(join(
                data_dir, params.upper_air_mean), join(data_dir, params.upper_air_delta_std), self.upper_air_variables, use_sigma_levels=self.use_sigma_levels)

        self.varying_boundary_mean, self.varying_boundary_std = self.load_mean_std(join(data_dir, params.boundary_mean),
                                                                                   join(data_dir, params.boundary_std),
                                                                                   self.varying_boundary_variables, upper_air = False)
        
        
        if hasattr(params, 'diagnostic_variables'):
            if len(params.diagnostic_variables) > 0:
                self.diagnostic_variables = params.diagnostic_variables
                self.diagnostic_mean, self.diagnostic_std = self.load_mean_std(join(data_dir, params.diagnostic_mean),
                                                                                join(data_dir, params.diagnostic_std),
                                                                                self.diagnostic_variables, upper_air = False)
            else:
                self.diagnostic_variables = []
        else:
            self.diagnostic_variables = []

        if self.init_from_nc:
            self._get_variable_list_nc()
        else:
            self._get_variable_list()
            

        #self.surface_transform = self._create_surface_transform()
        #self.boundary_transform = self._create_boundary_transform()
        #self.upper_air_transform = self._create_upper_air_transform()
        #self.surface_inv_transform = self._create_surface_inv_transform()
        #self.upper_air_inv_transform = self._create_upper_air_inv_transform()

        if self.epsilon_factor > 0.:
            torch.manual_seed(0)
            
        if self.ensemble and self.load_all_bcs:
            self.varying_boundary_data = torch.zeros((self.ensemble_inference_steps, len(self.params.varying_boundary_variables), self.params.horizontal_resolution[0], self.params.horizontal_resolution[1]), dtype = torch.float32)
            for i, hour in tqdm(enumerate(range(0, self.params.ensemble_inference_hours, self.params.timedelta_hours)), total = self.ensemble_inference_steps, desc = 'Loading boundary data'):
                data_datetime = self.start_date + timedelta(hours=hour)
                varying_boundary_data = self._get_boundary_data(data_datetime)
                varying_boundary_data = self._fill_mask(varying_boundary_data, self.varying_boundary_variables)
                self.varying_boundary_data[i] = self.boundary_transform(varying_boundary_data)
                

    def _get_variable_list(self, level_unit_str = '.0'):
        self.level_unit_str = level_unit_str
        self.variable_list_out = []
        for variable in self.upper_air_variables:
            if variable != 'zg' and variable != 'geopotential_height' and self.use_sigma_levels:
                for level in self.sigma_levels:
                    self.variable_list_out.append(f'{variable}_{level}')
            else:
                for level in self.levels:
                    self.variable_list_out.append(f'{variable}_{int(level)}{level_unit_str}')
        self.upper_air_len = len(self.variable_list_out)
        self.variable_list_out.extend(self.surface_variables)
        self.variable_list_in = self.variable_list_out.copy()
        self.variable_list_out.extend(self.diagnostic_variables)
        self.variable_list_in.extend(self.varying_boundary_variables)
        
    def _get_variable_list_nc(self):
        self.upper_air_len = self.params.num_levels*len(self.params.upper_air_variables)
        self.variable_list_in = self.upper_air_variables + self.surface_variables + self.varying_boundary_variables
        self.variable_list_out = self.upper_air_variables + self.surface_variables + self.diagnostic_variables

    def _reshape_and_mask_variables(self, data_array, out = False, from_nc = False):
        upper_air = torch.from_numpy(data_array[:self.upper_air_len].reshape(len(self.upper_air_variables),
                                                            len(self.levels),
                                                            self.params.horizontal_resolution[0],
                                                            self.params.horizontal_resolution[1])).to(torch.float32)
        surface = torch.from_numpy(data_array[self.upper_air_len:self.upper_air_len+len(self.surface_variables)]\
            .reshape(len(self.surface_variables),
                     self.params.horizontal_resolution[0],
                     self.params.horizontal_resolution[1])).to(torch.float32)
        surface = self._fill_mask(surface, self.surface_variables, self.land_variables + self.ocean_variables)
        if out:
            if len(self.diagnostic_variables) > 0:
                diagnostic = torch.from_numpy(data_array[self.upper_air_len+len(self.surface_variables):\
                                        self.upper_air_len+len(self.surface_variables)+len(self.diagnostic_variables)]\
                                .reshape(len(self.diagnostic_variables),
                                self.params.horizontal_resolution[0],
                                self.params.horizontal_resolution[1])).to(torch.float32)
                diagnostic = self._fill_mask(diagnostic, self.diagnostic_variables)
                return upper_air, surface, diagnostic
            else:
                return upper_air, surface
        else:
            if len(self.varying_boundary_variables) > 0 and not from_nc:
                varying_boundary = torch.from_numpy(data_array[self.upper_air_len+len(self.surface_variables):\
                                        self.upper_air_len+len(self.surface_variables)+len(self.varying_boundary_variables)]\
                                .reshape(len(self.varying_boundary_variables),
                                self.params.horizontal_resolution[0],
                                self.params.horizontal_resolution[1])).to(torch.float32)
                varying_boundary = self._fill_mask(varying_boundary, self.varying_boundary_variables)
                return upper_air, surface, varying_boundary
            else:
                return upper_air, surface
            
    def _load_bias(self):
        upper_air_bias = []
        surface_bias = []
        if self.params.timedelta_hours == 24:
            start_hour_z = int(self.start_date.hour)
            bias_hour_str = f'_{start_hour_z}z'
        elif self.params.timedelta_hours == 6:
            bias_hour_str = ''
        else:
            raise ValueError('Only 6 and 24 hour timesteps supported for long validation at this time.')
        for variable in self.upper_air_variables:
            if variable != 'zg' and variable != 'geopotential_height' and self.use_sigma_levels:
                for level in self.sigma_levels:
                    bias_file = os.path.join(self.params.bias_data_dir, f'{variable}_{level}_bias{bias_hour_str}.npy')
                    data = np.load(bias_file)
                    upper_air_bias.append(data)
            else:
                for level in self.levels:
                    bias_file = os.path.join(self.params.bias_data_dir, f'{variable}_{int(level)}{self.level_unit_str}_bias{bias_hour_str}.npy')
                    data = np.load(bias_file)
                    upper_air_bias.append(data)
        for variable in self.surface_variables:
            bias_file = os.path.join(self.params.bias_data_dir, f'{variable}_bias{bias_hour_str}.npy')
            data = np.load(bias_file)
            surface_bias.append(data)
        upper_air_bias = np.stack(upper_air_bias, axis = 0).reshape(len(self.upper_air_variables), -1, data.shape[0], data.shape[1])
        surface_bias = np.stack(surface_bias, axis = 0)
        if len(self.diagnostic_variables) > 0:
            diagnostic_bias = []
            for variable in self.diagnostic_variables:
                bias_file = os.path.join(self.params.bias_data_dir, f'{variable}_bias{bias_hour_str}.npy')
                data = np.load(bias_file)
                diagnostic_bias.append(data)
            diagnostic_bias = np.stack(diagnostic_bias, axis = 0)
            return torch.from_numpy(surface_bias), torch.from_numpy(upper_air_bias), torch.from_numpy(diagnostic_bias)
        else:
            return torch.from_numpy(surface_bias), torch.from_numpy(upper_air_bias)
            

    def _fill_mask(self, data, variables, optional_variables = None):
        if optional_variables:
            for i, var in enumerate(variables):
                if var in optional_variables:
                    nans = torch.isnan(data[i])
                    if torch.any(nans):
                        data[i] = data[i].masked_fill(nans, self.mask_fill[var])
        else:
            for i, var in enumerate(variables):
                nans = torch.isnan(data[i])
                if torch.any(nans):
                    data[i] = data[i].masked_fill(nans, self.mask_fill[var])
        return data

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
        constant_boundary_date = self.datetime_class(self.params.val_year_start, 1, 1, 0, has_year_zero = self.params.has_year_zero)
        constant_boundary_data = torch.from_numpy(self._get_data(constant_boundary_date, 
                                                                 variable_list = self.constant_boundary_variables)).to(torch.float32)
        constant_boundary_data = self._fill_mask(constant_boundary_data, self.constant_boundary_variables)
        land_mask = torch.clone(constant_boundary_data[np.array(self.constant_boundary_variables) == self.land_sea_mask_name].detach())
        constant_boundary_mean = torch.mean(constant_boundary_data, dim=(1,2))
        constant_boundary_std = torch.std(constant_boundary_data, dim=(1,2))
        constant_boundary_data = (constant_boundary_data - constant_boundary_mean.reshape(-1, 1, 1)) / constant_boundary_std.reshape(-1, 1, 1)
        return constant_boundary_data, land_mask

    def load_mean_std(self, mean_file, std_file, datavars, upper_air = True, use_sigma_levels = False, level_delta = 1e-4):
        if upper_air:
            if use_sigma_levels:
                coordinates = []
                levels_for_var = []
                for var in datavars:
                    if var == 'zg' or var == 'geopotential_height':
                        coordinates.append("Z")
                        levels_for_var.append(self.levels)
                    else:
                        coordinates.append("Z_2")
                        levels_for_var.append(self.sigma_levels)
            else:
                coordinates = ["Z"] * len(datavars)
                levels_for_var = [self.levels] * len(datavars)
                
            with xr.open_dataset(mean_file) as ds:
                mean = torch.stack([torch.from_numpy(ds[var].where(xr.DataArray(data=[any(np.abs(lev - levels) < level_delta) for lev in ds[coordinate].values], \
                                                                                dims = [coordinate]), drop = True).values).to(torch.float32)\
                                                                                        for var, coordinate, levels in zip(datavars, coordinates, levels_for_var)], dim=0)
            with xr.open_dataset(std_file) as ds:
                std = torch.stack([torch.from_numpy(ds[var].where(xr.DataArray(data=[any(np.abs(lev - levels) < level_delta) for lev in ds[coordinate].values], \
                                                                                dims = [coordinate]), drop = True).values).to(torch.float32)\
                                                                                        for var, coordinate, levels in zip(datavars, coordinates, levels_for_var)], dim=0)
        else:
            with xr.open_dataset(mean_file) as ds:
                mean = torch.stack([torch.from_numpy(ds[var].values).to(torch.float32) for var in datavars], dim=0)
            with xr.open_dataset(std_file) as ds:
                std = torch.stack([torch.from_numpy(ds[var].values).to(torch.float32) for var in datavars], dim=0)
        return mean, std
    
    def surface_transform(self, data):
        return (data - self.surface_mean.reshape(-1, 1, 1))/self.surface_std.reshape(-1, 1, 1)
    
    def diagnostic_transform(self, data):
        return (data - self.diagnostic_mean.reshape(-1, 1, 1))/self.diagnostic_std.reshape(-1, 1, 1)
    
    def boundary_transform(self, data):
        return (data - self.varying_boundary_mean.reshape(-1, 1, 1))/self.varying_boundary_std.reshape(-1, 1, 1)
    
    def upper_air_transform(self, data):
        return (data - self.upper_air_mean.reshape(len(self.upper_air_variables), -1, 1, 1))/ \
            self.upper_air_std.reshape(len(self.upper_air_variables), -1, 1, 1)
    
    def surface_inv_transform(self, data):
        return data * self.surface_std.reshape(1, -1, 1, 1) + self.surface_mean.reshape(1, -1, 1, 1)
    
    def upper_air_inv_transform(self, data):
        return data * self.upper_air_std.reshape(1, len(self.upper_air_variables), -1, 1, 1) + \
            self.upper_air_mean.reshape(1, len(self.upper_air_variables), -1, 1, 1)
    
    def diagnostic_inv_transform(self, data):
        return data * self.diagnostic_std.reshape(1, -1, 1, 1) + self.diagnostic_mean.reshape(1, -1, 1, 1)

    def surface_delta_transform(self, data):
        return data / self.surface_delta_std.reshape(-1, 1, 1)
    
    def upper_air_delta_transform(self, data):
        return data / self.upper_air_delta_std.reshape(len(self.upper_air_variables), -1, 1, 1)

    # Modification for the autoregressive parameter
    def _get_dates(self, hour_step=6.):
        if self.single_ic:
            start_date = self.datetime_class(self.year_start, 1, 1, has_year_zero = self.has_year_zero) + timedelta(hours=self.single_ic_offset)
            print(f'Start date: {start_date.strftime("%Y-%m-%d_%H:%M:%S")}')
            if hasattr(self.params, "prediction_duration_days"):
                print(f'Initializing data loader for {self.params.prediction_duration_days} day prediction.')
                end_date = start_date + timedelta(days=self.params.prediction_duration_days)
            else:
                end_date = self.datetime_class(self.year_start + self.long_rollout_years, 1, 1, has_year_zero = self.has_year_zero)
            print(f'End date: {end_date.strftime("%Y-%m-%d_%H:%M:%S")}')
        else:
            start_date = self.datetime_class(self.year_start, 1, 1, has_year_zero = self.has_year_zero)
            end_date = self.datetime_class(self.year_end, 1, 1, has_year_zero = self.has_year_zero)
            
        hours = (end_date - start_date).total_seconds() // 3600
        
        date_range = np.arange(0., hours, hour_step)
        print(f'Hours: {hours}')
        print(f'End data hour: {date_range[-1]}')
        return date_range, start_date, end_date
        

    def _get_data(self, data_datetime, out = False, variable_list = None):
        data_year = data_datetime.year
        data_idx = int((data_datetime - self.datetime_class(data_year, 1, 1, hour=0, has_year_zero=self.has_year_zero)).total_seconds())\
              // 3600 // self.data_timedelta_hours
        data_file_path = get_out_path(self.data_dir, data_year, data_idx)
        if variable_list:
            raw_data = get_data_given_path(data_file_path, variable_list)
        else:
            if out:
                raw_data = get_data_given_path(data_file_path, self.variable_list_out)
            else:
                raw_data = get_data_given_path(data_file_path, self.variable_list_in)
        return raw_data
    
    def _get_data_nc(self, index, out = False, variable_list_3D = None, variable_list_2D = None):
        data_file_path = self.params.init_nc_filepaths[index]
        if variable_list_3D or variable_list_2D:
            raw_data = get_data_given_path_nc(data_file_path, variable_list_3D, variable_list_2D, self.init_nc_timestep_offset[index])
        else:
            if out:
                raw_data = get_data_given_path_nc(data_file_path, self.upper_air_variables, self.surface_variables + self.diagnostic_variables, self.init_nc_timestep_offset[index])
            else:
                raw_data = get_data_given_path_nc(data_file_path, self.upper_air_variables, self.surface_variables, self.init_nc_timestep_offset[index])
        return raw_data
    
    def _get_boundary_data(self, data_datetime):
        data_year = data_datetime.year
        data_idx = int((data_datetime - self.datetime_class(data_year, 1, 1, hour=0, has_year_zero=self.has_year_zero)).total_seconds())\
              // 3600 // self.data_timedelta_hours
        #print(data_datetime.strftime("%Y-%m-%d %H:%M:%S"), data_year, data_idx)
        if cftime.is_leap_year(data_year, self.params.calendar, self.has_year_zero):
            data_file_path = get_out_path(self.data_dir, self.leap_year, data_idx)
        else:
            data_file_path = get_out_path(self.data_dir, self.no_leap_year, data_idx)
        raw_data = get_data_given_path(data_file_path, self.varying_boundary_variables)
        return torch.from_numpy(raw_data).to(torch.float32)

    def __len__(self):
        return len(self.inference_idxs)


    def __getitem__(self, index):
        #print('Loaded Boundary Data')
        #self.dates = self._get_dates(hour_step=params.timedelta_hours)
        #self.data_dss = self._load_data(initial=False)
        #self.lat = torch.from_numpy(self.data_dss[0].lat.values)
        #self.lev = torch.from_numpy(self.data_dss[0].lev.values)
        if self.single_ic:
            if index == 0:
                if self.start_date.year != self.params.val_year_start:
                    if cftime.is_leap_year(self.start_date.year, self.params.calendar, self.has_year_zero):
                        start_time = self.datetime_class(self.leap_year, self.start_date.month, self.start_date.day,
                                                         self.start_date.hour, has_year_zero = self.has_year_zero) +\
                                                             timedelta(hours=self.dates[index]) + timedelta(hours=self.nc_bc_offset)
                    else:
                        start_time = self.datetime_class(self.no_leap_year, self.start_date.month, self.start_date.day,
                                                         self.start_date.hour, has_year_zero = self.has_year_zero) +\
                                                             timedelta(hours=self.dates[index]) + timedelta(hours=self.nc_bc_offset)
                else:
                    start_time = self.start_date + timedelta(hours=self.dates[index]) + timedelta(hours=self.nc_bc_offset)
                data_in  = self._get_data(start_time, out = False)
                if len(self.varying_boundary_variables) > 0:
                    upper_air_t, surface_t, varying_boundary_data = self._reshape_and_mask_variables(data_in, out = False)
                    varying_boundary_data = self.boundary_transform(varying_boundary_data)
                else:
                    upper_air_t, surface_t = self._reshape_and_mask_variables(data_in, out = False)
                surface_t = self.surface_transform(surface_t)
                upper_air_t = self.upper_air_transform(upper_air_t)
                if self.epsilon_factor > 0.:
                    if 'surface_ff_std' in self.params:
                        surface_t_noise = torch.randn(*surface_t.shape) * (self.epsilon_factor * self.surface_ff_std / self.surface_std).reshape(len(self.surface_variables), 1, 1)
                    else:
                        surface_t_noise = torch.randn(*surface_t.shape) * self.epsilon_factor
                    surface_t = surface_t + surface_t_noise
                    if 'upper_air_ff_std' in self.params:
                        upper_air_t_noise = torch.randn(*upper_air_t.shape) * (self.epsilon_factor * self.upper_air_ff_std / self.upper_air_std).reshape(len(self.upper_air_variables), len(self.levels), 1, 1)
                    else:
                        upper_air_t_noise = torch.randn(*upper_air_t.shape) * self.epsilon_factor
                    upper_air_t = upper_air_t + upper_air_t_noise
                return surface_t, upper_air_t, varying_boundary_data, torch.tensor(start_time.year)
            else:
                start_time = self.start_date + timedelta(hours=self.dates[index]) + timedelta(hours=self.nc_bc_offset)
                varying_boundary_data = self._get_boundary_data(start_time)
                varying_boundary_data = self._fill_mask(varying_boundary_data, self.varying_boundary_variables)
                varying_boundary_data = self.boundary_transform(varying_boundary_data)
                return varying_boundary_data, torch.tensor(start_time.year)
        elif self.ensemble:
            if self.init_from_nc:
                data_in = self._get_data_nc(index, out = False)
            else:
                data_in  = self._get_data(start_time, out = False)
            if len(self.varying_boundary_variables) > 0 and not self.init_from_nc:
                upper_air_t, surface_t, varying_boundary_data = self._reshape_and_mask_variables(data_in, out = False)
                varying_boundary_data = self.boundary_transform(varying_boundary_data)
                surface_t = self.surface_transform(surface_t)
                upper_air_t = self.upper_air_transform(upper_air_t)
                return surface_t, upper_air_t, varying_boundary_data, torch.tensor(start_time.year)
            else:
                upper_air_t, surface_t = self._reshape_and_mask_variables(data_in, out = False, from_nc = True)
                surface_t = self.surface_transform(surface_t)
                upper_air_t = self.upper_air_transform(upper_air_t)
                return surface_t, upper_air_t, index
            
            
        lead_times = self.params.forecast_lead_times

        # Condition 1: Training
        if self.train:
            start_time = self.start_date + timedelta(hours=self.dates[index])
            end_time = self.start_date + timedelta(hours=self.dates[index] + self.timedelta_hours)
            data_in  = self._get_data(start_time, out = False)
            data_out = self._get_data(end_time, out = True)
            if len(self.varying_boundary_variables) > 0:
                upper_air_t, surface_t, varying_boundary_data = self._reshape_and_mask_variables(data_in, out = False)
            else:
                upper_air_t, surface_t = self._reshape_and_mask_variables(data_in, out = False)
            if len(self.diagnostic_variables) > 0:
                upper_air_t_1, surface_t_1, diagnostic_t_1 = self._reshape_and_mask_variables(data_out, out = True)
            else:
                upper_air_t_1, surface_t_1 = self._reshape_and_mask_variables(data_out, out = True)
            
            if self.params.predict_delta:
                surface_t_1 = surface_t_1 - surface_t
                upper_air_t_1 = upper_air_t_1 - upper_air_t
                surface_t = self.surface_transform(surface_t)
                surface_t_1 = self.surface_delta_transform(surface_t_1)
                upper_air_t = self.upper_air_transform(upper_air_t)
                upper_air_t_1 = self.upper_air_delta_transform(upper_air_t_1)
            else:
                surface_t = self.surface_transform(surface_t)
                surface_t_1 = self.surface_transform(surface_t_1)
                upper_air_t = self.upper_air_transform(upper_air_t)
                upper_air_t_1 = self.upper_air_transform(upper_air_t_1)
            if len(self.diagnostic_variables) > 0:
                diagnostic_t_1 = self.diagnostic_transform(diagnostic_t_1)
            varying_boundary_data = self.boundary_transform(varying_boundary_data)
            #print('Normalized Boundary')
            if self.epsilon_factor > 0.:
                if 'surface_ff_std' in self.params:
                    surface_t_noise = torch.randn(*surface_t.shape) * (self.epsilon_factor * self.surface_ff_std / self.surface_std).reshape(len(self.surface_variables), 1, 1)
                else:
                    surface_t_noise = torch.randn(*surface_t.shape) * self.epsilon_factor
                surface_t = surface_t + surface_t_noise
                if 'upper_air_ff_std' in self.params:
                    upper_air_t_noise = torch.randn(*upper_air_t.shape) * (self.epsilon_factor * self.upper_air_ff_std / self.upper_air_std).reshape(len(self.upper_air_variables), len(self.levels), 1, 1)
                else:
                    upper_air_t_noise = torch.randn(*upper_air_t.shape) * self.epsilon_factor
                upper_air_t = upper_air_t + upper_air_t_noise
        
        # Condition for autoregression
        elif lead_times:

            start_time = self.start_date + timedelta(hours=self.dates[self.inference_idxs[index]])

            # Load initial conditions
            data_in = self._get_data(start_time, out = False)
            if len(self.varying_boundary_variables) > 0:
                upper_air_t, surface_t, varying_boundary_data_t = self._reshape_and_mask_variables(data_in, out = False)
            else:
                upper_air_t, surface_t = self._reshape_and_mask_variables(data_in, out = False)

            max_lead_time = lead_times[-1]
            boundary_times = [start_time + timedelta(hours=self.timedelta_hours * lead_time) for lead_time in range(max_lead_time)]
            start_time_tensor = torch.tensor([start_time.year, start_time.month, start_time.day, start_time.hour])
            varying_boundary_data = [varying_boundary_data_t]
            varying_boundary_data.extend([self._fill_mask(\
                torch.from_numpy(self._get_data(boundary_time, variable_list = self.varying_boundary_variables)).to(torch.float32), self.varying_boundary_variables) for boundary_time in boundary_times])
            varying_boundary_data = torch.stack([self.boundary_transform(varying_boundary_data_i) for varying_boundary_data_i in varying_boundary_data], dim=0)


            if self.validate: 
                # Load targets for each time step up to the maximum lead time
                targets_surface = []
                targets_upper_air = []
                if self.params.predict_delta:
                    targets_delta_surface = []
                    targets_delta_upper_air = []
                if len(self.diagnostic_variables) > 0:
                    targets_diagnostic = []

                # Iterate over each time step up to the maximum lead time
                max_lead_time = lead_times[-1]

                for step in range(1, max_lead_time + 1):
                    target_time = start_time + timedelta(hours = self.timedelta_hours * step)
                    raw_target_data = self._get_data(target_time, out = True)

                    if len(self.diagnostic_variables) > 0:
                        upper_air_target, surface_target, diagnostic_target = self._reshape_and_mask_variables(raw_target_data, out = True)
                        targets_diagnostic.append(diagnostic_target)
                    else:
                        upper_air_target, surface_target = self._reshape_and_mask_variables(raw_target_data, out = True)
        
                    targets_surface.append(surface_target)
                    targets_upper_air.append(upper_air_target)

                    if self.params.predict_delta:
                        if step == 1:
                            surface_delta_target = targets_surface[-1] - surface_t
                            upper_air_delta_target = targets_upper_air[-1] - upper_air_t
                        else:
                            surface_delta_target = targets_surface[-1] - targets_surface[-2]
                            upper_air_delta_target = targets_upper_air[-1] - targets_upper_air[-2]
                        surface_delta_target = self.surface_delta_transform(surface_delta_target)
                        upper_air_delta_target = self.upper_air_delta_transform(upper_air_delta_target)

                        targets_delta_surface.append(surface_delta_target)
                        targets_delta_upper_air.append(upper_air_delta_target)

                for step in range(0, max_lead_time):
                    targets_surface[step] = self.surface_transform(targets_surface[step])
                    targets_upper_air[step] = self.upper_air_transform(targets_upper_air[step])
                    if len(self.diagnostic_variables) > 0:
                        targets_diagnostic[step] = self.diagnostic_transform(targets_diagnostic[step])
                
                surface_t = self.surface_transform(surface_t)
                upper_air_t = self.upper_air_transform(upper_air_t)

                targets_surface = torch.stack(targets_surface, dim=0)
                targets_upper_air = torch.stack(targets_upper_air, dim=0)
                if len(self.diagnostic_variables) > 0:
                    targets_diagnostic = torch.stack(targets_diagnostic, dim=0)
                if self.params.predict_delta:
                    targets_delta_surface = torch.stack(targets_delta_surface, dim=0)
                    targets_delta_upper_air = torch.stack(targets_delta_upper_air, dim=0)
                    

        else:
            start_time = self.start_date + timedelta(hours=self.dates[self.inference_idxs[index]])
            data_in = self._get_data(start_time, out = False)
            if len(self.varying_boundary_variables) > 0:
                surface_t, upper_air_t, varying_boundary_data = self._reshape_and_mask_variables(data_in, out=False)
                varying_boundary_data = self.boundary_transform(varying_boundary_data).unsqueeze(0)
            else:
                surface_t, upper_air_t = self._reshape_and_mask_variables(data_in, out=False)
            surface_t = self.surface_transform(surface_t)
            upper_air_t = self.upper_air_transform(upper_air_t)
        if torch.any(torch.isnan(varying_boundary_data)):
            print('Boundary data has nan')
            sys.exit(2)
        if torch.any(torch.isnan(surface_t)):
            print('Surface t has nan')
            sys.exit(2)
        if torch.any(torch.isnan(upper_air_t)):
            print('Upper air t has nan')
            sys.exit(2)

        if self.train:
            if torch.any(torch.isnan(surface_t_1)):
                print('Surface t+1 has nan')
                sys.exit(2)
            if torch.any(torch.isnan(upper_air_t_1)):
                print('Upper air t+1 has nan')
                sys.exit(2)
            if len(self.diagnostic_variables) > 0:
                if torch.any(torch.isnan(diagnostic_t_1)):
                    print('Diagnostic has nan')
                    sys.exit(2)
            if len(self.diagnostic_variables) > 0:
                return surface_t, upper_air_t, surface_t_1, upper_air_t_1, diagnostic_t_1, varying_boundary_data
            else:
                return surface_t, upper_air_t, surface_t_1, upper_air_t_1, varying_boundary_data
        ### ERROR - Need to have data loader return times for validation
        elif self.validate and lead_times:
            if self.params.predict_delta:
                if len(self.diagnostic_variables) > 0:
                    return surface_t, upper_air_t, targets_surface, targets_upper_air, targets_diagnostic, targets_delta_surface, targets_delta_upper_air, \
                        varying_boundary_data, start_time_tensor
                else:
                    return surface_t, upper_air_t, targets_surface, targets_upper_air, varying_boundary_data, targets_delta_surface, targets_delta_upper_air, start_time_tensor
            else:
                if len(self.diagnostic_variables) > 0:
                    return surface_t, upper_air_t, targets_surface, targets_upper_air, targets_diagnostic, \
                        varying_boundary_data, start_time_tensor
                else:
                    return surface_t, upper_air_t, targets_surface, targets_upper_air, varying_boundary_data, start_time_tensor
        elif lead_times:
            return surface_t, upper_air_t, varying_boundary_data
        else:
            if len(self.diagnostic_variables) > 0:
                return surface_t, upper_air_t, surface_t_1, upper_air_t_1, diagnostic_t_1, varying_boundary_data
            else:
                return surface_t, upper_air_t, surface_t_1, upper_air_t_1, varying_boundary_data
