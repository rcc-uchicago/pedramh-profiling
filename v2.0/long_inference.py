from networks.pangu import PanguModel_Plasim
from networks.modulus_sfno.sfnonet import SphericalFourierNeuralOperatorNet_v2 as SFNO
from tqdm import tqdm
from ruamel.yaml.comments import CommentedMap as ruamelDict
from ruamel.yaml import YAML
from collections import OrderedDict
import matplotlib.pyplot as plt
from utils.data_loader_multifiles import get_data_loader
from utils.YParams import YParams
import os
import time
import numpy as np
import argparse
import torch
import xarray as xr
import torchvision
from torchvision.utils import save_image
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils
from utils.power_spectrum import *
from utils.perturbation import Perturber
##########################################
## NEW IMPORTS
from utils.losses import Latitude_weighted_MSELoss, Latitude_weighted_L1Loss, Masked_L1Loss,\
    Masked_MSELoss, Latitude_weighted_masked_L1Loss, Latitude_weighted_masked_MSELoss,\
    Latitude_weighted_CRPSLoss
###############################@###########
logging_utils.config_logger()
#from apex import optimizers
from pathlib import Path
import dask
from datetime import timedelta
# import transformer_engine.pytorch as te
# from transformer_engine.common import recipe
# from transformer_engine.pytorch import fp8_autocast
from torch.profiler import profile, record_function, ProfilerActivity
from itertools import product
import time 
import multiprocessing as mp
import psutil
import shutil
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid
from utils.integrate import Integrator, forward_euler
import cftime
from copy import deepcopy
import json
from natsort import natsorted
import glob


#mp.set_start_method("spawn", force=True)


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
        
import atexit
atexit.register(cleanup)

def to_ensemble_batch(data, ens_members):
    """Convert batch of M samples (M, ...) to a batch of (M*ens_members, ...)."""
    mult_shape = [1] * (len(data.shape) + 1)
    mult_shape[1] = ens_members
    return (data.unsqueeze(1)*torch.ones(mult_shape, device = data.device)).flatten(0, 1)

def compute_A_ensemble(args):
    """
    Computes the A values for an ensemble forecast.
    """
    # Open the dataset for existing paths
    ds, particle_idx, ensemble_start, ensemble_end, save_basenames, target_duration, lead_time, var, regions, PATH_REGIONS = args
    # ds = xr.open_dataset(path, decode_times=True, use_cftime=True)
    # Load region boundaries from a JSON file
    print(f'Computing obs for particle {particle_idx}, members {ensemble_start} to {ensemble_end}')
    with open(PATH_REGIONS, 'r') as f:
        all_regions = json.load(f)
    for region in regions:
        lon_region = all_regions[region]['lon']
        lat_region = all_regions[region]['lat']
        # Select the region of interest from the dataset
        ds_region = ds.sel(lon=lon_region, lat=lat_region, method='nearest')
        # Compute the distribution of A values for the selected region and variable
        if var in ['tas', 'ta']:
            A = ds_region[var].mean(dim=['lon', 'lat']) - 273.15  # Convert temperature from Kelvin to Celsius
        else:
            A = ds_region[var].mean(dim=['lon', 'lat'])
        A = A.resample(time='1D').mean()  # Resample to daily mean
        A = A.isel(time=slice(lead_time, lead_time+target_duration))  # Select the first T days
        A = A.mean(dim='time')  # Compute the mean over the selected time period
        filepath = save_basenames[particle_idx] + f'_{ensemble_start:04d}-{ensemble_end:04d}_A_{region}.npy'
        np.save(filepath, A.values.flatten())
        
def combine_A_ensemble(save_basenames, regions):
    for save_basename, region in tqdm(product(save_basename, region), 
                                      total = len(save_basenames)*len(regions),
                                      desc = "Combining obs files..."):
        files = natsorted(glob.glob(save_basename + f'_**-**_A_{region}.npy'))
        data = []
        for file in files:
            data.append(np.load(data))
            #os.remove(file)
        data = np.concat(data, axis = 0)
        filepath_out = save_basename + f'_A_{region}.npy'
        np.save(filepath_out)


class Stepper():
    def count_parameters(self):
        if self.use_6h_24h_model:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad) + \
                sum(p.numel() for p in self.model_24h.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def __init__(self, params_list, world_rank, use_6h_24h_model=False,
                 async_save = False, obs_function = None, obs_args = None):
        self.params = params_list[0]
        self.use_6h_24h_model = use_6h_24h_model
        if use_6h_24h_model:
            self.params_24h = params_list[1]
        self.world_rank = world_rank
        self.async_save = async_save
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        self.early_stop_epoch = self.params['early_stop_epoch'] - 1 if 'early_stop_epoch' in self.params else None
        self.run_uuid = str(uuid.uuid4())
        self.has_land = False
        self.has_ocean = False
        self.mask_output = False
        if hasattr(self.params, 'land_variables'):
            if len(self.params.land_variables) > 0:
                self.has_land = True
        else:
            self.params['land_variables'] = []
        if hasattr(self.params, 'ocean_variables'):
            if len(self.params.ocean_variables) > 0:
                self.has_land = True
        else:
            self.params['ocean_variables'] = []
        if hasattr(self.params, 'mask_output'):
            self.mask_output = self.params.mask_output
        if use_6h_24h_model:
            has_land = False
            has_ocean = False
            mask_output = False
            if hasattr(self.params_24h, 'land_variables'):
                if len(self.params_24h.land_variables) > 0:
                    has_land = True
            else:
                self.params_24h['land_variables'] = []
            if hasattr(self.params_24h, 'ocean_variables'):
                if len(self.params_24h.ocean_variables) > 0:
                    has_land = True
            else:
                self.params_24h['ocean_variables'] = []
            if hasattr(self.params_24h, 'mask_output'):
                mask_output = self.params_24h.mask_output
            assert has_land == self.has_land
            assert has_ocean == self.has_ocean
            assert mask_output == self.mask_output
        self.obs_function = obs_function
        self.obs_args = obs_args
        self.save_forecasts = False
        if hasattr(self.params, 'save_forecasts'):
            self.save_forecasts = self.params.save_forecasts
        print(f'Save forecasts: {self.save_forecasts}')
            
        if hasattr(self.params, 'save_level_idxs'):
            self.save_level_idxs = np.array(self.params.save_level_idxs)
        else:
            self.save_level_idxs = np.arange(self.params.num_levels)
        
        if self.params.use_sigma_levels:
            if hasattr(self.params, 'save_sigma_level_idxs'):
                self.save_sigma_level_idxs = self.params.save_sigma_level_idxs
            else:
                self.save_sigma_level_idxs = np.arange(self.params.num_levels)


        logging.info('rank %d, begin data loader init' % world_rank)
        for params in params_list:
            print(params)
            
        self.params.long_rollout_years = self.params.final_datetime.year - self.params.init_datetime.year
                                                                                
        self.data_loader, self.dataset = get_data_loader(self.params, self.params.data_dir, dist.is_initialized(), 
                                                         year_start=self.params.val_year_start, 
                                                         year_end=self.params.val_year_end, train=False,
                                                         ensemble = True, init_from_nc = True,
                                                         load_all_bcs = False)
        print(f'Len(data_loader): {len(self.data_loader)}')
        self.params['single_ic_offset'] = int((self.params.init_datetime - \
            self.dataset.datetime_class(self.params.init_datetime.year, 1, 1, 0, has_year_zero = self.params.has_year_zero)).total_seconds() // 3600)
        self.data_loader_bcs, self.dataset_bcs = get_data_loader(self.params, self.params.data_dir, dist.is_initialized(), 
                                                         year_start=self.params.init_datetime.year, 
                                                         year_end=self.params.final_datetime.year, train=False,
                                                         single_ic = True, load_all_bcs = False)
        
        if self.params.epsilon_factor > 0.:
            self.perturber = Perturber(self.params, self.dataset, device = self.device,
                                    device_idx = self.world_rank, seed = self.params.run_iter*self.params.world_size)
        
        
        #print('Inference Idxs:')
        #print(self.valid_dataset.inference_idxs)

        # self.constant_boundary_data = self.train_dataset.constant_boundary_data.unsqueeze(0) * torch.ones(params.batch_size, 1, 1, 1)
        # self.constant_boundary_data = self.constant_boundary_data.to(self.device, non_blocking=True)
        self.constant_boundary_data = self.dataset.constant_boundary_data.unsqueeze(0) * torch.ones(self.params.batch_size, 1, 1, 1)
        self.constant_boundary_data = self.constant_boundary_data.to(self.device, non_blocking=True)
        if self.params.num_ensemble_members > 1:
            logging.info(f'Ensemble Mode. Ensemble size = {self.params.num_ensemble_members}\n')

         # Load climatology
        """
        climatology_path = os.path.join(params.data_dir, self.params.climatology_file)
        self.climatology = xr.open_dataset(climatology_path)
        if 'time_bnds' in self.climatology.data_vars:
            self.climatology = self.climatology.drop_vars('time_bnds')
        self.climatology = self.climatology.astype({var: np.float32 for var in self.climatology.data_vars})
        self.climatology = self.climatology.rename({'time':'dayofyear'})
        if self.long_validation:
            self.climatology_bias = self.climatology.mean(dim='dayofyear')
            self.clim_surface_bias = torch.from_numpy(np.stack([self.climatology_bias[var].values for var in self.params.surface_variables]))
            upper_air_clim = []
            for var in self.params.upper_air_variables:
                if var != 'zg' and var != 'geopotential' and self.params.use_sigma_levels:
                    upper_air_clim.append(self.climatology_bias[var].sel(lev = self.params.sigma_levels))
                else:
                    upper_air_clim.append(self.climatology_bias[var].sel(plev = self.params.levels))
            self.clim_upper_air_bias = torch.from_numpy(np.stack(upper_air_clim))
            if self.params.has_diagnostic:
                self.clim_diagnostic_bias = torch.from_numpy(np.stack([self.climatology_bias[var].values for var in self.params.diagnostic_variables]))
        """


        self.enable_amp = self.params.enable_amp
        self.enable_fp8 = self.params.enable_fp8
        
        if self.enable_fp8:
            global te, recipe, fp8_autocast
            import transformer_engine.pytorch as te
            from transformer_engine.common import recipe
            from transformer_engine.pytorch import fp8_autocast

            self.fp8_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID,
                                                    amax_history_len=16,
                                                    amax_compute_algo="max")
            
        logging.info('rank %d, data loader initialized' % world_rank)
        
        if self.params.nettype == 'pangu_plasim':
            if (self.has_land or self.has_ocean) and self.mask_output:
                land_mask = torch.clone(self.dataset.land_mask.detach()).to(self.device)
                print(f'Land Mask shape: {land_mask.shape}')
                mask_bool = []
                for var in self.params.surface_variables:
                    if var in self.params.land_variables:
                        mask_bool.append(torch.clone(land_mask).to(torch.bool))
                    elif var in self.params.ocean_variables:
                        mask_bool.append(torch.logical_not(torch.clone(land_mask).to(torch.bool)))
                    else:
                        mask_bool.append(torch.ones(land_mask.shape, device=self.device, dtype=torch.bool))
                mask_bool = torch.stack(mask_bool)
            else:
                land_mask = None
            if self.params.predict_delta:
                self.model = PanguModel_Plasim(self.params, land_mask = land_mask).to(self.device)
                self.integrator = Integrator(self.params, surface_ff_std=self.dataset.surface_std.detach().to(self.device),
                                            surface_delta_std=self.dataset.surface_delta_std.detach().to(self.device),
                                            upper_air_ff_std=self.dataset.upper_air_std.detach().to(self.device),
                                            upper_air_delta_std=self.dataset.upper_air_delta_std.detach().to(self.device)).to(self.device)
            else:
                if hasattr(params, 'mask_fill'):
                    self.model = PanguModel_Plasim(self.params, land_mask = land_mask, 
                                            mask_fill = self.params.mask_fill).to(self.device)
                else:
                    self.model = PanguModel_Plasim(self.params, land_mask = land_mask, 
                                                mask_fill = self.dataset.mask_fill).to(self.device)
            if self.use_6h_24h_model:
                _, self.dataset_24h = get_data_loader(self.params_24h, self.params_24h.data_dir, dist.is_initialized(), 
                                                 year_start=self.params_24h.val_year_start, 
                                                 year_end=self.params_24h.val_year_end, train=False,
                                                 ensemble = False, init_from_nc = True,
                                                 load_all_bcs = False)
                if self.params_24h.predict_delta:
                    self.model_24h = PanguModel_Plasim(self.params_24h, land_mask = land_mask).to(self.device)
                    self.integrator_24h = Integrator(self.params_24h, surface_ff_std=self.dataset_24h.surface_std.detach().to(self.device),
                                                surface_delta_std=self.dataset_24h.surface_delta_std.detach().to(self.device),
                                                upper_air_ff_std=self.dataset_24h.upper_air_std.detach().to(self.device),
                                                upper_air_delta_std=self.dataset_24h.upper_air_delta_std.detach().to(self.device)).to(self.device)
                else:
                    if hasattr(params, 'mask_fill'):
                        self.model_24h = PanguModel_Plasim(self.params_24h, land_mask = land_mask, 
                                                mask_fill = self.params_24h.mask_fill).to(self.device)
                    else:
                        self.model_24h = PanguModel_Plasim(self.params_24h, land_mask = land_mask, 
                                                    mask_fill = self.dataset_24h.mask_fill).to(self.device)
                
            #self.restore_checkpoint(params.best_checkpoint_path)
            #self.model = torch.compile(self.model, mode="max-autotune")
            #self.model = torch.compile(self.model, mode = 'default')
        elif params.nettype == 'sfno_plasim':
            print(f'\n\nRunning SFNO model\n\n')
            self.model = SFNO(params, self.dataset).to(self.device)
            if params.sync_norm:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            if self.params.predict_delta:
                self.integrator = Integrator(params, surface_ff_std=self.train_datasets[0].surface_std.detach().to(self.device),
                                               surface_delta_std=self.train_datasets[0].surface_delta_std.detach().to(self.device),
                                               upper_air_ff_std=self.train_datasets[0].upper_air_std.detach().to(self.device),
                                               upper_air_delta_std=self.train_datasets[0].upper_air_delta_std.detach().to(self.device)).to(self.device)
        else:
            raise Exception("not implemented")

        if dist.is_initialized():
            self.model = DistributedDataParallel(self.model,
                                                 device_ids=[
                                                     params.local_rank],
                                                 output_device=[params.local_rank], find_unused_parameters=True)
            if self.use_6h_24h_model:
                self.model_24h = DistributedDataParallel(self.model_24h,
                                                 device_ids=[
                                                     params.local_rank],
                                                 output_device=[params.local_rank], find_unused_parameters=True)
        self.restore_checkpoint(self.model, self.params.best_checkpoint_path)
        if self.use_6h_24h_model:
            self.restore_checkpoint(self.model_24h, self.params_24h.best_checkpoint_path)
            if 'pr_6h' in self.params.diagnostic_variables:
                assert 'pr_24h' in self.params_24h.diagnostic_variables
                self.pr_6h_idx = self.params.diagnostic_variables.index('pr_6h')
                assert self.pr_6h_idx == self.params_24h.diagnostic_variables.index('pr_24h')
        
        if params.log_to_screen:
            logging.info("Number of trainable model parameters: {}".format(self.count_parameters()))
            
        
    def _get_inference_duration(self):
        steps_per_year = [
            (self.dataset.datetime_class(year + 1, 1, 1, hour=0, has_year_zero = self.params.has_year_zero) - \
             self.dataset.datetime_class(year, 1, 1, hour=0, has_year_zero = self.params.has_year_zero)).total_seconds() \
             // 3600 / self.params.timedelta_hours for year in range(self.params.init_datetime.year, self.params.final_datetime.year)
        ]
        return np.array(steps_per_year)
            
    def restore_checkpoint(self, model, checkpoint_path):
        """ We intentionally require a checkpoint_dir to be passed
            in order to allow Ray Tune to use this function """
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.params.local_rank), weights_only=False)
        try:
            model.load_state_dict(checkpoint['model_state'])
        except:
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                name = key[7:]
                new_state_dict[name] = val
            model.load_state_dict(new_state_dict)
        #self.model = torch.compile(self.model)
        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch']
        print('START EPOCH:', self.startEpoch)
        # restore checkpoint is used for finetuning as well as resuming. If finetuning (i.e., not resuming), restore checkpoint does not load optimizer state, instead uses config specified lr.
        #if self.params.resuming:
        #    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
    def predict(self, obs_function = None, obs_args = None):
        if self.params.log_to_screen:
            logging.info("Starting Model Inference Loop...")

        start = time.time()
        #tr_time, data_time, train_logs = self.train_one_epoch()
        self.obs_function = obs_function
        self.obs_args = obs_args
        #if self.params.batch_size > 1 and self.obs_function:
        #    raise ValueError('Program assumes obs_function only works for a batch size of 1.')
        if self.async_save and (self.save_forecasts or type(self.obs_function) is not type(None)):
            valid_time, valid_logs = asyncio.run(self.predict_async())
        else:
            valid_time, valid_logs = self.predict_sync()
            
    async def save_prediction_async(self, datasets, prediction_idxs, ensemble_start, ensemble_end):
        await asyncio.to_thread(self.save_prediction, datasets, prediction_idxs, ensemble_start, ensemble_end)
        
    async def run_obs_function_async(self, dataset, particle_idx, ensemble_start, ensemble_end, obs_args):
        await asyncio.to_thread(self.obs_function, [dataset, particle_idx, ensemble_start, ensemble_end] + obs_args)
        
    async def save_results(self, queue):
        while True:
            item = await queue.get()
            if item is None:
                break
            datasets, prediction_idxs, ensemble_start, ensemble_end = item
            save_start = time.time()
            await self.save_prediction_async(datasets, prediction_idxs, ensemble_start, ensemble_end)
            self.save_time += time.time() - save_start
            queue.task_done()
            
    async def obs_results(self, queue):
        while True:
            item = await queue.get()
            if item is None:
                break
            dataset, particle_idx, ensemble_start, ensemble_end, obs_args = item
            obs_start = time.time()
            await self.run_obs_function_async(dataset, particle_idx, ensemble_start, ensemble_end, obs_args)
            self.obs_time += time.time() - obs_start
            queue.task_done()
            
    async def predict_async(self):
        self.model.eval()
        total_start = time.time()
        data_time = 0
        inference_time = 0
        transform_time = 0
        conversion_time = 0
        self.obs_time = 0
        self.save_time = 0

        if self.params.save_forecasts:
            save_queue = asyncio.Queue()
            save_task = asyncio.create_task(self.save_results(save_queue))
        
        if type(self.obs_function) is not type(None):
            obs_queue = asyncio.Queue()
            obs_task = asyncio.create_task(self.obs_results(obs_queue))

        with torch.inference_mode(), amp.autocast(enabled=self.params.enable_amp):
            for i, data in enumerate(self.data_loader, 0):
                # Load ith set of initial conditions from .nc files
                data_start = time.time()
                input_surface_in, input_upper_air_in = map(
                    lambda x: x.to(self.device, dtype=torch.float32), data[:-1])
                particle_idxs = data[-1]
                print(f'Particle idxs:{particle_idxs}, world rank {self.world_rank}')
                data_time += time.time() - data_start
                
                # Get beginning and end indices for ensemble sets
                ensemble_member_splits = np.arange(0, self.params.num_ensemble_members+self.params.ensemble_members_per_pred,
                                                   self.params.ensemble_members_per_pred)
                
                # For each ensemble set
                for ensemble_start, ensemble_end in zip(ensemble_member_splits[:-1], ensemble_member_splits[1:]):
                    # Create ensemble batch from ICs
                    ensemble_end = min(ensemble_end, self.params.num_ensemble_members)
                    input_surface = to_ensemble_batch(input_surface_in, ensemble_end - ensemble_start)
                    input_upper_air = to_ensemble_batch(input_upper_air_in, ensemble_end - ensemble_start)
                    constant_boundary_data = to_ensemble_batch(self.constant_boundary_data, ensemble_end - ensemble_start)
                    
                    #If using 6h and 24h models, also initialize initial condition for 24h model
                    if self.use_6h_24h_model:
                        input_surface_24h = torch.clone(input_surface)
                        input_upper_air_24h = torch.clone(input_upper_air)
                    #print('Input:', input_surface[0,1,16,1])
                    
                    # Create arrays for first year of output data
                    current_datetime = self.params.init_datetime
                    current_year = self.params.init_datetime.year
                    next_year_offset_hours = current_datetime.hour % self.params.timedelta_hours
                    time_step_in_year = 0
                    next_output_datetime = self.dataset.datetime_class(current_year+1, 1, 1, hour=next_year_offset_hours,
                                                                    has_year_zero = self.params.has_year_zero)
                    
                    # Perturb initial conditions if using perturbations
                    if self.params.epsilon_factor > 0. and self.params.init_datetime.year == 1:
                        print('Perturbing ICs...')
                        input_surface, input_upper_air = self.perturber(input_surface, input_upper_air)
                        
                    output_inference_steps = int((next_output_datetime - current_datetime).total_seconds() // 3600 / self.params.timedelta_hours)
                    output_surface = np.zeros((input_surface.shape[0], output_inference_steps,
                                                    input_surface.shape[1], input_surface.shape[2], input_surface.shape[3]),
                                                    dtype = np.float32)
                    output_upper_air = np.zeros((input_upper_air.shape[0], output_inference_steps,
                                                    input_upper_air.shape[1], input_upper_air.shape[2],
                                                    input_upper_air.shape[3], input_upper_air.shape[4]),
                                                    dtype = np.float32)
                    if self.params.has_diagnostic:
                        output_diagnostic = np.zeros((input_surface.shape[0], output_inference_steps,
                                                        len(self.params.diagnostic_variables), input_surface.shape[2], input_surface.shape[3]),
                                                        dtype = np.float32)
                    
                    # Set initial values in arrays from loaded and perturbed ICs
                    transform_start = time.time()
                    output_surface[:,time_step_in_year] = self.dataset.surface_inv_transform(input_surface.to('cpu')).numpy()
                    output_upper_air[:,time_step_in_year] = self.dataset.upper_air_inv_transform(input_upper_air.to('cpu')).numpy()
                    transform_time += time.time() - transform_start
                    
                    # For every time step in duration of ensemble emulation
                    pbar = tqdm(enumerate(self.data_loader_bcs, 0), total=len(self.data_loader_bcs), miniters=1,
                                desc = f'Emulating year {self.params.init_datetime.year}')
                    for i, bc_data in pbar:
                        # Load varying boundary conditions
                        data_start = time.time()
                        if i == 0:
                            _, _, varying_boundary_data, year = map(lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), bc_data)
                        else:
                            varying_boundary_data, year = map(lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), bc_data)
                        pbar.set_description(f'Emulating year {year}')
                        # Create ensemble of boundary conditions
                        varying_boundary_data = to_ensemble_batch(varying_boundary_data, ensemble_end - ensemble_start)
                        data_time += time.time() - data_start
                        
                        # If using 6h and 24h model and time is 0z
                        if self.use_6h_24h_model and time_step_in_year % 4 == 0:
                            # Predict to next 0z with 24h model
                            inference_start = time.time()
                            if self.params_24h.has_diagnostic:
                                out_surface_24h, out_upper_air_24h, out_diagnostic_24h = self.model_24h(input_surface_24h, 
                                                                                        constant_boundary_data, 
                                                                                        varying_boundary_data,
                                                                                        input_upper_air_24h)
                            else:
                                out_surface_24h, out_upper_air_24h = self.model_24h(input_surface_24h, constant_boundary_data, 
                                                                        varying_boundary_data, input_upper_air_24h)
                            #print(f'Output 24 at step {i}:', out_surface_24h[0,1,16,1])
                            # Predict to next 6z with 6h model
                            if self.params.has_diagnostic:
                                out_surface, out_upper_air, out_diagnostic = self.model(input_surface_24h, 
                                                                                        constant_boundary_data, 
                                                                                        varying_boundary_data,
                                                                                        input_upper_air_24h)
                            else:
                                out_surface, out_upper_air = self.model(input_surface_24h, constant_boundary_data, 
                                                                        varying_boundary_data, input_upper_air_24h)
                            #print(f'Output 6 at step {i}:', out_surface[0,1,16,1])
                            # If needed, integrate tendency predictions
                            if self.params.predict_delta:
                                input_surface, input_upper_air = self.integrator(input_surface_24h, input_upper_air_24h, out_surface, out_upper_air)
                            else:
                                input_surface, input_upper_air = out_surface, out_upper_air
                            if self.params_24h.predict_delta:
                                input_surface_24h, input_upper_air_24h = self.integrator(input_surface_24h, input_upper_air_24h, out_surface_24h, out_upper_air_24h)
                            else:
                                input_surface_24h, input_upper_air_24h = out_surface_24h, out_upper_air_24h
                            inference_time += time.time() - inference_start
                            
                            # If we have not reached the end of a year, save the non-diagnostic output from the 24h model at next 0z
                            if time_step_in_year + 4 < output_surface.shape[1]:
                                transform_start = time.time()
                                output_surface[:,time_step_in_year + 4] = self.dataset_24h.surface_inv_transform(input_surface_24h.to('cpu')).numpy()
                                output_upper_air[:,time_step_in_year + 4] = self.dataset_24h.upper_air_inv_transform(input_upper_air_24h.to('cpu')).numpy()
                                transform_time += time.time() - transform_start
                            
                            # If we have not reached the end of a year, save the output from the 6h model at next 6z
                            if time_step_in_year + 1 < output_surface.shape[1]:
                                transform_start = time.time()
                                if self.params.has_diagnostic:
                                    output_diagnostic[:, time_step_in_year + 1] = self.dataset.diagnostic_inv_transform(out_diagnostic.to('cpu')).numpy()
                                output_surface[:,time_step_in_year + 1] = self.dataset.surface_inv_transform(input_surface.to('cpu')).numpy()
                                output_upper_air[:,time_step_in_year + 1] = self.dataset.upper_air_inv_transform(input_upper_air.to('cpu')).numpy()
                                transform_time += time.time() - transform_start
                        # If not using 6h and 24h model, or time is 6z or 12z
                        elif (not self.use_6h_24h_model) or (time_step_in_year+1) % 4 != 0:
                            # Predict with base model
                            inference_start = time.time()
                            if self.params.has_diagnostic:
                                out_surface, out_upper_air, out_diagnostic = self.model(input_surface, 
                                                                                        constant_boundary_data, 
                                                                                        varying_boundary_data,
                                                                                        input_upper_air)
                            else:
                                out_surface, out_upper_air = self.model(input_surface, constant_boundary_data, 
                                                                        varying_boundary_data, input_upper_air)
                            #print(f'Output {self.params.timdelta_hours} at step {i}:', out_surface[0,1,16,1])
                            # If needed, integrate tendency predictions
                            if self.params.predict_delta:
                                input_surface, input_upper_air = self.integrator(input_surface, input_upper_air, out_surface, out_upper_air)
                            else:
                                input_surface, input_upper_air = out_surface, out_upper_air
                            inference_time += time.time() - inference_start
                            # If we have not reached the end of a year, save the output from the model
                            if time_step_in_year + 1 < output_surface.shape[1]:
                                transform_start = time.time()
                                if self.params.has_diagnostic:
                                    output_diagnostic[:, time_step_in_year + 1] = self.dataset.diagnostic_inv_transform(out_diagnostic.to('cpu')).numpy()
                                output_surface[:,time_step_in_year + 1] = self.dataset.surface_inv_transform(input_surface.to('cpu')).numpy()
                                output_upper_air[:,time_step_in_year + 1] = self.dataset.upper_air_inv_transform(input_upper_air.to('cpu')).numpy()
                                transform_time += time.time() - transform_start
                                
                        # If we are using the 6h and 24h model and we are at 18z
                        elif self.use_6h_24h_model and (time_step_in_year+1) % 4 == 0 and self.params_24h.has_diagnostic:
                            # Get the 24h model diagnostic output and subtract the accumulated precip from the 3 6h model predictions
                            transform_start = time.time()
                            output_diagnostic_24h = self.dataset_24h.diagnostic_inv_transform(out_diagnostic_24h.to('cpu')).numpy()
                            if 'pr_6h' in self.params.diagnostic_variables:
                                output_diagnostic_24h[:, self.pr_6h_idx] -= np.sum(output_diagnostic[:, time_step_in_year+1:time_step_in_year+4, self.pr_6h_idx], axis = 1)
                            if time_step_in_year + 1 < output_surface.shape[1]:
                                output_diagnostic[:, time_step_in_year + 1] = output_diagnostic_24h
                            transform_time += time.time() - transform_start
                        
                        # Go to next yearly time step
                        time_step_in_year += 1
                        
                        # If we have reached the end of a year of predictions
                        if time_step_in_year == output_surface.shape[1]:
                            # Reset the yearly timestep counter
                            time_step_in_year = 0
                            # Convert numpy data to xarray
                            conversion_start = time.time()
                            if self.params.has_diagnostic:
                                ensemble_datasets = self.convert_ensemble_to_xarray(\
                                    output_surface.reshape(input_surface_in.shape[0], ensemble_end - ensemble_start, *output_surface.shape[1:]).copy(),
                                    output_upper_air.reshape(input_upper_air_in.shape[0], ensemble_end - ensemble_start, *output_upper_air.shape[1:]).copy(),
                                    particle_idxs, current_datetime, next_output_datetime, np.arange(ensemble_start, ensemble_end),
                                    diagnostic_prediction = output_diagnostic.reshape(input_surface_in.shape[0], ensemble_end - ensemble_start, *output_diagnostic.shape[1:]).copy())
                            else:
                                ensemble_datasets = self.convert_ensemble_to_xarray(\
                                    output_surface.reshape(input_surface_in.shape[0], ensemble_end - ensemble_start, *output_surface.shape[1:]).copy(),
                                    output_upper_air.reshape(input_upper_air_in.shape[0], ensemble_end - ensemble_start, *output_upper_air.shape[1:]).copy(),
                                    particle_idxs, current_datetime, next_output_datetime, np.arange(ensemble_start, ensemble_end))
                            conversion_time += time.time() - conversion_start
                            
                            # If computing an observable, send this computatation to a separate thread
                            if type(self.obs_function) is not type(None):
                                await obs_queue.put((deepcopy(ensemble_datasets)[0], particle_idxs.numpy()[0], ensemble_start, ensemble_end, deepcopy(self.obs_args)))
                                await asyncio.sleep(0)
                                
                            # If saving the forecasts, send this computation to another thread
                            if self.params.save_forecasts:
                                # Queue the results for asynchronous saving
                                await save_queue.put((deepcopy(ensemble_datasets), particle_idxs.numpy(), ensemble_start, ensemble_end))
                                await asyncio.sleep(0)
                                        
                            
                            current_year += 1
                            # If this was not the final year
                            if current_year < self.params.final_datetime.year:
                                # Get the number of time steps in the next year
                                current_datetime = next_output_datetime
                                next_year_offset_hours = current_datetime.hour % self.params.timedelta_hours
                                next_output_datetime = self.dataset.datetime_class(current_year+1, 1, 1, hour=next_year_offset_hours,
                                                                                has_year_zero = self.params.has_year_zero)
                                output_inference_steps = int((next_output_datetime - current_datetime).total_seconds() // 3600 / self.params.timedelta_hours)
                                
                                # Create new numpy arrays for the next year of data
                                output_surface = np.zeros((input_surface.shape[0], output_inference_steps,
                                                                input_surface.shape[1], input_surface.shape[2], input_surface.shape[3]),
                                                                dtype = np.float32)
                                output_upper_air = np.zeros((input_upper_air.shape[0], output_inference_steps,
                                                                input_upper_air.shape[1], input_upper_air.shape[2],
                                                                input_upper_air.shape[3], input_upper_air.shape[4]),
                                                                dtype = np.float32)
                                if self.params.has_diagnostic:
                                    output_diagnostic = np.zeros((input_surface.shape[0], output_inference_steps,
                                                                    len(self.params.diagnostic_variables), input_surface.shape[2], input_surface.shape[3]),
                                                                    dtype = np.float32)
                                
                                # Set first time step of new array to last prediction from model
                                transform_start = time.time()
                                if self.use_6h_24h_model and self.params.init_datetime.hour == 0:
                                    if self.params.has_diagnostic:
                                        output_diagnostic[:, time_step_in_year] = output_diagnostic_24h
                                    output_surface[:,time_step_in_year] = self.dataset_24h.surface_inv_transform(input_surface_24h.to('cpu')).numpy()
                                    output_upper_air[:,time_step_in_year] = self.dataset_24h.upper_air_inv_transform(input_upper_air_24h.to('cpu')).numpy()
                                else:
                                    if self.params.has_diagnostic:
                                        output_diagnostic[:, time_step_in_year] = self.dataset.diagnostic_inv_transform(out_diagnostic.to('cpu')).numpy()
                                    output_surface[:,time_step_in_year] = self.dataset.surface_inv_transform(input_surface.to('cpu')).numpy()
                                    output_upper_air[:,time_step_in_year] = self.dataset.upper_air_inv_transform(input_upper_air.to('cpu')).numpy()
                                    if self.use_6h_24h_model:
                                        input_surface_24h, input_upper_air_24h = input_surface, input_upper_air
                                transform_time += time.time() - transform_start
                        

        if type(self.obs_function) is not type(None):
            # Signal that we're done
            await obs_queue.put(None)
            # Wait for all saves to complete
            await obs_task
        if self.params.save_forecasts:
            # Signal that we're done
            await save_queue.put(None)
            # Wait for all saves to complete
            await save_task

        total_time = time.time() - total_start

        logs = {
            'total_time': total_time,
            'data_time': data_time,
            'inference_time': inference_time,
            'transform_time': transform_time,
            'conversion_time': conversion_time,
            'obs_time': self.obs_time,
            'save_time': self.save_time
        }

        logging.info(f"Validation logs: {logs}")

        return total_time, logs
    
    def predict_sync(self):
        self.model.eval()
        total_start = time.time()
        data_time = 0
        inference_time = 0
        transform_time = 0
        conversion_time = 0
        obs_time = 0
        save_time = 0

        with torch.inference_mode(), amp.autocast(enabled=self.params.enable_amp):
            for i, data in enumerate(self.data_loader, 0):
                # Load ith set of initial conditions from .nc files
                data_start = time.time()
                input_surface_in, input_upper_air_in = map(
                    lambda x: x.to(self.device, dtype=torch.float32), data[:-1])
                particle_idxs = data[-1]
                print(f'Particle idxs:{particle_idxs}, world rank {self.world_rank}')
                data_time += time.time() - data_start
                
                # Get beginning and end indices for ensemble sets
                ensemble_member_splits = np.arange(0, self.params.num_ensemble_members+self.params.ensemble_members_per_pred,
                                                   self.params.ensemble_members_per_pred)
                
                # For each ensemble set
                for ensemble_start, ensemble_end in zip(ensemble_member_splits[:-1], ensemble_member_splits[1:]):
                    # Create ensemble batch from ICs
                    ensemble_end = min(ensemble_end, self.params.num_ensemble_members)
                    input_surface = to_ensemble_batch(input_surface_in, ensemble_end - ensemble_start)
                    input_upper_air = to_ensemble_batch(input_upper_air_in, ensemble_end - ensemble_start)
                    constant_boundary_data = to_ensemble_batch(self.constant_boundary_data, ensemble_end - ensemble_start)
                    
                    #If using 6h and 24h models, also initialize initial condition for 24h model
                    if self.use_6h_24h_model:
                        input_surface_24h = torch.clone(input_surface)
                        input_upper_air_24h = torch.clone(input_upper_air)
                    #print('Input:', input_surface[0,1,16,1])
                    
                    # Create arrays for first year of output data
                    current_datetime = self.params.init_datetime
                    current_year = self.params.init_datetime.year
                    next_year_offset_hours = current_datetime.hour % self.params.timedelta_hours
                    time_step_in_year = 0
                    
                    # Perturb initial conditions if using perturbations
                    if self.params.epsilon_factor > 0. and self.params.init_datetime.year == 1:
                        print('Perturbing ICs...')
                        input_surface, input_upper_air = self.perturber(input_surface, input_upper_air)
                        
                    next_output_datetime = self.dataset.datetime_class(current_year+1, 1, 1, hour=next_year_offset_hours,
                                                                    has_year_zero = self.params.has_year_zero)
                    output_inference_steps = int((next_output_datetime - current_datetime).total_seconds() // 3600 / self.params.timedelta_hours)
                    output_surface = np.zeros((input_surface.shape[0], output_inference_steps,
                                                    input_surface.shape[1], input_surface.shape[2], input_surface.shape[3]),
                                                    dtype = np.float32)
                    output_upper_air = np.zeros((input_upper_air.shape[0], output_inference_steps,
                                                    input_upper_air.shape[1], input_upper_air.shape[2],
                                                    input_upper_air.shape[3], input_upper_air.shape[4]),
                                                    dtype = np.float32)
                    if self.params.has_diagnostic:
                        output_diagnostic = np.zeros((input_surface.shape[0], output_inference_steps,
                                                        len(self.params.diagnostic_variables), input_surface.shape[2], input_surface.shape[3]),
                                                        dtype = np.float32)
                    
                    # Set initial values in arrays from loaded and perturbed ICs
                    transform_start = time.time()
                    output_surface[:,time_step_in_year] = self.dataset.surface_inv_transform(input_surface.to('cpu')).numpy()
                    output_upper_air[:,time_step_in_year] = self.dataset.upper_air_inv_transform(input_upper_air.to('cpu')).numpy()
                    transform_time += time.time() - transform_start
                    
                    # For every time step in duration of ensemble emulation
                    pbar = tqdm(enumerate(self.data_loader_bcs, 0), total=len(self.data_loader_bcs), miniters=1,
                                desc = f'Emulating year {self.params.init_datetime.year}')
                    for i, bc_data in pbar:
                        # Load varying boundary conditions
                        data_start = time.time()
                        if i == 0:
                            _, _, varying_boundary_data, year = map(lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), bc_data)
                        else:
                            varying_boundary_data, year = map(lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), bc_data)
                        pbar.set_description(f'Emulating year {year}')
                        # Create ensemble of boundary conditions
                        varying_boundary_data = to_ensemble_batch(varying_boundary_data, ensemble_end - ensemble_start)
                        data_time += time.time() - data_start
                        
                        # If using 6h and 24h model and time is 0z
                        if self.use_6h_24h_model and time_step_in_year % 4 == 0:
                            # Predict to next 0z with 24h model
                            inference_start = time.time()
                            if self.params_24h.has_diagnostic:
                                out_surface_24h, out_upper_air_24h, out_diagnostic_24h = self.model_24h(input_surface_24h, 
                                                                                        constant_boundary_data, 
                                                                                        varying_boundary_data,
                                                                                        input_upper_air_24h)
                            else:
                                out_surface_24h, out_upper_air_24h = self.model_24h(input_surface_24h, constant_boundary_data, 
                                                                        varying_boundary_data, input_upper_air_24h)
                            #print(f'Output 24 at step {i}:', out_surface_24h[0,1,16,1])
                            # Predict to next 6z with 6h model
                            if self.params.has_diagnostic:
                                out_surface, out_upper_air, out_diagnostic = self.model(input_surface_24h, 
                                                                                        constant_boundary_data, 
                                                                                        varying_boundary_data,
                                                                                        input_upper_air_24h)
                            else:
                                out_surface, out_upper_air = self.model(input_surface_24h, constant_boundary_data, 
                                                                        varying_boundary_data, input_upper_air_24h)
                            #print(f'Output 6 at step {i}:', out_surface[0,1,16,1])
                            # If needed, integrate tendency predictions
                            if self.params.predict_delta:
                                input_surface, input_upper_air = self.integrator(input_surface_24h, input_upper_air_24h, out_surface, out_upper_air)
                            else:
                                input_surface, input_upper_air = out_surface, out_upper_air
                            if self.params_24h.predict_delta:
                                input_surface_24h, input_upper_air_24h = self.integrator(input_surface_24h, input_upper_air_24h, out_surface_24h, out_upper_air_24h)
                            else:
                                input_surface_24h, input_upper_air_24h = out_surface_24h, out_upper_air_24h
                            inference_time += time.time() - inference_start
                            
                            # If we have not reached the end of a year, save the non-diagnostic output from the 24h model at next 0z
                            if time_step_in_year + 4 < output_surface.shape[1]:
                                transform_start = time.time()
                                output_surface[:,time_step_in_year + 4] = self.dataset_24h.surface_inv_transform(input_surface_24h.to('cpu')).numpy()
                                output_upper_air[:,time_step_in_year + 4] = self.dataset_24h.upper_air_inv_transform(input_upper_air_24h.to('cpu')).numpy()
                                transform_time += time.time() - transform_start
                            
                            # If we have not reached the end of a year, save the output from the 6h model at next 6z
                            if time_step_in_year + 1 < output_surface.shape[1]:
                                transform_start = time.time()
                                if self.params.has_diagnostic:
                                    output_diagnostic[:, time_step_in_year + 1] = self.dataset.diagnostic_inv_transform(out_diagnostic.to('cpu')).numpy()
                                output_surface[:,time_step_in_year + 1] = self.dataset.surface_inv_transform(input_surface.to('cpu')).numpy()
                                output_upper_air[:,time_step_in_year + 1] = self.dataset.upper_air_inv_transform(input_upper_air.to('cpu')).numpy()
                                transform_time += time.time() - transform_start
                        # If not using 6h and 24h model, or time is 6z or 12z
                        elif (not self.use_6h_24h_model) or (time_step_in_year+1) % 4 != 0:
                            # Predict with base model
                            inference_start = time.time()
                            if self.params.has_diagnostic:
                                out_surface, out_upper_air, out_diagnostic = self.model(input_surface, 
                                                                                        constant_boundary_data, 
                                                                                        varying_boundary_data,
                                                                                        input_upper_air)
                            else:
                                out_surface, out_upper_air = self.model(input_surface, constant_boundary_data, 
                                                                        varying_boundary_data, input_upper_air)
                            #print(f'Output {self.params.timdelta_hours} at step {i}:', out_surface[0,1,16,1])
                            # If needed, integrate tendency predictions
                            if self.params.predict_delta:
                                input_surface, input_upper_air = self.integrator(input_surface, input_upper_air, out_surface, out_upper_air)
                            else:
                                input_surface, input_upper_air = out_surface, out_upper_air
                            inference_time += time.time() - inference_start
                            # If we have not reached the end of a year, save the output from the model
                            if time_step_in_year + 1 < output_surface.shape[1]:
                                transform_start = time.time()
                                if self.params.has_diagnostic:
                                    output_diagnostic[:, time_step_in_year + 1] = self.dataset.diagnostic_inv_transform(out_diagnostic.to('cpu')).numpy()
                                output_surface[:,time_step_in_year + 1] = self.dataset.surface_inv_transform(input_surface.to('cpu')).numpy()
                                output_upper_air[:,time_step_in_year + 1] = self.dataset.upper_air_inv_transform(input_upper_air.to('cpu')).numpy()
                                transform_time += time.time() - transform_start
                                
                        # If we are using the 6h and 24h model and we are at 18z
                        elif self.use_6h_24h_model and (time_step_in_year+1) % 4 == 0 and self.params_24h.has_diagnostic:
                            # Get the 24h model diagnostic output and subtract the accumulated precip from the 3 6h model predictions
                            transform_start = time.time()
                            output_diagnostic_24h = self.dataset_24h.diagnostic_inv_transform(out_diagnostic_24h.to('cpu')).numpy()
                            if 'pr_6h' in self.params.diagnostic_variables:
                                output_diagnostic_24h[:, self.pr_6h_idx] -= np.sum(output_diagnostic[:, time_step_in_year+1:time_step_in_year+4, self.pr_6h_idx], axis = 1)
                            if time_step_in_year + 1 < output_surface.shape[1]:
                                output_diagnostic[:, time_step_in_year + 1] = output_diagnostic_24h
                            transform_time += time.time() - transform_start
                        
                        # Go to next yearly time step
                        time_step_in_year += 1
                        
                        # If we have reached the end of a year of predictions
                        if time_step_in_year == output_surface.shape[1]:
                            # Reset the yearly timestep counter
                            time_step_in_year = 0
                            # Convert numpy data to xarray
                            conversion_start = time.time()
                            if self.params.has_diagnostic:
                                ensemble_datasets = self.convert_ensemble_to_xarray(\
                                    output_surface.reshape(input_surface_in.shape[0], ensemble_end - ensemble_start, *output_surface.shape[1:]).copy(),
                                    output_upper_air.reshape(input_upper_air_in.shape[0], ensemble_end - ensemble_start, *output_upper_air.shape[1:]).copy(),
                                    particle_idxs, current_datetime, next_output_datetime, np.arange(ensemble_start, ensemble_end),
                                    diagnostic_prediction = output_diagnostic.reshape(input_surface_in.shape[0], ensemble_end - ensemble_start, *output_diagnostic.shape[1:]).copy())
                            else:
                                ensemble_datasets = self.convert_ensemble_to_xarray(\
                                    output_surface.reshape(input_surface_in.shape[0], ensemble_end - ensemble_start, *output_surface.shape[1:]).copy(),
                                    output_upper_air.reshape(input_upper_air_in.shape[0], ensemble_end - ensemble_start, *output_upper_air.shape[1:]).copy(),
                                    particle_idxs, current_datetime, next_output_datetime, np.arange(ensemble_start, ensemble_end))
                            conversion_time += time.time() - conversion_start
                            
                            # If computing an observable, send this computatation to a separate thread
                            if type(self.obs_function) is not type(None):
                                obs_start = time.time()
                                self.obs_functions(deepcopy(ensemble_datasets)[0], particle_idxs.numpy()[0], ensemble_start, ensemble_end, deepcopy(self.obs_args))
                                obs_time += time.time() - obs_start
                                
                            # If saving the forecasts, send this computation to another thread
                            if self.params.save_forecasts:
                                # Queue the results for asynchronous saving
                                save_start = time.time()
                                self.save_prediction(deepcopy(ensemble_datasets), particle_idxs.numpy(), ensemble_start, ensemble_end)
                                save_time += time.time() - save_start
                                        
                            
                            current_year += 1
                            # If this was not the final year
                            if current_year < self.params.final_datetime.year:
                                # Get the number of time steps in the next year
                                current_datetime = next_output_datetime
                                next_year_offset_hours = current_datetime.hour % self.params.timedelta_hours
                                next_output_datetime = self.dataset.datetime_class(current_year+1, 1, 1, hour=next_year_offset_hours,
                                                                                has_year_zero = self.params.has_year_zero)
                                output_inference_steps = int((next_output_datetime - current_datetime).total_seconds() // 3600 / self.params.timedelta_hours)
                                
                                # Create new numpy arrays for the next year of data
                                output_surface = np.zeros((input_surface.shape[0], output_inference_steps,
                                                                input_surface.shape[1], input_surface.shape[2], input_surface.shape[3]),
                                                                dtype = np.float32)
                                output_upper_air = np.zeros((input_upper_air.shape[0], output_inference_steps,
                                                                input_upper_air.shape[1], input_upper_air.shape[2],
                                                                input_upper_air.shape[3], input_upper_air.shape[4]),
                                                                dtype = np.float32)
                                if self.params.has_diagnostic:
                                    output_diagnostic = np.zeros((input_surface.shape[0], output_inference_steps,
                                                                    len(self.params.diagnostic_variables), input_surface.shape[2], input_surface.shape[3]),
                                                                    dtype = np.float32)
                                
                                # Set first time step of new array to last prediction from model
                                transform_start = time.time()
                                if self.use_6h_24h_model and self.params.init_datetime.hour == 0:
                                    if self.params.has_diagnostic:
                                        output_diagnostic[:, time_step_in_year] = output_diagnostic_24h
                                    output_surface[:,time_step_in_year] = self.dataset_24h.surface_inv_transform(input_surface_24h.to('cpu')).numpy()
                                    output_upper_air[:,time_step_in_year] = self.dataset_24h.upper_air_inv_transform(input_upper_air_24h.to('cpu')).numpy()
                                else:
                                    if self.params.has_diagnostic:
                                        output_diagnostic[:, time_step_in_year] = self.dataset.diagnostic_transform(out_diagnostic.to('cpu')).numpy()
                                    output_surface[:,time_step_in_year] = self.dataset.surface_inv_transform(input_surface.to('cpu')).numpy()
                                    output_upper_air[:,time_step_in_year] = self.dataset.upper_air_inv_transform(input_upper_air.to('cpu')).numpy()
                                    if self.use_6h_24h_model:
                                        input_surface_24h, input_upper_air_24h = input_surface, input_upper_air
                                transform_time += time.time() - transform_start

        total_time = time.time() - total_start

        logs = {
            'total_time': total_time,
            'data_time': data_time,
            'inference_time': inference_time,
            'transform_time': transform_time,
            'conversion_time': conversion_time,
            'obs_time': obs_time,
            'save_time': save_time
        }

        logging.info(f"Validation logs: {logs}")

        return total_time, logs
    
    def convert_ensemble_to_xarray(self, surface_prediction, upper_air_prediction, particle_idxs, current_year_datetime, next_year_datetime,
                                   ensemble_idxs = None, diagnostic_prediction = None):
        batch_size, num_ensemble_members, time_steps, num_surface_vars, lat, lon = surface_prediction.shape
        if type(ensemble_idxs) is type(None):
            ensemble_idxs = np.arange(num_ensemble_members)
        #if batch_size > 1:
        #    Warning('Obs functions assume a batch size of 1!')
        # print(f"TIME STEPS ARE: {time_steps}")
        datasets = []
        if next_year_datetime.year == self.params.final_datetime.year:
            save_level_idxs = np.arange(len(self.params.levels))
            if self.params.use_sigma_levels:
                save_sigma_level_idxs = np.arange(len(self.params.sigma_levels))
        else:
            save_level_idxs = self.save_level_idxs
            if self.params.use_sigma_levels:
                save_sigma_level_idxs = self.save_sigma_level_idxs

        for sample, particle_idx in enumerate(particle_idxs):
            # time_range = xr.cftime_range(
            #     start_time + timedelta(hours=params['timedelta_hours'] * sample * time_steps),
            #     periods=time_steps,
            #     freq=f"{params['timedelta_hours']}h"
            # )
            # time_range = [start_times[sample] + timedelta(hours=lt * params['timedelta_hours']) for lt in params['forecast_lead_times']]
            # time_range = [start_time + timedelta(hours=lt * params['timedelta_hours']) for lt in params['forecast_lead_times']]
            time_range = xr.date_range(
                current_year_datetime, next_year_datetime,
                freq = f'{self.params.timedelta_hours}h', inclusive='left', calendar = self.params.calendar,
                use_cftime = True
            )
            #time_range = time_range[-time_steps:]
            #print(time_range[0], time_range[-1])
            

            # Determine the level coordinate name based on params.lev
            level_coord_name = 'lev' if self.params.lev == 'lev' else 'plev'
            if hasattr(self.dataset, 'sigma_levels'):
                if self.params.lev == 'lev':
                    levels = self.dataset.sigma_levels
                else:
                    levels = self.dataset.levels
            else:
                levels = self.dataset.levels
                
            if self.params.lev == 'lev' and ('zg' in self.params.upper_air_variables or 'geopotential' in self.params.upper_air_variables):
                coordinates = {
                    'ensemble_idx': ensemble_idxs,
                    'time': time_range,
                    level_coord_name: levels[save_sigma_level_idxs],
                    'plev': self.dataset.levels[save_level_idxs],
                    'lat': self.params.lat,
                    'lon': self.params.lon
                }
            else:
                coordinates = {
                    'ensemble_idx': ensemble_idxs,
                    'time': time_range,
                    level_coord_name: levels[save_level_idxs],
                    'lat': self.params.lat,
                    'lon': self.params.lon
                }

            dataset = xr.Dataset(
                coords=coordinates,
                attrs=dict(description=f"Prediction from {self.params.nettype} model run, particle {particle_idx}")
            )

            for idx, var in enumerate(self.dataset.surface_variables):
                da = xr.DataArray(
                    data=surface_prediction[sample, :, :, idx],
                    dims=["ensemble_idx", "time", "lat", "lon"],
                    coords={'ensemble_idx': ensemble_idxs,
                            'time': time_range,
                            'lat': self.params.lat,
                            'lon': self.params.lon}
                )
                #da = da.assign_attrs(self.dataset.data_dss[0][var].attrs)
                dataset[var] = da

            if type(diagnostic_prediction) is not type(None):
                for idx, var in enumerate(self.dataset.diagnostic_variables):
                    da = xr.DataArray(
                        data=diagnostic_prediction[sample, :, :, idx],
                        dims=["ensemble_idx", "time", "lat", "lon"],
                        coords={'ensemble_idx': ensemble_idxs,
                                'time': time_range,
                                'lat': self.params.lat,
                                'lon': self.params.lon}
                    )
                    #da = da.assign_attrs(self.dataset.data_dss[0][var].attrs)
                    dataset[var] = da

            for idx, var in enumerate(self.dataset.upper_air_variables):
                if self.params.lev == 'lev' and (var == 'zg' or var == 'geopotential'):
                    da = xr.DataArray(
                        data=upper_air_prediction[sample, :, :, idx][:, :, save_level_idxs],
                        dims=["ensemble_idx", "time", "plev", "lat", "lon"],
                        coords = {
                            'ensemble_idx': ensemble_idxs,
                            'time': time_range,
                            'plev': dataset.plev.values,
                            'lat': self.params.lat,
                            'lon': self.params.lon
                        }
                    )
                elif self.params.lev == 'lev':
                    da = xr.DataArray(
                        data=upper_air_prediction[sample, :, :, idx][:, :, save_sigma_level_idxs],
                        dims=["ensemble_idx", "time", self.params.lev, "lat", "lon"],
                        coords = {
                            'ensemble_idx': ensemble_idxs,
                            'time': time_range,
                            self.params.lev: dataset.lev.values,
                            'lat': self.params.lat,
                            'lon': self.params.lon
                        }
                    )
                else:
                    da = xr.DataArray(
                        data=upper_air_prediction[sample, :, :, idx][:, :, save_level_idxs],
                        dims=["ensemble_idx", "time", level_coord_name, "lat", "lon"],
                        coords = {
                            'ensemble_idx': ensemble_idxs,
                            'time': time_range,
                            self.params.lev: dataset[self.params.lev].values,
                            'lat': self.params.lat,
                            'lon': self.params.lon
                        }
                    )
                #da = da.assign_attrs(self.dataset.data_dss[0][var].attrs)
                dataset[var] = da

            datasets.append(dataset)
                

        return datasets
    
    def save_prediction(self, ensemble_datasets, particle_idxs, ensemble_start, ensemble_end):
        savedir = self.params.output_dir
        save_basename = os.path.join(savedir, self.params.save_basename)
        if not os.path.isdir(savedir):
            os.makedirs(savedir, exist_ok=True)
        for i, (dataset, particle_idx) in enumerate(zip(ensemble_datasets, particle_idxs)):
            for j, ensemble_member in enumerate(range(ensemble_start, ensemble_end)):
                total_run_iter = (self.params.run_iter - 1) * (self.params.num_ensemble_members * len(self.params.init_nc_filepaths)) + \
                    self.params.num_ensemble_members * int(particle_idx) + ensemble_member
                current_year = dataset.time[0].item().year
                dataset_in = dataset.sel(ensemble_idx = ensemble_member)
                dataset_in = dataset_in.drop_vars("ensemble_idx")
                print(f'Saving prediction {total_run_iter} year {current_year}...')
                #dataset = dataset.chunk({'time': 1, self.params.lev: 1})
                #if self.params.use_sigma_levels and ('zg' in self.params.upper_air_variables or 'geopotential' in self.params.upper_air_variables):
                #    dataset = dataset.chunk({'plev': 1})
                #filename = f'{self.params.nettype}_{self.params.run_num}_{self.params['timedelta_hours']}h_{self.params['inference_steps']}step_{self.params.val_start_year}_{batch_idx * self.params.batch_size + sample}.nc'
                filename = save_basename + f'_member{total_run_iter:03}_y{current_year:04}.nc'
                if filename in self.params.init_nc_filepaths:
                    print(f'Skipping ic file: {os.path.basename(filename)}')
                else:
                    if os.path.isfile(filename):
                        os.remove(filename)
                    dataset_in.to_netcdf(filename)
            dataset.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='0305', type=str)
    parser.add_argument("--yaml_config", default='v2.0/config/PANGU_PLASIM_H5_DSI_4_test.yaml', type=str)
    parser.add_argument("--use_6h_24h_model", default = False, action="store_true")
    parser.add_argument("--config", default='PLASIM', type=str)
    parser.add_argument("--enable_amp", default=True, action='store_true')
    parser.add_argument("--epsilon_factor", default=0, type=float)
    parser.add_argument("--debug", default=False, action='store_true')
    parser.add_argument("--init_datetime", default="", type=str)
    parser.add_argument("--final_datetime", default="", type=str)
    parser.add_argument("--init_nc_filepaths", required=True, type=str)
    parser.add_argument("--async_save", default = False, action='store_true')
    parser.add_argument("--run_iter", default = 1, type = int)
    parser.add_argument("--output_dir", default = "", type = str)
    parser.add_argument("--save_basename", default = "", type = str)

    ####### for UCAR
    parser.add_argument("--local-rank", type=int)
    #######

    args = parser.parse_args()

    #config = 'PLASIM'
    #yaml_config = 'config/PANGU_PLASIM_H5_PERLMUTTER_0510_ensemble.yaml'
    #just_validate = True
    #debug = True
    #run_num = '0510'
    os.environ["WANDB_MODE"] = "offline"
    if args.use_6h_24h_model:
        run_nums = args.run_num.split(',')
        yaml_configs = args.yaml_config.split(',')
    else:
        run_nums = [args.run_num]
        yaml_configs = [args.yaml_config]
    params_list = [YParams(os.path.abspath(yaml_config), args.config) for yaml_config in yaml_configs]
    params = params_list[0]
    #params['epsilon_factor'] = args.epsilon_factor
    params['run_iter'] = args.run_iter
    if hasattr(params, 'diagnostic_variables'):
        if len(params.diagnostic_variables) > 0:
            params['has_diagnostic'] = True
        else:
            params['has_diagnostic'] = False
    else:
        params['has_diagnostic'] = False
    print(f'Has diagnostic: {params.has_diagnostic}')
    if args.use_6h_24h_model:
        params_list[1]['has_diagnostic'] = params.has_diagnostic
    if not hasattr(params, 'num_ensemble_members'):
        params['num_ensemble_members'] = 1
    params['init_nc_filepaths'] = args.init_nc_filepaths.split(',')
    params['nc_bc_offset'] = 18
    if len(args.output_dir) > 0:
        params['output_dir'] = args.output_dir
    if len(args.save_basename) > 0:
        params['save_basename'] = args.save_basename
    if not hasattr(params, 'ensemble_members_per_pred'):
        params['ensemble_members_per_pred'] = params.num_ensemble_members
    if args.debug:
        params['world_size'] = 1
    else:
        print('World size from OS: %d' % int(os.environ['WORLD_SIZE']))
        print('World size from Cuda: %d' % torch.cuda.device_count())
        if 'WORLD_SIZE' in os.environ:
            params['world_size'] = int(os.environ['WORLD_SIZE'])
            print(params['world_size'])
        else:
            params['world_size'] = torch.cuda.device_count()
            print(params['world_size'])
    params['batch_size'] = params['world_size']
    if params['world_size'] > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        if 'derecho' in str(Path(__file__)):
            local_rank = args.local_rank
        else:
            local_rank = int(os.environ["LOCAL_RANK"])

        gpu = local_rank
        world_rank = dist.get_rank()
        # print("##########WORLD RANK: TESTING ", world_rank)

        params['global_batch_size'] = params.batch_size
        params['batch_size'] = int(params.batch_size//params['world_size'])
    else:
        world_rank = 0
        local_rank = 0
    if world_rank == 0:
        print(f'World size: {params.world_size}')
            
    torch.manual_seed(world_rank)
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

    if world_rank == 0:
        logging.info("Resuming from existing checkpoint.")
        
    if len(args.init_datetime) == 0:
        if hasattr(params, "init_datetime"):
            params['init_datetime'] = cftime.datetime.strptime(params.init_datetime, "%Y-%m-%d_%H:%M:%S",
                                                                            has_year_zero = params.has_year_zero,
                                                                            calendar = 'proleptic_gregorian')
        else:
            params['init_datetime'] = cftime.datetime(params.val_year_start, 1, 1, 0, has_year_zero = params.has_year_zero,
                                                            calendar = 'proleptic_gregorian')
    else:
        params['init_datetime'] = cftime.datetime.strptime(args.init_datetime, "%Y-%m-%d_%H:%M:%S",
                                                                            has_year_zero = params.has_year_zero,
                                                                            calendar = 'proleptic_gregorian')
    params['init_datetime'] = cftime.DatetimeProlepticGregorian(params.init_datetime.year,
                                                                params.init_datetime.month,
                                                                params.init_datetime.day,
                                                                hour = params.init_datetime.hour,
                                                                has_year_zero = params.has_year_zero)
    
    params['init_nc_timestep_offset'] = []
    for file in params.init_nc_filepaths:
        ds = xr.open_dataset(file, engine = 'netcdf4')
        index = ds.get_index("time").get_loc(params["init_datetime"])
        params['init_nc_timestep_offset'].append(index)
        ds.close()
    print('Init filepaths:')
    print(params.init_nc_filepaths)
    print("Init .nc file time offset:")
    print(params['init_nc_timestep_offset'])
    if len(args.final_datetime) == 0:
        if hasattr(params, "final_datetime"):
            params['final_datetime'] = cftime.datetime.strptime(params.final_datetime, "%Y-%m-%d_%H:%M:%S",
                                                                            has_year_zero = params.has_year_zero,
                                                                            calendar = 'proleptic_gregorian')
        else:
            params['final_datetime'] = cftime.datetime(params.val_year_end, 1, 1, 0, has_year_zero = params.has_year_zero,
                                                            calendar = 'proleptic_gregorian')
    else:
        params['final_datetime'] = cftime.datetime.strptime(args.final_datetime, "%Y-%m-%d_%H:%M:%S",
                                                                            has_year_zero = params.has_year_zero,
                                                                            calendar = 'proleptic_gregorian')
    params['final_datetime'] = cftime.DatetimeProlepticGregorian(params.final_datetime.year,
                                                                params.final_datetime.month,
                                                                params.final_datetime.day,
                                                                hour = params.final_datetime.hour,
                                                                has_year_zero = params.has_year_zero)
    print('Init datetime:')
    print(params.init_datetime)
    print('Final datetime:')
    print(params.final_datetime)

    params['local_rank'] = local_rank
    if len(params_list) > 1:
        params_list[1]['local_rank'] = local_rank
    params['enable_amp'] = True

    # Add indicator for precision method and engine
    if params['use_transformer_engine']:
        print("Using Transformer Engine")
    else:
        print("Using PyTorch native")

    if params['enable_fp8']:
        print("with FP8 precision")
    elif params['enable_amp']:
        print("with Automatic Mixed Precision (AMP)")
    else:
        print("with full precision")
        
    params_list[0] = params
    
    # Set up directories
    for params_i, run_num in zip(params_list, run_nums):
        expDir = os.path.join(params_i.exp_dir, args.config, str(run_num))
        if world_rank == 0:
            if not os.path.isdir(expDir):
                os.makedirs(expDir)
                os.makedirs(os.path.join(expDir, 'training_checkpoints/'))

        params_i['experiment_dir'] = os.path.abspath(expDir)
        ckpt_path_globstr = 'training_checkpoints/ckpt_*.tar'
        best_ckpt_path = 'training_checkpoints/best_ckpt.tar'
        params_i['checkpoint_path_globstr'] = os.path.join(expDir, ckpt_path_globstr)
        params_i['best_checkpoint_path'] = os.path.join(expDir, best_ckpt_path)

        checkpoint_exists = os.path.isfile(params_i.best_checkpoint_path)

        # Determine whether to resume or start fresh
        params_i['resuming'] = True
    
    if world_rank == 0:
        log_file = f'out.log'
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(params_list[0].experiment_dir, log_file))
        logging_utils.log_versions()
        params_list[0].log()

        params_list[0]['log_to_wandb'] = False
        params_list[0]['log_to_screen'] = (world_rank == 0) and params_list[0]['log_to_screen']
        
    if world_rank == 0:
        for params_i in params_list:
            hparams = ruamelDict()
            yaml = YAML()
            for key, value in params_i.params.items():
                hparams[str(key)] = str(value)
            with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
                yaml.dump(hparams,  hpfile)
                
                
    stepper = Stepper(params_list, world_rank, use_6h_24h_model = args.use_6h_24h_model,
                      async_save = args.async_save)
    stepper.predict()


    logging.info('DONE ---- rank %d' % world_rank)
    if params['world_size'] > 1:
        dist.barrier()
        cleanup()