import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from networks.pangu import PanguModel_Plasim
from networks.pangu_legacy import PanguModel_Plasim as PanguModel_Plasim_Legacy
from networks.modulus_sfno.sfnonet import SphericalFourierNeuralOperatorNet_v2 as SFNO
from tqdm import tqdm
from ruamel.yaml.comments import CommentedMap as ruamelDict
from ruamel.yaml import YAML
from collections import OrderedDict
import matplotlib.pyplot as plt
from utils.data_loader_multifiles import get_data_loader, datetime_class_from_calendar
from utils.YParams import YParams
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
from datetime import timedelta
# import transformer_engine.pytorch as te
# from transformer_engine.common import recipe
# from transformer_engine.pytorch import fp8_autocast
from torch.amp import autocast
from torch.profiler import profile, record_function, ProfilerActivity
from itertools import product
import time 
from multiprocessing import Process
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
import warnings

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
        
import atexit
atexit.register(cleanup)

def to_ensemble_batch(data, ens_members):
    """Convert batch of M samples (M, ...) to a batch of (M*ens_members, ...)."""
    return data.repeat_interleave(ens_members, dim=0)

def compute_A_ensemble(args):
    """
    Computes the A values for an ensemble forecast.
    """
    # Open the dataset for existing paths
    ds_list, particle_idxs_list, ensemble_start, ensemble_end, save_basenames, target_duration, lead_time, var, regions, PATH_REGIONS = args
    # print("DEBUG: particle_idxs_list: ", particle_idxs_list)
    # ds = xr.open_dataset(path, decode_times=True, use_cftime=True)
    # Load region boundaries from a JSON file
    # print("DEBUG: ds_list: ", ds_list)
    # print("DEBUG: particle_idxs_list: ", particle_idxs_list)
    
    # if not isinstance(ds_list, list):
    #     ds_list = [ds_list]
    # if not isinstance(particle_idxs_list, list):
    #     particle_idxs_list = [particle_idxs_list]

    for ds, particle_idx in zip(ds_list, particle_idxs_list):
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
            #print(f'Obs filepath: {filepath}')
            np.save(filepath, A.values.flatten())
        
def combine_A_ensemble(save_basenames, regions):
    for save_basename, region in tqdm(product(save_basenames, regions),
                                      total=len(save_basenames)*len(regions),
                                      desc="Combining obs files...",
                                      dynamic_ncols=True, file=logging_utils.tqdm_stream):
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
        self.run_uuid = str(uuid.uuid4())
        self.check_land_ocean_variables()
        self.get_dataset()
        self.obs_function = obs_function
        self.obs_args = obs_args
        self.save_forecasts = False
        if hasattr(self.params, 'save_forecasts'):
            self.save_forecasts = self.params.save_forecasts
        self.save_ensemble_nc = getattr(self.params, 'save_ensemble_nc', False)


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
        self.setup_model()
        

    def get_dataset(self):
        logging.info('rank %d, begin data loader init' % self.world_rank)
        # for params in self.params:
        #     print(params)

        if hasattr(self.params, 'init_nc_filepaths'):
            self.init_from_nc = True
        else:
            self.init_from_nc = False
                                                                                
        self.data_loader, self.dataset = get_data_loader(self.params, self.params.data_dir, dist.is_initialized(), 
                                                         year_start=self.params.val_year_start, 
                                                         year_end=self.params.val_year_end, train=False,
                                                         ensemble = True, init_from_nc = self.init_from_nc)
        
        if self.params.epsilon_factor > 0.:
            self.perturber = Perturber(self.params, self.dataset, device = self.device,
                                    device_idx = self.world_rank, seed = 1)
        
        
        self.constant_boundary_data = self.dataset.constant_boundary_data.unsqueeze(0) * torch.ones(self.params.batch_size, 1, 1, 1)
        self.constant_boundary_data = self.constant_boundary_data.to(self.device, non_blocking=True)
        if self.params.num_ensemble_members > 1:
            logging.info('Ensemble Mode. Ensemble size = {self.params.num_ensemble_members}\n')
        logging.info('rank %d, data loader initialized' % self.world_rank)

    def setup_model(self):
        # Set up model
        self.mask_bool, self.land_mask = self.get_land_mask_bool() #Bing: need to double check if the return is static values.
        self.model, self.model_24h = self.get_model()
        if hasattr(self.params, 'best_checkpoint_path'):
            self.restore_checkpoint(self.model, self.params.best_checkpoint_path)
            logging.info("Loading model from checkpoint: %s", self.params.best_checkpoint_path)
            if self.use_6h_24h_model:
                self.restore_checkpoint(self.model_24h, self.params_24h.best_checkpoint_path)
                logging.info("Loading 24h model from checkpoint: %s", self.params_24h.best_checkpoint_path)

    def get_land_mask_bool(self) -> torch.Tensor:
        """
        Get a boolean mask for the land or ocean based on the variable name.
        """
        mask_bool = []
        land_mask = []
        if self.params.nettype == 'pangu_plasim' or self.params.nettype == 'sfno_plasim':
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
        else:
            raise Exception("not implemented")
        return mask_bool, land_mask

    def get_model(self):
        """ 
        Get the model based on the nettype specified in params.
        """
        self.model_24h = None
        if self.params.nettype == 'pangu_plasim':
            if self.params.use_legacy_model:
                model_class = PanguModel_Plasim_Legacy
            else:
                model_class = PanguModel_Plasim
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
                self.model = model_class(self.params, land_mask = land_mask).to(self.device)
                self.integrator = Integrator(self.params, surface_ff_std=self.dataset.surface_std.detach().to(self.device),
                                            surface_delta_std=self.dataset.surface_delta_std.detach().to(self.device),
                                            upper_air_ff_std=self.dataset.upper_air_std.detach().to(self.device),
                                            upper_air_delta_std=self.dataset.upper_air_delta_std.detach().to(self.device)).to(self.device)
            else:
                if hasattr(self.params, 'mask_fill'):
                    self.model = model_class(self.params, land_mask = land_mask, 
                                            mask_fill = self.params.mask_fill).to(self.device)
                else:
                    self.model = model_class(self.params, land_mask = land_mask, 
                                                mask_fill = self.dataset.mask_fill).to(self.device)
            if self.use_6h_24h_model:
                _, dataset_24h = get_data_loader(self.params_24h, self.params_24h.data_dir, dist.is_initialized(), 
                                                 year_start=self.params_24h.val_year_start, 
                                                 year_end=self.params_24h.val_year_end, train=False,
                                                 ensemble = True, init_from_nc = self.init_from_nc)
                if self.params_24h.predict_delta:
                    self.model_24h = model_class(self.params_24h, land_mask = land_mask).to(self.device)
                    self.integrator_24h = Integrator(self.params_24h, surface_ff_std=dataset_24h.surface_std.detach().to(self.device),
                                                surface_delta_std=dataset_24h.surface_delta_std.detach().to(self.device),
                                                upper_air_ff_std=dataset_24h.upper_air_std.detach().to(self.device),
                                                upper_air_delta_std=dataset_24h.upper_air_delta_std.detach().to(self.device)).to(self.device)
                else:
                    if hasattr(self.params_24h, 'mask_fill'):
                        self.model_24h = model_class(self.params_24h, land_mask = land_mask, 
                                                mask_fill = self.params_24h.mask_fill).to(self.device)
                    else:
                        self.model_24h = model_class(self.params_24h, land_mask = land_mask, 
                                                    mask_fill = dataset_24h.mask_fill).to(self.device)
                
            #self.restore_checkpoint(params.best_checkpoint_path)
            #self.model = torch.compile(self.model, mode="max-autotune")
            #self.model = torch.compile(self.model, mode = 'default')
        elif self.params.nettype == 'sfno_plasim':
            print(f'\n\nRunning SFNO model\n\n')
            # For SFNO, the dataset determines in_chans/out_chans from variable lists
            # To ensure same architecture as Trainer, create a training-style dataset for model initialization
            # This ensures variable_list_in/out match Trainer's configuration
            try:
                # Create a temporary training-style dataset for model initialization
                # This ensures the same variable lists as used during training
                # Note: get_data_loader returns (dataloader, dataset, sampler) when train=True
                _, model_init_dataset, _ = get_data_loader(
                    self.params, self.params.data_dir, dist.is_initialized(),
                    year_start=self.params.train_year_start if hasattr(self.params, 'train_year_start') else self.params.val_year_start,
                    year_end=self.params.train_year_end if hasattr(self.params, 'train_year_end') else self.params.val_year_end,
                    train=True,  # Use train=True to match Trainer's dataset configuration
                    ensemble=False, init_from_nc=False
                )
                self.model = SFNO(self.params, model_init_dataset).to(self.device)
            except Exception as e:
                # Fallback: use inference dataset if training dataset creation fails
                logging.warning(f"Could not create training-style dataset for model initialization: {e}. Using inference dataset.")
                self.model = SFNO(self.params, self.dataset).to(self.device)
            if self.params.sync_norm and dist.is_initialized():
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            if self.params.predict_delta:
                # Use dataset for integrator stds
                self.integrator = Integrator(self.params, surface_ff_std=self.dataset.surface_std.detach().to(self.device),
                                               surface_delta_std=self.dataset.surface_delta_std.detach().to(self.device),
                                               upper_air_ff_std=self.dataset.upper_air_ff_std.detach().to(self.device),
                                               upper_air_delta_std=self.dataset.upper_air_delta_std.detach().to(self.device)).to(self.device)
        else:
            raise Exception("not implemented")

        if dist.is_initialized():
            self.model = DistributedDataParallel(self.model,
                                                 device_ids=[
                                                     self.params.local_rank],
                                                 output_device=[self.params.local_rank], find_unused_parameters=True)
            if self.use_6h_24h_model:
                self.model_24h = DistributedDataParallel(self.model_24h,
                                                 device_ids=[
                                                     self.params.local_rank],
                                                 output_device=[self.params.local_rank], find_unused_parameters=True)
        if self.params.log_to_screen:
            logging.info("Number of trainable model parameters: {}".format(self.count_parameters()))
        return self.model, self.model_24h

    def check_land_ocean_variables(self) -> None:
        """
        The function is used to update the boolean variable to check if there is land/ocean variables
        """
        #initlisation
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
        if self.use_6h_24h_model:
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

    def restore_checkpoint(self, model, checkpoint_path):
        """ We intentionally require a checkpoint_dir to be passed
            in order to allow Ray Tune to use this function """
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.params.local_rank), weights_only=False)

        # Prefer EMA state for inference (typically better), fall back to model_state
        if 'ema_state' in checkpoint and checkpoint['ema_state'] is not None:
            state_str = 'ema_state'
            logging.info(f"Using EMA state from checkpoint (preferred for inference)")
        else:
            state_str = 'model_state'
            logging.info(f"No EMA state found, using model_state")

        checkpoint_state_dict = checkpoint[state_str]

        # Check if checkpoint keys have "module." prefix (check all keys, not just first)
        checkpoint_has_module = any(key.startswith('module.') for key in checkpoint_state_dict.keys())
        # Check if model keys have "module." prefix
        model_has_module = any(key.startswith('module.') for key in model.state_dict().keys())

        # Try loading directly first
        try:
            model.load_state_dict(checkpoint_state_dict)
            logging.info('Successfully loaded checkpoint state dict')
        except Exception as e:
            logging.warning(f'Direct load failed: {e}. Attempting to fix "module." prefix mismatch...')

            new_state_dict = OrderedDict()

            if checkpoint_has_module and not model_has_module:
                # Checkpoint has "module." prefix but model doesn't - remove it
                for key, val in checkpoint_state_dict.items():
                    if key.startswith('module.'):
                        new_state_dict[key[7:]] = val
                    else:
                        new_state_dict[key] = val
                logging.info('Removed "module." prefix from checkpoint keys')
            elif not checkpoint_has_module and model_has_module:
                # Checkpoint doesn't have "module." prefix but model does - add it
                for key, val in checkpoint_state_dict.items():
                    new_state_dict['module.' + key] = val
                logging.info('Added "module." prefix to checkpoint keys')
            else:
                # Both have or both don't have - try removing anyway as fallback
                for key, val in checkpoint_state_dict.items():
                    if key.startswith('module.'):
                        new_state_dict[key[7:]] = val
                    else:
                        new_state_dict[key] = val
                logging.info('Attempting to remove "module." prefix as fallback')

            try:
                model.load_state_dict(new_state_dict)
                logging.info('Successfully loaded checkpoint after fixing "module." prefix')
            except Exception as e2:
                logging.error(f'Failed to load checkpoint even after fixing "module." prefix: {e2}')
                raise e2

        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch']
        self.epoch = checkpoint['epoch']
        logging.info(f'Restored from epoch {self.startEpoch}, iteration {self.iters}')
            
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
            
    async def save_prediction_async(self, datasets, particle_idxs, ensemble_start, ensemble_end):
        await asyncio.to_thread(self.save_prediction, datasets, particle_idxs, ensemble_start, ensemble_end)
        
    async def run_obs_function_async(self, dataset, particle_idx, ensemble_start, ensemble_end, obs_args):
        await asyncio.to_thread(self.obs_function, [dataset, particle_idx, ensemble_start, ensemble_end] + obs_args)
        
    async def save_results(self, queue):
        while True:
            item = await queue.get()
            if item is None:
                break
            datasets, particle_idxs, ensemble_start, ensemble_end = item
            save_start = time.time()
            await self.save_prediction_async(datasets, particle_idxs, ensemble_start, ensemble_end)
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

        with torch.inference_mode(), autocast(enabled=self.params.enable_amp, device_type="cuda"):
            for i, data in enumerate(self.data_loader, 0):
                data_start = time.time()
                input_surface_in, input_upper_air_in, varying_boundary_data_in = map(
                    lambda x: x.to(self.device, dtype=torch.float32), data[:-1])
                particle_idxs = data[-1]
                #print(f'Particle idxs:{particle_idxs}, world rank {self.world_rank}')
                
                actual_batch = input_surface_in.shape[0]
                if actual_batch == 0:
                    continue
                ensemble_member_splits = np.arange(0, self.params.num_ensemble_members+self.params.ensemble_members_per_pred,
                                                   self.params.ensemble_members_per_pred)
                for ensemble_start, ensemble_end in zip(ensemble_member_splits[:-1], ensemble_member_splits[1:]):
                    ensemble_end = min(ensemble_end, self.params.num_ensemble_members)
                    input_surface = to_ensemble_batch(input_surface_in, ensemble_end - ensemble_start)
                    input_upper_air = to_ensemble_batch(input_upper_air_in, ensemble_end - ensemble_start)
                    varying_boundary_data = to_ensemble_batch(varying_boundary_data_in, ensemble_end - ensemble_start)
                    # Slice to actual batch size so partial last batches work when
                    # n_particles is not divisible by the (global) batch size.
                    constant_boundary_data = to_ensemble_batch(
                        self.constant_boundary_data[:actual_batch], ensemble_end - ensemble_start)
                    #varying_boundary_data_init = to_ensemble_batch(varying_boundary_data_init_in, ensemble_end - ensemble_start)

                    input_surface, input_upper_air = self.perturber(input_surface, input_upper_air)
                    # Clamp perturbed values to float16 representable range to prevent
                    # overflow when Conv2d casts inputs to float16 under AMP autocast.
                    # The perturbation noise scale (epsilon_factor * ff_std / std) can
                    # produce extreme outliers that exceed float16 max (~65504).
                    #_fp16_max = torch.finfo(torch.float16).max
                    #input_surface = input_surface.clamp(-_fp16_max, _fp16_max)
                    #input_upper_air = input_upper_air.clamp(-_fp16_max, _fp16_max)
                    
                    

                    inference_start = time.time()
                    output_surface = np.zeros((input_surface.shape[0], self.dataset.ensemble_inference_steps,
                                                    input_surface.shape[1], input_surface.shape[2], input_surface.shape[3]),
                                                    dtype = np.float32)
                    output_upper_air = np.zeros((input_upper_air.shape[0], self.dataset.ensemble_inference_steps,
                                                    input_upper_air.shape[1], input_upper_air.shape[2],
                                                    input_upper_air.shape[3], input_upper_air.shape[4]),
                                                    dtype = np.float32)
                    if self.params.has_diagnostic:
                        output_diagnostic = np.zeros((input_surface.shape[0], self.dataset.ensemble_inference_steps,
                                                        len(self.params.diagnostic_variables), input_surface.shape[2], input_surface.shape[3]),
                                                        dtype = np.float32)
                    
                    #output_surface[:,0] = self.dataset.surface_inv_transform(input_surface.to('cpu')).numpy()
                    #output_upper_air[:,0] = self.dataset.upper_air_inv_transform(input_upper_air.to('cpu')).numpy()
                    

                    # Only show progress bar on rank 0 to avoid jumping output in DDP
                    time_step_iter = tqdm(range(self.dataset.ensemble_inference_steps),
                                         desc=f'Ensemble forecast {i}, members {ensemble_start}-{ensemble_end}',
                                         dynamic_ncols=True, file=logging_utils.tqdm_stream,
                                         disable=(self.world_rank != 0))
                    for time_step in time_step_iter:
                        inference_start = time.time()
                        if self.params.has_diagnostic:
                            out_surface, out_upper_air, out_diagnostic, _, _, _, _ = self.model(input_surface, 
                                                                                    constant_boundary_data, 
                                                                                    varying_boundary_data[:,time_step],
                                                                                    input_upper_air)
                        else:
                            out_surface, out_upper_air, _, _, _, _ = self.model(input_surface, constant_boundary_data, 
                                                                    varying_boundary_data[:,time_step], input_upper_air)
                        if self.params.predict_delta:
                            input_surface, input_upper_air = self.integrator(input_surface, input_upper_air, out_surface, out_upper_air)
                        else:
                            input_surface, input_upper_air = out_surface, out_upper_air
                        inference_time += time.time() - inference_start
                        transform_start = time.time()
                        if self.params.has_diagnostic:
                            output_diagnostic[:, time_step] = self.dataset.diagnostic_transform(out_diagnostic.to('cpu')).numpy()
                        output_surface[:,time_step] = self.dataset.surface_inv_transform(input_surface.to('cpu')).numpy()
                        output_upper_air[:,time_step] = self.dataset.upper_air_inv_transform(input_upper_air.to('cpu')).numpy()
                        transform_time += time.time() - transform_start

                    inference_time += time.time() - inference_start
                    
                    conversion_start = time.time()
                    
                    if self.params.has_diagnostic:
                        ensemble_datasets = self.convert_ensemble_to_xarray(\
                            output_surface.reshape(input_surface_in.shape[0], ensemble_end - ensemble_start, *output_surface.shape[1:]).copy(),
                            output_upper_air.reshape(input_upper_air_in.shape[0], ensemble_end - ensemble_start, *output_upper_air.shape[1:]).copy(),
                            particle_idxs, np.arange(ensemble_start, ensemble_end), 
                            diagnostic_prediction = output_diagnostic.reshape(input_surface_in.shape[0], ensemble_end - ensemble_start, *output_diagnostic.shape[1:]).copy())
                    else:
                        ensemble_datasets = self.convert_ensemble_to_xarray(\
                            output_surface.reshape(input_surface_in.shape[0], ensemble_end - ensemble_start, *output_surface.shape[1:]).copy(),
                            output_upper_air.reshape(input_upper_air_in.shape[0], ensemble_end - ensemble_start, *output_upper_air.shape[1:]).copy(),
                            particle_idxs, np.arange(ensemble_start, ensemble_end))
                    #print(f'Ensemble dataset len: {len(ensemble_datasets)}')
                    conversion_time += time.time() - conversion_start
                    
                    if type(self.obs_function) is not type(None):
                        await obs_queue.put((deepcopy(ensemble_datasets)[0], particle_idxs.numpy()[0], ensemble_start, ensemble_end, deepcopy(self.obs_args)))
                        await asyncio.sleep(0)
                        
                    if self.params.save_forecasts:
                        # Queue the results for asynchronous saving
                        await save_queue.put((deepcopy(ensemble_datasets),  particle_idxs.numpy(), ensemble_start, ensemble_end))
                        await asyncio.sleep(0)

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
        conversion_time = 0
        obs_time = 0
        save_time = 0

        with torch.inference_mode(), amp.autocast(enabled=self.params.enable_amp):
            for i, data in enumerate(self.data_loader, 0):
                data_start = time.time()
                #particle_idxs = np.arange(self.params.batch_size*i, self.params.batch_size*(i+1))
                input_surface_in, input_upper_air_in, varying_boundary_data_in = map(
                    lambda x: x.to(self.device, dtype=torch.float32), data[:-1])
                particle_idxs = data[-1]
                #print(f'Particle idxs:{particle_idxs}')

                actual_batch = input_surface_in.shape[0]
                if actual_batch == 0:
                    continue
                ensemble_member_splits = np.arange(0, self.params.num_ensemble_members+self.params.ensemble_members_per_pred,
                                                   self.params.ensemble_members_per_pred)
                if self.save_ensemble_nc:
                    particle_datasets_buf = {}  # pid -> xr.Dataset accumulated across ensemble splits
                for ensemble_start, ensemble_end in zip(ensemble_member_splits[:-1], ensemble_member_splits[1:]):
                    ensemble_end = min(ensemble_end, self.params.num_ensemble_members)
                    input_surface = to_ensemble_batch(input_surface_in, ensemble_end - ensemble_start)
                    input_upper_air = to_ensemble_batch(input_upper_air_in, ensemble_end - ensemble_start)
                    varying_boundary_data = to_ensemble_batch(varying_boundary_data_in, ensemble_end - ensemble_start)
                    # Slice to actual batch size so partial last batches work when
                    # n_particles is not divisible by the (global) batch size.
                    constant_boundary_data = to_ensemble_batch(
                        self.constant_boundary_data[:actual_batch], ensemble_end - ensemble_start)
                    #varying_boundary_data_init = to_ensemble_batch(varying_boundary_data_init_in, ensemble_end - ensemble_start)

                    if self.params.epsilon_factor > 0.:
                        input_surface, input_upper_air = self.perturber(input_surface, input_upper_air)
                    # Clamp perturbed values to float16 representable range to prevent
                    # overflow when Conv2d casts inputs to float16 under AMP autocast.
                    # The perturbation noise scale (epsilon_factor * ff_std / std) can
                    # produce extreme outliers that exceed float16 max (~65504).
                    #_fp16_max = torch.finfo(torch.float16).max
                    #input_surface = input_surface.clamp(-_fp16_max, _fp16_max)
                    #input_upper_air = input_upper_air.clamp(-_fp16_max, _fp16_max)
                    
                    

                    inference_start = time.time()
                    output_surface = np.zeros((input_surface.shape[0], self.dataset.ensemble_inference_steps,
                                                    input_surface.shape[1], input_surface.shape[2], input_surface.shape[3]),
                                                    dtype = np.float32)
                    output_upper_air = np.zeros((input_upper_air.shape[0], self.dataset.ensemble_inference_steps,
                                                    input_upper_air.shape[1], input_upper_air.shape[2],
                                                    input_upper_air.shape[3], input_upper_air.shape[4]),
                                                    dtype = np.float32)
                    if self.params.has_diagnostic:
                        output_diagnostic = np.zeros((input_surface.shape[0], self.dataset.ensemble_inference_steps,
                                                        len(self.params.diagnostic_variables), input_surface.shape[2], input_surface.shape[3]),
                                                        dtype = np.float32)
                    
                    #output_surface[:,0] = self.dataset.surface_inv_transform(input_surface.to('cpu')).numpy()
                    #output_upper_air[:,0] = self.dataset.upper_air_inv_transform(input_upper_air.to('cpu')).numpy()
                    

                    # Only show progress bar on rank 0 to avoid jumping output in DDP
                    time_step_iter = tqdm(range(self.dataset.ensemble_inference_steps),
                                         desc=f'Ensemble forecast {i}, members {ensemble_start}-{ensemble_end}',
                                         dynamic_ncols=True, file=logging_utils.tqdm_stream,
                                         disable=(self.world_rank != 0))
                    for time_step in time_step_iter:
                        # DIAGNOSTIC: Log first forward pass inputs/outputs
                        if i == 0 and ensemble_start == 0 and time_step == 0 and self.world_rank == 0:
                            logging.info("===== STEPPER predict_sync: FIRST FORWARD PASS DIAGNOSTIC =====")
                            logging.info(f"input_surface shape: {input_surface.shape}, dtype: {input_surface.dtype}")
                            logging.info(f"input_surface[:1,:,0,0]: {input_surface[:1,:,0,0]}")
                            logging.info(f"input_surface stats: min={input_surface.min().item():.6f}, max={input_surface.max().item():.6f}, mean={input_surface.mean().item():.6f}")
                            logging.info(f"constant_boundary_data shape: {constant_boundary_data.shape}")
                            logging.info(f"constant_boundary_data[:1,:,0,0]: {constant_boundary_data[:1,:,0,0]}")
                            logging.info(f"varying_boundary_data[:,{time_step}] shape: {varying_boundary_data[:,time_step].shape}")
                            logging.info(f"varying_boundary_data[:1,{time_step},:,0,0]: {varying_boundary_data[:1,time_step,:,0,0]}")
                            logging.info(f"input_upper_air shape: {input_upper_air.shape}")
                            logging.info(f"input_upper_air[:1,:,:,0,0]: {input_upper_air[:1,:,:,0,0]}")
                            logging.info(f"Model training mode: {self.model.training}")
                            raw_model = self.model.module if hasattr(self.model, 'module') else self.model
                            logging.info(f"Raw model training mode: {raw_model.training}")
                        
                        if self.params.has_diagnostic:
                            out_surface, out_upper_air, out_diagnostic, _, _, _, _ = self.model(input_surface, 
                                                                                    constant_boundary_data, 
                                                                                    varying_boundary_data[:,time_step],
                                                                                    input_upper_air)
                            output_diagnostic[:, time_step] = self.dataset.diagnostic_transform(out_diagnostic.to('cpu')).numpy()
                        else:
                            out_surface, out_upper_air, _, _, _, _ = self.model(input_surface, constant_boundary_data, 
                                                                    varying_boundary_data[:,time_step], input_upper_air)
                        
                        # DIAGNOSTIC: Log first forward pass output
                        if i == 0 and ensemble_start == 0 and time_step == 0 and self.world_rank == 0:
                            logging.info(f"out_surface shape: {out_surface.shape}, dtype: {out_surface.dtype}")
                            logging.info(f"out_surface[:1,:,0,0]: {out_surface[:1,:,0,0]}")
                            logging.info(f"out_surface stats: min={out_surface.min().item():.6f}, max={out_surface.max().item():.6f}, mean={out_surface.mean().item():.6f}")
                            logging.info(f"out_upper_air[:1,:,:,0,0]: {out_upper_air[:1,:,:,0,0]}")
                            logging.info("===== END STEPPER FIRST FORWARD PASS DIAGNOSTIC =====")
                        
                        if self.params.predict_delta:
                            input_surface, input_upper_air = self.integrator(input_surface, input_upper_air, out_surface, out_upper_air)
                        else:
                            input_surface, input_upper_air = out_surface, out_upper_air
                        output_surface[:,time_step] = self.dataset.surface_inv_transform(input_surface.to('cpu')).numpy()
                        output_upper_air[:,time_step] = self.dataset.upper_air_inv_transform(input_upper_air.to('cpu')).numpy()

                    inference_time += time.time() - inference_start
                    
                    conversion_start = time.time()
                    
                    if self.params.has_diagnostic:
                        ensemble_datasets = self.convert_ensemble_to_xarray(\
                            output_surface.reshape(input_surface_in.shape[0], ensemble_end - ensemble_start, *output_surface.shape[1:]).copy(),
                            output_upper_air.reshape(input_upper_air_in.shape[0], ensemble_end - ensemble_start, *output_upper_air.shape[1:]).copy(),
                            particle_idxs, np.arange(ensemble_start, ensemble_end), 
                            diagnostic_prediction = output_diagnostic.reshape(input_surface_in.shape[0], ensemble_end - ensemble_start, *output_diagnostic.shape[1:]).copy())
                    else:
                        ensemble_datasets = self.convert_ensemble_to_xarray(\
                            output_surface.reshape(input_surface_in.shape[0], ensemble_end - ensemble_start, *output_surface.shape[1:]).copy(),
                            output_upper_air.reshape(input_upper_air_in.shape[0], ensemble_end - ensemble_start, *output_upper_air.shape[1:]).copy(),
                            particle_idxs, np.arange(ensemble_start, ensemble_end))
                    #print(f'Ensemble dataset len: {len(ensemble_datasets)}')
                    conversion_time += time.time() - conversion_start

                    if self.save_ensemble_nc:
                        for j, pid in enumerate(particle_idxs):
                            pid_int = int(pid)
                            if pid_int not in particle_datasets_buf:
                                particle_datasets_buf[pid_int] = ensemble_datasets[j]
                            else:
                                particle_datasets_buf[pid_int] = xr.concat(
                                    [particle_datasets_buf[pid_int], ensemble_datasets[j]],
                                    dim='ensemble_idx')

                    """
                    print(f'Comparing to particle {particle_idxs[0]}')

                    chicago_tas = ensemble_datasets[0]['tas'].sel(lat=41.8781, lon=-87.6298, method='nearest')
                    
                    # Get input tas value for Chicago location
                    # Find nearest lat/lon indices for Chicago
                    lat_idx = np.argmin(np.abs(np.array(self.params.lat) - 41.8781))
                    lon_idx = np.argmin(np.abs(np.array(self.params.lon) - (-87.6298)))
                    
                    # Get input tas value from input_surface_in (before perturbation)
                    # input_surface_in shape: (batch_size, num_vars, lat, lon)
                    # tas is at variable index 1
                    input_tas_value = input_surface_in[0, 1, lat_idx, lon_idx].cpu().numpy()
                    
                    # Unstandardize: value * std + mean
                    input_tas_unstandardized = input_tas_value * self.dataset.surface_std[1].cpu().numpy() + self.dataset.surface_mean[1].cpu().numpy()
                    
                    # Get init_datetime for this particle
                    init_datetime = self.params.init_datetime if not hasattr(self.params, 'init_datetimes') else self.params.init_datetimes[particle_idxs[0]]
                    
                    # Create input time point (6 hours before first prediction)
                    input_time = init_datetime
                    
                    # Append input value to chicago_tas
                    # Create a new DataArray for the input time point with same structure as chicago_tas
                    input_tas_data = np.full((len(chicago_tas.ensemble_idx), 1), input_tas_unstandardized)
                    input_tas_da = xr.DataArray(
                        data=input_tas_data,
                        dims=['ensemble_idx', 'time'],
                        coords={
                            'ensemble_idx': chicago_tas.ensemble_idx,
                            'time': [input_time]
                        }
                    )
                    # Concatenate along time dimension
                    chicago_tas = xr.concat([input_tas_da, chicago_tas], dim='time')
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        true_data = xr.open_dataset(self.dataset.params.init_nc_filepaths[particle_idxs[0]])
                    true_tas = true_data['tas'].sel(lat=41.8781, lon=-87.6298, method='nearest')
                    
                    # Adjust true data time range to include input time (timedelta_hours before first prediction)
                    true_tas_time_start = input_time
                    true_tas_time_end = chicago_tas.time.values[-1]
                    true_tas_selected = true_tas.sel(time=(true_tas.time >= true_tas_time_start) & (true_tas.time <= true_tas_time_end))

                    fig = plt.figure()
                    for idx in chicago_tas.ensemble_idx:
                        (chicago_tas.sel(ensemble_idx=idx) - true_tas_selected).plot()
                    plt.savefig(os.path.join(self.params.plots_dir, f'chicago_tas_comparison_particle_{particle_idxs[0]}_ensemble_{ensemble_start}_{ensemble_end}.png'))
                    plt.close()

                    true_data.close()
                    """


                    if self.save_forecasts:
                        save_start = time.time()
                        self.save_prediction(ensemble_datasets, particle_idxs, ensemble_start, ensemble_end)
                        save_time += time.time() - save_start
                    
                    if self.obs_function:
                        obs_start = time.time()
                        # print("DEBUG: particle_idxs from ensemble_inference.py", particle_idxs)
                        self.obs_function([ensemble_datasets, particle_idxs, ensemble_start, ensemble_end] + self.obs_args)
                        obs_time += time.time() - obs_start

                if self.save_ensemble_nc:
                    save_start = time.time()
                    self._save_ensemble_nc_files(particle_datasets_buf, particle_idxs)
                    save_time += time.time() - save_start

        total_time = time.time() - total_start

        logs = {
            'total_time': total_time,
            'data_time': data_time,
            'inference_time': inference_time,
            'conversion_time': conversion_time,
            'obs_time': obs_time,
            'save_time': save_time
        }

        logging.info(f"Validation logs: {logs}")

        return total_time, logs
    
    def convert_ensemble_to_xarray(self, surface_prediction, upper_air_prediction, particle_idxs, ensemble_idxs = None, diagnostic_prediction = None):
        batch_size, num_ensemble_members, time_steps, num_surface_vars, lat, lon = surface_prediction.shape
        if type(ensemble_idxs) is type(None):
            ensemble_idxs = np.arange(num_ensemble_members)
        #if batch_size > 1:
        #    Warning('Obs functions assume a batch size of 1!')
        # print(f"TIME STEPS ARE: {time_steps}")
        datasets = []

        for sample, particle_idx in enumerate(particle_idxs):
            # time_range = xr.cftime_range(
            #     start_time + timedelta(hours=params['timedelta_hours'] * sample * time_steps),
            #     periods=time_steps,
            #     freq=f"{params['timedelta_hours']}h"
            # )
            # time_range = [start_times[sample] + timedelta(hours=lt * params['timedelta_hours']) for lt in params['forecast_lead_times']]
            # time_range = [start_time + timedelta(hours=lt * params['timedelta_hours']) for lt in params['forecast_lead_times']]
            init_datetime = self.params.init_datetime if not hasattr(self.params, 'init_datetimes') else self.params.init_datetimes[particle_idx]
            time_range = [init_datetime + timedelta(hours=step * self.params['timedelta_hours']) for step in range(1, time_steps + 1)]
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
                    level_coord_name: levels,
                    'plev': self.dataset.levels,
                    'lat': self.params.lat,
                    'lon': self.params.lon
                }
            else:
                coordinates = {
                    'ensemble_idx': ensemble_idxs,
                    'time': time_range,
                    level_coord_name: levels,
                    'lat': self.params.lat,
                    'lon': self.params.lon
                }

            dataset = xr.Dataset(
                coords=coordinates,
                attrs=dict(description=f"Prediction from {self.params.nettype} model run, particle {particle_idx}")
            )
            # Get save_var_dict for filtering (None means save all)
            save_var_dict = getattr(self.params, 'save_var_dict', None)

            for idx, var in enumerate(self.dataset.surface_variables):
                # Skip if save_var_dict exists and var not in it
                if save_var_dict is not None and var not in save_var_dict:
                    continue
                da = xr.DataArray(
                    data=surface_prediction[sample, :, :, idx],
                    dims=["ensemble_idx", "time", "lat", "lon"],
                    coords={'ensemble_idx': ensemble_idxs,
                            'time': time_range,
                            'lat': self.params.lat,
                            'lon': self.params.lon}
                )
                dataset[var] = da

            # for idx, var in enumerate(self.dataset.surface_variables):
            #     da = xr.DataArray(
            #         data=surface_prediction[sample, :, :, idx],
            #         dims=["ensemble_idx", "time", "lat", "lon"],
            #         coords={'ensemble_idx': ensemble_idxs,
            #                 'time': time_range,
            #                 'lat': self.params.lat,
            #                 'lon': self.params.lon}
            #     )
            #     #da = da.assign_attrs(self.dataset.data_dss[0][var].attrs)
            #     dataset[var] = da

            if type(diagnostic_prediction) is not type(None):
                for idx, var in enumerate(self.dataset.diagnostic_variables):
                    # Skip if save_var_dict exists and var not in it
                    if save_var_dict is not None and var not in save_var_dict:
                        continue
                    da = xr.DataArray(
                        data=diagnostic_prediction[sample, :, :, idx],
                        dims=["ensemble_idx", "time", "lat", "lon"],
                        coords={'ensemble_idx': ensemble_idxs,
                                'time': time_range,
                                'lat': self.params.lat,
                                'lon': self.params.lon}
                    )
                    dataset[var] = da

            # if type(diagnostic_prediction) is not type(None):
            #     for idx, var in enumerate(self.dataset.diagnostic_variables):
            #         da = xr.DataArray(
            #             data=diagnostic_prediction[sample, :, :, idx],
            #             dims=["ensemble_idx", "time", "lat", "lon"],
            #             coords={'ensemble_idx': ensemble_idxs,
            #                     'time': time_range,
            #                     'lat': self.params.lat,
            #                     'lon': self.params.lon}
            #         )
            #         #da = da.assign_attrs(self.dataset.data_dss[0][var].attrs)
            #         dataset[var] = da

            # print(f"upper_air_prediction shape: {upper_air_prediction.shape}")
            for idx, var in enumerate(self.dataset.upper_air_variables):
                # Skip if save_var_dict exists and var not in it
                if save_var_dict is not None and var not in save_var_dict:
                    continue
                
                # Get target levels for this variable
                target_levels = save_var_dict.get(var, []) if save_var_dict else []
                
                # Determine which levels array to use
                if self.params.lev == 'lev' and (var == 'zg' or var == 'geopotential'):
                    all_levels = self.dataset.levels
                elif self.params.lev == 'lev':
                    all_levels = levels
                else:
                    all_levels = self.dataset.levels if hasattr(self.dataset, 'levels') else levels
                
                # If target_levels specified, save as 2D variables with level suffix
                if save_var_dict is not None and len(target_levels) > 0:
                    for target_lev in target_levels:
                        # Find nearest level index
                        lev_idx = np.argmin(np.abs(np.array(all_levels) - target_lev))
                        actual_lev = all_levels[lev_idx]
                        
                        # Create 2D variable with level suffix (e.g., zg_50000)
                        var_name = f"{var}_{int(actual_lev)}"
                        da = xr.DataArray(
                            data=upper_air_prediction[sample, :, :, idx, lev_idx],
                            dims=["ensemble_idx", "time", "lat", "lon"],
                            coords={
                                'ensemble_idx': ensemble_idxs,
                                'time': time_range,
                                'lat': self.params.lat,
                                'lon': self.params.lon
                            }
                        )
                        dataset[var_name] = da
                else:
                    # No filtering - save full 3D variable (original behavior)
                    if self.params.lev == 'lev' and (var == 'zg' or var == 'geopotential'):
                        da = xr.DataArray(
                            data=upper_air_prediction[sample, :, :, idx],
                            dims=["ensemble_idx", "time", "plev", "lat", "lon"],
                            coords={
                                'ensemble_idx': ensemble_idxs,
                                'time': time_range,
                                'plev': self.dataset.levels,
                                'lat': self.params.lat,
                                'lon': self.params.lon
                            }
                        )
                    elif self.params.lev == 'lev':
                        da = xr.DataArray(
                            data=upper_air_prediction[sample, :, :, idx],
                            dims=["ensemble_idx", "time", self.params.lev, "lat", "lon"],
                            coords={
                                'ensemble_idx': ensemble_idxs,
                                'time': time_range,
                                self.params.lev: levels,
                                'lat': self.params.lat,
                                'lon': self.params.lon
                            }
                        )
                    else:
                        da = xr.DataArray(
                            data=upper_air_prediction[sample, :, :, idx],
                            dims=["ensemble_idx", "time", level_coord_name, "lat", "lon"],
                            coords=coordinates
                        )
                    dataset[var] = da

            # for idx, var in enumerate(self.dataset.upper_air_variables):
            #     if self.params.lev == 'lev' and (var == 'zg' or var == 'geopotential'):
            #         da = xr.DataArray(
            #             data=upper_air_prediction[sample, :, :, idx],
            #             dims=["ensemble_idx", "time", "plev", "lat", "lon"],
            #             coords = {
            #                 'ensemble_idx': ensemble_idxs,
            #                 'time': time_range,
            #                 'plev': self.dataset.plev.values,
            #                 'lat': self.params.lat,
            #                 'lon': self.params.lon
            #             }
            #         )
            #     elif self.params.lev == 'lev':
            #         da = xr.DataArray(
            #             data=upper_air_prediction[sample, :, :, idx],
            #             dims=["ensemble_idx", "time", self.params.lev, "lat", "lon"],
            #             coords = {
            #                 'ensemble_idx': ensemble_idxs,
            #                 'time': time_range,
            #                 self.params.lev: dataset.lev.values,
            #                 'lat': self.params.lat,
            #                 'lon': self.params.lon
            #             }
            #         )
            #     else:
            #         da = xr.DataArray(
            #             data=upper_air_prediction[sample, :, :, idx],
            #             dims=["ensemble_idx", "time", level_coord_name, "lat", "lon"],
            #             coords=coordinates
            #         )
            #     #da = da.assign_attrs(self.dataset.data_dss[0][var].attrs)
            #     dataset[var] = da

            datasets.append(dataset)
                

        return datasets
    
    # def save_prediction(self, datasets, particle_idxs, ensemble_start, ensemble_end):
    #     savedirs = [self.params.output_dirs[particle_idx] for particle_idx in particle_idxs]
    #     save_basenames = [self.params.save_basenames[particle_idx] for particle_idx in particle_idxs]
    #     for savedir in savedirs:
    #         os.makedirs(savedir, exist_ok = True)
    #     for i, (dataset, save_basename) in enumerate(zip(datasets, save_basenames)):
    #         print(f'Saving prediction {particle_idxs[i]} members {ensemble_start}-{ensemble_end}...')
    #         dataset = dataset.chunk({'ensemble_idx': 1, 'time': 1, self.params.lev: 1})
    #         if self.params.use_sigma_levels and ('zg' in self.params.upper_air_variables or 'geopotential' in self.params.upper_air_variables):
    #             dataset = dataset.chunk({'plev': 1})
    #         #filename = f'{self.params.nettype}_{self.params.run_num}_{self.params['timedelta_hours']}h_{self.params['inference_steps']}step_{self.params.val_start_year}_{batch_idx * self.params.batch_size + sample}.nc'
    #         filename = save_basename + f'_run.{ensemble_start:04d}-{ensemble_end:04d}_output.nc'
    #         dataset.to_netcdf(os.path.join(savedir, filename))

    def save_prediction(self, datasets, particle_idxs, ensemble_start, ensemble_end):
        savedirs = [self.params.output_dirs[particle_idx] for particle_idx in particle_idxs]
        save_basenames = [self.params.save_basenames[particle_idx] for particle_idx in particle_idxs]
        for savedir in savedirs:
            os.makedirs(savedir, exist_ok = True)
        for i, (dataset, save_basename, savedir) in enumerate(zip(datasets, save_basenames, savedirs)):
            print(f'Saving prediction {particle_idxs[i]} members {ensemble_start}-{ensemble_end}...')
            # Check if dask is imported and available before chunking
            #dask_is_available = "dask" in globals() or "dask" in locals()
            #if dask_is_available:
            #    # If dask is available, proceed to chunk as normal
            #    dataset = dataset.chunk({'ensemble_idx': 1, 'time': 1, self.params.lev: 1})
            #    if self.params.use_sigma_levels and ('zg' in self.params.upper_air_variables or 'geopotential' in self.params.upper_air_variables):
            #        dataset = dataset.chunk({'plev': 1})
            #filename = f'{self.params.nettype}_{self.params.run_num}_{self.params['timedelta_hours']}h_{self.params['inference_steps']}step_{self.params.val_start_year}_{batch_idx * self.params.batch_size + sample}.nc'
            filename = save_basename + f'_run.{ensemble_start:04d}-{ensemble_end:04d}_output.nc'
            
            # dataset.to_netcdf(os.path.join(savedir, filename))
            filepath = os.path.join(savedir, filename)
            # FIX 1: specify mode='w', compute=True ensure immediate close
            dataset.to_netcdf(filepath, mode='w', compute=True)
            # FIX 2: close every time we saved.
            dataset.close()

    def _save_ensemble_nc_files(self, particle_datasets_buf, particle_idxs):
        """Save one .nc file per particle containing all ensemble members.

        *particle_datasets_buf* maps global particle-id (int) to a merged
        xr.Dataset whose ``ensemble_idx`` dimension spans all members.
        Paths come from ``self.params.ensemble_nc_basenames`` and
        ``self.params.ensemble_nc_dirs``.
        """
        for pid in particle_idxs:
            pid_int = int(pid)
            if pid_int not in particle_datasets_buf:
                continue
            savedir = self.params.ensemble_nc_dirs[pid_int]
            os.makedirs(savedir, exist_ok=True)
            save_basename = self.params.ensemble_nc_basenames[pid_int]
            filepath = save_basename + '_all_members_output.nc'
            ds = particle_datasets_buf[pid_int]
            ds.to_netcdf(filepath, mode='w', compute=True)
            ds.close()

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
    parser.add_argument("--init_datetimes", default="", type=str)
    parser.add_argument("--init_nc_filepaths", default="", type=str)
    parser.add_argument("--save_basenames", default="", type=str)
    parser.add_argument("--output_dirs", default="", type=str)
    parser.add_argument("--region_file", default="", type=str)
    parser.add_argument("--regions", default="", type=str)
    parser.add_argument("--async_save", default = False, action='store_true')
    parser.add_argument("--use_legacy_model", default=False, action='store_true')
    parser.add_argument("--ensemble_inference_hours", default=0, type=int)
    parser.add_argument("--num_ensemble_members", default=0, type=int)
    parser.add_argument("--ensemble_members_per_pred", default=0, type=int)
    parser.add_argument("--batch_size", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int, help="Random seed used during training (for checkpoint path)")
    parser.add_argument("--seed_24h", default=None, type=int, help="Random seed for 24h model (for 6h_24h mode)")

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
    #params['epsilon_factor'] = args.epsilon_factor
    params = params_list[0]
    params['use_legacy_model'] = args.use_legacy_model
    if len(args.save_basenames) > 0:
        params['save_basenames'] = args.save_basenames.split(',')
    if len(args.output_dirs) > 0:
        params['output_dirs'] = args.output_dirs.split(',')
    if len(args.region_file) > 0:
        params['region_file'] = args.region_file
    if not hasattr(params, 'output_dirs') and hasattr(params, 'save_basenames'):
        params['output_dirs'] = [os.path.dirname(save_basename) for save_basename in params['save_basenames']]
    try:
        assert len(params['save_basenames']) == len(params['output_dirs'])
    except AssertionError:
        raise ValueError("Number of save basenames and output directories must match")

    if args.ensemble_inference_hours > 0:
        params['ensemble_inference_hours'] = args.ensemble_inference_hours
    if args.num_ensemble_members > 0:
        params['num_ensemble_members'] = args.num_ensemble_members
    if args.ensemble_members_per_pred > 0:
        params['ensemble_members_per_pred'] = args.ensemble_members_per_pred
    if args.batch_size > 0:
        params['batch_size'] = args.batch_size
    
    # Allow train_data_sets / validation_data_sets in the YAML to be a path to a .json file
    for _key in ('train_data_sets', 'validation_data_sets'):
        _val = getattr(params, _key, None)
        if isinstance(_val, str) and _val.endswith('.json'):
            with open(_val, 'r') as f:
                params[_key] = json.load(f)

    params['run_iter'] = 1
    if hasattr(params, 'diagnostic_variables'):
        if len(params.diagnostic_variables) > 0:
            params['has_diagnostic'] = True
        else:
            params['has_diagnostic'] = False
    else:
        params['has_diagnostic'] = False
    print(f'Has diagnostic: {params.has_diagnostic}')
    if not hasattr(params, 'num_ensemble_members'):
        params['num_ensemble_members'] = 1
    if len(args.init_nc_filepaths) > 0:
        params['init_nc_filepaths'] = args.init_nc_filepaths.split(',')
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
    params_list[0] = params
            
    torch.manual_seed(world_rank)
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

    # Set up directories
    # Set up directories
    # Determine seeds for each model (6h uses args.seed, 24h uses args.seed_24h or falls back to args.seed)
    if args.use_6h_24h_model:
        seeds = [args.seed, args.seed_24h if args.seed_24h is not None else args.seed]
    else:
        seeds = [args.seed]
    
    for params, run_num, seed in zip(params_list, run_nums, seeds):
        expDir = os.path.join(params.exp_dir, args.config, str(run_num))

        params['experiment_dir'] = os.path.abspath(expDir)
        # Construct checkpoint path following train.py convention.
        params['checkpoint_dir'] = os.path.join(expDir, 'checkpoints')
        params['best_checkpoint_path'] = os.path.join(params['checkpoint_dir'], 'best_ckpt.tar')
        params['latest_checkpoint_path'] = os.path.join(params['checkpoint_dir'], 'ckpt_latest.tar')
        params['checkpoint_path_globstr'] = os.path.join(params['checkpoint_dir'], 'ckpt_epoch_*.tar')

        # Check for checkpoints with priority: best > latest > numbered
        if os.path.isfile(params.best_checkpoint_path):
            checkpoint_path = params.best_checkpoint_path
            checkpoint_exists = True
        elif os.path.isfile(params.latest_checkpoint_path):
            checkpoint_path = params.latest_checkpoint_path
            checkpoint_exists = True
        else:
            checkpoint_paths = natsorted([
                f for f in glob.glob(params.checkpoint_path_globstr) if os.path.isfile(f)
            ])
            if len(checkpoint_paths) > 0:
                checkpoint_path = checkpoint_paths[-1]  # Use most recent
                checkpoint_exists = True
            else:
                checkpoint_exists = False
        
        if checkpoint_exists:
            params['best_checkpoint_path'] = checkpoint_path  # Update to actual path found
        else:
            raise FileNotFoundError(
                f"No checkpoint found.\n"
                f"Searched: {params.best_checkpoint_path}, {params.latest_checkpoint_path}, {params.checkpoint_path_globstr}"
            )

        # Determine whether to resume or start fresh
        params['resuming'] = True
    if world_rank == 0:
        logging.info("Resuming from existing checkpoint.")

    # # Do not comment this line out please:
    # # args.resuming = True if os.path.isfile(params.checkpoint_path) else False
    # args.resuming = False
    # params['resuming'] = args.resuming

    params = params_list[0]
    params['local_rank'] = local_rank
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

    if world_rank == 0:
        #log_file = f'out_{i}.log'
        log_file = 'out.log'
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(params.experiment_dir, log_file))
        logging_utils.log_versions()
        params.log()

        params['log_to_wandb'] = False
        params['log_to_screen'] = (world_rank == 0) and params['log_to_screen']
    params_list[0] = params

    if world_rank == 0:
        for params in params_list:
            hparams = ruamelDict()
            yaml = YAML()
            for key, value in params.params.items():
                hparams[str(key)] = str(value)
            with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
                yaml.dump(hparams,  hpfile)

    params = params_list[0]
    _dt_cls = datetime_class_from_calendar(params.calendar)
    if len(args.init_datetime) == 0:
        if hasattr(params, "init_datetime"):
            params['init_datetime'] = cftime.datetime.strptime(params.init_datetime, "%Y-%m-%d %H:%M:%S",
                                                                            has_year_zero = params.has_year_zero,
                                                                            calendar = params.calendar)
        else:
            params['init_datetime'] = cftime.datetime(params.val_year_start, 1, 1, 0, has_year_zero = params.has_year_zero,
                                                            calendar = params.calendar)
        params['init_datetime'] = _dt_cls(params.init_datetime.year,
                                          params.init_datetime.month,
                                          params.init_datetime.day,
                                          hour = params.init_datetime.hour,
                                          has_year_zero = params.has_year_zero)
    else:
        params['init_datetime'] = cftime.datetime.strptime(args.init_datetime, "%Y-%m-%d %H:%M:%S",
                                                                            has_year_zero = params.has_year_zero,
                                                                            calendar = params.calendar)
        params['init_datetime'] = _dt_cls(params.init_datetime.year,
                                          params.init_datetime.month,
                                          params.init_datetime.day,
                                          hour = params.init_datetime.hour,
                                          has_year_zero = params.has_year_zero)
    if len(args.init_datetimes) == 0 and len(args.init_datetime) == 0:
        if hasattr(params, "init_datetimes"):
            params['init_datetimes'] = [cftime.datetime.strptime(datetime, "%Y-%m-%d %H:%M:%S",
                                                                            has_year_zero = params.has_year_zero,
                                                                            calendar = params.calendar) for \
                                                                                datetime in params.init_datetimes]
            params['init_datetimes'] = [_dt_cls(init_datetime.year,
                                                init_datetime.month,
                                                init_datetime.day,
                                                hour = init_datetime.hour,
                                                has_year_zero = params.has_year_zero) for \
                                                init_datetime in params.init_datetimes]
    else:
        params['init_datetimes'] = [cftime.datetime.strptime(datetime, "%Y-%m-%d %H:%M:%S",
                                                                            has_year_zero = params.has_year_zero,
                                                                            calendar = params.calendar) for \
                                                                                datetime in args.init_datetimes.split(',')]
        params['init_datetimes'] = [_dt_cls(init_datetime.year,
                                            init_datetime.month,
                                            init_datetime.day,
                                            hour = init_datetime.hour,
                                            has_year_zero = params.has_year_zero) for \
                                            init_datetime in params.init_datetimes]


    # # @Amaury ADD a condition to generate the output_dirs and save_basenames if they are not provided (consistent with the total number of particles)
    # if args.output_dir:
    #    if  len(params['init_datetimes']) > 0:
    #        N_ICs = len(params['init_datetimes'])
    #    else:
    #        N_ICs = len(args.init_nc_filepaths.split(','))
    #    params['output_dirs'] = [args.output_dir] * N_ICs
    #    print(f'Output dirs: {params["output_dirs"]}')
    #    params['save_basenames'] = [args.output_dir + f'/particle_{i}' for i in range(N_ICs)]
    #    print(f'Save basenames: {params["save_basenames"]}')

    params_list[0] = params
    # === ADD THIS BLOCK ===
    # For 6h_24h mode, copy shared parameters to the 24h model params
    if args.use_6h_24h_model and len(params_list) > 1:
        shared_keys = [
            'init_datetime', 'init_datetimes', 'init_nc_filepaths',
            'save_basenames', 'output_dirs', 'region_file',
            'ensemble_inference_hours', 'num_ensemble_members',
            'ensemble_members_per_pred', 'batch_size', 'world_size',
            'has_diagnostic', 'use_legacy_model', 'run_iter'
        ]
        for key in shared_keys:
            if hasattr(params_list[0], key):
                params_list[1][key] = getattr(params_list[0], key)
    # === END OF BLOCK ===

    # if output_dirs dosen't exist, create it #Added by Amaury on 12/22/2025
    # print("DEBUG: output_dirs: ", params['output_dirs'])
    # # output_dirs = params['output_dirs'] if isinstance(params['output_dirs'], (list, tuple)) else [params['output_dirs']]
    # for dirname in params['output_dirs']:
    #     print("DEBUG: dirname: ", dirname + "/")
    #     if not os.path.isdir(dirname + "/"):
    #         os.makedirs(dirname + "/", exist_ok=True)

    stepper = Stepper(params_list, world_rank, use_6h_24h_model = args.use_6h_24h_model,
                      async_save = args.async_save)
    
    if hasattr(params, 'use_default_obs'):
        if params.use_default_obs:
            target_duration = 7
            lead_time = params.ensemble_inference_hours // 24 - target_duration
            print("DEBUG: lead_time: ", lead_time)
            print("DEBUG: target_duration: ", target_duration)
            var = "tas"
            if len(args.regions) > 0:
                regions = args.regions.split(',')
            else:
                regions = ["France", "Chicago"]
            PATH_REGIONS = params.region_file
            stepper.predict(obs_function=compute_A_ensemble, 
                            obs_args=[params.save_basenames, target_duration, lead_time, var, regions, PATH_REGIONS])
        else:
            stepper.predict()
    else:
        stepper.predict()



    logging.info('DONE ---- rank %d' % world_rank)
    if params['world_size'] > 1:
        dist.barrier()
        cleanup()