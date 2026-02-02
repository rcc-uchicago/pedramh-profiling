from networks.pangu import PanguModel_Plasim
from networks.pangu_legacy import PanguModel_Plasim as PanguModel_Plasim_Legacy
from networks.modulus_sfno.sfnonet import SphericalFourierNeuralOperatorNet_v2 as SFNO
from tqdm import tqdm
from pathlib import Path
from datetime import timedelta
from datetime import datetime
from ruamel.yaml.comments import CommentedMap as ruamelDict
from ruamel.yaml import YAML
from collections import OrderedDict
import matplotlib.pyplot as plt
import wandb
from utils.data_loader_multifiles import get_data_loader, get_date_range, datetime_class_from_calendar, create_dataloader
from utils.YParams import YParams
import os, glob
import time
from natsort import natsorted
import numpy as np
import argparse
import xarray as xr
import logging
import torch
import torchvision
from torchvision.utils import save_image
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.profiler import profile, record_function, ProfilerActivity
from utils import logging_utils
from utils.power_spectrum import *
from utils.perturbation import Perturber
##########################################
## NEW IMPORTS
from utils.losses import Latitude_weighted_MSELoss, Latitude_weighted_L1Loss, Masked_L1Loss,\
    Masked_MSELoss, Latitude_weighted_masked_L1Loss, Latitude_weighted_masked_MSELoss,\
    Latitude_weighted_CRPSLoss, Kl_divergence_gaussians
from utils.lr_scheduler_sfno import LinearWarmupCosineAnnealingLR
###############################@###########
logging_utils.config_logger()
#from apex import optimizers
from pathlib import Path
from datetime import timedelta
# import transformer_engine.pytorch as te
# from transformer_engine.common import recipe
# from transformer_engine.pytorch import fp8_autocast
from torch.profiler import profile, record_function, ProfilerActivity
from itertools import product
import time 
from multiprocessing import Process
import psutil
import shutil
from datetime import datetime
import uuid
from utils.integrate import Integrator, forward_euler
import copy
import json
from ensemble_inference import Stepper
import cftime
import warnings
import pickle

#dask.config.set(scheduler='synchronous')
torch._dynamo.config.optimize_ddp = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
torch.cuda.empty_cache()           
logging.info("Torch version: {}".format(torch.__version__))
# is_ddp = False

# if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
#     if int(os.environ["WORLD_SIZE"]) > 1:
#         torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
#         dist.init_process_group(backend="nccl", init_method="env://")
#         is_ddp = True

# if is_ddp and dist.get_rank() == 0:
#     print("DDP initialized")
#     world_rank = dist.get_rank()
#     print(f"World rank: {world_rank}")

# if not is_ddp:
#     print("Single-process mode")

#@torch.jit.script
def latitude_weighting_factor_torch(latitudes):
    lat_weights_unweighted = torch.cos(3.1416/180. * latitudes)
    return latitudes.size()[0] * lat_weights_unweighted/torch.sum(lat_weights_unweighted)

#@torch.jit.script
def weighted_rmse_torch_channels(pred, target, latitudes):
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    #lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    #s = torch.sum(torch.cos(3.1416/180. * latitudes))
    weight = torch.reshape(latitude_weighting_factor_torch(latitudes), (1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1,-2)))
    return result

#@torch.jit.script
def weighted_rmse_torch_3D(pred, target, latitudes):
    num_lat = pred.shape[3]
    weight = torch.reshape(latitude_weighting_factor_torch(latitudes), (1, 1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1,-2)))
    return result

def grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def grad_max(model):
    max_grad = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_max = torch.max(torch.abs(p.grad.detach().data))
        if max_grad < param_max.item():
            max_grad = param_max.item()
    return param_max


def evaluate_iterative_forecast(da_fc, da_true, func, clim, mean_dims=['lat', 'lon', 'time'], weighted=True):
    scores = []
    for f in da_fc.lead_time:
        fc = da_fc.sel(lead_time=f)
        true = da_true.sel(lead_time=f)
        score = func(fc, true, clim, mean_dims=mean_dims, weighted=weighted)
        scores.append(score)
    return xr.concat(scores, dim='lead_time')

def get_climatology_index(date):
    """
    Maps a date to the correct index in a 366-day climatology,
    adjusting for leap years vs. non-leap years.
    
    Args:
        date: A cftime.datetime or datetime object
        
    Returns:
        int: The index in the climatology (0-based)
    """
    # Get day of year (1-based)
    if hasattr(date, 'dayofyr'):
        day_of_year = date.dayofyr
    elif hasattr(date, 'dayofyear'):
        day_of_year = date.dayofyear
    else:
        day_of_year = date.timetuple().tm_yday
    
    # Check if it's a leap year
    if hasattr(date, 'is_leap_year'):
        is_leap_year = date.is_leap_year
    elif hasattr(date, 'calendar'):
        if date.calendar in ['standard', 'gregorian', 'proleptic_gregorian']:
            year = date.year
            is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        else:
            is_leap_year = False
    else:
        year = date.year
        is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    
    # For non-leap years, if date is after Feb 28, adjust the index
    if not is_leap_year and day_of_year > 59:  # Feb 28 is the 59th day
        clim_index = day_of_year  # Skip leap day (Feb 29)
    else:
        clim_index = day_of_year - 1  # Convert to 0-based index
    
    return clim_index

# # ACC Scores
def compute_weighted_acc(da_fc, da_true, clim=None, weighted=True, mean_dims=xr.ALL_DIMS, **kwargs):
    """
    Compute Anomaly Correlation Coefficient with proper handling of leap years and
    support for both pressure levels and sigma levels.
    
    Args:
        da_fc: xarray Dataset, forecast data
        da_true: xarray Dataset, ground truth data
        clim: xarray Dataset, climatology data
        weighted: bool, whether to use weighted averaging
        mean_dims: list, dimensions to average over
        
    Returns:
        xarray DataArray containing ACC values
    """
    # Assign dayofyear coordinate
    da_fc = da_fc.assign_coords(dayofyear=da_fc['time'].dt.dayofyear)
    da_true = da_true.assign_coords(dayofyear=da_true['time'].dt.dayofyear)
    
    # Create a DataArray with the same lead_time dimension for proper error handling
    empty_result = xr.ones_like(da_fc.isel(time=0, lat=[0], lon=[0]))
    empty_result = empty_result.mean(['lat', 'lon']) * np.nan  # Creates proper dimensions
    
    if clim is not None:
        try:
            # Rename dayofyear to time in climatology if needed
            if 'dayofyear' in clim.dims:
                clim = clim.rename({'dayofyear':'time'})
            
            # Remove 'zsfc' from climatology if it exists
            if 'zsfc' in clim:
                clim = clim.drop_vars('zsfc')
                
            # Add missing variables with zero values
            #if 'pr_12h' in da_fc and 'pr_12h' not in clim:
            #    clim['pr_12h'] = clim['tas'].copy()
            #    clim['pr_12h'][:] = 0.
            #if 'pr_6h' in da_fc and 'pr_6h' not in clim:
            #    clim['pr_6h'] = clim['tas'].copy()
            #    clim['pr_6h'][:] = 0.
            #if 'mrso' in da_fc and 'mrso' not in clim:
            #    clim['mrso'] = clim['tas'].copy()
            #    clim['mrso'][:] = 0.
            
            # Reorder variables in climatology to match forecast data
            clim = clim[list(da_fc.data_vars)]
            
            # Map each time to the correct climatology index
            # Don't use .item() which doesn't work with cftime objects
            mapped_indices = []
            for time in da_fc['time'].values:
                clim_index = get_climatology_index(time)
                mapped_indices.append(clim_index)
            
            # Select climatology using the mapped indices
            climatology_aligned = clim.isel(time=mapped_indices)
            climatology_aligned['time'] = da_fc['time']
            
            # Handle both coordinate systems
            if "plev" in da_fc.coords and "plev" in climatology_aligned.coords:
                climatology_aligned = climatology_aligned.sel(plev=da_fc.plev)
            if "lev" in da_fc.coords and "lev" in climatology_aligned.coords:
                climatology_aligned = climatology_aligned.sel(lev=da_fc.lev)
            
            # Ensure climatology has the same dimensions
            climatology_aligned = climatology_aligned.transpose(*da_fc.dims)
            
            # Match latitude coordinates
            climatology_aligned = climatology_aligned.assign_coords(lat=da_fc.lat)
            
            # Calculate anomalies
            fa = da_fc - climatology_aligned
            a = da_true - climatology_aligned
            
        except Exception as e:
            print(f"Error during climatology processing: {str(e)}")
            return empty_result  # Return properly dimensioned NaN array
    else:
        fa = da_fc
        a = da_true
    
    # Remove temporary dayofyear coordinate
    fa = fa.drop_vars('dayofyear', errors='ignore')
    a = a.drop_vars('dayofyear', errors='ignore')
    
    # Calculate latitude weights
    if weighted:
        weights_lat = np.cos(np.deg2rad(a.lat))
        weights_lat /= weights_lat.mean()
    else:
        weights_lat = 1.
    w = weights_lat
    
    # Calculate anomalies from mean
    fa_prime = fa - fa.mean()
    a_prime = a - a.mean()
    
    # Calculate ACC
    numerator = (w * fa_prime * a_prime).sum(mean_dims)
    denominator = np.sqrt((w * fa_prime ** 2).sum(mean_dims) * (w * a_prime ** 2).sum(mean_dims))
    acc = numerator / denominator
    
    return acc

def to_ensemble_batch(data, ens_members):
    """Convert batch of M samples (M, ...) to a batch of (M*ens_members, ...)."""
    return (data.unsqueeze(1) * torch.ones(1, ens_members, *data.shape[1:]).to(data.device)).flatten(0, 1)


class Trainer():
    def __init__(self, params, world_rank):
        self.params = params
        self.world_rank = world_rank
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        ###Setup the initia epochs and iteration #######
        self.iters = 0
        self.wandb_step = 0  # Track wandb step to ensure monotonicity
        if world_rank == 0:
            logging.info(f"Initialized wandb_step: {self.wandb_step}")
        self.startEpoch = 0
        self.early_stop_epoch = params['early_stop_epoch'] - 1 if 'early_stop_epoch' in params else None
        #################################################
        self.run_uuid = str(uuid.uuid4())
        self.check_land_ocean_variables()
        # get dataset
        self.get_dataset()
        # Create output directories
        self.spectra_dir, self.diagnostics_dir, self.output_dir, self.bias_dir = self.create_dirs(self.run_uuid)
        # Initial wandb
        self.init_wandb(self.params)
        logging.info('Params' % params)
        
    def setup_model(self):
        # Set up model
        self.mask_bool, self.land_mask = self.get_land_mask_bool() #Bing: need to double check if the return is static values.
        self.model = self.get_model()
        
        if self.params.enable_amp == True:
            self.scaler = GradScaler()
        if self.params.resuming or self.params.finetuning:
            checkpoint_path = None
            finetune = self.params.finetuning
            
            # Priority 1: If start_epoch is assigned, load from that epoch's checkpoint file
            if hasattr(self.params, 'start_epoch'):
                epoch_checkpoint_pattern = os.path.join(
                    self.params.checkpoint_dir_load,
                    f'ckpt_epoch_{self.params.start_epoch}.tar'
                )
                if os.path.isfile(epoch_checkpoint_pattern):
                    checkpoint_path = epoch_checkpoint_pattern
                    if self.world_rank == 0:
                        logging.info(f"Loading checkpoint from specified epoch {self.params.start_epoch}: {checkpoint_path}")
            
            # Priority 2: If just_validate and no epochs are set, load from best_checkpoint_path_load
            if checkpoint_path is None:
                if hasattr(self.params, 'just_validate') and self.params.just_validate:
                    validation_epochs = getattr(self.params, 'validation_epochs', '')
                    if validation_epochs == '' or (isinstance(validation_epochs, (list, str)) and len(validation_epochs) == 0):
                        if os.path.isfile(self.params.best_checkpoint_path_load):
                            checkpoint_path = self.params.best_checkpoint_path_load
                            if self.world_rank == 0:
                                logging.info(f"Validation mode: using best checkpoint: {checkpoint_path}")
            
            # Priority 3: Load from latest_checkpoint_path_load
            if checkpoint_path is None:
                if hasattr(self.params, 'latest_checkpoint_path_load') and os.path.isfile(self.params.latest_checkpoint_path_load):
                    checkpoint_path = self.params.latest_checkpoint_path_load
                    if self.world_rank == 0:
                        logging.info(f"Loading latest checkpoint: {checkpoint_path}")
            
            # Priority 4: Use checkpoint_path_globstr_load to search for the latest epoch checkpoint path
            if checkpoint_path is None:
                checkpoint_paths = natsorted([
                    file for file in glob.glob(self.params.checkpoint_path_globstr_load) 
                    if os.path.isfile(file)
                ])
                if len(checkpoint_paths) > 0:
                    checkpoint_path = checkpoint_paths[-1]
                    if self.world_rank == 0:
                        logging.info(f"Loading latest epoch checkpoint: {checkpoint_path}")
            
            # Priority 5: If no epoch checkpoint paths found, use best_checkpoint_path_load
            if checkpoint_path is None:
                if os.path.isfile(self.params.best_checkpoint_path_load):
                    checkpoint_path = self.params.best_checkpoint_path_load
                    if self.world_rank == 0:
                        logging.info(f"Loading best checkpoint: {checkpoint_path}")
            
            # Raise error if no checkpoint found
            if checkpoint_path is None:
                raise FileNotFoundError(
                    f"No checkpoint files found for resuming.\n"
                    f"Searched: start_epoch={getattr(self.params, 'start_epoch', None)}, "
                    f"best_checkpoint_path_load={self.params.best_checkpoint_path_load}, "
                    f"latest_checkpoint_path_load={getattr(self.params, 'latest_checkpoint_path_load', 'N/A')}, "
                    f"checkpoint_path_globstr_load={self.params.checkpoint_path_globstr_load}"
                )
            
            self.restore_checkpoint(checkpoint_path, finetune=finetune)
            if finetune:
                logging.info("Finetuning: Loaded checkpoint from %s (optimizer state not loaded)", checkpoint_path)
            else:
                logging.info("Resuming from checkpoint: %s", checkpoint_path)
            if self.params.debug:
                self.params.max_epochs = self.startEpoch + 1
        else:
            logging.info("Starting fresh training run")
            if self.params.debug:
                self.params.max_epochs = self.startEpoch + 1
        
        if dist.is_initialized():
            self.model = DistributedDataParallel(self.model,
                                                 device_ids=[params.local_rank],
                                                 output_device=[params.local_rank], 
                                                 find_unused_parameters=True)
        #Logging
        if self.params.log_to_wandb:
            wandb.watch(self.model)
        '''if params.log_to_screen:
        logging.info(self.model)'''
        if params.log_to_screen:
            logging.info("Number of trainable model parameters: {}".format(self.count_parameters()))
        
        self.epoch = self.startEpoch
        self.optimizer = self.get_optimizer()
        self.setup_scheduler()
        self.loss_obj_pl,self.loss_obj_sfc, self.loss_obj_diagnostic = self.setup_loss_fun()


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
            self.mask_output = params.mask_output
            
        self.long_validation = False
        if hasattr(params, 'long_validation'):
            self.long_validation = params.long_validation
            
        if self.long_validation:
            if not hasattr(params, 'bias_data_dir'):
                params['bias_data_dir'] = os.path.join(os.path.dirname(params.data_dir), 'bias')
                



    def create_dirs(self, run_uuid: int) -> tuple[str, str, str, str]:
        """
        Returns the output directories for plots (already created in main block).
        Kept for backward compatibility - directories are now created in main.
        """
        spectra_dir = self.params.spectra_dir
        diagnostics_dir = self.params.gif_dir
        output_dir = self.params.acc_dir
        bias_dir = self.params.bias_dir if self.params.long_validation else None
        
        if self.world_rank == 0:
            logging.info(f"Output directories under: {self.params.experiment_dir}")
            logging.info(f"  Spectra: {spectra_dir}")
            logging.info(f"  GIFs: {diagnostics_dir}")
            logging.info(f"  ACC plots: {output_dir}")
            if bias_dir:
                logging.info(f"  Bias plots: {bias_dir}")
        
        return spectra_dir, diagnostics_dir, output_dir, bias_dir
             
    # @log_memory_usage(rank=world_rank)
    # @log_gpu_memory
    def get_dataset(self):
        """
        setup data loader
        """
        logging.info('rank %d, begin data loader init' % self.world_rank)
        if self.world_rank == 0:
            self.generator = np.random.default_rng(seed = 0)
        if hasattr(self.params, 'train_data_sets') and self.params.curriculum_learning:
            self.data_sizes = get_date_range([], self.params.train_data_sets, self.params.data_timedelta_hours,
                                            self.params.calendar, self.params.has_year_zero, 
                                            datetime_class_from_calendar(self.params.calendar), get_size = True)
                
        if self.params.train_year_to_year:
            self.train_data_loaders = []
            self.train_datasets = []
            self.train_samplers = []
            for year_start in range(params.train_year_start, params.train_year_end):
                year_end = year_start + 1
                train_data_loader, train_dataset, train_sampler = get_data_loader(
                    params, 
                    params.data_dir, 
                    dist.is_initialized(), 
                    year_start=year_start, 
                    year_end=year_end, 
                    train=True,
                    drop_last=False if self.params.curriculum_learning else True
                )
                self.train_data_loaders.append(train_data_loader)
                self.train_datasets.append(train_dataset)
                self.train_samplers.append(train_sampler)
        else:
            train_data_loader, train_dataset, train_sampler = get_data_loader(
                    params, 
                    params.data_dir, 
                    dist.is_initialized(), 
                    year_start=params.train_year_start, 
                    year_end=params.train_year_end, 
                    train=True, 
                    drop_last=False if self.params.curriculum_learning else True
                )
            self.train_data_loaders = [train_data_loader]
            self.train_datasets = [train_dataset]
            self.train_samplers = [train_sampler]

                                                                        
        self.valid_data_loader, self.valid_dataset = get_data_loader(params, params.data_dir, dist.is_initialized(), 
                                                                     year_start=params.val_year_start, 
                                                                     year_end=params.val_year_end, train=False,
                                                                     num_inferences = params.num_inferences,
                                                                     validate = True)
        
        if self.long_validation:
            self.long_valid_data_loader, self.long_valid_dataset = get_data_loader(params, params.data_dir, dist.is_initialized(), 
                                                                                year_start=params.long_val_year_start,
                                                                                year_end=params.long_val_year_start + params.long_rollout_years,
                                                                                train=False, single_ic=True, ensemble=False)
            self.epochs_per_long_validation = 1
            if hasattr(params, "epochs_per_long_validation"):
                self.epochs_per_long_validation = params.epochs_per_long_validation
            self.long_validation_spinup_years = 1
            if hasattr(params, "long_validation_spinup_years"):
                self.long_validation_spinup_years = params.long_validation_spinup_years
            
            if len(self.params.diagnostic_variables) > 0:
                self.clim_surface_bias, self.clim_upper_air_bias, self.clim_diagnostic_bias = self.long_valid_dataset._load_bias()
            else:
                self.clim_surface_bias, self.clim_upper_air_bias = self.long_valid_dataset._load_bias()
            self.climatology_bias = self.convert_to_xarray(self.clim_surface_bias.unsqueeze(0).unsqueeze(0).numpy(),
                                                        self.clim_upper_air_bias.unsqueeze(0).unsqueeze(0).numpy(),
                                                        [self.long_valid_dataset.start_date], self.params, self.long_valid_dataset, acc = True,
                                                        diagnostic_prediction = None if not self.params.has_diagnostic else self.clim_diagnostic_bias.unsqueeze(0).unsqueeze(0).numpy())[0].squeeze()
            
            perturb_params = copy.deepcopy(params)
            perturb_params['epsilon_factor'] = 0.1
            perturb_params['perturbation_type'] = 'gaussian_noise'
            
            self.perturber = Perturber(perturb_params, self.valid_dataset, device = self.device,
                                    device_idx = self.world_rank, seed = self.params.run_iter*self.params.world_size)
        
        #print('Inference Idxs:')
        #print(self.valid_dataset.inference_idxs)

        # self.constant_boundary_data = self.train_dataset.constant_boundary_data.unsqueeze(0) * torch.ones(params.batch_size, 1, 1, 1)
        # self.constant_boundary_data = self.constant_boundary_data.to(self.device, non_blocking=True)
        self.constant_boundary_data = self.train_datasets[0].constant_boundary_data.unsqueeze(0) * torch.ones(params.batch_size, 1, 1, 1)
        self.constant_boundary_data = self.constant_boundary_data.to(self.device, non_blocking=True)
        if params.num_ensemble_members > 1:
            self.constant_boundary_data = to_ensemble_batch(self.constant_boundary_data, params.num_ensemble_members)
            logging.info('Ensemble Mode. Ensemble size = {params.num_ensemble_members}\n')

         # Load climatology
        climatology_path = os.path.join(params.data_dir, self.params.climatology_file)
        self.climatology = xr.open_dataset(climatology_path)
        if 'time_bnds' in self.climatology.variables:
            self.climatology = self.climatology.drop_vars('time_bnds')
        self.climatology = self.climatology.astype({var: np.float32 for var in self.climatology.data_vars})
        self.climatology = self.climatology.rename({'time':'dayofyear'})
        if world_rank == 0:
            logging.info('rank %d, data loader initialized' % self.world_rank)


    def init_wandb(self, params:dict):    
        """
        Initialise wandb, setup metrics to log
        """
        if params.log_to_wandb:
            if params.resuming:
                resume = "allow"
            else:
                resume = "never"
            wandb.init(config=params, name=f'{params.name}-{params.run_iter}', 
                entity=params.entity, group=params.group, 
                project=params.project, resume=resume)
            logging.info("WandB initialized with config: %s", params)

            wandb.init(config=params, entity=params.entity, name=f'{params.name}-{params.run_iter}',
                        group=params.group, project=params.project, resume=resume)#, entity=params.entity)

            #wandb.define_metric("custom_step")
            #wandb.define_metric("power_spectrum_plot", step_metric="custom_step")


        


            # entity=params.entity)
            wandb.define_metric("epoch")
            wandb.define_metric("ACC_plot", step_metric="epoch")
            wandb.define_metric("power_spectrum_plot", step_metric="epoch")
            if self.params.diagnostic_logs:
                epoch_metrics = ['lr', 'train_loss', 'valid_loss', 'valid_loss_sfc', 'valid_loss_upper_air', 'valid_mean_norm_lwrmse']
                for l, steps in enumerate(self.params.forecast_lead_times):
                    epoch_metrics.append(f"valid_lwrmse_sfc_{steps}step")
                    epoch_metrics.append(f"valid_lwrmse_pl_{steps}step")
                    epoch_metrics.append(f"valid_loss_{steps}step")
                    for j, var in enumerate(self.valid_dataset.surface_variables):
                        epoch_metrics.append(f'valid_{var}_{steps}step_lwrmse')
                    for j, var in enumerate(self.valid_dataset.upper_air_variables):
                        if var != 'zg' and var != 'geopotential_height' and self.valid_dataset.use_sigma_levels:
                            for k, level in enumerate(self.valid_dataset.sigma_levels):
                                epoch_metrics.append(f'valid_{var}_level{level:.3f}_{steps}step_lwrmse')
                        else:
                            for k, level in enumerate(self.valid_dataset.levels):
                                epoch_metrics.append(f'valid_{var}_level{level:.3f}_{steps}step_lwrmse')
                if self.long_validation:
                    wandb.define_metric("bias_plot", step_metric="epoch")
                    for j, var in enumerate(self.valid_dataset.surface_variables):
                        epoch_metrics.append(f'valid_{var}_bias_lwrmse')
                    for j, var in enumerate(self.valid_dataset.upper_air_variables):
                        if var != 'zg' and var != 'geopotential_height' and self.valid_dataset.use_sigma_levels:
                            for k, level in enumerate(self.valid_dataset.sigma_levels):
                                epoch_metrics.append(f'valid_{var}_level{level:.3f}_bias_lwrmse')
                        else:
                            for k, level in enumerate(self.valid_dataset.levels):
                                epoch_metrics.append(f'valid_{var}_level{level:.3f}_bias_lwrmse')
                    
            else:
                epoch_metrics = ['lr', 'train_loss', 'valid_loss', 'valid_loss_sfc', 'valid_loss_upper_air']
            # Add this line to ensure power_spectrum_plot is always defined as a metric
            for metric in epoch_metrics:
                wandb.define_metric(metric, step_metric="epoch")


    def get_land_mask_bool(self) -> torch.Tensor:
        """
        Get a boolean mask for the land or ocean based on the variable name.
        """
        mask_bool = []
        land_mask = []
        if self.params.nettype == 'pangu_plasim' or self.params.nettype == 'sfno_plasim':
            if (self.has_land or self.has_ocean) and self.mask_output:
                land_mask = torch.clone(self.train_datasets[0].land_mask.detach()).to(self.device)
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
        if self.params.nettype == 'pangu_plasim':
            if self.params.use_legacy_model:
                model_class = PanguModel_Plasim_Legacy
            else:
                model_class = PanguModel_Plasim
            if self.params.predict_delta:
                self.model = model_class(params, land_mask = self.land_mask).to(self.device)
                self.integrator = Integrator(params, surface_ff_std=self.train_datasets[0].surface_std.detach().to(self.device),
                                               surface_delta_std=self.train_datasets[0].surface_delta_std.detach().to(self.device),
                                               upper_air_ff_std=self.train_datasets[0].upper_air_std.detach().to(self.device),
                                               upper_air_delta_std=self.train_datasets[0].upper_air_delta_std.detach().to(self.device)).to(self.device)
            else:
                if hasattr(params, 'mask_fill'):
                    self.model = model_class(params, land_mask = self.land_mask, 
                                               mask_fill = params.mask_fill).to(self.device)
                else:
                    self.model = model_class(params, land_mask = self.land_mask, 
                                                mask_fill = self.train_datasets[0].mask_fill).to(self.device)
            # self.model = torch.compile(self.model, mode = 'default')
        elif params.nettype == 'sfno_plasim':
            print(f'\n\nRunning SFNO model\n\n')
            self.model = SFNO(params, self.train_datasets[0]).to(self.device)
            if params.sync_norm:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            if self.params.predict_delta:
                self.integrator = Integrator(params, surface_ff_std=self.train_datasets[0].surface_std.detach().to(self.device),
                                               surface_delta_std=self.train_datasets[0].surface_delta_std.detach().to(self.device),
                                               upper_air_ff_std=self.train_datasets[0].upper_air_std.detach().to(self.device),
                                               upper_air_delta_std=self.train_datasets[0].upper_air_delta_std.detach().to(self.device)).to(self.device)
        else:
            raise Exception("not implemented")

        return self.model
        

    def count_parameters(self):
        """
        Count the trainable parameters
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


    def get_optimizer(self):
        if hasattr(self.params, 'loglr'):
            self.lr = (2 ** self.params.loglr) * self.params["global_batch_size"] / 16.0
        else:
            self.lr = self.params.lr
        if self.params.optimizer_type == 'FusedAdam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.params.weight_decay, fused=True)
        elif self.params.optimizer_type == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.params.weight_decay, fused=True)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.params.weight_decay)

        return self.optimizer

    def setup_scheduler(self):

        if self.params.scheduler == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=5, mode='min')
        elif self.params.scheduler == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.params.max_epochs, 
                                                                        last_epoch=self.startEpoch-1)
        elif params.scheduler == 'LinearWarmupCosineAnnealingLR':
            steps_per_epoch = sum(len(loader) for loader in self.train_data_loaders)
            self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                           warmup_epochs=self.params.num_warmup_epochs*steps_per_epoch,
                                                           max_epochs=self.params.max_epochs*steps_per_epoch,
                                                           warmup_start_lr=self.params.warmup_start_lr,
                                                           eta_min = self.params.eta_min,
                                                           last_epoch = -1 if self.startEpoch < 1 else (self.startEpoch-1) * steps_per_epoch)
        elif params.scheduler == 'OneCycleLR':
            # total_steps = len(self.train_data_loader) * params.max_epochs
            steps_per_epoch = sum(len(loader) for loader in self.train_data_loaders)
            total_steps = steps_per_epoch * self.params.max_epochs
            if hasattr(self.params, 'oc_pct_start'):
                pct_start = self.params.oc_pct_start
            else:
                pct_start = 0.3
            if hasattr(self.params, 'oc_div_factor'):
                div_factor = self.params.oc_div_factor
            else:
                div_factor = 25
            if hasattr(self.params, 'oc_final_div_factor'):
                final_div_factor = self.params.oc_final_div_factor
            else:
                final_div_factor = 1e4
            
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=self.lr,
                    total_steps=total_steps,
                    steps_per_epoch=steps_per_epoch,
                    pct_start = pct_start,
                    div_factor = div_factor,
                    final_div_factor=final_div_factor,
                    last_epoch = -1 if self.startEpoch < 1 else (self.startEpoch-1) * steps_per_epoch
                )
            logging.info("Scheduler is setup")
        else:
            self.scheduler = None

        

    def setup_loss_fun(self):
        """
        Set up loss function to return the loss for pl, sfc and diagnoistic
        """
        #initialisation
        self.loss_obj_pl = 0 
        self.loss_obj_sfc = 0
        self.loss_obj_diagnostic = 0
        self.loss_vae = 0
        if self.params.vae_loss:
            self.loss_vae = Kl_divergence_gaussians()
            logging.info("VAE loss is setup")
        if self.params.loss == 'l1' or self.params.loss == 'raw_l1':
            self.loss_obj_pl = torch.nn.L1Loss()
            if (self.has_land or self.has_ocean) and self.mask_output:
                self.loss_obj_sfc = Masked_L1Loss(self.mask_bool)
            else:
                self.loss_obj_sfc = torch.nn.L1Loss()
            if self.params.has_diagnostic:
                self.loss_obj_diagnostic = torch.nn.L1Loss()
        elif params.loss == 'l2' or params.loss == 'raw_l2':
            self.loss_obj_pl = torch.nn.MSELoss()
            if (self.has_land or self.has_ocean) and self.mask_output:
                self.loss_obj_sfc = Masked_MSELoss(self.mask_bool)
            else:
                self.loss_obj_sfc = torch.nn.MSELoss()
            if self.params.has_diagnostic:
                self.loss_obj_diagnostic = torch.nn.MSELoss()
        elif self.params.loss == 'weightedl1':
            self.lat = torch.from_numpy(np.array(self.params.lat)).to(self.device)
            # self.lat = self.train_dataset.lat.to(self.device, non_blocking=True)
            self.loss_obj_pl = Latitude_weighted_L1Loss(self.lat)
            if (self.has_land or self.has_ocean) and self.mask_output:
                self.loss_obj_sfc = Latitude_weighted_masked_L1Loss(self.lat, self.mask_bool)
            else:
                self.loss_obj_sfc = Latitude_weighted_L1Loss(self.lat)
            if self.params.has_diagnostic:
                self.loss_obj_diagnostic = Latitude_weighted_L1Loss(self.lat)
        elif self.params.loss == 'weightedl2':
            self.lat = torch.from_numpy(np.array(self.params.lat)).to(self.device)
            self.loss_obj_pl = Latitude_weighted_MSELoss(self.lat)
            if (self.has_land or self.has_ocean) and self.mask_output:
                self.loss_obj_sfc = Latitude_weighted_masked_MSELoss(self.lat, self.mask_bool)
            else:
                self.loss_obj_sfc = Latitude_weighted_MSELoss(self.lat)
            if self.params.has_diagnostic:
                self.loss_obj_diagnostic = Latitude_weighted_MSELoss(self.lat)
        elif self.params.loss == 'weightedCRPS':
            self.lat = torch.from_numpy(np.array(self.params.lat)).to(self.device)
            self.loss_obj_pl = Latitude_weighted_CRPSLoss(self.lat, params.num_ensemble_members)
            if self.has_land or self.has_ocean:
                self.loss_obj_sfc = Latitude_weighted_CRPSLoss(self.lat, params.num_ensemble_members,
                                                               self.mask_bool)
            else:
                self.loss_obj_sfc = Latitude_weighted_CRPSLoss(self.lat, params.num_ensemble_members)
            if self.params.has_diagnostic:
                self.loss_obj_diagnostic = Latitude_weighted_CRPSLoss(self.lat, params.num_ensemble_members)
        else:
            raise NotImplementedError
        logging.info("Losses is setup")
        return self.loss_obj_pl, self.loss_obj_sfc, self.loss_obj_diagnostic


    def train(self):
        if self.params.log_to_screen:
            logging.info("Starting Training Loop...")
        best_valid_loss = 1.e6
        early_stopping_counter = 0
        early_stop_epoch_triggered = False

        for epoch in range(self.startEpoch, self.params.max_epochs):
            if world_rank == 0:
                logging.info(f'Starting epoch {epoch + 1}/{self.params.max_epochs}')

            if self.early_stop_epoch is not None and epoch > self.early_stop_epoch:
                if self.params.log_to_screen:
                    logging.info(f'Completed early stop epoch {self.early_stop_epoch}. Terminating training.')
                early_stop_epoch_triggered = True
                break
            
            if dist.is_initialized():
                for sampler in self.train_samplers:
                    sampler.set_epoch(epoch)

            if self.params.curriculum_learning:
                # Generate shuffled indices on rank 0
                if self.world_rank == 0:
                    shuffled_date_idxs = torch.tensor(self.generator.permutation(self.data_sizes[0]), device=self.device)
                
                # Broadcast to all ranks if using distributed training
                if dist.is_initialized():
                    dist.barrier()
                    if self.world_rank != 0:
                        # Initialize tensor on non-root ranks before broadcast
                        shuffled_date_idxs = torch.zeros(self.data_sizes[0], dtype=torch.long, device=self.device)
                    dist.broadcast(shuffled_date_idxs, src = 0)
                
                shuffled_date_idxs = shuffled_date_idxs.tolist()
                logging.info(f"Shuffling training dates with curriculum learning fraction {self.params.curriculum_learning_fraction:.2f} for epoch {epoch + 1}")
                self.train_datasets[0]._shuffle_training_dates(shuffled_date_idxs, self.params.curriculum_learning_fraction, self.data_sizes)
                self.train_data_loaders[0], self.train_samplers[0] = \
                    create_dataloader(self.train_datasets[0], 
                                      int(params.batch_size), 
                                      params.num_data_workers, 
                                      dist.is_initialized(), 
                                      False, True, False, False)
                logging.info(f"Shuffled training data len: {len(self.train_datasets[0])}")

            start = time.time()
            tr_time, data_time, train_logs = self.train_one_epoch()
            logging.info(f"Epoch {epoch + 1} training time: {tr_time:.2f} seconds, data loading time: {data_time:.2f} seconds")
            # Run ensemble forecast validation if enabled
            ensemble_val_time = 0.0
            if hasattr(self.params, 'ensemble_validation') and self.params.ensemble_validation:
                ensemble_val_time = self.validate_ensemble_forecast()
                logging.info(f"Epoch {epoch + 1} ensemble validation time: {ensemble_val_time:.2f} seconds")
            valid_time, valid_logs = self.validate_one_epoch()
            logging.info(f"Epoch {epoch + 1} validation time: {valid_time:.2f} seconds")    
            
            torch.cuda.empty_cache()

            if self.params.scheduler == 'ReduceLROnPlateau':
                self.scheduler.step(valid_logs['valid_loss'])
            elif self.params.scheduler == 'CosineAnnealingLR':
                self.scheduler.step()
                if self.epoch >= self.params.max_epochs:
                    logging.info("Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR")
                    # exit()
                    break
            
            # Early stopping logic should be outside of world_rank check
            # Check if validation improved BEFORE updating best_valid_loss
            is_best = valid_logs['valid_loss'] <= best_valid_loss
            
            if is_best:
                best_valid_loss = valid_logs['valid_loss']
                early_stopping_counter = 0  # Reset the counter
            else:
                early_stopping_counter += 1  # Increment the counter
            
            if self.world_rank == 0:
                if self.params.save_checkpoint:
                    # checkpoint at the end of every epoch
                    self.save_checkpoint(self.params.checkpoint_path_globstr_save, self.epoch)

                    # Save best checkpoint only if validation improved
                    if is_best:
                        self.save_checkpoint(self.params.best_checkpoint_path_save)
            if self.params.log_to_wandb and self.world_rank == 0:
                self.log_wandb_epoch(epoch)
                self.log_screen_epoch(epoch, start, train_logs, valid_logs, early_stopping_counter)
            # Early stopping check
            if self.params.early_stopping and early_stopping_counter >= self.params.early_stopping_patience:
                if self.params.log_to_screen and world_rank == 0:
                    logging.info('Early stopping triggered. Terminating training.')
                break # Exit the train method
            
        # After the training loop ends
        #if self.params.log_to_wandb:
        #    if self.world_rank == 0:
        #        self.log_all_plots_to_wandb()
        if self.world_rank == 0:
            if self.params.diagnostic_acc:
                self.cleanup_acc_plots()
            if self.params.diagnostic_gif:
                self.cleanup_gifs()
            if self.params.diagnostic_spectra:
                self.cleanup_power_spectrum_plots()
            if self.long_validation:
                self.cleanup_bias()
        # If we've reached this point, we've completed all epochs
        if self.params.log_to_screen and self.world_rank == 0:
            if early_stop_epoch_triggered:
                logging.info(f'Training finished early at epoch {self.early_stop_epoch} due to early_stop_epoch setting.')
            else:
                logging.info('Completed all epochs. Training finished normally.')


    def log_wandb_epoch(self, epoch:int)->None:
        """
        Log to wandb
        """
        if self.params.log_to_wandb:
            for pg in self.optimizer.param_groups:
                lr = pg['lr']
            wandb.log({'lr': lr, 'epoch': self.epoch}, step=self.wandb_step)
    

    def log_screen_epoch(self, epoch:int, start, train_logs, valid_logs, early_stopping_counter, **kwargs) ->None:
        """
        Log to screen 
        """
        if self.params.log_to_screen:
                logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
                logging.info('Train loss: {}. Validation loss: {}. Surface Val loss: {}. Upper Air Val loss: {}'.format(
                    train_logs['train_loss'], valid_logs['valid_loss'], valid_logs['valid_loss_sfc'], valid_logs['valid_loss_upper_air']))
                # Add logging for multi-day losses
                lead_times_steps = self.params.forecast_lead_times
                multi_step_loss_str = '. '.join([f"{step}-step Val loss: {valid_logs.get(f'valid_loss_{step}step', 'N/A')}" for step in lead_times_steps])
                logging.info(f'Multi-step validation losses: {multi_step_loss_str}')
                if self.params.early_stopping:
                    logging.info(f'EarlyStopping counter: {early_stopping_counter} out of {self.params.early_stopping_patience}')

    # @log_memory_usage(rank=world_rank)
    # @log_gpu_memory
    def train_one_epoch(self)->None:
        self.epoch += 1
        tr_time = 0
        data_time = 0
        total_iterations = sum(len(loader) for loader in self.train_data_loaders)

        diagnostic_logs = {}
        loss = 0

        logging.info(f"Expected total batches: {total_iterations}")
        if not self.train_data_loaders:
            logging.warning("No training data loaders available.")
            return 0, 0, {"train_loss": 0.0}

        # What does this do?
        self.model.train()

        pbar = tqdm(total=total_iterations, bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')
        running_results = {"batch_sizes": 0, "loss": 0.0}

        for year_idx, train_data_loader in enumerate(self.train_data_loaders):
            logging.debug(f"Processing year idx {year_idx}")
            current_dataset = self.train_datasets[year_idx]
            if self.params.train_year_to_year:
                logging.debug(f"Processing year {self.params.train_year_start + year_idx}")
            else:
                logging.debug(f"Processing years {self.params.train_year_start} to {self.params.train_year_end}")
      
            for i, data in enumerate(train_data_loader):
                if self.params.mode == "test":
                    logging.info("training on batch %d of year %d" % (i, self.params.train_year_start + year_idx))
                if self.params.mode == "test" and i >= self.params.test_iterations:
                    logging.info("Test mode: only processing first 30 batches")
                    pbar.update(total_iterations - self.iters)
                    data_time += time.time() - data_start

                else:
                    self.iters += 1
                    data_start = time.time()
                    input_surface, input_upper_air, target_surface, target_upper_air, target_diagnostic, varying_boundary_data = self._prepare_inputs_batch(data)
                    data_time += time.time() - data_start
                    if self.params.mode == "test":
                        logging.info(f"Data preparation took {time.time() - data_start:.4f} seconds per iteration")

                    tr_start = time.time()
                    self.model.zero_grad()                
                    #define loss
                    output_surface, output_upper_air, output_diagnostic, loss_sfc, loss_pl, loss_diagnostic, loss_vae, loss= self.cal_loss(
                        input_surface, self.constant_boundary_data, varying_boundary_data, input_upper_air,
                        target_diagnostic, target_surface, target_upper_air
                    )
                    
                    
                    if self.params.enable_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()
                    tr_end_time = time.time()
                    if self.params.mode == "test":
                        logging.info(f"Backpropagation and optimizer step took {tr_end_time - tr_start:.4f} seconds/ iteration")
                    if self.params.scheduler in ['OneCycleLR', 'LinearWarmupCosineAnnealingLR']:
                        self.scheduler.step()

                    with torch.no_grad():

                        if self.params.predict_delta:
                            output_surface, output_upper_air = self.integrator(input_surface, input_upper_air, output_surface, output_upper_air)
                            target_surface, target_upper_air = self.integrator(input_surface, input_upper_air, target_surface, target_upper_air)

                        latitudes = torch.from_numpy(np.array(self.params.lat)).to(self.device, non_blocking=True)
                        surface_lwrmse = weighted_rmse_torch_channels(output_surface, target_surface, latitudes)
                        upper_air_lwrmse = weighted_rmse_torch_3D(output_upper_air, target_upper_air, latitudes)

                        if self.params.has_diagnostic:
                            diagnostic_lwrmse = weighted_rmse_torch_channels(output_diagnostic, target_diagnostic, latitudes)
                            mean_norm_lwrmse = torch.mean(torch.cat((surface_lwrmse, diagnostic_lwrmse, upper_air_lwrmse.reshape(output_upper_air.shape[0], -1)), dim = -1))
                        else:
                            diagnostic_lwrmse  = 0
                            mean_norm_lwrmse = torch.mean(torch.cat((surface_lwrmse, upper_air_lwrmse.reshape(output_upper_air.shape[0], -1)), dim = -1))

                        ######diagnoistic logging per iteration ###################
                        diagnostic_logs = self.diagnostic_log_per_iter(diagnostic_logs, diagnostic_lwrmse, surface_lwrmse, upper_air_lwrmse, current_dataset,
                                                                        train_batch_loss = loss, 
                                                                        train_batch_loss_sfc = loss_sfc, 
                                                                        train_batch_loss_upper_air = loss_pl,
                                                                        train_batch_loss_diagnostic =loss_diagnostic,
                                                                        # train_batch_loss_vae = loss_vae,
                                                                        train_mean_norm_lwrmse = mean_norm_lwrmse)
                    ##########################################################
                        if self.world_rank == 0:
                            #wandb.log(diagnostic_logs, step=(self.epoch-1) * total_iterations + self.iters)
                            # Use wandb_step to ensure monotonicity, then increment it
                            wandb.log(diagnostic_logs, step=self.wandb_step)
                            self.wandb_step += 1

                    torch.cuda.empty_cache()
                    tr_time += time.time() - tr_start
                
                    pbar.set_description(f"Year {self.params.train_year_start + year_idx}, Loss: {diagnostic_logs['train_batch_loss']:.4f}")

                    pbar.update(1)
        pbar.close()

        logs = self.diagnostic_log_per_epoch(diagnostic_logs, train_loss = loss, epoch = self.epoch)
        return tr_time, data_time, logs



    # @log_gpu_memory
    def _prepare_inputs_batch(self, data:torch.Tensor):
        """
        prepare input variables for each iteration from data loader.
        The return must contain input_surface, input_upper_air, target_surface, target_upper_air, target_diagnostic, varying_boundary_data
        """
        #Initilaise variables
        input_surface = 0
        input_upper_air = 0
        target_surface = 0
        target_upper_air = 0
        target_diagnostic = 0
        varying_boundary_data = 0
        if self.params.has_diagnostic:
            input_surface, input_upper_air, target_surface, target_upper_air, target_diagnostic, varying_boundary_data = map(
                lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), data)
        else:
            input_surface, input_upper_air, target_surface, target_upper_air, varying_boundary_data = map(
                lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), data)
        
        if self.params.num_ensemble_members > 1:
            if self.params.has_diagnostic:
                ensemble_batches = [to_ensemble_batch(temp_batch, params.num_ensemble_members) for temp_batch in 
                                    [input_surface, input_upper_air, target_surface, target_upper_air, 
                                    target_diagnostic, varying_boundary_data]]
                input_surface, input_upper_air, target_surface, target_upper_air, target_diagnostic, varying_boundary_data = ensemble_batches

                
            else:
                ensemble_batches = [to_ensemble_batch(temp_batch, params.num_ensemble_members) for temp_batch in 
                                    [input_surface, input_upper_air, target_surface, target_upper_air, 
                                    varying_boundary_data]]
                input_surface, input_upper_air, target_surface, target_upper_air, varying_boundary_data = ensemble_batches
        
        return input_surface, input_upper_air, target_surface, target_upper_air, target_diagnostic, varying_boundary_data


    def cal_loss(self, input_surface, constant_boundary_data, varying_boundary_data, input_upper_air, 
                 target_diagnostic, target_surface, target_upper_air, **kwargs):
        """
        Calculates the model predictions and corresponding loss values for surface, upper air, and (optionally) diagnostic outputs.
            Args:
                    input_surface (Tensor): Input data for the surface variables.
                    constant_boundary_data (Tensor): Input data for constant boundary conditions.
                    varying_boundary_data (Tensor): Input data for varying boundary conditions.
                    input_upper_air (Tensor): Input data for upper air variables.
                    target_diagnostic (Tensor): Target data for diagnostic variables.
                    target_surface (Tensor): Target data for surface variables.
                    target_upper_air (Tensor): Target data for upper air variables.
                    **kwargs: Additional keyword arguments.
            Returns:
                    Tuple[Tensor, Tensor, Tensor, Tensor]: 
                        - output_surface: Predicted surface variables.
                        - output_upper_air: Predicted upper air variables.
                        - output_diagnostic: Predicted diagnostic variables (zero if diagnostics are not used).
                        - loss: Computed total loss value.
        """
        output_surface = 0 
        output_upper_air = 0 
        output_diagnostic = 0 
        loss = 0 
        loss_diagnostic = 0
        loss_pl = 0 
        loss_sfc = 0
        loss_vae = 0
        with autocast(device_type="cuda"):
            if self.params.has_diagnostic:
                output_surface, output_upper_air, output_diagnostic, mu, sigma , mu2, sigma2 = self.model(input_surface, constant_boundary_data, 
                                                                    varying_boundary_data, input_upper_air, train = True)
                loss_diagnostic = self.loss_obj_diagnostic(output_diagnostic, target_diagnostic)
                
            else: 
                output_surface, output_upper_air, mu, sigma,  mu2, sigma2 = self.model(input_surface, constant_boundary_data, 
                                                            varying_boundary_data, input_upper_air, 
                                                            target_surface, target_upper_air, train = True)
                
            loss_sfc = self.loss_obj_sfc(output_surface, target_surface)
            loss_pl = self.loss_obj_pl(output_upper_air, target_upper_air)

            if 'raw_' in self.params.loss:
                if self.params.has_diagnostic:
                    loss = ((loss_pl * output_upper_air.shape[1] * output_upper_air.shape[2]) +\
                                loss_sfc * output_surface.shape[1] + \
                                loss_diagnostic * output_diagnostic.shape[1]) / \
                                    (output_upper_air.shape[1] * output_upper_air.shape[2] + output_surface.shape[1] + output_diagnostic.shape[1])
                else:
                    loss = ((loss_pl * output_upper_air.shape[1] * output_upper_air.shape[2]) +\
                                loss_sfc * output_surface.shape[1]) / \
                            (output_upper_air.shape[1] * output_upper_air.shape[2] + output_surface.shape[1])  
            else:
                if self.params.has_diagnostic:
                    loss = (loss_sfc + loss_diagnostic) * 0.25 + loss_pl
                else:
                    loss = (loss_sfc * 0.25) + loss_pl

            if self.params.vae_loss:    
                loss_vae = self.loss_vae(mu, sigma, mu2, sigma2)
                loss += self.params.vae_loss_weight * loss_vae


        return output_surface, output_upper_air, output_diagnostic, loss_sfc, loss_pl, loss_diagnostic, loss_vae, loss


    def diagnostic_log_per_iter(self, diagnostic_logs, diagnostic_lwrmse, surface_lwrmse, upper_air_lwrmse, current_dataset, **kwargs)->dict:
        """
        This function is used for logging the results from each iteration.
        Given the diagnostic logging input and return the update diagnostic_logs
        """

        diagnostic_logs['batch_grad_norm'] = torch.tensor([grad_norm(self.model)]).to(self.device)
        diagnostic_logs['batch_grad_max'] = torch.tensor([grad_max(self.model)]).to(self.device)
        
        for key, value in kwargs.items():
            diagnostic_logs[key] = value

        if self.params.has_diagnostic:
            for j, var in enumerate(current_dataset.diagnostic_variables):
                diagnostic_logs[f'train_{var}_lwrmse'] = torch.mean(diagnostic_lwrmse[:, j]) * current_dataset.diagnostic_std[j]
        
        for j, var in enumerate(current_dataset.surface_variables):
            diagnostic_logs[f'train_{var}_lwrmse'] = torch.mean(surface_lwrmse[:, j]) * current_dataset.surface_std[j]

        for j, var in enumerate(current_dataset.upper_air_variables):
            for k, level in enumerate(current_dataset.levels):
                diagnostic_logs[f'train_{var}_level{level:.4f}_lwrmse'] = torch.mean(upper_air_lwrmse[:, j, k]) * current_dataset.upper_air_std[j, k]

        if dist.is_initialized():
            for key in sorted(diagnostic_logs.keys()):
                if key == 'batch_grad_max':
                    grad_max_tensor = torch.zeros(dist.get_world_size(), dtype = torch.float32, device=self.device)
                    dist.all_gather_into_tensor(grad_max_tensor, diagnostic_logs[key])
                    diagnostic_logs[key] = torch.max(grad_max_tensor)
                else:
                    if type(diagnostic_logs[key]) in [int, float]:
                        diagnostic_logs[key] = torch.tensor([diagnostic_logs[key]]).to(self.device)
                    dist.all_reduce(diagnostic_logs[key].detach())
                    diagnostic_logs[key] = float(diagnostic_logs[key]/dist.get_world_size())

        return diagnostic_logs


    def diagnostic_log_per_epoch(self, diagnostic_logs, train_loss, epoch, **kwargs)->dict:
        """
        Generate logging information for each epoch.
        """
        logs = {}
        if self.params.diagnostic_logs:
            with torch.no_grad():
                diagnostic_logs['train_loss'] = train_loss
                if dist.is_initialized():
                    dist.all_reduce(torch.tensor(diagnostic_logs['train_loss']).to(self.device))
                    diagnostic_logs['train_loss'] = float(diagnostic_logs['train_loss']/dist.get_world_size())
                logs = {'train_loss': diagnostic_logs['train_loss'], 'epoch': self.epoch}
                if self.params.log_to_wandb:
                    wandb.log(logs, step=self.wandb_step)
                return diagnostic_logs
        else:
            with torch.no_grad():
                logs = {'train_loss': train_loss, 'epoch': self.epoch}
            
            if dist.is_initialized():
                for key in sorted(logs.keys()):
                    if isinstance(logs[key], (int, float)):
                        logs[key] = torch.tensor(logs[key]).to(self.device)
                    dist.all_reduce(logs[key])
                    logs[key] = float(logs[key]/dist.get_world_size())

            if self.params.log_to_wandb:
                wandb.log(logs, step=self.wandb_step)
            return logs





    def inti_valid_loss(self, lead_times_steps) -> tuple:
        """
        Initialise the validation loss variables.
        """
        if self.params.has_diagnostic:
            valid_buff = torch.zeros((5), dtype=torch.float32, device=self.device)
            valid_loss_diag = valid_buff[3].view(-1)
        else:
            valid_buff = torch.zeros((4), dtype=torch.float32, device=self.device)

        valid_loss = valid_buff[0].view(-1)
        valid_loss_sfc = valid_buff[1].view(-1)
        valid_loss_pl = valid_buff[2].view(-1)
        valid_steps = valid_buff[-1].view(-1)


        valid_surface_lwrmse = torch.zeros((len(lead_times_steps), len(self.valid_dataset.surface_variables)), dtype=torch.float32, device=self.device)
        valid_upper_air_lwrmse = torch.zeros((len(lead_times_steps), len(self.valid_dataset.upper_air_variables), len(self.valid_dataset.levels)), dtype=torch.float32, device=self.device)
        if self.params.has_diagnostic:
            valid_diagnostic_lwrmse = torch.zeros((len(lead_times_steps), len(self.valid_dataset.diagnostic_variables)), dtype=torch.float32, device=self.device)
        
        multi_step_losses = {f"valid_loss_{step}step": torch.zeros(1, dtype=torch.float32, device=self.device) for step in lead_times_steps}
        # Add RMSE storage for multiple lead times
        if self.params.has_diagnostic:
            multi_step_rmse = {f"valid_lwrmse_sfc_{step}step": torch.zeros(1, dtype=torch.float32, device=self.device) for step in lead_times_steps} |\
                {f"valid_lwrmse_pl_{step}step": torch.zeros(1, dtype=torch.float32, device=self.device) for step in lead_times_steps}|\
                {f"valid_lwrmse_diag_{step}step": torch.zeros(1, dtype=torch.float32, device=self.device) for step in lead_times_steps}
        else:
            multi_step_rmse = {f"valid_lwrmse_sfc_{step}step": torch.zeros(1, dtype=torch.float32, device=self.device) for step in lead_times_steps} |\
                {f"valid_lwrmse_pl_{step}step": torch.zeros(1, dtype=torch.float32, device=self.device) for step in lead_times_steps}
        
        return valid_loss_diag, valid_buff, valid_loss, valid_loss_sfc, valid_loss_pl, valid_steps, valid_surface_lwrmse, valid_upper_air_lwrmse, valid_diagnostic_lwrmse, multi_step_losses, multi_step_rmse
   
    # @log_memory_usage(rank=world_rank)
    # @log_gpu_memory
    def validate_one_epoch(self):
        if world_rank == 0:
            print("Validating...")
        self.model.eval()
        #n_valid_batches = 50  # do validation on first 50 images, just for LR scheduler
        # define the lead times to evaluate (in time steps)
        
        lead_times_steps = self.params.forecast_lead_times
        with torch.no_grad():
                latitudes = torch.from_numpy(np.array(self.params.lat)).to(self.device, non_blocking=True)

        # Initialize validation loss variables
        valid_loss_diag, valid_buff, valid_loss, valid_loss_sfc, valid_loss_pl, valid_steps, \
        valid_surface_lwrmse, valid_upper_air_lwrmse, valid_diagnostic_lwrmse, \
        multi_step_losses, multi_step_rmse = self.inti_valid_loss(lead_times_steps)
        

        valid_start = time.time()
        nb = len(self.valid_data_loader)

        diagnostic_logs = {}

        #sample_idx = np.random.randint(len(self.valid_data_loader))

        all_predictions = []
        all_ground_truths = []
        acc_predictions = []
        acc_ground_truths = []

        # with torch.inference_mode():

        with torch.no_grad():
            
            precision_context = fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe) if self.params.enable_fp8 else \
                autocast(enabled=self.params.enable_amp, device_type="cuda")
            
            no_nans = True
            if self.long_validation and self.epoch % self.epochs_per_long_validation == 0:
                print('Performing long validation...')
                cnt = 0
                no_nans = True
                if self.world_rank == 0:
                    pbar = tqdm(enumerate(self.long_valid_data_loader, 0), total=len(self.long_valid_data_loader), miniters=1)
                else:
                    pbar = enumerate(self.long_valid_data_loader, 0)
                plt.ion()
                for i, data in pbar:
                    if i == 0:
                        val_input_surface, val_input_upper_air, val_varying_boundary_data, year = map(lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), data)
                        val_input_surface, val_input_upper_air = self.perturber(val_input_surface, val_input_upper_air)
                    else:
                        val_varying_boundary_data, year = map(lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), data)
                    if self.params.has_diagnostic:
                        val_output_surface, val_output_upper_air, val_output_diagnostic, _, _, _, _ = self.model(
                            val_input_surface, self.constant_boundary_data[[0]], val_varying_boundary_data, val_input_upper_air)
                    else:
                        val_output_surface, val_output_upper_air, _, _ = self.model(val_input_surface, self.constant_boundary_data[[0]], 
                                                                            val_varying_boundary_data, val_input_upper_air)
                    if self.params.predict_delta:
                        val_output_surface, val_output_upper_air = self.integrator(val_input_surface, val_input_upper_air, val_output_surface,
                                                                                        val_output_upper_air)
                    val_input_surface, val_input_upper_air = val_output_surface, val_output_upper_air
                    if torch.any(torch.isnan(val_output_surface)) or torch.any(torch.isnan(val_output_upper_air)):
                        print(f'Long emulation diverged after {i} steps')
                        no_nans = False
                        break
                    if int(year.item()) >= self.params.long_val_year_start + self.long_validation_spinup_years:
                        if cnt == 0:
                            val_surface_bias, val_upper_air_bias = val_output_surface, val_output_upper_air
                            if self.params.has_diagnostic:
                                val_diagnostic_bias = val_output_diagnostic
                        else:
                            val_surface_bias = val_surface_bias * (cnt / (cnt + 1)) + val_output_surface / (cnt + 1)
                            val_upper_air_bias = val_upper_air_bias * (cnt / (cnt + 1)) + val_output_upper_air / (cnt + 1)
                            if self.params.has_diagnostic:
                                val_diagnostic_bias = val_diagnostic_bias * (cnt / (cnt + 1)) + val_output_diagnostic / (cnt + 1)
                        cnt += 1
                print(f'Completed {len(self.long_valid_data_loader)} step long validation')
                if dist.is_initialized():
                    dist.all_reduce(val_surface_bias, op = dist.ReduceOp.AVG)
                    dist.all_reduce(val_upper_air_bias, op = dist.ReduceOp.AVG)
                    if self.params.has_diagnostic:
                        dist.all_reduce(val_diagnostic_bias, op = dist.ReduceOp.AVG)
                    if torch.any(torch.isnan(val_output_surface)) or torch.any(torch.isnan(val_output_upper_air)):
                        no_nans = False
                
                if no_nans and self.world_rank == 0:
                    val_surface_bias = self.long_valid_dataset.surface_inv_transform(val_surface_bias.cpu())
                    val_surface_bias_lwrmse = weighted_rmse_torch_channels(val_surface_bias, self.clim_surface_bias.cpu(), latitudes.cpu()).squeeze(0)
                    val_upper_air_bias = self.long_valid_dataset.upper_air_inv_transform(val_upper_air_bias.cpu())
                    val_upper_air_bias_lwrmse = weighted_rmse_torch_3D(val_upper_air_bias, self.clim_upper_air_bias.cpu(), latitudes.cpu()).squeeze(0)
                    if self.params.has_diagnostic:
                        val_diagnostic_bias = self.long_valid_dataset.diagnostic_inv_transform(val_diagnostic_bias.cpu())
                        val_diagnostic_bias_lwrmse = weighted_rmse_torch_channels(val_diagnostic_bias, self.clim_diagnostic_bias.cpu(), latitudes.cpu()).squeeze(0)
                    start_times = [self.long_valid_dataset.datetime_class(self.params.long_val_year_start + self.long_validation_spinup_years,
                                                                        1, 1, has_year_zero = self.long_valid_dataset.has_year_zero) - \
                                                                            timedelta(hours=self.params.timedelta_hours)]
                    bias_datasets = self.convert_to_xarray(np.expand_dims(val_surface_bias.numpy(), axis = 1),
                                                        np.expand_dims(val_upper_air_bias.numpy(), axis = 1),
                                                        start_times, self.params, self.long_valid_dataset, acc = True,
                                                        diagnostic_prediction = None if not self.params.has_diagnostic else np.expand_dims(val_diagnostic_bias.numpy(), axis=1))
                        
                    print('Plotting Bias')
                    bias_filename = os.path.join(self.bias_dir, f"bias_epoch_{self.epoch}.png")
                    #print("Bias datasets")
                    #print(bias_datasets[0])
                    #print("Climatology bias")
                    #print(self.climatology_bias)
                    
                    self.plot_in_separate_process(bias_datasets[0].squeeze("time"), self.climatology_bias, [], bias_filename)

                    print("\nFinished Bias Plots...")
                    
                    if self.params.log_to_wandb:
                        wandb.log({
                            "bias_plot": wandb.Image(bias_filename),
                            "epoch": self.epoch
                        }, step=self.wandb_step)
                    
                
                
                                
            for i, data in tqdm(enumerate(self.valid_data_loader, 0), total=nb, bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}', miniters=1):
                if world_rank == 0:
                    print(f"Validating batch {i+1}/{nb}")
                if self.params.predict_delta:
                    if self.params.has_diagnostic:
                        val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air, val_target_diagnostic, val_target_surface_delta, val_target_upper_air_delta,\
                            val_varying_boundary_data, times = map(lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), data)
                    else:
                        val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air, val_target_surface_delta, val_target_upper_air_delta,\
                            val_varying_boundary_data, times = map(lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), data)
                else:
                    if self.params.has_diagnostic:
                        val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air, val_target_diagnostic, val_varying_boundary_data, times = map(
                            lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), data)
                    else:
                        val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air, val_varying_boundary_data, times = map(
                            lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), data)


                if self.params.num_ensemble_members > 1:
            
                    if self.params.has_diagnostic:
                        ensemble_batches = [to_ensemble_batch(temp_batch, params.num_ensemble_members) for temp_batch in 
                                        [val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air, 
                                        val_target_diagnostic, val_varying_boundary_data]]
                        val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air, \
                        val_target_diagnostic, val_varying_boundary_data = ensemble_batches
                
                    else:
                        ensemble_batches = [to_ensemble_batch(temp_batch, params.num_ensemble_members) for temp_batch in 
                                        [val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air, 
                                        val_varying_boundary_data]]
                        val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air, \
                        val_varying_boundary_data = ensemble_batches
        
       

                # get the correct start times for each sample
                start_times = []
                for i in range(times.shape[0]):  # Iterate over all samples in the batch
                    
                    start_time = self.valid_dataset.datetime_class(times[i,0].item(), times[i,1].item(), times[i,2].item(), hour=times[i,3].item())
                    start_times.append(start_time)
                
                max_lead_time = max(lead_times_steps)

                # Tensor for all time steps (ACC calculation)
                val_output_surface_acc = np.zeros((val_input_surface.shape[0], max_lead_time,
                                               val_input_surface.shape[1], val_input_surface.shape[2], val_input_surface.shape[3]),
                                              dtype=np.float32)
                val_output_upper_air_acc = np.zeros((val_input_upper_air.shape[0], max_lead_time,
                                                 val_input_upper_air.shape[1], val_input_upper_air.shape[2],
                                                 val_input_upper_air.shape[3], val_input_upper_air.shape[4]),
                                                dtype=np.float32)
                if self.params.has_diagnostic:
                    val_output_diagnostic_acc = np.zeros((val_target_diagnostic.shape[0], max_lead_time,
                                                    val_target_diagnostic.shape[2], val_target_diagnostic.shape[3],
                                                    val_target_diagnostic.shape[4]), dtype=np.float32)
                
                    val_output_diagnostic_t = np.zeros((val_target_diagnostic.shape[0], len(lead_times_steps),
                                                        val_target_diagnostic.shape[2], val_target_diagnostic.shape[3],
                                                        val_target_diagnostic.shape[4]), dtype=np.float32)


                # Tensor for specific lead times (power spectrum and GIF)
                val_output_surface_t = np.zeros((val_input_surface.shape[0], len(lead_times_steps),
                                       val_input_surface.shape[1], val_input_surface.shape[2], val_input_surface.shape[3]),
                                      dtype=np.float32)
                val_output_upper_air_t = np.zeros((val_input_upper_air.shape[0], len(lead_times_steps),
                                         val_input_upper_air.shape[1], val_input_upper_air.shape[2],
                                         val_input_upper_air.shape[3], val_input_upper_air.shape[4]),
                                        dtype=np.float32)
                with precision_context:                        
                    step_idx = 0
                    for step in range(max_lead_time):
                        if self.params.has_diagnostic:
                            val_output_surface, val_output_upper_air, val_output_diagnostic, _, _, _, _  = self.model(
                                val_input_surface, self.constant_boundary_data, val_varying_boundary_data[:, step], val_input_upper_air)
                        else:
                            val_output_surface, val_output_upper_air,  _, _, _, _ = self.model(val_input_surface, self.constant_boundary_data, 
                                                                                    val_varying_boundary_data[:, step], val_input_upper_air)
                        # Calculate losses for different lead times
                        if (step + 1) in lead_times_steps:
                            # target_index = lead_times_steps.index(step + 1)
                            target_index = step
                            if self.params.predict_delta:
                                loss_sfc = self.loss_obj_sfc(val_output_surface, val_target_surface_delta[:,target_index])
                                loss_pl = self.loss_obj_pl(val_output_upper_air, val_target_upper_air_delta[:,target_index])
                            else:
                                loss_sfc = self.loss_obj_sfc(val_output_surface, val_target_surface[:,target_index])
                                loss_pl = self.loss_obj_pl(val_output_upper_air, val_target_upper_air[:,target_index])
                            if self.params.has_diagnostic:
                                loss_diag = self.loss_obj_diagnostic(val_output_diagnostic, val_target_diagnostic[:,target_index])
                                loss = (loss_sfc + loss_diag) * 0.25 + loss_pl
                            else:
                                loss = (loss_sfc * 0.25 + loss_pl)
                            multi_step_losses[f"valid_loss_{step+1}step"] += loss

                            if step == 0:
                                valid_loss += loss
                                valid_loss_sfc += loss_sfc
                                valid_loss_pl += loss_pl
                                if self.params.has_diagnostic:
                                    valid_loss_diag += loss_diag

                        if self.params.predict_delta:
                            val_output_surface, val_output_upper_air = self.integrator(val_input_surface, val_input_upper_air, val_output_surface,
                                                                                            val_output_upper_air)
                        # Store output for ACC calculation (all time steps)
                        val_output_surface_acc[:, step] = self.valid_dataset.surface_inv_transform(val_output_surface.cpu()).numpy()
                        val_output_upper_air_acc[:, step] = self.valid_dataset.upper_air_inv_transform(val_output_upper_air.cpu()).numpy()
                        if self.params.has_diagnostic:
                            val_output_diagnostic_acc[:, step] = self.valid_dataset.diagnostic_inv_transform(val_output_diagnostic.cpu()).numpy()

                    
                        # Calculate losses for different lead times
                        if (step + 1) in lead_times_steps:
                            # Calculate RMSE
                            rmse_sfc = weighted_rmse_torch_channels(val_output_surface, val_target_surface[:,target_index], latitudes)
                            rmse_pl = weighted_rmse_torch_3D(val_output_upper_air, val_target_upper_air[:,target_index], latitudes)
                            if self.params.has_diagnostic:
                                rmse_diag = weighted_rmse_torch_channels(val_output_diagnostic, val_target_diagnostic[:,target_index], latitudes)
                                multi_step_rmse[f"valid_lwrmse_diag_{step+1}step"] += torch.mean(rmse_diag)
                                valid_diagnostic_lwrmse[step_idx] += torch.mean(rmse_diag, dim = 0)
                                val_output_diagnostic_t[:, step_idx] = self.valid_dataset.diagnostic_inv_transform(val_output_diagnostic.cpu()).numpy()

                            multi_step_rmse[f"valid_lwrmse_sfc_{step+1}step"] += torch.mean(rmse_sfc)
                            multi_step_rmse[f"valid_lwrmse_pl_{step+1}step"] += torch.mean(rmse_pl)

                            valid_surface_lwrmse[step_idx] += torch.mean(rmse_sfc, dim = 0)
                            valid_upper_air_lwrmse[step_idx] += torch.mean(rmse_pl, dim=0)

                            val_output_surface_t[:, step_idx] = self.valid_dataset.surface_inv_transform(val_output_surface.cpu()).numpy()
                            val_output_upper_air_t[:, step_idx] = self.valid_dataset.upper_air_inv_transform(val_output_upper_air.cpu()).numpy()
                            
                            if step + 1 == max_lead_time:

                                # Prepare datasets for ACC (all time steps)
                                if self.params.diagnostic_acc or self.params.diagnostic_gif:
                                    if self.params.has_diagnostic:
                                        acc_datasets = self.convert_to_xarray(val_output_surface_acc, val_output_upper_air_acc, start_times, self.params, self.valid_dataset, acc = True,
                                                                            diagnostic_prediction=val_output_diagnostic_acc)
                                    else:
                                        acc_datasets = self.convert_to_xarray(val_output_surface_acc, val_output_upper_air_acc, start_times, self.params, self.valid_dataset, acc = True)
                                    acc_prepared_datasets = [self.prepare_preds(ds, acc = True) for ds in acc_datasets]
                                    acc_combined_dataset = self.combine_datasets(acc_prepared_datasets)
                                    acc_predictions.append(acc_combined_dataset)

                                    acc_gt_surface = self.valid_dataset.surface_inv_transform(val_target_surface.cpu()).numpy()
                                    acc_gt_upper_air = self.valid_dataset.upper_air_inv_transform(val_target_upper_air.cpu()).numpy()
                                    if self.params.has_diagnostic:
                                        acc_gt_diagnostic = self.valid_dataset.diagnostic_inv_transform(val_target_diagnostic.cpu()).numpy()
                                        acc_gt_datasets = self.convert_to_xarray(acc_gt_surface, acc_gt_upper_air, start_times, self.params, self.valid_dataset, acc = True,
                                                                                diagnostic_prediction = acc_gt_diagnostic)
                                    else:
                                        acc_gt_datasets = self.convert_to_xarray(acc_gt_surface, acc_gt_upper_air, start_times, self.params, self.valid_dataset, acc = True)
                                    acc_gt_prepared_datasets = [self.prepare_preds(ds, acc=True) for ds in acc_gt_datasets]
                                    acc_gt_combined_dataset = self.combine_datasets(acc_gt_prepared_datasets)
                                    acc_ground_truths.append(acc_gt_combined_dataset)

                                # Prepare the ground truths
                                lead_time_indices = [lt - 1 for lt in self.params.forecast_lead_times]  # Convert to 0-based index

                                if self.params.diagnostic_spectra:
                                    # Prepare the predictions (only forecast lead times)
                                    if self.params.has_diagnostic:
                                        datasets = self.convert_to_xarray(val_output_surface_t, val_output_upper_air_t, start_times, self.params, self.valid_dataset, acc = False,
                                                                        diagnostic_prediction = val_output_diagnostic_t)
                                    else:
                                        datasets = self.convert_to_xarray(val_output_surface_t, val_output_upper_air_t, start_times, self.params, self.valid_dataset, acc = False)
                                    prepared_datasets = [self.prepare_preds(ds, acc = False) for ds in datasets]
                                    combined_dataset = self.combine_datasets(prepared_datasets)

                                    # gt_surface = self.valid_dataset.surface_inv_transform(val_target_surface.cpu()).numpy()
                                    # gt_upper_air = self.valid_dataset.upper_air_inv_transform(val_target_upper_air.cpu()).numpy()

                                    # only take the necessary indices as the dataloader now returns all time steps. 
                                    gt_surface = self.valid_dataset.surface_inv_transform(val_target_surface[:, lead_time_indices].cpu()).numpy()
                                    gt_upper_air = self.valid_dataset.upper_air_inv_transform(val_target_upper_air[:, lead_time_indices].cpu()).numpy()
                                    if self.params.has_diagnostic:
                                        gt_diagnostic = self.valid_dataset.diagnostic_inv_transform(val_target_diagnostic[:, lead_time_indices].cpu()).numpy()
                                        gt_datasets = self.convert_to_xarray(gt_surface, gt_upper_air, start_times, self.params, self.valid_dataset, acc = False,
                                                                            diagnostic_prediction = gt_diagnostic)
                                    else:
                                        gt_datasets = self.convert_to_xarray(gt_surface, gt_upper_air, start_times, self.params, self.valid_dataset, acc = False)
                                    gt_prepared_datasets = [self.prepare_preds(ds, acc = False) for ds in gt_datasets]
                                    gt_combined_dataset = self.combine_datasets(gt_prepared_datasets)

                                    all_predictions.append(combined_dataset)
                                    all_ground_truths.append(gt_combined_dataset)

                            step_idx += 1
                        
                        val_input_surface, val_input_upper_air = val_output_surface, val_output_upper_air
                        del val_output_surface, val_output_upper_air
                    
                del val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air
                torch.cuda.empty_cache()
                valid_steps += 1.
                
            print("Finished batch validation.")
        

        # After the loop, combine all predictions and ground truthsacc_combined_predictions.to_netcdf(os.path.join(val_data_dir, 'predictions.nc'))
        if self.params.diagnostic_spectra:
            combined_predictions = xr.concat(all_predictions, dim='time')
            combined_ground_truths = xr.concat(all_ground_truths, dim='time')
        print("\nFinished combining predictions and ground truths.")
           

        if self.params.diagnostic_acc or self.params.diagnostic_gif:
            acc_combined_predictions = xr.concat(acc_predictions, dim='time')
            acc_combined_ground_truths = xr.concat(acc_ground_truths, dim='time')
            if self.params.just_validate or (not self.params.just_validate and self.epoch == 100):
                val_data_dir = os.path.join(self.params.experiment_dir, 'validation_data')
                os.makedirs(val_data_dir, exist_ok=True)
                for start_time in acc_combined_predictions.time.values:
                    pred_out = acc_combined_predictions.sel(time = start_time)
                    pred_out = pred_out.drop_vars("time")
                    pred_out['lead_time'] = [start_time + timedelta(hours = int(elem)) for elem in pred_out.lead_time.values]
                    pred_out = pred_out.rename({'lead_time': 'time'})
                    pred_out.to_netcdf(os.path.join(val_data_dir, f'prediction_{start_time.strftime("%Y-%m-%d_%H")}.nc'))
                for start_time in acc_combined_ground_truths.time.values:
                    truth_out = acc_combined_ground_truths.sel(time = start_time)
                    truth_out = truth_out.drop_vars("time")
                    truth_out['lead_time'] = [start_time + timedelta(hours = int(elem)) for elem in truth_out.lead_time.values]
                    truth_out = truth_out.rename({'lead_time': 'time'})
                    truth_out.to_netcdf(os.path.join(val_data_dir, f'ground_truth_{start_time.strftime("%Y-%m-%d_%H")}.nc'))
        print("\nFinished combining ACC predictions and ground truths.")

        max_lead_time = max(self.params.forecast_lead_times)
        acc_times_hours = [(lt + 1) * self.params.timedelta_hours for lt in range(max_lead_time)]

        if self.params.diagnostic_acc:
            # Compute ACC
            print("\nComputing ACC...")
            # Compute ACC for all data
            acc = OrderedDict({
                'Pangu': evaluate_iterative_forecast(
                    acc_combined_predictions, 
                    acc_combined_ground_truths, 
                    compute_weighted_acc,
                    # mean over these dimensions
                    mean_dims=['lat', 'lon', 'time'],
                    clim=self.climatology
                )
            })
            

            # Plot ACC over lead time
            fig, axs = plot_acc_over_lead_time(acc, acc_times_hours, var_dict=self.params.diagnostic_acc_var_dict)

        if self.params.diagnostic_spectra:
            print("\nCalculating power spectrum...")
            k_x_pred, power_spectrum_avg_pred = zonal_averaged_power_spectrum(combined_predictions, time_avg=True) 
            k_x_gt, power_spectrum_avg_gt = zonal_averaged_power_spectrum(combined_ground_truths, time_avg= True)
            preds_times = combined_predictions.time.values
            preds_times = preds_times.cpu().numpy() if isinstance(preds_times, torch.Tensor) else preds_times
            print("\nFinished calculating power spectrum.")
        
        # Save the plot
        if self.world_rank == 0:
            if self.params.diagnostic_acc:
                plot_filename = os.path.join(self.output_dir, f"acc_plot_epoch_{self.epoch}.png")
                fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.close(fig)  # Close the figure to free up memory
                print("\nFinished ACC..")

            if self.params.diagnostic_gif:
                print("\nMaking GIF...")

                gif_filename = os.path.join(self.diagnostics_dir, f"geopotential_height_animation_epoch_{self.epoch}.gif")
                make_gif(acc_combined_predictions, acc_combined_ground_truths, "Model Forecast",
                    list(set(['geopotential', 'zg']) & set(self.params.upper_air_variables))[0],
                    gif_filename, climatology=self.climatology, plev=50000)
                print("\nFinished creating GIF animation.")

            if self.params.diagnostic_spectra:
                print("\nMaking Power Spectrum...")
                # Calculate zonal averaged power spectrum. 

                # k_x_pred, power_spectrum_avg_pred = zonal_averaged_power_spectrum(combined_dataset, time_avg=True) 
                # k_x_gt, power_spectrum_avg_gt = zonal_averaged_power_spectrum(gt_combined_dataset, time_avg= True)
                # preds_times = combined_dataset.time.values

                path_filename = os.path.join(self.spectra_dir, f"power_spectrum_epoch_{self.epoch}.png")
                preds_times = preds_times.cpu().numpy() if isinstance(preds_times, torch.Tensor) else preds_times
                self.plot_in_separate_process(power_spectrum_avg_pred, power_spectrum_avg_gt,preds_times, path_filename)
                print("\nFinished Power Spectrum...")

        if dist.is_initialized():
            dist.all_reduce(valid_buff)
            dist.all_reduce(valid_surface_lwrmse)
            dist.all_reduce(valid_upper_air_lwrmse)
            for loss_tensor in multi_step_losses.values():
                dist.all_reduce(loss_tensor)


        # divide by number of steps
        valid_buff[0:-1] = valid_buff[0:-1] / valid_buff[-1]
        valid_surface_lwrmse = (valid_surface_lwrmse / valid_buff[-1]).detach()
        valid_upper_air_lwrmse = (valid_upper_air_lwrmse / valid_buff[-1]).detach()
        if self.params.has_diagnostic:
            valid_diagnostic_lwrmse = (valid_diagnostic_lwrmse / valid_buff[-1]).detach()
        for key in multi_step_losses:
            multi_step_losses[key] /= valid_buff[-1]

        valid_buff_cpu = valid_buff.detach()

        diagnostic_logs['epoch'] = self.epoch
        diagnostic_logs['valid_loss'] = valid_buff_cpu[0]
        diagnostic_logs['valid_loss_sfc'] = valid_buff_cpu[1]
        diagnostic_logs['valid_loss_upper_air'] = valid_buff_cpu[2]
        if self.params.has_diagnostic:
            diagnostic_logs['valid_loss_diag'] = valid_buff_cpu[3]

            #mean_norm_lwrmse = torch.mean(torch.cat((valid_surface_lwrmse, valid_upper_air_lwrmse.flatten()), dim = -1))
            #diagnostic_logs['valid_mean_norm_lwrmse'] = mean_norm_lwrmse
            for l, steps in enumerate(lead_times_steps):
                for j, var in enumerate(self.valid_dataset.surface_variables):
                    diagnostic_logs[f'valid_{var}_{steps}step_lwrmse'] = valid_surface_lwrmse[l, j] * self.valid_dataset.surface_std[j]
                for j, var in enumerate(self.valid_dataset.upper_air_variables):
                    if var != 'zg' and var != 'geopotential_height' and self.valid_dataset.use_sigma_levels:
                        for k, level in enumerate(self.valid_dataset.sigma_levels):
                            diagnostic_logs[f'valid_{var}_level{level:.3f}_{steps}step_lwrmse'] = valid_upper_air_lwrmse[l, j, k] * self.valid_dataset.upper_air_std[j, k]
                    else:
                        for k, level in enumerate(self.valid_dataset.levels):
                            diagnostic_logs[f'valid_{var}_level{level:.3f}_{steps}step_lwrmse'] = valid_upper_air_lwrmse[l, j, k] * self.valid_dataset.upper_air_std[j, k]
                if self.params.has_diagnostic:
                    for j, var in enumerate(self.valid_dataset.diagnostic_variables):
                        diagnostic_logs[f'valid_{var}_{steps}step_lwrmse'] = valid_diagnostic_lwrmse[l, j] * self.valid_dataset.diagnostic_std[j]
                        
            if self.long_validation and self.world_rank == 0 and self.epoch % self.epochs_per_long_validation == 0 and no_nans:
                for j, var in enumerate(self.valid_dataset.surface_variables):
                    diagnostic_logs[f'valid_{var}_bias_lwrmse'] = val_surface_bias_lwrmse[j]
                for j, var in enumerate(self.valid_dataset.upper_air_variables):
                    if var != 'zg' and var != 'geopotential_height' and self.valid_dataset.use_sigma_levels:
                        for k, level in enumerate(self.valid_dataset.sigma_levels):
                            diagnostic_logs[f'valid_{var}_level{level:.3f}_bias_lwrmse'] = val_upper_air_bias_lwrmse[j, k]
                    else:
                        for k, level in enumerate(self.valid_dataset.levels):
                            diagnostic_logs[f'valid_{var}_level{level:.3f}_bias_lwrmse'] = val_upper_air_bias_lwrmse[j, k]
                if self.params.has_diagnostic:
                    for j, var in enumerate(self.valid_dataset.diagnostic_variables):
                        diagnostic_logs[f'valid_{var}_bias_lwrmse'] = val_diagnostic_bias_lwrmse[j]
                
            #if dist.is_initialized():
            #    for key in sorted(diagnostic_logs.keys()):
            #        dist.all_reduce(diagnostic_logs[key].detach())
            #        diagnostic_logs[key] = float(diagnostic_logs[key]/dist.get_world_size())

        # Add multi-day losses to diagnostic logs
        for key, value in multi_step_losses.items():
            diagnostic_logs[key] = value.item()

        if self.params.log_to_wandb:
            wandb.log(diagnostic_logs, step=self.wandb_step)
            if self.params.diagnostic_acc:
                wandb.log({
                    "ACC_plot": wandb.Image(plot_filename),
                    "epoch": self.epoch
                }, step=self.wandb_step)
            if self.params.diagnostic_gif:
                if gif_filename:
                    wandb.log({
                        "Evolution_GIF": wandb.Video(gif_filename),
                        "epoch": self.epoch
                    }, step=self.wandb_step)
            if self.params.diagnostic_spectra:
                wandb.log({
                    "power_spectrum_plot": wandb.Image(path_filename),
                    "epoch": self.epoch,
                }, step=self.wandb_step)
        
        valid_time = time.time() - valid_start
       
        return valid_time, diagnostic_logs

    def validate_ensemble_forecast(self):
        """
        Run ensemble forecast validation using Stepper from ensemble_inference.
        This function makes ensemble predictions and computes observables without saving predictions.
        
        Returns:
            float: Total time taken for ensemble validation in seconds
        """
        if not hasattr(self.params, 'ensemble_validation') or not self.params.ensemble_validation:
            return 0.0
        
        ensemble_val_start = time.time()
        
        if self.world_rank == 0:
            logging.info("Starting ensemble forecast validation...")
        
        try:
            # Create a copy of params for ensemble inference
            ensemble_params = copy.deepcopy(self.params)
            
            # Initialize event_type_mapping (maps file index to event_type)
            event_type_mapping = {}
            init_data = {}
            
            # Set up ensemble inference parameters
            # Store mapping from filepath to (event_type, final_datetime) for datetime extraction
            filepath_to_datetime_info = {}
            
            if hasattr(self.params, 'init_nc_filepath_files') and len(self.params.init_nc_filepath_files) > 0:
                with open(self.params.init_nc_filepath_files, 'r') as f:
                    init_data = json.load(f)
                ensemble_params['init_nc_filepaths'] = []
                ensemble_params['save_basenames'] = []
                ensemble_params['output_dirs'] = []
                
                # Extract file paths from JSON structure (hierarchical: event_type -> {filepath: [first_datetime, final_datetime]})
                # Create mapping from file index to event_type (particle_idx maps to file index)
                file_idx = 0
                for event_type, event_data in init_data.items():
                    if isinstance(event_data, dict):
                        # The JSON structure has filepaths as keys, and [first_datetime, final_datetime] as values
                        for filepath, datetime_list in event_data.items():
                            if isinstance(datetime_list, list) and len(datetime_list) >= 2:
                                # Extract final datetime (second element)
                                final_datetime_str = datetime_list[1]
                                ensemble_params['init_nc_filepaths'].append(filepath)
                                event_type_mapping[file_idx] = event_type
                                # Store mapping for later use in computing init_datetimes
                                filepath_to_datetime_info[filepath] = {
                                    'event_type': event_type,
                                    'final_datetime_str': final_datetime_str
                                }
                                file_idx += 1
                                
                                # Create deterministic output location for ensemble validation artifacts
                                # (observables + combined netCDF + error metrics)
                                base_dir = os.path.join(self.params.experiment_dir, 'ensemble_validation')
                                os.makedirs(base_dir, exist_ok=True)
                                temp_dir = os.path.join(base_dir, 'tmp')
                                os.makedirs(temp_dir, exist_ok=True)

                                # Stepper expects lists for these fields; our observation functions use a single save_basename.
                                # We'll still populate Stepper fields for compatibility.
                                basename = os.path.join(temp_dir, f"ensemble_val_{event_type}_epoch{self.epoch:04d}")
                                ensemble_params['save_basenames'].append(basename)
                                ensemble_params['output_dirs'].append(temp_dir)
            
            # Parse ensemble_inference_hours (comma-separated list)
            if hasattr(self.params, 'ensemble_inference_hours') and len(self.params.ensemble_inference_hours) > 0:
                if isinstance(self.params.ensemble_inference_hours, str):
                    inference_hours_list = [int(h.strip()) for h in self.params.ensemble_inference_hours.split(',')]
                elif isinstance(self.params.ensemble_inference_hours, (list, tuple)):
                    inference_hours_list = [int(h) for h in self.params.ensemble_inference_hours]
                else:
                    inference_hours_list = [int(self.params.ensemble_inference_hours)]
            else:
                inference_hours_list = []
                logging.warning("No ensemble_inference_hours specified. Skipping ensemble validation.")
                ensemble_val_time = time.time() - ensemble_val_start
                return ensemble_val_time
            
            # Get has_year_zero from params (default to False if not present)
            has_year_zero = getattr(self.params, 'has_year_zero', False)
            
            # Extract final datetimes and compute init_datetimes for each inference_hours value
            # For each filepath, we have the final_datetime_str from the JSON
            # For each inference_hours, compute init_datetime = final_datetime - inference_hours
            init_datetimes_by_lead_time = {}  # Maps inference_hours -> list of init_datetimes
            
            for inference_hours in inference_hours_list:
                init_datetimes_for_lead = []
                
                for filepath in ensemble_params['init_nc_filepaths']:
                    if filepath in filepath_to_datetime_info:
                        final_datetime_str = filepath_to_datetime_info[filepath]['final_datetime_str']
                        try:
                            # Parse the datetime string (format: "YYYY-MM-DD HH:MM:SS")
                            final_datetime = cftime.datetime.strptime(
                                final_datetime_str,
                                "%Y-%m-%d %H:%M:%S",
                                has_year_zero=has_year_zero,
                                calendar='proleptic_gregorian'
                            )
                            
                            # Convert to DatetimeProlepticGregorian format expected by Stepper
                            final_datetime_dt = cftime.DatetimeProlepticGregorian(
                                final_datetime.year,
                                final_datetime.month,
                                final_datetime.day,
                                hour=final_datetime.hour,
                                has_year_zero=has_year_zero
                            )
                            
                            # Compute init_datetime by subtracting inference_hours
                            init_datetime = final_datetime_dt - timedelta(hours=int(inference_hours))
                            
                            init_datetimes_for_lead.append(init_datetime)
                            
                        except Exception as e:
                            logging.error(f"Error parsing datetime '{final_datetime_str}' for filepath {filepath}: {e}")
                            import traceback
                            logging.error(traceback.format_exc())
                            # Fallback: use a default datetime if parsing fails
                            default_dt = cftime.DatetimeProlepticGregorian(1, 1, 1, 0, has_year_zero=has_year_zero)
                            init_datetimes_for_lead.append(default_dt)
                    else:
                        logging.warning(f"Filepath {filepath} not found in datetime info mapping. Using default datetime.")
                        default_dt = cftime.DatetimeProlepticGregorian(1, 1, 1, 0, has_year_zero=has_year_zero)
                        init_datetimes_for_lead.append(default_dt)
                
                init_datetimes_by_lead_time[inference_hours] = init_datetimes_for_lead
                
                if self.world_rank == 0:
                    logging.info(f"Computed init_datetimes for inference_hours={inference_hours}: {[str(dt) for dt in init_datetimes_for_lead]}")
            
            # Disable saving forecasts
            ensemble_params['save_forecasts'] = False
            
            # Set up data directory for ensemble inference
            if not hasattr(ensemble_params, 'data_dir'):
                ensemble_params['data_dir'] = self.params.data_dir
            
            # Set ensemble batch size (compute local batch size for distributed training)
            if hasattr(self.params, 'ensemble_batch_size'):
                ensemble_batch_size = self.params.ensemble_batch_size
            else:
                ensemble_batch_size = self.params.batch_size

            # Set ensemble epsilon factor
            if hasattr(self.params, 'ensemble_epsilon_factor'):
                ensemble_params['epsilon_factor'] = self.params.ensemble_epsilon_factor
            else:
                ensemble_params['epsilon_factor'] = self.params.epsilon_factor

            # Set number of validation ensemble members
            if hasattr(self.params, 'num_validation_ensemble_members'):
                ensemble_params['num_ensemble_members'] = self.params.num_validation_ensemble_members
            else:
                ensemble_params['num_ensemble_members'] = self.params.num_ensemble_members

            # Set number of validation ensemble members
            if hasattr(self.params, 'validation_ensemble_members_per_pred'):
                ensemble_params['ensemble_members_per_pred'] = self.params.validation_ensemble_members_per_pred
            else:
                ensemble_params['ensemble_members_per_pred'] = self.params.ensemble_members_per_pred
            
            # Compute local batch size for distributed training (same logic as training batch_size)
            if dist.is_initialized():
                ensemble_params['global_batch_size'] = ensemble_batch_size
                ensemble_params['batch_size'] = int(ensemble_batch_size // dist.get_world_size())
            else:
                ensemble_params['global_batch_size'] = ensemble_batch_size
                ensemble_params['batch_size'] = ensemble_batch_size
            
            # Ensure required ensemble parameters are set
            if not hasattr(ensemble_params, 'num_ensemble_members'):
                ensemble_params['num_ensemble_members'] = 1
            if not hasattr(ensemble_params, 'ensemble_members_per_pred'):
                ensemble_params['ensemble_members_per_pred'] = ensemble_params['num_ensemble_members']
            
            # Parse error metrics to compute (optional)
            error_metrics_list = []
            if hasattr(self.params, 'error_metrics') and isinstance(self.params.error_metrics, str) and len(self.params.error_metrics) > 0:
                error_metrics_list = [m.strip() for m in self.params.error_metrics.split(',') if len(m.strip()) > 0]

            # Set up observable functions (same for all inference hours)
            obs_functions = []
            obs_args_list = []
            obs_function_names = []
            
            if hasattr(self.params, 'obs_functions') and len(self.params.obs_functions) > 0:
                obs_function_names = [f.strip() for f in self.params.obs_functions.split(',')]
                
                # Use obs_args_dict if provided (already parsed as dictionary in __main__)
                # If it doesn't exist, default to empty dict
                obs_args_dict = {}
                if hasattr(self.params, 'obs_args_dict') and isinstance(self.params.obs_args_dict, dict):
                    obs_args_dict = self.params.obs_args_dict
                
                # Import and set up observable functions
                for obs_func_name in obs_function_names:
                    try:
                        # Import from v2.0/utils/observations.py
                        from utils import observations as obs_mod
                        if not hasattr(obs_mod, obs_func_name):
                            logging.warning(f"Unknown observable function: {obs_func_name}. Skipping.")
                            continue
                        obs_func = getattr(obs_mod, obs_func_name)
                        obs_functions.append(obs_func)

                        # For observations.py functions, obs_args_dict[func] should be a list containing
                        # the function-specific args *excluding* save_basename.
                        # Handle two formats:
                        # 1. Function-name keyed: {"unweighted_nday_mean": [target_duration, var, regions, region_file_path]}
                        # 2. Flat dictionary: {"target_duration": 7, "var": "tas", "regions": [...], "region_file_path": "..."}
                        if obs_func_name in obs_args_dict:
                            # Format 1: function name is a key with list value
                            obs_args_list.append(obs_args_dict[obs_func_name])
                        elif isinstance(obs_args_dict, dict) and len(obs_args_dict) > 0:
                            # Format 2: flat dictionary - convert to list based on function requirements
                            if obs_func_name == 'unweighted_nday_mean':
                                # Expected order: [target_duration, var, regions, region_file_path]
                                required_keys = ['target_duration', 'var', 'regions', 'region_file_path']
                                if all(key in obs_args_dict for key in required_keys):
                                    obs_args_list.append([
                                        obs_args_dict['target_duration'],
                                        obs_args_dict['var'],
                                        obs_args_dict['regions'],
                                        obs_args_dict['region_file_path']
                                    ])
                                else:
                                    missing_keys = [key for key in required_keys if key not in obs_args_dict]
                                    logging.warning(f"Missing required keys for {obs_func_name}: {missing_keys}. Using empty args.")
                                    obs_args_list.append([])
                            else:
                                # For other functions, try to use as-is or log warning
                                logging.warning(f"Unknown function {obs_func_name} with flat dict format. Using empty args.")
                                obs_args_list.append([])
                        else:
                            obs_args_list.append([])
                    except Exception as e:
                        logging.warning(f"Could not import observable function {obs_func_name}: {e}")
            
            # Helper: build a deterministic save_basename (single path) for observables
            base_dir = os.path.join(self.params.experiment_dir, 'ensemble_validation')
            os.makedirs(base_dir, exist_ok=True)
            tmp_dir = os.path.join(base_dir, 'tmp')
            os.makedirs(tmp_dir, exist_ok=True)
            save_basename_forecast = os.path.join(tmp_dir, f"ensemble_forecast_epoch{self.epoch:04d}")
            save_basename_truth = os.path.join(tmp_dir, "ensemble_truth")

            # Helper: check if truth combined files exist already
            def truth_combined_exists() -> bool:
                truth_glob = os.path.join(os.path.dirname(save_basename_truth), f"{os.path.basename(save_basename_truth)}_*_truth_combined.nc")
                return len(glob.glob(truth_glob)) > 0

            # If truth observables are missing, compute them once on rank 0.
            # If error metrics are requested, ensure truth observables are computed (even if they exist, 
            # we may need to recompute them to ensure they're up to date)
            truth_exists = truth_combined_exists()
            needs_truth_for_metrics = len(error_metrics_list) > 0
            
            if self.world_rank == 0 and len(obs_functions) > 0 and (not truth_exists or needs_truth_for_metrics):
                try:
                    from utils import observations as obs_mod
                    if not truth_exists:
                        logging.info("Truth observables not found; computing truth observables on world_rank=0...")
                    elif needs_truth_for_metrics:
                        logging.info("Error metrics requested; ensuring truth observables are computed on world_rank=0...")

                    # Compute truth observables by loading the event datasets directly and creating
                    # a pseudo-ensemble with a single member.
                    # The event_data structure is: {filepath: [first_datetime, final_datetime]}
                    # Use the same file_idx mapping as forecast processing to ensure particle_idx matches
                    file_idx = 0
                    for event_type, event_data in init_data.items():
                        if not isinstance(event_data, dict):
                            continue
                        # Iterate through filepaths (keys) in event_data, same as forecast processing
                        for data_path, datetime_list in event_data.items():
                            if not isinstance(datetime_list, list) or len(datetime_list) < 2:
                                continue
                            try:
                                with warnings.catch_warnings():
                                    warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*use_cftime.*')
                                    ds_truth = xr.open_dataset(data_path, use_cftime=True)
                            except Exception as e:
                                logging.warning(f"Could not open truth dataset for {event_type} at {data_path}: {e}")
                                file_idx += 1  # Still increment to keep alignment
                                continue

                            # Ensure an ensemble dimension exists (single member)
                            # We mimic Stepper output where ensemble dimension is named 'ensemble_idx'
                            if 'ensemble_idx' not in ds_truth.dims:
                                ds_truth = ds_truth.expand_dims({'ensemble_idx': [0]})

                            # Use file_idx as particle_idx to match forecast processing
                            # This ensures particle indices are consistent between forecast and truth
                            particle_idxs = [file_idx]

                            # Compute truth observables for each forecast horizon by slicing truth time dimension.
                            # IMPORTANT: do NOT pass lead_time_hours into observation functions for truth.
                            for inference_hours in inference_hours_list:
                                time_steps = int(inference_hours // self.params.timedelta_hours)
                                ds_truth_h = ds_truth
                                if 'time' in ds_truth.dims and ds_truth.sizes.get('time', 0) > 0 and time_steps > 0:
                                    ds_truth_h = ds_truth.isel(time=slice(0, min(time_steps, ds_truth.sizes['time'])))

                                for obs_func, obs_func_name, obs_args in zip(obs_functions, obs_function_names, obs_args_list):
                                    # observations.py standard order (truth mode):
                                    # [datasets, particle_idxs, ensemble_start, ensemble_end, event_type, ...func_args..., save_basename]
                                    func_args = [[ds_truth_h], particle_idxs, 0, 1, event_type] + list(obs_args) + [save_basename_truth]
                                    try:
                                        obs_func(tuple(func_args))
                                    except Exception as e:
                                        logging.warning(f"Error computing truth observable {obs_func_name} at inference_hours={inference_hours}: {e}")

                            ds_truth.close()
                            file_idx += 1  # Increment file index to match forecast processing

                    # Combine truth observables (truth files omit lead/ens labels) into *_truth_combined.nc
                    obs_mod.combine_observation_truth(
                        save_basename_truth,
                        obs_function_names,
                        output_dir=os.path.dirname(save_basename_truth),
                        data_dict=init_data
                    )

                    logging.info("Finished computing truth observables.")
                except Exception as e:
                    logging.error(f"Error computing truth observables: {e}")

            if dist.is_initialized():
                dist.barrier()

            # Core wrapper for observation functions.
            # Stepper calls obs_function([ensemble_datasets, particle_idxs, ensemble_start, ensemble_end] + obs_args)
            # We inject lead_time_hours ONLY as a label for forecast files (optional), plus save_basename.
            def combined_obs_function_core(args, lead_time_hours_label: int):
                """Wrapper that calls all observable functions with their respective args."""
                # args format: [ensemble_datasets, particle_idxs, ensemble_start, ensemble_end] + additional_args
                if len(args) < 4:
                    logging.error(f"Invalid args for observable function: expected at least 4 args, got {len(args)}")
                    return
                
                ensemble_datasets, particle_idxs, ensemble_start, ensemble_end = args[0], args[1], args[2], args[3]
                
                # Extract event_type from particle_idx (particle_idx maps to file index in init_nc_filepaths)
                # Handle particle_idxs which can be: single value, list, numpy array, or torch tensor
                # With batch_size > 1, particle_idxs will have multiple elements - use first one for event_type mapping
                if isinstance(particle_idxs, torch.Tensor):
                    # Convert tensor to numpy and get first element
                    particle_idxs = particle_idxs.cpu().numpy()
                
                if isinstance(particle_idxs, (list, np.ndarray)):
                    if len(particle_idxs) > 0:
                        particle_idx = int(particle_idxs[0])
                    else:
                        raise ValueError("particle_idxs is empty")
                else:
                    # Single scalar value
                    particle_idx = int(particle_idxs)
                
                # Map particle_idx to event_type (particle_idx corresponds to file index)
                event_type = event_type_mapping.get(particle_idx, 'unknown')
                
                for obs_func, obs_args, obs_name in zip(obs_functions, obs_args_list, obs_function_names):
                    try:
                        # observations.py expects:
                        # (datasets, particle_idxs, ensemble_start, ensemble_end, event_type, [lead_time_hours label], *func_specific_args, save_basename)
                        func_args = [ensemble_datasets, particle_idxs, ensemble_start, ensemble_end, event_type, int(lead_time_hours_label)] + list(obs_args) + [save_basename_forecast]
                        obs_func(tuple(func_args))
                    except Exception as e:
                        logging.error(f"Error in observable function {obs_func.__name__}: {e}")
                        import traceback
                        logging.error(traceback.format_exc())
            
            # Loop through each ensemble_inference_hours value
            for inference_hours in inference_hours_list:
                if self.world_rank == 0:
                    logging.info(f"Running ensemble validation for {inference_hours} hours...")
                
                # Create a copy of ensemble_params for this specific inference hours value
                current_ensemble_params = copy.deepcopy(ensemble_params)
                current_ensemble_params['ensemble_inference_hours'] = inference_hours
                current_ensemble_params['ensemble_inference_steps'] = inference_hours // self.params.timedelta_hours
                
                # Set init_datetimes for this lead time
                if inference_hours in init_datetimes_by_lead_time:
                    current_ensemble_params['init_datetimes'] = init_datetimes_by_lead_time[inference_hours]
                    if self.world_rank == 0:
                        logging.info(f"Using init_datetimes for inference_hours={inference_hours}: {[str(dt) for dt in current_ensemble_params['init_datetimes']]}")
                else:
                    if self.world_rank == 0:
                        logging.warning(f"No init_datetimes computed for inference_hours={inference_hours}. Stepper may fail if init_datetime/init_datetimes are required.")
                
                # Create Stepper instance for this inference hours value
                # Stepper will create its model using a training-style dataset to ensure same architecture
                with open(f'params_ensemble_{args.run_num}.pkl', 'wb') as f:
                    pickle.dump(current_ensemble_params, f)
                stepper = Stepper([current_ensemble_params], self.world_rank, use_6h_24h_model=False, async_save=False)
                
                # Replace Stepper's model with the current training model to use the latest weights
                # This ensures we use the model from the current epoch, not from checkpoint
                # Get state dict from training model (handle DDP wrapping)
                if hasattr(self.model, 'module'):
                    # DDP wrapped model
                    training_state_dict = self.model.module.state_dict()
                else:
                    training_state_dict = self.model.state_dict()
                
                # Load into stepper model (handle DDP wrapping if present)
                if hasattr(stepper.model, 'module'):
                    # Stepper model is also DDP wrapped
                    stepper.model.module.load_state_dict(training_state_dict)
                else:
                    stepper.model.load_state_dict(training_state_dict)
                
                # Also copy integrator if it exists
                if hasattr(self, 'integrator') and hasattr(stepper, 'integrator'):
                    stepper.integrator = self.integrator
                
                # Run ensemble prediction with observables for this inference hours value
                try:
                    if len(obs_functions) > 0:
                        # Bind the lead_time_hours label for this run into the callable
                        def _obs_fn(bound_args, _lead=int(inference_hours)):
                            return combined_obs_function_core(bound_args, lead_time_hours_label=_lead)
                        stepper.predict(obs_function=_obs_fn, obs_args=[])
                    else:
                        logging.warning("No observable functions loaded. Running ensemble prediction without observables.")
                        stepper.predict()
                    
                    if self.world_rank == 0:
                        logging.info(f"Completed ensemble validation for {inference_hours} hours.")
                except Exception as e:
                    logging.error(f"Error running ensemble validation for {inference_hours} hours: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
            
            if dist.is_initialized():
                dist.barrier()

            if self.world_rank == 0:
                logging.info("Ensemble forecast validation completed for all inference hours.")

                # Combine forecast observables and compute error metrics (if requested)
                error_metrics_results = {}
                try:
                    from utils import observations as obs_mod
                    obs_mod.combine_observations(save_basename_forecast, obs_function_names, output_dir=os.path.dirname(save_basename_forecast), data_dict=init_data)
                    if len(error_metrics_list) > 0:
                        # Ensure truth observables exist before computing error metrics
                        if not truth_combined_exists():
                            logging.warning("Truth observables not found when computing error metrics. Attempting to compute them now...")
                            # Truth observables should have been computed earlier, but if they're missing, 
                            # we can't compute error metrics. Log error and skip.
                            logging.error("Cannot compute error metrics: truth observables are missing. Please ensure truth observables are computed first.")
                        else:
                            error_metrics_results = obs_mod.compute_error_metrics(
                                save_basename=save_basename_forecast,
                                save_basename_truth=save_basename_truth,
                                obs_function_names=obs_function_names,
                                error_metrics=error_metrics_list,
                                output_dir=os.path.dirname(save_basename_forecast),
                            )
                        
                        # Log error metrics to wandb if enabled
                        if self.params.log_to_wandb and len(error_metrics_results) > 0:
                            wandb_error_logs = {}
                            metrics_to_define = set()  # Track which metrics need to be defined
                            
                            # First pass: collect all metric names that will be logged
                            # Structure: error_metrics_results[event_type][obs_function_name][function_specific_string]
                            for event_type, obs_func_dict in error_metrics_results.items():
                                for obs_function_name, func_specific_dict in obs_func_dict.items():
                                    for function_specific_string, metrics_dict in func_specific_dict.items():
                                        if 'lead_times' not in metrics_dict:
                                            continue
                                        
                                        lead_times = metrics_dict['lead_times']
                                        
                                        # For each error metric
                                        for error_metric in error_metrics_list:
                                            if error_metric not in metrics_dict:
                                                continue
                                            
                                            errors_by_lead_time = metrics_dict[error_metric]
                                            
                                            # Create wandb log entry for each lead_time
                                            for lead_time_hours, error_value in zip(lead_times, errors_by_lead_time):
                                                # Only process finite values
                                                if not np.isfinite(error_value):
                                                    continue
                                                
                                                # Create metric name: event_type_obs_function_func_specific_lead_time_error_metric
                                                # Sanitize function_specific_string for wandb (replace problematic chars)
                                                func_specific_safe = function_specific_string.replace('/', '_').replace('\\', '_')
                                                metric_name = f"{event_type}_{obs_function_name}_{func_specific_safe}_{int(lead_time_hours)}h_{error_metric}"
                                                
                                                metrics_to_define.add(metric_name)
                                                wandb_error_logs[metric_name] = float(error_value)
                            
                            # Check and define metrics that haven't been created yet
                            if len(metrics_to_define) > 0:
                                # Initialize set of defined metrics if it doesn't exist
                                if not hasattr(self, '_defined_wandb_metrics'):
                                    self._defined_wandb_metrics = set()
                                
                                # Define any metrics that haven't been defined yet
                                for metric_name in metrics_to_define:
                                    if metric_name not in self._defined_wandb_metrics:
                                        try:
                                            wandb.define_metric(metric_name, step_metric="epoch")
                                            self._defined_wandb_metrics.add(metric_name)
                                        except Exception as e:
                                            # If metric definition fails, log warning but continue
                                            logging.warning(f"Could not define wandb metric {metric_name}: {e}")
                            
                            # Log all error metrics to wandb at once
                            if len(wandb_error_logs) > 0:
                                wandb.log(wandb_error_logs, step=self.wandb_step)
                                logging.info(f"Logged {len(wandb_error_logs)} ensemble validation error metrics to wandb.")
                                
                except Exception as e:
                    logging.warning(f"Error combining observables / computing error metrics: {e}")
                    import traceback
                    logging.warning(traceback.format_exc())
                
        except Exception as e:
            logging.error(f"Error in ensemble forecast validation: {e}")
            import traceback
            logging.error(traceback.format_exc())
        
        ensemble_val_time = time.time() - ensemble_val_start
        return ensemble_val_time

    def prepare_preds(self, preds, acc = False):
        preds = preds.rename({'time': 'lead_time'})
        # If bug, change this back to values[0]
        preds['time'] = preds.lead_time.values[0:1]
        preds = preds.set_coords('time')
        if acc:
        # For ACC, use all time steps
            # lead_times = range(len(preds.lead_time))
            lead_times = range(1, len(preds.lead_time) + 1)  # Start from 1 to match forecast lead times
        else:
        # For non-ACC, use forecast lead times
            lead_times = self.params['forecast_lead_times']

        preds['lead_time'] = [lt * self.params['timedelta_hours'] for lt in lead_times]
        return preds
    

    def convert_to_xarray(self, surface_prediction, upper_air_prediction, start_times, params, valid_dataset, acc = True, diagnostic_prediction = None):
        batch_size, time_steps, num_surface_vars, lat, lon = surface_prediction.shape
        # print(f"TIME STEPS ARE: {time_steps}")
        datasets = []

        for sample in range(batch_size):
            # time_range = xr.cftime_range(
            #     start_time + timedelta(hours=params['timedelta_hours'] * sample * time_steps),
            #     periods=time_steps,
            #     freq=f"{params['timedelta_hours']}h"
            # )
            # time_range = [start_times[sample] + timedelta(hours=lt * params['timedelta_hours']) for lt in params['forecast_lead_times']]
            # time_range = [start_time + timedelta(hours=lt * params['timedelta_hours']) for lt in params['forecast_lead_times']]
            if acc:
            # For ACC, create time_range for all time steps
                # time_range = [start_times[sample] + timedelta(hours=step * params['timedelta_hours']) for step in range(time_steps)]
                time_range = [start_times[sample] + timedelta(hours=step * params['timedelta_hours']) for step in range(1, time_steps + 1)]
                # print(time_range)
            else:
            # For specific lead times, use forecast_lead_times
                time_range = [start_times[sample] + timedelta(hours=lt * params['timedelta_hours']) for lt in params['forecast_lead_times']]
                # print(time_range)
            

            # Determine the level coordinate name based on params.lev
            level_coord_name = 'lev' if params.lev == 'lev' else 'plev'

            coordinates = {
                'time': time_range,
                level_coord_name: valid_dataset.levels,
                'lat': self.params.lat,
                'lon': self.params.lon
            }

            dataset = xr.Dataset(
                coords=coordinates,
                attrs=dict(description=f"Prediction from {params.nettype} model run, sample {sample}")
            )

            for idx, var in enumerate(valid_dataset.surface_variables):
                da = xr.DataArray(
                    data=surface_prediction[sample, :, idx],
                    dims=["time", "lat", "lon"],
                    coords={'time': time_range,
                            'lat': dataset.lat.values,
                            'lon': dataset.lon.values}
                )
                #da = da.assign_attrs(valid_dataset.data_dss[0][var].attrs)
                dataset[var] = da

            if type(diagnostic_prediction) is not type(None):
                for idx, var in enumerate(valid_dataset.diagnostic_variables):
                    da = xr.DataArray(
                        data=diagnostic_prediction[sample, :, idx],
                        dims=["time", "lat", "lon"],
                        coords={'time': time_range,
                                'lat': dataset.lat.values,
                                'lon': dataset.lon.values}
                    )
                    #da = da.assign_attrs(valid_dataset.data_dss[0][var].attrs)
                    dataset[var] = da

            for idx, var in enumerate(valid_dataset.upper_air_variables):
                da = xr.DataArray(
                    data=upper_air_prediction[sample, :, idx],
                    dims=["time", level_coord_name, "lat", "lon"],
                    coords=coordinates
                )
                #da = da.assign_attrs(valid_dataset.data_dss[0][var].attrs)
                dataset[var] = da

            datasets.append(dataset)
        return datasets
    
    def combine_datasets(self, datasets):
        return xr.concat(datasets, dim='time')


    def plot_in_separate_process(self, avg_preds, avg_gt, preds_times, filename):
        # convert lead times to hours
        lead_times_hours = [step * self.params.timedelta_hours for step in self.params.forecast_lead_times]
                
        if len(preds_times) > 0:
            # Get variable dictionary from params 
            var_dict = self.params.diagnostic_spectrum_var_dict if hasattr(self.params, 'diagnostic_spectrum_var_dict') else None

            p = Process(target=plot_power_spectrum, args=(avg_preds, avg_gt, preds_times, filename, lead_times_hours, self.params.use_sigma_levels, var_dict))
        else:
            if hasattr(self.params, 'diagnostic_bias_var_dict'):
                var_dict = self.params.diagnostic_bias_var_dict
            elif 'mrso' in self.params.land_variables:
                vars = ["tas", "mrso", "zg", "ua", "hus"]
                levels = [None, None, 50000, 25000, 85000]
                var_dict = {var: [level] if level is not None else [] for var, level in zip(vars, levels)}
            else:
                # basic fallback
                var_dict = {"tas": [], "zg": [50000], "ua": [0.4368000030517578]}
                
            p = Process(target=plot_bias, args=(avg_preds, avg_gt, filename, var_dict))
            # if 'mrso' in self.params.land_variables:
            #     p = Process(target=plot_bias, args=(avg_preds, avg_gt, filename, ["tas", "mrso", "zg", "ua", "hus"], [None, None, 50000, 25000, 85000]))
            # else:
            #     p = Process(target=plot_bias, args=(avg_preds, avg_gt, filename))
        p.start()
        p.join()

    def log_all_plots_to_wandb(self):
        if self.params.log_to_wandb:
            output_dir = self.spectra_dir
            # # Create the directory if it doesn't exist
            # os.makedirs(output_dir, exist_ok=True)
            # print(f"Created directory: {output_dir}")
            plot_files = sorted([f for f in os.listdir(output_dir) if f.startswith("power_spectrum_epoch_")])
            
            print(f"Found plot files: {plot_files}")
            
            for plot_file in plot_files:
                # Extract epoch number from filename
                try:
                    epoch = int(plot_file.split("_")[-1].split(".")[0])
                    print(f"Processing file: {plot_file}, extracted epoch: {epoch}")
                except ValueError as e:
                    print(f"Error parsing epoch from filename {plot_file}: {e}")
                    continue
                
                # Log plot to wandb
                try:
                    wandb.log({
                        "power_spectrum_plot": wandb.Image(os.path.join(output_dir, plot_file)),
                        "custom_step": epoch,
                    }, step=self.wandb_step)
                    print(f"Logged file: {plot_file} with epoch: {epoch}")
                except Exception as e:
                    print(f"Error logging file {plot_file} to wandb: {e}")
            
            print(f"Logged {len(plot_files)} power spectrum plots to wandb")

            # Delete the directory after logging
            shutil.rmtree(output_dir)
            print(f"Deleted directory: {output_dir}")

    def print_acc(self, acc):
        print("\nACC Results:")
        
        # Define the variables and pressure levels you're interested in
        variables = ["2m_temperature", "temperature", "geopotential", "u_component_of_wind"]
        pressure_levels = [None, 850, 500, 250]  # in hPa, None for surface variables
        
        # Convert pressure levels to Pa for selection
        pressure_levels_pa = [p * 100 if p is not None else None for p in pressure_levels]
        
        for var, plev, plev_pa in zip(variables, pressure_levels, pressure_levels_pa):
            print(f"\nVariable: {var}" + (f" at {plev} hPa" if plev else " (Surface)"))
            
            if isinstance(acc['Pangu'], xr.DataArray):
                data = acc['Pangu'][var]
                if plev_pa and 'plev' in data.dims:
                    data = data.sel(plev=plev_pa, method='nearest')
                
                for lt in self.params.forecast_lead_times:
                    hours = lt * self.params.timedelta_hours
                    acc_value = data.sel(lead_time=lt).values
                    print(f"  Lead time {hours:2d}h (Step {lt:2d}): {acc_value:.4f}")
            else:
                print("  Unexpected data type for ACC score. Please check the output of evaluate_iterative_forecast.")
    

    def cleanup_acc_plots(self):
        if os.path.exists(self.params.acc_dir):
            shutil.rmtree(self.params.acc_dir)
            logging.info(f"Deleted ACC plots directory: {self.params.acc_dir}")

    def cleanup_power_spectrum_plots(self):
        if os.path.exists(self.params.spectra_dir):
            shutil.rmtree(self.params.spectra_dir)
            logging.info(f"Deleted Power Spectrum plots directory: {self.params.spectra_dir}")

    def cleanup_gifs(self):
        if os.path.exists(self.params.gif_dir):
            shutil.rmtree(self.params.gif_dir)
            logging.info(f"Deleted GIF directory: {self.params.gif_dir}")
    
    def cleanup_bias(self):
        if os.path.exists(self.params.bias_dir):
            shutil.rmtree(self.params.bias_dir)
            logging.info(f"Deleted Bias plots directory: {self.params.bias_dir}")

    def save_checkpoint(self, checkpoint_path, epoch=-1, model=None):
        """
        Save checkpoint with the following strategy:
        - Always save ckpt_latest.tar (overwritten each epoch)
        - Save ckpt_epoch_{N}.tar every checkpoint_save_interval epochs
        - Maintain at most max_checkpoints_to_keep numbered checkpoints
        - best_ckpt.tar is saved separately when validation improves
        """
        if not model:
            model = self.model

        checkpoint_data = {
            'iters': self.iters,
            'epoch': self.epoch,
            'model_state': model.module.state_dict() if dist.is_initialized() else model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'wandb_step': self.wandb_step
        }
        if hasattr(self, 'finetune_startEpoch'):
            checkpoint_data['finetune_startEpoch'] = self.finetune_startEpoch

        # Always save latest checkpoint
        torch.save(checkpoint_data, self.params.latest_checkpoint_path_save)
        logging.info(f"Saved latest checkpoint: {self.params.latest_checkpoint_path_save}")

        checkpoint_save_interval = 1
        if hasattr(self.params, 'checkpoint_save_interval'):
            checkpoint_save_interval = self.params.checkpoint_save_interval
        
        # Save numbered checkpoint at intervals
        if epoch >= 0 and (epoch + 1) % checkpoint_save_interval == 0:
            numbered_path = os.path.join(
                self.params.checkpoint_dir_save, 
                f'ckpt_epoch_{epoch}.tar'
            )
            torch.save(checkpoint_data, numbered_path)
            logging.info(f"Saved numbered checkpoint: {numbered_path}")
            
            # Clean up old numbered checkpoints
            self._cleanup_old_checkpoints()

        # Save best checkpoint (called separately from train loop)
        if checkpoint_path == self.params.best_checkpoint_path_save:
            torch.save(checkpoint_data, checkpoint_path)
            logging.info(f"Saved best checkpoint: {checkpoint_path}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old numbered checkpoints, keeping only the most recent N."""
        if hasattr(self.params, "max_checkpoints_to_keep"):
            checkpoint_paths = natsorted([
                file for file in glob.glob(self.params.checkpoint_path_globstr_save) 
                if os.path.isfile(file)
            ])
            
            num_to_delete = len(checkpoint_paths) - self.params.max_checkpoints_to_keep
            if num_to_delete > 0:
                for old_ckpt in checkpoint_paths[:num_to_delete]:
                    os.remove(old_ckpt)
                    logging.info(f"Removed old checkpoint: {old_ckpt}")


    def restore_checkpoint(self, checkpoint_path, finetune=False):
        """ We intentionally require a checkpoint_dir to be passed
            in order to allow Ray Tune to use this function """
        logging.info(f'Restoring from checkpoint: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.params.local_rank), weights_only=False)
        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except:
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                name = key[7:]
                new_state_dict[name] = val
            self.model.load_state_dict(new_state_dict)
        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch']
        self.epoch = checkpoint['epoch']
        # Restore wandb_step from checkpoint if available, otherwise use iters
        if 'wandb_step' in checkpoint:
            self.wandb_step = checkpoint['wandb_step']
            if self.world_rank == 0:
                logging.info(f"Restored wandb_step from checkpoint: {self.wandb_step} (iters: {self.iters})")
        else:
            self.wandb_step = self.iters
            if self.world_rank == 0:
                logging.info(f"Initialized wandb_step from iters (checkpoint had no wandb_step): {self.wandb_step}")
        
        # During fine-tuning, ensure wandb_step is at least as large as wandb's current step
        # to prevent monotonicity violations. This is critical because wandb might continue
        # from a previous run even with resume="never" if the run name is the same.
        if finetune and self.params.log_to_wandb:
            try:
                # Get wandb's current step if available
                if wandb.run is not None:
                    current_wandb_step = None
                    try:
                        # Try multiple methods to get wandb's current step
                        # Method 1: wandb.run.step (most direct)
                        if hasattr(wandb.run, 'step') and wandb.run.step is not None:
                            current_wandb_step = wandb.run.step
                        # Method 2: Try to get from summary (last logged step)
                        elif hasattr(wandb.run, 'summary') and '_step' in wandb.run.summary:
                            current_wandb_step = wandb.run.summary.get('_step', None)
                        # Method 3: Try to get from config
                        elif hasattr(wandb.run, 'config') and '_step' in wandb.run.config:
                            current_wandb_step = wandb.run.config.get('_step', None)
                    except Exception as e:
                        # If we can't get wandb's step, log a warning but continue
                        if self.world_rank == 0:
                            logging.warning(f"Could not determine wandb current step: {e}")
                    
                    # If we got a current step, ensure our wandb_step is larger
                    if current_wandb_step is not None:
                        if self.wandb_step <= current_wandb_step:
                            if self.world_rank == 0:
                                logging.info(f"Adjusting wandb_step from {self.wandb_step} to {current_wandb_step + 1} to maintain monotonicity")
                            self.wandb_step = current_wandb_step + 1
                    else:
                        # If we couldn't get wandb's step directly, try to query history
                        # This is a fallback for cases where wandb continues from a previous run
                        try:
                            if hasattr(wandb.run, 'history') and callable(wandb.run.history):
                                history_df = wandb.run.history()
                                if history_df is not None and not history_df.empty and '_step' in history_df.columns:
                                    max_step = int(history_df['_step'].max())
                                    if self.wandb_step <= max_step:
                                        if self.world_rank == 0:
                                            logging.info(f"Adjusting wandb_step from {self.wandb_step} to {max_step + 1} based on history")
                                        self.wandb_step = max_step + 1
                        except Exception as e:
                            # If history query fails, use a safety margin
                            if self.world_rank == 0:
                                logging.warning(f"Could not query wandb history: {e}. Using safety margin.")
                            if self.wandb_step < self.iters:
                                self.wandb_step = self.iters
            except Exception as e:
                # If we can't access wandb, ensure wandb_step is at least iters
                if self.world_rank == 0:
                    logging.warning(f"Error accessing wandb during checkpoint restoration: {e}")
                if self.wandb_step < self.iters:
                    self.wandb_step = self.iters
        
        # Print final wandb_step after all adjustments
        if self.world_rank == 0:
            logging.info(f"Final wandb_step after checkpoint restoration: {self.wandb_step} (finetune={finetune})")
        
        if finetune:
            self.finetune_startEpoch = self.startEpoch
            self.params.max_epochs = self.startEpoch + self.params.finetune_num_epochs
        print('START EPOCH:', self.startEpoch)
        # restore checkpoint is used for finetuning as well as resuming. If finetuning (i.e., not resuming), restore checkpoint does not load optimizer state, instead uses config specified lr.
        if self.params.resuming:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def seed_torch(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def setup_distributed(params: YParams, args: argparse.Namespace) -> tuple[int, int]:
    """Return (world_rank, local_rank)."""
    if params['world_size'] > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        seed = args.global_seed * dist.get_world_size() + rank
        params['global_batch_size'] = params['batch_size'] * params['world_size']
    else:
        rank = 0
        device = 0
        seed = args.global_seed
        params['global_batch_size'] = params['batch_size']
    
    seed_torch(seed)
    torch.cuda.set_device(device)
    return rank, device

if __name__ == '__main__':
    """
    Main training script entry point.
    This script handles argument parsing, configuration loading, distributed training setup,
    checkpoint management, and initiates the training or validation process.
    """
    
    # ============================================================================
    # SECTION 1: Parse command-line arguments
    # ============================================================================
    # Define all command-line arguments that control training behavior
    parser = argparse.ArgumentParser()
    
    # Core configuration arguments
    parser.add_argument("--run_num", default='0514', type=str, help="Unique identifier for this training run")
    parser.add_argument("--yaml_config", default='/project/pedramh/awikner/PanguWeather/v2.0/config/PANGU_PLASIM_H5_MIDWAY_0514.yaml', type=str, help="Path to YAML configuration file")
    parser.add_argument("--config", default='PLASIM', type=str, help="Configuration section name in YAML file")
    
    # Training control arguments
    parser.add_argument("--epsilon_factor", default=0, type=float, help="Epsilon factor for training")
    parser.add_argument("--epochs", default=0, type=int, help="Override max_epochs from config (0 means use config value)")
    parser.add_argument("--run_iter", default=1, type=int, help="Iteration number for this run")
    parser.add_argument("--debug", default=False, action='store_true', help="Enable debug mode (reduces data, disables wandb)")
    parser.add_argument("--no_amp", default=False, action='store_true', help="Disable automatic mixed precision training")
    parser.add_argument("--vae_loss", default=False, action='store_true', help="Use VAE loss function")
    parser.add_argument("--mode", default='train', type=str, choices=['train', 'test'], help="Execution mode: train or test")
    parser.add_argument("--test_iterations", default=30, type=int, help="Number of test iterations to run")
    parser.add_argument("--global_seed", type=int, default=0, help="Global seed for random number generators")
    
    # Checkpoint and validation arguments
    parser.add_argument("--fresh_start", default = False, action="store_true", help="Start training from scratch, ignoring existing checkpoints")
    parser.add_argument("--just_validate", default = False, action="store_true", help="Only run single epoch of validation")
    parser.add_argument("--validation_epochs", default="", type = str, help="List of epoch to validate when using just_validate. Comma separated list. If empty, validate best_ckpt.")
    parser.add_argument("--validate_before_train", default = False, action="store_true", help="Run validation before training when resuming from checkpoint or finetuning")
    
    # Model selection arguments
    parser.add_argument("--use_legacy_model", default=False, action='store_true', help="Use legacy model architecture")
    
    # Finetuning arguments
    parser.add_argument("--finetune_num_epochs", default=0, type=int, help="Number of epochs to finetune")
    parser.add_argument("--finetune_run_num", default=None, type=str, help="Run number to finetune from")
    parser.add_argument("--finetune_lr", default=0, type=float, help="Learning rate to finetune at")
    
    parser.add_argument("--train_data_sets_json", default=None, type=str, help="JSON file containing train data sets")
    parser.add_argument("--validation_data_sets_json", default=None, type=str, help="JSON file containing validation data sets")
    parser.add_argument("--start_epoch", default=None, type=int, help="Starting epoch to resume training from")
    
    # Ensemble forecast validation arguments
    parser.add_argument("--ensemble_validation", default=False, action='store_true', help="Enable ensemble forecast validation")
    parser.add_argument("--init_nc_filepath_files", default="", type=str, help="Path to JSON file containing input/target data info (event_type, data_path, start/end datetimes)")
    parser.add_argument("--ensemble_inference_hours", default="", type=str, help="Comma-separated list of total duration of ensemble predictions in hours")
    parser.add_argument("--obs_functions", default="", type=str, help="Comma-separated list of observable function names to use")
    parser.add_argument("--obs_args", default="", type=str, help="JSON string with additional arguments for each obs_function (e.g., target_duration, variable, region, path to regions file)")
    parser.add_argument("--error_metrics", default="", type=str, help="Comma-separated list of error metrics to compute for ensemble validation (e.g., rmse, lat_weighted_rmse, fair_crps, pearson_correlation)")
    parser.add_argument("--ensemble_batch_size", default=0, type=int, help="Batch size for ensemble validation (0 means use training batch_size)")

    ####### for UCAR
    parser.add_argument("--local-rank", type=int)
    #######
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    
    # ============================================================================
    # SECTION 3: Update parameters from command-line arguments
    # ============================================================================
    # Override or set parameters based on command-line arguments
    
    # Training precision and loss settings
    params['enable_amp'] = not args.no_amp  # Enable/disable automatic mixed precision
    params['vae_loss'] = args.vae_loss  # Use VAE loss if specified
    params['mode'] = args.mode  # Set execution mode (train/test)
    params['test_iterations'] = args.test_iterations  # Number of test iterations
    
    # Override max epochs if specified via command line
    if args.epochs > 0:
        params['max_epochs'] = args.epochs
    
    # Model and run settings
    params['run_iter'] = args.run_iter  # Run iteration number
    params['use_legacy_model'] = args.use_legacy_model  # Use legacy model architecture
    
    # Check if diagnostic variables are configured
    if hasattr(params, 'diagnostic_variables'):
        params['has_diagnostic'] = len(params.diagnostic_variables) > 0
    else:
        params['has_diagnostic'] = False
    
    print(f'Has diagnostic: {params.has_diagnostic}')
    
    # Set default number of ensemble members if not specified
    if not hasattr(params, 'num_ensemble_members'):
        params['num_ensemble_members'] = 1
    
    # Validation settings
    params['just_validate'] = args.just_validate  # Only validate, don't train
    if params.just_validate:
        os.environ["WANDB_MODE"] = "offline"  # Disable wandb for validation-only runs
    # Parse validation epochs from comma-separated string
    print("validation epochs arg:", args.validation_epochs)
    params['validation_epochs'] = sorted([int(i) for i in args.validation_epochs.split(',')]) if len(args.validation_epochs) > 0 else []
    params['validate_before_train'] = args.validate_before_train  # Validate before training when resuming/finetuning
    
    # Finetuning configuration
    if args.finetune_num_epochs > 0:
        params['finetune_num_epochs'] = args.finetune_num_epochs
    if args.finetune_run_num is not None:
        params['finetune_run_num'] = args.finetune_run_num
    # Validate finetuning arguments
    if params.finetune_num_epochs > 0 and params.finetune_run_num is None:
        raise ValueError("Finetuning epochs specified but finetuning run number is not specified")
    if args.finetune_lr > 0:
        params['lr'] = args.finetune_lr  # Override learning rate for finetuning
    
    # Load data set configurations from JSON files if provided
    if args.train_data_sets_json is not None:
        with open(args.train_data_sets_json, 'r') as f:
            params['train_data_sets'] = json.load(f)
    if args.validation_data_sets_json is not None:
        with open(args.validation_data_sets_json, 'r') as f:
            params['validation_data_sets'] = json.load(f)
    
    # Checkpoint selection
    if args.start_epoch is not None:
        params['start_epoch'] = args.start_epoch
    
    # Ensemble forecast validation configuration
    if not hasattr(params, 'ensemble_validation') or args.ensemble_validation:
        params['ensemble_validation'] = args.ensemble_validation
    if params.ensemble_validation:
        # Check for required parameters - can come from YAML or command line
        # Command line arguments override YAML values if provided
        
        # init_nc_filepath_files
        if len(args.init_nc_filepath_files) > 0:
            params['init_nc_filepath_files'] = args.init_nc_filepath_files
        elif not hasattr(params, 'init_nc_filepath_files') or (isinstance(params.init_nc_filepath_files, str) and len(params.init_nc_filepath_files) == 0):
            raise ValueError("ensemble_validation requires init_nc_filepath_files to be specified (either in YAML or via --init_nc_filepath_files)")
        # If not set from args, keep the YAML value (already in params)
        
        # ensemble_inference_hours
        if len(args.ensemble_inference_hours) > 0:
            params['ensemble_inference_hours'] = [int(h.strip()) for h in args.ensemble_inference_hours.split(',')]
        elif not hasattr(params, 'ensemble_inference_hours') or (isinstance(params.ensemble_inference_hours, str) and len(params.ensemble_inference_hours) == 0):
            raise ValueError("ensemble_validation requires ensemble_inference_hours to be specified (either in YAML or via --ensemble_inference_hours)")
        # If not set from args, keep the YAML value (already in params)
        
        # obs_functions
        if len(args.obs_functions) > 0:
            params['obs_functions'] = args.obs_functions
        elif not hasattr(params, 'obs_functions') or (isinstance(params.obs_functions, str) and len(params.obs_functions) == 0):
            raise ValueError("ensemble_validation requires obs_functions to be specified (either in YAML or via --obs_functions)")
        # If not set from args, keep the YAML value (already in params)
        
        # obs_args (optional - can be empty)
        # Parse obs_args as JSON string and store as obs_args_dict
        # If params already has obs_args_dict, assume it's correctly formatted as a dictionary
        if hasattr(params, 'obs_args_dict') and isinstance(params.obs_args_dict, dict):
            # Already correctly formatted as dictionary, use as-is
            pass
        else:
            obs_args_dict = {}
            if len(args.obs_args) > 0:
                # Parse from command line argument (JSON string)
                try:
                    obs_args_dict = json.loads(args.obs_args)
                except json.JSONDecodeError as e:
                    logging.warning(f"Could not parse obs_args JSON from command line: {e}")
            elif hasattr(params, 'obs_args') and isinstance(params.obs_args, str) and len(params.obs_args) > 0:
                # Parse from YAML config (JSON string) - for backward compatibility
                try:
                    obs_args_dict = json.loads(params.obs_args)
                except json.JSONDecodeError as e:
                    logging.warning(f"Could not parse obs_args JSON from config: {e}")
            params['obs_args_dict'] = obs_args_dict

        # error_metrics (optional - can be empty; command line overrides YAML)
        if hasattr(args, 'error_metrics') and len(args.error_metrics) > 0:
            params['error_metrics'] = args.error_metrics
        elif not hasattr(params, 'error_metrics'):
            params['error_metrics'] = ""
        
        # Set ensemble_batch_size (0 means use training batch_size)
        if args.ensemble_batch_size > 0:
            params['ensemble_batch_size'] = args.ensemble_batch_size
        elif not hasattr(params, 'ensemble_batch_size'):
            params['ensemble_batch_size'] = params['batch_size']
    
    # ============================================================================
    # SECTION 4: Debug mode configuration
    # ============================================================================
    # Configure debug mode settings (reduces data size, disables wandb, etc.)
    params['debug'] = False
    if args.debug:
        params['debug'] = True
        params['world_size'] = 1  # Single GPU for debugging
        os.environ['WANDB_MODE'] = 'offline'  # Disable wandb logging
        params['train_year_start'] = params['train_year_end'] - 1  # Reduce training data
        params['batch_size'] = params['num_data_workers']  # Match batch size to workers
        params['epochs_per_long_validation'] = 1  # Reduce validation frequency
        params['num_inferences'] = params['num_data_workers']  # Reduce inference count
    else:
        # Normal mode: determine world size from environment or available GPUs
        
        
        if 'WORLD_SIZE' in os.environ:
            print('World size from OS: %d' % int(os.environ['WORLD_SIZE']))
            params['world_size'] = int(os.environ['WORLD_SIZE'])
            print(params['world_size'])
        else: 
            params['world_size'] = torch.cuda.device_count()
            print(params['world_size'])
        print('World size from Cuda: %d' % torch.cuda.device_count())
    # Force wandb offline mode if configured
    if hasattr(params, "wandb_offline"):
        if params.wandb_offline:
            os.environ['WANDB_MODE'] = 'offline'

    # initialize DDP.
    world_rank, local_rank = setup_distributed(params, args)
    # ============================================================================
    # SECTION 5: Display GPU information and memory status
    # ============================================================================
    # Print GPU device information and current memory usage
    print('World size from OS: %d' % int(os.environ['WORLD_SIZE']))
    print('World size from Cuda: %d' % torch.cuda.device_count())
    
    # Check and display GPU memory status
    print(torch.cuda.get_device_name(0))
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    print(f"Memory Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
    
    # Set up directory structure
    if hasattr(params, 'finetune_run_num'):
        # Finetuning mode: save to finetune_run_num directory, load from original run_num
        save_expDir = os.path.join(params.exp_dir, args.config, str(params.finetune_run_num))
        load_expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    else:
        # Normal training: save and load from same directory
        save_expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
        load_expDir = save_expDir
    params['experiment_dir'] = os.path.abspath(save_expDir)
    
    # Define subdirectories
    params['checkpoint_dir_save'] = os.path.join(save_expDir, 'checkpoints')
    params['checkpoint_dir_load'] = os.path.join(load_expDir, 'checkpoints')
    params['plots_dir'] = os.path.join(save_expDir, 'plots')
    params['spectra_dir'] = os.path.join(params['plots_dir'], 'spectra')
    params['acc_dir'] = os.path.join(params['plots_dir'], 'acc')
    params['gif_dir'] = os.path.join(params['plots_dir'], 'gif')
    params['bias_dir'] = os.path.join(params['plots_dir'], 'bias')
    params['validation_data_dir'] = os.path.join(save_expDir, 'validation_data')
    
    # Checkpoint paths
    params['checkpoint_path_globstr_save'] = os.path.join(params['checkpoint_dir_save'], 'ckpt_epoch_*.tar')
    params['checkpoint_path_globstr_load'] = os.path.join(params['checkpoint_dir_load'], 'ckpt_epoch_*.tar')
    params['best_checkpoint_path_save'] = os.path.join(params['checkpoint_dir_save'], 'best_ckpt.tar')
    params['best_checkpoint_path_load'] = os.path.join(params['checkpoint_dir_load'], 'best_ckpt.tar')
    params['latest_checkpoint_path_save'] = os.path.join(params['checkpoint_dir_save'], 'ckpt_latest.tar')
    params['latest_checkpoint_path_load'] = os.path.join(params['checkpoint_dir_load'], 'ckpt_latest.tar')
    
    # Set default config values if not specified
    if not hasattr(params, 'checkpoint_save_interval'):
        params['checkpoint_save_interval'] = 10
    if not hasattr(params, 'max_checkpoints_to_keep'):
        params['max_checkpoints_to_keep'] = 5
    

    torch.manual_seed(world_rank)
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

    if params['validation_epochs']:
        if world_rank == 0:
            print(f"Validation epochs specified: {params['validation_epochs']}")

    if world_rank == 0:
        os.makedirs(params['experiment_dir'], exist_ok=True)
        os.makedirs(params['checkpoint_dir_save'], exist_ok=True)
        os.makedirs(params['spectra_dir'], exist_ok=True)
        os.makedirs(params['acc_dir'], exist_ok=True)
        os.makedirs(params['gif_dir'], exist_ok=True)
        if params.long_validation:
            os.makedirs(params['bias_dir'], exist_ok=True)
        os.makedirs(params['validation_data_dir'], exist_ok=True)

    checkpoint_paths = [file for file in glob.glob(params.checkpoint_path_globstr_load) if os.path.isfile(file)]
    best_checkpoint_exists = os.path.isfile(params['best_checkpoint_path_load'])
    latest_checkpoint_exists = False
    if hasattr(params, 'latest_checkpoint_path'):
        latest_checkpoint_exists = os.path.isfile(params['latest_checkpoint_path'])
        if latest_checkpoint_exists:
            params['latest_checkpoint_path_load'] = params['latest_checkpoint_path']
    params['latest_checkpoint_path_save'] = os.path.join(params['checkpoint_dir_save'], 'ckpt_latest.tar')
    checkpoint_exists = len(checkpoint_paths) > 0 or best_checkpoint_exists or latest_checkpoint_exists
    
    # Determine whether to resume from checkpoint or start fresh
    # Note: If finetuning is enabled, fresh_start is ignored (finetuning always requires loading a checkpoint)
    params.finetuning = hasattr(params, 'finetune_run_num')
    
    if params.finetuning:
        # Finetuning mode: always try to load checkpoint from original run (fresh_start is ignored)
        # Note: resuming is set to False for finetuning so optimizer state is not loaded
        if checkpoint_exists:
            params['resuming'] = False  # Don't load optimizer state for finetuning
            params['finetuning'] = True  # Flag to indicate finetuning mode
            if world_rank == 0:
                logging.info("Finetuning mode: Loading checkpoint from original run (fresh_start ignored, optimizer state will not be loaded).")
        else:
            raise FileNotFoundError(
                f"No checkpoint found for finetuning.\n"
                f"Searched: {params.checkpoint_path_globstr_load}, {params.best_checkpoint_path_load}, {params.latest_checkpoint_path_load}"
            )
    elif params.just_validate:
        # Validation requires a trained model - must load checkpoint
        if not checkpoint_exists:
            raise FileNotFoundError(
                "just_validate=True but no checkpoint found. "
                f"Searched: {params.checkpoint_path_globstr_load}, {params.best_checkpoint_path_load}"
            )
        params['resuming'] = True
        params['finetuning'] = False
        if world_rank == 0:
            logging.info("Validation mode: will load checkpoint.")
    elif params.fresh_start or args.fresh_start:
        # Normal training mode: use fresh_start flag to determine whether to start fresh or resume
        
        params['resuming'] = False
        params['finetuning'] = False
        if checkpoint_exists and world_rank == 0:
            logging.info("Fresh start requested. Ignoring existing checkpoint.")
    elif checkpoint_exists:
        # Resume: checkpoint found, will resume training
        params['resuming'] = True
        params['finetuning'] = False
        if world_rank == 0:
            logging.info("Resuming from existing checkpoint.")
    else:
        # No checkpoint: start fresh training
        params['resuming'] = False
        params['finetuning'] = False
        if world_rank == 0:
            logging.info("No checkpoint found. Starting fresh training run.")


    # ============================================================================
    # SECTION 11: Final parameter setup
    # ============================================================================
    # Store local rank in params for use in Trainer
    params['local_rank'] = local_rank
    
    # Display which precision engine is being used
    if params['use_transformer_engine']:
        print("Using Transformer Engine")
    else:
        print("Using PyTorch native")
    
    # ============================================================================
    # SECTION 12: Setup logging and save hyperparameters
    # ============================================================================
    # Setup file logging (only on rank 0 to avoid duplicate logs)
    if world_rank == 0:
        log_file = 'out.log'
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(save_expDir, log_file))
        logging_utils.log_versions()  # Log versions of key libraries
        params.log()  # Log all parameters
    
    # Configure logging flags (only rank 0 logs to wandb and screen)
    params['log_to_wandb'] = (world_rank == 0) and params['log_to_wandb']
    params['log_to_screen'] = (world_rank == 0) and params['log_to_screen']
    
    # Save hyperparameters to YAML file for reproducibility (only on rank 0)
    if world_rank == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(save_expDir, 'hyperparams.yaml'), 'w') as hpfile:
            yaml.dump(hparams,  hpfile)
    
    # ============================================================================
    # SECTION 13: Initialize Trainer and configure diagnostics
    # ============================================================================
    # Create Trainer instance (this loads datasets, initializes model, etc.)
    trainer = Trainer(params, world_rank)
    
    # Configure diagnostic GIF settings if enabled
    if params.diagnostic_gif:
        if not hasattr(params, "diagnostic_gif_var_dict"):
            # Default diagnostic variable for GIF: geopotential height at 500 hPa
            params['diagnostic_gif_var_dict'] = {'zg': [50000]}
    
    # Disable certain diagnostics for sigma-level training
    if hasattr(params, 'use_sigma_levels'):
        if params.use_sigma_levels:
            print('For sigma level training, disabling diagnostic ACC and diagnostic spectra')
            params['diagnostic_acc'] = False
            params['diagnostic_spectra'] = False
            params['diagnostic_gif'] = False
    
    # ============================================================================
    # SECTION 14: Setup model and start training/validation
    # ============================================================================
    # Initialize the model (load weights, setup optimizer, etc.)
    trainer.setup_model()

    with open(f'params_train_{args.run_num}.pkl', 'wb') as f:
        pickle.dump(params, f)
    
    # Run validation before training if requested and resuming/finetuning
    if not params.just_validate and params.validate_before_train:
        if params.resuming or params.finetuning:
            if world_rank == 0:
                logging.info("Running validation before training (resuming/finetuning mode)...")
            trainer.validate_one_epoch()
            # Run ensemble validation if enabled
            if hasattr(params, 'ensemble_validation') and params.ensemble_validation:
                ensemble_val_time = trainer.validate_ensemble_forecast()
                if world_rank == 0:
                    logging.info(f"Pre-training ensemble validation time: {ensemble_val_time:.2f} seconds")
    
    # Execute training or validation based on mode
    if not params.just_validate:
        # Normal training mode: run full training loop
        trainer.train()
    else:
        # Validation-only mode: run validation without training
        if len(params.validation_epochs) == 0:
            # No specific epochs specified: validate best checkpoint
            trainer.validate_one_epoch()
            # Run ensemble validation if enabled
            if hasattr(params, 'ensemble_validation') and params.ensemble_validation:
                ensemble_val_time = trainer.validate_ensemble_forecast()
                logging.info(f"Ensemble validation time: {ensemble_val_time:.2f} seconds")
        else:
            # Validate specific epochs: load each checkpoint and validate
            for ckpt_i in params.validation_epochs:
                print(f'Validating epoch {ckpt_i}...')
                # Construct checkpoint path by replacing wildcard with epoch number
                ckpt_path = params.checkpoint_path_globstr_load.replace('*', str(ckpt_i))
                if not os.path.isfile(ckpt_path):
                    logging.warning(f"Checkpoint not found: {ckpt_path}, skipping epoch {ckpt_i}")
                    continue
                trainer.restore_checkpoint(ckpt_path)
                trainer.epoch = trainer.startEpoch
                # Run validation for this checkpoint
                trainer.validate_one_epoch()
                # Run ensemble validation if enabled (for each epoch)
                if hasattr(params, 'ensemble_validation') and params.ensemble_validation:
                    ensemble_val_time = trainer.validate_ensemble_forecast()
                    logging.info(f"Epoch {ckpt_i} ensemble validation time: {ensemble_val_time:.2f} seconds")
    
    # Training/validation complete
    logging.info('DONE ---- rank %d' % world_rank)
