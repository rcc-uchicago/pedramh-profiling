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
from utils.data_loader_multifiles import get_data_loader, get_date_range, datetime_class_from_calendar
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
        self.epoch = self.startEpoch
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
        if self.params.resuming:
            checkpoint_path = None
            finetune = False
            if hasattr(self.params, 'start_epoch'):
                checkpoint_path = [file for file in glob.glob(self.params.checkpoint_path_globstr_load) \
                    if os.path.isfile(file) and f'_{params.start_epoch}.tar' in file][0]
            elif hasattr(self.params, 'finetuning'):
                if self.params.finetuning:
                    # Finetuning mode: load checkpoint but don't load optimizer state
                    finetune = True
                    if hasattr(self.params, 'start_epoch'):
                        checkpoint_paths = [file for file in glob.glob(self.params.checkpoint_path_globstr_load) if os.path.isfile(file)]
                        print(checkpoint_paths)
                        print(f'_{self.params.start_epoch}.tar')
                        checkpoint_path = [file for file in checkpoint_paths if f'_{self.params.start_epoch}.tar' in file][0]
                    else:
                        checkpoint_path = natsorted([file for file in glob.glob(self.params.checkpoint_path_globstr_load) if os.path.isfile(file)])[-1]
            else:
                # For validation without specific epochs, prefer best checkpoint
                if self.params.just_validate and len(self.params.validation_epochs) == 0:
                    if os.path.isfile(self.params.best_checkpoint_path_load):
                        checkpoint_path = self.params.best_checkpoint_path_load
                        logging.info("Validation mode: using best checkpoint")
                # For training resume: priority is latest > numbered > best
                if checkpoint_path is None:
                    load_latest = False
                    if hasattr(self.params, 'latest_checkpoint_path_load'):
                        if os.path.isfile(self.params.latest_checkpoint_path_load):
                            checkpoint_path = self.params.latest_checkpoint_path_load
                            load_latest = True
                    if not load_latest:
                        checkpoint_paths = natsorted([
                            file for file in glob.glob(self.params.checkpoint_path_globstr_load) 
                            if os.path.isfile(file)
                        ])
                        if len(checkpoint_paths) > 0:
                            checkpoint_path = checkpoint_paths[-1]
                        elif os.path.isfile(self.params.best_checkpoint_path):
                            checkpoint_path = self.params.best_checkpoint_path
            if checkpoint_path is None:
                raise FileNotFoundError(
                    f"No checkpoint files found for resuming.\n"
                    f"Searched: {self.params.checkpoint_path_globstr}, "
                    f"{self.params.latest_checkpoint_path}, {self.params.best_checkpoint_path}"
                )
            self.restore_checkpoint(checkpoint_path, finetune = finetune)
            if finetune:
                logging.info("Finetuning: Loaded checkpoint from %s (optimizer state not loaded)", checkpoint_path)
            else:
                logging.info("Resuming from checkpoint: %s", params.checkpoint_path)
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
            
        self.ensemble_validation = False
        if hasattr(params, 'ensemble_validation'):
            self.ensemble_validation = params.ensemble_validation
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
                                                                                train=False, single_ic=True, ensemble=self.ensemble_validation)
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
                if self.long_validation and not self.ensemble_validation:
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
        if self.params.nettype == 'pangu_plasim':
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
        self.lr = (2 ** params.loglr) * params["global_batch_size"] / 16.0
        if params.optimizer_type == 'FusedAdam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=params.weight_decay, fused=True)
        elif params.optimizer_type == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=params.weight_decay, fused=True)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=params.weight_decay)

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
                logging.info(f"Shuffled training data len: {self.train_datasets[0].num_inferences}")

            start = time.time()
            tr_time, data_time, train_logs = self.train_one_epoch()
            logging.info(f"Epoch {epoch + 1} training time: {tr_time:.2f} seconds, data loading time: {data_time:.2f} seconds")
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
                                                                    varying_boundary_data, input_upper_air, 
                                                                    target_surface, target_upper_air,train = True)
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

        sample_idx = np.random.randint(len(self.valid_data_loader))

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
                if self.ensemble_validation:
                    val_data_dir = self.params.validation_data_dir
                    print(val_data_dir)
                    os.makedirs(val_data_dir, exist_ok=True)
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
                        val_output_surface, val_output_upper_air, val_output_diagnostic, _, _ = self.model(
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
                    if self.ensemble_validation:
                        val_surface_numpy = val_output_surface.cpu().numpy()
                        np.save(os.path.join(val_data_dir, f'surface_{int(year.item())}_{i:04}.npy'), val_surface_numpy)
                        val_upper_air_numpy = val_output_upper_air.cpu().numpy()
                        np.save(os.path.join(val_data_dir, f'upper_air_{int(year.item())}_{i:04}.npy'), val_upper_air_numpy)
                        if self.params.has_diagnostic:
                            val_diagnostic_numpy = val_output_diagnostic.cpu().numpy()
                            np.save(os.path.join(val_data_dir, f'diagnostic_{int(year.item())}_{i:04}.npy'), val_diagnostic_numpy)
                    else:
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
                if no_nans and not self.ensemble_validation:
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
        torch.save(checkpoint_data, self.params.latest_checkpoint_path)
        logging.info(f"Saved latest checkpoint: {self.params.latest_checkpoint_path}")

        checkpoint_save_interval = 1
        if hasattr(self.params, 'checkpoint_save_interval'):
            checkpoint_save_interval = self.params.checkpoint_save_interval
        
        # Save numbered checkpoint at intervals
        if epoch >= 0 and (epoch + 1) % checkpoint_save_interval == 0:
            numbered_path = os.path.join(
                self.params.checkpoint_dir, 
                f'ckpt_epoch_{epoch}.tar'
            )
            torch.save(checkpoint_data, numbered_path)
            logging.info(f"Saved numbered checkpoint: {numbered_path}")
            
            # Clean up old numbered checkpoints
            self._cleanup_old_checkpoints()

        # Save best checkpoint (called separately from train loop)
        if checkpoint_path == self.params.best_checkpoint_path:
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
    
    # Model selection arguments
    parser.add_argument("--use_legacy_model", default=False, action='store_true', help="Use legacy model architecture")
    
    # Finetuning arguments
    parser.add_argument("--finetune_num_epochs", default=0, type=int, help="Number of epochs to finetune")
    parser.add_argument("--finetune_run_num", default=None, type=str, help="Run number to finetune from")
    parser.add_argument("--finetune_lr", default=0, type=float, help="Learning rate to finetune at")
    
    parser.add_argument("--train_data_sets_json", default=None, type=str, help="JSON file containing train data sets")
    parser.add_argument("--validation_data_sets_json", default=None, type=str, help="JSON file containing validation data sets")
    parser.add_argument("--start_epoch", default=None, type=int, help="Starting epoch to resume training from")

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
    params['validation_epochs'] = sorted([int(i) for i in args.validation_epochs.split(',')]) if len(args.validation_epochs) > 0 else []
    
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
    if hasattr(params, 'latest_checkpoint_path'):
        latest_checkpoint_exists = os.path.isfile(params['latest_checkpoint_path'])
        if latest_checkpoint_exists:
            params['latest_checkpoint_path_load'] = params['latest_checkpoint_path']
    checkpoint_exists = len(checkpoint_paths) > 0 or best_checkpoint_exists or latest_checkpoint_exists

    # Determine whether to resume from checkpoint or start fresh
    # Note: If finetuning is enabled, fresh_start is ignored (finetuning always requires loading a checkpoint)
    is_finetuning = hasattr(params, 'finetune_run_num')

    if is_finetuning:
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
                f"Searched: {params.checkpoint_path_globstr}, {params.best_checkpoint_path}"
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
    
    # Execute training or validation based on mode
    if not params.just_validate:
        # Normal training mode: run full training loop
        trainer.train()
    else:
        # Validation-only mode: run validation without training
        if len(params.validation_epochs) == 0:
            # No specific epochs specified: validate best checkpoint
            trainer.validate_one_epoch()
        else:
            # Validate specific epochs: load each checkpoint and validate
            for ckpt_i in params.validation_epochs:
                print(f'Validating epoch {ckpt_i}...')
                # Construct checkpoint path by replacing wildcard with epoch number
                ckpt_path = params.checkpoint_path_globstr.replace('*', str(ckpt_i))
                if not os.path.isfile(ckpt_path):
                    logging.warning(f"Checkpoint not found: {ckpt_path}, skipping epoch {ckpt_i}")
                    continue
                trainer.restore_checkpoint(ckpt_path)
                trainer.epoch = trainer.startEpoch
                # Run validation for this checkpoint
                trainer.validate_one_epoch()
    
    # Training/validation complete
    logging.info('DONE ---- rank %d' % world_rank)
