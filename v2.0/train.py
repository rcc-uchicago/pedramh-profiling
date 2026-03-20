import sys, os, threading
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from networks.pangu import PanguModel_Plasim
from networks.pangu_legacy import PanguModel_Plasim as PanguModel_Plasim_Legacy
from networks.modulus_sfno.sfnonet import SphericalFourierNeuralOperatorNet_v2 as SFNO
from tqdm import tqdm
from pathlib import Path
from datetime import timedelta, datetime
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
from collections import OrderedDict
from copy import deepcopy
import multiprocessing
from multiprocessing import Process
import concurrent.futures
from itertools import product
import matplotlib.pyplot as plt

import pandas as pd
import wandb
import glob
import time
import os
import uuid
import copy
import json
import shutil
import traceback
import cftime
import warnings

from natsort import natsorted
import numpy as np
import argparse
import xarray as xr
import logging
import torch
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.profiler import profile, record_function, ProfilerActivity
from utils import logging_utils
from utils.power_spectrum import (
    zonal_averaged_power_spectrum,
    plot_bias,
    plot_power_spectrum,
    make_gif
)
from utils.data_loader_multifiles import get_data_loader,\
    get_date_range, datetime_class_from_calendar, create_dataloader, \
    shuffle_balanced_dates
from utils.YParams import YParams
from utils.metrics import create_metrics_aggregator_new
from utils.perturbation import Perturber
from utils.integrate import Integrator, forward_euler

from utils.losses import Latitude_weighted_MSELoss, Latitude_weighted_L1Loss, Masked_L1Loss,\
     Masked_MSELoss, Latitude_weighted_masked_L1Loss, Latitude_weighted_masked_MSELoss,\
     Latitude_weighted_CRPSLoss, Kl_divergence_gaussians
from utils.lr_scheduler_sfno import LinearWarmupCosineAnnealingLR
from ensemble_inference import Stepper
logging_utils.config_logger()

torch._dynamo.config.optimize_ddp = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
torch.cuda.empty_cache()           
logging.info("Torch version: {}".format(torch.__version__))

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

#@torch.jit.script
def latitude_weighting_factor_torch(latitudes):
    lat_weights_unweighted = torch.cos(3.1416/180. * latitudes)
    return latitudes.size()[0] * lat_weights_unweighted/torch.sum(lat_weights_unweighted)

#@torch.jit.script
def weighted_rmse_torch_channels(pred, target, latitudes):
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    weight = torch.reshape(latitude_weighting_factor_torch(latitudes), (1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1,-2)))
    return result

#@torch.jit.script
def weighted_rmse_torch_3D(pred, target, latitudes):
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

def make_grad_contiguous_hook(grad):
    """Hook to ensure gradients are contiguous for DDP."""
    if grad is not None and not grad.is_contiguous():
        return grad.contiguous()
    return grad

def to_ensemble_batch(data, ens_members):
    """Convert batch of M samples (M, ...) to a batch of (M*ens_members, ...).

    Each input sample is repeated consecutively ens_members times, so the output
    ordering is [s0, s0, ..., s1, s1, ...] matching the original flatten(0,1) behaviour.
    Uses repeat_interleave to keep all work on the GPU with no auxiliary tensor creation
    or CPU/GPU data transfers.
    """
    return data.repeat_interleave(ens_members, dim=0)


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
        # Track if this is the first call to validate_ensemble_forecast in this run
        self._first_ensemble_validation = True
        
    def setup_model(self):
        # Set up model
        self.mask_bool, self.land_mask = self.get_land_mask_bool() #Bing: need to double check if the return is static values.
        self.model = self.get_model()

        if getattr(self.params, 'use_ema', False):
            logging.info(f"Rank {self.params.local_rank} Loaded EMA with decay = {self.params.ema_decay}")
            # Note that parameter initialization is done within the DiT constructor
            self.ema = deepcopy(self.model).to(self.device)  # Create an EMA of the model for use after training
            requires_grad(self.ema, False)
            update_ema(self.ema, self.model, decay=0)  # Ensure EMA is initialized with synced weights
        else:
            self.ema = None

        # Warn about incompatible configurations
        if self.params.use_legacy_model and self.params.vae_loss:
            logging.warning("Legacy model does not support VAE dual encoder architecture. "
                            "VAE loss will be ignored during training. "
                            "Set --vae_loss=False or use non-legacy model.")
        
        if self.params.enable_amp == True:
            self.scaler = GradScaler()
        
        if dist.is_initialized():
            # Make all parameters contiguous to avoid stride warnings
            for param in self.model.parameters():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()
            
            # Register hooks to make gradients contiguous (fixes stride mismatch)
            for param in self.model.parameters():
                if param.requires_grad:
                    param.register_hook(make_grad_contiguous_hook)

            self.model = DistributedDataParallel(self.model,
                                                device_ids=[self.params.local_rank],
                                                output_device=self.params.local_rank,  # Remove the list brackets
                                                find_unused_parameters=True,
                                                gradient_as_bucket_view=True,  # More memory efficient
            )
        
        self.optimizer = self.get_optimizer()
        self.setup_scheduler()
        self.loss_obj_pl,self.loss_obj_sfc, self.loss_obj_diagnostic = self.setup_loss_fun()
        self.latitudes = torch.from_numpy(np.array(self.params.lat)).to(self.device, non_blocking=True)

        # When just_validate with specific validation_epochs, skip initial checkpoint load.
        # The validation loop (lines below) calls restore_checkpoint for each epoch directly.
        _just_validate_specific_epochs = (
            getattr(self.params, 'just_validate', False)
            and len(getattr(self.params, 'validation_epochs', [])) > 0
        )

        if _just_validate_specific_epochs:
            if self.world_rank == 0:
                logging.info(
                    "just_validate with specific validation_epochs: skipping initial checkpoint load; "
                    "each epoch's checkpoint will be loaded in the validation loop."
                )
        elif self.params.resuming or self.params.finetuning:
            checkpoint_path = None
            finetune = self.params.finetuning

            # Priority 1: If start_epoch is assigned, load from that epoch's checkpoint file.
            # Skipped for just_validate: the per-epoch loop or the best-checkpoint path
            # handles checkpoint selection; start_epoch is a curriculum/finetune config
            # field and should not redirect validation to a specific epoch.
            if hasattr(self.params, 'start_epoch') and self.params.start_epoch is not None \
                    and not getattr(self.params, 'just_validate', False):
                epoch_checkpoint_pattern = os.path.join(
                    self.params.checkpoint_dir_load,
                    f'ckpt_epoch_{self.params.start_epoch}.tar'
                )
                if os.path.isfile(epoch_checkpoint_pattern):
                    checkpoint_path = epoch_checkpoint_pattern
                    if self.world_rank == 0:
                        logging.info(f"Loading checkpoint from specified epoch {self.params.start_epoch}: {checkpoint_path}")
                else:
                    # Do not fall back to other checkpoints when start_epoch is explicit (e.g. validate_before_train).
                    # Load dir is from run_num; use the run that actually contains this epoch (e.g. base run for finetuning).
                    raise FileNotFoundError(
                        f"Checkpoint for start_epoch={self.params.start_epoch} not found at {epoch_checkpoint_pattern}. "
                        f"checkpoint_dir_load is set from the run directory. For validate_before_train or finetuning, "
                        f"pass --run_num to the run that contains the checkpoint (e.g. the base run 260207), not the "
                        f"finetune run number."
                    )
            
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
        
        #Logging
        if self.params.log_to_wandb:
            # wandb.watch(self.model)
            wandb.watch(self.model, log="parameters", log_freq=1000)
        '''if params.log_to_screen: logging.info(self.model)'''
        if params.log_to_screen:
            logging.info("Number of trainable model parameters: {}".format(self.count_parameters()))
        self.epoch = self.startEpoch
        

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
            _partition_date_ranges = getattr(self.params, 'train_date_range', None)
            self.data_sizes = get_date_range([], self.params.train_data_sets, self.params.data_timedelta_hours,
                                            self.params.calendar, self.params.has_year_zero,
                                            datetime_class_from_calendar(self.params.calendar), get_size = True,
                                            partition_date_ranges = _partition_date_ranges)
                
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
        
        self.constant_boundary_data = self.train_datasets[0].constant_boundary_data.unsqueeze(0) * torch.ones(params.batch_size, 1, 1, 1)
        self.constant_boundary_data = self.constant_boundary_data.to(self.device, non_blocking=True)
        if params.num_ensemble_members > 1:
            self.constant_boundary_data = to_ensemble_batch(self.constant_boundary_data, params.num_ensemble_members)
            logging.info(f'Ensemble Mode. Ensemble size = {params.num_ensemble_members}\n')

         # Load climatology
        climatology_path = os.path.join(params.data_dir, self.params.climatology_file)
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        self.climatology = xr.open_dataset(climatology_path, decode_times=time_coder)
        if 'time_bnds' in self.climatology.variables:
            self.climatology = self.climatology.drop_vars('time_bnds')
        self.climatology = self.climatology.astype({var: np.float32 for var in self.climatology.data_vars})
        self.climatology = self.climatology.rename({'time':'dayofyear'})
        if world_rank == 0:
            logging.info('rank %d, data loader initialized' % self.world_rank)

    def init_wandb(self, params: dict):
        """
        Initialise wandb, setup metrics to log
        """
        if params.log_to_wandb:
            resume_mode = "allow" if params.resuming else "never"
            logging.info(f"WandB resume mode: {resume_mode}")
            wandb.init(config=params, 
                    name=f'{params.name}-{params.run_iter}', 
                    entity=params.entity, 
                    group=params.group, 
                    project=params.project, 
                    resume=resume_mode,
                    settings=wandb.Settings(
                            init_timeout=300,
                            _disable_stats=True,
                        )
                    )
            logging.info("WandB initialized with config: %s", params)

            wandb.define_metric("epoch")
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
            if params.sync_norm and dist.is_initialized():
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
            self.epoch = epoch
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
            # valid_time, valid_logs = self.validate_one_epoch()  # <-- Debug validation loop
            start = time.time()

            self.model.train()
            if self._ema_active():
                self.ema.eval()
            elif self.ema is not None:
                # First epoch where EMA becomes active: sync EMA weights from model
                warmup = getattr(self.params, 'ema_warmup_epochs', 0)
                if self.epoch == warmup:
                    model_for_ema = self.model.module if hasattr(self.model, 'module') else self.model
                    update_ema(self.ema, model_for_ema, decay=0)
                    logging.info(f"EMA warmup complete at epoch {self.epoch}. Synced EMA from model weights.")

            if self.params.curriculum_learning:
                # Generate shuffled indices on rank 0
                if self.world_rank == 0:
                    if self.params.balanced_learning:
                        shuffled_date_idxs = torch.tensor(
                            shuffle_balanced_dates(self.params,
                                                   self.generator,
                                                   self.train_datasets[0].all_dates,
                                                   self.data_sizes,
                                                   self.train_datasets[0].start_date,
                                                   self.train_datasets[0].has_year_zero,
                                                   self.train_datasets[0].datetime_class),
                            device = self.device)
                    else:
                        shuffled_date_idxs = torch.tensor(self.generator.permutation(self.data_sizes[0]), device=self.device)
                
                # Broadcast to all ranks if using distributed training
                if dist.is_initialized():
                    dist.barrier(device_ids=[self.device])
                    if self.world_rank != 0:
                        # Initialize tensor on non-root ranks before broadcast.
                        # balanced_learning returns sum(data_sizes[1:]) indices;
                        # standard curriculum returns data_sizes[0] indices.
                        if getattr(self.params, 'balanced_learning', False):
                            shuffled_size = sum(self.data_sizes[1:])
                        else:
                            shuffled_size = self.data_sizes[0]
                        shuffled_date_idxs = torch.zeros(shuffled_size, dtype=torch.long, device=self.device)
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
            # Run ensemble forecast validation if enabled, subject to frequency gate.
            ensemble_val_time = 0.0
            if hasattr(self.params, 'ensemble_validation') and self.params.ensemble_validation:
                _ens_freq = getattr(self.params, 'ensemble_validation_frequency', 1)
                _epochs_since_start = epoch - self.startEpoch + 1
                if _epochs_since_start % _ens_freq == 0:
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
                    break
            
            # Early stopping logic should be outside of world_rank check
            # Check if validation improved BEFORE updating best_valid_loss
            is_best = valid_logs['valid_loss'] <= best_valid_loss
            
            if is_best:
                best_valid_loss = valid_logs['valid_loss']
                early_stopping_counter = 0  # Reset the counter
            else:
                early_stopping_counter += 1  # Increment the counter

            self.model.eval() # important! This disables randomized embedding dropout
            
            if self.world_rank == 0:
                if self.params.save_checkpoint:
                    # checkpoint at the end of every epoch
                    self.save_checkpoint(self.params.checkpoint_path_globstr_save, self.epoch)

                    # Save best checkpoint only if validation improved
                    if is_best:
                        self.save_checkpoint(self.params.best_checkpoint_path_save)
                        logging.info(f'Find best checkpoint at {epoch}.')
            if self.params.log_to_wandb and self.world_rank == 0:
                self.log_wandb_epoch(epoch)
                self.log_screen_epoch(epoch, start, train_logs, valid_logs, early_stopping_counter)
            # Early stopping check
            if self.params.early_stopping and early_stopping_counter >= self.params.early_stopping_patience:
                if self.params.log_to_screen and world_rank == 0:
                    logging.info('Early stopping triggered. Terminating training.')
                break # Exit the train method
            
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

        self.model.train()

        pbar = tqdm(total=total_iterations, bar_format='{l_bar}{bar:30}{r_bar}',
                    dynamic_ncols=True, file=logging_utils.tqdm_stream,
                    disable=(self.world_rank != 0))
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
                        # Unscale gradients before clipping
                        self.scaler.unscale_(self.optimizer)
                        # Clip gradients to prevent explosion
                        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        # Clip gradients to prevent explosion
                        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                        self.optimizer.step()
                    if self._ema_active():
                        if dist.is_initialized():
                            update_ema(self.ema, self.model.module, decay=self.params.ema_decay)
                        else:
                            update_ema(self.ema, self.model, decay=self.params.ema_decay)
                    tr_end_time = time.time()
                    if self.params.mode == "test":
                        logging.info(f"Backpropagation and optimizer step took {tr_end_time - tr_start:.4f} seconds/ iteration")
                    if self.params.scheduler in ['OneCycleLR', 'LinearWarmupCosineAnnealingLR']:
                        self.scheduler.step()

                    with torch.no_grad():

                        if self.params.predict_delta:
                            output_surface, output_upper_air = self.integrator(input_surface, input_upper_air, output_surface, output_upper_air)
                            target_surface, target_upper_air = self.integrator(input_surface, input_upper_air, target_surface, target_upper_air)

                        surface_lwrmse = weighted_rmse_torch_channels(output_surface, target_surface, self.latitudes)
                        upper_air_lwrmse = weighted_rmse_torch_3D(output_upper_air, target_upper_air, self.latitudes)

                        if self.params.has_diagnostic:
                            diagnostic_lwrmse = weighted_rmse_torch_channels(output_diagnostic, target_diagnostic, self.latitudes)
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
                                                                        train_batch_loss_vae = loss_vae,
                                                                        train_mean_norm_lwrmse = mean_norm_lwrmse)
                    ##########################################################
                        if self.world_rank == 0 and self.params.log_to_wandb:
                            #wandb.log(diagnostic_logs, step=(self.epoch-1) * total_iterations + self.iters)
                            # Use wandb_step to ensure monotonicity, then increment it
                            wandb.log(diagnostic_logs, step=self.wandb_step)
                            self.wandb_step += 1

                    # torch.cuda.empty_cache()
                    tr_time += time.time() - tr_start
                
                    pbar.set_description(f"Epoch [{self.epoch}/{self.params.max_epochs}], Year {self.params.train_year_start + year_idx}, Loss: {diagnostic_logs['train_batch_loss']:.4f}")
                    # pbar.set_description(f"Year {self.params.train_year_start + year_idx}, Loss: {diagnostic_logs['train_batch_loss']:.4f}")
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
        
        Handles both legacy and non-legacy model interfaces:
        - Legacy model: does not accept target_surface/target_upper_air, returns 4-5 values
        - Non-legacy model: accepts targets for VAE encoder 2, returns 6-7 values
        
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
                Tuple[Tensor, ...]: 
                    - output_surface: Predicted surface variables.
                    - output_upper_air: Predicted upper air variables.
                    - output_diagnostic: Predicted diagnostic variables (zero if diagnostics are not used).
                    - loss_sfc: Surface loss.
                    - loss_pl: Pressure level (upper air) loss.
                    - loss_diagnostic: Diagnostic loss.
                    - loss_vae: VAE KL divergence loss.
                    - loss: Computed total loss value.
        """
        # Initialize outputs
        output_surface = torch.zeros(1, device=self.device) 
        output_upper_air = torch.zeros(1, device=self.device) 
        output_diagnostic = torch.zeros(1, device=self.device) 
        loss = torch.zeros(1, device=self.device) 
        loss_diagnostic = torch.zeros(1, device=self.device)
        loss_pl = torch.zeros(1, device=self.device) 
        loss_sfc = torch.zeros(1, device=self.device)
        loss_vae = torch.zeros(1, device=self.device)
        
        with autocast(enabled = self.params.enable_amp, device_type="cuda"):
            # Branch based on model type (legacy vs non-legacy)
            if self.params.use_legacy_model:
                # Legacy model: doesn't accept target_surface/target_upper_air
                # Returns: (output_surface, output_upper_air, [output_diagnostic,] mu, sigma)
                # Note: mu and sigma are placeholder tensors (torch.tensor(0.)) in legacy
                if self.params.has_diagnostic:
                    output_surface, output_upper_air, output_diagnostic, mu, sigma = self.model(
                        input_surface, constant_boundary_data, 
                        varying_boundary_data, input_upper_air, train=True)
                    loss_diagnostic = self.loss_obj_diagnostic(output_diagnostic, target_diagnostic)
                else: 
                    output_surface, output_upper_air, mu, sigma = self.model(
                        input_surface, constant_boundary_data, 
                        varying_boundary_data, input_upper_air, train=True)
                # Legacy model doesn't have second encoder for VAE
                mu2, sigma2 = None, None
            else:
                # Non-legacy model
                # Returns: (output_surface, output_upper_air, [output_diagnostic,] mu, sigma, mu2, sigma2)
                if self.params.has_diagnostic:
                    output_surface, output_upper_air, output_diagnostic, mu, sigma, mu2, sigma2 = self.model(
                        input_surface, constant_boundary_data,
                        varying_boundary_data, input_upper_air, train=True)
                    loss_diagnostic = self.loss_obj_diagnostic(output_diagnostic, target_diagnostic)
                else:
                    output_surface, output_upper_air, mu, sigma, mu2, sigma2 = self.model(
                        input_surface, constant_boundary_data,
                        varying_boundary_data, input_upper_air, train=True)
                
            # Compute losses (same for both model types)
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

            # VAE loss: only applicable for non-legacy model with dual encoder
            if self.params.vae_loss:
                if mu2 is not None and sigma2 is not None:
                    loss_vae = self.loss_vae(mu, sigma, mu2, sigma2)
                    loss += self.params.vae_loss_weight * loss_vae
                    # print(f"loss_vae = {loss_vae}, loss = {loss}")
                else:
                    # Legacy model doesn't support VAE loss - warn once and skip
                    if not hasattr(self, '_vae_legacy_warning_shown'):
                        logging.warning("VAE loss is enabled but legacy model doesn't support dual encoder. "
                                        "VAE loss will be skipped. Consider using non-legacy model for VAE training.")
                        self._vae_legacy_warning_shown = True

        return output_surface, output_upper_air, output_diagnostic, loss_sfc, loss_pl, loss_diagnostic, loss_vae, loss

    def diagnostic_log_per_iter(self, diagnostic_logs, diagnostic_lwrmse, surface_lwrmse, upper_air_lwrmse, current_dataset, **kwargs)->dict:
        """
        This function is used for logging the results from each iteration.
        Given the diagnostic logging input and return the update diagnostic_logs
        """

        # ADD THIS: Log current learning rate
        current_lr = torch.tensor(self.optimizer.param_groups[0]['lr'], device = self.device)
        diagnostic_logs['lr_step'] = current_lr

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
                    try:
                        dist.all_reduce(diagnostic_logs[key].detach())
                    except:
                        raise ValueError(f'Diagnostic log {key} is type {type(diagnostic_logs[key])} and cannot be reduced.')
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
        Initialize validation loss variables.
        
        Returns a tuple of tensors for accumulating validation metrics.
        Note: valid_loss_diag and valid_diagnostic_lwrmse are None when has_diagnostic=False.
        """
        if self.params.has_diagnostic:
            valid_buff = torch.zeros((5), dtype=torch.float32, device=self.device)
            valid_loss_diag = valid_buff[3].view(-1)
            valid_diagnostic_lwrmse = torch.zeros(
                (len(lead_times_steps), len(self.valid_dataset.diagnostic_variables)), 
                dtype=torch.float32, device=self.device
            )
        else:
            valid_buff = torch.zeros((4), dtype=torch.float32, device=self.device)
            valid_loss_diag = None  # Explicitly None when no diagnostics
            valid_diagnostic_lwrmse = None  # Explicitly None when no diagnostics

        valid_loss = valid_buff[0].view(-1)
        valid_loss_sfc = valid_buff[1].view(-1)
        valid_loss_pl = valid_buff[2].view(-1)
        valid_steps = valid_buff[-1].view(-1)

        valid_surface_lwrmse = torch.zeros(
            (len(lead_times_steps), len(self.valid_dataset.surface_variables)), 
            dtype=torch.float32, device=self.device
        )
        valid_upper_air_lwrmse = torch.zeros(
            (len(lead_times_steps), len(self.valid_dataset.upper_air_variables), len(self.valid_dataset.levels)), 
            dtype=torch.float32, device=self.device
        )
        
        multi_step_losses = {
            f"valid_loss_{step}step": torch.zeros(1, dtype=torch.float32, device=self.device) 
            for step in lead_times_steps
        }
        
        # Build multi_step_rmse dict
        multi_step_rmse = {}
        for step in lead_times_steps:
            multi_step_rmse[f"valid_lwrmse_sfc_{step}step"] = torch.zeros(1, dtype=torch.float32, device=self.device)
            multi_step_rmse[f"valid_lwrmse_pl_{step}step"] = torch.zeros(1, dtype=torch.float32, device=self.device)
            if self.params.has_diagnostic:
                multi_step_rmse[f"valid_lwrmse_diag_{step}step"] = torch.zeros(1, dtype=torch.float32, device=self.device)
        
        return (valid_loss_diag, valid_buff, valid_loss, valid_loss_sfc, valid_loss_pl, 
                valid_steps, valid_surface_lwrmse, valid_upper_air_lwrmse, 
                valid_diagnostic_lwrmse, multi_step_losses, multi_step_rmse)
    
    def _ema_active(self) -> bool:
        """Return True when EMA should be used this epoch.

        EMA is suppressed for the first ``ema_warmup_epochs`` epochs (1-indexed).
        With the default of 0 warmup epochs the behaviour is identical to always
        using EMA (fully backwards-compatible).
        """
        if self.ema is None:
            return False
        warmup = getattr(self.params, 'ema_warmup_epochs', 0)
        return self.epoch >= warmup

    def _get_model_for_eval(self) -> torch.nn.Module:
        """Select and prepare model for validation (EMA if available and active)."""
        if self._ema_active():
            model = self.ema
            if self.params.log_to_screen:
                logging.info('Using EMA model for validation')
        else:
            model = self.model.module if dist.is_initialized() else self.model
        # model = self.model.module if dist.is_initialized() else self.model
        model.eval()
        return model

    def _create_metrics_logs(self, results: dict, lead_times_steps: list = None) -> dict:
        """
        Create wandb-ready log entries from MetricsAggregator results.

        Only logs metrics for variables/levels specified in diagnostic_acc_var_dict.

        Args:
            results: Dictionary from MetricsAggregator.compute()
            lead_times_steps: The full list of lead-time steps that was passed to the
                MetricsAggregator (e.g. [1,2,...,60] in just_validate mode, or
                forecast_lead_times in training mode).  Used to map each step in
                forecast_lead_times to the correct index in the results tensors.
                If None, falls back to using forecast_lead_times directly (training mode
                behaviour, where the two lists are identical).

        Returns:
            Dictionary of metric name -> value for wandb logging
        """
        logs = {}
        var_dict = getattr(self.params, 'diagnostic_acc_var_dict', {})
        lead_times = self.params.forecast_lead_times

        # Build a mapping from step number → result-tensor index.
        # In just_validate mode lead_times_steps = [1,2,...,max] while
        # lead_times = forecast_lead_times = [1,12,20,40,60].  Using step_i
        # (position in lead_times) as the index into results is wrong in that
        # case because step_i=1 would give the 2-step result, not the 12-step.
        if lead_times_steps is not None:
            step_to_idx = {step: idx for idx, step in enumerate(lead_times_steps)}
        else:
            step_to_idx = {step: idx for idx, step in enumerate(lead_times)}

        # Get variable lists
        surface_vars = list(self.valid_dataset.surface_variables)
        upper_air_vars = list(self.valid_dataset.upper_air_variables)
        levels = list(self.valid_dataset.levels)

        for step in lead_times:
            if step not in step_to_idx:
                continue
            result_idx = step_to_idx[step]
            for var, var_levels in var_dict.items():
                if not var_levels:
                    # Surface variable
                    if var in surface_vars:
                        var_idx = surface_vars.index(var)
                        acc_val = results['acc_surface'][result_idx, var_idx].item()
                        rmse_val = results['rmse_surface'][result_idx, var_idx].item()
                        logs[f'valid_{var}_{step}step_acc'] = acc_val
                        logs[f'valid_{var}_{step}step_rmse'] = rmse_val
                else:
                    # Upper air variable with specific levels
                    if var in upper_air_vars:
                        var_idx = upper_air_vars.index(var)
                        for level in var_levels:
                            # Find closest level index
                            level_idx = None
                            for li, lev in enumerate(levels):
                                if abs(lev - level) < 1e-3:  # Close enough match
                                    level_idx = li
                                    break
                            if level_idx is not None:
                                acc_val = results['acc_upper_air'][result_idx, var_idx, level_idx].item()
                                rmse_val = results['rmse_upper_air'][result_idx, var_idx, level_idx].item()
                                logs[f'valid_{var}_{int(level)}_{step}step_acc'] = acc_val
                                logs[f'valid_{var}_{int(level)}_{step}step_rmse'] = rmse_val

        return logs

    def _should_save_plots(self, epoch: int) -> bool:
        """
        Determine if plots should be saved this epoch.
        
        Saves at:
        - Every plot_save_interval epochs (during training)
        - Last epoch (max_epochs)
        - During just_validate mode (always)
        """
        if self.params.just_validate:
            return True
        if (epoch + 1) % self.params.plot_save_interval == 0:
            return True
        if (epoch + 1) == self.params.max_epochs:
            return True
        return False

    def _cleanup_old_plots(self, directory: str, pattern: str):
        """
        Remove old plot files, keeping only the most recent N.
        
        Args:
            directory: Directory containing the plots
            pattern: Glob pattern for the files (e.g., 'power_spectrum_epoch_*.png')
        """
        plot_files = natsorted([
            f for f in glob.glob(os.path.join(directory, pattern))
            if os.path.isfile(f)
        ])
        
        num_to_delete = len(plot_files) - self.params.max_plots_to_keep
        if num_to_delete > 0:
            for old_plot in plot_files[:num_to_delete]:
                os.remove(old_plot)
                logging.info(f"Removed old plot: {old_plot}")

    def _plot_acc_rmse_vs_leadtime(self, results: dict, lead_time_steps: list, save_dir: str):
        """
        Create combined ACC/RMSE vs lead time plots with dual y-axes.
        
        Generates one figure per variable/level in diagnostic_acc_var_dict,
        with ACC on left y-axis and RMSE on right y-axis.
        
        Args:
            results: Dictionary from MetricsAggregator.compute()
            lead_time_steps: List of lead time steps used for metrics
            save_dir: Directory to save the plots
        """
        var_dict = getattr(self.params, 'diagnostic_acc_var_dict', {})
        if not var_dict:
            logging.info("No diagnostic_acc_var_dict specified, skipping ACC/RMSE plots")
            return
        
        # Convert lead time steps to hours
        lead_time_hours = [lt * self.params.timedelta_hours for lt in lead_time_steps]
        
        surface_vars = list(self.valid_dataset.surface_variables)
        upper_air_vars = list(self.valid_dataset.upper_air_variables)
        diag_vars = list(self.valid_dataset.diagnostic_variables) if hasattr(self.valid_dataset, 'diagnostic_variables') else []
        levels = list(self.valid_dataset.levels)
        
        for var, var_levels in var_dict.items():
            if not var_levels:
                # Surface or diagnostic variable (no levels)
                if var in surface_vars:
                    var_idx = surface_vars.index(var)
                    acc_values = [results['acc_surface'][step_i, var_idx].item() 
                                for step_i in range(len(lead_time_steps))]
                    rmse_values = [results['rmse_surface'][step_i, var_idx].item() 
                                for step_i in range(len(lead_time_steps))]
                elif var in diag_vars:
                    var_idx = diag_vars.index(var)
                    acc_values = [results['acc_diagnostic'][step_i, var_idx].item() 
                                for step_i in range(len(lead_time_steps))]
                    rmse_values = [results['rmse_diagnostic'][step_i, var_idx].item() 
                                for step_i in range(len(lead_time_steps))]
                else:
                    logging.warning(f"Variable {var} not found in surface or diagnostic variables, skipping")
                    continue
                
                self._save_dual_axis_plot(var, None, lead_time_hours, acc_values, rmse_values, save_dir)
            else:
                # Upper air variable with specific levels
                if var not in upper_air_vars:
                    logging.warning(f"Variable {var} not found in upper air variables, skipping")
                    continue
                var_idx = upper_air_vars.index(var)
                
                for level in var_levels:
                    # Find closest level index
                    level_idx = None
                    for li, lev in enumerate(levels):
                        if abs(lev - level) < 1e-3:
                            level_idx = li
                            break
                    if level_idx is None:
                        logging.warning(f"Level {level} not found for variable {var}, skipping")
                        continue
                    
                    acc_values = [results['acc_upper_air'][step_i, var_idx, level_idx].item() 
                                for step_i in range(len(lead_time_steps))]
                    rmse_values = [results['rmse_upper_air'][step_i, var_idx, level_idx].item() 
                                for step_i in range(len(lead_time_steps))]
                    
                    self._save_dual_axis_plot(var, level, lead_time_hours, acc_values, rmse_values, save_dir)

    def _save_dual_axis_plot(self, var: str, level, lead_time_hours: list, 
                        acc_values: list, rmse_values: list, save_dir: str):
        """
        Save a dual y-axis plot with ACC on left and RMSE on right.
        Also saves metrics to a CSV file for future analysis.
        
        Args:
            var: Variable name
            level: Pressure level (None for surface/diagnostic variables)
            lead_time_hours: List of lead times in hours
            acc_values: ACC values for each lead time
            rmse_values: RMSE values for each lead time
            save_dir: Directory to save the plot
        """
        # Convert hours to days for x-axis
        lead_time_days = [h / 24.0 for h in lead_time_hours]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Title suffix for level
        level_str = f"_{int(level)}Pa" if level is not None else ""
        title_level = f" at {int(level)} Pa" if level is not None else " (surface)"
        
        # Left y-axis: ACC (blue)
        color_acc = 'tab:blue'
        ax1.set_xlabel('Lead Time (days)', fontsize=12)
        ax1.set_ylabel('ACC', color=color_acc, fontsize=12)
        line1, = ax1.plot(lead_time_days, acc_values, color=color_acc, linestyle='-', 
                        marker='o', linewidth=2, markersize=5, label='ACC')
        ax1.tick_params(axis='y', labelcolor=color_acc)
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3)
        
        # Right y-axis: RMSE (red)
        ax2 = ax1.twinx()
        color_rmse = 'tab:red'
        ax2.set_ylabel('RMSE', color=color_rmse, fontsize=12)
        line2, = ax2.plot(lead_time_days, rmse_values, color=color_rmse, linestyle='--', 
                        marker='s', linewidth=2, markersize=5, label='RMSE')
        ax2.tick_params(axis='y', labelcolor=color_rmse)
        
        # Combined legend - positioned outside plot area to avoid overlap
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='lower right')
        
        plt.title(f'ACC & RMSE vs Lead Time: {var}{title_level}', fontsize=14)
        fig.tight_layout()
        
        # Save plot
        filename = os.path.join(save_dir, f"acc_rmse_{var}{level_str}_epoch_{self.epoch}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logging.info(f"Saved ACC/RMSE plot: {filename}")
        
        # Save metrics to CSV file for future analysis
        csv_filename = os.path.join(save_dir, f"acc_rmse_{var}{level_str}_epoch_{self.epoch}.csv")

        df = pd.DataFrame({
            'lead_time_hours': lead_time_hours,
            'lead_time_days': lead_time_days,
            'ACC': acc_values,
            'RMSE': rmse_values
        })
        df.to_csv(csv_filename, index=False)
        logging.info(f"Saved ACC/RMSE metrics: {csv_filename}")

    def validate_one_epoch(self):
        """
        Validation loop using MetricsAggregator for ACC/RMSE computation.
        
        Changes from original:
        - Uses PyTorch-based MetricsAggregator instead of xarray
        - Only computes metrics at forecast_lead_times
        - Removes ACC plot generation (keeps GIF and power spectrum)
        - Logs ACC/RMSE for variables in diagnostic_acc_var_dict
        """
        if world_rank == 0:
            print("Validating...")
        
        model_to_eval = self._get_model_for_eval()
        # For just_validate, compute metrics at all lead times (1 to max_lead_time)
        # For training validation, use forecast_lead_times only
        if self.params.just_validate:
            max_lead_time = max(self.params.forecast_lead_times)
            lead_times_steps = list(range(1, max_lead_time + 1))
            logging.info(f"Just validate mode: computing metrics at all {max_lead_time} lead times")
        else:
            lead_times_steps = self.params.forecast_lead_times
        
        # Initialize validation loss variables
        valid_loss_diag, valid_buff, valid_loss, valid_loss_sfc, valid_loss_pl, valid_steps, \
        valid_surface_lwrmse, valid_upper_air_lwrmse, valid_diagnostic_lwrmse, \
        multi_step_losses, multi_step_rmse = self.inti_valid_loss(lead_times_steps)
        
        valid_start = time.time()
        nb = len(self.valid_data_loader)
        
        diagnostic_logs = {}
        gif_filename = None
        spectra_filename = None
        
        # Initialize MetricsAggregator for ACC/RMSE computation
        metrics = create_metrics_aggregator_new(self, lead_time_steps=lead_times_steps)
        
        # Lists for GIF and power spectrum (still use xarray for these)
        all_predictions = []
        all_ground_truths = []
        all_predictions = []
        all_ground_truths = []
        acc_predictions = []  # For GIF
        acc_ground_truths = []  # For GIF
        
        with torch.no_grad():
            if self.params.enable_fp8:
                logging.warning("FP8 is not fully configured for validation. Falling back to AMP.")
            precision_context = autocast(enabled=self.params.enable_amp, device_type="cuda")
            
            no_nans = True
            
            # Long validation section (unchanged)
            if self.long_validation and self.epoch % self.epochs_per_long_validation == 0:
                print('Performing long validation...')
                cnt = 0
                no_nans = True
                val_surface_bias = None
                val_upper_air_bias = None
                if self.params.has_diagnostic:
                    val_diagnostic_bias = None
                if self.params.ensemble_validation:
                    val_data_dir = self.params.validation_data_dir
                    if self.world_rank == 0:
                        os.makedirs(val_data_dir, exist_ok=True)
                    if dist.is_initialized():
                        dist.barrier(device_ids=[self.device])
                    print(val_data_dir)
                    os.makedirs(val_data_dir, exist_ok=True)
                pbar = tqdm(enumerate(self.long_valid_data_loader, 0),
                            total=len(self.long_valid_data_loader), miniters=1,
                            dynamic_ncols=True, file=logging_utils.tqdm_stream,
                            disable=(self.world_rank != 0))
                plt.ion()
                for i, data in pbar:
                    if i == 0:
                        val_input_surface, val_input_upper_air, val_varying_boundary_data, year = map(lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), data)
                        val_input_surface, val_input_upper_air = self.perturber(val_input_surface, val_input_upper_air)
                    else:
                        val_varying_boundary_data, year = map(lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), data)
                    if self.params.use_legacy_model:
                        if self.params.has_diagnostic:
                            val_output_surface, val_output_upper_air, val_output_diagnostic, _, _ = model_to_eval(
                                val_input_surface, self.constant_boundary_data[[0]], val_varying_boundary_data, val_input_upper_air)
                        else:
                            val_output_surface, val_output_upper_air, _, _ = model_to_eval(
                                val_input_surface, self.constant_boundary_data[[0]], val_varying_boundary_data, val_input_upper_air)
                    else:
                        if self.params.has_diagnostic:
                            val_output_surface, val_output_upper_air, val_output_diagnostic, _, _, _, _ = model_to_eval(
                                val_input_surface, self.constant_boundary_data[[0]], val_varying_boundary_data, val_input_upper_air)
                        else:
                            val_output_surface, val_output_upper_air, _, _, _, _ = model_to_eval(
                                val_input_surface, self.constant_boundary_data[[0]], val_varying_boundary_data, val_input_upper_air)
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

                if val_surface_bias is not None:
                    if dist.is_initialized():
                        dist.all_reduce(val_surface_bias, op = dist.ReduceOp.AVG)
                        dist.all_reduce(val_upper_air_bias, op = dist.ReduceOp.AVG)
                        if self.params.has_diagnostic:
                            dist.all_reduce(val_diagnostic_bias, op = dist.ReduceOp.AVG)
                        if torch.any(torch.isnan(val_output_surface)) or torch.any(torch.isnan(val_output_upper_air)):
                            no_nans = False
                else:
                    logging.warning(f'No bias accumulated during long validation. Year threshold ({self.params.long_val_year_start + self.long_validation_spinup_years}) may not have been reached.')
                    no_nans = False
                
                if no_nans and self.world_rank == 0:
                    val_surface_bias = self.long_valid_dataset.surface_inv_transform(val_surface_bias.cpu())
                    val_surface_bias_lwrmse = weighted_rmse_torch_channels(val_surface_bias, self.clim_surface_bias.cpu(), self.latitudes.cpu()).squeeze(0)
                    val_upper_air_bias = self.long_valid_dataset.upper_air_inv_transform(val_upper_air_bias.cpu())
                    val_upper_air_bias_lwrmse = weighted_rmse_torch_3D(val_upper_air_bias, self.clim_upper_air_bias.cpu(), self.latitudes.cpu()).squeeze(0)
                    if self.params.has_diagnostic:
                        val_diagnostic_bias = self.long_valid_dataset.diagnostic_inv_transform(val_diagnostic_bias.cpu())
                        val_diagnostic_bias_lwrmse = weighted_rmse_torch_channels(val_diagnostic_bias, self.clim_diagnostic_bias.cpu(), self.latitudes.cpu()).squeeze(0)
                    start_times = [self.long_valid_dataset.datetime_class(self.params.long_val_year_start + self.long_validation_spinup_years,
                                                                        1, 1, has_year_zero = self.long_valid_dataset.has_year_zero) - \
                                                                            timedelta(hours=self.params.timedelta_hours)]
                    bias_datasets = self.convert_to_xarray(np.expand_dims(val_surface_bias.numpy(), axis = 1),
                                                        np.expand_dims(val_upper_air_bias.numpy(), axis = 1),
                                                        start_times, self.params, self.long_valid_dataset, acc = True,
                                                        diagnostic_prediction = None if not self.params.has_diagnostic else np.expand_dims(val_diagnostic_bias.numpy(), axis=1))
                        
                    print('Plotting Bias')
                    bias_filename = os.path.join(self.bias_dir, f"bias_epoch_{self.epoch}.png")                    
                    self.plot_in_separate_process(bias_datasets[0].squeeze("time"), self.climatology_bias, [], bias_filename)

                    print("\nFinished Bias Plots...")
                    
                    if self.params.log_to_wandb:
                        wandb.log({
                            "bias_plot": wandb.Image(bias_filename),
                            "epoch": self.epoch
                        }, step=self.wandb_step)
                    
                
                
                                
            for i, data in tqdm(enumerate(self.valid_data_loader, 0), total=nb,
                                bar_format='{l_bar}{bar:30}{r_bar}', miniters=1,
                                dynamic_ncols=True, file=logging_utils.tqdm_stream,
                                disable=(self.world_rank != 0)):
                if world_rank == 0:
                    print(f"Validating batch {i+1}/{nb}")
                
                # Load data
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

                # Get start times for each sample
                start_times = []
                for j in range(times.shape[0]):
                    start_time = self.valid_dataset.datetime_class(times[j,0].item(), times[j,1].item(), times[j,2].item(), hour=times[j,3].item())
                    start_times.append(start_time)
                
                max_lead_time = max(lead_times_steps)
                
                # Storage for GIF/spectra (xarray-based, only if needed)
                if self.params.diagnostic_gif:
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
                
                if self.params.diagnostic_spectra:
                    val_output_surface_t = np.zeros((val_input_surface.shape[0], len(lead_times_steps),
                                        val_input_surface.shape[1], val_input_surface.shape[2], val_input_surface.shape[3]),
                                        dtype=np.float32)
                    val_output_upper_air_t = np.zeros((val_input_upper_air.shape[0], len(lead_times_steps),
                                            val_input_upper_air.shape[1], val_input_upper_air.shape[2],
                                            val_input_upper_air.shape[3], val_input_upper_air.shape[4]),
                                            dtype=np.float32)
                    if self.params.has_diagnostic:
                        val_output_diagnostic_t = np.zeros((val_target_diagnostic.shape[0], len(lead_times_steps),
                                                            val_target_diagnostic.shape[2], val_target_diagnostic.shape[3],
                                                            val_target_diagnostic.shape[4]), dtype=np.float32)
                
                with precision_context:
                    step_idx = 0
                    for step in range(max_lead_time):
                        # DIAGNOSTIC: Log first forward pass inputs/outputs in validate_one_epoch
                        if i == 0 and step == 0 and self.world_rank == 0:
                            logging.info("===== TRAINER validate_one_epoch: FIRST FORWARD PASS DIAGNOSTIC =====")
                            logging.info(f"val_input_surface shape: {val_input_surface.shape}, dtype: {val_input_surface.dtype}")
                            logging.info(f"val_input_surface[:1,:,0,0]: {val_input_surface[:1,:,0,0]}")
                            logging.info(f"val_input_surface stats: min={val_input_surface.min().item():.6f}, max={val_input_surface.max().item():.6f}, mean={val_input_surface.mean().item():.6f}")
                            logging.info(f"constant_boundary_data shape: {self.constant_boundary_data.shape}")
                            logging.info(f"constant_boundary_data[:1,:,0,0]: {self.constant_boundary_data[:1,:,0,0]}")
                            logging.info(f"val_varying_boundary_data[:,{step}] shape: {val_varying_boundary_data[:, step].shape}")
                            logging.info(f"val_varying_boundary_data[:1,{step},:,0,0]: {val_varying_boundary_data[:1,step,:,0,0]}")
                            logging.info(f"val_input_upper_air shape: {val_input_upper_air.shape}")
                            logging.info(f"val_input_upper_air[:1,:,:,0,0]: {val_input_upper_air[:1,:,:,0,0]}")
                            logging.info(f"Model training mode: {self.model.training}")
                        
                        if self.params.use_legacy_model:
                            if self.params.has_diagnostic:
                                val_output_surface, val_output_upper_air, val_output_diagnostic, _, _ = model_to_eval(
                                    val_input_surface, self.constant_boundary_data, val_varying_boundary_data[:, step], val_input_upper_air)
                            else:
                                val_output_surface, val_output_upper_air, _, _ = model_to_eval(
                                    val_input_surface, self.constant_boundary_data, val_varying_boundary_data[:, step], val_input_upper_air)
                        else:
                            if self.params.has_diagnostic:
                                val_output_surface, val_output_upper_air, val_output_diagnostic, _, _, _, _ = model_to_eval(
                                    val_input_surface, self.constant_boundary_data, val_varying_boundary_data[:, step], val_input_upper_air)
                            else:
                                val_output_surface, val_output_upper_air, _, _, _, _ = model_to_eval(
                                    val_input_surface, self.constant_boundary_data, val_varying_boundary_data[:, step], val_input_upper_air)
                        
                        # DIAGNOSTIC: Log first forward pass output in validate_one_epoch
                        if i == 0 and step == 0 and self.world_rank == 0:
                            logging.info(f"val_output_surface shape: {val_output_surface.shape}, dtype: {val_output_surface.dtype}")
                            logging.info(f"val_output_surface[:1,:,0,0]: {val_output_surface[:1,:,0,0]}")
                            logging.info(f"val_output_surface stats: min={val_output_surface.min().item():.6f}, max={val_output_surface.max().item():.6f}, mean={val_output_surface.mean().item():.6f}")
                            logging.info(f"val_output_upper_air[:1,:,:,0,0]: {val_output_upper_air[:1,:,:,0,0]}")
                            logging.info("===== END TRAINER FIRST FORWARD PASS DIAGNOSTIC =====")
                        # Calculate losses for different lead times
                        if (step + 1) in lead_times_steps:
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
                        
                        # Apply delta integration if needed
                        if self.params.predict_delta:
                            val_output_surface, val_output_upper_air = self.integrator(
                                val_input_surface, val_input_upper_air, val_output_surface, val_output_upper_air)
                        
                        # Store for GIF (all steps)
                        if self.params.diagnostic_gif:
                            val_output_surface_acc[:, step] = self.valid_dataset.surface_inv_transform(val_output_surface.cpu()).numpy()
                            val_output_upper_air_acc[:, step] = self.valid_dataset.upper_air_inv_transform(val_output_upper_air.cpu()).numpy()
                            if self.params.has_diagnostic:
                                val_output_diagnostic_acc[:, step] = self.valid_dataset.diagnostic_inv_transform(val_output_diagnostic.cpu()).numpy()
                        
                        # At forecast_lead_times: update metrics and store for spectra
                        if (step + 1) in lead_times_steps:
                            # Get TARGET timestamps for this step (initial time + lead time).
                            # The climatology lookup in MetricsAggregator must use the
                            # day-of-year of the TARGET, not the initial condition.
                            advance_hours = int((step + 1) * self.params.timedelta_hours)
                            timestamps = times.clone()
                            for _j in range(times.shape[0]):
                                _y = int(times[_j, 0]); _m = int(times[_j, 1])
                                _d = int(times[_j, 2]); _h = int(times[_j, 3])
                                _tgt = datetime(_y, _m, _d, _h) + timedelta(hours=advance_hours)
                                timestamps[_j, 0] = _tgt.year
                                timestamps[_j, 1] = _tgt.month
                                timestamps[_j, 2] = _tgt.day
                                timestamps[_j, 3] = _tgt.hour

                            # Update MetricsAggregator (uses denormalized values internally)
                            if self.params.has_diagnostic:
                                metrics.update(
                                    pred_surface=val_output_surface,
                                    pred_upper_air=val_output_upper_air,
                                    target_surface=val_target_surface[:, target_index],
                                    target_upper_air=val_target_upper_air[:, target_index],
                                    timestamps=timestamps,
                                    step_idx=step_idx,
                                    pred_diagnostic=val_output_diagnostic,
                                    target_diagnostic=val_target_diagnostic[:, target_index],
                                )
                            else:
                                metrics.update(
                                    pred_surface=val_output_surface,
                                    pred_upper_air=val_output_upper_air,
                                    target_surface=val_target_surface[:, target_index],
                                    target_upper_air=val_target_upper_air[:, target_index],
                                    timestamps=timestamps,
                                    step_idx=step_idx,
                                )
                            
                            # Calculate RMSE for legacy logging
                            rmse_sfc = weighted_rmse_torch_channels(val_output_surface, val_target_surface[:,target_index], self.latitudes)
                            rmse_pl = weighted_rmse_torch_3D(val_output_upper_air, val_target_upper_air[:,target_index], self.latitudes)
                            if self.params.has_diagnostic:
                                rmse_diag = weighted_rmse_torch_channels(val_output_diagnostic, val_target_diagnostic[:,target_index], self.latitudes)
                                multi_step_rmse[f"valid_lwrmse_diag_{step+1}step"] += torch.mean(rmse_diag)
                                valid_diagnostic_lwrmse[step_idx] += torch.mean(rmse_diag, dim = 0)
                            
                            multi_step_rmse[f"valid_lwrmse_sfc_{step+1}step"] += torch.mean(rmse_sfc)
                            multi_step_rmse[f"valid_lwrmse_pl_{step+1}step"] += torch.mean(rmse_pl)
                            valid_surface_lwrmse[step_idx] += torch.mean(rmse_sfc, dim = 0)
                            valid_upper_air_lwrmse[step_idx] += torch.mean(rmse_pl, dim=0)
                            
                            # Store for spectra
                            if self.params.diagnostic_spectra:
                                val_output_surface_t[:, step_idx] = self.valid_dataset.surface_inv_transform(val_output_surface.cpu()).numpy()
                                val_output_upper_air_t[:, step_idx] = self.valid_dataset.upper_air_inv_transform(val_output_upper_air.cpu()).numpy()
                                if self.params.has_diagnostic:
                                    val_output_diagnostic_t[:, step_idx] = self.valid_dataset.diagnostic_inv_transform(val_output_diagnostic.cpu()).numpy()
                            
                            # At max lead time, prepare xarray datasets for GIF and spectra
                            if step + 1 == max_lead_time:
                                if self.params.diagnostic_gif:
                                    if self.params.has_diagnostic:
                                        acc_datasets = self.convert_to_xarray(val_output_surface_acc, val_output_upper_air_acc, start_times, self.params, self.valid_dataset, acc=True,
                                                                            diagnostic_prediction=val_output_diagnostic_acc)
                                    else:
                                        acc_datasets = self.convert_to_xarray(val_output_surface_acc, val_output_upper_air_acc, start_times, self.params, self.valid_dataset, acc=True)
                                    acc_prepared_datasets = [self.prepare_preds(ds, acc=True) for ds in acc_datasets]
                                    acc_combined_dataset = self.combine_datasets(acc_prepared_datasets)
                                    acc_predictions.append(acc_combined_dataset)

                                    acc_gt_surface = self.valid_dataset.surface_inv_transform(val_target_surface.cpu()).numpy()
                                    acc_gt_upper_air = self.valid_dataset.upper_air_inv_transform(val_target_upper_air.cpu()).numpy()
                                    if self.params.has_diagnostic:
                                        acc_gt_diagnostic = self.valid_dataset.diagnostic_inv_transform(val_target_diagnostic.cpu()).numpy()
                                        acc_gt_datasets = self.convert_to_xarray(acc_gt_surface, acc_gt_upper_air, start_times, self.params, self.valid_dataset, acc=True,
                                                                                diagnostic_prediction=acc_gt_diagnostic)
                                    else:
                                        acc_gt_datasets = self.convert_to_xarray(acc_gt_surface, acc_gt_upper_air, start_times, self.params, self.valid_dataset, acc=True)
                                    acc_gt_prepared_datasets = [self.prepare_preds(ds, acc=True) for ds in acc_gt_datasets]
                                    acc_gt_combined_dataset = self.combine_datasets(acc_gt_prepared_datasets)
                                    acc_ground_truths.append(acc_gt_combined_dataset)

                                if self.params.diagnostic_spectra:
                                    # lead_time_indices = [lt - 1 for lt in self.params.forecast_lead_times]
                                    lead_time_indices = [lt - 1 for lt in lead_times_steps]
                                    if self.params.has_diagnostic:
                                        datasets = self.convert_to_xarray(val_output_surface_t, val_output_upper_air_t, start_times, self.params, self.valid_dataset, acc=False,
                                                                        diagnostic_prediction=val_output_diagnostic_t, lead_times=lead_times_steps)
                                    else:
                                        datasets = self.convert_to_xarray(val_output_surface_t, val_output_upper_air_t, start_times, self.params, self.valid_dataset, acc=False,
                                                                          lead_times=lead_times_steps)
                                    prepared_datasets = [self.prepare_preds(ds, acc=False, lead_times=lead_times_steps) for ds in datasets]
                                    combined_dataset = self.combine_datasets(prepared_datasets)

                                    gt_surface = self.valid_dataset.surface_inv_transform(val_target_surface[:, lead_time_indices].cpu()).numpy()
                                    gt_upper_air = self.valid_dataset.upper_air_inv_transform(val_target_upper_air[:, lead_time_indices].cpu()).numpy()
                                    if self.params.has_diagnostic:
                                        gt_diagnostic = self.valid_dataset.diagnostic_inv_transform(val_target_diagnostic[:, lead_time_indices].cpu()).numpy()
                                        gt_datasets = self.convert_to_xarray(gt_surface, gt_upper_air, start_times, self.params, self.valid_dataset, acc=False,
                                                                            diagnostic_prediction=gt_diagnostic, lead_times=lead_times_steps)
                                    else:
                                        gt_datasets = self.convert_to_xarray(gt_surface, gt_upper_air, start_times, self.params, self.valid_dataset, acc=False,
                                                                             lead_times=lead_times_steps)
                                    gt_prepared_datasets = [self.prepare_preds(ds, acc=False, lead_times=lead_times_steps) for ds in gt_datasets]
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
        
        # Combine xarray datasets for GIF/spectra
        if self.params.diagnostic_spectra:
            combined_predictions = xr.concat(all_predictions, dim='time')
            combined_ground_truths = xr.concat(all_ground_truths, dim='time')
        print("\nFinished combining predictions and ground truths.")
        
        if self.params.diagnostic_gif:
            acc_combined_predictions = xr.concat(acc_predictions, dim='time')
            acc_combined_ground_truths = xr.concat(acc_ground_truths, dim='time')
            if self.params.just_validate or (not self.params.just_validate and self.epoch == 100):
                if self.world_rank == 0:
                    val_data_dir = self.params.validation_data_dir
                    os.makedirs(val_data_dir, exist_ok=True)
                    for start_time in acc_combined_predictions.time.values:
                        pred_out = acc_combined_predictions.sel(time=start_time)
                        pred_out = pred_out.drop_vars("time")
                        pred_out['lead_time'] = [start_time + timedelta(hours=int(elem)) for elem in pred_out.lead_time.values]
                        pred_out = pred_out.rename({'lead_time': 'time'})
                        pred_out.to_netcdf(os.path.join(val_data_dir, f'prediction_{start_time.strftime("%Y-%m-%d_%H")}.nc'))
                    for start_time in acc_combined_ground_truths.time.values:
                        truth_out = acc_combined_ground_truths.sel(time=start_time)
                        truth_out = truth_out.drop_vars("time")
                        truth_out['lead_time'] = [start_time + timedelta(hours=int(elem)) for elem in truth_out.lead_time.values]
                        truth_out = truth_out.rename({'lead_time': 'time'})
                        truth_out.to_netcdf(os.path.join(val_data_dir, f'ground_truth_{start_time.strftime("%Y-%m-%d_%H")}.nc'))
            
        # DDP synchronization for MetricsAggregator
        if dist.is_initialized():
            metrics.all_reduce()
        
        # Compute final ACC and RMSE from MetricsAggregator
        metrics_results = metrics.compute()
        # Plot ACC/RMSE vs lead time (just_validate mode or at save intervals)
        if self.world_rank == 0:
            save_plots_this_epoch = self._should_save_plots(self.epoch)
            if save_plots_this_epoch:
                self._plot_acc_rmse_vs_leadtime(metrics_results, lead_times_steps, self.params['acc_dir'])
                # Cleanup old ACC/RMSE plots (only during training)
                if not self.params.just_validate:
                    for var, var_levels in getattr(self.params, 'diagnostic_acc_var_dict', {}).items():
                        if not var_levels:
                            self._cleanup_old_plots(self.params['acc_dir'], f"acc_rmse_{var}_epoch_*.png")
                        else:
                            for level in var_levels:
                                self._cleanup_old_plots(self.params['acc_dir'], f"acc_rmse_{var}_{int(level)}Pa_epoch_*.png")
            
        
        # Create GIF and power spectrum plots (rank 0 only)
        if self.world_rank == 0:
            if self.params.diagnostic_gif and save_plots_this_epoch:
                print("\nMaking GIFs...")
                gif_var_dict = getattr(self.params, 'diagnostic_gif_var_dict', {'zg': [50000]})
                
                for var, var_levels in gif_var_dict.items():
                    # Check if variable exists in data
                    if var in self.params.upper_air_variables:
                        for plev in var_levels:
                            gif_filename = os.path.join(
                                self.diagnostics_dir, 
                                f"{var}_{int(plev)}_animation_epoch_{self.epoch}.gif"
                            )
                            try:
                                make_gif(
                                    acc_combined_predictions, 
                                    acc_combined_ground_truths, 
                                    "Model Forecast",
                                    var,
                                    gif_filename, 
                                    climatology=self.climatology, 
                                    plev=plev
                                )
                                logging.info(f"Saved GIF: {gif_filename}")
                            except Exception as e:
                                logging.warning(f"Failed to create GIF for {var} at {plev}: {e}")
                    elif var in self.params.surface_variables:
                        # Surface variable (no pressure level)
                        gif_filename = os.path.join(
                            self.diagnostics_dir, 
                            f"{var}_animation_epoch_{self.epoch}.gif"
                        )
                        try:
                            make_gif(
                                acc_combined_predictions, 
                                acc_combined_ground_truths, 
                                "Model Forecast",
                                var,
                                gif_filename, 
                                climatology=self.climatology, 
                                plev=None
                            )
                            logging.info(f"Saved GIF: {gif_filename}")
                        except Exception as e:
                            logging.warning(f"Failed to create GIF for {var}: {e}")
                    else:
                        logging.warning(f"Variable {var} not found in upper_air or surface variables")
                
                print("\nFinished creating GIF animations.")
    
                # Cleanup old GIFs (only during training)
                if not self.params.just_validate:
                    for var, var_levels in gif_var_dict.items():
                        if var in self.params.upper_air_variables:
                            for plev in var_levels:
                                self._cleanup_old_plots(
                                    self.diagnostics_dir, 
                                    f"{var}_{int(plev)}_animation_epoch_*.gif"
                                )
                        else:
                            self._cleanup_old_plots(self.diagnostics_dir, f"{var}_animation_epoch_*.gif")

            if self.params.diagnostic_spectra and save_plots_this_epoch:
                print("\nCalculating power spectrum...")
                k_x_pred, power_spectrum_avg_pred = zonal_averaged_power_spectrum(combined_predictions, time_avg=True) 
                k_x_gt, power_spectrum_avg_gt = zonal_averaged_power_spectrum(combined_ground_truths, time_avg=True)
                preds_times = combined_predictions.time.values
                preds_times = preds_times.cpu().numpy() if isinstance(preds_times, torch.Tensor) else preds_times
                print("\nFinished calculating power spectrum.")
                
                print("\nMaking Power Spectrum...")
                spectra_filename = os.path.join(self.spectra_dir, f"power_spectrum_epoch_{self.epoch}.png")
                preds_times = preds_times.cpu().numpy() if isinstance(preds_times, torch.Tensor) else preds_times
                self.plot_in_separate_process(power_spectrum_avg_pred, power_spectrum_avg_gt, preds_times, spectra_filename)
                print("\nFinished Power Spectrum...")
                
                # Cleanup old power spectrum plots (only during training)
                if not self.params.just_validate:
                    self._cleanup_old_plots(self.spectra_dir, "power_spectrum_epoch_*.png")
        
        # DDP synchronization for legacy metrics
        if dist.is_initialized():
            dist.all_reduce(valid_buff)
            dist.all_reduce(valid_surface_lwrmse)
            dist.all_reduce(valid_upper_air_lwrmse)
            if self.params.has_diagnostic and valid_diagnostic_lwrmse is not None:
                dist.all_reduce(valid_diagnostic_lwrmse)
            for loss_tensor in multi_step_losses.values():
                dist.all_reduce(loss_tensor)
            for rmse_tensor in multi_step_rmse.values():
                dist.all_reduce(rmse_tensor)

        # Normalize legacy metrics by number of steps
        valid_buff[0:-1] = valid_buff[0:-1] / valid_buff[-1]
        valid_surface_lwrmse = (valid_surface_lwrmse / valid_buff[-1]).detach()
        valid_upper_air_lwrmse = (valid_upper_air_lwrmse / valid_buff[-1]).detach()
        if self.params.has_diagnostic:
            valid_diagnostic_lwrmse = (valid_diagnostic_lwrmse / valid_buff[-1]).detach()
        for key in multi_step_losses:
            multi_step_losses[key] /= valid_buff[-1]
        for key in multi_step_rmse:
            multi_step_rmse[key] /= valid_buff[-1]

        valid_buff_cpu = valid_buff.detach()

        # Build diagnostic logs
        diagnostic_logs['epoch'] = self.epoch
        diagnostic_logs['valid_loss'] = valid_buff_cpu[0]
        diagnostic_logs['valid_loss_sfc'] = valid_buff_cpu[1]
        diagnostic_logs['valid_loss_upper_air'] = valid_buff_cpu[2]
        if self.params.has_diagnostic:
            diagnostic_logs['valid_loss_diag'] = valid_buff_cpu[3]

        # Add multi-step losses
        for key, value in multi_step_losses.items():
            diagnostic_logs[key] = value.item()

        # Add ACC/RMSE metrics from MetricsAggregator (only for diagnostic_acc_var_dict vars)
        acc_rmse_logs = self._create_metrics_logs(metrics_results, lead_times_steps)
        diagnostic_logs.update(acc_rmse_logs)

        # Add bias RMSE logs (if long validation)
        if self.long_validation and self.world_rank == 0 and self.epoch % self.epochs_per_long_validation == 0 and no_nans:
            if val_surface_bias is not None:
                for j, var in enumerate(self.valid_dataset.surface_variables):
                    diagnostic_logs[f'valid_{var}_bias_lwrmse'] = val_surface_bias_lwrmse[j].item()
                for j, var in enumerate(self.valid_dataset.upper_air_variables):
                    if var != 'zg' and var != 'geopotential_height' and self.valid_dataset.use_sigma_levels:
                        for k, level in enumerate(self.valid_dataset.sigma_levels):
                            diagnostic_logs[f'valid_{var}_level{level:.3f}_bias_lwrmse'] = val_upper_air_bias_lwrmse[j, k].item()
                    else:
                        for k, level in enumerate(self.valid_dataset.levels):
                            diagnostic_logs[f'valid_{var}_level{level:.3f}_bias_lwrmse'] = val_upper_air_bias_lwrmse[j, k].item()
                if self.params.has_diagnostic:
                    for j, var in enumerate(self.valid_dataset.diagnostic_variables):
                        diagnostic_logs[f'valid_{var}_bias_lwrmse'] = val_diagnostic_bias_lwrmse[j].item()

        print(f"diagnostic_logs: {diagnostic_logs}")
        if self.params.log_to_wandb:
            wandb_log_dict = dict(diagnostic_logs)

            if save_plots_this_epoch:
                if self.params.diagnostic_gif and gif_filename:
                    wandb_log_dict["Evolution_GIF"] = wandb.Video(gif_filename)

                if self.params.diagnostic_spectra and spectra_filename:
                    wandb_log_dict["power_spectrum_plot"] = wandb.Image(spectra_filename)

            wandb.log(wandb_log_dict, step=self.wandb_step)
        
        valid_time = time.time() - valid_start
        self.model.eval()
        return valid_time, diagnostic_logs

    def _load_ensemble_init_data(self, ensemble_params):
        """
        Parse the events JSON (new format: filepath → list of init_datetimes).

        New JSON format::

            {event_type: {filepath: [init_dt_str1, init_dt_str2, ...]}}

        Each (filepath, init_datetime) pair becomes one independent particle.

        Returns:
            tuple: (init_data, init_nc_filepaths, init_datetimes,
                    event_type_mapping, particle_idx_within_event)

            - init_data:               Raw JSON dict.
            - init_nc_filepaths:       List[str] – filepath repeated once per
                                       init_datetime it contributes.
            - init_datetimes:          List of cftime datetimes – one per particle.
            - event_type_mapping:      Dict[int, str] – particle_idx → event_type.
            - particle_idx_within_event: Dict[int, int] – particle_idx → per-event-type counter.
        """
        init_data: dict = {}
        init_nc_filepaths: list = []
        init_datetimes: list = []
        event_type_mapping: dict = {}
        particle_idx_within_event: dict = {}

        if not (hasattr(self.params, 'init_nc_filepath_files')
                and self.params.init_nc_filepath_files):
            ensemble_params['init_nc_filepaths'] = []
            ensemble_params['init_datetimes'] = []
            ensemble_params['save_basenames_obs'] = []
            ensemble_params['output_dirs_obs'] = []
            return init_data, init_nc_filepaths, init_datetimes, event_type_mapping, particle_idx_within_event

        with open(self.params.init_nc_filepath_files, 'r') as f:
            init_data = json.load(f)

        has_year_zero = getattr(self.params, 'has_year_zero', False)
        particle_idx = 0

        for event_type, event_data in init_data.items():
            if not isinstance(event_data, dict):
                continue
            particle_idx_in_event = 0
            for filepath, datetime_list in event_data.items():
                if not isinstance(datetime_list, list) or len(datetime_list) == 0:
                    continue
                for init_dt_str in datetime_list:
                    try:
                        raw_dt = cftime.datetime.strptime(
                            str(init_dt_str), "%Y-%m-%d %H:%M:%S",
                            has_year_zero=has_year_zero,
                            calendar=self.params.calendar,
                        )
                        init_dt = self.valid_dataset.datetime_class(
                            raw_dt.year, raw_dt.month, raw_dt.day,
                            hour=raw_dt.hour, has_year_zero=has_year_zero,
                        )
                    except Exception as e:
                        logging.error(
                            f"Error parsing init_datetime '{init_dt_str}' "
                            f"for {filepath}: {e}"
                        )
                        continue

                    init_nc_filepaths.append(filepath)
                    init_datetimes.append(init_dt)
                    event_type_mapping[particle_idx] = event_type
                    particle_idx_within_event[particle_idx] = particle_idx_in_event
                    particle_idx += 1
                    particle_idx_in_event += 1

        ensemble_params['init_nc_filepaths'] = init_nc_filepaths
        ensemble_params['init_datetimes'] = init_datetimes
        ensemble_params['save_basenames_obs'] = []
        ensemble_params['output_dirs_obs'] = []

        if self.world_rank == 0:
            logging.info(
                f"Loaded {len(init_nc_filepaths)} particles from "
                f"{self.params.init_nc_filepath_files} "
                f"({len(init_data)} event type(s))"
            )
            for p_idx, (fp, dt) in enumerate(zip(init_nc_filepaths, init_datetimes)):
                et = event_type_mapping.get(p_idx, 'unknown')
                logging.info(f"  particle {p_idx}: [{et}] {fp}  init={dt}")

        return (init_data, init_nc_filepaths, init_datetimes,
                event_type_mapping, particle_idx_within_event)

    def _build_ensemble_params(self):
        """
        Deep-copy self.params into ensemble_params and set all ensemble-specific overrides.
        Parse ensemble_inference_hours into a list.

        Returns:
            tuple: (ensemble_params, inference_hours_list, error_metrics_list)
        """
        # Create a copy of params for ensemble inference
        ensemble_params = copy.deepcopy(self.params)

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

        # Set save_forecasts flag (default to False if not specified)
        if not hasattr(ensemble_params, 'save_forecasts'):
            ensemble_params['save_forecasts'] = getattr(self.params, 'save_forecasts', False)

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

        # Set number of validation ensemble members per prediction
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

        return ensemble_params, inference_hours_list, error_metrics_list

    def _setup_obs_functions(self):
        """
        Import observable functions from utils.observations and build the per-function argument lists.

        Returns:
            tuple: (obs_functions, obs_args_list, obs_function_names)
        """
        obs_functions = []
        obs_args_list = []
        obs_function_names = []

        if hasattr(self.params, 'obs_functions') and len(self.params.obs_functions) > 0:
            raw_obs = self.params.obs_functions
            if isinstance(raw_obs, str):
                obs_function_names = [f.strip() for f in raw_obs.split(',')]
            else:
                obs_function_names = [str(f).strip() for f in raw_obs]

            # Use obs_args_dict if provided (already parsed as dictionary in __main__)
            # If it doesn't exist, default to empty dict
            obs_args_dict = {}
            if hasattr(self.params, 'obs_args_dict') and isinstance(self.params.obs_args_dict, dict):
                obs_args_dict = self.params.obs_args_dict

            # Import and set up observable functions
            # Mapping from old file-saving function names to their accumulator-based
            # _ts equivalents.  The old functions expect save_basename as the last
            # positional argument; the new pipeline passes an ObservationAccumulator
            # instead.  Redirect transparently so existing configs keep working.
            _ts_redirect = {
                'unweighted_nday_mean': 'unweighted_nday_mean_ts',
                'unweighted_nday_max':  'unweighted_nday_max_ts',
            }

            for obs_func_name in obs_function_names:
                try:
                    # Import from v2.0/utils/observations.py
                    from utils import observations as obs_mod

                    # Redirect legacy function names to their _ts counterparts.
                    redirected_name = _ts_redirect.get(obs_func_name)
                    if redirected_name is not None:
                        logging.info(
                            f"obs_function '{obs_func_name}' redirected to "
                            f"'{redirected_name}' for accumulator-based pipeline."
                        )
                        obs_func_name = redirected_name

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
                        # Functions needing [target_duration, var, regions, region_file_path]
                        _nday_funcs = (
                            'unweighted_nday_mean', 'unweighted_nday_max',
                            'unweighted_nday_mean_ts', 'unweighted_nday_max_ts',
                            'unweighted_nday_min_ts',
                            'unweighted_nday_mean_field_ts',
                            'unweighted_nday_max_field_ts',
                            'unweighted_nday_min_field_ts',
                        )
                        # Functions needing [var, regions, region_file_path] (no target_duration)
                        _spatial_funcs = (
                            'unweighted_spatial_mean_ts',
                            'unweighted_spatial_field_ts',
                        )
                        if obs_func_name in _nday_funcs:
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
                        elif obs_func_name in _spatial_funcs:
                            # Expected order: [var, regions, region_file_path]
                            required_keys = ['var', 'regions', 'region_file_path']
                            if all(key in obs_args_dict for key in required_keys):
                                obs_args_list.append([
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

        return obs_functions, obs_args_list, obs_function_names

    def _ensemble_truth_combined_exists(self, save_basename_truth) -> bool:
        """
        Return True if any *_truth_combined.nc files matching save_basename_truth glob exist.
        Log the found files if self.world_rank == 0.
        """
        truth_glob = os.path.join(os.path.dirname(save_basename_truth), f"{os.path.basename(save_basename_truth)}_*_truth_combined.nc")
        truth_files = glob.glob(truth_glob)
        if len(truth_files) > 0:
            if self.world_rank == 0:
                logging.info(f"Found {len(truth_files)} existing truth combined file(s):")
                for tf in truth_files:
                    logging.info(f"  - {os.path.basename(tf)}")
            return True
        return False

    def _compute_truth_observables(self, accumulator, ensemble_params, init_nc_filepaths,
                                   init_datetimes, event_type_mapping, obs_function_names,
                                   obs_args_list, max_forecast_hours):
        """Compute truth observable time series and store in *accumulator* (rank-0 only).

        Uses :func:`~utils.observations.truth_file_worker_ts` in a spawned
        ``ProcessPoolExecutor``.  After all workers finish the method calls
        ``dist.barrier`` so all ranks stay in sync.
        """
        if self.world_rank == 0 and obs_function_names:
            try:
                from utils import observations as obs_mod
                n_particles = len(init_nc_filepaths)
                max_forecast_steps = max_forecast_hours // self.params.timedelta_hours

                local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
                n_workers = min(
                    max((os.cpu_count() or 1) // max(local_world_size, 1), 1),
                    n_particles,
                    16,
                )
                logging.info(
                    f"Computing truth observables for {n_particles} particles "
                    f"using {n_workers} worker processes "
                    f"(LOCAL_WORLD_SIZE={local_world_size}, cpu_count={os.cpu_count()})"
                )

                _spawn_ctx = multiprocessing.get_context('spawn')
                futures_map: dict = {}
                with concurrent.futures.ProcessPoolExecutor(
                        max_workers=n_workers, mp_context=_spawn_ctx) as executor:
                    for p_idx, (filepath, init_dt) in enumerate(
                            zip(init_nc_filepaths, init_datetimes)):
                        event_type = event_type_mapping.get(p_idx, 'unknown')
                        future = executor.submit(
                            obs_mod.truth_file_worker_ts,
                            p_idx, filepath, init_dt, event_type, p_idx,
                            max_forecast_steps, self.params.timedelta_hours,
                            obs_function_names, obs_args_list,
                        )
                        futures_map[future] = p_idx

                    for future in concurrent.futures.as_completed(futures_map):
                        p_idx = futures_map[future]
                        filepath = init_nc_filepaths[p_idx]
                        event_type = event_type_mapping.get(p_idx, 'unknown')
                        try:
                            _, success, open_error, result_dict = future.result()
                        except Exception as exc:
                            logging.error(
                                f"Truth worker error for particle {p_idx} ({filepath}): "
                                f"{exc}\n{traceback.format_exc()}"
                            )
                            continue
                        if not success:
                            logging.warning(
                                f"Could not open truth file for particle {p_idx} "
                                f"({filepath}): {open_error}"
                            )
                            continue
                        accumulator.add_truth_from_worker_result(result_dict, event_type)
                        logging.info(
                            f"Truth particle {p_idx} done "
                            f"(event_type={event_type}, qualifiers={list(result_dict.keys())})"
                        )

                logging.info("Finished computing truth observables.")
            except Exception as exc:
                logging.error(f"Error computing truth observables: {exc}")
                logging.error(traceback.format_exc())

        if dist.is_initialized():
            dist.barrier(device_ids=[self.device])

    def _make_ensemble_obs_function_ts(self, event_type_mapping, obs_functions,
                                        obs_args_list, obs_function_names, accumulator):
        """Return a callable that feeds all *_ts obs functions with the accumulator.

        The returned closure has the signature expected by
        ``Stepper.predict(obs_function=...)``:

        .. code-block:: text

            obs_fn([ensemble_datasets, particle_idxs, ensemble_start, ensemble_end])

        Each *_ts function receives::

            (datasets, particle_idxs, ens_start, ens_end,
             event_type, *obs_args, accumulator)
        """
        def _obs_fn(args):
            if len(args) < 4:
                logging.error(
                    f"Invalid obs_fn args: expected >= 4, got {len(args)}")
                return
            ensemble_datasets, particle_idxs, ensemble_start, ensemble_end = (
                args[0], args[1], args[2], args[3])

            # Normalise particle_idxs
            if isinstance(particle_idxs, torch.Tensor):
                pids = particle_idxs.cpu().numpy().tolist()
            elif isinstance(particle_idxs, (list, np.ndarray)):
                pids = [int(p) for p in particle_idxs]
            else:
                pids = [int(particle_idxs)]

            if not pids:
                return

            # Compute per-particle event types (particles in a batch may span
            # multiple event types at the boundary between event groups).
            event_types = [event_type_mapping.get(pid, 'unknown') for pid in pids]

            for obs_func, obs_args, obs_name in zip(
                    obs_functions, obs_args_list, obs_function_names):
                try:
                    func_args = (
                        [ensemble_datasets, pids, ensemble_start, ensemble_end, event_types]
                        + list(obs_args)
                        + [accumulator]
                    )
                    obs_func(tuple(func_args))
                except Exception as exc:
                    logging.error(f"Error in obs function '{obs_name}': {exc}")
                    logging.error(traceback.format_exc())

        return _obs_fn

    def _run_ensemble_from_init_conditions(self, max_forecast_hours, ensemble_params,
                                           init_datetimes, event_type_mapping,
                                           obs_functions, obs_args_list, obs_function_names,
                                           accumulator):
        """Run one Stepper for all particles for *max_forecast_hours* steps.

        The obs function closure writes accumulated time-series data directly
        into *accumulator* (no file I/O for observables).
        """
        current_ensemble_params = copy.deepcopy(ensemble_params)
        current_ensemble_params['ensemble_inference_hours'] = max_forecast_hours
        current_ensemble_params['ensemble_inference_steps'] = (
            max_forecast_hours // self.params.timedelta_hours)
        current_ensemble_params['init_datetimes'] = init_datetimes
        current_ensemble_params['save_basenames_obs'] = []
        current_ensemble_params['output_dirs_obs'] = []

        if getattr(self.params, 'save_forecasts', False):
            if hasattr(self.params, 'output_dir'):
                forecast_base_dir = self.params.output_dir
            else:
                forecast_base_dir = os.path.join(
                    self.params.experiment_dir, 'ensemble_validation', 'forecasts')
            forecast_dir = os.path.join(forecast_base_dir, f"{max_forecast_hours}h")
            os.makedirs(forecast_dir, exist_ok=True)
            current_ensemble_params['save_basenames'] = [
                os.path.join(
                    forecast_dir,
                    f"forecast_{event_type_mapping.get(i, 'unknown')}_particle{i:04d}_epoch{self.epoch:04d}"
                )
                for i in range(len(init_datetimes))
            ]
            current_ensemble_params['output_dirs'] = [forecast_dir] * len(init_datetimes)
        else:
            current_ensemble_params['save_basenames'] = []
            current_ensemble_params['output_dirs'] = []

        # Prevent Stepper from loading a stale checkpoint (weights will be overwritten)
        for attr in ('best_checkpoint_path',):
            try:
                del current_ensemble_params.params[attr]
            except (KeyError, AttributeError):
                pass
            try:
                delattr(current_ensemble_params, attr)
            except AttributeError:
                pass

        stepper = Stepper(
            [current_ensemble_params], self.world_rank,
            use_6h_24h_model=False, async_save=False)

        # Copy current training weights into stepper.
        # Prefer EMA weights for inference (consistent with _get_model_for_eval).
        if self._ema_active():
            training_sd = self.ema.state_dict()
            if self.world_rank == 0:
                logging.info("Using EMA model weights for ensemble inference.")
        else:
            training_sd = (self.model.module.state_dict()
                           if hasattr(self.model, 'module')
                           else self.model.state_dict())
        if hasattr(stepper.model, 'module'):
            stepper.model.module.load_state_dict(training_sd, strict=True)
        else:
            stepper.model.load_state_dict(training_sd, strict=True)

        if hasattr(self, 'integrator') and hasattr(stepper, 'integrator'):
            stepper.integrator = self.integrator

        if dist.is_initialized():
            dist.barrier(device_ids=[self.device])

        try:
            if obs_functions:
                obs_fn = self._make_ensemble_obs_function_ts(
                    event_type_mapping, obs_functions, obs_args_list,
                    obs_function_names, accumulator)
                stepper.predict(obs_function=obs_fn, obs_args=[])
            else:
                logging.warning("No obs functions loaded; running without observables.")
                stepper.predict()
            if self.world_rank == 0:
                logging.info(
                    f"Completed ensemble forecast for {max_forecast_hours} hours.")
        except Exception as exc:
            logging.error(f"Error in ensemble forecast: {exc}")
            logging.error(traceback.format_exc())

    def _combine_and_log_ensemble_metrics(self, accumulator, error_metrics_list,
                                          output_nc_path):
        """Compute error metrics from *accumulator*, log to wandb, save to .nc.

        Runs on rank-0 only (caller's responsibility).
        """
        try:
            if error_metrics_list:
                error_results = accumulator.compute_errors(error_metrics_list)

                if getattr(self.params, 'log_to_wandb', False) and error_results:
                    wandb_logs: dict = {}
                    metrics_to_define: set = set()

                    for qualifier, et_dict in error_results.items():
                        q_safe = qualifier.replace('/', '_').replace('\\', '_')
                        for event_type, metrics_dict in et_dict.items():
                            for metric, errors in metrics_dict.items():
                                if not isinstance(errors, np.ndarray):
                                    continue
                                for d, err_val in enumerate(errors):
                                    if not np.isfinite(float(err_val)):
                                        continue
                                    lead_hours = (d + 1) * 24
                                    mn = f"{event_type}_{q_safe}_{lead_hours}h_{metric}"
                                    metrics_to_define.add(mn)
                                    wandb_logs[mn] = float(err_val)

                    if not hasattr(self, '_defined_wandb_metrics'):
                        self._defined_wandb_metrics = set()
                    for mn in metrics_to_define:
                        if mn not in self._defined_wandb_metrics:
                            try:
                                wandb.define_metric(mn, step_metric="epoch")
                                self._defined_wandb_metrics.add(mn)
                            except Exception:
                                pass

                    if wandb_logs:
                        wandb.log(wandb_logs, step=self.wandb_step)
                        logging.info(
                            f"Logged {len(wandb_logs)} ensemble error metrics to wandb.")

            if output_nc_path:
                logging.info(f"Saving ensemble validation data to {output_nc_path}")
                accumulator.save(output_nc_path)
                logging.info("Ensemble validation data saved.")

        except Exception as exc:
            logging.warning(f"Error in _combine_and_log_ensemble_metrics: {exc}")
            logging.warning(traceback.format_exc())

    def validate_ensemble_forecast(self):
        """Run ensemble forecast validation using the time-series observable paradigm.

        For each (filepath, init_datetime) particle defined in the events JSON:

        1. Truth observable time series are computed via parallel workers.
        2. One Stepper run of *max_forecast_hours* steps is executed for all
           particles.
        3. The *_ts obs functions compute rolling time series at each daily step
           and store them in an :class:`~utils.observations.ObservationAccumulator`.
        4. Error metrics are computed as a function of lead time and logged to
           wandb.
        5. All data (truth, forecast, error metrics) are saved to a .nc file.

        Returns:
            float: Total time taken in seconds.
        """
        if not getattr(self.params, 'ensemble_validation', False):
            return 0.0

        # Join any metrics-computation thread left over from the previous call
        # before touching shared logging / wandb state again.
        if self.world_rank == 0:
            prev_thread = getattr(self, '_metrics_thread', None)
            if prev_thread is not None and prev_thread.is_alive():
                logging.info("Waiting for previous metrics thread to finish...")
                prev_thread.join()

        ensemble_val_start = time.time()
        if self.world_rank == 0:
            logging.info(
                "Starting ensemble forecast validation (time-series paradigm)...")

        try:
            ensemble_params, inference_hours_list, error_metrics_list = (
                self._build_ensemble_params())

            (init_data, init_nc_filepaths, init_datetimes,
             event_type_mapping, particle_idx_within_event) = (
                self._load_ensemble_init_data(ensemble_params))

            if not init_nc_filepaths:
                if self.world_rank == 0:
                    logging.warning(
                        "No particles found in events JSON. "
                        "Skipping ensemble validation.")
                return time.time() - ensemble_val_start

            if not inference_hours_list:
                if self.world_rank == 0:
                    logging.warning(
                        "ensemble_inference_hours not set. "
                        "Skipping ensemble validation.")
                return time.time() - ensemble_val_start

            max_forecast_hours = max(inference_hours_list)

            obs_functions, obs_args_list, obs_function_names = (
                self._setup_obs_functions())

            from utils.observations import ObservationAccumulator
            accumulator = ObservationAccumulator()

            # Step 1: truth observables (rank-0 + barrier)
            self._compute_truth_observables(
                accumulator, ensemble_params, init_nc_filepaths, init_datetimes,
                event_type_mapping, obs_function_names, obs_args_list,
                max_forecast_hours)

            # Step 2: ensemble forecast (all ranks)
            if self.world_rank == 0:
                logging.info(
                    f"Running ensemble forecast for {max_forecast_hours} hours "
                    f"from {len(init_nc_filepaths)} particle(s)...")
            self._run_ensemble_from_init_conditions(
                max_forecast_hours, ensemble_params, init_datetimes,
                event_type_mapping, obs_functions, obs_args_list,
                obs_function_names, accumulator)

            # Gather forecast observations from all DDP ranks to rank 0.
            # Each rank only processes a subset of particles (via DistributedSampler),
            # so we must collect all subsets before saving.
            if dist.is_initialized() and dist.get_world_size() > 1:
                world_size = dist.get_world_size()
                if self.world_rank == 0:
                    all_forecast_obs = [None] * world_size
                else:
                    all_forecast_obs = None
                dist.gather_object(
                    accumulator.forecast_obs, all_forecast_obs, dst=0)
                if self.world_rank == 0:
                    for rank_fc_obs in all_forecast_obs[1:]:
                        for qualifier, et_dict in rank_fc_obs.items():
                            for event_type, pid_dict in et_dict.items():
                                for particle_idx, ens_dict in pid_dict.items():
                                    for ens_member, values in ens_dict.items():
                                        accumulator.add_forecast(
                                            qualifier, event_type,
                                            int(particle_idx), ens_member,
                                            values)
                    logging.info(
                        f"Gathered forecast obs from {world_size} DDP ranks.")

            if dist.is_initialized():
                dist.barrier(device_ids=[self.device])

            # Step 3: metrics + save (rank-0 only).
            # _combine_and_log_ensemble_metrics can take many minutes (much
            # longer than the NCCL watchdog timeout of 600 s).  Running it
            # inline while other ranks wait at a barrier would cause an NCCL
            # timeout.  Instead we launch it in a daemon thread so rank 0
            # reaches the barrier immediately and all ranks exit this function
            # together.  The thread is joined at the next entry to this
            # function (see above) before any shared state is touched again.
            if self.world_rank == 0:
                logging.info("Computing error metrics and saving results "
                             "(background thread)...")
                ens_val_dir = os.path.join(
                    self.params.experiment_dir, 'ensemble_validation')
                os.makedirs(ens_val_dir, exist_ok=True)
                output_nc_path = os.path.join(
                    ens_val_dir,
                    f"ensemble_validation_epoch{self.epoch:04d}.nc")
                self._metrics_thread = threading.Thread(
                    target=self._combine_and_log_ensemble_metrics,
                    args=(accumulator, error_metrics_list, output_nc_path),
                    daemon=True,
                )
                self._metrics_thread.start()

            # All ranks (including rank 0 which just started the thread) sync
            # here.  The barrier completes in milliseconds, well within the
            # NCCL watchdog timeout.
            if dist.is_initialized():
                dist.barrier(device_ids=[self.device])

        except Exception as exc:
            logging.error(f"Error in ensemble forecast validation: {exc}")
            logging.error(traceback.format_exc())

        return time.time() - ensemble_val_start


    def prepare_preds(self, preds, acc = False):
        preds = preds.rename({'time': 'lead_time'})
        # If bug, change this back to values[0]
        preds['time'] = preds.lead_time.values[0:1]
        preds = preds.set_coords('time')
        if acc:
            # For ACC, use all time steps
            actual_lead_times = range(1, len(preds.lead_time) + 1)
        else:
            # For non-ACC, use provided lead_times or fall back to forecast_lead_times
            actual_lead_times = preds.lead_times if preds.lead_times is not None else self.params['forecast_lead_times']

        preds['lead_time'] = [lt * self.params['timedelta_hours'] for lt in actual_lead_times]
        return preds
    
    def convert_to_xarray(self, surface_prediction, upper_air_prediction, start_times, params, valid_dataset, acc=False, diagnostic_prediction=None, lead_times=None):
        batch_size, time_steps, num_surface_vars, lat, lon = surface_prediction.shape
        datasets = []

        for sample in range(batch_size):
            if acc:
                # For ACC, create time_range for all time steps
                time_range = [start_times[sample] + timedelta(hours=step * params['timedelta_hours']) for step in range(1, time_steps + 1)]
            else:
                # For specific lead times, use provided lead_times or fall back to forecast_lead_times
                actual_lead_times = lead_times if lead_times is not None else params['forecast_lead_times']
                time_range = [start_times[sample] + timedelta(hours=lt * params['timedelta_hours']) for lt in actual_lead_times]

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
                    dataset[var] = da

            for idx, var in enumerate(valid_dataset.upper_air_variables):
                da = xr.DataArray(
                    data=upper_air_prediction[sample, :, idx],
                    dims=["time", level_coord_name, "lat", "lon"],
                    coords=coordinates
                )
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
            'wandb_step': self.wandb_step,
            'ema_state': self.ema.state_dict() if self.ema is not None else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
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
        
        # Get model state dict keys (accounting for DDP wrapper)
        model_state_dict = self.model.state_dict()
        checkpoint_state_dict = checkpoint['model_state']
        
        # When start_epoch is set, require strict match so config/model mismatch fails clearly (e.g. validate_before_train).
        strict_load = getattr(self.params, 'start_epoch', None) is not None
        
        # Check if checkpoint keys have "module." prefix
        checkpoint_has_module = any(key.startswith('module.') for key in checkpoint_state_dict.keys())
        # Check if model keys have "module." prefix
        model_has_module = any(key.startswith('module.') for key in model_state_dict.keys())
        
        # Try loading directly first
        try:
            result = self.model.load_state_dict(checkpoint_state_dict, strict=strict_load)
            if not strict_load and (getattr(result, 'missing_keys', None) or getattr(result, 'unexpected_keys', None)):
                mk, uk = getattr(result, 'missing_keys', []), getattr(result, 'unexpected_keys', [])
                # Do not silently accept a partial load — mismatched keys almost always indicate
                # a "module." DDP prefix mismatch.  Raise so the prefix-fix fallback runs.
                raise RuntimeError(
                    f"Partial load: {len(mk)} missing keys and {len(uk)} unexpected keys. "
                    "Attempting 'module.' prefix fix."
                )
            else:
                logging.info('Successfully loaded checkpoint state dict')
        except Exception as e:
            logging.warning(f'Direct load failed: {e}. Attempting to fix "module." prefix mismatch...')
            
            # Handle "module." prefix mismatch
            new_state_dict = OrderedDict()
            
            if checkpoint_has_module and not model_has_module:
                # Checkpoint has "module." prefix but model doesn't - remove it
                for key, val in checkpoint_state_dict.items():
                    if key.startswith('module.'):
                        new_key = key[7:]  # Remove "module." prefix
                        new_state_dict[new_key] = val
                    else:
                        new_state_dict[key] = val
                logging.info('Removed "module." prefix from checkpoint keys')
            elif not checkpoint_has_module and model_has_module:
                # Checkpoint doesn't have "module." prefix but model does - add it
                for key, val in checkpoint_state_dict.items():
                    new_key = 'module.' + key
                    new_state_dict[new_key] = val
                logging.info('Added "module." prefix to checkpoint keys')
            else:
                # Both have or both don't have - try removing anyway as fallback
                for key, val in checkpoint_state_dict.items():
                    if key.startswith('module.'):
                        new_key = key[7:]
                        new_state_dict[new_key] = val
                    else:
                        new_state_dict[key] = val
                logging.info('Attempting to remove "module." prefix as fallback')
            
            # Try loading the modified state dict
            try:
                self.model.load_state_dict(new_state_dict, strict=strict_load)
                logging.info('Successfully loaded checkpoint after fixing "module." prefix')
            except Exception as e2:
                logging.error(f'Failed to load checkpoint even after fixing "module." prefix: {e2}')
                raise e2
        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch']
        logging.info(f'Restoring from epoch {self.startEpoch}, iteration {self.iters}')

        ema_warmup = getattr(self.params, 'ema_warmup_epochs', 0)
        ema_was_active_at_checkpoint = (self.ema is not None) and (self.startEpoch >= ema_warmup)
        if ema_was_active_at_checkpoint and 'ema_state' in checkpoint and checkpoint['ema_state'] is not None:
            self.ema.load_state_dict(checkpoint['ema_state'])
            logging.info("Restored EMA state")
        elif self.ema is not None and not ema_was_active_at_checkpoint:
            # EMA warmup not yet complete; EMA weights will be synced from model at warmup boundary
            logging.info(f"EMA warmup not complete at checkpoint epoch {self.startEpoch} "
                         f"(ema_warmup_epochs={ema_warmup}). EMA state not loaded.")
        elif self.ema is not None:
            # Checkpoint has no EMA state but EMA is enabled: initialise from model weights
            logging.warning("No EMA state in checkpoint. Initializing EMA from loaded model weights.")
            model_for_ema = self.model.module if hasattr(self.model, 'module') else self.model
            update_ema(self.ema, model_for_ema, decay=0)  # decay=0 means direct copy
            
        # Restore optimizer and scheduler state only when resuming training (not finetuning)
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
            logging.info("Restored optimizer state")
            
            # Load scheduler state if available
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logging.info("Restored scheduler state")
            else:
                logging.warning("Scheduler state not found in checkpoint, scheduler will start fresh")

def seed_torch(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def setup_distributed(params: YParams, args: argparse.Namespace) -> tuple[int, int]:
    """Return (world_rank, local_rank).
    
    Always initializes the distributed process group, even for single-GPU
    execution. This ensures that DDP wrapping, SyncBatchNorm, and all
    dist-dependent code paths behave identically regardless of whether the
    script is launched with ``torchrun`` or plain ``python``.
    """
    if params['world_size'] > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        seed = args.global_seed * dist.get_world_size() + rank
        params['global_batch_size'] = params['batch_size'] * params['world_size']
    else:
        # Single-GPU mode: still initialize a (trivial) process group so that
        # dist.is_initialized() returns True and all downstream code (DDP
        # wrapping, SyncBatchNorm conversion, all_reduce calls, etc.) follows
        # the same path as multi-GPU execution.
        rank = 0
        device = 0
        seed = args.global_seed
        params['global_batch_size'] = params['batch_size']
        
        # Ensure required environment variables are set for the process group
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '29500')
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=1, rank=0)
        logging.info("Initialized single-GPU process group (nccl) so that "
                      "DDP and dist-dependent code paths are consistent.")
    
    seed_torch(seed)
    params['seed'] = seed - rank
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
    #if params.just_validate:
    #    os.environ["WANDB_MODE"] = "offline"  # Disable wandb for validation-only runs
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
    if hasattr(params, "finetune_num_epochs"):
        if params.finetune_num_epochs > 0 and params.finetune_run_num is None:
            raise ValueError("Finetuning epochs specified but finetuning run number is not specified")
    if args.finetune_lr > 0:
        params['lr'] = args.finetune_lr  # Override learning rate for finetuning
    params['curriculum_learning'] = params['curriculum_learning'] if hasattr(params, 'curriculum_learning') else False
    params['balanced_learning'] = params['balanced_learning'] if hasattr(params, 'balanced_learning') else False


    # Curriculum learning configuration
    if hasattr(params, "curriculum_learning"):
        if params.curriculum_learning:
            if hasattr(params, "balanced_learning"):
                if params.balanced_learning:
                    print('Using balanced learning, setting curriculum_learning_fraction to 0.5 and train_date_range = None')
                    params['curriculum_learning_fraction'] = 0.5
                    params['train_date_range'] = None
    
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
        elif not hasattr(params, 'obs_functions') or len(params.obs_functions) == 0:
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
        # Finetuning mode: save to finetune_run_num directory, load from original run_num.
        # load_exp_dir can be set independently when the original checkpoint lives under
        # a different base directory than the finetuned model (e.g. exp_dir is the
        # finetuning output dir but the source checkpoint is in 'results/').
        save_expDir = os.path.join(params.exp_dir, args.config, str(params.finetune_run_num))
        load_base = params.load_exp_dir if hasattr(params, 'load_exp_dir') else params.exp_dir
        load_expDir = os.path.join(load_base, args.config, str(args.run_num))
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


    # Set default config values for plot saving
    if not hasattr(params, 'plot_save_interval'):
        params['plot_save_interval'] = 10  # Save plots every N epochs
    if not hasattr(params, 'max_plots_to_keep'):
        params['max_plots_to_keep'] = 5   # Keep only M most recent plot sets
    
    if world_rank == 0:
        os.makedirs(params['experiment_dir'], exist_ok=True)
        os.makedirs(params['checkpoint_dir_save'], exist_ok=True)
        os.makedirs(params['spectra_dir'], exist_ok=True)
        os.makedirs(params['acc_dir'], exist_ok=True)
        os.makedirs(params['gif_dir'], exist_ok=True)
        if params.long_validation:
            os.makedirs(params['bias_dir'], exist_ok=True)
        os.makedirs(params['validation_data_dir'], exist_ok=True)
    
    # Synchronize all ranks after directory creation
    if dist.is_initialized():
        dist.barrier(device_ids=[local_rank])

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
    
    # Run validation before training if requested and resuming/finetuning
    if not params.just_validate and params.validate_before_train:
        if params.resuming or params.finetuning:
            if world_rank == 0:
                logging.info(
                    "Running validation before training (resuming/finetuning mode)... "
                    "Using checkpoint loaded in setup_model (epoch=%s).",
                    getattr(trainer, 'epoch', getattr(trainer, 'startEpoch', '?')),
                )
            # Run ensemble validation first if enabled
            if hasattr(params, 'ensemble_validation') and params.ensemble_validation:
                ensemble_val_time = trainer.validate_ensemble_forecast()
                if world_rank == 0:
                    logging.info(f"Pre-training ensemble validation time: {ensemble_val_time:.2f} seconds")
            trainer.validate_one_epoch()
    
    # Execute training or validation based on mode
    if not params.just_validate:
        # Normal training mode: run full training loop
        trainer.train()
    else:
        # Validation-only mode: run validation without training
        if len(params.validation_epochs) == 0:
            # No specific epochs specified: validate best checkpoint
            # Run ensemble validation first if enabled
            if hasattr(params, 'ensemble_validation') and params.ensemble_validation:
                ensemble_val_time = trainer.validate_ensemble_forecast()
                logging.info(f"Ensemble validation time: {ensemble_val_time:.2f} seconds")
            trainer.validate_one_epoch()
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
                # Run ensemble validation first if enabled (for each epoch)
                if hasattr(params, 'ensemble_validation') and params.ensemble_validation:
                    ensemble_val_time = trainer.validate_ensemble_forecast()
                    logging.info(f"Epoch {ckpt_i} ensemble validation time: {ensemble_val_time:.2f} seconds")
                # Run validation for this checkpoint
                trainer.validate_one_epoch()
    
    # Training/validation complete.
    # Join any pending background metrics-saving thread before exiting so that
    # the ensemble validation nc file is fully written (daemon threads are killed
    # on process exit, which silently drops in-progress saves).
    if world_rank == 0:
        final_thread = getattr(trainer, '_metrics_thread', None)
        if final_thread is not None and final_thread.is_alive():
            logging.info("Waiting for metrics thread to finish saving ensemble validation results...")
            final_thread.join()

    logging.info('DONE ---- rank %d' % world_rank)

    if dist.is_initialized():
        dist.destroy_process_group()  # cleanup.