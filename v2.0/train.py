import uuid
import copy
import matplotlib.pyplot as plt

from copy import deepcopy
from multiprocessing import Process
from tqdm import tqdm
from datetime import timedelta
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
from collections import OrderedDict

import pandas as pd
import wandb
import os, glob
import time
from natsort import natsorted
import numpy as np
import argparse
import xarray as xr
import logging
import torch
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from utils import logging_utils
from utils.power_spectrum import (
    zonal_averaged_power_spectrum,
    plot_bias,
    plot_power_spectrum,
    make_gif
)
from utils.data_loader_multifiles import get_data_loader
from utils.YParams import YParams
from utils.metrics import create_metrics_aggregator_new
from utils.perturbation import Perturber
from utils.integrate import Integrator

from utils.losses import Latitude_weighted_MSELoss, Latitude_weighted_L1Loss, Masked_L1Loss,\
     Masked_MSELoss, Latitude_weighted_masked_L1Loss, Latitude_weighted_masked_MSELoss,\
     Latitude_weighted_CRPSLoss, Kl_divergence_gaussians
from utils.lr_scheduler_sfno import LinearWarmupCosineAnnealingLR
logging_utils.config_logger()

from networks.pangu import PanguModel_Plasim
from networks.pangu_legacy import PanguModel_Plasim as PanguModel_Plasim_Legacy
from networks.modulus_sfno.sfnonet import SphericalFourierNeuralOperatorNet_v2 as SFNO

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
    """Convert batch of M samples (M, ...) to a batch of (M*ens_members, ...)."""
    return (data.unsqueeze(1) * torch.ones(1, ens_members, *data.shape[1:]).to(data.device)).flatten(0, 1)

class Trainer():
    def __init__(self, params, world_rank):
        self.params = params
        self.world_rank = world_rank
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        ###Setup the initia epochs and iteration #######
        self.iters = 0
        self.startEpoch = 0
        # self.epoch = self.startEpoch
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

        if self.params.resuming:
            checkpoint_path = None
            
            # Only rank 0 determines which checkpoint to use
            if self.world_rank == 0:
                # For validation without specific epochs, prefer best checkpoint
                if self.params.just_validate and len(self.params.validation_epochs) == 0:
                    if os.path.isfile(self.params.best_checkpoint_path):
                        checkpoint_path = self.params.best_checkpoint_path
                        logging.info("Validation mode: using best checkpoint")
                
                # For training resume: priority is latest > numbered > best
                if checkpoint_path is None:
                    if os.path.isfile(self.params.latest_checkpoint_path):
                        checkpoint_path = self.params.latest_checkpoint_path
                    else:
                        checkpoint_paths = natsorted([
                            file for file in glob.glob(self.params.checkpoint_path_globstr) 
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
            
            # Broadcast checkpoint path from rank 0 to all ranks
            if dist.is_initialized():
                # Convert path to list of chars for broadcasting
                if self.world_rank == 0:
                    path_bytes = checkpoint_path.encode('utf-8')
                    path_len = torch.tensor([len(path_bytes)], dtype=torch.long, device='cuda')
                else:
                    path_len = torch.tensor([0], dtype=torch.long, device='cuda')
                
                dist.broadcast(path_len, src=0)
                
                if self.world_rank == 0:
                    path_tensor = torch.tensor(list(path_bytes), dtype=torch.uint8, device='cuda')
                else:
                    path_tensor = torch.zeros(path_len.item(), dtype=torch.uint8, device='cuda')
                
                dist.broadcast(path_tensor, src=0)
                checkpoint_path = bytes(path_tensor.cpu().tolist()).decode('utf-8')
            
            self.restore_checkpoint(checkpoint_path)
            logging.info(f"Rank {self.params.local_rank} Loaded checkpoint: {checkpoint_path}")
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
                    train=True
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
                    train=True
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
                epoch_metrics = ['lr', 'train_loss', 'valid_loss', 'valid_loss_sfc', 'valid_loss_upper_air']
                
                # Add RMSE and ACC metrics for variables/levels in diagnostic_acc_var_dict
                var_dict = getattr(self.params, 'diagnostic_acc_var_dict', {})
                for step in self.params.forecast_lead_times:
                    epoch_metrics.append(f"valid_loss_{step}step")
                    epoch_metrics.append(f"valid_lwrmse_sfc_{step}step")
                    epoch_metrics.append(f"valid_lwrmse_pl_{step}step")
                    
                    # Add per-variable ACC and RMSE metrics based on diagnostic_acc_var_dict
                    for var, levels in var_dict.items():
                        if not levels:  # Surface variable
                            epoch_metrics.append(f'valid_{var}_{step}step_acc')
                            epoch_metrics.append(f'valid_{var}_{step}step_rmse')
                        else:  # Upper air variable with levels
                            for level in levels:
                                epoch_metrics.append(f'valid_{var}_{int(level)}_{step}step_acc')
                                epoch_metrics.append(f'valid_{var}_{int(level)}_{step}step_rmse')
                
                # Bias metrics
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
            if self.ema:
                self.ema.eval()

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
                    # Save latest checkpoint (and numbered checkpoint at intervals)
                    self.save_checkpoint(self.params.latest_checkpoint_path, epoch)
                    
                    # Save best checkpoint only if validation improved
                    if is_best:
                        self.save_checkpoint(self.params.best_checkpoint_path)
                        logging.info(f'Find best checkpoint at {epoch}.')
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
        # if self.world_rank == 0:
        #     if self.params.diagnostic_acc:
        #         self.cleanup_acc_plots()
        #     if self.params.diagnostic_gif:
        #         self.cleanup_gifs()
        #     if self.params.diagnostic_spectra:
        #         self.cleanup_power_spectrum_plots()
        #     if self.long_validation:
        #         self.cleanup_bias()
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
            wandb.log({'lr': lr, 'epoch': self.epoch})
    
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
                        # Unscale gradients before clipping
                        self.scaler.unscale_(self.optimizer)
                        # Clip gradients to prevent explosion
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        # Clip gradients to prevent explosion
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                        self.optimizer.step()
                    if self.ema:
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
                        if self.world_rank == 0:
                            #wandb.log(diagnostic_logs, step=(self.epoch-1) * total_iterations + self.iters)
                            wandb.log(diagnostic_logs, step= self.iters)

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
        output_surface = 0 
        output_upper_air = 0 
        output_diagnostic = 0 
        loss = 0 
        loss_diagnostic = 0
        loss_pl = 0 
        loss_sfc = 0
        loss_vae = 0
        
        with autocast(device_type="cuda"):
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
                # Non-legacy model: accepts targets for VAE dual encoder architecture
                # Returns: (output_surface, output_upper_air, [output_diagnostic,] mu, sigma, mu2, sigma2)
                if self.params.has_diagnostic:
                    output_surface, output_upper_air, output_diagnostic, mu, sigma, mu2, sigma2 = self.model(
                        input_surface, constant_boundary_data, 
                        varying_boundary_data, input_upper_air, 
                        target_surface, target_upper_air, train=True)
                    loss_diagnostic = self.loss_obj_diagnostic(output_diagnostic, target_diagnostic)
                else: 
                    output_surface, output_upper_air, mu, sigma, mu2, sigma2 = self.model(
                        input_surface, constant_boundary_data, 
                        varying_boundary_data, input_upper_air, 
                        target_surface, target_upper_air, train=True)
                
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
        current_lr = self.optimizer.param_groups[0]['lr']
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
                    wandb.log(logs)
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
                wandb.log(logs)
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
    
    def _get_model_for_eval(self) -> torch.nn.Module:
        """Select and prepare model for validation (EMA if available)."""
        if self.ema is not None:
            model = self.ema
            if self.params.log_to_screen:
                logging.info('Using EMA model for validation')
        else:
            model = self.model.module if dist.is_initialized() else self.model
        # model = self.model.module if dist.is_initialized() else self.model
        model.eval()
        return model

    def _create_metrics_logs(self, results: dict) -> dict:
        """
        Create wandb-ready log entries from MetricsAggregator results.
        
        Only logs metrics for variables/levels specified in diagnostic_acc_var_dict.
        
        Args:
            results: Dictionary from MetricsAggregator.compute()
            
        Returns:
            Dictionary of metric name -> value for wandb logging
        """
        logs = {}
        var_dict = getattr(self.params, 'diagnostic_acc_var_dict', {})
        lead_times = self.params.forecast_lead_times
        
        # Get variable lists
        surface_vars = list(self.valid_dataset.surface_variables)
        upper_air_vars = list(self.valid_dataset.upper_air_variables)
        levels = list(self.valid_dataset.levels)
        
        for step_i, step in enumerate(lead_times):
            for var, var_levels in var_dict.items():
                if not var_levels:
                    # Surface variable
                    if var in surface_vars:
                        var_idx = surface_vars.index(var)
                        acc_val = results['acc_surface'][step_i, var_idx].item()
                        rmse_val = results['rmse_surface'][step_i, var_idx].item()
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
                                acc_val = results['acc_upper_air'][step_i, var_idx, level_idx].item()
                                rmse_val = results['rmse_upper_air'][step_i, var_idx, level_idx].item()
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
                if self.ensemble_validation:
                    val_data_dir = self.params.validation_data_dir
                    if self.world_rank == 0:
                        os.makedirs(val_data_dir, exist_ok=True)
                    if dist.is_initialized():
                        dist.barrier()
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
                        val_output_surface, val_output_upper_air, val_output_diagnostic, _, _ = model_to_eval(
                            val_input_surface, self.constant_boundary_data[[0]], val_varying_boundary_data, val_input_upper_air)
                    else:
                        val_output_surface, val_output_upper_air, _, _ = model_to_eval(val_input_surface, self.constant_boundary_data[[0]], 
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
                        })
            
            # Main validation loop
            for i, data in tqdm(enumerate(self.valid_data_loader, 0), total=nb, bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}', miniters=1):
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
                        # Forward pass
                        if self.params.has_diagnostic:
                            val_output_surface, val_output_upper_air, val_output_diagnostic, _, _ = model_to_eval(
                                val_input_surface, self.constant_boundary_data, val_varying_boundary_data[:, step], val_input_upper_air)
                        else:
                            val_output_surface, val_output_upper_air, _, _ = model_to_eval(
                                val_input_surface, self.constant_boundary_data, val_varying_boundary_data[:, step], val_input_upper_air)
                        
                        # Compute loss at forecast_lead_times
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
                            # Get timestamps for this step
                            timestamps = times.clone()
                            
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
        acc_rmse_logs = self._create_metrics_logs(metrics_results)
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

        # Log to wandb
        if self.params.log_to_wandb:
            wandb_log_dict = dict(diagnostic_logs)
            
            if save_plots_this_epoch:
                if self.params.diagnostic_gif and gif_filename:
                    wandb_log_dict["Evolution_GIF"] = wandb.Video(gif_filename)
            
                if self.params.diagnostic_spectra and spectra_filename:
                    wandb_log_dict["power_spectrum_plot"] = wandb.Image(spectra_filename)
            
            wandb.log(wandb_log_dict)
        
        valid_time = time.time() - valid_start
        self.model.eval()
        return valid_time, diagnostic_logs

    def prepare_preds(self, preds, acc=False, lead_times=None):
        preds = preds.rename({'time': 'lead_time'})
        # If bug, change this back to values[0]
        preds['time'] = preds.lead_time.values[0:1]
        preds = preds.set_coords('time')
        if acc:
            # For ACC, use all time steps
            actual_lead_times = range(1, len(preds.lead_time) + 1)
        else:
            # For non-ACC, use provided lead_times or fall back to forecast_lead_times
            actual_lead_times = lead_times if lead_times is not None else self.params['forecast_lead_times']

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
            'ema_state': self.ema.state_dict() if self.ema is not None else None,  # <-- ADD THIS
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,  # <-- ADD THIS
        }

        # Always save latest checkpoint
        torch.save(checkpoint_data, self.params.latest_checkpoint_path)
        logging.info(f"Saved latest checkpoint: {self.params.latest_checkpoint_path}")

        # Save numbered checkpoint at intervals
        if epoch >= 0 and (epoch + 1) % self.params.checkpoint_save_interval == 0:
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
        checkpoint_paths = natsorted([
            file for file in glob.glob(self.params.checkpoint_path_globstr) 
            if os.path.isfile(file)
        ])
        
        num_to_delete = len(checkpoint_paths) - self.params.max_checkpoints_to_keep
        if num_to_delete > 0:
            for old_ckpt in checkpoint_paths[:num_to_delete]:
                os.remove(old_ckpt)
                logging.info(f"Removed old checkpoint: {old_ckpt}")

    def restore_checkpoint(self, checkpoint_path):
        """ We intentionally require a checkpoint_dir to be passed
            in order to allow Ray Tune to use this function """
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.params.local_rank), weights_only=False)
        
        # Get the raw model (without DDP wrapper if present)
        model = self.model.module if dist.is_initialized() else self.model
        
        # Normalize state_dict: always strip 'module.' prefix if present
        state_dict = checkpoint['model_state']
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            name = key[7:] if key.startswith('module.') else key
            new_state_dict[name] = val
        
        # Load the normalized state_dict
        model.load_state_dict(new_state_dict)
        logging.info(f"Loaded model state from {checkpoint_path}")
        
        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch']
        logging.info(f'Restoring from epoch {self.startEpoch}, iteration {self.iters}')

        if self.ema is not None and 'ema_state' in checkpoint and checkpoint['ema_state'] is not None:
            self.ema.load_state_dict(checkpoint['ema_state'])
            logging.info("Restored EMA state")
        
        else:
            # Old checkpoint without EMA state: sync EMA from loaded model weights
            logging.warning("No EMA state in checkpoint. Initializing EMA from loaded model weights.")
            update_ema(self.ema, model, decay=0)  # decay=0 means direct copy
            
        # Restore optimizer and scheduler state only when resuming training (not finetuning)
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
    params['seed'] = seed - rank
    torch.cuda.set_device(device)
    return rank, device

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--run_num", default='0514', type=str)
    parser.add_argument("--yaml_config", default='/project/pedramh/awikner/PanguWeather/v2.0/config/PANGU_PLASIM_H5_MIDWAY_0514.yaml', type=str)
    parser.add_argument("--config", default='PLASIM', type=str) 
    parser.add_argument("--epsilon_factor", default=0, type=float)
    parser.add_argument("--epochs", default=0, type=int)
    parser.add_argument("--run_iter", default=1, type=int)
    parser.add_argument("--debug", default=False, action='store_true')
    parser.add_argument("--no_amp", default=False, action='store_true')
    parser.add_argument("--vae_loss", default=False, action='store_true')
    parser.add_argument("--mode", default='train', type=str, choices=['train', 'test'])
    parser.add_argument("--test_iterations", default=30, type=int)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--just_validate", default = False, action="store_true", help="Only run single epoch of validation")
    parser.add_argument("--validation_epochs", default="", type = str, help="List of epoch to validate when using just_validate. Comma separated list. If empty, validate best_ckpt.")
    parser.add_argument("--use_legacy_model", default=False, action='store_true', help="Use legacy model")

    ####### for UCAR
    parser.add_argument("--local-rank", type=int)
    #######
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    print("This is the starting point f")
    params['enable_amp'] = not args.no_amp
    params['vae_loss'] = args.vae_loss
    params['mode'] = args.mode
    params['test_iterations'] = args.test_iterations
    if args.epochs > 0:
        params['max_epochs'] = args.epochs
    #params['epsilon_factor'] = args.epsilon_factor
    params['run_iter'] = args.run_iter
    params['use_legacy_model'] = args.use_legacy_model
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
    params['just_validate'] = args.just_validate
    if params.just_validate:
        os.environ["WANDB_MODE"] = "offline"
    params['validation_epochs'] = sorted([int(i) for i in args.validation_epochs.split(',')]) if len(args.validation_epochs) > 0 else []
    
    params['debug'] = False
    if args.debug:
        params['debug'] = True
        params['world_size'] = 1
        os.environ['WANDB_MODE'] = 'offline'
        params['train_year_start'] = params['train_year_end'] - 1
        params['batch_size'] = params['num_data_workers']
        #params['long_rollout_years'] = 2
        params['epochs_per_long_validation'] = 1
        params['num_inferences'] = params['num_data_workers']
    else:
        print('World size from OS: %d' % int(os.environ['WORLD_SIZE']))
        print('World size from Cuda: %d' % torch.cuda.device_count())
        if 'WORLD_SIZE' in os.environ:
            params['world_size'] = int(os.environ['WORLD_SIZE'])
            print(params['world_size'])
        else:
            params['world_size'] = torch.cuda.device_count()
            print(params['world_size'])
    if hasattr(params, "wandb_offline"):
        if params.wandb_offline:
            os.environ['WANDB_MODE'] = 'offline'
    
    # initialize DDP.
    world_rank, local_rank = setup_distributed(params, args)

    print('World size from OS: %d' % int(os.environ['WORLD_SIZE']))
    print('World size from Cuda: %d' % torch.cuda.device_count())

    ##Check GPU memory 
    print(torch.cuda.get_device_name(0))
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    print(f"Memory Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")

    # Set up directory structure
    expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    params['experiment_dir'] = os.path.abspath(expDir)
    
    # Define subdirectories
    params['checkpoint_dir'] = os.path.join(expDir, 'checkpoints', f'seed-{params.seed}')
    # params['checkpoint_dir'] = os.path.join(expDir, 'train_checkpoints')
    params['plots_dir'] = os.path.join(expDir, 'plots')
    params['spectra_dir'] = os.path.join(params['plots_dir'], 'spectra', f'seed-{params.seed}')
    params['acc_dir'] = os.path.join(params['plots_dir'], 'acc', f'seed-{params.seed}')
    params['gif_dir'] = os.path.join(params['plots_dir'], 'gif', f'seed-{params.seed}')
    params['bias_dir'] = os.path.join(params['plots_dir'], 'bias', f'seed-{params.seed}')
    params['validation_data_dir'] = os.path.join(expDir, 'validation_data', f'seed-{params.seed}')
    
    # Checkpoint paths
    params['checkpoint_path_globstr'] = os.path.join(params['checkpoint_dir'], 'ckpt_epoch_*.tar')
    params['best_checkpoint_path'] = os.path.join(params['checkpoint_dir'], 'best_ckpt.tar')
    params['latest_checkpoint_path'] = os.path.join(params['checkpoint_dir'], 'ckpt_latest.tar')
    
    # Set default config values if not specified
    if not hasattr(params, 'checkpoint_save_interval'):
        params['checkpoint_save_interval'] = 10
    if not hasattr(params, 'max_checkpoints_to_keep'):
        params['max_checkpoints_to_keep'] = 5
    
    # Set default config values for plot saving
    if not hasattr(params, 'plot_save_interval'):
        params['plot_save_interval'] = 10  # Save plots every N epochs
    if not hasattr(params, 'max_plots_to_keep'):
        params['max_plots_to_keep'] = 5   # Keep only M most recent plot sets
    
    if world_rank == 0:
        os.makedirs(params['checkpoint_dir'], exist_ok=True)
        os.makedirs(params['spectra_dir'], exist_ok=True)
        os.makedirs(params['acc_dir'], exist_ok=True)
        os.makedirs(params['gif_dir'], exist_ok=True)
        if params.long_validation:
            os.makedirs(params['bias_dir'], exist_ok=True)
        os.makedirs(params['validation_data_dir'], exist_ok=True)
    
    # Synchronize all ranks after directory creation
    if dist.is_initialized():
        dist.barrier()

    # CRITICAL FIX: Synchronize checkpoint existence check across all ranks
    # On HPC parallel filesystems, file visibility can vary between nodes
    if world_rank == 0:
        checkpoint_paths = [file for file in glob.glob(params.checkpoint_path_globstr) if os.path.isfile(file)]
        best_checkpoint_exists = os.path.isfile(params['best_checkpoint_path'])
        if hasattr(params, 'latest_checkpoint_path'):
            latest_checkpoint_exists = os.path.isfile(params['latest_checkpoint_path'])
        else:
            latest_checkpoint_exists = False
        checkpoint_exists = len(checkpoint_paths) > 0 or best_checkpoint_exists or latest_checkpoint_exists
    else:
        checkpoint_exists = False  # Will be overwritten by broadcast
    
    # Broadcast checkpoint_exists from rank 0 to all other ranks
    if dist.is_initialized():
        checkpoint_exists_tensor = torch.tensor([checkpoint_exists], dtype=torch.int, device='cuda')
        dist.broadcast(checkpoint_exists_tensor, src=0)
        checkpoint_exists = bool(checkpoint_exists_tensor.item())

    # Determine whether to resume or start fresh
    if params.just_validate:
        # Validation requires a trained model - must load checkpoint
        if not checkpoint_exists:
            raise FileNotFoundError(
                "just_validate=True but no checkpoint found. "
                f"Searched: {params.checkpoint_path_globstr}, {params.best_checkpoint_path}"
            )
        params['resuming'] = True
        if world_rank == 0:
            logging.info("Validation mode: will load checkpoint.")
    elif checkpoint_exists:
        params['resuming'] = True
        if world_rank == 0:
            logging.info("Resuming from existing checkpoint.")
    else:
        params['resuming'] = False
        if world_rank == 0:
            logging.info("No checkpoint found. Starting fresh training run.")

    params['local_rank'] = local_rank
    # Add indicator for precision method and engine
    if params['use_transformer_engine']:
        print("Using Transformer Engine")
    else:
        print("Using PyTorch native")

    if world_rank == 0:
        log_file = 'out.log'
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, log_file))
        logging_utils.log_versions()
        params.log()

    params['log_to_wandb'] = (world_rank == 0) and params['log_to_wandb']
    params['log_to_screen'] = (world_rank == 0) and params['log_to_screen']

    if world_rank == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
            yaml.dump(hparams,  hpfile)

    trainer = Trainer(params, world_rank)
    if params.diagnostic_gif:
        if not hasattr(params, "diagnostic_gif_var_dict"):
            params['diagnostic_gif_var_dict'] = {'zg': [50000]}
    
    if hasattr(params, 'use_sigma_levels'):
        if params.use_sigma_levels:
            print('For sigma level training, disabling diagnostic ACC and diagnostic spectra')
            params['diagnostic_acc'] = False
            params['diagnostic_spectra'] = False
            params['diagnostic_gif'] = False

    trainer.setup_model()

    if not params.just_validate:
        trainer.train()
    else:
        if len(params.validation_epochs) == 0:
            trainer.epoch = trainer.startEpoch
            trainer.validate_one_epoch()
        else:
            for ckpt_i in params.validation_epochs:
                print(f'Validating epoch {ckpt_i}...')
                ckpt_path = params.checkpoint_path_globstr.replace('*', str(ckpt_i))
                if not os.path.isfile(ckpt_path):
                    logging.warning(f"Checkpoint not found: {ckpt_path}, skipping epoch {ckpt_i}")
                    continue
                trainer.restore_checkpoint(ckpt_path)
                trainer.epoch = trainer.startEpoch
                trainer.validate_one_epoch()
    logging.info('DONE ---- rank %d' % world_rank)
    
    if dist.is_initialized():
        dist.destroy_process_group()  # cleanup.