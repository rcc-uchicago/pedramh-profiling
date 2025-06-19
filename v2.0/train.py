# SPDX-FileCopyrightText: 2025 The unvierstiy of Chicago
__author__ = "Yongqiang,Sun"
__updatedate__ = "2025-7-17"

from tqdm import tqdm
from pathlib import Path
from datetime import timedelta
from datetime import datetime
from ruamel.yaml.comments import CommentedMap as ruamelDict
from ruamel.yaml import YAML
from collections import OrderedDict
import matplotlib.pyplot as plt
import wandb
from itertools import product
import time 
from multiprocessing import Process
import psutil
import shutil
import uuid
import os
import numpy as np
import argparse
import xarray as xr
import logging
import torch
import torchvision
from torchvision.utils import save_image
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.profiler import profile, record_function, ProfilerActivity
from utils import logging_utils
from utils.power_spectrum import *
from utils.losses import Latitude_weighted_MSELoss, Latitude_weighted_L1Loss, Masked_L1Loss,\
    Masked_MSELoss, Latitude_weighted_masked_L1Loss, Latitude_weighted_masked_MSELoss,\
    Latitude_weighted_CRPSLoss
from utils.data_loader_multifiles import get_data_loader
from utils.YParams import YParams
from utils.integrate import Integrator, forward_euler
from networks.pangu import PanguModel_Plasim
logging_utils.config_logger()
torch._dynamo.config.optimize_ddp = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
torch.cuda.empty_cache()           
global te, recipe, fp8_autocast
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch import fp8_autocast



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


def compute_weighted_acc(da_fc, da_true, clim=None, weighted=True, mean_dims=xr.ALL_DIMS, **kwargs):
    da_fc = da_fc.assign_coords(dayofyear=da_fc['time'].dt.dayofyear)
    da_true = da_true.assign_coords(dayofyear=da_true['time'].dt.dayofyear)
    if clim is not None:
        if True:
            if 'zsfc' in clim:
                clim = clim.drop_vars('zsfc')
            if 'pr_12h' in da_fc:
                clim['pr_12h'] = clim['tas'].copy()
                clim['pr_12h'][:] = 0.
            if 'pr_6h' in da_fc:
                clim['pr_6h'] = clim['tas'].copy()
                clim['pr_6h'][:] = 0.
            if 'mrso' in da_fc:
                clim['mrso'] = clim['tas'].copy()
                clim['mrso'][:] = 0.
            
            # Reorder variables in climatology to match forecast data
            clim = clim[list(da_fc.data_vars)]
            
            # Transpose climatology to match forecast data dimensions
            clim = clim.transpose('dayofyear', 'plev', 'lat', 'lon')
            
            # print("\nSelecting climatology based on dayofyear:")
            climatology_aligned = clim.sel(dayofyear=da_fc['dayofyear'])
            
            # Ensure climatology has the same dimensions as da_fc
            climatology_aligned = climatology_aligned.transpose(*da_fc.dims)
            # print_info(climatology_aligned, "Aligned Climatology")
            climatology_aligned = climatology_aligned.assign_coords(lat=da_fc.lat)
            fa = da_fc - climatology_aligned
            a = da_true - climatology_aligned
            #except Exception as e:
        else:
            print(f"Error during climatology alignment or subtraction: {str(e)}")
            return xr.DataArray(np.nan, dims=['time'])
    else:
        fa = da_fc
        a = da_true

    fa = fa.drop_vars('dayofyear', errors='ignore')
    a = a.drop_vars('dayofyear', errors='ignore')


    if weighted:
        weights_lat = np.cos(np.deg2rad(a.lat))
        weights_lat /= weights_lat.mean()
    else:
        weights_lat = 1.
    w = weights_lat

    fa_prime = fa - fa.mean()
    a_prime = a - a.mean()
    numerator = (w * fa_prime * a_prime).sum(mean_dims)
    denominator = np.sqrt((w * fa_prime ** 2).sum(mean_dims) * (w * a_prime ** 2).sum(mean_dims))
    acc = numerator / denominator
    # print_info(acc, "ACC")
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
        self.startEpoch = 0
        if params.resuming:
            self.restore_checkpoint(params.checkpoint_path)
        else:
            logging.info("Starting fresh training run")
        self.epoch = self.startEpoch
        self.early_stop_epoch = params['early_stop_epoch'] - 1 if 'early_stop_epoch' in params else None
        #################################################
        self.run_uuid = str(uuid.uuid4())
        self.check_land_ocean_variables()
        if self.params.enable_amp == True:  self.gscaler = amp.GradScaler()
        # Create output directories
        self.spectra_dir, self.diagnostics_dir, self.output_dir = self.create_dirs(self.run_uuid)    
        # Enable mixed precision training
        self.enable_mix_prcision()
        # Initial wandb
        self.init_wandb(self.params)
        # Setup model
        self.setup_model()
        logging.info('Params' % params)
        



    def setup_model(self):
        # Set up model
        self.model = self.get_model()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.setup_scheduler()
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



    def create_dirs(run_uuid:int)->tuple[str, str, str]:
        """
        To create the output directories to store the spectra_output, image, and evaluation metrics plot.
        """

        main_dirs = ["spectra_out", "gif_out", "acc_plots"]
        for dir_name in main_dirs:
            os.makedirs(os.path.join(os.getcwd(), dir_name), exist_ok=True)
        spectra_dir = os.path.join(os.getcwd(), "spectra_out", self.run_uuid)
        diagnostics_dir = os.path.join(os.getcwd(), "gif_out", self.run_uuid)
        output_dir = os.path.join(os.getcwd(), "acc_plots", self.run_uuid) 
        logging.info('The output directories %s ; %s; %s', spectra_dir,diagnostics_dir,output_dir)

        if world_rank == 0:
            os.makedirs(self.spectra_dir, exist_ok=True)
            os.makedirs(self.diagnostics_dir, exist_ok=True)
            os.makedirs(self.output_dir, exist_ok=True)
            logging.info(f"Created directory: {spectra_dir}")
            logging.info(f"Created directory: {diagnostics_dir}")
            logging.info(f"Created directory: {output_dir}")
        return spectra_dir, diagnostics_dir, output_dir 
             

    def enable_mix_prcision(self) -> None:
        """
        Enable mixed precision training with FP8
        """
        if self.params.enable_fp8:
            self.fp8_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID,
                                                    amax_history_len=16,
                                                    amax_compute_algo="max")
            self.precision_context = fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe)
        else:
            self.precision_context = amp.autocast(enabled=self.params.enable_amp)


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
        
        self.constant_boundary_data = self.train_datasets[0].constant_boundary_data.unsqueeze(0) * torch.ones(params.batch_size, 1, 1, 1)
        self.constant_boundary_data = self.constant_boundary_data.to(self.device, non_blocking=True)
        if params.num_ensemble_members > 1:
            self.constant_boundary_data = to_ensemble_batch(self.constant_boundary_data, params.num_ensemble_members)
            logging.info('Ensemble Mode. Ensemble size = {params.num_ensemble_members}\n')

         # Load climatology
        climatology_path = os.path.join(params.data_dir, self.params.climatology_file)
        self.climatology = xr.open_dataset(climatology_path)
        self.climatology = self.climatology.rename({'time':'dayofyear'})
        logging.info('rank %d, data loader initialized' % self.world_rank)



    @staticmethod
    def init_wandb(params:dict):    
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
            wandb.define_metric("epoch")
            wandb.define_metric("ACC_plot", step_metric="epoch")
            wandb.define_metric("power_spectrum_plot", step_metric="epoch")
            if params.diagnostic_logs:
                epoch_metrics = ['lr', 'train_loss', 'valid_loss', 'valid_loss_sfc', 'valid_loss_upper_air', 'valid_mean_norm_lwrmse']
                for l, steps in enumerate(params.forecast_lead_times):
                    epoch_metrics.append(f"valid_lwrmse_sfc_{steps}step")
                    epoch_metrics.append(f"valid_lwrmse_pl_{steps}step")
                    epoch_metrics.append(f"valid_loss_{steps}step")
                    for j, var in enumerate(self.valid_dataset.surface_variables):
                        epoch_metrics.append(f'valid_{var}_{steps}step_lwrmse')
                    for j, var in enumerate(self.valid_dataset.upper_air_variables):
                        for k, level in enumerate(self.valid_dataset.levels):
                            epoch_metrics.append(f'valid_{var}_level{level:.3f}_{steps}step_lwrmse')
            else:
                epoch_metrics = ['lr', 'train_loss', 'valid_loss', 'valid_loss_sfc', 'valid_loss_upper_air']

            # Add this line to ensure power_spectrum_plot is always defined as a metric
            for metric in epoch_metrics:
                wandb.define_metric(metric, step_metric="epoch")


        

    def get_model(self):
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
            if self.params.predict_delta:
                self.model = PanguModel_Plasim(params, land_mask = land_mask).to(self.device)
                self.integrator = Integrator(params, surface_ff_std=self.train_datasets[0].surface_std.detach().to(self.device),
                                               surface_delta_std=self.train_datasets[0].surface_delta_std.detach().to(self.device),
                                               upper_air_ff_std=self.train_datasets[0].upper_air_std.detach().to(self.device),
                                               upper_air_delta_std=self.train_datasets[0].upper_air_delta_std.detach().to(self.device)).to(self.device)
            else:
                if hasattr(params, 'mask_fill'):
                    self.model = PanguModel_Plasim(params, land_mask = land_mask, 
                                               mask_fill = params.mask_fill).to(self.device)
                else:
                    self.model = PanguModel_Plasim(params, land_mask = land_mask, 
                                                mask_fill = self.train_datasets[0].mask_fill).to(self.device)
            # self.model = torch.compile(self.model, mode = 'default')
        else:
            raise Exception("not implemented")

        
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
        return self.model
        

    def count_parameters(self):
        """
        Count the trainable parameters
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


    def get_optimizer(self):
        if params.optimizer_type == 'FusedAdam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr, weight_decay=params.weight_decay, fused=True)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
        return self.optimizer 


    def setup_scheduler(self):
        if self.params.scheduler == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=5, mode='min')
        elif self.params.scheduler == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.params.max_epochs, 
                                                                        last_epoch=self.startEpoch-1)
        elif self.params.scheduler == 'OneCycleLR':
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

            if self.startEpoch < 1:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=self.params.lr,
                    total_steps=total_steps,
                    steps_per_epoch=steps_per_epoch,
                    pct_start = pct_start,
                    div_factor = div_factor,
                    final_div_factor=final_div_factor
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=params.lr,
                    total_steps=total_steps,
                    steps_per_epoch=steps_per_epoch,
                    last_epoch=(self.startEpoch-1) * steps_per_epoch,
                    pct_start = pct_start,
                    div_factor = div_factor,
                    final_div_factor=final_div_factor
                )
        else:
            self.scheduler = None
        logging.info("Scheduler is setup")


    def setup_loss_fun(self):
        """
        Set up loss function to return the loss for pl, sfc and diagnoistic
        """
        #initialisation
        self.loss_obj_pl = 0 
        self.loss_obj_sfc = 0
        self.loss_obj_diagnostic = 0
        if self.params.loss == 'l1':
            self.loss_obj_pl = torch.nn.L1Loss()
            if (self.has_land or self.has_ocean) and self.mask_output:
                self.loss_obj_sfc = Masked_L1Loss(mask_bool)
            else:
                self.loss_obj_sfc = torch.nn.L1Loss()
            if self.params.has_diagnostic:
                self.loss_obj_diagnostic = torch.nn.L1Loss()
        elif self.params.loss == 'l2':
            self.loss_obj_pl = torch.nn.MSELoss()
            if (self.has_land or self.has_ocean) and self.mask_output:
                self.loss_obj_sfc = Masked_MSELoss(mask_bool)
            else:
                self.loss_obj_sfc = torch.nn.MSELoss()
            if self.params.has_diagnostic:
                self.loss_obj_diagnostic = torch.nn.MSELoss()
        elif self.params.loss == 'weightedl1':
            self.lat = torch.from_numpy(np.array(self.params.lat)).to(self.device)
            # self.lat = self.train_dataset.lat.to(self.device, non_blocking=True)
            self.loss_obj_pl = Latitude_weighted_L1Loss(self.lat)
            if (self.has_land or self.has_ocean) and self.mask_output:
                self.loss_obj_sfc = Latitude_weighted_masked_L1Loss(self.lat, mask_bool)
            else:
                self.loss_obj_sfc = Latitude_weighted_L1Loss(self.lat)
            if self.params.has_diagnostic:
                self.loss_obj_diagnostic = Latitude_weighted_L1Loss(self.lat)
        elif self.params.loss == 'weightedl2':
            self.lat = torch.from_numpy(np.array(self.params.lat)).to(self.device)
            self.loss_obj_pl = Latitude_weighted_MSELoss(self.lat)
            if (self.has_land or self.has_ocean) and self.mask_output:
                self.loss_obj_sfc = Latitude_weighted_masked_MSELoss(self.lat, mask_bool)
            else:
                self.loss_obj_sfc = Latitude_weighted_MSELoss(self.lat)
            if self.params.has_diagnostic:
                self.loss_obj_diagnostic = Latitude_weighted_MSELoss(self.lat)
        elif self.params.loss == 'weightedCRPS':
            self.lat = torch.from_numpy(np.array(self.params.lat)).to(self.device)
            self.loss_obj_pl = Latitude_weighted_CRPSLoss(self.lat, params.num_ensemble_members)
            if self.has_land or self.has_ocean:
                self.loss_obj_sfc = Latitude_weighted_CRPSLoss(self.lat, params.num_ensemble_members,
                                                               mask_bool)
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
            if self.early_stop_epoch is not None and epoch > self.early_stop_epoch:
                if self.params.log_to_screen:
                    logging.info(f'Completed early stop epoch {self.early_stop_epoch}. Terminating training.')
                early_stop_epoch_triggered = True
                break
            
            if dist.is_initialized():
                for sampler in self.train_samplers:
                    sampler.set_epoch(epoch)

            start = time.time()
            tr_time, data_time, train_logs = self.train_one_epoch()
            valid_time, valid_logs = self.validate_one_epoch()

            if self.params.scheduler == 'ReduceLROnPlateau':
                self.scheduler.step(valid_logs['valid_loss'])
            elif self.params.scheduler == 'CosineAnnealingLR':
                self.scheduler.step()
                if self.epoch >= self.params.max_epochs:
                    logging.info("Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR")
                    # exit()
                    break
            
            # Early stopping logic should be outside of world_rank check
            if valid_logs['valid_loss'] <= best_valid_loss:
                best_valid_loss = valid_logs['valid_loss']
                early_stopping_counter = 0  # Reset the counter
            else:
                early_stopping_counter += 1  # Increment the counter
            
            if self.world_rank == 0:
                if self.params.save_checkpoint:
                    # checkpoint at the end of every epoch
                    self.save_checkpoint(self.params.checkpoint_path)
                    if valid_logs['valid_loss'] <= best_valid_loss:
                        self.save_checkpoint(self.params.best_checkpoint_path)

            #Bing: Is this a bug? why log the same lr ?
            # if self.params.log_to_wandb:
            #     for pg in self.optimizer.param_groups:
            #         lr = pg['lr']
            #     wandb.log({'lr': lr, 'epoch': self.epoch})
            self.log_wandb_epoch(epoch)
            self.log_screen_epoch(epcoh,time)
                    # Early stopping check
            if self.params.early_stopping and early_stopping_counter >= self.params.early_stopping_patience:
                if self.params.log_to_screen:
                    logging.info('Early stopping triggered. Terminating training.')
                    break # Exit the train method

        if self.params.log_to_screen:
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
    

    def log_screen_epoch(self, epoch:int, start, **kwargs) ->None:
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

    def train_one_epoch(self)->None:
        self.epoch += 1
        tr_time = 0
        data_time = 0
        if self.params.diagnostic_logs:
            diagnostic_logs = {}

        total_iterations = sum(len(loader) for loader in self.train_data_loaders)
        logging.info(f"Expected total batches: {total_iterations}")
        if not self.train_data_loaders:
            logging.warning("No training data loaders available.")
            return 0, 0, {"train_loss": 0.0}

        self.model.train()

        pbar = tqdm(total=total_iterations, bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')
        running_results = {"batch_sizes": 0, "loss": 0.0}

        for year_idx, train_data_loader in enumerate(self.train_data_loaders):
            current_dataset = self.train_datasets[year_idx]
            with torch.no_grad():
                latitudes = torch.from_numpy(np.array(self.params.lat)).to(self.device, non_blocking=True)
                
            if self.params.train_year_to_year:
                logging.debug(f"Processing year {self.params.train_year_start + year_idx}")
            else:
                logging.debug(f"Processing years {self.params.train_year_start} to {self.params.train_year_end}")
      
            for i, data in enumerate(train_data_loader):
                logging.debug(f"Batch {i}, data shape: {data[0].shape}")
                self.iters += 1
                data_start = time.time()
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
            
                data_time += time.time() - data_start
                tr_start = time.time()

                self.model.zero_grad()

               

                with self.precision_context:
                    if self.params.has_diagnostic:
                        output_surface, output_upper_air, output_diagnostic = self.model(input_surface, self.constant_boundary_data, 
                                                                    varying_boundary_data, input_upper_air, train = True)
                
                        loss_diagnostic = self.loss_obj_diagnostic(output_diagnostic, target_diagnostic)
                    else:
                        output_surface, output_upper_air = self.model(input_surface, self.constant_boundary_data, 
                                                                    varying_boundary_data, input_upper_air, train = True)
                    
                    loss_sfc = self.loss_obj_sfc(output_surface, target_surface)
                    loss_pl = self.loss_obj_pl(output_upper_air, target_upper_air)

                    if self.params.has_diagnostic:
                        loss = (loss_sfc + loss_diagnostic) * 0.25 + loss_pl
                    else:
                        loss = (loss_sfc * 0.25) + loss_pl

                if self.params.enable_amp:
                    self.gscaler.scale(loss).backward()
                    self.gscaler.step(self.optimizer)
                    self.gscaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                if self.params.scheduler == 'OneCycleLR':
                    self.scheduler.step()

                with torch.no_grad():
                    if self.params.predict_delta:
                        output_surface, output_upper_air = self.integrator(input_surface, input_upper_air, output_surface, output_upper_air)
                        target_surface, target_upper_air = self.integrator(input_surface, input_upper_air, target_surface, target_upper_air)
                    surface_lwrmse = weighted_rmse_torch_channels(output_surface, target_surface, latitudes)
                    upper_air_lwrmse = weighted_rmse_torch_3D(output_upper_air, target_upper_air, latitudes)

                    if self.params.has_diagnostic:
                        diagnostic_lwrmse = weighted_rmse_torch_channels(output_diagnostic, target_diagnostic, latitudes)

                    if self.params.diagnostic_logs:
                        diagnostic_logs['batch_grad_norm'] = torch.tensor([grad_norm(self.model)]).to(self.device)
                        diagnostic_logs['batch_grad_max'] = torch.tensor([grad_max(self.model)]).to(self.device)
                        diagnostic_logs['train_batch_loss'] = loss
                        diagnostic_logs['train_batch_loss_sfc'] = loss_sfc
                        diagnostic_logs['train_batch_loss_upper_air'] = loss_pl

                        if self.params.has_diagnostic:
                            diagnostic_logs['train_batch_loss_diagnostic'] = loss_diagnostic
                            mean_norm_lwrmse = torch.mean(torch.cat((surface_lwrmse, diagnostic_lwrmse, upper_air_lwrmse.reshape(output_upper_air.shape[0], -1)), dim = -1))
                        else:
                            mean_norm_lwrmse = torch.mean(torch.cat((surface_lwrmse, upper_air_lwrmse.reshape(output_upper_air.shape[0], -1)), dim = -1))
                        diagnostic_logs['train_mean_norm_lwrmse'] = mean_norm_lwrmse
                        for j, var in enumerate(current_dataset.surface_variables):
                            diagnostic_logs[f'train_{var}_lwrmse'] = torch.mean(surface_lwrmse[:, j]) * current_dataset.surface_std[j]
                        if self.params.has_diagnostic:
                            for j, var in enumerate(current_dataset.diagnostic_variables):
                                diagnostic_logs[f'train_{var}_lwrmse'] = torch.mean(diagnostic_lwrmse[:, j]) * current_dataset.diagnostic_std[j]
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
                        if self.params.log_to_wandb:
                            wandb.log(diagnostic_logs, step=(self.epoch-1) * total_iterations + self.iters)

                torch.cuda.empty_cache()
                tr_time += time.time() - tr_start

                if self.params.diagnostic_logs:
                    pbar.set_description(f"Year {self.params.train_year_start + year_idx}, Loss: {diagnostic_logs['train_batch_loss']:.4f}")
                else:
                    running_results["loss"] += loss.item() * self.params['batch_size']
                    running_results["batch_sizes"] += self.params['batch_size']
                    pbar.set_description(f"Year {self.params.train_year_start + year_idx}, Loss: {running_results['loss'] / running_results['batch_sizes']:.4f}")
                
                pbar.update(1)
        pbar.close()

        if self.params.diagnostic_logs:
            with torch.no_grad():
                diagnostic_logs['train_loss'] = loss
                if dist.is_initialized():
                    dist.all_reduce(torch.tensor(diagnostic_logs['train_loss']).to(self.device))
                    diagnostic_logs['train_loss'] = float(diagnostic_logs['train_loss']/dist.get_world_size())
                logs = {'train_loss': diagnostic_logs['train_loss'], 'epoch': self.epoch}
                if self.params.log_to_wandb:
                    wandb.log(logs)
                return tr_time, data_time, diagnostic_logs
        else:
            with torch.no_grad():
                logs = {'train_loss': loss, 'epoch': self.epoch}
            
            if dist.is_initialized():
                for key in sorted(logs.keys()):
                    if isinstance(logs[key], (int, float)):
                        logs[key] = torch.tensor(logs[key]).to(self.device)
                    dist.all_reduce(logs[key])
                    logs[key] = float(logs[key]/dist.get_world_size())

            if self.params.log_to_wandb:
                wandb.log(logs)
        return tr_time, data_time, logs




    def validate_one_epoch(self):
        self.model.eval()
        #n_valid_batches = 50  # do validation on first 50 images, just for LR scheduler
        # define the lead times to evaluate (in time steps)
        lead_times_steps = self.params.forecast_lead_times
        with torch.no_grad():
                latitudes = torch.from_numpy(np.array(self.params.lat)).to(self.device, non_blocking=True)

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
        
        valid_start = time.time()
        nb = len(self.valid_data_loader)
        if self.params.diagnostic_logs:
            diagnostic_logs = {}

        sample_idx = np.random.randint(len(self.valid_data_loader))

        all_predictions = []
        all_ground_truths = []
        acc_predictions = []
        acc_ground_truths = []

        # OPTIMIZATION
        # with torch.inference_mode():
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.valid_data_loader, 0), total=nb, bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}'):
                #if i >= n_valid_batches:
                #    break
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
                
                # Tensor for specific lead times (power spectrum and GIF)
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

                     # Autoregressive prediction
                    # val_output_surface, val_output_upper_air = val_input_surface, val_input_upper_air
                    step_idx = 0
                    for step in range(max_lead_time):
                        if self.params.has_diagnostic:
                            val_output_surface, val_output_upper_air, val_output_diagnostic = self.model(
                                val_input_surface, self.constant_boundary_data, val_varying_boundary_data[:, step], val_input_upper_air)
                        else:
                            val_output_surface, val_output_upper_air = self.model(val_input_surface, self.constant_boundary_data, 
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
                            #print(f'val_output_diagnostic_acc[:,step].shape: {val_output_diagnostic_acc[:, step].shape}')
                            #print(f'val_output_diagnostic.cpu().shape: {val_output_diagnostic.cpu().shape}')
                            #print(f'self.valid_dataset.diagnostic_inv_transform(val_output_diagnostic.cpu()).numpy().shape: {self.valid_dataset.diagnostic_inv_transform(val_output_diagnostic.cpu()).numpy().shape}')
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
                valid_steps += 1.
        

        # After the loop, combine all predictions and ground truths
        if self.params.diagnostic_spectra:
            combined_predictions = xr.concat(all_predictions, dim='time')
            combined_ground_truths = xr.concat(all_ground_truths, dim='time')

        if self.params.diagnostic_acc or self.params.diagnostic_gif:
            acc_combined_predictions = xr.concat(acc_predictions, dim='time')
            acc_combined_ground_truths = xr.concat(acc_ground_truths, dim='time')

        # lead_times_hours = [lt * self.params.timedelta_hours for lt in self.params.forecast_lead_times]
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
            fig, axs = plot_acc_over_lead_time(acc, acc_times_hours)

        if self.params.diagnostic_spectra:
            k_x_pred, power_spectrum_avg_pred = zonal_averaged_power_spectrum(combined_predictions, time_avg=True) 
            k_x_gt, power_spectrum_avg_gt = zonal_averaged_power_spectrum(combined_ground_truths, time_avg= True)
            preds_times = combined_predictions.time.values
            preds_times = preds_times.cpu().numpy() if isinstance(preds_times, torch.Tensor) else preds_times
        
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
                make_gif(acc_combined_predictions, acc_combined_ground_truths, self.climatology, "Model Forecast", "geopotential", gif_filename, plev=50000)
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

        if self.params.diagnostic_logs:
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
                    for k, level in enumerate(self.valid_dataset.levels):
                        diagnostic_logs[f'valid_{var}_level{level:.3f}_{steps}step_lwrmse'] = valid_upper_air_lwrmse[l, j, k] * self.valid_dataset.upper_air_std[j, k]
                if self.params.has_diagnostic:
                    for j, var in enumerate(self.valid_dataset.diagnostic_variables):
                        diagnostic_logs[f'valid_{var}_{steps}step_lwrmse'] = valid_diagnostic_lwrmse[l, j] * self.valid_dataset.diagnostic_std[j]
            #if dist.is_initialized():
            #    for key in sorted(diagnostic_logs.keys()):
            #        dist.all_reduce(diagnostic_logs[key].detach())
            #        diagnostic_logs[key] = float(diagnostic_logs[key]/dist.get_world_size())

            # Add multi-day losses to diagnostic logs
            for key, value in multi_step_losses.items():
                diagnostic_logs[key] = value.item()

            if self.params.log_to_wandb:
                wandb.log(diagnostic_logs)
                if self.params.diagnostic_acc:
                    wandb.log({
                        "ACC_plot": wandb.Image(plot_filename),
                        "epoch": self.epoch
                    })
                if self.params.diagnostic_gif:
                    if gif_filename:
                        wandb.log({
                            "Evolution_GIF": wandb.Video(gif_filename),
                            "epoch": self.epoch
                        })
                if self.params.diagnostic_spectra:
                    wandb.log({
                        "power_spectrum_plot": wandb.Image(path_filename),
                        "epoch": self.epoch,
                    })
            
            valid_time = time.time() - valid_start

            return valid_time, diagnostic_logs
        else:
            try:
                if self.params.has_diagnostic:
                    logs = {'valid_loss': valid_buff_cpu[0], 
                            'valid_loss_sfc': valid_buff_cpu[1], 'valid_loss_upper_air': valid_buff_cpu[2],
                            'valid_loss_diag': valid_buff_cpu[3],
                            'epoch': self.epoch}
                else:
                    logs = {'valid_loss': valid_buff_cpu[0], 
                            'valid_loss_sfc': valid_buff_cpu[1], 'valid_loss_upper_air': valid_buff_cpu[2],
                            'epoch': self.epoch}
                # Add multi-day losses to logs
                for key, value in multi_step_losses.items():
                    logs[key] = value.item()

            except:
                pass

            if self.params.log_to_wandb:
                wandb.log(logs)

                    # Log ACC plot
                if self.params.diagnostic_acc:
                    wandb.log({
                        "ACC_plot": wandb.Image(plot_filename),
                        "epoch": self.epoch
                    })
                if gif_filename and self.params.diagnostic_gif:
                    wandb.log({
                        "Evolution_GIF": wandb.Video(gif_filename),
                        "epoch": self.epoch
                    })
                if self.params.diagnostic_spectra:
                    wandb.log({
                        "power_spectrum_plot": wandb.Image(path_filename),
                        "epoch": self.epoch,
                    })

            valid_time = time.time() - valid_start

            return valid_time, logs



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


    def plot_in_separate_process(self, power_spectrum_avg_preds, power_spectrum_avg_gt, preds_times, filename):
        # convert lead times to hours
        lead_times_hours = [step * self.params.timedelta_hours for step in self.params.forecast_lead_times]
        p = Process(target=plot_power_spectrum_test, args=(power_spectrum_avg_preds, power_spectrum_avg_gt, preds_times, filename, lead_times_hours))
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
                    })
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
        output_dir = os.path.join(os.getcwd(), "acc_plots", self.run_uuid)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"Deleted ACC plots directory: {output_dir}")

    def cleanup_power_spectrum_plots(self):
        output_dir = os.path.join(os.getcwd(), "spectra_out", self.run_uuid)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"Deleted Power Spectrum plots directory: {output_dir}")

    def cleanup_gifs(self):
        if os.path.exists(self.diagnostics_dir):
            shutil.rmtree(self.diagnostics_dir)
            print(f"Deleted GIF directory: {self.diagnostics_dir}")
    
    def save_checkpoint(self, checkpoint_path, model=None):
        """ We intentionally require a checkpoint_dir to be passed
            in order to allow Ray Tune to use this function """

        if not model:
            model = self.model

        torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, checkpoint_path)


    def restore_checkpoint(self, checkpoint_path):
        """ We intentionally require a checkpoint_dir to be passed
            in order to allow Ray Tune to use this function """
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.params.local_rank))
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
        print('START EPOCH:', self.startEpoch)
        # restore checkpoint is used for finetuning as well as resuming. If finetuning (i.e., not resuming), restore checkpoint does not load optimizer state, instead uses config specified lr.
        if self.params.resuming:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='0100', type=str)
    parser.add_argument("--yaml_config", default='v2.0/config/PANGU_S2S.yaml', type=str)
    parser.add_argument("--config", default='S2S', type=str)
    parser.add_argument("--enable_amp", default=True, action='store_true')
    parser.add_argument("--epsilon_factor", default=0, type=float)
    parser.add_argument("--epochs", default=0, type=int)
    parser.add_argument("--run_iter", default=1, type=int)
    # parser.add_argument("--num_inferences", type = int)
    # parser.add_argument("--window_size", default = '2,2,2', type = str)
    parser.add_argument("--fresh_start", default=False, action="store_true", help="Start training from scratch, ignoring existing checkpoints")
    ####### for UCAR
    parser.add_argument("--local-rank", type=int)
    #######
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    print("This is the starting point f")
    if args.epochs > 0:
        params['max_epochs'] = args.epochs
    params['epsilon_factor'] = args.epsilon_factor
    params['run_iter'] = args.run_iter
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

    if hasattr(params, "wandb_offline"):
        if params.wandb_offline:
            os.environ['WANDB_MODE'] = 'offline'

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

        args.gpu = local_rank
        world_rank = dist.get_rank()
        # print("##########WORLD RANK: TESTING ", world_rank)
        params['global_batch_size'] = params.batch_size
        params['batch_size'] = int(params.batch_size//params['world_size'])
    else:
        world_rank = 0
        local_rank = 0

    torch.manual_seed(world_rank)
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

    # Set up directory
    expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    if world_rank == 0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)
            os.makedirs(os.path.join(expDir, 'training_checkpoints/'))

    params['experiment_dir'] = os.path.abspath(expDir)
    ckpt_path = 'training_checkpoints/ckpt.tar'
    best_ckpt_path = 'training_checkpoints/best_ckpt.tar'
    params['checkpoint_path'] = os.path.join(expDir, ckpt_path)
    params['best_checkpoint_path'] = os.path.join(expDir, best_ckpt_path)

    checkpoint_exists = os.path.isfile(params.checkpoint_path)

    # Determine whether to resume or start fresh
    if params.fresh_start or args.fresh_start:
        params['resuming'] = False
        if checkpoint_exists and world_rank == 0:
            logging.info("Fresh start requested. Ignoring existing checkpoint.")
    elif checkpoint_exists:
        params['resuming'] = True
        if world_rank == 0:
            logging.info("Resuming from existing checkpoint.")
    else:
        params['resuming'] = False
        if world_rank == 0:
            logging.info("No checkpoint found. Starting fresh training run.")

    # # Do not comment this line out please:
    # # args.resuming = True if os.path.isfile(params.checkpoint_path) else False
    # args.resuming = False
    # params['resuming'] = args.resuming

    params['local_rank'] = local_rank
    params['enable_amp'] = False if params['enable_fp8'] else args.enable_amp

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
    train.setup()
    # trainer.train()
    # logging.info('DONE ---- rank %d' % world_rank)

