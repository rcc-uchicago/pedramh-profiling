from networks.pangu import PanguModel_Plasim
from tqdm import tqdm
from ruamel.yaml.comments import CommentedMap as ruamelDict
from ruamel.yaml import YAML
from collections import OrderedDict
import matplotlib.pyplot as plt
import wandb
from utils.data_loader_multifiles import get_data_loader
from utils.YParams import YParams
import os
import time
import numpy as np
import argparse
import torch
import torchvision
from torchvision.utils import save_image
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils
##########################################
## NEW IMPORTS
from utils.losses import Latitude_weighted_MSELoss, Latitude_weighted_L1Loss
###############################@###########
logging_utils.config_logger()
from apex import optimizers
from pathlib import Path
import dask
# import transformer_engine.pytorch as te
# from transformer_engine.common import recipe
# from transformer_engine.pytorch import fp8_autocast
from torch.profiler import profile, record_function, ProfilerActivity
from itertools import product



# from utils.weighted_acc_rmse import weighted_rmse_torch_channels, weighted_rmse_torch_3D

# os.environ['WANDB_MODE'] = 'offline'
# os.environ['WANDB_DIR'] = '/home/tvallabh/PanguWeather/v2.0/wandb'
# os.environ['WANDB_SERVICE_WAIT'] = '300'  # Wait for 300 seconds


dask.config.set(scheduler='synchronous')
torch._dynamo.config.optimize_ddp = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

torch.cuda.empty_cache()

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

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
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[3]
    #num_long = target.shape[2]
    #lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    #s = torch.sum(torch.cos(3.1416/180. * latitudes))
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

class Trainer():
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def __init__(self, params, world_rank):

        self.params = params
        self.world_rank = world_rank
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'


        logging.info('rank %d, begin data loader init' % world_rank)
        print(params)

        self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(params, params.data_dir, dist.is_initialized(), 
                                                                                         year_start=params.train_year_start, 
                                                                                         year_end=params.train_year_end, train=True)
        self.valid_data_loader, self.valid_dataset = get_data_loader(params, params.data_dir, dist.is_initialized(), 
                                                                     year_start=params.val_year_start, 
                                                                     year_end=params.val_year_end, train=False,
                                                                     num_inferences = params.num_inferences,
                                                                     validate = True)

        self.constant_boundary_data = self.train_dataset.constant_boundary_data.unsqueeze(0) * torch.ones(params.batch_size, 1, 1, 1)
        self.constant_boundary_data = self.constant_boundary_data.to(self.device, non_blocking=True)

        self.enable_amp = params.enable_amp
        self.enable_fp8 = params.enable_fp8
        
        if self.enable_fp8:
            global te, recipe, fp8_autocast
            import transformer_engine.pytorch as te
            from transformer_engine.common import recipe
            from transformer_engine.pytorch import fp8_autocast

            self.fp8_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID,
                                                    amax_history_len=16,
                                                    amax_compute_algo="max")
        if params.log_to_wandb:
            wandb.init(config=params, name=params.name, group=params.group, project=params.project)#,
            #           entity=params.entity)
            wandb.define_metric("epoch")
            if self.params.diagnostic_logs:
                epoch_metrics = ['lr', 'train_loss', 'valid_loss', 'valid_loss_sfc', 'valid_loss_upper_air', 'valid_mean_norm_lwrmse']
                for l, steps in enumerate(self.params.forecast_lead_times):
                    epoch_metrics.append(f"valid_lwrmse_sfc_{steps}step")
                    epoch_metrics.append(f"valid_lwrmse_pl_{steps}step")
                    epoch_metrics.append(f"valid_loss_{steps}step")
                    for j, var in enumerate(self.valid_dataset.surface_variables):
                        epoch_metrics.append(f'valid_{var}_{steps}step_lwrmse')
                    for j, var in enumerate(self.valid_dataset.upper_air_variables):
                        for k, level in enumerate(self.valid_dataset.lev):
                            epoch_metrics.append(f'valid_{var}_level{level:.3f}_{steps}step_lwrmse')
            else:
                epoch_metrics = ['lr', 'train_loss', 'valid_loss', 'valid_loss_sfc', 'valid_loss_upper_air']
            for metric in epoch_metrics:
                wandb.define_metric(metric, step_metric="epoch")


        logging.info('rank %d, data loader initialized' % world_rank)


        if params.nettype == 'pangu_plasim':
            self.model = PanguModel_Plasim(params).to(self.device)
            # self.model = torch.compile(self.model, mode = 'default')
        else:
            raise Exception("not implemented")

        if params.log_to_wandb:
            wandb.watch(self.model)

        if params.optimizer_type == 'FusedAdam':
            self.optimizer = optimizers.FusedAdam(self.model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

        if params.enable_amp == True:
            self.gscaler = amp.GradScaler()

        if dist.is_initialized():
            self.model = DistributedDataParallel(self.model,
                                                 device_ids=[
                                                     params.local_rank],
                                                 output_device=[params.local_rank], find_unused_parameters=True)

        self.iters = 0
        self.startEpoch = 0
        if params.resuming:
            self.restore_checkpoint(params.checkpoint_path)
        else:
            logging.info("Starting fresh training run")

        self.epoch = self.startEpoch

        if params.scheduler == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=5, mode='min')
        elif params.scheduler == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=params.max_epochs, 
                                                                        last_epoch=self.startEpoch-1)
        elif params.scheduler == 'OneCycleLR':
            total_steps = len(self.train_data_loader) * params.max_epochs
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=params.lr,
                total_steps=total_steps
            )
        else:
            self.scheduler = None

        '''if params.log_to_screen:
      logging.info(self.model)'''
        if params.log_to_screen:
            logging.info("Number of trainable model parameters: {}".format(self.count_parameters()))
        if params.loss == 'l1':
            self.loss_obj_sfc = torch.nn.L1Loss() 
            self.loss_obj_pl = torch.nn.L1Loss()
        elif params.loss == 'l2':
            self.loss_obj_sfc = torch.nn.MSELoss()
            self.loss_obj_pl = torch.nn.MSELoss()
        elif params.loss == 'weightedl1':
            self.lat = self.train_dataset.lat.to(self.device, non_blocking=True)
            self.loss_obj_sfc = Latitude_weighted_L1Loss(self.lat)
            self.loss_obj_pl = Latitude_weighted_L1Loss(self.lat)
        elif params.loss == 'weightedl2':
            self.lat = self.train_dataset.lat.to(self.device)
            self.loss_obj_sfc = Latitude_weighted_MSELoss(self.lat)
            self.loss_obj_pl = Latitude_weighted_MSELoss(self.lat)
        else:
            raise NotImplementedError


    def train(self):
        if self.params.log_to_screen:
            logging.info("Starting Training Loop...")

        best_valid_loss = 1.e6
        early_stopping_counter = 0
        for epoch in range(self.startEpoch, self.params.max_epochs):
            if dist.is_initialized():
                self.train_sampler.set_epoch(epoch)
#        self.valid_sampler.set_epoch(epoch)

            start = time.time()
            tr_time, data_time, train_logs = self.train_one_epoch()
            valid_time, valid_logs = self.validate_one_epoch()

            if self.params.scheduler == 'ReduceLROnPlateau':
                self.scheduler.step(valid_logs['valid_loss'])
            elif self.params.scheduler == 'CosineAnnealingLR':
                self.scheduler.step()
                if self.epoch >= self.params.max_epochs:
                    logging.info("Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR")
                    exit()

            if self.params.log_to_wandb:
                for pg in self.optimizer.param_groups:
                    lr = pg['lr']
                wandb.log({'lr': lr, 'epoch': self.epoch})
            
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
            
            # Early stopping check
            if self.params.early_stopping and early_stopping_counter >= self.params.early_stopping_patience:
                if self.params.log_to_screen:
                    logging.info('Early stopping triggered. Terminating training.')
                return # Exit the train method
            
        # If we've reached this point, we've completed all epochs
        if self.params.log_to_screen:
            logging.info('Completed all epochs. Training finished.')
            


    def train_one_epoch(self):
        self.epoch += 1
        tr_time = 0
        data_time = 0
        self.model.train()

        nb = len(self.train_data_loader)
        pbar = enumerate(self.train_data_loader, 0)
        pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')

        running_results = {"batch_sizes": 0, "loss": 0}

        if self.params.diagnostic_logs:
            diagnostic_logs = {}
        
        # For each epoch, we iterate from 1979 to 2017
        
        for i, data in pbar:
            # Load weather data at time t as the input; load weather data at time t+1/3/6/24 as the output
            # Note the data need to be randomly shuffled
            # Note the input and target need to be normalized, see Inference() for details
            
            self.iters += 1
            # adjust_LR(optimizer, params, iters)
            data_start = time.time()
            #inp_sfc, inp_pl, tar_sfc, tar_pl = map(lambda x: x.to(self.device, dtype=torch.float32), data)
            input_surface, input_upper_air, target_surface, target_upper_air, varying_boundary_data, index_info = map(
                lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), data)
            
            #print(index_info.shape)
            # add noise to the input if self.params.noise_training is not 0.0
            #if self.params.noise_training!=0.0:
            #    input_surface = input_surface + torch.normal(mean=0.0, std=self.params.noise_training, size=input_surface.shape).to(self.device)
            #    input_upper_air = input_upper_air + torch.normal(mean=0.0, std=self.params.noise_training, size=input_upper_air.shape).to(self.device)
            # add a clip to the input to avoid overflow, but need to be careful with the range of the input
            # input_surface = torch.clamp(input_surface, min=-1.0, max=1.0)
            # input_upper_air = torch.clamp(input_upper_air, min=-1.0, max=1.0)

            index_info_names = ['index', 'start_time', 'start_idx', 'start_leap_idx', 'start_hour_diff', 'end_time', 'end_idx', 'end_hour_diff']

            data_time += time.time() - data_start

            tr_start = time.time()

            self.model.zero_grad()

            # #OPTIMIZATION
            # self.model.zero_grad(set_to_none=True)

            if self.params.enable_fp8:
                precision_context = fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe)
            else:
                precision_context = amp.autocast(enabled=self.params.enable_amp)

            with precision_context:
                '''input, input_surface, target, target_surface = LoadData(step)

                # Call the model and get the output
                output, output_surface = model(input, input_surface)

                # Call the backward algorithm and calculate the gratitude of parameters
                Backward(loss)

                # Update model parameters with Adam optimizer
                # The learning rate is 5e-4 as in the paper, while the weight decay is 3e-6
                # A example solution is using torch.optim.adam
                UpdateModelParametersWithAdam()'''
                output_surface, output_upper_air = self.model(input_surface, self.constant_boundary_data, 
                                                            varying_boundary_data, input_upper_air)
                
                loss_sfc = self.loss_obj_sfc(output_surface, target_surface)
                loss_pl = self.loss_obj_pl(output_upper_air, target_upper_air)

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
                surface_lwrmse = weighted_rmse_torch_channels(output_surface, target_surface, self.train_dataset.lat.to(self.device, non_blocking=True))
                upper_air_lwrmse = weighted_rmse_torch_3D(output_upper_air, target_upper_air, self.train_dataset.lat.to(self.device, non_blocking=True))

                if self.params.diagnostic_logs:
                    #for batch_idx in range(index_info.shape[0]):
                    #    for j, index_type in enumerate(index_info_names):
                    #        diagnostic_logs[f'{index_type}_batch{batch_idx}_gpu{self.world_rank}'] = index_info[batch_idx, j]
                    diagnostic_logs['batch_grad_norm'] = torch.tensor([grad_norm(self.model)]).to(self.device)
                    diagnostic_logs['batch_grad_max'] = torch.tensor([grad_max(self.model)]).to(self.device)
                    diagnostic_logs['train_batch_loss'] = loss
                    diagnostic_logs['train_batch_loss_sfc'] = loss_sfc
                    diagnostic_logs['train_batch_loss_upper_air'] = loss_pl
                    mean_norm_lwrmse = torch.mean(torch.cat((surface_lwrmse, upper_air_lwrmse.reshape(output_upper_air.shape[0], -1)), dim = -1))
                    diagnostic_logs['train_mean_norm_lwrmse'] = mean_norm_lwrmse
                    for j, var in enumerate(self.train_dataset.surface_variables):
                        diagnostic_logs[f'train_{var}_lwrmse'] = torch.mean(surface_lwrmse[:, j]) * self.train_dataset.surface_std[j]
                    for j, var in enumerate(self.train_dataset.upper_air_variables):
                        for k, level in enumerate(self.train_dataset.lev):
                            diagnostic_logs[f'train_{var}_level{level:.4f}_lwrmse'] = torch.mean(upper_air_lwrmse[:, j, k]) * self.train_dataset.upper_air_std[j, k]
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
                        wandb.log(diagnostic_logs, step=(self.epoch-1) * len(self.train_data_loader) + i)
                

            torch.cuda.empty_cache() #Check

            tr_time += time.time() - tr_start

            if self.params.diagnostic_logs:
                pbar.set_description(desc="Loss: %.4f" % diagnostic_logs['train_batch_loss'])
            else:
                running_results["loss"] += loss.item() * self.params['batch_size']
                running_results["batch_sizes"] += self.params['batch_size']

                pbar.set_description(desc="Loss: %.4f" % (running_results["loss"] / running_results["batch_sizes"]))
        

        if self.params.diagnostic_logs:
            with torch.no_grad():
                diagnostic_logs['train_loss'] = loss
                if dist.is_initialized():
                    dist.all_reduce(diagnostic_logs['train_loss'].detach())
                    diagnostic_logs['train_loss'] = float(diagnostic_logs['train_loss']/dist.get_world_size())
                logs = {'train_loss': diagnostic_logs['train_loss'], 'epoch': self.epoch}
                if self.params.log_to_wandb:
                    wandb.log(logs)
                return tr_time, data_time, diagnostic_logs
        else:
            with torch.no_grad():
                # logs = {'train_loss': loss, 'epoch': self.epoch}
                logs = {'train_loss': loss, 'epoch': self.epoch}
            
            if dist.is_initialized():
                for key in sorted(logs.keys()):
                    if isinstance(logs[key], (int, float)):
                        logs[key] = torch.tensor(logs[key]).to(self.device)
                    dist.all_reduce(logs[key])
                    logs[key] = float(logs[key]/dist.get_world_size())


            # if dist.is_initialized():
            #     for key in sorted(logs.keys()):
            #         dist.all_reduce(logs[key].detach())
            #         logs[key] = float(logs[key]/dist.get_world_size())

            if self.params.log_to_wandb:
                wandb.log(logs)

            return tr_time, data_time, logs


    def validate_one_epoch(self):
        self.model.eval()
        #n_valid_batches = 50  # do validation on first 50 images, just for LR scheduler

        # define the lead times to evaluate (in time steps)
        lead_times_steps = self.params.forecast_lead_times

        valid_buff = torch.zeros((4), dtype=torch.float32, device=self.device)
        valid_loss = valid_buff[0].view(-1)
        valid_loss_sfc = valid_buff[1].view(-1)
        valid_loss_pl = valid_buff[2].view(-1)
        valid_steps = valid_buff[3].view(-1)
        valid_surface_lwrmse = torch.zeros((len(lead_times_steps), len(self.valid_dataset.surface_variables)), dtype=torch.float32, device=self.device)
        valid_upper_air_lwrmse = torch.zeros((len(lead_times_steps), len(self.valid_dataset.upper_air_variables), self.valid_dataset.num_levels), dtype=torch.float32, device=self.device)

        

        multi_step_losses = {f"valid_loss_{step}step": torch.zeros(1, dtype=torch.float32, device=self.device) for step in lead_times_steps}
        # Add RMSE storage for multiple lead times
        multi_step_rmse = {f"valid_lwrmse_sfc_{step}step": torch.zeros(1, dtype=torch.float32, device=self.device) for step in lead_times_steps} |\
            {f"valid_lwrmse_pl_{step}step": torch.zeros(1, dtype=torch.float32, device=self.device) for step in lead_times_steps}



        valid_start = time.time()
        nb = len(self.valid_data_loader)
        if self.params.diagnostic_logs:
            diagnostic_logs = {}
        


        sample_idx = np.random.randint(len(self.valid_data_loader))

        # OPTIMIZATION
        # with torch.inference_mode():
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.valid_data_loader, 0), total=nb, bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}'):
                #if i >= n_valid_batches:
                #    break
                val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air, val_varying_boundary_data, times = map(
                    lambda x: x.to(self.device, dtype=torch.float32, non_blocking=True), data)
                
                max_lead_time = max(lead_times_steps)

                precision_context = fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe) if self.params.enable_fp8 else amp.autocast(enabled=self.params.enable_amp)

                with precision_context:
                     # Autoregressive prediction
                    val_output_surface, val_output_upper_air = val_input_surface, val_input_upper_air
                    step_idx = 0
                    for step in range(max_lead_time):
                        val_output_surface, val_output_upper_air = self.model(val_output_surface, self.constant_boundary_data, 
                                                                            val_varying_boundary_data[:, step], val_output_upper_air)
                        
                        # Calculate losses for different lead times
                        if (step + 1) in lead_times_steps:
                            target_index = lead_times_steps.index(step + 1)
                            loss_sfc = self.loss_obj_sfc(val_output_surface, val_target_surface[:,target_index])
                            loss_pl = self.loss_obj_pl(val_output_upper_air, val_target_upper_air[:,target_index])
                            loss = (loss_sfc * 0.25 + loss_pl)
                            multi_step_losses[f"valid_loss_{step+1}step"] += loss

                            # Calculate RMSE
                            rmse_sfc = weighted_rmse_torch_channels(val_output_surface, val_target_surface[:,target_index], self.valid_dataset.lat.to(self.device, non_blocking=True))
                            rmse_pl = weighted_rmse_torch_3D(val_output_upper_air, val_target_upper_air[:,target_index], self.valid_dataset.lat.to(self.device, non_blocking=True))
                            multi_step_rmse[f"valid_lwrmse_sfc_{step+1}step"] += torch.mean(rmse_sfc)
                            multi_step_rmse[f"valid_lwrmse_pl_{step+1}step"] += torch.mean(rmse_pl)

                            valid_surface_lwrmse[step_idx] += torch.mean(rmse_sfc, dim = 0)
                            valid_upper_air_lwrmse[step_idx] += torch.mean(rmse_pl, dim=0)

                            if step + 1 == max_lead_time:
                                valid_loss += loss
                                valid_loss_sfc += loss_sfc
                                valid_loss_pl += loss_pl

                                #surface_lwrmse = weighted_rmse_torch_channels(val_output_surface, val_target_surface[:, target_index], self.valid_dataset.lat.to(self.device, non_blocking=True))
                                #upper_air_lwrmse = weighted_rmse_torch_3D(val_output_upper_air, val_target_upper_air[:, target_index], self.valid_dataset.lat.to(self.device, non_blocking=True))

                                #valid_surface_lwrmse += torch.mean(surface_lwrmse, dim=0)
                                #valid_upper_air_lwrmse += torch.mean(upper_air_lwrmse, dim=0)
                            
                            step_idx += 1

                valid_steps += 1.

                            

                # loss_sfc = self.loss_obj_sfc(val_output_surface, val_target_surface[-1])
                # loss_pl = self.loss_obj_pl(val_output_upper_air, val_target_upper_air[-1])
                
                # loss = (loss_sfc * 0.25 + loss_pl)
                # valid_loss += loss
                # #valid_l1 += (torch.nn.functional.l1_loss(val_output_surface, val_target_surface) + \
                # #    torch.nn.functional.l1_loss(val_output_upper_air, val_target_upper_air))
                
                # valid_loss_sfc += loss_sfc 
                # valid_loss_pl += loss_pl
                    
                # surface_lwrmse = weighted_rmse_torch_channels(val_output_surface, val_target_surface, self.valid_dataset.lat.to(self.device, non_blocking=True))
                # upper_air_lwrmse = weighted_rmse_torch_3D(val_output_upper_air, val_target_upper_air, self.valid_dataset.lat.to(self.device, non_blocking=True))

                # valid_surface_lwrmse += torch.mean(surface_lwrmse, dim = 0)
                # valid_upper_air_lwrmse += torch.mean(upper_air_lwrmse, dim = 0)

                # valid_steps += 1.



                # save first channel of first 5 images
                #if i < 5:
                #    try:
                #        os.mkdir(params['experiment_dir'] + "/" + str(i))
                #    except:
                #        pass
                #
                #    save_image(torch.cat((val_output_surface[0, 0], torch.zeros((val_output_surface.shape[2], 4)).to(self.device, dtype=torch.float), 
                #                      val_target_surface[0, 0]), axis=1), params['experiment_dir'] + "/" + str(i) + "/" + str(self.epoch) + ".png")
        if dist.is_initialized():
            dist.all_reduce(valid_buff)
            dist.all_reduce(valid_surface_lwrmse)
            dist.all_reduce(valid_upper_air_lwrmse)
            for loss_tensor in multi_step_losses.values():
                dist.all_reduce(loss_tensor)

        # divide by number of steps
        valid_buff[0:3] = valid_buff[0:3] / valid_buff[3]
        valid_surface_lwrmse = (valid_surface_lwrmse / valid_buff[3]).detach()
        valid_upper_air_lwrmse = (valid_upper_air_lwrmse / valid_buff[3]).detach()
        for key in multi_step_losses:
            multi_step_losses[key] /= valid_buff[3]

        valid_buff_cpu = valid_buff.detach()
                    
        if self.params.diagnostic_logs:
            diagnostic_logs['epoch'] = self.epoch
            diagnostic_logs['valid_loss'] = valid_buff_cpu[0]
            diagnostic_logs['valid_loss_sfc'] = valid_buff_cpu[1]
            diagnostic_logs['valid_loss_upper_air'] = valid_buff_cpu[2]
            #mean_norm_lwrmse = torch.mean(torch.cat((valid_surface_lwrmse, valid_upper_air_lwrmse.flatten()), dim = -1))
            #diagnostic_logs['valid_mean_norm_lwrmse'] = mean_norm_lwrmse
            for l, steps in enumerate(lead_times_steps):
                for j, var in enumerate(self.valid_dataset.surface_variables):
                    diagnostic_logs[f'valid_{var}_{steps}step_lwrmse'] = valid_surface_lwrmse[l, j] * self.valid_dataset.surface_std[j]
                for j, var in enumerate(self.valid_dataset.upper_air_variables):
                    for k, level in enumerate(self.valid_dataset.lev):
                        diagnostic_logs[f'valid_{var}_level{level:.3f}_{steps}step_lwrmse'] = valid_upper_air_lwrmse[l, j, k] * self.valid_dataset.upper_air_std[j, k]
            #if dist.is_initialized():
            #    for key in sorted(diagnostic_logs.keys()):
            #        dist.all_reduce(diagnostic_logs[key].detach())
            #        diagnostic_logs[key] = float(diagnostic_logs[key]/dist.get_world_size())

            # Add multi-day losses to diagnostic logs
            for key, value in multi_step_losses.items():
                diagnostic_logs[key] = value.item()

            if self.params.log_to_wandb:
                wandb.log(diagnostic_logs)
            
            valid_time = time.time() - valid_start

            return valid_time, diagnostic_logs
        else:
            try:
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

            valid_time = time.time() - valid_start

            return valid_time, logs

    
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
    parser.add_argument("--run_num", default='0001', type=str)
    parser.add_argument("--yaml_config", default='v2.0/config/PANGU_PLASIM.yaml', type=str)
    parser.add_argument("--config", default='PLASIM', type=str)
    parser.add_argument("--enable_amp", default=True, action='store_true')
    parser.add_argument("--epsilon_factor", default=0, type=float)
    parser.add_argument("--epochs", default=0, type=int)
    parser.add_argument("--num_inferences", default = 0, type = int)
    # parser.add_argument("--window_size", default = '2,2,2', type = str)

    parser.add_argument("--fresh_start", action="store_true", help="Start training from scratch, ignoring existing checkpoints")


    ####### for UCAR
    parser.add_argument("--local-rank", type=int)
    #######

    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    if args.epochs > 0:
        params['max_epochs'] = args.epochs
    params['epsilon_factor'] = args.epsilon_factor
    params['num_inferences'] = args.num_inferences
    #params['loss'] = args.loss

    # Add mandatory check for autoregressive steps
    #max_forecast_lead_time = max(params.forecast_lead_times)
    #if params.autoreg_steps < max_forecast_lead_time:
    #    raise ValueError(f"autoregressive steps ({params.autoreg_steps}) must be >= "
    #                     f"the maximum forecast lead time ({max_forecast_lead_time})")
    
    print('World size from OS: %d' % int(os.environ['WORLD_SIZE']))
    print('World size from Cuda: %d' % torch.cuda.device_count())
    if 'WORLD_SIZE' in os.environ:
        params['world_size'] = int(os.environ['WORLD_SIZE'])
        print(params['world_size'])
    else:
        params['world_size'] = torch.cuda.device_count()
        print(params['world_size'])

    #params['world_size'] = 1
    '''if torch.cuda.device_count() == 1:
        world_rank = 0
        local_rank = 0
        params['batch_size'] = params['batch_size']//4'''
    
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
    if params.fresh_start:
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

    # this will be the wandb name
    params['name'] = args.config + '_' + str(args.run_num)
    params['group'] = "Pangu_plasim_" + args.config  
    params['project'] = "Pangu-PLASIM"  
    #params['entity'] = "proj-ai-weather"
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
    trainer.train()
    logging.info('DONE ---- rank %d' % world_rank)

