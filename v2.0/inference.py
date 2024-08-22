from networks.pangu import PanguModel_Plasim
from tqdm import tqdm
from ruamel.yaml.comments import CommentedMap as ruamelDict
from ruamel.yaml import YAML
from collections import OrderedDict
import matplotlib.pyplot as plt
import wandb
from utils.data_loader_multifiles import get_data_loader
from utils.YParams import YParams
import os, shutil, sys
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
logging_utils.config_logger()
from apex import optimizers
from pathlib import Path
import dask
import cftime
import xarray as xr
from datetime import timedelta
# import transformer_engine.pytorch as te
# from transformer_engine.common import recipe
# from transformer_engine.pytorch import fp8_autocast, fp8_model_init
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor


# fp8_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID,
#                                    amax_history_len=16,
#                                    amax_compute_algo="max")


dask.config.set(scheduler='synchronous')
torch._dynamo.config.optimize_ddp = False
torch.set_float32_matmul_precision('high')

torch.cuda.empty_cache() 

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



class Stepper():
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def __init__(self, params, world_rank, async_save=False):

        self.params = params
        self.world_rank = world_rank
        self.async_save = async_save
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

        if self.async_save:
            logging.info('Asynchronous Saving')
        else: 
            logging.info('Synchronous Saving')


        if params.log_to_wandb:
            wandb.init(config=params, name=params.name, group=params.group, project=params.project,
                       entity=params.entity)

        logging.info('rank %d, begin data loader init' % world_rank)
        self.valid_data_loader, self.valid_dataset = get_data_loader(params, params.data_dir, dist.is_initialized(), 
                                                                     year_start=params.val_year_start, 
                                                                     year_end=params.val_year_end, train=False,
                                                                     num_inferences = params.num_inferences)

        self.constant_boundary_data = self.valid_dataset.constant_boundary_data.unsqueeze(0) * torch.ones(params.batch_size, 1, 1, 1)
        self.constant_boundary_data = self.constant_boundary_data.to(self.device)


        logging.info('rank %d, data loader initialized' % world_rank)



        if params.nettype == 'pangu_plasim':
            # with fp8_model_init(enabled=True):
            self.model = PanguModel_Plasim(params).to(self.device)
            # self.model = PanguModel_Plasim(params).to(self.device)
            # self.model = torch.compile(self.model, mode = 'default')
            # self.model = torch.compile(self.model, backend="torch_tensorrt", dynamic=False)

            # Assuming val_input_surface has the correct shape

        else:
            raise Exception("not implemented")

        if params.log_to_wandb:
            wandb.watch(self.model)

        if dist.is_initialized():
            self.model = DistributedDataParallel(self.model,
                                                 device_ids=[
                                                     params.local_rank],
                                                 output_device=[params.local_rank], find_unused_parameters=True)

        self.iters = 0
        self.startEpoch = 0
        #if params.resuming:
        self.restore_checkpoint(params.checkpoint_path)

        self.epoch = self.startEpoch


        '''if params.log_to_screen:
      logging.info(self.model)'''
        if params.log_to_screen:
            logging.info("Number of trainable model parameters: {}".format(self.count_parameters()))

    def predict(self):
        if self.params.log_to_screen:
            logging.info("Starting Model Inference Loop...")

        start = time.time()
        #tr_time, data_time, train_logs = self.train_one_epoch()
        if self.async_save:
            valid_time, valid_logs = asyncio.run(self.validate_one_epoch_async())
        else:
            valid_time, valid_logs = self.validate_one_epoch_sync()
        

  
        #if self.world_rank == 0:
        #    if self.params.save_checkpoint:
        #        # checkpoint at the end of every epoch
        #        self.save_checkpoint(self.params.checkpoint_path)
        #        if valid_logs['valid_loss'] <= best_valid_loss:
        #            # logging.info('Val loss improved from {} to {}'.format(best_valid_loss, valid_logs['valid_loss']))
        #            self.save_checkpoint(self.params.best_checkpoint_path)
        #            best_valid_loss = valid_logs['valid_loss']

        #if self.params.log_to_screen:
        #    logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        #    logging.info('Train loss: {}. Surface MSE: {}. Upper Air MSE:{}'.format(
        #        train_logs['loss'], valid_logs['Surface MSE'], valid_logs['Upper Air MSE']))


    async def save_prediction_async(self, surface_prediction, upper_air_prediction, start_time, pred_idx):
        await asyncio.to_thread(self.save_prediction, surface_prediction, upper_air_prediction, start_time, pred_idx)

    async def save_results(self, queue):
        while True:
            item = await queue.get()
            if item is None:
                break
            surface_prediction, upper_air_prediction, start_time, pred_idx = item
            save_start = time.time()
            await self.save_prediction_async(surface_prediction, upper_air_prediction, start_time, pred_idx)
            self.save_time += time.time() - save_start
            queue.task_done()

    async def validate_one_epoch_async(self):
        self.model.eval()
        total_start = time.time()
        data_time = 0
        inference_time = 0
        self.save_time = 0

        save_queue = asyncio.Queue()
        save_task = asyncio.create_task(self.save_results(save_queue))

        with torch.inference_mode(), amp.autocast(enabled=self.params.enable_amp):
            for i, data in enumerate(self.valid_data_loader, 0):
                data_start = time.time()
                val_input_surface, val_input_upper_air, val_varying_boundary_data = map(
                    lambda x: x.to(self.device, dtype=torch.float32), data[:-1])
                
                start_idx = data[-1][0,0].item()
                start_hour_diff = data[-1][0,1].item()
                start_time = self.valid_dataset.datetime_class(start_idx + self.params.val_year_start, 1, 1, hour = 0) + timedelta(hours = start_hour_diff)
                pred_year_hours = (self.valid_dataset.datetime_class(start_idx + self.params.val_year_start, 1, 1, hour = 0) - 
                    self.valid_dataset.datetime_class(self.params.val_year_start, 1, 1, hour = 0)).seconds // 3600
                pred_idx = (pred_year_hours + start_hour_diff) // self.params.timedelta_hours
                data_time += time.time() - data_start

                inference_start = time.time()
                val_output_surface = np.zeros((val_input_surface.shape[0], self.params['inference_steps']+1,
                                                val_input_surface.shape[1], val_input_surface.shape[2], val_input_surface.shape[3]),
                                                dtype = np.float32)
                val_output_upper_air = np.zeros((val_input_upper_air.shape[0], self.params['inference_steps']+1,
                                                val_input_upper_air.shape[1], val_input_upper_air.shape[2],
                                                    val_input_upper_air.shape[3], val_input_upper_air.shape[4]),
                                                dtype = np.float32)
                
                val_output_surface[:,0] = self.valid_dataset.surface_inv_transform(val_input_surface.to('cpu')).numpy()
                val_output_upper_air[:,0] = self.valid_dataset.upper_air_inv_transform(val_input_upper_air.to('cpu')).numpy()

                for time_step in range(self.params['inference_steps']):
                    val_input_surface, val_input_upper_air = self.model(val_input_surface, self.constant_boundary_data, 
                                                                        val_varying_boundary_data[:,time_step], val_input_upper_air)
                    val_output_surface[:,time_step + 1] = self.valid_dataset.surface_inv_transform(val_input_surface.to('cpu')).numpy()
                    val_output_upper_air[:,time_step + 1] = self.valid_dataset.upper_air_inv_transform(val_input_upper_air.to('cpu')).numpy()

                inference_time += time.time() - inference_start

                # Queue the results for asynchronous saving
                await save_queue.put((val_output_surface.copy(), val_output_upper_air.copy(), start_time, pred_idx))

        # Signal that we're done
        await save_queue.put(None)
        # Wait for all saves to complete
        await save_task

        total_time = time.time() - total_start

        logs = {
            'total_time': total_time,
            'data_time': data_time,
            'inference_time': inference_time,
            'save_time': self.save_time
        }

        logging.info(f"Validation logs: {logs}")

        if self.params.log_to_wandb:
            wandb.log(logs, step=self.epoch)

        return total_time, logs
    

    def validate_one_epoch_sync(self):
        self.model.eval()
        total_start = time.time()
        data_time = 0
        inference_time = 0
        save_time = 0

        with torch.inference_mode(), amp.autocast(enabled=self.params.enable_amp):
            for i, data in enumerate(self.valid_data_loader, 0):
                data_start = time.time()
                val_input_surface, val_input_upper_air, val_varying_boundary_data = map(
                    lambda x: x.to(self.device, dtype=torch.float32), data[:-1])
                
                start_idx = data[-1][0,0].item()
                start_hour_diff = data[-1][0,1].item()
                start_time = self.valid_dataset.datetime_class(start_idx + self.params.val_year_start, 1, 1, hour = 0) + timedelta(hours = start_hour_diff)
                pred_year_hours = (self.valid_dataset.datetime_class(start_idx + self.params.val_year_start, 1, 1, hour = 0) - 
                    self.valid_dataset.datetime_class(self.params.val_year_start, 1, 1, hour = 0)).seconds // 3600
                pred_idx = (pred_year_hours + start_hour_diff) // self.params.timedelta_hours
                data_time += time.time() - data_start

                inference_start = time.time()
                val_output_surface = np.zeros((val_input_surface.shape[0], self.params['inference_steps']+1,
                                                val_input_surface.shape[1], val_input_surface.shape[2], val_input_surface.shape[3]),
                                                dtype = np.float32)
                val_output_upper_air = np.zeros((val_input_upper_air.shape[0], self.params['inference_steps']+1,
                                                val_input_upper_air.shape[1], val_input_upper_air.shape[2],
                                                    val_input_upper_air.shape[3], val_input_upper_air.shape[4]),
                                                dtype = np.float32)
                
                val_output_surface[:,0] = self.valid_dataset.surface_inv_transform(val_input_surface.to('cpu')).numpy()
                val_output_upper_air[:,0] = self.valid_dataset.upper_air_inv_transform(val_input_upper_air.to('cpu')).numpy()

                for time_step in range(self.params['inference_steps']):
                    val_input_surface, val_input_upper_air = self.model(val_input_surface, self.constant_boundary_data, 
                                                                        val_varying_boundary_data[:,time_step], val_input_upper_air)
                    val_output_surface[:,time_step + 1] = self.valid_dataset.surface_inv_transform(val_input_surface.to('cpu')).numpy()
                    val_output_upper_air[:,time_step + 1] = self.valid_dataset.upper_air_inv_transform(val_input_upper_air.to('cpu')).numpy()

                inference_time += time.time() - inference_start

                save_start = time.time()
                self.save_prediction(val_output_surface, val_output_upper_air, start_time, pred_idx)
                save_time += time.time() - save_start

        total_time = time.time() - total_start

        logs = {
            'total_time': total_time,
            'data_time': data_time,
            'inference_time': inference_time,
            'save_time': save_time
        }

        logging.info(f"Validation logs: {logs}")

        if self.params.log_to_wandb:
            wandb.log(logs, step=self.epoch)

        return total_time, logs
        


    def save_checkpoint(self, checkpoint_path, model=None):
        """ We intentionally require a checkpoint_dir to be passed
            in order to allow Ray Tune to use this function """

        if not model:
            model = self.model

        torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, checkpoint_path)


    # def restore_checkpoint(self, checkpoint_path):
    #     """ We intentionally require a checkpoint_dir to be passed
    #         in order to allow Ray Tune to use this function """
    #     checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.params.local_rank))
    #     try:
    #         self.model.load_state_dict(checkpoint['model_state'])
    #     except:
    #         new_state_dict = OrderedDict()
    #         for key, val in checkpoint['model_state'].items():
    #             name = key[7:]
    #             new_state_dict[name] = val
    #         self.model.load_state_dict(new_state_dict)
    #     self.iters = checkpoint['iters']
    #     self.startEpoch = checkpoint['epoch']
    #     print('START EPOCH:', self.startEpoch)
    #     # restore checkpoint is used for finetuning as well as resuming. If finetuning (i.e., not resuming), restore checkpoint does not load optimizer state, instead uses config specified lr.
    #     if self.params.resuming:
    #         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    def restore_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.params.local_rank))
        model_state_dict = checkpoint['model_state']

        # Remove 'module.' prefix if it exists
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        # Filter out unnecessary keys
        model_dict = self.model.state_dict()
        new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}

        # Update model_dict
        model_dict.update(new_state_dict)

        # Load the filtered state dict
        self.model.load_state_dict(model_dict, strict=False)

        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch']
        print('START EPOCH:', self.startEpoch)

        # Restore optimizer state if resuming
        if self.params.resuming and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print("Checkpoint restored successfully")

    def save_prediction(self, surface_prediction, upper_air_prediction, start_time, pred_idx):
        inference_results_dir = "/home/tvallabh/PanguWeather/v2.0/inference_results"
        savedir = os.path.join(inference_results_dir, 'predictions')
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        pred_config = os.path.join(self.params['experiment_dir'], os.path.basename(params['config_filepath']))
        if not os.path.exists(pred_config):
            shutil.copy(params['config_filepath'], pred_config)
        for sample in range(surface_prediction.shape[0]):
            time_range = xr.cftime_range(start_time + timedelta(hours = self.params['timedelta_hours'] * sample), 
                                         start_time + timedelta(hours = self.params['timedelta_hours'] * (sample + self.params['inference_steps'])),
                                         freq = "%dh" % self.params['timedelta_hours'], inclusive = "both")
            coordinates = {'time': time_range,
                               'lev': self.valid_dataset.data_dss[0].lev.values,
                               'lat': self.valid_dataset.data_dss[0].lat.values,
                               'lon': self.valid_dataset.data_dss[0].lon.values}
            filename = '%s_%s_%dh_%dstep_%d_%d.nc' % (self.params.nettype, self.params.run_num, self.params['timedelta_hours'],
                                                      self.params['inference_steps'], self.params.val_year_start, 
                                                      pred_idx + sample)
            print(filename)
            dataset = xr.Dataset(data_vars = dict(),
                                 coords = coordinates,
                                 attrs = dict(description = f"Prediction from {self.params.nettype} model run {self.params.run_num}"))
            for idx, var in enumerate(self.valid_dataset.surface_variables):
                da = xr.DataArray(data = surface_prediction[sample, :, idx],
                                  dims=["time", "lat", "lon"],
                                  coords = {'time': time_range,
                                                'lat': dataset.lat.values,
                                                'lon': dataset.lon.values
                                                     })
                da = da.assign_attrs(self.valid_dataset.data_dss[0][var].attrs)
                dataset[var] = da
            for idx, var in enumerate(self.valid_dataset.upper_air_variables):
                da = xr.DataArray(data = upper_air_prediction[sample, :, idx],
                                  dims=["time", "lev", "lat", "lon"],
                                  coords = coordinates)
                da = da.assign_attrs(self.valid_dataset.data_dss[0][var].attrs)
                dataset[var] = da
            dataset = dataset.chunk({'time': 1, 'lev': 1})
            #filename = f'{self.params.nettype}_{self.params.run_num}_{self.params['timedelta_hours']}h_{self.params['inference_steps']}step_{self.params.val_start_year}_{batch_idx * self.params.batch_size + sample}.nc'
            dataset.to_netcdf(os.path.join(savedir, filename))

            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='0006', type=str)
    parser.add_argument("--yaml_config", default='v2.0/config/PANGU_PLASIM.yaml', type=str)
    parser.add_argument("--config", default='PLASIM', type=str)
    parser.add_argument("--enable_amp", default=True, action='store_true')
    parser.add_argument("--epsilon_factor", default=0, type=float)
    parser.add_argument("--epochs", default=0, type=int)
    parser.add_argument("--inference_steps", default=0, type=int)
    parser.add_argument("--num_inferences", default = 0, type = int)
    parser.add_argument("--async_save", default = False, action="store_true", help="Enable asynchronous saving")

    ####### for UCAR
    parser.add_argument("--local-rank", type=int)
    #######

    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    if args.epochs > 0:
        params['max_epochs'] = args.epochs
    params['config_filepath'] = args.yaml_config
    params['epsilon_factor'] = args.epsilon_factor
    params['run_num'] = args.run_num
    if args.inference_steps > 0:
        params['inference_steps'] = args.inference_steps
    else:
        try:
            params['inference_steps'] = max(params.forecast_lead_times)
        except:
            print('args.inference_steps and params.forecast_lead_times not set, exiting...')
            sys.exit(2)
    params['num_inferences'] = args.num_inferences
    
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

    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

    # Set up directory
    expDir = os.path.join('/project/pedramh/awikner/PanguWeather/v2.0/results', args.config, str(args.run_num))
    if world_rank == 0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)
            os.makedirs(os.path.join(expDir, 'training_checkpoints/'))

    params['experiment_dir'] = os.path.abspath(expDir)
    ckpt_path = 'training_checkpoints/ckpt.tar'
    best_ckpt_path = 'training_checkpoints/best_ckpt.tar'
    params['checkpoint_path'] = os.path.join(expDir, ckpt_path)
    params['best_checkpoint_path'] = os.path.join(expDir, best_ckpt_path)

    # Do not comment this line out please:
    args.resuming = True if os.path.isfile(params.checkpoint_path) else False

    params['resuming'] = False
    params['local_rank'] = local_rank
    params['enable_amp'] = args.enable_amp

    # this will be the wandb name
    params['name'] = args.config + '_' + str(args.run_num)
    params['group'] = "Pangu_plasim_" + args.config  
    params['project'] = "Pangu"  
    params['entity'] = "proj-ai-weather"
    if world_rank == 0:
        log_file = 'out.log'
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join('/home/tvallabh/PanguWeather/v2.0/inference_results', log_file))
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
    
    inference = Stepper(params, world_rank, args.async_save)
    inference.predict()
    logging.info('DONE ---- rank %d' % world_rank)
