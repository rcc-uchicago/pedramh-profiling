from networks.pangu import PanguModel_Plasim
from tqdm import tqdm
from ruamel.yaml.comments import CommentedMap as ruamelDict
from ruamel.yaml import YAML
from collections import OrderedDict
import matplotlib.pyplot as plt
import wandb
from utils.data_loader_multifiles_test import get_data_loader
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
logging_utils.config_logger()
from apex import optimizers
from pathlib import Path
import dask
dask.config.set(scheduler='synchronous')



class Trainer():
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def __init__(self, params, world_rank):

        self.params = params
        self.world_rank = world_rank
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

        if params.log_to_wandb:
            wandb.init(config=params, name=params.name, group=params.group, project=params.project,
                       entity=params.entity)

        logging.info('rank %d, begin data loader init' % world_rank)
        self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(params, self.world_rank, params.data_dir, dist.is_initialized(), 
                                                                                         year_start=params.train_year_start, 
                                                                                         year_end=params.train_year_end, train=True)
        self.valid_data_loader, self.valid_dataset = get_data_loader(params, self.world_rank, params.data_dir, dist.is_initialized(), 
                                                                     year_start=params.val_year_start, 
                                                                     year_end=params.val_year_end, train=False)

        self.constant_boundary_data = self.train_dataset.constant_boundary_data.unsqueeze(0) * torch.ones(params.batch_size, 1, 1, 1)
        self.constant_boundary_data = self.constant_boundary_data.to(self.device)


        logging.info('rank %d, data loader initialized' % world_rank)


        if params.nettype == 'pangu_plasim':
            self.model = PanguModel_Plasim(params).to(self.device)
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
            logging.info("Loading checkpoint %s" % params.checkpoint_path)
            self.restore_checkpoint(params.checkpoint_path)

        self.epoch = self.startEpoch

        if params.scheduler == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=5, mode='min')
        elif params.scheduler == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=params.max_epochs, 
                                                                        last_epoch=self.startEpoch-1)
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
        else:
            raise NotImplementedError


    def train(self):
        if self.params.log_to_screen:
            logging.info("Starting Training Loop...")

        best_valid_loss = 1.e6
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
                wandb.log({'lr': lr})

            if self.world_rank == 0:
                if self.params.save_checkpoint:
                    # checkpoint at the end of every epoch
                    self.save_checkpoint(self.params.checkpoint_path)
                    if valid_logs['valid_loss'] <= best_valid_loss:
                        # logging.info('Val loss improved from {} to {}'.format(best_valid_loss, valid_logs['valid_loss']))
                        self.save_checkpoint(self.params.best_checkpoint_path)
                        best_valid_loss = valid_logs['valid_loss']

            if self.params.log_to_screen:
                logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
                logging.info('Train loss: {}. Surface MSE: {}. Upper Air MSE:{}'.format(
                    train_logs['loss'], valid_logs['Surface MSE'], valid_logs['Upper Air MSE']))


    def train_one_epoch(self):
        self.epoch += 1
        tr_time = 0
        data_time = 0
        self.model.train()

        nb = len(self.train_data_loader)
        pbar = enumerate(self.train_data_loader, 0)
        pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')

        running_results = {"batch_sizes": 0, "loss": 0}
        
        # For each epoch, we iterate from 1979 to 2017
        for i, data in pbar:
            # Load weather data at time t as the input; load weather data at time t+1/3/6/24 as the output
            # Note the data need to be randomly shuffled
            # Note the input and target need to be normalized, see Inference() for details
            self.iters += 1
            # adjust_LR(optimizer, params, iters)
            data_start = time.time()
            #inp_sfc, inp_pl, tar_sfc, tar_pl = map(lambda x: x.to(self.device, dtype=torch.float32), data)
            input_surface, input_upper_air, target_surface, target_upper_air, varying_boundary_data = map(
                lambda x: x.to(self.device, dtype=torch.float32), data)

            data_time += time.time() - data_start

            tr_start = time.time()

            self.model.zero_grad()

            with amp.autocast(self.params.enable_amp):
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

                
                # We use the MAE loss to train the model
                # The weight of surface loss is 0.25
                # Different weight can be applied for differen fields if needed
                #loss = TensorAbs(output-target) + TensorAbs(output_surface-target_surface) * 0.25
                
                loss_sfc = self.loss_obj_sfc(output_surface, target_surface)
                loss_pl = self.loss_obj_pl(output_upper_air, target_upper_air)

                loss = (loss_sfc * 0.25) + loss_pl

            if self.params.enable_amp:
                self.gscaler.scale(loss).backward()
                self.gscaler.step(self.optimizer)
            else:
                loss.backward()
                self.optimizer.step()

            if self.params.enable_amp:
                self.gscaler.update()

            tr_time += time.time() - tr_start

            running_results["loss"] += loss.item() * self.params['batch_size']
            running_results["batch_sizes"] += self.params['batch_size']

            pbar.set_description(desc="Loss: %.4f" % (running_results["loss"] / running_results["batch_sizes"]))


        logs = {'loss': loss}

        if dist.is_initialized():
            for key in sorted(logs.keys()):
                dist.all_reduce(logs[key].detach())
                logs[key] = float(logs[key]/dist.get_world_size())

        if self.params.log_to_wandb:
            wandb.log(logs, step=self.epoch)

        return tr_time, data_time, logs


    def validate_one_epoch(self):
        self.model.eval()
        n_valid_batches = 50  # do validation on first 50 images, just for LR scheduler

        valid_buff = torch.zeros((5), dtype=torch.float32, device=self.device)
        valid_loss = valid_buff[0].view(-1)
        valid_loss_sfc = valid_buff[1].view(-1)
        valid_loss_pl = valid_buff[2].view(-1)
        valid_l1 = valid_buff[3].view(-1)
        valid_steps = valid_buff[4].view(-1)

        valid_start = time.time()

        sample_idx = np.random.randint(len(self.valid_data_loader))
        with torch.no_grad():
            for i, data in enumerate(self.valid_data_loader, 0):
                if i >= n_valid_batches:
                    break
                val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air, val_varying_boundary_data, times = map(
                    lambda x: x.to(self.device, dtype=torch.float32), data)

                val_output_surface, val_output_upper_air = self.model(val_input_surface, self.constant_boundary_data, 
                                                                      val_varying_boundary_data, val_input_upper_air)

                val_output_surface = val_output_surface.squeeze(0)
                val_output_upper_air = val_output_upper_air.squeeze(0)

                val_target_surface = val_target_surface.squeeze(0)
                val_target_upper_air = val_target_upper_air.squeeze(0)

                loss_sfc = self.loss_obj_sfc(val_output_surface, val_target_surface)
                loss_pl = self.loss_obj_pl(val_output_upper_air, val_target_upper_air)
                
                valid_loss += (loss_sfc + loss_pl) 
                valid_l1 += (torch.nn.functional.l1_loss(val_output_surface, val_target_surface) + \
                    torch.nn.functional.l1_loss(val_output_upper_air, val_target_upper_air))
                
                valid_loss_sfc += loss_sfc 
                valid_loss_pl += loss_pl
                valid_steps += 1.

                # save first channel of first 5 images
                if i < 5:
                    try:
                        os.mkdir(params['experiment_dir'] + "/" + str(i))
                    except:
                        pass

                    save_image(torch.cat((val_output_surface[0, 0], torch.zeros((val_output_surface.shape[2], 4)).to(self.device, dtype=torch.float), 
                                      val_target_surface[0, 0]), axis=1), params['experiment_dir'] + "/" + str(i) + "/" + str(self.epoch) + ".png")

        
        if dist.is_initialized():
            dist.all_reduce(valid_buff)

        # divide by number of steps
        valid_buff[0:4] = valid_buff[0:4] / valid_buff[4]
        
        # download buffers
        valid_buff_cpu = valid_buff.detach().cpu().numpy()

        valid_time = time.time() - valid_start

        try:
            logs = {'valid_l1': valid_buff_cpu[3], 'valid_loss': valid_buff_cpu[0], 
                    'Surface MSE': valid_buff_cpu[1], 'Upper Air MSE': valid_buff_cpu[2]}
        except:
            pass

        if self.params.log_to_wandb:
            wandb.log(logs, step=self.epoch)

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

    ####### for UCAR
    parser.add_argument("--local-rank", type=int)
    #######

    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    if args.epochs > 0:
        params['max_epochs'] = args.epochs
    params['epsilon_factor'] = args.epsilon_factor
    
    print('World size from OS: %d' % int(os.environ['WORLD_SIZE']))
    print('World size from Cuda: %d' % torch.cuda.device_count())
    if 'WORLD_SIZE' in os.environ:
        params['world_size'] = int(os.environ['WORLD_SIZE'])
    else:
        params['world_size'] = torch.cuda.device_count()

    #params['world_size'] = 1
    '''if torch.cuda.device_count() == 1:
        world_rank = 0
        local_rank = 0
        params['batch_size'] = params['batch_size']//4'''
    
    if params['world_size'] > 1:
        dist.init_process_group(backend='nccl', init_method='env://',  world_size=params['world_size'])
        if 'derecho' in str(Path(__file__)):
            local_rank = args.local_rank
        else:
            local_rank = int(os.environ["LOCAL_RANK"])

        args.gpu = local_rank
        world_rank = dist.get_rank()

        params['global_batch_size'] = params.batch_size
        params['batch_size'] = int(params.batch_size//params['world_size'])
    else:
        world_rank = 0
        local_rank = 0

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

    # Do not comment this line out please:
    args.resuming = True if os.path.isfile(params.checkpoint_path) else False

    params['resuming'] = args.resuming
    params['local_rank'] = local_rank
    params['enable_amp'] = args.enable_amp

    # this will be the wandb name
    params['name'] = args.config + '_' + str(args.run_num)
    params['group'] = "Pangu_plasim_" + args.config  
    params['project'] = "Pangu"  
    params['entity'] = "proj-ai-weather"
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
