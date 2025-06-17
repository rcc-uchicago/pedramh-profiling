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
import xarray as xr
import torchvision
from torchvision.utils import save_image
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils
from utils.power_spectrum import *
##########################################
## NEW IMPORTS
from utils.losses import Latitude_weighted_MSELoss, Latitude_weighted_L1Loss, Masked_L1Loss,\
    Masked_MSELoss, Latitude_weighted_masked_L1Loss, Latitude_weighted_masked_MSELoss,\
    Latitude_weighted_CRPSLoss
###############################@###########
logging_utils.config_logger()
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='0194', type=str)
    parser.add_argument("--yaml_config", default='v2.0/config/PANGU_S2S_222_0222.yaml', type=str)
    parser.add_argument("--config", default='S2S', type=str)
    parser.add_argument("--enable_amp", default=True, action='store_true')
    parser.add_argument("--epsilon_factor", default=0, type=float)
    parser.add_argument("--epochs", default=0, type=int)
    parser.add_argument("--run_iter", default=1, type=int)
    # parser.add_argument("--num_inferences", type = int)
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
    # params['num_inferences'] = args.num_inferences
    #params['loss'] = args.loss

    # # Add mandatory check for autoregressive steps
    # max_forecast_lead_time = max(params.forecast_lead_times)
    # if params.autoreg_steps < max_forecast_lead_time:
    #     raise ValueError(f"autoregressive steps ({params.autoreg_steps}) must be >= "
    #                      f"the maximum forecast lead time ({max_forecast_lead_time})")
    
    params['world_size'] = 1
    os.environ['WANDB_MODE'] = 'offline'

    #print('World size from OS: %d' % int(os.environ['WORLD_SIZE']))
    #print('World size from Cuda: %d' % torch.cuda.device_count())
    #if 'WORLD_SIZE' in os.environ:
    #    params['world_size'] = int(os.environ['WORLD_SIZE'])
    #    print(params['world_size'])
    #else:
    #    params['world_size'] = torch.cuda.device_count()
    #    print(params['world_size'])


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

    params['log_to_wandb'] = (world_rank == 0) and params['log_to_wandb']
    params['log_to_screen'] = (world_rank == 0) and params['log_to_screen']
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    data_loader, dataset, sampler = get_data_loader(params, params.data_dir, dist.is_initialized(), 
                                                                     year_start=params.val_year_start, 
                                                                     year_end=params.val_year_end, train=True)
for i, data in tqdm(enumerate(data_loader), total = len(data_loader)):
        if params.has_diagnostic:
            input_surface, input_upper_air, target_surface, target_upper_air, target_diagnostic, varying_boundary_data= map(
                lambda x: x.to(device, dtype=torch.float32, non_blocking=True), data)
        else:
            input_surface, input_upper_air, target_surface, target_upper_air, varying_boundary_data= map(
                lambda x: x.to(device, dtype=torch.float32, non_blocking=True), data)

