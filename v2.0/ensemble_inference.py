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
#from apex import optimizers
from pathlib import Path
#import dask
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

from train import Trainer

config = 'PLASIM'
yaml_config = 'config/PANGU_PLASIM_H5_PERLMUTTER_0510_ensemble.yaml'
just_validate = True
debug = True
run_num = '0510'

params = YParams(os.path.abspath(yaml_config), config)
#params['epsilon_factor'] = args.epsilon_factor
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
params['just_validate'] = just_validate
if params.just_validate:
    os.environ["WANDB_MODE"] = "offline"
# params['num_inferences'] = args.num_inferences
#params['loss'] = args.loss

# # Add mandatory check for autoregressive steps
# max_forecast_lead_time = max(params.forecast_lead_times)
# if params.autoreg_steps < max_forecast_lead_time:
#     raise ValueError(f"autoregressive steps ({params.autoreg_steps}) must be >= "
#                      f"the maximum forecast lead time ({max_forecast_lead_time})")

#os.environ['WANDB_MODE'] = 'offline'

if debug:
    params['world_size'] = 1
    os.environ['WANDB_MODE'] = 'offline'
else:
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

    gpu = local_rank
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
expDir = os.path.join(params.exp_dir, config, str(run_num))
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
params['resuming'] = True
if world_rank == 0:
    logging.info("Resuming from existing checkpoint.")

# # Do not comment this line out please:
# # args.resuming = True if os.path.isfile(params.checkpoint_path) else False
# args.resuming = False
# params['resuming'] = args.resuming


params['local_rank'] = local_rank
params['enable_amp'] = False

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
# params['name'] = args.config + '_' + str(args.run_num)
# params['group'] = "Pangu_plasim_" + args.config  
# params['project'] = "Pangu-PLASIM"  
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


init_date = cftime.Date
trainer = Trainer(params, world_rank)

if params.diagnostic_gif:
    if not hasattr(params, "diagnostic_gif_var_dict"):
        params['diagnostic_gif_var_dict'] = {'zg': [50000]}
        

if hasattr(params, 'use_sigma_levels'):
    if params.use_sigma_levels:
        print('For sigma level training, disabling diagnostic ACC and diagnostic spectra')
        params['diagnostic_acc'] = False
        params['diagnostic_spectra'] = False


trainer.validate_one_epoch()
logging.info('DONE ---- rank %d' % world_rank)