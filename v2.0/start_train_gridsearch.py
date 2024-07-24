import os, shutil, sys
import numpy as np
import subprocess
from itertools import product

def create_config_file(run_num, loss = 'l1', lr = 1e-4, window_size = (2, 6, 12), epsilon_factor = 0.,
    base_config = 'config/PANGU_PLASIM_FASTER_BASE.yaml'):
    new_config = '_'.join(base_config.split('_')[:-1]) + f'_{run_num}.yaml'
    with open(base_config, 'r') as f:
        lines = f.readlines()
    with open(new_config, 'w') as f:
        for line in lines:
            line = line.replace('%LOSS%', loss)
            line = line.replace('%LR%', str(lr))
            line = line.replace('%WINDOW_SIZE%', str(window_size))
            line = line.replace('%EPSILON%', str(epsilon_factor))
            f.write(line)
    return new_config

losses = ['l1', 'weightedl1', 'weightedl2']
lrs = [1e-4]
window_sizes = [[2, 6, 12], [2, 4, 8], [2, 2, 4]]
epsilon_factors = [0, 5e-2]
num_runs = 18

run_num = 100
for i, (loss, lr, window_size, epsilon_factor) in enumerate(product(losses, lrs, window_sizes, epsilon_factors)):
    run_num_str = f'{run_num + i + 1:04}'
    config_file = create_config_file(run_num_str, loss, lr, window_size, epsilon_factor)
    if i > 13 and i != 14:
        gpu_str = '--gpus=a40:4'
        job_str = ['sbatch', '-J', f'pp-{run_num_str}', '--time=4-00:00:00', gpu_str, '--ntasks=4',
             '--mem=120G', 'faster_ddp_2.sh', run_num_str, config_file]
        subprocess.run(job_str)
    elif i > 3 and i != 14:
        gpu_str = '--gpus=a100:4'
        job_str = ['sbatch', '-J', f'pp-{run_num_str}', gpu_str, '--mem=200G', 'faster_ddp_2.sh', run_num_str, config_file]
        subprocess.run(job_str)
    #else:
    #    gpu_str = '--gpus=a100:4'
    #job_str = ['sbatch', '-J', f'pp-{run_num_str}', gpu_str, 'faster_ddp.sh', run_num_str, config_file]
    #subprocess.run(job_str)



