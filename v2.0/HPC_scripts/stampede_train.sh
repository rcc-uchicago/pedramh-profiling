#!/bin/bash

#SBATCH -A TG-ATM170020
#SBATCH -p h100
#SBATCH -t 24:00:00
#SBATCH -N 1 # nodes
#SBATCH -n 1

#SBATCH -o stampede_ddp_%x_%j.out
#SBATCH -e stampede_ddp_%x_%j.err


### Modeules needed for cuda to work
module load gcc/15.1.0 nvidia/25.3 opencilk/2.1.0
module load cuda/12.8

source  /home1/10786/bgong1/.bashrc
conda activate /home1/10786/bgong1/stampede3/env


export NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)


config_file=../config/exp2.yaml

torchrun --standalone -m torch.distributed.launch --nproc_per_node=gpu ../train.py --yaml_config=$config_file --run_num=2
#/project/pedramh/anaconda/py311/bin/python -u train.py --yaml_config=$2 --run_num=$1
