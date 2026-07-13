#!/bin/bash

#SBATCH -A TG-ATM170020
#SBATCH -p h100
#SBATCH -t 24:00:00
#SBATCH -N 1 # nodes
#SBATCH -n 4
#SBATCH  --mem=1000G
#SBATCH -o stampede_ddp_%x_%j.out
#SBATCH -e stampede_ddp_%x_%j.err


### Modeules needed for cuda to work
module load gcc/15.1.0 nvidia/25.3 opencilk/2.1.0
module load cuda/12.8
module load nvidia/24.5

source  /home1/10786/bgong1/.bashrc
conda activate /home1/10786/bgong1/stampede3/env


export NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)


config_file=../config/exp2.yaml


nsys profile -o output_profile_baseline --trace=cuda,nvtx,osrt,cudnn,cublas --force-overwrite=true \
 torchrun --nproc_per_node=$NUM_TASKS_PER_NODE --standalone \
 ../train.py --yaml_config=$config_file --run_num=1stampede

# torchrun --standalone -m torch.distributed.launch --nproc_per_node=gpu ../train.py --yaml_config=$config_file --run_num=1stampede

