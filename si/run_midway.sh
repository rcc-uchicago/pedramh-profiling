#!/bin/bash
#SBATCH --account=pi-pedramh
#SBATCH --time=48:00:00
#SBATCH --mem=500G
#SBATCH -p pedramh-gpu 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4 
#SBATCH -o SI_%x_%j.out
#SBATCH -e SI_%x_%j.err

module load python/miniforge-25.3.0

eval "$(conda shell.bash hook)"
conda activate /project/pedramh/shared/anthonyz/venv

# sinteractive --account=pi-pedramh --partition=pedramh-gpu --nodes=1 --time=48:00:00 --ntasks-per-node=4 --gres=gpu:4 --cpus-per-task=8 --mem=500G

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

config_file=configs/SI_midway.yaml

srun python train.py --config=$config_file