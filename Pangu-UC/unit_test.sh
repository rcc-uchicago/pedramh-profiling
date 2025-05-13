#!/bin/bash

#SBATCH --job-name=pangu_plasim_unit_test
#SBATCH --time=00:05:00
#SBATCH --mem=4096M
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --account=145439188689
#SBATCH --output=pangu_plasim_unit_test.%j

module purge
ml Anaconda3
ml CUDA/11.8.0

source activate pangu-uc
pwd
cd /scratch/group/p.atm170020.000/Pangu-UC
/scratch/user/u.aw164890/.conda/envs/pangu-uc/bin/python -m unittest tests/models/pangu/test_main.py
