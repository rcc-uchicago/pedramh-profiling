#!/bin/bash -l
#SBATCH -A atm170020-gpu
#SBATCH --time=2-00:00:00
#SBATCH -p gpu
#SBATCH --mem-per-cpu=6G       #mem=128G
#SBATCH --nodes=1
#SBATCH --gres=gpu:4

#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH -o outs/anvil_ddp.out


echo $SLURM_NTASKS   # WORLD_SIZE
echo $SLURM_PROCID   # WORLD_RANK
echo $SLURM_LOCALID  # LOCAL_RANK


export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(hostname)
 
module load hdf5
module load anaconda/2024.02-py311
conda activate /anvil/projects/x-atm170020/anaconda/py311

set -x
srun -u --mpi=pmi2 \
    bash -c "
    source export_DDP_vars.sh
    TORCH_USE_CUDA_DSA=1 python train.py 
    "
