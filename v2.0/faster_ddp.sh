#!/bin/bash -l
#SBATCH --time=2-00:00:00
#SBATCH -p gpu
#SBATCH --mem-per-cpu=4G 
#SBATCH --nodes=1
#SBATCH --gpus=a40:4       #gpus=a100:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16 
#SBATCH -o outs/faster_ddp-%j.out

#echo $SLURM_NTASKS   # WORLD_SIZE
#echo $SLURM_PROCID   # WORLD_RANK
#echo $SLURM_LOCALID  # LOCAL_RANK


export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(hostname)


set -x
srun -u --mpi=pmi2 \
    bash -c "
    source export_DDP_vars.sh
    python train.py --window_size=${1} --epsilon_factor=${2} --loss=${3}
    "
