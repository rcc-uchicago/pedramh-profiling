#!/bin/bash -l
#SBATCH --time=2-00:00:00
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=a100:4       #gpus=a40:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16 
#SBATCH --mem=200G
#SBATCH -o outs/faster_ddp_%x_%j.out
#SBATCH -e outs/faster_ddp_%x_%j.err

#echo $SLURM_NTASKS   # WORLD_SIZE
#echo $SLURM_PROCID   # WORLD_RANK
#echo $SLURM_LOCALID  # LOCAL_RANK


export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(hostname)

ml Anaconda3
conda activate /scratch/group/p.atm170020.000/anaconda/py311
# source /home/u.aw164890/venvs/pangu/bin/activate
source ~/venvs/amaury-env/bin/activate


source export_DDP_vars.sh

export NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)

echo "NUM_OF_NODES= ${NNODES} NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE} WORLD_SIZE= ${WORLD_SIZE}"

export OMP_NUM_THREADS=1

python -m torch.distributed.launch --nproc_per_node=$NUM_TASKS_PER_NODE train.py --run_num=$1 --yaml_config=$2

#set -x
#srun -u --mpi=pmi2 \
#    bash -c "
#    source export_DDP_vars.sh
#    python train.py --run_num=${1} --yaml_config=${2}
#    "
