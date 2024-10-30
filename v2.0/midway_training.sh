#!/bin/bash -l
#SBATCH --account=pi-pedramh
#SBATCH --time=6-00:00:00
#SBATCH -p pedramh-gpu
##SBATCH -p schmidt-gpu 
##SBATCH --account=pi-dfreeman
##SBATCH --time=8:00:00
#SBATCH --nodes=1
###SBATCH --nodelist=midway3-0559
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8 #16 
#SBATCH -o outs/midway_ddp_%x_%j.out
#SBATCH -e outs/midway_ddp_%x_%j.err

#echo $SLURM_NTASKS   # WORLD_SIZE
#echo $SLURM_PROCID   # WORLD_RANK
#echo $SLURM_LOCALID  # LOCAL_RANK
export MPICH_GPU_SUPPORT_ENABLED=1

ulimit -l unlimited

ml python
source activate /project/pedramh/anaconda/py311
source /home/awikner/venvs/pangu-wandb/bin/activate  
export WANDB_MODE=offline

# Change to working directory
#cd $SLURM_SUBMIT_DIR
source export_DDP_vars.sh

# MPI and OpenMP settings
# NNODES=`wc -l < $SLURM_JOB_NODELIST`
#Follwing will be the number of GPUs on each node, so 4 in our case as each node has 4 GPUs
export NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
#NUM_TASKS_PER_NODE=2
#WORLD_SIZE=$((NNODES * NUM_TASKS_PER_NODE))

echo "NUM_OF_NODES= ${NNODES} NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE} WORLD_SIZE= ${WORLD_SIZE}"

# Set up the PyTorch distributed environment
#export MASTER_ADDR=$(hostname)
#export MASTER_PORT=12345
#export WORLD_SIZE
#export RANK=$SLURM_ARRAY_TASK_ID
#export OMP_NUM_THREAD=8

export NUM_TASKS_PER_NODE=1

# Launch your script using torch.distributed.launch
/project/pedramh/anaconda/py311/bin/python -m torch.distributed.launch --nproc_per_node=$NUM_TASKS_PER_NODE train.py --yaml_config=$2 --run_num=$1
