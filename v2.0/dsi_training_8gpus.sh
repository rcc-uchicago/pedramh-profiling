#!/bin/bash -l
#SBATCH -p general
#SBATCH --time=12:00:00
##SBATCH --mem-per-gpu=60G 
##SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --nodelist=i001-ds,j002-ds,j003-ds,j001-ds,j004-ds,j005-ds
#SBATCH --gpus=8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH -o outs/dsi_%x_%j.out
#SBATCH -e outs/dsi_%x_%j.err

#echo $SLURM_NTASKS   # WORLD_SIZE
#echo $SLURM_PROCID   # WORLD_RANK
#echo $SLURM_LOCALID  # LOCAL_RANK
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export HDF5_USE_FILE_LOCKING=FALSE


#./home/awikner/miniconda3/bin/conda init; bash
conda activate py311_pip
#export cuda_version=12.1
#export CUDA_HOME=/usr/local/cuda-${cuda_version}
#export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
#export PATH=$CUDA_HOME/bin:$PATH

# source activate /scratch/midway3/tvallabh/tarun_pangu



# Change to working directory
cd /net/scratch2/awikner/PanguWeather/v2.0
#source export_DDP_vars.sh

nvidia-smi

# MPI and OpenMP settings
NNODES=`wc -l < $SLURM_JOB_NODELIST`
#Follwing will be the number of GPUs on each node, so 4 in our case as each node has 4 GPUs
export NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
#NUM_TASKS_PER_NODE=2
WORLD_SIZE=$((NNODES * NUM_TASKS_PER_NODE))

echo "NUM_OF_NODES= ${NNODES} NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE} WORLD_SIZE= ${WORLD_SIZE}"

# Set up the PyTorch distributed environment
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12345
export WORLD_SIZE
export RANK=$SLURM_ARRAY_TASK_ID
export OMP_NUM_THREADS=1


# Launch your script using torch.distributed.launch
python -m torch.distributed.launch --nproc_per_node=$NUM_TASKS_PER_NODE train.py --yaml_config=$2 --run_num=$1
# --enable_amp if needed
