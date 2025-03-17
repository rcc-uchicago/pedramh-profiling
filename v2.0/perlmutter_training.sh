#!/bin/bash -l
#SBATCH --account=m4416
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH -o outs/%x_%j.out
#SBATCH -e outs/%x_%j.err

#echo $SLURM_NTASKS   # WORLD_SIZE
#echo $SLURM_PROCID   # WORLD_RANK
#echo $SLURM_LOCALID  # LOCAL_RANK
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export HDF5_USE_FILE_LOCKING=FALSE


#./home/awikner/miniconda3/bin/conda init; bash
module load conda
conda activate py311_pip_sfno
#export cuda_version=12.1
#export CUDA_HOME=/usr/local/cuda-${cuda_version}
#export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
#export PATH=$CUDA_HOME/bin:$PATH

# source activate /scratch/midway3/tvallabh/tarun_pangu



# Change to working directory
cd /pscratch/sd/a/awikner/PanguWeather/v2.0
#source export_DDP_vars.sh
which conda
#python test_torch.py
export WANDB_MODE=offline

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


if [[ "$3" == "1" ]]; then
	CMD="python train.py --yaml_config=${2} --run_num=${1} --debug"
else
	CMD="python -m torch.distributed.launch --nproc_per_node=${NUM_TASKS_PER_NODE} train.py --yaml_config=${2} --run_num=${1}"
fi
if [[ -z "$4" ]]; then
	CMD+=" --fresh_start"
fi
# Launch your script using torch.distributed.launch
eval "$CMD"
