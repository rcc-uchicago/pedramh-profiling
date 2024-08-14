#!/bin/bash -l
#PBS -N parallel_test
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -q debug 
#PBS -l walltime=1:00:00
#PBS -l filesystems=home:eagle                          
#PBS -A lighthouse-uchicago
#PBS -e logs/
#PBS -o logs/

# Enable GPU-MPI (if supported by application)
export MPICH_GPU_SUPPORT_ENABLED=1

# Change to working directory
cd $PBS_O_WORKDIR

echo "Job ID: ${PBS_JOBID}"
export PLASIM_TRAIN_ITER=$PLASIM_TRAIN_ITER+1
echo "PLASIM Emulator training epoch: ${PLASIM_TRAIN_ITER}"

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
#Following will be the number of GPUs on each node, so 4 in our case as each node has 4 GPUs
NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
WORLD_SIZE=$((NNODES * NUM_TASKS_PER_NODE))

echo "NUM_OF_NODES= ${NNODES} NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE} WORLD_SIZE= ${WORLD_SIZE}"

# Set up the PyTorch distributed environment
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12345
export WORLD_SIZE
export RANK=$PBS_ARRAYID

module use /soft/modulefiles
module load conda 
conda activate /eagle/MDClimSim/hyadav/Pangu_env_2
module load cudatoolkit-standalone/12.2.2

# Launch your script using torch.distributed.launch
python -m torch.distributed.launch --nproc_per_node=$NUM_TASKS_PER_NODE train.py --yaml_config=/eagle/lighthouse-uchicago/members/hyadav/PanguWeather/v2.0/config/PANGU_PLASIM_POLARIS.yaml
