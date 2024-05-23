#!/bin/bash -l
#PBS -N random
##PBS -l select=2:system=polaris:ncpus=128:ngpus=8:gputype=A100
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -q debug 
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle                          
#PBS -A MDClimSim
#PBS -e logs/
#PBS -o logs/

# Enable GPU-MPI (if supported by application)
export MPICH_GPU_SUPPORT_ENABLED=1

module use /soft/modulefiles
ml conda
conda activate /eagle/MDClimSim/hyadav/Pangu_env_2 
ml cudatoolkit-standalone/12.2.2

echo "Job ID: ${PBS_JOBID}"
FINAL_MAX_EPOCHS=50

EPOCHS_PER_JOB=2
if [ -z "${MAX_EPOCHS}" ]; then MAX_EPOCHS=$EPOCHS_PER_JOB; fi

MAX_EPOCHS_NEXT=$((MAX_EPOCHS + EPOCHS_PER_JOB))
#JOB_ID="${PBS_JOBID%%.*}"
JOB_ID=$PBS_JOBID
echo "Max Epochs: ${MAX_EPOCHS}"
echo "Max Epochs next: ${MAX_EPOCHS}"
echo "Job ID: ${JOB_ID}"

if [ "$MAX_EPOCHS_NEXT" -le "$FINAL_MAX_EPOCHS" ]; then qsub -v MAX_EPOCHS=$MAX_EPOCHS_NEXT -W depend=afterok:$JOB_ID /eagle/MDClimSim/awikner/PanguWeather-UC/v2.0/polaris_ddp.sh; fi

# Change to working directory
cd $PBS_O_WORKDIR

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
#Follwing will be the number of GPUs on each node, so 4 in our case as each node has 4 GPUs
NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
#NUM_TASKS_PER_NODE=2
WORLD_SIZE=$((NNODES * NUM_TASKS_PER_NODE))

echo "NUM_OF_NODES= ${NNODES} NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE} WORLD_SIZE= ${WORLD_SIZE}"

# Set up the PyTorch distributed environment
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12345
export WORLD_SIZE
export RANK=$PBS_ARRAYID

# Launch your script using torch.distributed.launch
python -m torch.distributed.launch --nproc_per_node=$NUM_TASKS_PER_NODE train.py --yaml_config=/eagle/MDClimSim/awikner/PanguWeather-UC/v2.0/config/PANGU_PLASIM_POLARIS.yaml --epochs=$MAX_EPOCHS
