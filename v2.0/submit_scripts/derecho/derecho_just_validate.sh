#!/bin/bash -l
#PBS -N pangu_validate
#PBS -l select=1:ncpus=32:ngpus=4
#PBS -q main
#PBS -l walltime=01:00:00
#PBS -A UCHI0014
#PBS -e logs/
#PBS -o logs/

#echo $SLURM_NTASKS   # WORLD_SIZE
#echo $SLURM_PROCID   # WORLD_RANK
#echo $SLURM_LOCALID  # LOCAL_RANK
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export HDF5_USE_FILE_LOCKING=FALSE

# Set default values for variables that can be passed via qsub
CONFIG=${CONFIG:-PLASIM}
DEBUG=${DEBUG:-0}
USE_LEGACY_MODEL=${USE_LEGACY_MODEL:-0}

module load conda
conda activate aires_panguplasim

# Change to working directory
cd /glade/work/awikner/PanguWeather/v2.0
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


if [[ "$DEBUG" == "1" ]]; then
	CMD="python train.py --config=${CONFIG} --yaml_config=${YAML_CONFIG} --run_num=${RUN_NUM} --debug --just_validate"
else
	CMD="torchrun --nproc_per_node=${NUM_TASKS_PER_NODE} train.py --config=${CONFIG} --yaml_config=${YAML_CONFIG} --run_num=${RUN_NUM} --just_validate"
fi
if [[ -z "$JOBID" ]]; then
	CMD+=" --fresh_start"
fi
# Add validation_epochs argument if VALIDATION_EPOCHS is set
if [[ -n "$VALIDATION_EPOCHS" ]]; then
	CMD+=" --validation_epochs=${VALIDATION_EPOCHS}"
fi
# Add use_legacy_model argument if USE_LEGACY_MODEL is set
if [[ "$USE_LEGACY_MODEL" == "1" ]]; then
	CMD+=" --use_legacy_model"
fi
# Launch your script using torch.distributed.launch
echo "CMD: $CMD"
eval "$CMD"