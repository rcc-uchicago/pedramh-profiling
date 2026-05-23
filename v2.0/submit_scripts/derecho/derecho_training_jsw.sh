#!/bin/bash -l
#PBS -N e3sm_sfno_train
#PBS -l select=1:ncpus=64:ngpus=1:mem=480G
#PBS -q develop
#PBS -l walltime=01:00:00
#PBS -A UCHI0018
#PBS -j oe

#echo $SLURM_NTASKS   # WORLD_SIZE
#echo $SLURM_PROCID   # WORLD_RANK
#echo $SLURM_LOCALID  # LOCAL_RANK
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export HDF5_USE_FILE_LOCKING=FALSE

PRELOAD="module load conda;"
conda activate sfno_pangu
# conda activate /glade/work/awikner/conda-envs/aires_panguplasim
# Change to working directory
cd /glade/work/jesswan/UChicago/PanguWeather/v2.0
#source export_DDP_vars.sh
which conda
#python test_torch.py
export WANDB_MODE=online

nvidia-smi

# MPI and OpenMP settings
if [[ -z "${PBS_NODEFILE}" ]]; then
    RANKS=$HOSTNAME
    NNODES=1
else
    MASTER_RANK=$(head -n 1 $PBS_NODEFILE)
    RANKS=$(tr '\n' ' ' < $PBS_NODEFILE)
    NNODES=$(< $PBS_NODEFILE wc -l)
fi

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
export OMP_NUM_THREADS=1 #OMP_NUM_THREADS=(cpus/gpu)/batch_size
# export OMP_NUM_THREADS=2 #OMP_NUM_THREADS=(cpus/gpu)/batch_size

if [[ "$DEBUG" == "1" ]]; then
	CMD="python train.py --yaml_config=${YAML_CONFIG} --run_num=${RUN_NUM} --debug --config=SFNO"
else
	# CMD="python -m torch.distributed.launch --nproc_per_node=${NUM_TASKS_PER_NODE} train.py --yaml_config=${YAML_CONFIG} --run_num=${RUN_NUM} --config=SFNO"
    CMD="python -m torch.distributed.launch --nproc_per_node=${NUM_TASKS_PER_NODE} train.py --yaml_config=${YAML_CONFIG} --run_num=${RUN_NUM} --config=SFNO --amp_dtype=bfloat16 --use_zero_optimizer='true'"
fi
# if [[ -z "$JOBID" ]]; then
# 	CMD+=" --fresh_start"
# fi
# Launch your script using torch.distributed.launch
eval "$CMD"







