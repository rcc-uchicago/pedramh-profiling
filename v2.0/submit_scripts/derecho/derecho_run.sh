#!/bin/bash -l
#===============================================================================
# PBS submission script for Pangu-PLASIM training/validation
#===============================================================================
# Usage (direct qsub):
#   qsub -v RUN_NUM=0515 derecho_run.sh                                 # train (default)
#   qsub -v RUN_NUM=0515,SEED=0 derecho_run.sh                          # train with seed
#   qsub -v RUN_NUM=0515,MODE=validate derecho_run.sh                   # validate best_ckpt
#   qsub -v RUN_NUM=0515,MODE=validate,VAL_EPOCHS=10,20 derecho_run.sh  # validate specific epochs
#   qsub -v RUN_NUM=0515,MODE=debug derecho_run.sh                      # debug (single GPU)
#===============================================================================

#--- (defaults for training; validate/debug override below) ---
#PBS -N pangu_run
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -A UCHI0014
#PBS -j oe

#--- Environment Setup ---
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export HDF5_USE_FILE_LOCKING=FALSE

module load conda
conda activate aires_panguplasim

#--- Configuration ---
# Paths (edit these if your setup differs)
CONFIG_DIR="/glade/u/home/plyu/aproject/PanguWeather/v2.0/config"
WORK_DIR="/glade/u/home/plyu/aproject/PanguWeather/v2.0"

# Defaults
MODE="${MODE:-train}"           # train, validate, or debug
SEED="${SEED:-0}"               # global seed
VAL_EPOCHS="${VAL_EPOCHS:-}"    # comma-separated epochs for validation (empty = best_ckpt)

# Derive YAML config path from RUN_NUM
YAML_CONFIG="${CONFIG_DIR}/PANGU_PLASIM_H5_DERECHO_${RUN_NUM}.yaml"

# Validate required variables
if [[ -z "$RUN_NUM" ]]; then
    echo "ERROR: RUN_NUM is required. Use: qsub -v RUN_NUM=0516 $0"
    exit 1
fi

if [[ ! -f "$YAML_CONFIG" ]]; then
    echo "ERROR: Config file not found: $YAML_CONFIG"
    exit 1
fi

#--- Change to working directory ---
cd "$WORK_DIR" || exit 1

echo "=============================================="
echo "Pangu-PLASIM Run Configuration"
echo "=============================================="
echo "MODE:        $MODE"
echo "RUN_NUM:     $RUN_NUM"
echo "SEED:        $SEED"
echo "YAML_CONFIG: $YAML_CONFIG"
echo "VAL_EPOCHS:  ${VAL_EPOCHS:-best_ckpt}"
echo "=============================================="

nvidia-smi

#--- Compute distributed training parameters ---
NNODES=$(wc -l < "$PBS_NODEFILE" 2>/dev/null || echo 1)
NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
WORLD_SIZE=$((NNODES * NUM_TASKS_PER_NODE))

echo "NUM_OF_NODES= ${NNODES} NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE} WORLD_SIZE= ${WORLD_SIZE}"

# PyTorch distributed environment
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12345
export WORLD_SIZE
export OMP_NUM_THREADS=1

#--- Build command ---
# Common arguments (always included)
COMMON_ARGS="--config=PLASIM --yaml_config=${YAML_CONFIG} --run_num=${RUN_NUM} --global_seed=${SEED} --use_legacy_model"

case "$MODE" in
    train)
        export WANDB_MODE=online
        CMD="torchrun --nproc_per_node=${NUM_TASKS_PER_NODE} train.py ${COMMON_ARGS}"
        ;;
    validate)
        export WANDB_MODE=offline
        if [[ -n "$VAL_EPOCHS" ]]; then
            CMD="torchrun --nproc_per_node=${NUM_TASKS_PER_NODE} train.py ${COMMON_ARGS} --just_validate --validation_epochs=${VAL_EPOCHS}"
        else
            CMD="torchrun --nproc_per_node=${NUM_TASKS_PER_NODE} train.py ${COMMON_ARGS} --just_validate"
        fi
        ;;
    debug)
        export WANDB_MODE=offline
        CMD="python train.py ${COMMON_ARGS} --debug"
        ;;
    *)
        echo "ERROR: Unknown MODE '$MODE'. Use: train, validate, or debug"
        exit 1
        ;;
esac

echo "Executing: $CMD"
echo "=============================================="

eval "$CMD"
