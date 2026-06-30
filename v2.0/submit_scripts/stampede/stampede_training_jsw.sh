#!/bin/bash -l
#SBATCH -J e3sm_train
#SBATCH -A TG-ATM170020
#SBATCH -p h100
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-user=jesswan@uchicago.edu
#SBATCH --mail-type=all
#SBATCH -o e3sm_train_%j.o
#SBATCH -e e3sm_train_%j.e


export HDF5_USE_FILE_LOCKING=FALSE

# DEBUG=${DEBUG:-0}
# RUN_NUM=${RUN_NUM:-0000}
INPUT_NUM_THREADS=${NUM_THREADS:-1}

module load  gcc/15.1.0 nvidia/25.3 opencilk/2.1.0 cuda/12.8
conda activate /work/11095/jwan4/conda-envs/sfno_pangu

cd /work/11095/jwan4/PanguWeather/v2.0

which conda
export WANDB_MODE=online

nvidia-smi

NNODES=$SLURM_JOB_NUM_NODES
export NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
WORLD_SIZE=$((NNODES * NUM_TASKS_PER_NODE))

echo "NUM_OF_NODES= ${NNODES} NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE} WORLD_SIZE= ${WORLD_SIZE}"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=12345
export WORLD_SIZE
export OMP_NUM_THREADS=$INPUT_NUM_THREADS



if [[ "$DEBUG" == "1" ]]; then
    CMD="python train.py --yaml_config=${YAML_CONFIG} --run_num=${RUN_NUM} --debug --config=SFNO"
else
    CMD="torchrun --nproc_per_node=${NUM_TASKS_PER_NODE} train.py --yaml_config=${YAML_CONFIG} --run_num=${RUN_NUM} --config=SFNO --amp_dtype=bfloat16 --use_zero_optimizer='false'"
fi

eval "$CMD"
