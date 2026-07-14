#!/bin/bash -l
#SBATCH -J e3sm_inf
#SBATCH -A TG-ATM170020
#SBATCH -p h100
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o e3sm_inf_%j.o
#SBATCH -e e3sm_inf_%j.e



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

# Launch your script using torch.distributed.launch
CMD="torchrun --nproc_per_node=${NUM_TASKS_PER_NODE} /work/11095/jwan4/PanguWeather/v2.0/ensemble_inference.py \
    --run_num=0016 \
    --yaml_config=/work/11095/jwan4/PanguWeather/v2.0/config/E3SM_SFNO_H5_STAMPEDE_jsw.yaml \
    --config=SFNO \
    --init_datetime='2045-01-01 00:00:00' --init_datetimes='2045-01-01 00:00:00' \
    --init_nc_filepaths=/scratch/11095/jwan4/E3SM/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/sigma_data/2045_Combined_EAM_ELM.nc \
    --ensemble_inference_hours=336 --num_ensemble_members=1  --epsilon_factor=0.0 \
    --output_dir=/work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0016 \
    --save_basenames=/work/11095/jwan4/PanguWeather/v2.0/results/SFNO/0016/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101_2045_0016_001"

echo "Running: $CMD"
eval $CMD
