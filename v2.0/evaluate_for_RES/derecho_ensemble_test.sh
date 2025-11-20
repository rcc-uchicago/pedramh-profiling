#!/bin/bash -l
#PBS -N inference
#PBS -l select=1:ncpus=16:ngpus=1:mem=60G
#PBS -q main
#PBS -l walltime=0:30:00
#PBS -A UCHI0014
#PBS -o /glade/u/home/aasche/PanguWeather/v2.0/logs
#PBS -j oe
# PBS -J 1-127

# Enable GPU-MPI (if supported by application)
export MPICH_GPU_SUPPORT_ENABLED=1

# DIR PANGUPLASIM

ml conda
conda activate aires

DIR_PANGUPLASIM="/glade/work/alancelin/PanguWeather/v2.0"
cd $DIR_PANGUPLASIM
mkdir -p logs # Optional: safer to make logs directory

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
#Follwing will be the number of GPUs on each node, so 4 in our case as each node has 4 GPUs
NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
WORLD_SIZE=$((NNODES * NUM_TASKS_PER_NODE))

echo "NUM_OF_NODES= ${NNODES} NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE} WORLD_SIZE= ${WORLD_SIZE}"

CONFIG="/glade/work/alancelin/PanguWeather/v2.0/evaluate_for_RES/configs/forecasts/PANGU_PLASIM_H5_DERECHO_0515_evalRES.yaml"
CONFIG_NAME="PLASIM"
RUN_NUM="0515"

# Read the INIT_DATETIME corresponding to this array job index
# INIT_DATETIME=$(sed -n "${PBS_ARRAY_INDEX}p" /glade/u/home/aasche/PanguWeather/v2.0/init_datetimes.txt)
# INIT_DATETIME="0011-01-06_00:00:00"
# INIT_NC_FILEPATHS="/glade/derecho/scratch/awikner/PLASIM/data/train_val_test_data_res/data_11.nc"
OUTPUT_DIR="/glade/derecho/scratch/alancelin/PLASIM/PanguPlasim/evaluate_for_RES/test2/"

# echo "PBS_ARRAY_INDEX is $PBS_ARRAY_INDEX"

# Launch your script using torch.distributed.launch
torchrun --nproc_per_node=$NUM_TASKS_PER_NODE ensemble_inference.py --yaml_config=$CONFIG --config=$CONFIG_NAME --run_num=$RUN_NUM --output_dir=$OUTPUT_DIR #--async_save # --init_datetime=$INIT_DATETIME # --init_nc_filepaths=$INIT_NC_FILEPATHS 
# --enable_amp if needed
# --enable_amp if needed
