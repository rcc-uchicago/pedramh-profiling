#!/bin/bash -l
#PBS -N inference
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -A UCHI0014

# Enable GPU-MPI (if supported by application)
export MPICH_GPU_SUPPORT_ENABLED=1

ml conda
#conda activate py311_nompi
conda activate aires_panguplasim
#ml cudatoolkit-standalone/12.2.2

# cd $PBS_O_WORKDIR

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
#Follwing will be the number of GPUs on each node, so 4 in our case as each node has 4 GPUs
NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
#NUM_TASKS_PER_NODE=2
NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
#NUM_TASKS_PER_NODE=2
WORLD_SIZE=$((NNODES * NUM_TASKS_PER_NODE))

echo "NUM_OF_NODES= ${NNODES} NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE} WORLD_SIZE= ${WORLD_SIZE}"
which python
# Launch your script using torch.distributed.launch
python -m torch.distributed.launch --nproc_per_node=$NUM_TASKS_PER_NODE long_inference.py \
    --run_num=0515,0514 --yaml_config=config/PANGU_PLASIM_H5_DERECHO_0515_longtest_3.yaml,config/PANGU_PLASIM_H5_DERECHO_0514_longtest.yaml \
    --use_6h_24h_model \
    --init_datetime=$INIT_DATETIME --final_datetime=$FINAL_DATETIME --init_nc_filepaths=$INIT_NC_FILEPATHS\
    --run_iter=$RUN_ITER --output_dir=$OUTPUT_DIR
