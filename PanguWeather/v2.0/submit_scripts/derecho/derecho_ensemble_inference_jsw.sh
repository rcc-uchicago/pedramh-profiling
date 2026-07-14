#!/bin/bash -l
#PBS -N sfno_e3sm_inf
#PBS -l select=1:ncpus=32:ngpus=1
#PBS -q develop
#PBS -l walltime=01:00:00
#PBS -A UCHI0018


# Enable GPU-MPI (if supported by application)
export MPICH_GPU_SUPPORT_ENABLED=1
ml conda
conda activate sfno_pangu
#conda activate py311_nompi
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
python -m torch.distributed.launch --nproc_per_node=$NUM_TASKS_PER_NODE /glade/work/jesswan/UChicago/PanguWeather/v2.0/ensemble_inference.py \
    --run_num=0008 \
    --yaml_config=/glade/work/jesswan/UChicago/PanguWeather/v2.0/config/E3SM_SFNO_H5_DERECHO_jsw.yaml \
    --config=SFNO \
    --init_datetime="2045-01-01 00:00:00" --init_datetimes="2045-01-01 00:00:00" \
    --init_nc_filepaths=/glade/derecho/scratch/jesswan/E3SM/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101/sigma_data/2045_Combined_EAM_ELM.nc \
    --ensemble_inference_hours=336 --num_ensemble_members=1  --epsilon_factor=0.0 \
    --output_dir=/glade/work/jesswan/UChicago/PanguWeather/v2.0/results/SFNO/0008 \
    --save_basenames=/glade/work/jesswan/UChicago/PanguWeather/v2.0/results/SFNO/0008/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101_2045_0008_002 

