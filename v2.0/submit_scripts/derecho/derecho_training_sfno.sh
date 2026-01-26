#!/bin/bash -l
#PBS -N inference
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -A UCHI0014
#PBS -e ./logs/SFNO_derecho_training_err.log
#PBS -o ./logs/SFNO_derecho_training_out.log

#echo $SLURM_NTASKS   # WORLD_SIZE
#echo $SLURM_PROCID   # WORLD_RANK
#echo $SLURM_LOCALID  # LOCAL_RANK

pwd 
mkdir -p logs

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export HDF5_USE_FILE_LOCKING=FALSE


#./home/awikner/miniconda3/bin/conda init; bash
module load conda
conda activate /glade/work/awikner/conda-envs/aires_panguplasim
#export cuda_version=12.1
#export CUDA_HOME=/usr/local/cuda-${cuda_version}
#export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
#export PATH=$CUDA_HOME/bin:$PATH

# source activate /scratch/midway3/tvallabh/tarun_pangu



# Change to working directory
cd /glade/work/marchakitus/PLASIM/PanguWeather/v2.0/

pwd 
mkdir -p logs

#source export_DDP_vars.sh
which conda
#python test_torch.py
# export WANDB_MODE=offline

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
	CMD="python train.py --yaml_config=${YAML_CONFIG} --run_num=${RUN_NUM} --debug --config=SFNO"
else
	CMD="python -m torch.distributed.run --nproc_per_node=${NUM_TASKS_PER_NODE} train.py --yaml_config=${YAML_CONFIG} --run_num=${RUN_NUM} --config=SFNO"
fi
# if [[ -z "$JOBID" ]]; then
# 	CMD+=" --fresh_start"
# fi
# Launch your script using torch.distributed.launch
eval "$CMD"