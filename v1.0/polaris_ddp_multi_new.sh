#!/bin/bash -l
#PBS -N multi_node
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -q debug 
#PBS -l walltime=0:20:00
#PBS -l filesystems=home:eagle                          
#PBS -A lighthouse-uchicago
#PBS -e logs/
#PBS -o logs/

. /etc/profile

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
echo "Job started at: {$TSTAMP}"

# Enable GPU-MPI (if supported by application)
export MPICH_GPU_SUPPORT_ENABLED=1

#NCCL Settings
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Change to working directory
cd $PBS_O_WORKDIR

echo "Job ID: ${PBS_JOBID}"
export PLASIM_TRAIN_ITER=$PLASIM_TRAIN_ITER+1
echo "PLASIM Emulator training epoch: ${PLASIM_TRAIN_ITER}"

# Figure out training environment
if [[ -z "${PBS_NODEFILE}" ]]; then
    RANKS=$HOSTNAME
    NNODES=1
else
    MASTER_RANK=$(head -n 1 $PBS_NODEFILE)
    RANKS=$(tr '\n' ' ' < $PBS_NODEFILE)
    NNODES=$(< $PBS_NODEFILE wc -l)
fi

NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
WORLD_SIZE=$((NNODES * NUM_TASKS_PER_NODE))

echo "NUM_OF_NODES= ${NNODES}"

# Commands to run prior to the Python script for setting up the environment
PRELOAD="source /etc/profile ; "
PRELOAD+="module use /soft/modulefiles;"
PRELOAD+="module load conda;"
PRELOAD+="conda activate /eagle/MDClimSim/hyadav/Pangu_env_2;"
PRELOAD+="module load cudatoolkit-standalone/12.2.2;"
PRELOAD+="export NODES=1;"
PRELOAD+="export MASTER_ADDR=$(hostname);"
PRELOAD+="export MASTER_PORT=12345;"
PRELOAD+="export WORLD_SIZE; "
# PRELOAD+="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;"

# time python process to ensure timely job exit
TIMER="timeout 718m "

# torchrun launch configuration
LAUNCHER="python3 -m torch.distributed.run "
LAUNCHER+="--nnodes=$NNODES --nproc_per_node=auto --max_restarts 0 "
if [[ "$NNODES" -eq 1 ]]; then
    LAUNCHER+="--standalone "
else
    LAUNCHER+="--rdzv_backend=c10d --rdzv_endpoint=$MASTER_RANK "
fi

CMD="train.py --yaml_config=/eagle/lighthouse-uchicago/members/hyadav/PanguWeather/v1.0/config/PANGU_24HR.yaml"

FULL_CMD=" $PRELOAD $TIMER $LAUNCHER $CMD $@ "
echo "Training Command: $FULL_CMD"

# Launch the pytorch processes on each worker (use ssh for remote nodes)
RANK=0
for NODE in $RANKS; do #${RANKS[*]:0:21}; do #$RANKS; do
    if [[ "$NODE" == "$HOSTNAME" ]]; then
        echo "Launching rank $RANK on local node $NODE"
        eval $FULL_CMD &
    else
        echo "Launching rank $RANK on remote node $NODE"
        ssh $NODE "cd $PWD; $FULL_CMD" &
    fi
    RANK=$((RANK+1))
done

wait