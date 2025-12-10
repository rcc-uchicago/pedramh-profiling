#!/bin/bash
#PBS -A UCHI0014
#PBS -N train_exp03
#PBS -q main
#PBS -l walltime=01:00:00 
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -e ucar_exp0804_workflow6_error.txt
#PBS -o ucar_exp0804_workflow6.out
#PBS -l gpu_type=a100
#export WORLD_SIZE=$((PBS_NUM_NODES * PBS_NUM_PPN))
#echo "Total tasks: $WORLD_SIZE"

# Use scratch for temporary files to avoid space limits in /tmp
export TMPDIR=/glade/derecho/scratch/$USER/tmp
mkdir -p $TMPDIR

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
echo "Job started at: {$TSTAMP}"
Sqstat -u $USER
echo nvidia-smi

cd /glade/work/bgong/PanguWeather2/PanguWeather/v2.0


export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1


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

PRELOAD="module load conda;"
PRELOAD+="conda activate  /glade/work/zand/anaconda/py311;"
PRELOAD+="export NODES=1;"
PRELOAD+="export MASTER_ADDR=$(hostname);"
PRELOAD+="export MASTER_PORT=12345;"
PRELOAD+="export WORLD_SIZE; "

# time python process to ensure timely job exit
TIMER="timeout 718m "

# torchrun launch configuration
LAUNCHER="python -m torch.distributed.run "
LAUNCHER+="--nnodes=$NNODES --nproc_per_node=$NUM_TASKS_PER_NODE --max_restarts 0 "
if [[ "$NNODES" -eq 1 ]]; then
    LAUNCHER+="--standalone "
else
    LAUNCHER+="--rdzv_backend=c10d --rdzv_endpoint=$MASTER_RANK "
fi


CMD="train.py --yaml_config=./config/exp3.yaml --run_num=3_workflow"

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

