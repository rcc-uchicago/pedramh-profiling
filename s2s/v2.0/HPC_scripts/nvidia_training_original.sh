#!/bin/bash -l

#SBATCH --time=01:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8 
#SBATCH -o midway_ddp_%x_%j.out
#SBATCH -e midway_ddp_%x_%j.err

# Enable GPU support for MPI
export MPICH_GPU_SUPPORT_ENABLED=1

ulimit -l unlimited
ml python

source activate /home/ucg-aepmn/uchigaco
export WANDB_MODE=offline
module unload cuda
module load cuda/12.6

# NCCL optimizations for H100
export NCCL_DEBUG=INFO  # Set to WARN in production
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_NET_GDR_LEVEL=5  # GPU Direct RDMA
export NCCL_P2P_LEVEL=5  # Enable P2P
export NCCL_SOCKET_IFNAME=^lo,docker0  # Exclude loopback

# PyTorch NCCL settings
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Remove in production

# CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1


echo nvidia-smi
nvidia-smi

# Get GPU count
export NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)

echo "NUM_OF_NODES= ${SLURM_JOB_NUM_NODES} NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE} WORLD_SIZE= ${SLURM_NTASKS}"

# Configuration
config_file=../config/exp2.yaml


echo "--- STARTING NSYS PROFILING RUN (TARGET: EPOCH 1) ---"
    torchrun \
    --standalone \
    --nproc_per_node=$NUM_TASKS_PER_NODE \
    ../train.py \
    --yaml_config=$config_file \
    --run_num=0105_nsys \
#   --amp-dtype bf16 \
#   --torch-compile \
#   --compile-mode reduce-overhead \
#   --enable-sdp-flash \
#   --ddp-static-graph \
#   --ddp-bucket-cap-mb 200 \
#   --ddp-fp16-compress \
#   --log-every-n-steps 20 \
#   --metrics-every 100 \
#   --fp32-matmul-precision high \

# --- END MODIFICATION
