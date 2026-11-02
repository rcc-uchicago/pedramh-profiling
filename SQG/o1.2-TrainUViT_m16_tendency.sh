#!/bin/bash
# o1.2-TrainUViT_m16_tendency.sh -- DDP launcher for o1.2-TrainUViT_m16_tendency.py (ONLINE twin of 1-TrainUViT_m16_tendency, design 3: all 3500 checkpoints x 1.0, S=480, 2x host buffer, resumable, same 195k optimizer steps as offline).
# All hyper-parameters live in the .py; this file only picks the GPUs and the environment.
#
#   nohup bash o1.2-TrainUViT_m16_tendency.sh > logs/uvit_m16_tendency_online3_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -eo pipefail

SCRIPT=o1.2-TrainUViT_m16_tendency.py
GPUS=1,2,3,4,5,6,7
NPROC=7

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p logs

source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate /data/weidong/wdenv

export CUDA_VISIBLE_DEVICES=$GPUS
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
# JAX solver and PyTorch share each GPU: JAX must not preallocate 75% of it; JIT cache shared with the other solver runs
export XLA_PYTHON_CLIENT_PREALLOCATE=false HDF5_USE_FILE_LOCKING=FALSE JAX_COMPILATION_CACHE_DIR=/data/weidong/SQG_data/jax_cache
# wandb key of beauwdli (0600 file)
[ -f /data/weidong/.wandb_key ] && export WANDB_API_KEY="$(cat /data/weidong/.wandb_key)"
# hostname resolves to a public IP on this box: rendezvous and NCCL on loopback
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo

echo "=== $SCRIPT on GPUs $GPUS: start $(date '+%F %T') ==="
torchrun --nnodes=1 --nproc_per_node=$NPROC --master_addr=127.0.0.1 --master_port=$((49500 + RANDOM % 2000)) \
    --max_restarts=0 "$HERE/$SCRIPT"
echo "=== end $(date '+%F %T') ==="
