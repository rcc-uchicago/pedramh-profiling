#!/bin/bash

#SBATCH --account=bdiu-dtai-gh
#SBATCH --job-name=nsight_ace2_train_test
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --time=06:00:00
#SBATCH --output=/project/pedramh/shared/ACE2_retrain/seed1_training_%j.log
#SBATCH --error=/project/pedramh/shared/ACE2_retrain/errors_%j.err
#SBATCH --qos=bdiu-dtai-gh


cd /project/pedramh/shared/ACE2_retrain

# export WANDB_API_KEY=$(cat /u/krucker/.wandb_api_key)
# export WANDB_RUN_NAME="test_nsight__${SLURM_JOB_ID}"
# export WANDB_MODE=online
# export OMP_NUM_THREADS=64 # was 18

# --- Resolve the head node for torchrun rendezvous ------------------------
nodes=( $( scontrol show hostnames ${SLURM_JOB_NODELIST} ) )
head_node="${nodes[0]}"
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "${head_node}" hostname -I | awk '{print $1}')
echo "Head node: ${head_node} (${head_node_ip})"

# --- Load Pytorch environment ---------------------------------------------
module load python/miniforge3_pytorch
source activate /scratch/midway3/krucker01/envs/fme

# --- fail fast from here on ------------------------------------------------
# NOTE: deliberately NOT before this point, and deliberately without -u.
# conda's activate.d scripts are not `set -u` clean: the env carries
# activate-gcc_linux-64.sh (from the Makefile's gxx_linux-64 toolchain),
# whose line 194 dereferences ${CONDA_BUILD_SYSROOT} with no default and
# nothing ever sets it -> "unbound variable" kills the job at activation.
set -eo pipefail

# --- NCCL environment (Slingshot 11) --------------------------------------
export NCCL_DEBUG=INFO           # set to WARN for less verbose output
export NCCL_SOCKET_IFNAME=hsn0
export LOGLEVEL=INFO
export NCCL_CROSS_NIC=1
export GLOO_SOCKET_IFNAME=hsn0
export MASTER_ADDR=${head_node_ip}
export MASTER_PORT=29500
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

module list
echo "Job is strating on $(hostname)"

# ---- Data transfer first -------------------------------------------------
#bash /work/nvme/bdiu/krucker/ace_base_run1/nsight/test_tmp_transfer.sh /work/nvme/bdiu/jlandsberg/ace_training/merged_ACE2_ERA5_final.nc

# --- Launch DDP training --------------------------------------------------
time srun python -m torch.distributed.run \
    --nnodes ${SLURM_NNODES} \
    --nproc_per_node ${SLURM_GPUS_PER_NODE} \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint="${head_node_ip}:29500" \
    -m fme.ace.train /project/pedramh/shared/ACE2_retrain/config_nsight.yaml
