#!/bin/bash -l

#SBATCH --time=01:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    # one launcher; torchrun spawns 4 GPU workers internally
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32     # 4 GPUs x 8 CPUs = 32 total
#SBATCH -o nv_nsys_ddp_%x_%j.out
#SBATCH -e nv_nsys_ddp_%x_%j.err

# Enable GPU support for MPI
export MPICH_GPU_SUPPORT_ENABLED=1

ulimit -l unlimited

export WANDB_MODE=offline

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

# Load apptainer
module load apptainer

echo nvidia-smi
nvidia-smi

# Get GPU count from host before entering container
export NUM_GPUS=$(nvidia-smi -L | wc -l)

echo "NUM_OF_NODES= ${SLURM_JOB_NUM_NODES} NUM_GPUS= ${NUM_GPUS} JOB_ID= ${SLURM_JOB_ID}"

# Configuration
CONFIG_FILE=../config/exp2.yaml


# NGC credentials — set before pulling (requires NGC API key)
export APPTAINER_DOCKER_USERNAME='$oauthtoken'
export APPTAINER_DOCKER_PASSWORD="${NGC_API_KEY:?Set NGC_API_KEY to your NGC API key (https://ngc.nvidia.com) before submitting}"

echo "--- STARTING NSYS PROFILING RUN IN NGC CONTAINER (nvcr.io/nvidia/pytorch:26.01-py3) ---"

# apptainer exec --nv pytorch_25.10.sif which nsys
apptainer exec \
    --nv \
    --bind /lustre/fs01 \
    /home/ucg-aepmn/uchigaco/pytorch_25.10.sif \
    bash -c "
        pip uninstall netCDF4 -y -q 2>/dev/null; pip install ruamel.yaml ruamel.base wandb xarray cartopy h5py h5netcdf timm cftime dask seaborn cdsapi onnx -q --user &&
        PYTHONPATH=/home/ucg-aepmn/uchigaco/S2S/v2.0 \
        nsys profile \
            -w true \
            -t cuda,nvtx,cudnn \
            -o /home/ucg-aepmn/uchigaco/S2S/v2.0/HPC_scripts/nsys_report_%q{SLURM_JOB_ID} \
            --force-overwrite=true \
            torchrun \
            --standalone \
            --nproc_per_node=${NUM_GPUS} \
            /home/ucg-aepmn/uchigaco/S2S/v2.0/train_optimized.py \
            --yaml_config=${CONFIG_FILE} \
            --run_num=01_nsys
"
 #cublas 