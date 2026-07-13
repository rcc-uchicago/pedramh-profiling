#!/bin/bash -l
#SBATCH --account=rcc-staff
#SBATCH --time=01:00:00
#SBATCH -p test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodelist=midway3-0602   # Intel Gold-6542Y, 1TB, H200 DLC
                                   # alternatives: midway3-0603/0604/0605/0606
#SBATCH -o midway_infer_nsys_h200_intel_%N_%j.out
#SBATCH -e midway_infer_nsys_h200_intel_%N_%j.err

# Nsight Systems inference profile on Midway H200 (Intel CPU) — bare metal,
# no container, same nsys flags as the DSI collection command so profiles
# are directly comparable with compare_nsys.py:
#
#   nsys profile -w true -t cuda,nvtx,cudnn --force-overwrite=true
#   torchrun --standalone --nproc_per_node=4
#   inference_optimized.py --yaml_config=exp2.yaml
#
# This gives us H200 Intel on Midway vs H200 on DSI, isolating hardware
# topology differences (NVLink, NUMA, PCIe) from software environment.
# Also compare against midway_infer_nsys.sh (H100) for GPU generation effect.

ulimit -l unlimited

module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv

module unload cuda
module load cuda/12.9

echo "nsys version: $(nsys --version 2>&1 | head -1)"

export WANDB_MODE=offline

echo "=== midway_infer_nsys H200 Intel: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODE=${SLURM_NODELIST}"
nvidia-smi -L
nvidia-smi topo -m

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "NUM_GPUS=${NUM_GPUS}"

config_file=/project/pedramh/shared/S2S/v2.0/config/exp2.yaml
NSYS_OUT="${SLURM_SUBMIT_DIR}/midway_h200_intel_4gpus_inference_${SLURM_JOB_ID}_$(date +%F)"

echo "nsys output: ${NSYS_OUT}.nsys-rep"

nsys profile \
    -w true \
    -t cuda,nvtx,cudnn \
    -o "${NSYS_OUT}" \
    --force-overwrite=true \
    torchrun \
        --standalone \
        --nproc_per_node="${NUM_GPUS}" \
        /project/pedramh/shared/S2S/v2.0/inference_optimized.py \
        --yaml_config="${config_file}" \
        --run_num="infer_nsys_h200_intel" \
        --async_save

echo "Profile written: ${NSYS_OUT}.nsys-rep"
echo "scp the .nsys-rep locally and run: nsys export --type=sqlite <file>.nsys-rep"
