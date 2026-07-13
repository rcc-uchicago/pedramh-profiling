#!/bin/bash -l
#SBATCH --account=rcc-staff
#SBATCH --time=01:00:00
#SBATCH -p test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodelist=midway3-0601   # AMD EPYC-9335, 768GB, H200 DLC
                                   # alternative: midway3-0600
#SBATCH -o midway_infer_nsys_h200_amd_%N_%j.out
#SBATCH -e midway_infer_nsys_h200_amd_%N_%j.err

# Nsight Systems inference profile on Midway H200 (AMD EPYC CPU) — bare metal,
# no container, same nsys flags as the DSI collection command.
#
# AMD EPYC-9335 (Genoa) uses a chiplet design with multiple NUMA nodes per
# socket. Compare the inter-kernel gap histogram from this profile against:
#   - midway_h200_intel_4gpus_inference.sqlite  (same GPU, different CPU)
#   - midway_h100_4gpus_inference.sqlite        (different GPU, Intel CPU)
#   - dsi_h200_4gpus_inference.sqlite           (original mystery profile)
# Any difference between Intel and AMD H200 profiles points to CPU/NUMA
# topology as the cause rather than the GPU hardware itself.

ulimit -l unlimited

module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv

module unload cuda
module load cuda/12.9

echo "nsys version: $(nsys --version 2>&1 | head -1)"

export WANDB_MODE=offline

echo "=== midway_infer_nsys H200 AMD EPYC: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODE=${SLURM_NODELIST}"
nvidia-smi -L
nvidia-smi topo -m

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "NUM_GPUS=${NUM_GPUS}"

config_file=/project/pedramh/shared/S2S/v2.0/config/exp2.yaml
NSYS_OUT="${SLURM_SUBMIT_DIR}/midway_h200_amd_4gpus_inference_${SLURM_JOB_ID}_$(date +%F)"

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
        --run_num="infer_nsys_h200_amd" \
        --async_save

echo "Profile written: ${NSYS_OUT}.nsys-rep"
echo "scp the .nsys-rep locally and run: nsys export --type=sqlite <file>.nsys-rep"
