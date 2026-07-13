#!/bin/bash -l
#SBATCH --account=rcc-staff
#SBATCH --time=00:45:00
#SBATCH -p test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodelist=midway3-0601   # AMD EPYC-9335, H200 DLC
#SBATCH -o midway_dispatch_real_amd_%N.out
#SBATCH -e midway_dispatch_real_amd_%N.err

# Real-PanguModel dispatch gap test on Midway H200 (AMD CPU).
# See midway_dispatch_real_intel.sh for what this test answers and why.

ulimit -l unlimited

module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv

module unload cuda
module load cuda/12.9

echo "=== midway_dispatch_real AMD: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODE=${SLURM_NODELIST}"
nvidia-smi -L
echo
echo "--- nvidia-smi topo -m ---"
nvidia-smi topo -m
echo
echo "--- numactl --hardware ---"
numactl --hardware
echo
echo "--- cpu / power state (dispatch-latency suspects) ---"
cpupower frequency-info 2>/dev/null | head -20 || true
cpupower idle-info     2>/dev/null | head -20 || true
echo
echo "--- nvidia IRQ steering ---"
grep -E "nvidia" /proc/interrupts 2>/dev/null | head -8 || true
echo
echo "--- driver / cuda versions ---"
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}')"
echo

SCRIPT=/project/pedramh/shared/S2S/v2.0/test/inference_dispatch_real.py
CONFIG=/project/pedramh/shared/S2S/v2.0/config/exp2.yaml

############################################
# Phase 1: single GPU
############################################
echo "================================================================"
echo "Phase 1: single-GPU dispatch test (CUDA_VISIBLE_DEVICES=0)"
echo "================================================================"
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT}" \
    --yaml_config="${CONFIG}" \
    --steps=60 --warmup=3 --reps=4

############################################
# Phase 2: 4 GPUs via torchrun
############################################
echo
echo "================================================================"
echo "Phase 2: 4-GPU dispatch test (torchrun --nproc_per_node=4)"
echo "================================================================"
torchrun --standalone --nproc_per_node=4 "${SCRIPT}" \
    --yaml_config="${CONFIG}" \
    --steps=60 --warmup=3 --reps=4

echo
echo "=== done: $(date -Iseconds) ==="
