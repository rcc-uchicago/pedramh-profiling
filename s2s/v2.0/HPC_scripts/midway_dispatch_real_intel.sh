#!/bin/bash -l
#SBATCH --account=rcc-staff
#SBATCH --time=00:45:00
#SBATCH -p test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodelist=midway3-0602   # Intel Gold-6542Y, 1TB, H200 DLC
#SBATCH -o midway_dispatch_real_intel_%N.out
#SBATCH -e midway_dispatch_real_intel_%N.err

# Real-PanguModel dispatch gap test on Midway H200 (Intel CPU).
#
# Uses torch.cuda.Event for in-process gap measurement — no profiler,
# no ptrace, works on the test partition.
#
# Runs two phases back-to-back on the same allocation:
#   Phase 1: single GPU  → directly comparable to DSI 1-GPU (41 gaps of 10-50ms baseline)
#   Phase 2: 4 GPUs via torchrun → directly comparable to DSI 4-GPU (423 gaps on GPU0)
#
# If Phase 2 shows a similar 10x gap explosion, DSI's pattern reproduces on
# Midway H200 — cause is H200-related. If Phase 2 stays clean, DSI is doing
# something the test partition isn't (NUMA, C-states, IRQ steering, driver).

ulimit -l unlimited

module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv

module unload cuda
module load cuda/12.9

echo "=== midway_dispatch_real Intel: $(date -Iseconds) ==="
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
cat /sys/module/intel_idle/parameters/max_cstate 2>/dev/null || true
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
