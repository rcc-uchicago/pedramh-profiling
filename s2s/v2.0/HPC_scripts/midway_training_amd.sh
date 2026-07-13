#!/bin/bash -l
#SBATCH --account=rcc-staff
#SBATCH --partition=test
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodelist=midway3-0601   # AMD EPYC-9335, H200 DLC
#SBATCH -o midway_training_amd_%N.out
#SBATCH -e midway_training_amd_%N.err

# Training benchmark on Midway H200 (AMD EPYC-9335, midway3-0601).
#
# Runs train.py directly (no nsys — test partition ptrace restriction blocks
# kernel tracing). Benchmark instrumentation via S2S_BENCH=1 env-var framework:
#   - 20 warmup steps (discarded)
#   - 80 timed steps -> CSV at ${SLURM_SUBMIT_DIR}/midway_training_amd_bench.csv
#
# Baseline to beat: H100 median 0.639 s/step, p90 0.643 s, std 0.005 s.
# AMP dtype: bf16 (matches bench_report.md Ablation 2 "best confirmed").

ulimit -l unlimited

module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv

module unload cuda
module load cuda/12.9

export WANDB_MODE=offline

echo "=== midway_training_amd: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODE=${SLURM_NODELIST}"

echo "--- nvidia-smi -L ---"
nvidia-smi -L
echo

echo "--- nvidia-smi topo -m ---"
nvidia-smi topo -m
echo

echo "--- numactl --hardware ---"
numactl --hardware
echo

echo "--- gpu name / vbios ---"
nvidia-smi --query-gpu=name,vbios_version --format=csv
echo

echo "--- cpu / power state (dispatch-latency suspects) ---"
cpupower frequency-info 2>/dev/null | head -20 || true
cpupower idle-info     2>/dev/null | head -20 || true
echo

echo "--- nvidia IRQ steering ---"
grep nvidia /proc/interrupts 2>/dev/null | head -8 || true
echo

echo "--- driver / torch versions ---"
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}')"
echo

SCRIPT=/project/pedramh/shared/S2S/v2.0/train.py
CONFIG=/project/pedramh/shared/S2S/v2.0/config/exp2.yaml
CSV_OUT="${SLURM_SUBMIT_DIR}/midway_training_amd_bench.csv"

echo "================================================================"
echo "Training benchmark: 4 GPUs, bf16 AMP, 20 warmup + 80 timed steps"
echo "CSV -> ${CSV_OUT}"
echo "================================================================"

S2S_BENCH=1 \
S2S_BENCH_WARMUP=20 \
S2S_BENCH_STEPS=80 \
S2S_BENCH_CSV="${CSV_OUT}" \
S2S_AMP_DTYPE=bf16 \
torchrun --standalone --nproc_per_node=4 \
    "${SCRIPT}" \
    --yaml_config="${CONFIG}" \
    --run_num=train_h200_amd_$(date +%s)

echo
echo "=== done: $(date -Iseconds) ==="
