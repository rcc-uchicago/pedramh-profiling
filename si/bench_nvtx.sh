#!/bin/bash
#SBATCH --account=pi-pedramh
#SBATCH --time=00:30:00
#SBATCH -p pedramh-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --mem=500G
#SBATCH -o si_nvtx_%x_%j.out
#SBATCH -e si_nvtx_%x_%j.err

# NVTX/nsys profiling run for the SI bench.
#
# Mirrors bench_midway.sh (same data-loading config, same SLURM shape) but
# wraps srun with `nsys profile --capture-range=cudaProfilerApi`.  The
# BenchCallback calls cudaProfilerStart at the first measured step and
# cudaProfilerStop after the last, so only the measurement window lands in
# the trace.
#
# Measurement window is shortened to 20 steps (vs 80 in bench_midway.sh)
# because each measured step contributes to the .nsys-rep file size.  20 is
# plenty for spotting the hotspots; bump it later if a long-tail effect is
# suspected.
#
# After the job, transfer the per-rank .nsys-rep files locally and run:
#   nsys export --type=sqlite si_nvtx_<job>_<rank>.nsys-rep
# then query the SQLite per CLAUDE.md ("nsys analysis" section).

module load python/miniforge-25.3.0

eval "$(conda shell.bash hook)"
conda activate /project/pedramh/shared/anthonyz/venv

module unload cuda
module load cuda/12.6

unset NCCL_DEBUG
unset TORCH_DISTRIBUTED_DEBUG

export WANDB_MODE=offline
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Same OMP setup as bench_midway.sh — dataloader workers + main process win cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- Bench knobs for profiling ---
export SI_NVTX=1
export SI_BENCH_WARMUP=20
export SI_BENCH_STEPS=20

# Match the precision used by the production bench so the profile reflects it.
export SI_PRECISION=bf16-mixed

# Keep CSV output separate from production bench rows; the profile job is
# not for benchmark numbers, but we still get one row appended as a sanity
# check.
export SI_BENCH_CSV="${SLURM_SUBMIT_DIR}/bench_nvtx_results.csv"

config_file=configs/bench_midway.yaml

echo "=== si_nvtx: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODELIST=${SLURM_NODELIST}"
nvidia-smi -L
which nsys
nsys --version

echo "config=${config_file}  csv=${SI_BENCH_CSV}"
echo "warmup=${SI_BENCH_WARMUP}  steps=${SI_BENCH_STEPS}"

cd "${SLURM_SUBMIT_DIR}"

# Per-rank .nsys-rep so we can compare load balance across GPUs.
# --trace=cuda,nvtx,osrt: kernel timeline + NVTX ranges + OS runtime
# --capture-range=cudaProfilerApi: only capture between cudaProfilerStart/Stop
# --cuda-memory-usage=true: track CUDA allocations (small overhead, very useful)
# --force-overwrite=true: tolerate re-runs with the same job ID
srun --export=ALL nsys profile \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop-shutdown \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --output="si_nvtx_${SLURM_JOB_ID}_rank%q{SLURM_PROCID}" \
    python bench.py --config "${config_file}"
