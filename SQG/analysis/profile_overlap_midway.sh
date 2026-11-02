#!/bin/bash
# profile_overlap_midway.sh -- Job B: does the JAX solver overlap the PyTorch UViT step on one GPU?
#   sbatch SQG/analysis/profile_overlap_midway.sh
# Runs profile_overlap.py (ml_only, solver_only, serial = production order, background-thread solver; for the
# pyqg_jax AB3 and the fixed-level AB3). Needs one solver checkpoint, no training data.
# PASS = "OVERLAP_OK" in the .out. Output: SQG/analysis/results/profile_overlap_<jobid>/profile_overlap.json
#SBATCH --job-name=sqg-prof-overlap
#SBATCH --account=pi-pedramh
#SBATCH --partition=pedramh-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=SQG/analysis/logs/prof_overlap_%j.out
#SBATCH --error=SQG/analysis/logs/prof_overlap_%j.err
set -eo pipefail

module load cuda/12.9                                       # same environment as job A
PY=/project/pedramh/weidong/miniconda3/envs/wdenv/bin/python   # jax 0.4.28, torch 2.3.1, timm 0.9.12
export XLA_PYTHON_CLIENT_PREALLOCATE=false HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=8 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # as production (o1.2 launcher)

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
OUT=SQG/analysis/results/profile_overlap_${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
echo "=== profile_overlap on $(hostname) job ${SLURM_JOB_ID:-local}: $(date '+%F %T') ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv | tee "$OUT/gpu.csv"
nvidia-smi dmon -s pucvm -d 1 -i 0 > "$OUT/dmon.txt" 2>&1 &
DMON=$!
"$PY" SQG/analysis/profile_overlap.py --steps 20 --warmup 4 --out "$OUT"
kill "$DMON" 2>/dev/null || true
echo "=== end $(date '+%F %T') ==="
