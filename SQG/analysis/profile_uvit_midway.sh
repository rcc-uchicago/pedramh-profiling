#!/bin/bash
# profile_uvit_midway.sh -- Job C: torch.profiler breakdown of the SQG UViT training step (random input, no data).
#   sbatch SQG/analysis/profile_uvit_midway.sh
# PASS = "UVIT_OK" in the .out. Output: SQG/analysis/results/profile_uvit_<jobid>/ (profile_uvit.json, uvit_ops.txt)
#SBATCH --job-name=sqg-prof-uvit
#SBATCH --account=pi-pedramh
#SBATCH --partition=pedramh-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=SQG/analysis/logs/prof_uvit_%j.out
#SBATCH --error=SQG/analysis/logs/prof_uvit_%j.err
set -eo pipefail

module load cuda/12.9
PY=/project/pedramh/weidong/miniconda3/envs/wdenv/bin/python   # torch 2.3.1, timm 0.9.12
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
OUT=SQG/analysis/results/profile_uvit_${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
echo "=== profile_uvit on $(hostname) job ${SLURM_JOB_ID:-local}: $(date '+%F %T') ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$OUT/gpu.csv"
"$PY" SQG/analysis/profile_uvit.py --out "$OUT"
echo "=== end $(date '+%F %T') ==="
