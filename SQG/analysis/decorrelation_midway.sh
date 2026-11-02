#!/bin/bash
# decorrelation_midway.sh -- CPU job for decorrelation.py (read-only analysis of the stored SQG trajectories).
#   sbatch SQG/analysis/decorrelation_midway.sh            # full run
#   sbatch SQG/analysis/decorrelation_midway.sh --smoke    # 2 windows, a few lags
# PASS = "DECORR_OK" in the .out, plus results/decorrelation_<tag>.csv and _summary.json.
#SBATCH --job-name=sqg-decorr
#SBATCH --account=pi-pedramh
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=SQG/analysis/logs/decorr_%j.out
#SBATCH --error=SQG/analysis/logs/decorr_%j.err
set -eo pipefail

PY=/project/pedramh/shared/S2S/v2.0/venv/bin/python     # has h5py + numpy; no GPU needed
export OMP_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "=== decorrelation on $(hostname) job ${SLURM_JOB_ID:-local}: $(date '+%F %T') ==="
"$PY" SQG/analysis/decorrelation.py --workers "${SLURM_CPUS_PER_TASK:-8}" "$@"
echo "=== end $(date '+%F %T') ==="
