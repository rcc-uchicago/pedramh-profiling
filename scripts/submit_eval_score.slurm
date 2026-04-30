#!/bin/bash
#SBATCH -J sfno_eval_score
#SBATCH -p skx                      # CPU node — climatology + scoring is I/O bound
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o logs/sfno_eval_score_%j.out
#SBATCH -e logs/sfno_eval_score_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# SFNO emulator scoring (docs/sfno_eval_plan.md §G.2 row 2).
# 1) Build climatology from sim52_full/train/ (~30-60 min, I/O bound).
# 2) Score NWP NetCDFs and write scorecard CSV + bias maps.
# 3) Apply the §D.6 sanity gate (non-zero exit on FAIL).
#
# Required env vars (defaults shown):
#   RUN_DIR, RUN_TAG (required), OUT_ROOT, TRAIN_DIR
# ============================================================================

set -euo pipefail

REPO_ROOT="$HOME/AI-RES"

: "${RUN_DIR:=$SCRATCH/AI-RES/runs/sfno_full/plasim_sim52_full/0}"
: "${TRAIN_DIR:=$SCRATCH/AI-RES/data/makani/sim52_full/train}"

if [[ -z "${RUN_TAG:-}" ]]; then
    echo "ERROR: RUN_TAG must be set" >&2
    exit 2
fi

: "${OUT_ROOT:=$WORK2/AI-RES/results/sfno_eval/$RUN_TAG}"

source "$REPO_ROOT/.venv/bin/activate"

cd "$REPO_ROOT"
mkdir -p logs

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

set -x

# 1) Climatology — only build if missing (idempotent).
CLIM_NC="$OUT_ROOT/baselines/climatology_proleptic.nc"
if [ ! -f "$CLIM_NC" ]; then
    python scripts/compute_climatology.py \
        --train-dir "$TRAIN_DIR" \
        --out "$CLIM_NC" \
        --source-files-out "$OUT_ROOT/diagnostics/climatology_source_files.json"
fi

# 2) NWP scoring — RMSE/ACC + bias maps + sanity gate.
python scripts/score_nwp.py \
    --out-root "$OUT_ROOT" \
    --clim-nc "$CLIM_NC"
