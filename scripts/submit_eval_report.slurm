#!/bin/bash
#SBATCH -J sfno_eval_report
#SBATCH -p skx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o logs/sfno_eval_report_%j.out
#SBATCH -e logs/sfno_eval_report_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# SFNO emulator report rendering (docs/sfno_eval_plan.md §G.2 row 3).
# Produces $OUT_ROOT/report.md from the scorecard CSV.
# ============================================================================

set -euo pipefail

REPO_ROOT="$HOME/AI-RES"

: "${RUN_DIR:=$SCRATCH/AI-RES/runs/sfno_full/plasim_sim52_full/0}"

if [[ -z "${RUN_TAG:-}" || -z "${EVAL_SHA7:-}" || -z "${DATA_SHA7:-}" || -z "${TRAIN_SHA7:-}" ]]; then
    echo "ERROR: RUN_TAG, EVAL_SHA7, DATA_SHA7, TRAIN_SHA7 must be set" >&2
    exit 2
fi

: "${OUT_ROOT:=$WORK2/AI-RES/results/sfno_eval/$RUN_TAG}"
: "${CKPT:=$RUN_DIR/training_checkpoints/best_ckpt_mp0.tar}"

source "$REPO_ROOT/.venv/bin/activate"

cd "$REPO_ROOT"
mkdir -p logs

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

set -x
python scripts/render_eval_report.py \
    --out-root "$OUT_ROOT" \
    --run-tag "$RUN_TAG" \
    --eval-sha7 "$EVAL_SHA7" \
    --data-sha7 "$DATA_SHA7" \
    --train-sha7 "$TRAIN_SHA7" \
    --ckpt-path "$CKPT"
