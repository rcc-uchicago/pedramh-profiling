#!/bin/bash
# eval_run_score_inline.sh — eval-score stage body, sourceable.
# Used by scripts/submit_eval_score.slurm (standalone) and bundled_eval.sh.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/projects/SFNO_Climate_Emulator}"

: "${RUN_DIR:=$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_full/plasim_sim52_zgplev_full/0}"
: "${TRAIN_DIR:=$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full/train}"

if [[ -z "${RUN_TAG:-}" ]]; then
    echo "ERROR: RUN_TAG must be set" >&2
    exit 2
fi

: "${OUT_ROOT:=$WORK2/SFNO_Climate_Emulator/results/sfno_eval/$RUN_TAG}"
mkdir -p "$OUT_ROOT/baselines" "$OUT_ROOT/diagnostics" "$OUT_ROOT/scores"

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
