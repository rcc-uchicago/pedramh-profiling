#!/bin/bash
#SBATCH -J sfno_eval_inf
#SBATCH -p h100                     # Stampede3 H100 partition (1 GPU)
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -o logs/sfno_eval_inf_%j.out
#SBATCH -e logs/sfno_eval_inf_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# SFNO emulator inference (docs/sfno_eval_plan.md §G.2 row 1).
# 96 NWP ICs × 56 steps + 8 climate × ~1456 steps = ~17,016 forward passes.
# Estimated runtime ~1.5 h on one H100; 6 h wallclock leaves ample margin.
#
# Required env vars:
#   RUN_DIR     — training run directory.
#                  Default: $SCRATCH/AI-RES/runs/sfno_full/plasim_sim52_full/0
#   CKPT        — checkpoint path.
#                  Default: $RUN_DIR/training_checkpoints/best_ckpt_mp0.tar
#   TEST_HOLDOUT — directory of test h5 files.
#                  Default: $SCRATCH/AI-RES/data/makani/sim52_full/test_holdout
#   OUT_ROOT    — eval output root.
#                  Default: $WORK2/AI-RES/results/sfno_eval/$RUN_TAG
#   RUN_TAG     — required; format
#                 ``YYYYMMDD_eval-{eval}_data-{data}_train-{train}_ckpt-{name}``
#   EVAL_SHA7, DATA_SHA7, TRAIN_SHA7 — provenance, baked into NetCDF attrs.
#   MODE        — 'nwp' (default) or 'climate'
# ============================================================================

set -euo pipefail

REPO_ROOT="$HOME/AI-RES"

# ---- defaults -----------------------------------------------------------
: "${RUN_DIR:=$SCRATCH/AI-RES/runs/sfno_full/plasim_sim52_full/0}"
: "${CKPT:=$RUN_DIR/training_checkpoints/best_ckpt_mp0.tar}"
: "${TEST_HOLDOUT:=$SCRATCH/AI-RES/data/makani/sim52_full/test_holdout}"
: "${MODE:=nwp}"

if [[ -z "${RUN_TAG:-}" ]]; then
    echo "ERROR: RUN_TAG must be set (see docs/sfno_eval_plan.md §G.4)" >&2
    exit 2
fi
if [[ -z "${EVAL_SHA7:-}" || -z "${DATA_SHA7:-}" || -z "${TRAIN_SHA7:-}" ]]; then
    echo "ERROR: EVAL_SHA7, DATA_SHA7, TRAIN_SHA7 must be set" >&2
    exit 2
fi

: "${OUT_ROOT:=$WORK2/AI-RES/results/sfno_eval/$RUN_TAG}"

# ---- venv + dist env ----------------------------------------------------
source "$REPO_ROOT/.venv/bin/activate"

# Single-rank distributed init (the eval driver builds Makani's
# DistributedManager which expects MASTER_ADDR/PORT to be available).
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"
export LOCAL_RANK="${LOCAL_RANK:-0}"

cd "$REPO_ROOT"
mkdir -p logs

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -c "import sfno_inference; print('sfno_inference import:', sfno_inference.__file__)"

set -x

# Build the test-holdout directory (idempotent symlink farm) if missing.
if [ ! -d "$TEST_HOLDOUT" ] || [ -z "$(ls -A "$TEST_HOLDOUT")" ]; then
    python scripts/build_test_split.py \
        --src "$SCRATCH/AI-RES/data/makani/sim52_astro_64x128/test" \
        --dst "$TEST_HOLDOUT" \
        --years 0121-0128
fi

# Optional 15-second sanity check (§3 P-4).
python scripts/trace_calendar_anchors.py \
    --src "$TEST_HOLDOUT" \
    --files MOST.0121.h5,MOST.0122.h5,MOST.0128.h5 \
    --csv "$OUT_ROOT/diagnostics/calendar_trace.csv"

# Main inference run. Optional debug flags: LIMIT_FILES / LIMIT_ICS env vars
# pass through to --limit-files / --limit-ics for smoke testing
# (e.g. LIMIT_FILES=1 LIMIT_ICS=1 ... sbatch submit_eval_inference.slurm).
LIMIT_ARGS=()
if [[ -n "${LIMIT_FILES:-}" ]]; then
    LIMIT_ARGS+=(--limit-files "$LIMIT_FILES")
fi
if [[ -n "${LIMIT_ICS:-}" ]]; then
    LIMIT_ARGS+=(--limit-ics "$LIMIT_ICS")
fi

python scripts/eval_inference.py \
    --run-dir "$RUN_DIR" \
    --ckpt "$CKPT" \
    --test-holdout "$TEST_HOLDOUT" \
    --out-root "$OUT_ROOT" \
    --mode "$MODE" \
    --eval-sha7 "$EVAL_SHA7" \
    --data-sha7 "$DATA_SHA7" \
    --train-sha7 "$TRAIN_SHA7" \
    --run-tag "$RUN_TAG" \
    "${LIMIT_ARGS[@]}"
