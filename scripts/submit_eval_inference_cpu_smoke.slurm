#!/bin/bash
#SBATCH -J sfno_eval_inf_cpu_smoke
#SBATCH -p skx-dev                  # Stampede3 develop CPU partition (max 2h)
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o logs/sfno_eval_inf_cpu_smoke_%j.out
#SBATCH -e logs/sfno_eval_inf_cpu_smoke_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# CPU smoke variant of submit_eval_inference.slurm.
# Forces --device cpu and runs LIMIT_FILES=1 LIMIT_ICS=1 to validate the
# rollout chain end-to-end without queueing for an H100.
#
# Use when the h100 queue is saturated and you only need to confirm orchestration.
# ============================================================================

set -euo pipefail

REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"

# ---- defaults -----------------------------------------------------------
: "${RUN_DIR:=$SCRATCH/SFNO_Climate_Emulator/runs/sfno_full/plasim_sim52_full/0}"
: "${CKPT:=$RUN_DIR/training_checkpoints/best_ckpt_mp0.tar}"
: "${TEST_HOLDOUT:=$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_full/test_holdout}"
: "${MODE:=nwp}"
: "${LIMIT_FILES:=1}"
: "${LIMIT_ICS:=1}"

if [[ -z "${RUN_TAG:-}" ]]; then
    echo "ERROR: RUN_TAG must be set (see docs/sfno_eval_plan.md §G.4)" >&2
    exit 2
fi
if [[ -z "${EVAL_SHA7:-}" || -z "${DATA_SHA7:-}" || -z "${TRAIN_SHA7:-}" ]]; then
    echo "ERROR: EVAL_SHA7, DATA_SHA7, TRAIN_SHA7 must be set" >&2
    exit 2
fi

: "${OUT_ROOT:=$WORK2/SFNO_Climate_Emulator/results/sfno_eval/$RUN_TAG}"
mkdir -p "$OUT_ROOT/diagnostics"

# ---- venv + dist env ----------------------------------------------------
source "$REPO_ROOT/.venv/bin/activate"

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

if [ ! -d "$TEST_HOLDOUT" ] || [ -z "$(ls -A "$TEST_HOLDOUT")" ]; then
    python scripts/build_test_split.py \
        --src "$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128/test" \
        --dst "$TEST_HOLDOUT" \
        --years 0121-0128
fi

python scripts/trace_calendar_anchors.py \
    --src "$TEST_HOLDOUT" \
    --files MOST.0121.h5,MOST.0122.h5,MOST.0128.h5 \
    --csv "$OUT_ROOT/diagnostics/calendar_trace.csv"

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
    --device cpu \
    --limit-files "$LIMIT_FILES" \
    --limit-ics "$LIMIT_ICS"
