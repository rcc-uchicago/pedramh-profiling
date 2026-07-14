#!/bin/bash
# eval_run_inference_inline.sh — eval-inference stage body, sourceable.
# Used by scripts/submit_eval_inference.slurm (standalone path) and by
# src/sfno_training/bundled_eval.sh (bundled-in-training path).
#
# Expects: RUN_TAG, EVAL_SHA7, DATA_SHA7, TRAIN_SHA7 set by the caller
# (the prelude function does this). RUN_DIR, CKPT, MODE, TEST_HOLDOUT,
# PACKAGER_TEST_SRC, OUT_ROOT default to v10 zgplev when missing.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/projects/SFNO_Climate_Emulator}"

: "${RUN_DIR:=$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_full/plasim_sim52_zgplev_full/0}"
: "${CKPT:=$RUN_DIR/training_checkpoints/best_ckpt_mp0.tar}"
: "${TEST_HOLDOUT:=$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full/test_holdout}"
: "${PACKAGER_TEST_SRC:=$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev/test}"
: "${MODE:=nwp}"

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

source "$REPO_ROOT/.venv/bin/activate"

# Single-rank distributed init (the eval driver builds Makani's
# DistributedManager which expects MASTER_ADDR/PORT to be available).
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"
export LOCAL_RANK="${LOCAL_RANK:-0}"
# Pin to a single GPU. On multi-GPU nodes (e.g. Stampede3 amd-rtx, which has
# no GRES config so the partition does not restrict CUDA_VISIBLE_DEVICES for
# us), NCCL otherwise probes every GPU on the host during init and crashes
# if any is bad (`nvmlDeviceGetHandleByIndex(N) failed`). WORLD_SIZE=1 only
# needs one GPU; this honors any pre-set value (e.g. when slurm did set it).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$REPO_ROOT"
mkdir -p logs

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -c "import sfno_inference; print('sfno_inference import:', sfno_inference.__file__)"

set -x

# Build the test-holdout directory (idempotent symlink farm) if missing.
if [ ! -d "$TEST_HOLDOUT" ] || [ -z "$(ls -A "$TEST_HOLDOUT")" ]; then
    python scripts/build_test_split.py \
        --src "$PACKAGER_TEST_SRC" \
        --dst "$TEST_HOLDOUT" \
        --years 0121,0122,0123,0124,0125,0126,0127,0128
fi

# Optional 15-second sanity check (§3 P-4).
python scripts/trace_calendar_anchors.py \
    --src "$TEST_HOLDOUT" \
    --files MOST.0121.h5,MOST.0122.h5,MOST.0128.h5 \
    --csv "$OUT_ROOT/diagnostics/calendar_trace.csv"

# Main inference run.
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
