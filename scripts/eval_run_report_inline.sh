#!/bin/bash
# eval_run_report_inline.sh — eval-report stage body, sourceable.
# Used by scripts/submit_eval_report.slurm (standalone) and bundled_eval.sh.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/projects/SFNO_Climate_Emulator}"

: "${RUN_DIR:=$SCRATCH/SFNO_Climate_Emulator/runs/sfno_full/plasim_sim52_full/0}"

if [[ -z "${RUN_TAG:-}" || -z "${EVAL_SHA7:-}" || -z "${DATA_SHA7:-}" || -z "${TRAIN_SHA7:-}" ]]; then
    echo "ERROR: RUN_TAG, EVAL_SHA7, DATA_SHA7, TRAIN_SHA7 must be set" >&2
    exit 2
fi

: "${OUT_ROOT:=$WORK2/SFNO_Climate_Emulator/results/sfno_eval/$RUN_TAG}"
: "${CKPT:=$RUN_DIR/training_checkpoints/best_ckpt_mp0.tar}"
: "${BENCHMARK_5410_OUT_ROOT:=/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval_5410/20260509_blocking_96ic_h100_packed_derecho_env_valid}"
: "${TRACK:=own}"
: "${PR6H_UNIT_ALIGN:=suppress}"

source "$REPO_ROOT/.venv/bin/activate"

cd "$REPO_ROOT"
mkdir -p logs

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

REPORT_ARGS=(
    --out-root "$OUT_ROOT"
    --run-tag "$RUN_TAG"
    --eval-sha7 "$EVAL_SHA7"
    --data-sha7 "$DATA_SHA7"
    --train-sha7 "$TRAIN_SHA7"
    --ckpt-path "$CKPT"
    --track "$TRACK"
    --pr6h-unit-align "$PR6H_UNIT_ALIGN"
)
# Empty BENCHMARK_5410_OUT_ROOT disables the overlay entirely.
if [[ -n "${BENCHMARK_5410_OUT_ROOT:-}" ]]; then
    REPORT_ARGS+=( --benchmark-5410-out-root "$BENCHMARK_5410_OUT_ROOT" )
fi
if [[ -n "${RUN_DIR:-}" ]]; then
    REPORT_ARGS+=( --run-dir "$RUN_DIR" )
fi

set -x
python scripts/render_eval_report.py "${REPORT_ARGS[@]}"
