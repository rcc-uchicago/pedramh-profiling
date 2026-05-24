#!/bin/bash
# eval_run_figures_inline.sh — eval-figures stage body, sourceable.
# Used by scripts/submit_eval_figures.slurm (standalone) and bundled_eval.sh.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/projects/SFNO_Climate_Emulator}"

if [[ -z "${RUN_TAG:-}" ]]; then
    echo "ERROR: RUN_TAG must be set" >&2
    exit 2
fi

: "${OUT_ROOT:=$WORK2/SFNO_Climate_Emulator/results/sfno_eval/$RUN_TAG}"
: "${BENCHMARK_5410_OUT_ROOT:=/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval_5410/20260509_blocking_96ic_h100_packed_derecho_env_valid}"
: "${TRACK:=own}"

source "$REPO_ROOT/.venv/bin/activate"

cd "$REPO_ROOT"
mkdir -p logs

FIG_ARGS=( --out-root "$OUT_ROOT" --track "$TRACK" )
if [[ -n "${BENCHMARK_5410_OUT_ROOT:-}" ]]; then
    FIG_ARGS+=( --benchmark-5410-out-root "$BENCHMARK_5410_OUT_ROOT" )
fi

set -x
python scripts/render_eval_figures.py "${FIG_ARGS[@]}"
