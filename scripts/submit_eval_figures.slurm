#!/bin/bash
#SBATCH -J sfno_eval_figures
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o logs/sfno_eval_figures_%j.out
#SBATCH -e logs/sfno_eval_figures_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# SFNO emulator figure rendering (4th job in eval chain).
# Reads scores/nwp_scorecard_summary.csv + scores/bias_maps_*.npy under
# $OUT_ROOT and writes PNGs into $OUT_ROOT/figures/:
#   - rmse_vs_lead.png, acc_vs_lead.png  (4-panel line plots)
#   - bias_<channel>.png x 5             (1x6 lead panels per channel)
# CPU-only; runs on skx-dev so the chain doesn't tie up an h100 slot.
# ============================================================================

set -euo pipefail

REPO_ROOT="$HOME/AI-RES"

if [[ -z "${RUN_TAG:-}" ]]; then
    echo "ERROR: RUN_TAG must be set" >&2
    exit 2
fi

: "${OUT_ROOT:=$WORK2/AI-RES/results/sfno_eval/$RUN_TAG}"
: "${BENCHMARK_5410_OUT_ROOT:=/work2/11114/zhixingliu/stampede3/AI-RES/results/sfno_eval_5410/20260509_blocking_96ic_h100_packed_derecho_env_valid}"

source "$REPO_ROOT/.venv/bin/activate"

cd "$REPO_ROOT"
mkdir -p logs

FIG_ARGS=( --out-root "$OUT_ROOT" )
# Empty BENCHMARK_5410_OUT_ROOT disables the overlay entirely; default
# above points at the H100 + packed-env settled valid 5410 run.
if [[ -n "${BENCHMARK_5410_OUT_ROOT:-}" ]]; then
    FIG_ARGS+=( --benchmark-5410-out-root "$BENCHMARK_5410_OUT_ROOT" )
fi

set -x
python scripts/render_eval_figures.py "${FIG_ARGS[@]}"
