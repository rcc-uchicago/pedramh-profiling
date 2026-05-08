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

source "$REPO_ROOT/.venv/bin/activate"

cd "$REPO_ROOT"
mkdir -p logs

set -x
python scripts/render_eval_figures.py --out-root "$OUT_ROOT"
