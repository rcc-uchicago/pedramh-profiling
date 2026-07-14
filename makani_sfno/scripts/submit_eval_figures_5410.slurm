#!/bin/bash
#SBATCH -J 5410_figures
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o logs/5410_figures_%j.out
#SBATCH -e logs/5410_figures_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# 5410 figure-rendering SLURM (per docs/2026-05-08_sfno_5410_scoring_plan.md v4.4).
#
# Invokes the existing scripts/render_eval_figures.py with --track 5410 so the
# pr_6h colorbar / axis stays in native units (kg m^-2, 6h accumulation)
# instead of the own-track m s^-1 → mm day^-1 conversion. CPU-only, runs on
# skx-dev to avoid tying up an h100 slot.
#
# Required env (set by submit_eval_5410.sh):
#   OUT_ROOT
# ============================================================================

set -euo pipefail
REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
cd "$REPO_ROOT"
mkdir -p logs

source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

: "${OUT_ROOT:?OUT_ROOT required}"

echo "[5410_figures] $(date) — start"
set -x
time python -u "$REPO_ROOT/scripts/render_eval_figures.py" \
    --out-root "$OUT_ROOT" \
    --track 5410
set +x
echo "[5410_figures] $(date) — done; figures at $OUT_ROOT/figures/"
