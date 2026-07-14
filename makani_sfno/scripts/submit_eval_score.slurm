#!/bin/bash
#SBATCH -J sfno_eval_score
# CPU-only job (compute_climatology.py + score_nwp.py are numpy/xarray; no
# CUDA imports). skx-dev: 2 h cap fits first-eval climatology build
# (~30–60 min) + scoring, with near-zero queue vs skx.
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o logs/sfno_eval_score_%j.out
#SBATCH -e logs/sfno_eval_score_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# SFNO emulator scoring (docs/sfno_eval_plan.md §G.2 row 2).
# Body extracted to scripts/eval_run_score_inline.sh (plan §4.3).
# ============================================================================

REPO_ROOT="${REPO_ROOT:-$HOME/projects/SFNO_Climate_Emulator}"
source "$REPO_ROOT/scripts/eval_run_score_inline.sh"
