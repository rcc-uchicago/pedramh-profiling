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
# Body extracted to scripts/eval_run_figures_inline.sh (plan §4.3).
# ============================================================================

REPO_ROOT="${REPO_ROOT:-$HOME/projects/SFNO_Climate_Emulator}"
source "$REPO_ROOT/scripts/eval_run_figures_inline.sh"
