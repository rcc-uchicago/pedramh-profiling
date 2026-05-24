#!/bin/bash
#SBATCH -J sfno_eval_report
# CPU-only job (render_eval_report.py reads CSVs, writes markdown).
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o logs/sfno_eval_report_%j.out
#SBATCH -e logs/sfno_eval_report_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# SFNO emulator report rendering (docs/sfno_eval_plan.md §G.2 row 3).
# Body extracted to scripts/eval_run_report_inline.sh (plan §4.3).
# ============================================================================

REPO_ROOT="${REPO_ROOT:-$HOME/projects/SFNO_Climate_Emulator}"
source "$REPO_ROOT/scripts/eval_run_report_inline.sh"
