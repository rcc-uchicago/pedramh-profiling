#!/bin/bash
#SBATCH -J 5410_ab_C
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -o logs/5410_ab_C_%j.out
#SBATCH -e logs/5410_ab_C_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Gate C — A/B equivalence: cross-year pair (Y=121 s=0 then Y=122 s=0).
# Exercises Stepper.reconfigure_for_ic with val_year_changed=True on the
# second call. Verifies that:
#   * constant_boundary_data refreshes correctly on Y boundary;
#   * val_year_start, val_year_end, leap_year, no_leap_year all update;
#   * the data loaders rebuild for the new Y.
#
# This is the most state-mutating reconfigure_for_ic path. Codex called
# it out as the must-have gate before any production submit.
#
# Per docs/2026-05-08_sfno_5410_inproc_orchestrator_plan.md verification §6.
# Run AFTER gates A and B pass. Stop on failure — do NOT submit production
# if C fails.
#
# Estimated runtime:
#   * legacy: 2 ICs × ~2:30 each = ~5 min
#   * in-process: 1 setup + 2 reconfigure-and-predict (with Y refresh) = ~3 min
#   * Total: ~10 min compute. 45 min budget for headroom.
# ============================================================================

set -euo pipefail
REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
cd "$REPO_ROOT"
mkdir -p logs

source "$REPO_ROOT/.venv/bin/activate"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export RUN_AB_TESTS=1

echo "[ab-gate-C] $(date) — starting"
set -x
time python -m pytest \
    tests/sfno_inference_5410/integration/test_ab_equivalence.py::test_cross_year_pair_equivalence \
    -v --tb=long
RC=$?
set +x
echo "[ab-gate-C] $(date) — done (rc=$RC)"
exit $RC
