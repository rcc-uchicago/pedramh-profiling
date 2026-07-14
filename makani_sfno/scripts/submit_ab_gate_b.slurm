#!/bin/bash
#SBATCH -J 5410_ab_B
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -o logs/5410_ab_B_%j.out
#SBATCH -e logs/5410_ab_B_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Gate B — A/B equivalence: two same-year ICs (Y=121, s=0 then s=122).
# Exercises Stepper.reconfigure_for_ic within ONE year (val_year_changed=False
# on the second call). Verifies that the data loaders rebuild correctly
# without any Y-boundary refresh, and that no state leaks from IC #1 into IC #2.
#
# Per docs/2026-05-08_sfno_5410_inproc_orchestrator_plan.md verification §5.
# Run AFTER gate A passes. Stop on failure — do NOT proceed to gate C if B fails.
#
# Estimated runtime:
#   * legacy: 2 ICs × ~2:30 each = ~5 min
#   * in-process: 1 setup + 2 reconfigure-and-predict = ~3 min
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

echo "[ab-gate-B] $(date) — starting"
set -x
time python -m pytest \
    tests/sfno_inference_5410/integration/test_ab_equivalence.py::test_two_same_year_ics_equivalence \
    -v --tb=long
RC=$?
set +x
echo "[ab-gate-B] $(date) — done (rc=$RC)"
exit $RC
