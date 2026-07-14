#!/bin/bash
#SBATCH -J 5410_ab_A
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -o logs/5410_ab_A_%j.out
#SBATCH -e logs/5410_ab_A_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Gate A — A/B equivalence: legacy subprocess vs new in-process orchestrator
# for ONE IC (Y=121, s=0, K=60). Both paths run on the H100 in turn; pytest
# compares the two output NetCDFs at rtol=1e-5 across dims, var names, time
# coords, and per-variable values.
#
# Per docs/2026-05-08_sfno_5410_inproc_orchestrator_plan.md verification §4
# (gate A). Stop on failure — do NOT proceed to gate B/C if A fails.
#
# Both paths use async_save=False (the legacy launcher omits --async_save;
# the new orchestrator default is False).
#
# Estimated runtime:
#   * legacy subprocess: ~2:30 (cold-import + ckpt load + 1 IC inference)
#   * in-process: ~2:00 (one process, one model load, 1 IC)
#   * pytest overhead + comparison: ~30 s
#   * Total: ~5 min compute. 45 min budget gives plenty of headroom.
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

echo "[ab-gate-A] $(date) — starting"
set -x
time python -m pytest \
    tests/sfno_inference_5410/integration/test_ab_equivalence.py::test_one_ic_equivalence \
    -v --tb=long
RC=$?
set +x
echo "[ab-gate-A] $(date) — done (rc=$RC)"
exit $RC
