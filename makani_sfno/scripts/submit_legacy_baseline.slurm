#!/bin/bash
#SBATCH -J 5410_legacy_floor
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -o logs/5410_legacy_floor_%j.out
#SBATCH -e logs/5410_legacy_floor_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Legacy-vs-legacy nondeterminism floor — Codex round-3 diagnostic.
# Runs upstream long_inference.py twice for Y=121 s=0 K=60 in fresh
# subprocesses, then prints per-variable max_abs / max_rel between the
# two outputs. Establishes the empirical floor for cudnn.benchmark + AMP
# nondeterminism on this exact pipeline.
#
# Used to inform the A/B equivalence tolerance for gates A/B/C. If
# legacy floor ≈ 4e-5 (matching the gate-A failure magnitude), the
# refactor is fine and we set rtol = 10× the floor.
#
# Estimated runtime: 2× legacy = ~5 min. 45 min budget for headroom.
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

echo "[legacy-floor] $(date) — starting"
set -x
time python -m pytest \
    tests/sfno_inference_5410/integration/test_legacy_baseline.py::test_legacy_vs_legacy_baseline \
    -v -s --tb=long
RC=$?
set +x
echo "[legacy-floor] $(date) — done (rc=$RC)"
exit $RC
