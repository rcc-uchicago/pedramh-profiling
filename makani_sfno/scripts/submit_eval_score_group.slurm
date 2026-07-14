#!/bin/bash
#SBATCH -J sfno_eval_score_group
# CPU-only job (compute_climatology.py + score_nwp.py are numpy/xarray;
# no CUDA imports). `skx` is the long-running CPU queue (no 2 h cap).
#SBATCH -p skx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o logs/sfno_eval_score_group_%j.out
#SBATCH -e logs/sfno_eval_score_group_%j.err

# ============================================================================
# Phase 1 v10-shim eval scoring — informative-mode fork of submit_eval_score.slurm.
#
# Behaviour:
#   - climatology_proleptic.nc built off TRAIN_DIR (same as submit_eval_score.slurm).
#   - score_nwp.py invoked with `set +e`; non-zero exit captured to scores/gate_status.txt
#     instead of halting the chain. Phase 1 smoke is unlikely to pass the scientific
#     gate (2-day rollout from a green-but-tiny train), so the gate is recorded as
#     INFORMATIVE rather than blocking.
#
# Inputs (env or default):
#   RUN_DIR, RUN_TAG (required), OUT_ROOT, TRAIN_DIR
# ============================================================================

# Note: we deliberately drop `-e` from the strict mode so a non-zero score
# exit does not abort the chain. We keep `-uo pipefail` for the rest.
set -uo pipefail

REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"

: "${RUN_DIR:?RUN_DIR must be set}"
: "${TRAIN_DIR:=$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full/train}"

if [[ -z "${RUN_TAG:-}" ]]; then
    echo "ERROR: RUN_TAG must be set" >&2
    exit 2
fi

: "${OUT_ROOT:=$WORK2/SFNO_Climate_Emulator/results/sfno_eval_group/$RUN_TAG}"
mkdir -p "$OUT_ROOT/baselines" "$OUT_ROOT/diagnostics" "$OUT_ROOT/scores"

source "$REPO_ROOT/.venv/bin/activate"
cd "$REPO_ROOT"
mkdir -p logs
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# 1) Climatology — idempotent; reuse the same v10 train pool as the Makani path.
CLIM_NC="$OUT_ROOT/baselines/climatology_proleptic.nc"
if [ ! -f "$CLIM_NC" ]; then
    python scripts/compute_climatology.py \
        --train-dir "$TRAIN_DIR" \
        --out "$CLIM_NC" \
        --source-files-out "$OUT_ROOT/diagnostics/climatology_source_files.json"
fi

# 2) NWP scoring — informative gate.
echo "[score_group] running score_nwp.py in informative mode..."
set +e
python scripts/score_nwp.py \
    --out-root "$OUT_ROOT" \
    --clim-nc "$CLIM_NC"
GATE_RC=$?
set -e

echo "score_nwp gate exit code: $GATE_RC" | tee "$OUT_ROOT/scores/gate_status.txt"
if [[ "$GATE_RC" -ne 0 ]]; then
    echo "[score_group] INFO: sanity gate exited non-zero (rc=$GATE_RC)."
    echo "[score_group] INFO: Phase 1 smoke informative-only — continuing chain."
fi

# Always exit 0 from this fork. The gate result is preserved in gate_status.txt
# and surfaced by the report job.
exit 0
