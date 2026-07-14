#!/bin/bash
#SBATCH -J backfill_tas_no_ice_own
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -o logs/backfill_tas_no_ice_own_%j.out
#SBATCH -e logs/backfill_tas_no_ice_own_%j.err
#SBATCH --mail-user=feynmannliu@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Backfill truth_sic into existing own-track per-IC inference NetCDFs and
# re-score for tas_no_ice. Produces a NEW OUT_ROOT (does NOT overwrite the
# original eval root).
#
# Required env:
#   IN_ROOT        — existing eval out_root (has inference/nwp/, baselines/)
#   OUT_ROOT       — new out_root for the tas_no_ice run
#   TEST_HOLDOUT   — dir of MOST.{YEAR}.h5 with forcing array
#   RUN_TAG        — run-tag for the report header
#
# Optional:
#   CKPT_PATH      — checkpoint path string for the report (default: ic_file
#                    from one inference NC if unset)
# ============================================================================
set -euo pipefail
REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
cd "$REPO_ROOT"
source .venv/bin/activate
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs

: "${IN_ROOT:?IN_ROOT required}"
: "${OUT_ROOT:?OUT_ROOT required}"
: "${TEST_HOLDOUT:?TEST_HOLDOUT required}"
: "${RUN_TAG:?RUN_TAG required}"

EVAL_SHA7="$(git -C "$REPO_ROOT" rev-parse --short=7 HEAD)"

# Pull ckpt_path from one of the original inference NCs so the report is
# faithful to where the predictions came from.
if [[ -z "${CKPT_PATH:-}" ]]; then
    CKPT_PATH="$(python - <<PY
import sys
from pathlib import Path
import xarray as xr
ncs = sorted(Path("$IN_ROOT/inference/nwp").glob("*.nc"))
if not ncs:
    sys.exit("no inference NCs at $IN_ROOT")
with xr.open_dataset(ncs[0]) as ds:
    print(ds.attrs.get("ckpt_path", "unknown"))
PY
)"
fi

echo "=== backfill_tas_no_ice_own ==="
echo "  IN_ROOT      = $IN_ROOT"
echo "  OUT_ROOT     = $OUT_ROOT"
echo "  TEST_HOLDOUT = $TEST_HOLDOUT"
echo "  RUN_TAG      = $RUN_TAG"
echo "  CKPT_PATH    = $CKPT_PATH"
echo "  EVAL_SHA7    = $EVAL_SHA7"

set -x

# 1. Backfill truth_sic into new out_root
python scripts/backfill_truth_sic.py \
  --in-root "$IN_ROOT" \
  --out-root "$OUT_ROOT" \
  --test-holdout "$TEST_HOLDOUT"

# 2. Re-score
CLIM_NC="$OUT_ROOT/baselines/climatology_proleptic.nc"
mkdir -p "$OUT_ROOT/scores"
python scripts/score_nwp.py \
  --out-root "$OUT_ROOT" \
  --clim-nc "$CLIM_NC"

# 3. Report
python scripts/render_eval_report.py \
  --out-root "$OUT_ROOT" \
  --run-tag "$RUN_TAG" \
  --eval-sha7 "$EVAL_SHA7" \
  --data-sha7 "backfill-truth_sic" \
  --train-sha7 "backfill-truth_sic" \
  --ckpt-path "$CKPT_PATH"

# 4. Figures
python scripts/render_eval_figures.py --out-root "$OUT_ROOT"

set +x
echo "=== DONE ==="
echo "scorecard: $OUT_ROOT/scores/nwp_scorecard_summary.csv"
echo "report:    $OUT_ROOT/report.md"
echo "figures:   $OUT_ROOT/figures/"
