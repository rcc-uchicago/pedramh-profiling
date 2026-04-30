#!/bin/bash
# submit_eval.sh — chain the three eval SLURM jobs with afterok dependencies.
#
# Per docs/sfno_eval_plan.md §G.3.
#
# Required env vars:
#   RUN_DIR     (default: $SCRATCH/AI-RES/runs/sfno_full/plasim_sim52_full/0)
#   CKPT        (default: $RUN_DIR/training_checkpoints/best_ckpt_mp0.tar)
#   MODE        ('nwp' (default) or 'climate' — passed to inference job)
#
# Auto-derived:
#   EVAL_SHA7   — git SHA of AI-RES at submit time (--short=7).
#   DATA_SHA7   — read from any test h5 file's packager_git_sha attr.
#   TRAIN_SHA7  — preferred: $RUN_DIR/train_code_sha.txt (§G.5);
#                  fallback: grep "git hash:" $RUN_DIR/out.log;
#                  fallback: literal "unknown".
#   RUN_TAG     — composed from the four SHAs and ckpt basename.
#   OUT_ROOT    — $WORK2/AI-RES/results/sfno_eval/$RUN_TAG.
#
# Usage:
#   scripts/submit_eval.sh
#   MODE=climate scripts/submit_eval.sh

set -euo pipefail

REPO_ROOT="$HOME/AI-RES"
cd "$REPO_ROOT"

: "${RUN_DIR:=$SCRATCH/AI-RES/runs/sfno_full/plasim_sim52_full/0}"
: "${CKPT:=$RUN_DIR/training_checkpoints/best_ckpt_mp0.tar}"
: "${MODE:=nwp}"

# --- EVAL_SHA7 ---
if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git is required" >&2
    exit 2
fi
EVAL_SHA7="$(git rev-parse --short=7 HEAD)"

# --- DATA_SHA7 ---
TEST_FILE_FOR_SHA="$SCRATCH/AI-RES/data/makani/sim52_astro_64x128/test/MOST.0121.h5"
if [ ! -f "$TEST_FILE_FOR_SHA" ]; then
    echo "WARNING: $TEST_FILE_FOR_SHA not found; setting DATA_SHA7=unknown" >&2
    DATA_SHA7="unknown"
else
    DATA_SHA7="$(.venv/bin/python -c "
import h5py, sys
with h5py.File(sys.argv[1], 'r') as f:
    sha = f.attrs.get('packager_git_sha', b'unknown')
    if isinstance(sha, bytes):
        sha = sha.decode('utf-8')
    print(sha[:7])
" "$TEST_FILE_FOR_SHA")"
fi

# --- TRAIN_SHA7 ---
if [ -f "$RUN_DIR/train_code_sha.txt" ]; then
    TRAIN_SHA7="$(head -c 7 "$RUN_DIR/train_code_sha.txt")"
elif [ -f "$RUN_DIR/out.log" ]; then
    TRAIN_SHA7="$(grep -m1 "git hash:" "$RUN_DIR/out.log" | sed -E "s/.*git hash: b?'?([0-9a-f]{7}).*/\1/")"
    [ -z "$TRAIN_SHA7" ] && TRAIN_SHA7="unknown"
else
    TRAIN_SHA7="unknown"
fi

# --- RUN_TAG ---
DATE_STR="$(date +%Y%m%d)"
CKPT_BASENAME="$(basename "$CKPT" .tar)"
: "${RUN_TAG:=${DATE_STR}_eval-${EVAL_SHA7}_data-${DATA_SHA7}_train-${TRAIN_SHA7}_ckpt-${CKPT_BASENAME}}"

# --- OUT_ROOT ---
: "${OUT_ROOT:=$WORK2/AI-RES/results/sfno_eval/$RUN_TAG}"

mkdir -p "$OUT_ROOT" logs

# Echo the resolved provenance.
cat <<EOF
[submit_eval]
  RUN_DIR    = $RUN_DIR
  CKPT       = $CKPT
  MODE       = $MODE
  EVAL_SHA7  = $EVAL_SHA7
  DATA_SHA7  = $DATA_SHA7
  TRAIN_SHA7 = $TRAIN_SHA7
  RUN_TAG    = $RUN_TAG
  OUT_ROOT   = $OUT_ROOT
EOF

export RUN_DIR CKPT MODE EVAL_SHA7 DATA_SHA7 TRAIN_SHA7 RUN_TAG OUT_ROOT

JOB_INF="$(sbatch --parsable scripts/submit_eval_inference.slurm)"
echo "[submit_eval] inference   job: $JOB_INF"

JOB_SCO="$(sbatch --parsable --dependency=afterok:$JOB_INF scripts/submit_eval_score.slurm)"
echo "[submit_eval] scoring     job: $JOB_SCO  (afterok:$JOB_INF)"

JOB_REP="$(sbatch --parsable --dependency=afterok:$JOB_SCO scripts/submit_eval_report.slurm)"
echo "[submit_eval] report      job: $JOB_REP  (afterok:$JOB_SCO)"

echo
echo "Final report path on success:"
echo "  $OUT_ROOT/report.md"
