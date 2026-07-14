#!/bin/bash
# Phase F.J + F.M — production eval orchestrator for the group SFNO sigma10 ckpt.
#
# Per test year (YEAR in YEARS list):
#   1. submit_long_inference_full.slurm  -> long_rollout NetCDF + 53-channel scorer NC
#   2. submit_eval_score.slurm          -> climatology + scorecard CSV + bias maps
#                                          (informative-mode for F.K2; fail-stop for F.M)
#   3. submit_eval_report.slurm         -> report.md (after all per-IC scoring done)
#   4. submit_eval_figures.slurm        -> figures dir
#
# Required env:
#   RUN_DIR    Train run dir (e.g. $EXP_DIR/SFNO/prod_<TS>); must contain rendered.yaml
#   YEARS      Space-separated test years (default "121 124" for F.K2; "121 122 123 124 125 126 127 128" for F.M)
# Optional:
#   MODE       "informative" (default for F.K2) or "production" (fail-stop, default for F.M)
#   OUT_ROOT   default $WORK2/SFNO_Climate_Emulator/results/sfno_eval_group/${RUN_TAG}
#   RUN_TAG    default phaseF_${SHA7}_${TS}
#   DATA_DIR   default $SCRATCH/SFNO_Climate_Emulator/data/group_sfno/sim52_full
#   TEST_SRC   default $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev/test
#   TRAIN_DIR  default $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full/train
#   MAX_OUT_LEADS  default 60 (= 360h scorecard horizon)
#   AFTEROK    optional jobid to wait for before submitting

set -euo pipefail
REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
cd "$REPO_ROOT"

: "${RUN_DIR:?need RUN_DIR (e.g. \$EXP_DIR/SFNO/prod_<TS>)}"
: "${YEARS:=121 124}"
: "${MODE:=informative}"
: "${DATA_DIR:=$SCRATCH/SFNO_Climate_Emulator/data/group_sfno/sim52_full}"
: "${TEST_SRC:=$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev/test}"
: "${TRAIN_DIR:=$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full/train}"
: "${MAX_OUT_LEADS:=60}"

EVAL_SHA7=$(git rev-parse --short=7 HEAD 2>/dev/null || echo unknown)
TS=$(date +%Y%m%d_%H%M)
: "${RUN_TAG:=phaseF_${EVAL_SHA7}_${TS}}"
: "${OUT_ROOT:=$WORK2/SFNO_Climate_Emulator/results/sfno_eval_group/$RUN_TAG}"

mkdir -p "$OUT_ROOT/inference/nwp" "$OUT_ROOT/scores" "$OUT_ROOT/baselines" \
         "$OUT_ROOT/diagnostics" "$REPO_ROOT/logs"

# Pick scoring script per MODE.
case "$MODE" in
  informative) SCORE_SLURM="$REPO_ROOT/scripts/submit_eval_score_group.slurm" ;;
  production)  SCORE_SLURM="$REPO_ROOT/scripts/submit_eval_score.slurm" ;;
  *) echo "ERROR: MODE must be 'informative' or 'production' (got $MODE)"; exit 2 ;;
esac

# Extract RUN_NUM from RUN_DIR (last path segment).
RUN_NUM=$(basename "$RUN_DIR")
EXP_DIR=$(dirname "$(dirname "$RUN_DIR")")  # strip /SFNO/<RUN_NUM>
RENDERED="${RENDERED:-$RUN_DIR/rendered.yaml}"
if [[ ! -f "$RENDERED" ]]; then
  echo "ERROR: rendered YAML not found at $RENDERED. Render it first."
  exit 2
fi

if [[ -n "${AFTEROK:-}" ]]; then
  DEP_FLAG="--dependency=afterok:$AFTEROK"
else
  DEP_FLAG=""
fi

echo "[eval_group_prod] RUN_DIR=$RUN_DIR  RUN_NUM=$RUN_NUM"
echo "[eval_group_prod] OUT_ROOT=$OUT_ROOT  RUN_TAG=$RUN_TAG"
echo "[eval_group_prod] MODE=$MODE -> $SCORE_SLURM"
echo "[eval_group_prod] YEARS: $YEARS"

# --- Step 1+2 per IC: long_inference + converter (in same slurm job) ---
INF_JOBIDS=()
for YEAR in $YEARS; do
  INIT_NC="${EXP_DIR}/init_year${YEAR}.nc"
  TEST_H5="$TEST_SRC/MOST.$(printf '%04d' $YEAR).h5"

  EXPORT_LIST="ALL,YEAR=$YEAR,RUN_DIR=$RUN_DIR,RUN_NUM=$RUN_NUM,DATA_DIR=$DATA_DIR,INIT_NC=$INIT_NC,OUT_ROOT=$OUT_ROOT,TEST_H5=$TEST_H5,RUN_TAG=$RUN_TAG,MAX_OUT_LEADS=$MAX_OUT_LEADS,IC_GLOBAL_IDX=0"
  jobid=$(sbatch --parsable $DEP_FLAG \
    --export="$EXPORT_LIST" \
    "$REPO_ROOT/src/sfno_training_group/slurm/submit_long_inference_full.slurm" \
    2>/dev/null | grep -oE '^[0-9]+$' | tail -1)
  echo "[eval_group_prod]   YEAR=$YEAR  inf+convert jobid=$jobid"
  INF_JOBIDS+=("$jobid")
done

# --- Step 3 per IC: scoring (depends on that IC's inference job) ---
SCORE_JOBIDS=()
for i in "${!INF_JOBIDS[@]}"; do
  prev=${INF_JOBIDS[$i]}
  EXPORT_LIST="ALL,RUN_DIR=$RUN_DIR,RUN_TAG=$RUN_TAG,OUT_ROOT=$OUT_ROOT,TRAIN_DIR=$TRAIN_DIR"
  jobid=$(sbatch --parsable --dependency=afterok:$prev \
    --export="$EXPORT_LIST" "$SCORE_SLURM" \
    2>/dev/null | grep -oE '^[0-9]+$' | tail -1)
  echo "[eval_group_prod]   score jobid=$jobid (afterok $prev)"
  SCORE_JOBIDS+=("$jobid")
done

# --- Step 4: report (after all scoring done) ---
ALL_SCORE_DEP=$(IFS=:; echo "${SCORE_JOBIDS[*]}")
EXPORT_LIST="ALL,RUN_DIR=$RUN_DIR,RUN_TAG=$RUN_TAG,OUT_ROOT=$OUT_ROOT"
REPORT_JOBID=$(sbatch --parsable --dependency=afterok:$ALL_SCORE_DEP \
  --export="$EXPORT_LIST" \
  "$REPO_ROOT/scripts/submit_eval_report.slurm" \
  2>/dev/null | grep -oE '^[0-9]+$' | tail -1)
echo "[eval_group_prod] report jobid=$REPORT_JOBID (afterok $ALL_SCORE_DEP)"

# --- Step 5: figures (after report) ---
FIGURES_JOBID=$(sbatch --parsable --dependency=afterok:$REPORT_JOBID \
  --export="$EXPORT_LIST" \
  "$REPO_ROOT/scripts/submit_eval_figures.slurm" \
  2>/dev/null | grep -oE '^[0-9]+$' | tail -1)
echo "[eval_group_prod] figures jobid=$FIGURES_JOBID (afterok $REPORT_JOBID)"

cat <<EOF

[eval_group_prod] CHAIN SUBMITTED.
  RUN_TAG  = $RUN_TAG
  OUT_ROOT = $OUT_ROOT
  MODE     = $MODE
  inference jobs: ${INF_JOBIDS[*]}
  score jobs:     ${SCORE_JOBIDS[*]}
  report job:     $REPORT_JOBID
  figures job:    $FIGURES_JOBID

Watch:    squeue -j $(IFS=,; echo "${INF_JOBIDS[*]} ${SCORE_JOBIDS[*]} $REPORT_JOBID $FIGURES_JOBID")
After completion:
  $OUT_ROOT/inference/nwp/MOST.0YYY_ic000.nc   (per-IC 53-channel NetCDFs)
  $OUT_ROOT/scores/                            (scorecards, bias maps)
  $OUT_ROOT/scores/gate_status.txt             (informative-mode only)
  $OUT_ROOT/report.md                          (with GB4 baseline overlay)
  $OUT_ROOT/figures/                           (rendered plots)
EOF
