#!/bin/bash
# submit_eval_group.sh — Phase 1 v10-shim eval chain.
#
# Chains: submit_eval_inference_group.slurm (inference + converter)
#       → submit_eval_score_group.slurm (informative-gate scoring)
#
# Phase 1 is intentionally minimal: we skip the existing report/figures jobs
# (which assume the Makani 4-job chain). For a Phase F production run, fold
# them back in once the eval contract is settled.
#
# Required env vars:
#   RUN_NUM   (the smoke train run dir suffix; e.g. smoke_20260509_2300)
# Optional:
#   EXP_DIR, DATA_DIR, INIT_NC, INIT_DT, STEPS, OUT_ROOT, RUN_TAG, TRAIN_DIR,
#   TEST_H5, IC_GLOBAL_IDX
#   AFTEROK   (slurm jobid to wait for before submitting; e.g. the Phase D smoke train)

set -euo pipefail
REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
cd "$REPO_ROOT"

: "${RUN_NUM:?need RUN_NUM}"
: "${EXP_DIR:=$SCRATCH/SFNO_Climate_Emulator/runs/sfno_group_sigma10_smoke}"
: "${DATA_DIR:=$SCRATCH/SFNO_Climate_Emulator/data/group_sfno/sim52_smoke}"
: "${INIT_NC:=$EXP_DIR/init_smoke.nc}"
: "${INIT_DT:=0121-01-01 00:00:00}"
: "${STEPS:=8}"
: "${TRAIN_DIR:=$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full/train}"
: "${TEST_H5:=$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev/test/MOST.0121.h5}"
: "${IC_GLOBAL_IDX:=0}"

EVAL_SHA7="$(git rev-parse --short=7 HEAD || echo unknown)"
TS="$(date +%Y%m%d_%H%M)"
: "${RUN_TAG:=phase1_smoke_${EVAL_SHA7}_${TS}}"
: "${OUT_ROOT:=$WORK2/SFNO_Climate_Emulator/results/sfno_eval_group/$RUN_TAG}"

mkdir -p "$OUT_ROOT" logs

if [[ -n "${AFTEROK:-}" ]]; then
  DEP_FLAG="--dependency=afterok:$AFTEROK"
else
  DEP_FLAG=""
fi

# Step 1: inference + converter.
INF_JOBID=$(sbatch --parsable $DEP_FLAG 2>/dev/null \
  --export=ALL,RUN_NUM="$RUN_NUM",EXP_DIR="$EXP_DIR",DATA_DIR="$DATA_DIR",INIT_NC="$INIT_NC",INIT_DT="$INIT_DT",STEPS="$STEPS",OUT_ROOT="$OUT_ROOT",TEST_H5="$TEST_H5",IC_GLOBAL_IDX="$IC_GLOBAL_IDX",RUN_TAG="$RUN_TAG" \
  "$REPO_ROOT/scripts/submit_eval_inference_group.slurm" 2>/dev/null | grep -oE '^[0-9]+$' | tail -1)
echo "[eval_group] inference+converter job: $INF_JOBID"

# Step 2: scoring (informative gate).
SCO_JOBID=$(sbatch --parsable --dependency=afterok:$INF_JOBID 2>/dev/null \
  --export=ALL,RUN_DIR="$EXP_DIR/SFNO/$RUN_NUM",RUN_TAG="$RUN_TAG",OUT_ROOT="$OUT_ROOT",TRAIN_DIR="$TRAIN_DIR" \
  "$REPO_ROOT/scripts/submit_eval_score_group.slurm" 2>/dev/null | grep -oE '^[0-9]+$' | tail -1)
echo "[eval_group] score job: $SCO_JOBID"

cat <<EOF

[eval_group] chain submitted.
  RUN_TAG = $RUN_TAG
  OUT_ROOT = $OUT_ROOT
  inference job:  $INF_JOBID
  score job:      $SCO_JOBID  (depends on $INF_JOBID)

Watch: squeue -j $INF_JOBID,$SCO_JOBID
After both complete, check:
  $OUT_ROOT/inference/nwp/MOST.0121_ic000.nc   (converted scorer NetCDF)
  $OUT_ROOT/scores/scores.csv                  (if score_nwp succeeded)
  $OUT_ROOT/scores/gate_status.txt             (informative gate exit code)
EOF
