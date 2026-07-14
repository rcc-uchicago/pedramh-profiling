#!/bin/bash
# ============================================================================
# run_backfill_tas_no_ice.sh — one-shot dispatcher for tas_no_ice across
# the three eval tracks the user asked for on 2026-05-14:
#
#   1. 5410 production (re-adapt + re-score; the adapter now writes truth_sic)
#   2. group_clone v10 (best_ckpt_mp0) — backfill truth_sic + re-score
#   3. v11_clip (best_ckpt_ema_mp0)    — backfill truth_sic + re-score
#
# All three land in fresh OUT_ROOTs alongside the originals; no prior eval
# is overwritten. Login-side invocation; the work runs on compute nodes.
# ============================================================================
set -euo pipefail

REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
cd "$REPO_ROOT"
mkdir -p logs

DATE_STR=$(date +%Y%m%d_%H%M)

# ----------------------------------------------------------------------------
# Track 2: group_clone v10 (best_ckpt_mp0)
# ----------------------------------------------------------------------------
GC_V10_IN="/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval/20260510_eval-8b395eb_data-e3c934b"
GC_V10_OUT="/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval/tas_no_ice_${DATE_STR}_group_clone_v10_mp0"
GC_V10_TEST_HOLDOUT="/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full/test_holdout"

# ----------------------------------------------------------------------------
# Track 3: v11_clip (best_ckpt_ema_mp0; canonical per [[feedback_ema_is_canonical_ckpt]])
# ----------------------------------------------------------------------------
V11_CLIP_IN="/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval/20260513_eval-8b395eb_data-e3c934b_family-sfno_zgplev_group_clone_v11_clip_ckpt-best_ckpt_ema_mp0"
V11_CLIP_OUT="/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval/tas_no_ice_${DATE_STR}_v11_clip_ema"
V11_CLIP_TEST_HOLDOUT="/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full_v11/test_holdout"

# ----------------------------------------------------------------------------
# Track 1: 5410 production (re-adapt via submit_eval_5410.sh)
# ----------------------------------------------------------------------------
S5410_RUN_ROOT="/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval_5410/20260509_blocking_96ic_h100_packed_derecho_env_valid"
S5410_OUT_ROOT="/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval_5410/tas_no_ice_${DATE_STR}_5410_prod"
S5410_RUN_TAG="tas_no_ice_${DATE_STR}_5410_prod"
S5410_CKPT="/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_blocking/sfno5410_blocking_epoch48_20260509/checkpoints/ckpt_epoch_48.tar"

echo "============================================================"
echo "Dispatching tas_no_ice backfill jobs (date_str=$DATE_STR)"
echo "============================================================"
echo
echo "Track 2 (group_clone v10):"
echo "  IN  = $GC_V10_IN"
echo "  OUT = $GC_V10_OUT"
echo
echo "Track 3 (v11_clip EMA):"
echo "  IN  = $V11_CLIP_IN"
echo "  OUT = $V11_CLIP_OUT"
echo
echo "Track 1 (5410 prod):"
echo "  RUN_ROOT = $S5410_RUN_ROOT  (existing upstream_raw)"
echo "  OUT_ROOT = $S5410_OUT_ROOT"
echo
echo "============================================================"

# Track 2: group_clone v10
JOB_GC_V10=$(sbatch \
  --export=ALL,IN_ROOT="$GC_V10_IN",OUT_ROOT="$GC_V10_OUT",TEST_HOLDOUT="$GC_V10_TEST_HOLDOUT",RUN_TAG="tas_no_ice_${DATE_STR}_group_clone_v10_mp0" \
  scripts/submit_backfill_tas_no_ice_own.slurm | awk '{print $NF}')
echo "[track 2] submitted job $JOB_GC_V10 (group_clone v10)"

# Track 3: v11_clip EMA
JOB_V11=$(sbatch \
  --export=ALL,IN_ROOT="$V11_CLIP_IN",OUT_ROOT="$V11_CLIP_OUT",TEST_HOLDOUT="$V11_CLIP_TEST_HOLDOUT",RUN_TAG="tas_no_ice_${DATE_STR}_v11_clip_ema" \
  scripts/submit_backfill_tas_no_ice_own.slurm | awk '{print $NF}')
echo "[track 3] submitted job $JOB_V11 (v11_clip EMA)"

# Track 1: 5410 prod — delegate to submit_eval_5410.sh which submits its own
# afterok chain (adapt+score → report → figures). SKIP_INF=1 reuses the
# existing upstream_raw; FORCE not needed since OUT_ROOT is new.
echo "[track 1] handing off to submit_eval_5410.sh ..."
SKIP_INF=1 \
RUN_ROOT="$S5410_RUN_ROOT" \
OUT_ROOT="$S5410_OUT_ROOT" \
RUN_TAG="$S5410_RUN_TAG" \
CKPT="$S5410_CKPT" \
GROUP_SHA7="5410-blocking-epoch48" \
MODEL_SHA7="ckpt_epoch_48" \
bash scripts/submit_eval_5410.sh

echo
echo "============================================================"
echo "All jobs dispatched. Monitor with:  squeue -u $USER"
echo "============================================================"
echo "Scorecards (when done):"
echo "  $GC_V10_OUT/scores/nwp_scorecard_summary.csv"
echo "  $V11_CLIP_OUT/scores/nwp_scorecard_summary.csv"
echo "  $S5410_OUT_ROOT/scores/nwp_scorecard_summary.csv"
