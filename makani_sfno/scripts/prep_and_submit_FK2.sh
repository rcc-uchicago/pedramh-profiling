#!/bin/bash
# Phase F.K2 driver — smoke gate against the existing 1-epoch smoke ckpt.
#
# Prerequisites (must already be done):
#   F.B1 — 101 native years converted at $DATA_DIR_FULL              [DONE]
#   F.C  — stats + climatology under $DATA_DIR_FULL                  [JOBID]
#   F.B2 — year 11 padded to 1460 frames; year 12 to 1464            [JOBID]
#   F.D  — init NCs at $EXP_DIR_FULL/init_year{121,124}.nc           [JOBID]
#   smoke ckpt at $SCRATCH/SFNO_Climate_Emulator/runs/sfno_group_sigma10_smoke/SFNO/smoke_*/checkpoints/best_ckpt.tar
#
# What this script does (all in ~3 minutes, mostly I/O):
#   1. Render production YAML against production data_dir + smoke EXP_DIR.
#   2. Stage smoke ckpt under a new RUN_DIR=$EXP_DIR_FULL/SFNO/smoke_FK2_<TS>/checkpoints/best_ckpt.tar.
#   3. Run pre_train_full preflight (Checks 1-10 + 16 + 17 + 20).
#   4. Submit submit_eval_group_prod.sh YEARS="121 124" MODE=informative.
#
# Tier 1 acceptance (must pass): no schema/runtime errors, NetCDFs written,
# leads 1..60 finite, scorer runs to completion. Tier 2 (science gates) may fail.

set -euo pipefail
REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"

DATA_DIR_FULL="${DATA_DIR_FULL:-$SCRATCH/SFNO_Climate_Emulator/data/group_sfno/sim52_full}"
EXP_DIR_FULL="${EXP_DIR_FULL:-$SCRATCH/SFNO_Climate_Emulator/runs/sfno_group_sigma10_full}"
SMOKE_CKPT="${SMOKE_CKPT:-}"
TS=$(date +%Y%m%d_%H%M)
RUN_NUM="smoke_FK2_${TS}"
RUN_DIR="$EXP_DIR_FULL/SFNO/$RUN_NUM"

echo "[FK2] DATA_DIR_FULL = $DATA_DIR_FULL"
echo "[FK2] EXP_DIR_FULL  = $EXP_DIR_FULL"
echo "[FK2] RUN_DIR       = $RUN_DIR"

# --- Locate smoke ckpt automatically if not specified ---
if [[ -z "$SMOKE_CKPT" ]]; then
  SMOKE_BEST=$(find $SCRATCH/SFNO_Climate_Emulator/runs/sfno_group_sigma10_smoke/SFNO -maxdepth 3 \
    -name 'best_ckpt.tar' -type f -o -name 'best_ckpt.tar' -type l 2>/dev/null | head -1)
  if [[ -z "$SMOKE_BEST" ]]; then
    echo "ERROR: no smoke ckpt found under $SCRATCH/SFNO_Climate_Emulator/runs/sfno_group_sigma10_smoke/"
    echo "       Set SMOKE_CKPT explicitly to override."
    exit 2
  fi
  SMOKE_CKPT="$SMOKE_BEST"
fi
echo "[FK2] smoke ckpt: $SMOKE_CKPT"
[[ -f "$SMOKE_CKPT" ]] || { echo "ERROR: ckpt not a regular file"; exit 2; }

# --- Sanity: F.B2 done (manifest has n_timesteps_padded for years 11, 12) ---
python3 -c "
import json, sys
m = json.load(open('$DATA_DIR_FULL/_v10_calendar_manifest.json'))
by = {y['year']: y for y in m['years']}
for y, target in [(11, 1460), (12, 1464)]:
    e = by.get(y)
    if e is None:
        print(f'ERROR: year {y} missing from manifest'); sys.exit(2)
    if e.get('n_timesteps_padded') != target:
        print(f'ERROR: year {y} n_timesteps_padded={e.get(\"n_timesteps_padded\")} != {target}; F.B2 not yet done'); sys.exit(2)
    print(f'  year {y}: padded={e[\"n_timesteps_padded\"]} (donor={e.get(\"pad_source\", [{}])[0].get(\"src_year\")})')
print('manifest pad metadata OK')
"

# --- Sanity: stats / climatology / init NCs present ---
for f in data_train_mean.nc data_train_std.nc climatology.nc; do
  [[ -f "$DATA_DIR_FULL/$f" ]] || { echo "ERROR: $DATA_DIR_FULL/$f missing (F.C not done?)"; exit 2; }
done
for y in 121 124; do
  [[ -f "$EXP_DIR_FULL/init_year${y}.nc" ]] || { echo "ERROR: $EXP_DIR_FULL/init_year${y}.nc missing (F.D not done?)"; exit 2; }
done
echo "[FK2] stats / climatology / init NCs all present"

# --- Step 1: render production YAML ---
mkdir -p "$RUN_DIR/checkpoints" "$REPO_ROOT/logs"
RENDERED="$RUN_DIR/rendered.yaml"
TRAIN_YEARS=$(seq 12 111 | tr '\n' ' ')

# Use .venv (no cuda needed for render).
source "$REPO_ROOT/.venv/bin/activate"
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m sfno_training_group.tools.render_yaml \
  --tpl "$REPO_ROOT/src/sfno_training_group/config/plasim_sim52_sigma10_sfno_full.yaml" \
  --out "$RENDERED" \
  --data-dir "$DATA_DIR_FULL" \
  --exp-dir "$EXP_DIR_FULL" \
  --manifest "$DATA_DIR_FULL/_v10_calendar_manifest.json" \
  --train-years $TRAIN_YEARS \
  --val-years 11

echo "[FK2] rendered YAML: $RENDERED"

# --- Step 2: stage smoke ckpt ---
cp -f "$SMOKE_CKPT" "$RUN_DIR/checkpoints/best_ckpt.tar"
ln -sfn "$RUN_DIR/checkpoints" "$RUN_DIR/training_checkpoints"
echo "[FK2] staged smoke ckpt at $RUN_DIR/checkpoints/best_ckpt.tar"

# --- Step 3: pre_train_full preflight (Checks 1-10 + 16 + 17 + 20) ---
echo "[FK2] running pre_train_full preflight..."
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m sfno_training_group.tools.preflight_checks \
  --yaml "$RENDERED" --config SFNO --phase pre_train_full \
  --data-dir "$DATA_DIR_FULL" --manifest "$DATA_DIR_FULL/_v10_calendar_manifest.json" \
  --run-dir "$RUN_DIR"

# --- Step 4: submit eval chain ---
echo "[FK2] submitting eval chain (long_inference + converter + score informative)..."
RUN_DIR="$RUN_DIR" \
RENDERED="$RENDERED" \
YEARS="121 124" \
MODE="informative" \
DATA_DIR="$DATA_DIR_FULL" \
"$REPO_ROOT/scripts/submit_eval_group_prod.sh"

echo
echo "[FK2] DONE. After all jobs complete, check Tier 1 acceptance:"
echo "  - submit_eval_group_prod.sh report block above lists all jobids"
echo "  - per-IC NetCDFs at $WORK2/SFNO_Climate_Emulator/results/sfno_eval_group/<RUN_TAG>/inference/nwp/"
echo "  - finite K=60 leads (T.10 contract); converter exit 0; scorer exit 0 (informative gate may report science fail)"
