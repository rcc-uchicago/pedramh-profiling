#!/bin/bash
# Phase F.G — Per-year long_inference for the group SFNO sigma10 production ckpt.
# Plan: docs/2026-05-10_phaseF_group_production_plan.md (v5).
#
# Required env:
#   YEAR        Test year (one of 121..128)
#   RUN_DIR     $EXP_DIR/SFNO/$RUN_NUM (the trained run dir; must contain
#               training_checkpoints/best_ckpt.tar via post-train symlink)
#   RUN_NUM     same suffix
#   DATA_DIR    Group data_dir (must have years 11/12 padded per F.B2)
#   INIT_NC     Year-specific init NetCDF (built by F.D)
# Output:
#   $RUN_DIR/inference_full_year${YEAR}/long_rollout_year${YEAR}_member000_y0${YEAR}.nc
#
# CLI contract (verified vs upstream long_inference.py argparse line 1217-1232):
#   - NO --nc_bc_offset (hardcoded to 18 internally at line 1267)
#   - --init_datetime "%Y-%m-%d_%H:%M:%S" with UNDERSCORE between date and time
#   - --final_datetime explicit (line 1347 fallback uses val_year_end which is wrong)
#   - torchrun --standalone (sets WORLD_SIZE=1 properly)

#SBATCH -J sfno_group_li
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 04:00:00
#SBATCH -o logs/sfno_group_li_%j.out
#SBATCH -e logs/sfno_group_li_%j.err

set -euo pipefail
REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
source "$REPO_ROOT/src/sfno_training_group/env_activate.sh"

YEAR="${YEAR:?need YEAR (e.g. 121)}"
RUN_DIR="${RUN_DIR:?need RUN_DIR}"
RUN_NUM="${RUN_NUM:?need RUN_NUM}"
DATA_DIR="${DATA_DIR:?need DATA_DIR}"
INIT_NC="${INIT_NC:?need INIT_NC}"

# Format dates with UNDERSCORE between date and time, zero-padded year width 4.
INIT_DT=$(printf "%04d-01-01_00:00:00" "$YEAR")
NEXT_YEAR=$((YEAR + 1))
FINAL_DT=$(printf "%04d-01-01_00:00:00" "$NEXT_YEAR")

OUT_DIR="$RUN_DIR/inference_full_year${YEAR}"
mkdir -p "$OUT_DIR" "$REPO_ROOT/logs"

echo "[long_inference] YEAR=$YEAR INIT=$INIT_DT FINAL=$FINAL_DT"
echo "[long_inference] RUN_DIR=$RUN_DIR DATA_DIR=$DATA_DIR INIT_NC=$INIT_NC"

# Pre-inference preflight (Checks 11-15 + 18 + 19; runs the offset-everywhere
# enumeration to verify year-11 padded zone idx 1459 (or year-12 idx 1463 for leap)
# files exist).
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m sfno_training_group.tools.preflight_checks \
  --yaml "$RUN_DIR/rendered.yaml" --config SFNO --phase pre_inference_long \
  --run-dir "$RUN_DIR" --init-nc "$INIT_NC" --init-dt "$INIT_DT" \
  --final-dt "$FINAL_DT" --data-dir "$DATA_DIR"

# torchrun --standalone sets WORLD_SIZE=1 properly (long_inference reads it
# at line 1284). NO --nc_bc_offset (hardcoded to 18 internally).
cd "$GROUP_PANGU_ROOT"
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  "$GROUP_PANGU_ROOT/long_inference.py" \
  --config SFNO \
  --yaml_config "$RUN_DIR/rendered.yaml" \
  --run_num "$RUN_NUM" \
  --init_nc_filepaths "$INIT_NC" \
  --init_datetime "$INIT_DT" \
  --final_datetime "$FINAL_DT" \
  --output_dir "$OUT_DIR" \
  --save_basename "long_rollout_year${YEAR}"

# Year-bounded save filename: <save_basename>_member000_y0<YEAR>.nc
EXPECTED_NC="$OUT_DIR/long_rollout_year${YEAR}_member000_y$(printf '%04d' $YEAR).nc"
if [[ ! -f "$EXPECTED_NC" ]]; then
  echo "ERROR: expected NetCDF not found: $EXPECTED_NC"
  echo "Files in $OUT_DIR:"
  find "$OUT_DIR" -maxdepth 1 -type f
  exit 3
fi
echo "[long_inference] success — wrote $EXPECTED_NC ($(stat -c%s "$EXPECTED_NC") bytes)"

# --- Step 2: convert long_inference output to scorer 53-channel NetCDF ---
# Optional — runs if OUT_ROOT and TEST_H5 are set (eval orchestration path).
if [[ -n "${OUT_ROOT:-}" && -n "${TEST_H5:-}" ]]; then
  RUN_TAG="${RUN_TAG:-phaseF_year${YEAR}}"
  EVAL_SHA7="$(cd "$REPO_ROOT" && git rev-parse --short=7 HEAD 2>/dev/null || echo unknown)"
  MAX_OUT_LEADS="${MAX_OUT_LEADS:-60}"
  mkdir -p "$OUT_ROOT/inference/nwp"

  echo "[long_inference] converting to scorer 53-channel NC (K=$MAX_OUT_LEADS)..."
  PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python "$REPO_ROOT/scripts/convert_group_long_inference_to_aires_nc.py" \
    --long-inference-nc "$EXPECTED_NC" \
    --test-h5 "$TEST_H5" \
    --ic-global-idx "${IC_GLOBAL_IDX:-0}" \
    --max-output-leads "$MAX_OUT_LEADS" \
    --ckpt-path "$RUN_DIR/checkpoints/best_ckpt.tar" \
    --eval-sha7 "$EVAL_SHA7" \
    --data-sha7 "v10_zgplev" \
    --train-sha7 "group_full" \
    --run-tag "$RUN_TAG" \
    --rollout-mode "nwp" \
    --out "$OUT_ROOT/inference/nwp/MOST.$(printf '%04d' $YEAR)_ic000.nc"

  echo "[long_inference] converter wrote: $OUT_ROOT/inference/nwp/MOST.$(printf '%04d' $YEAR)_ic000.nc"
fi
