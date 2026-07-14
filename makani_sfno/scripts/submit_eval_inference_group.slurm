#!/bin/bash
#SBATCH -J sfno_eval_inf_group
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 01:00:00
#SBATCH -o logs/sfno_eval_inf_group_%j.out
#SBATCH -e logs/sfno_eval_inf_group_%j.err

# Phase E.1+E.2 combined: run inference smoke via the score-function wrapper,
# then convert the wrapper output to a 53-channel scorer-compatible NetCDF
# under $OUT_ROOT/inference/nwp/.
#
# Required env (defaults shown):
#   DATA_DIR        $SCRATCH/SFNO_Climate_Emulator/data/group_sfno/sim52_smoke
#   EXP_DIR         $SCRATCH/SFNO_Climate_Emulator/runs/sfno_group_sigma10_smoke
#   RUN_NUM         (required — train smoke run dir suffix)
#   INIT_NC         $EXP_DIR/init_smoke.nc
#   INIT_DT         "0121-01-01 00:00:00"
#   STEPS           8
#   OUT_ROOT        (required — where to write inference/<mode>/*.nc)
#   TEST_H5         $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev/test/MOST.0121.h5
#   IC_GLOBAL_IDX   0
#   RUN_TAG         (required — passed through to NetCDF attrs)

set -euo pipefail
REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"

# Step 1: group env for inference smoke.
source "$REPO_ROOT/src/sfno_training_group/env_activate.sh"

DATA_DIR="${DATA_DIR:-$SCRATCH/SFNO_Climate_Emulator/data/group_sfno/sim52_smoke}"
EXP_DIR="${EXP_DIR:-$SCRATCH/SFNO_Climate_Emulator/runs/sfno_group_sigma10_smoke}"
RUN_NUM="${RUN_NUM:?need RUN_NUM}"
RUN_DIR="$EXP_DIR/SFNO/$RUN_NUM"
INIT_NC="${INIT_NC:-$EXP_DIR/init_smoke.nc}"
INIT_DT="${INIT_DT:-0121-01-01 00:00:00}"
STEPS="${STEPS:-8}"
OUT_ROOT="${OUT_ROOT:?need OUT_ROOT}"
TEST_H5="${TEST_H5:-$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev/test/MOST.0121.h5}"
IC_GLOBAL_IDX="${IC_GLOBAL_IDX:-0}"
RUN_TAG="${RUN_TAG:?need RUN_TAG}"

mkdir -p "$OUT_ROOT/inference/nwp"

# Pre-inference preflight (checks 11-15).
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m sfno_training_group.tools.preflight_checks \
  --yaml "$RUN_DIR/rendered.yaml" --config SFNO --phase pre_inference \
  --run-dir "$RUN_DIR" --init-nc "$INIT_NC" --init-dt "$INIT_DT" \
  --data-dir "$DATA_DIR" --steps "$STEPS"

# Step 1: rollout via wrapper.
ROLLOUT_NC="$RUN_DIR/inference_smoke/score_fn_rollout.nc"
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m sfno_training_group.score_function.run_smoke_rollout \
  --ckpt "$RUN_DIR/checkpoints/best_ckpt.tar" \
  --yaml "$RUN_DIR/rendered.yaml" --config SFNO \
  --init-nc "$INIT_NC" \
  --boundary-data-dir "$DATA_DIR" \
  --init-dt "$INIT_DT" \
  --steps "$STEPS" \
  --out "$ROLLOUT_NC"

echo "[eval_inf_group] rollout written: $ROLLOUT_NC"

# Step 2: converter uses only h5py + numpy + xarray, which are also in the
# group env (verified). Switching envs is fragile because conda deactivate
# doesn't fully unset LD_LIBRARY_PATH, so we just keep the group env active.

EVAL_SHA7="$(cd "$REPO_ROOT" && git rev-parse --short=7 HEAD || echo unknown)"

PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python "$REPO_ROOT/scripts/convert_group_inference_to_aires_nc.py" \
  --rollout-nc "$ROLLOUT_NC" \
  --test-h5 "$TEST_H5" \
  --ic-global-idx "$IC_GLOBAL_IDX" \
  --ckpt-path "$RUN_DIR/checkpoints/best_ckpt.tar" \
  --eval-sha7 "$EVAL_SHA7" \
  --data-sha7 "v10_zgplev" \
  --train-sha7 "group_smoke" \
  --run-tag "$RUN_TAG" \
  --rollout-mode "nwp" \
  --out "$OUT_ROOT/inference/nwp/MOST.0121_ic000.nc"

echo "[eval_inf_group] converter wrote: $OUT_ROOT/inference/nwp/MOST.0121_ic000.nc"
