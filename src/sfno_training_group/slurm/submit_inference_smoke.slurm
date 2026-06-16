#!/bin/bash
#SBATCH -J sfno_group_inf_smoke
#SBATCH -p h100
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 00:30:00
#SBATCH -o logs/sfno_group_inf_smoke_%j.out
#SBATCH -e logs/sfno_group_inf_smoke_%j.err

# Phase E.1 inference smoke. Inputs (env or default):
#   DATA_DIR  default $SCRATCH/SFNO_Climate_Emulator/data/group_sfno/sim52_smoke
#   EXP_DIR   default $SCRATCH/SFNO_Climate_Emulator/runs/sfno_group_sigma10_smoke
#   RUN_NUM   required (the smoke train run number, e.g. smoke_20260509_2300)
#   INIT_NC   default $EXP_DIR/init_smoke.nc
#   INIT_DT   default "0121-01-01 00:00:00"
#   STEPS     default 8

set -euo pipefail
REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
source "$REPO_ROOT/src/sfno_training_group/env_activate.sh"

DATA_DIR="${DATA_DIR:-$SCRATCH/SFNO_Climate_Emulator/data/group_sfno/sim52_smoke}"
EXP_DIR="${EXP_DIR:-$SCRATCH/SFNO_Climate_Emulator/runs/sfno_group_sigma10_smoke}"
RUN_NUM="${RUN_NUM:?need RUN_NUM (the train smoke run dir suffix)}"
RUN_DIR="$EXP_DIR/SFNO/$RUN_NUM"
INIT_NC="${INIT_NC:-$EXP_DIR/init_smoke.nc}"
INIT_DT="${INIT_DT:-0121-01-01 00:00:00}"
STEPS="${STEPS:-8}"

# Pre-inference preflight (checks 11-15).
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m sfno_training_group.tools.preflight_checks \
  --yaml "$RUN_DIR/rendered.yaml" --config SFNO --phase pre_inference \
  --run-dir "$RUN_DIR" --init-nc "$INIT_NC" --init-dt "$INIT_DT" \
  --data-dir "$DATA_DIR" --steps "$STEPS"

# Run smoke rollout.
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m sfno_training_group.score_function.run_smoke_rollout \
  --ckpt "$RUN_DIR/checkpoints/best_ckpt.tar" \
  --yaml "$RUN_DIR/rendered.yaml" --config SFNO \
  --init-nc "$INIT_NC" \
  --boundary-data-dir "$DATA_DIR" \
  --init-dt "$INIT_DT" \
  --steps "$STEPS" \
  --out "$RUN_DIR/inference_smoke/score_fn_rollout.nc"

echo "[smoke] Inference done: $RUN_DIR/inference_smoke/score_fn_rollout.nc"
