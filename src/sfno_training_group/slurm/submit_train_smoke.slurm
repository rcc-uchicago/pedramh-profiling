#!/bin/bash
#SBATCH -J sfno_group_smoke
#SBATCH -p h100               # single H100 (Stampede3 h100 partition is 1 GPU/node)
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00
#SBATCH -o logs/sfno_group_smoke_%j.out
#SBATCH -e logs/sfno_group_smoke_%j.err

# Phase D smoke train. Inputs (env or default):
#   DATA_DIR  default $SCRATCH/SFNO_Climate_Emulator/data/group_sfno/sim52_smoke
#   EXP_DIR   default $SCRATCH/SFNO_Climate_Emulator/runs/sfno_group_sigma10_smoke
#   RUN_NUM   default smoke_<timestamp>
# Output:
#   $EXP_DIR/SFNO/$RUN_NUM/checkpoints/best_ckpt.tar  (training writes here)
#   $EXP_DIR/SFNO/$RUN_NUM/training_checkpoints      (post-train symlink for long_inference)

set -euo pipefail
REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"

source "$REPO_ROOT/src/sfno_training_group/env_activate.sh"

DATA_DIR="${DATA_DIR:-$SCRATCH/SFNO_Climate_Emulator/data/group_sfno/sim52_smoke}"
EXP_DIR="${EXP_DIR:-$SCRATCH/SFNO_Climate_Emulator/runs/sfno_group_sigma10_smoke}"
RUN_NUM="${RUN_NUM:-smoke_$(date +%Y%m%d_%H%M)}"
RUN_DIR="$EXP_DIR/SFNO/$RUN_NUM"

mkdir -p "$RUN_DIR" "$REPO_ROOT/logs"

SMOKE_TPL="$REPO_ROOT/src/sfno_training_group/config/plasim_sim52_sigma10_sfno_smoke.yaml"
RENDERED="$RUN_DIR/rendered.yaml"

# Render YAML (substitutes {{DATA_DIR}}, {{EXP_DIR}}, end-exclusive train/val date strings).
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m sfno_training_group.tools.render_yaml \
  --tpl "$SMOKE_TPL" --out "$RENDERED" \
  --data-dir "$DATA_DIR" --exp-dir "$EXP_DIR" \
  --manifest "$DATA_DIR/_v10_calendar_manifest.json" \
  --train-years 12 13 \
  --val-years 11

# Pre-train preflight (checks 1-10).
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m sfno_training_group.tools.preflight_checks \
  --yaml "$RENDERED" --config SFNO --phase pre_train \
  --data-dir "$DATA_DIR" --manifest "$DATA_DIR/_v10_calendar_manifest.json"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# Single-GPU smoke: torchrun --standalone --nproc_per_node=1 sets WORLD_SIZE=1
# without fanning out DDP, so train.py initializes a 1-rank dist group.
cd "$GROUP_PANGU_ROOT"
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  "$GROUP_PANGU_ROOT/train.py" \
  --config SFNO \
  --yaml_config "$RENDERED" \
  --run_num "$RUN_NUM" \
  --fresh_start

# Post-train: bridge ckpt dir for long_inference (Phase F). Train.py writes
# <run_dir>/checkpoints/, long_inference reads <run_dir>/training_checkpoints/.
if [[ -d "$RUN_DIR/checkpoints" && ! -e "$RUN_DIR/training_checkpoints" ]]; then
  ln -sfn "$RUN_DIR/checkpoints" "$RUN_DIR/training_checkpoints"
  echo "[smoke] Bridged checkpoints/ -> training_checkpoints/ at $RUN_DIR"
fi

# With max_epochs=1, group's trainer saves ckpt_latest.tar but no best_ckpt.tar
# (it tracks "best" across epochs, so 1-epoch runs have nothing to compare).
# The wrapper + preflight expect best_ckpt.tar, so fall back to a symlink.
CKPTS="$RUN_DIR/checkpoints"
if [[ ! -e "$CKPTS/best_ckpt.tar" && -e "$CKPTS/ckpt_latest.tar" ]]; then
  ln -sfn ckpt_latest.tar "$CKPTS/best_ckpt.tar"
  echo "[smoke] No best_ckpt.tar; symlinked best_ckpt.tar -> ckpt_latest.tar"
fi

echo "[smoke] Done. Checkpoint: $RUN_DIR/checkpoints/best_ckpt.tar"
