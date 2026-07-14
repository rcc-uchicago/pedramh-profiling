#!/bin/bash
#SBATCH -J sfno_tiny
#SBATCH -p amd-rtx                  # Stampede3 AMD+RTX GPU partition
#SBATCH -N 1
#SBATCH -n 1
# Stampede3 GPU partitions do not advertise GPU GRES; -p amd-rtx selects GPU nodes.
#SBATCH -t 00:30:00
#SBATCH -o logs/sfno_tiny_%j.out
#SBATCH -e logs/sfno_tiny_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Tiny training launch (docs/sfno_tiny_short_training_plan.md §A).
#
# Runs preflight (scripts/preflight.py) before training. Preflight does:
#   1. makani import path under makani-src/
#   2. re-run rollout sentinel tests in the launch venv
#   3. single-batch 58/53 contract dry-run + forcing-from-truth assertion
#   4. print resolved sizes (len(dataset), len(dataloader), max_epochs, ...)
#   5. YAML diff vs unrendered template (no leftover {{...}})
#
# Required env vars (defaults match the planned subset layout):
#   OUTPUT_ROOT — packager-output root (or symlink subset root). Default:
#                 $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_tiny
#   EXP_DIR     — experiment output dir. Default:
#                 $SCRATCH/SFNO_Climate_Emulator/runs/sfno_tiny
#
# Build the subset (one-time) before launching:
#
#   scripts/build_subset_dataset.py \
#       --src $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128 \
#       --dst $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_tiny \
#       --train-years 3 --valid-years 101
# ============================================================================

set -euo pipefail

REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
SFNO_JOB_NAME="sfno_tiny"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_tiny}"
EXP_DIR="${EXP_DIR:-$SCRATCH/SFNO_Climate_Emulator/runs/sfno_tiny}"
MAIL_TO="${MAIL_TO:-zhixingliu@uchicago.edu}"
source "$REPO_ROOT/src/sfno_training/slurm_helpers.sh"
sfno_install_status_trap

# Activate the project venv (must include torch + makani + physicsnemo).
source "$REPO_ROOT/.venv/bin/activate"

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the dataset (subset) root}"
: "${EXP_DIR:?Set EXP_DIR to the experiment output dir}"

# Render the tiny YAML with concrete paths.
TINY_TPL="$REPO_ROOT/src/sfno_training/config/plasim_sim52_tiny.yaml"
TINY_YAML="$EXP_DIR/plasim_sim52_tiny.rendered.yaml"
mkdir -p "$EXP_DIR" "$EXP_DIR/training_checkpoints"
sed -e "s|{{OUTPUT_ROOT}}|$OUTPUT_ROOT|g" \
    -e "s|{{EXP_DIR}}|$EXP_DIR|g" \
    "$TINY_TPL" > "$TINY_YAML"

cd "$REPO_ROOT"
mkdir -p logs

sfno_send_status_mail "BEGIN" 0 || true

set -x

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -c "import sfno_training; print('sfno_training import:', sfno_training.__file__)"

# 1) Preflight (scripts/preflight.py) — exits non-zero on any failure.
python scripts/preflight.py \
    --yaml_config "$TINY_YAML" \
    --config plasim_sim52_tiny \
    --template "$TINY_TPL" \
    --log "$EXP_DIR/preflight_log.txt"

# 2) Train.
python -m sfno_training.train_plasim \
    --yaml_config "$TINY_YAML" \
    --config plasim_sim52_tiny \
    --run_num 0 \
    --batch_size 2 \
    --multistep_count 1 \
    --amp_mode bf16 \
    --disable_ddp

# 3) Post-training NaN/Inf scan (added in chunk 2; harmless if missing).
if [ -f scripts/scan_for_nans.py ]; then
    python scripts/scan_for_nans.py \
        --exp-dir "$EXP_DIR/plasim_sim52_tiny/0" \
        --slurm-logs "logs/sfno_tiny_${SLURM_JOB_ID}.out" "logs/sfno_tiny_${SLURM_JOB_ID}.err" \
        || true
fi

echo "tiny complete: $EXP_DIR/plasim_sim52_tiny/0/"
