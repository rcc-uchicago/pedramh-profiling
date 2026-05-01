#!/bin/bash

# v10 zg sigma → zg_plev migration. v9 sibling stays in place for the
# frozen contract; see docs/plasim_zg_plev_migration_plan.md (v7).
#SBATCH -J sfno_zgplev_full
#SBATCH -p h100                     # Stampede3 H100 partition
#SBATCH -N 1
#SBATCH -n 1
# Stampede3's h100 partition does not advertise GPU GRES; -p h100 selects GPU nodes.
#SBATCH -t 47:30:00
#SBATCH -o logs/sfno_zgplev_full_%j.out
#SBATCH -e logs/sfno_zgplev_full_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Full SFNO emulator training (docs/sfno_full_training_plan.md v1.1).
#
# Year ranges (group convention, half-open):
#   train: 12-111  (100 yrs)
#   valid: 11      (1 yr)
#   no test split
#
# Resubmitting picks up from the latest checkpoint
# (train_plasim.py:110-115 detects a checkpoint via load_checkpoint=legacy
#  default at argument_parser.py:61).
#
# Required env vars (defaults match the planned subset layout):
#   OUTPUT_ROOT — packager-output (or symlink subset) root.
#                 Default: $SCRATCH/AI-RES/data/makani/sim52_zgplev_full
#   EXP_DIR     — experiment output dir.
#                 Default: $SCRATCH/AI-RES/runs/sfno_zgplev_full
#
# Build the subset (one-time) before launching. --src is the v10 zgplev
# packaged root produced by plasim_makani_packager (NOT the v9 sigma-level
# sim52_astro_64x128/, which was the previous example here):
#
#   scripts/build_subset_dataset.py \
#       --src $SCRATCH/AI-RES/data/makani/sim52_astro_64x128_zgplev \
#       --dst $SCRATCH/AI-RES/data/makani/sim52_zgplev_full \
#       --train-years 12-111 \
#       --valid-years 11
# ============================================================================

set -euo pipefail

REPO_ROOT="$HOME/AI-RES"
SFNO_JOB_NAME="sfno_zgplev_full"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/AI-RES/data/makani/sim52_zgplev_full}"
EXP_DIR="${EXP_DIR:-$SCRATCH/AI-RES/runs/sfno_zgplev_full}"
MAIL_TO="${MAIL_TO:-zhixingliu@uchicago.edu}"
source "$REPO_ROOT/src/sfno_training/slurm_helpers.sh"
sfno_install_status_trap

# Activate the project venv (must include torch + makani + physicsnemo).
source "$REPO_ROOT/.venv/bin/activate"

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the dataset (subset) root}"
: "${EXP_DIR:?Set EXP_DIR to the experiment output dir}"

# Render the full YAML with concrete paths.
FULL_TPL="$REPO_ROOT/src/sfno_training/config/plasim_sim52_zgplev_full.yaml"
FULL_YAML="$EXP_DIR/plasim_sim52_zgplev_full.rendered.yaml"
mkdir -p "$EXP_DIR" "$EXP_DIR/training_checkpoints"
sed -e "s|{{OUTPUT_ROOT}}|$OUTPUT_ROOT|g" \
    -e "s|{{EXP_DIR}}|$EXP_DIR|g" \
    "$FULL_TPL" > "$FULL_YAML"

cd "$REPO_ROOT"
mkdir -p logs

sfno_send_status_mail "BEGIN" 0 || true

set -x

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -c "import sfno_training; print('sfno_training import:', sfno_training.__file__)"

# 1) Preflight — same bf16 + checkpointing_level=2 as training so the
#    memory probe represents the actual training path
#    (docs/sfno_full_training_plan.md §A.2).
python scripts/preflight.py \
    --yaml_config "$FULL_YAML" \
    --config plasim_sim52_zgplev_full \
    --template "$FULL_TPL" \
    --amp-mode bf16 \
    --checkpointing-level 2 \
    --log "$EXP_DIR/preflight_log.txt"

# 2) Train. --checkpointing_level 2 enables activation checkpointing
#    (default 0 in argument_parser.py:41; group convention is 2).
python -m sfno_training.train_plasim \
    --yaml_config "$FULL_YAML" \
    --config plasim_sim52_zgplev_full \
    --run_num 0 \
    --batch_size 4 \
    --multistep_count 1 \
    --amp_mode bf16 \
    --checkpointing_level 2 \
    --disable_ddp

# 3) Post-training NaN/Inf scan (chunk 2 deliverable; harmless if missing).
if [ -f scripts/scan_for_nans.py ]; then
    python scripts/scan_for_nans.py \
        --exp-dir "$EXP_DIR/plasim_sim52_zgplev_full/0" \
        --slurm-logs "logs/sfno_zgplev_full_${SLURM_JOB_ID}.out" "logs/sfno_zgplev_full_${SLURM_JOB_ID}.err" \
        || true
fi

echo "full complete: $EXP_DIR/plasim_sim52_zgplev_full/0/"
