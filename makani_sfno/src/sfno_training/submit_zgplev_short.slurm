#!/bin/bash

# v10 zg sigma → zg_plev migration. v9 sibling stays in place for the
# frozen contract; see docs/plasim_zg_plev_migration_plan.md (v7).
#SBATCH -J sfno_zgplev_short
#SBATCH -p amd-rtx                  # Stampede3 AMD+RTX GPU partition
#SBATCH -N 1
#SBATCH -n 1
# Stampede3 GPU partitions do not advertise GPU GRES; -p amd-rtx selects GPU nodes.
#SBATCH -t 06:00:00
#SBATCH -o logs/sfno_zgplev_short_%j.out
#SBATCH -e logs/sfno_zgplev_short_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Short training launch (docs/sfno_tiny_short_training_plan.md Part B).
#
# Runs preflight (scripts/preflight.py) before training. The default subset is:
#   train: MOST.0003.h5 ... MOST.0007.h5
#   valid: MOST.0101.h5 ... MOST.0102.h5
#
# Build the subset one time before launching:
#
#   scripts/build_subset_dataset.py \
#       --src $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128 \
#       --dst $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_short \
#       --train-years 3-7 --valid-years 101-102
# ============================================================================

set -euo pipefail

REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
SFNO_JOB_NAME="sfno_zgplev_short"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_short}"
EXP_DIR="${EXP_DIR:-$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_short}"
MAIL_TO="${MAIL_TO:-zhixingliu@uchicago.edu}"
source "$REPO_ROOT/src/sfno_training/slurm_helpers.sh"
sfno_install_status_trap

# Activate the project venv (must include torch + makani + physicsnemo).
source "$REPO_ROOT/.venv/bin/activate"

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the dataset (subset) root}"
: "${EXP_DIR:?Set EXP_DIR to the experiment output dir}"

# Render the short YAML with concrete paths.
SHORT_TPL="$REPO_ROOT/src/sfno_training/config/plasim_sim52_zgplev_short.yaml"
SHORT_YAML="$EXP_DIR/plasim_sim52_zgplev_short.rendered.yaml"
mkdir -p "$EXP_DIR" "$EXP_DIR/training_checkpoints"
sed -e "s|{{OUTPUT_ROOT}}|$OUTPUT_ROOT|g" \
    -e "s|{{EXP_DIR}}|$EXP_DIR|g" \
    "$SHORT_TPL" > "$SHORT_YAML"

cd "$REPO_ROOT"
mkdir -p logs

sfno_send_status_mail "BEGIN" 0 || true

set -x

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -c "import sfno_training; print('sfno_training import:', sfno_training.__file__)"

# 1) Preflight (scripts/preflight.py) - exits non-zero on any failure.
python scripts/preflight.py \
    --yaml_config "$SHORT_YAML" \
    --config plasim_sim52_zgplev_short \
    --template "$SHORT_TPL" \
    --amp-mode bf16 \
    --log "$EXP_DIR/preflight_log.txt"

# 2) Train.
python -m sfno_training.train_plasim \
    --yaml_config "$SHORT_YAML" \
    --config plasim_sim52_zgplev_short \
    --run_num 0 \
    --batch_size 4 \
    --multistep_count 1 \
    --amp_mode bf16 \
    --disable_ddp

# 3) Post-training NaN/Inf scan (harmless if missing).
if [ -f scripts/scan_for_nans.py ]; then
    python scripts/scan_for_nans.py \
        --exp-dir "$EXP_DIR/plasim_sim52_zgplev_short/0" \
        --slurm-logs "logs/sfno_zgplev_short_${SLURM_JOB_ID}.out" "logs/sfno_zgplev_short_${SLURM_JOB_ID}.err" \
        || true
fi

echo "short complete: $EXP_DIR/plasim_sim52_zgplev_short/0/"
