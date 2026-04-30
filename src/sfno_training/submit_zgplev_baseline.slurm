#!/bin/bash

# v10 zg sigma → zg_plev migration. v9 sibling stays in place for the
# frozen contract; see docs/plasim_zg_plev_migration_plan.md (v7).
#SBATCH -J sfno_zgplev_baseline
#SBATCH -p gh                       # Stampede3 GH200 partition
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=4
#SBATCH -t 24:00:00
#SBATCH -o logs/sfno_zgplev_baseline_%j.out
#SBATCH -e logs/sfno_zgplev_baseline_%j.err

# ============================================================================
# Production training run. Do NOT launch until submit_smoke.slurm has
# completed cleanly (plan v9 hard gate).
#
# Inference is OUT OF SCOPE — see src/sfno_training/README.md.
# ============================================================================

# Required — fill in for your run. Either point at the packager-rendered
# YAML directly, or render src/sfno_training/config/plasim_sim52_zgplev_baseline.yaml
# via envsubst-style substitution before this script runs.
OUTPUT_ROOT=            # e.g. $SCRATCH/AI-RES/data/makani/sim52_astro_64x128_zgplev
EXP_DIR=                # e.g. $SCRATCH/AI-RES/runs/sfno_zgplev_baseline
CONFIG_NAME="plasim_sim52_zgplev_baseline"

# Activate the project venv (torch + makani + physicsnemo).
source "$HOME/AI-RES/.venv/bin/activate"

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the packager output dir}"
: "${EXP_DIR:?Set EXP_DIR to the experiment output dir}"

# Prefer the packager-rendered YAML if present; fall back to rendering our
# baseline template.
PACKAGER_RENDERED="$OUTPUT_ROOT/config/${CONFIG_NAME}.yaml"
if [[ -f "$PACKAGER_RENDERED" ]]; then
    YAML_CONFIG="$PACKAGER_RENDERED"
else
    BASELINE_TPL="$HOME/AI-RES/src/sfno_training/config/plasim_sim52_zgplev_baseline.yaml"
    YAML_CONFIG="$EXP_DIR/${CONFIG_NAME}.rendered.yaml"
    mkdir -p "$EXP_DIR"
    sed -e "s|{{OUTPUT_ROOT}}|$OUTPUT_ROOT|g" \
        -e "s|{{EXP_DIR}}|$EXP_DIR|g" \
        "$BASELINE_TPL" > "$YAML_CONFIG"
fi

cd "$HOME/AI-RES"
mkdir -p logs

set -euo pipefail
set -x

# Multi-GPU launch via torchrun (one process per visible GPU).
torchrun --nproc_per_node=4 -m sfno_training.train_plasim \
    --yaml_config "$YAML_CONFIG" \
    --config "$CONFIG_NAME" \
    --run_num 0 \
    --multistep_count 1 \
    --amp_mode bf16

echo "training run complete: $EXP_DIR/$CONFIG_NAME/0/"
