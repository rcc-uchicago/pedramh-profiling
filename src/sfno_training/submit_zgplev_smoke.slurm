#!/bin/bash

# v10 zg sigma → zg_plev migration. v9 sibling stays in place for the
# frozen contract; see docs/plasim_zg_plev_migration_plan.md (v7).
#SBATCH -J sfno_zgplev_smoke
#SBATCH -p amd-rtx                  # Stampede3 AMD+RTX GPU partition
#SBATCH -N 1
#SBATCH -n 1
# Stampede3 GPU partitions do not advertise GPU GRES; -p amd-rtx selects GPU nodes.
#SBATCH -t 00:30:00
#SBATCH -o logs/sfno_zgplev_smoke_%j.out
#SBATCH -e logs/sfno_zgplev_smoke_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# GPU smoke (HARD GATE — plan v9): tiny SFNO, 1 epoch, 1 file.
#
# Required: a packager-output dataset directory. After this completes
# cleanly (loss finite, checkpoint written, reloadable) you may launch
# `submit_train.slurm` for full training.
#
# Inference is OUT OF SCOPE — see src/sfno_training/README.md.
# ============================================================================

set -euo pipefail

REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
SFNO_JOB_NAME="sfno_zgplev_smoke"
MAIL_TO="${MAIL_TO:-zhixingliu@uchicago.edu}"
source "$REPO_ROOT/src/sfno_training/slurm_helpers.sh"
sfno_install_status_trap

# Required — overridable via env (sbatch propagates exported env by default).
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev}"
EXP_DIR="${EXP_DIR:-$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_smoke}"

# Activate the project venv (must include torch + makani + physicsnemo).
source "$REPO_ROOT/.venv/bin/activate"

# Fail fast if unset.
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the packager output dir}"
: "${EXP_DIR:?Set EXP_DIR to the experiment output dir}"

# Render the smoke YAML with concrete paths.
SMOKE_TPL="$REPO_ROOT/src/sfno_training/config/plasim_sim52_zgplev_smoke.yaml"
SMOKE_YAML="$EXP_DIR/plasim_sim52_zgplev_smoke.rendered.yaml"
mkdir -p "$EXP_DIR" "$EXP_DIR/training_checkpoints"
sed -e "s|{{OUTPUT_ROOT}}|$OUTPUT_ROOT|g" \
    -e "s|{{EXP_DIR}}|$EXP_DIR|g" \
    "$SMOKE_TPL" > "$SMOKE_YAML"

cd "$REPO_ROOT"
mkdir -p logs

sfno_send_status_mail "BEGIN" 0 || true

set -x

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -c "import sfno_training; print('sfno_training import:', sfno_training.__file__)"

python -m sfno_training.train_plasim \
    --yaml_config "$SMOKE_YAML" \
    --config plasim_sim52_zgplev_smoke \
    --run_num 0 \
    --batch_size 1 \
    --multistep_count 1 \
    --amp_mode bf16 \
    --disable_ddp

echo "smoke complete: $EXP_DIR/plasim_sim52_zgplev_smoke/0/"
