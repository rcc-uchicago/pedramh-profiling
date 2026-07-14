#!/bin/bash

# Smoke for plasim_sim52_zgplev_group_clone — verifies that the
# group-recipe knobs (input_noise perturb, betas (0.9, 0.999), grad clip
# off, lr=1e-4, target=state) all wire up cleanly with the FULL SFNO
# architecture. 1 epoch, 4 train + 4 eval samples, batch=1, single GPU.
# Should complete in <5 minutes.
#
# This is a hard gate: do NOT submit submit_zgplev_group_clone.slurm
# until this smoke completes with a finite loss and a sane one-step
# output range.

#SBATCH -J sfno_zgplev_group_clone_smoke
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 00:30:00
#SBATCH -o logs/sfno_zgplev_group_clone_smoke_%j.out
#SBATCH -e logs/sfno_zgplev_group_clone_smoke_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Required env vars (defaults match the v10 zgplev full subset, which is
# fine for a 4-sample smoke):
#   OUTPUT_ROOT — packager-output (or symlink subset) root.
#                 Default: $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full
#   EXP_DIR     — experiment output dir; **fresh dir** (the smoke writes
#                 a small ckpt + log; do not point at any production run).
#                 Default: $SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_smoke
# ============================================================================

set -euo pipefail

REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
SFNO_JOB_NAME="sfno_zgplev_group_clone_smoke"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full}"
EXP_DIR="${EXP_DIR:-$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_smoke}"
MAIL_TO="${MAIL_TO:-zhixingliu@uchicago.edu}"
source "$REPO_ROOT/src/sfno_training/slurm_helpers.sh"
sfno_install_status_trap

source "$REPO_ROOT/.venv/bin/activate"
source "$REPO_ROOT/src/sfno_training/env_nccl.sh"

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the dataset (subset) root}"
: "${EXP_DIR:?Set EXP_DIR to the experiment output dir}"

# Refuse to overwrite a previous smoke (memory: feedback_protect_prior_runs.md).
RUN0_DIR="$EXP_DIR/plasim_sim52_zgplev_group_clone_smoke/0"
if [[ -d "$RUN0_DIR/training_checkpoints" ]] && \
   ls "$RUN0_DIR/training_checkpoints"/ckpt_mp0_v*.tar 2>/dev/null | grep -q .; then
    echo "FATAL: $RUN0_DIR already has checkpoints — refuse to overwrite. Use a fresh EXP_DIR." >&2
    exit 2
fi

SMOKE_TPL="$REPO_ROOT/src/sfno_training/config/plasim_sim52_zgplev_group_clone_smoke.yaml"
SMOKE_YAML="$EXP_DIR/plasim_sim52_zgplev_group_clone_smoke.rendered.yaml"
mkdir -p "$EXP_DIR" "$EXP_DIR/training_checkpoints"
sed -e "s|{{OUTPUT_ROOT}}|$OUTPUT_ROOT|g" \
    -e "s|{{EXP_DIR}}|$EXP_DIR|g" \
    "$SMOKE_TPL" > "$SMOKE_YAML"

cd "$REPO_ROOT"
mkdir -p logs

sfno_send_status_mail "BEGIN" 0 || true

set -x

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-16} ))
python -c "import sfno_training; print('sfno_training import:', sfno_training.__file__)"

# Single GPU: torchrun --nproc_per_node=1.
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    -m sfno_training.train_plasim \
    --yaml_config "$SMOKE_YAML" \
    --config plasim_sim52_zgplev_group_clone_smoke \
    --run_num 0 \
    --multistep_count 1 \
    --amp_mode bf16 \
    --checkpointing_level 2

echo "smoke complete: $EXP_DIR/plasim_sim52_zgplev_group_clone_smoke/0/"
