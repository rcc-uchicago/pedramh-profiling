#!/bin/bash

# Warm-start continuation diagnostic for the own-track v11_clip SFNO emulator.
# Single-knob delta vs submit_zgplev_group_clone_v11_clip.slurm:
#   --pretrained_checkpoint_path "$PRETRAINED_CKPT"
# pointing at the v11_clip raw-best ckpt. The recipe (lr, schedule, EMA,
# input_noise, grad clip, batch_size, max_epochs) is unchanged. Optimizer +
# scheduler + counters start fresh; only model_state is loaded from the
# pretrained ckpt. See docs/2026-05-14_v11_clip_warmstart_continuation_plan.md.

#SBATCH -J sfno_zgplev_group_clone_v11_clip_warmstart
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
# Same budget as v11_clip; warm-start is identical compute per step.
#SBATCH -t 18:30:00
#SBATCH -o logs/sfno_zgplev_group_clone_v11_clip_warmstart_%j.out
#SBATCH -e logs/sfno_zgplev_group_clone_v11_clip_warmstart_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=c561-004

# ============================================================================
# Required env vars (defaults match the v11 zgplev subset):
#   OUTPUT_ROOT     — packager-output (or symlink subset) root.
#                     Default: $SCRATCH/AI-RES/data/makani/sim52_zgplev_full_v11
#   EXP_DIR         — experiment output dir. **Use a fresh dir** for this run.
#                     Default: $SCRATCH/AI-RES/runs/sfno_zgplev_group_clone_v11_clip_warmstart
#   PRETRAINED_CKPT — Absolute path to the legacy raw-best ckpt to warm-start
#                     from. Default: v11_clip /0 best_ckpt_mp0.tar.
# ============================================================================

set -euo pipefail

REPO_ROOT="$HOME/AI-RES"
SFNO_JOB_NAME="sfno_zgplev_group_clone_v11_clip_warmstart"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/AI-RES/data/makani/sim52_zgplev_full_v11}"
EXP_DIR="${EXP_DIR:-$SCRATCH/AI-RES/runs/sfno_zgplev_group_clone_v11_clip_warmstart}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-$SCRATCH/AI-RES/runs/sfno_zgplev_group_clone_v11_clip/plasim_sim52_zgplev_group_clone_v11_clip/0/training_checkpoints/best_ckpt_mp0.tar}"
MAIL_TO="${MAIL_TO:-zhixingliu@uchicago.edu}"
source "$REPO_ROOT/src/sfno_training/slurm_helpers.sh"
sfno_install_status_trap

source "$REPO_ROOT/.venv/bin/activate"
source "$REPO_ROOT/src/sfno_training/env_nccl.sh"

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the dataset (subset) root}"
: "${EXP_DIR:?Set EXP_DIR to the experiment output dir}"
: "${PRETRAINED_CKPT:?Set PRETRAINED_CKPT to the warm-start source ckpt}"

if [[ ! -s "$PRETRAINED_CKPT" ]]; then
    echo "FATAL: PRETRAINED_CKPT not readable or empty: $PRETRAINED_CKPT" >&2
    exit 2
fi

# Refuse to write into a directory that already has a checkpoint or run/0
# tree (memory: protect-prior-runs). Set RESUME=1 to bypass when intentionally
# continuing a prior warm-start run.
RUN0_DIR="$EXP_DIR/plasim_sim52_zgplev_group_clone_v11_clip_warmstart/0"
RESUME="${RESUME:-0}"
if [[ -d "$RUN0_DIR/training_checkpoints" ]] && \
   ls "$RUN0_DIR/training_checkpoints"/ckpt_mp0_v*.tar 2>/dev/null | grep -q .; then
    if [[ "$RESUME" != "1" ]]; then
        echo "FATAL: $RUN0_DIR already has checkpoints — refuse to overwrite. Use a fresh EXP_DIR, or set RESUME=1 to resume." >&2
        exit 2
    fi
    echo "INFO: RESUME=1 — bypassing the protect-prior-runs guard; will resume in-place at $RUN0_DIR" >&2
fi

# Render the full YAML with concrete paths.
FULL_TPL="$REPO_ROOT/src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_clip_warmstart.yaml"
FULL_YAML="$EXP_DIR/plasim_sim52_zgplev_group_clone_v11_clip_warmstart.rendered.yaml"
mkdir -p "$EXP_DIR" "$EXP_DIR/training_checkpoints"
sed -e "s|{{OUTPUT_ROOT}}|$OUTPUT_ROOT|g" \
    -e "s|{{EXP_DIR}}|$EXP_DIR|g" \
    "$FULL_TPL" > "$FULL_YAML"

cd "$REPO_ROOT"
mkdir -p logs

sfno_send_status_mail "BEGIN" 0 || true

set -x

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-64} / 4 ))
python -c "import sfno_training; print('sfno_training import:', sfno_training.__file__)"

# 1) Preflight (same bf16 + checkpointing_level=2 as training).
python scripts/preflight.py \
    --yaml_config "$FULL_YAML" \
    --config plasim_sim52_zgplev_group_clone_v11_clip_warmstart \
    --template "$FULL_TPL" \
    --amp-mode bf16 \
    --checkpointing-level 2 \
    --log "$EXP_DIR/preflight_log.txt"

# 2) Train. world_size=4, GB=8 from yaml → per-rank=2.
#    Warm-start loads model_state ONLY from $PRETRAINED_CKPT; optimizer +
#    scheduler + counters remain fresh (trainer-side patch in
#    src/sfno_training/trainer/plasim_trainer.py).
RESUME_CKPT_DIR="$EXP_DIR/plasim_sim52_zgplev_group_clone_v11_clip_warmstart/0/training_checkpoints"
RESUME_SENTINEL="$RESUME_CKPT_DIR/ckpt_mp0_v0.tar"
if [ -f "$RESUME_SENTINEL" ]; then
    LATEST_CKPT="$(ls -t "$RESUME_CKPT_DIR"/ckpt_mp0_v*.tar 2>/dev/null | head -1)"
    echo "INFO resume: ckpt_mp0_v0.tar present — trainer will resume from latest by mtime: $LATEST_CKPT"
    ls -lt "$RESUME_CKPT_DIR"/ckpt_mp0_v*.tar
else
    echo "INFO warm-start: pretrained_checkpoint_path = $PRETRAINED_CKPT"
fi
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m sfno_training.train_plasim \
    --yaml_config "$FULL_YAML" \
    --config plasim_sim52_zgplev_group_clone_v11_clip_warmstart \
    --run_num 0 \
    --multistep_count 1 \
    --amp_mode bf16 \
    --pretrained_checkpoint_path "$PRETRAINED_CKPT" \
    --checkpointing_level 2

# 3) Post-training NaN/Inf scan (no-op if missing).
if [ -f scripts/scan_for_nans.py ]; then
    python scripts/scan_for_nans.py \
        --exp-dir "$EXP_DIR/plasim_sim52_zgplev_group_clone_v11_clip_warmstart/0" \
        --slurm-logs "logs/sfno_zgplev_group_clone_v11_clip_warmstart_${SLURM_JOB_ID}.out" "logs/sfno_zgplev_group_clone_v11_clip_warmstart_${SLURM_JOB_ID}.err" \
        || true
fi

echo "group clone v11_clip warmstart complete: $EXP_DIR/plasim_sim52_zgplev_group_clone_v11_clip_warmstart/0/"
