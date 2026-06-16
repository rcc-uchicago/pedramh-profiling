#!/bin/bash

# v11 dataset @ GB=32, single-knob LR probe: lr=1.13e-3 (×√2 from current
# winner lr=8e-4). Conservative next-rung probe; sister of lr=1.6e-3 (×2).
# All other knobs locked to the v11_gb32 control: betas=(0.9, 0.999),
# input_noise.sigma=0.05, weight_decay=3e-6, 50 epochs, 5-epoch warmup +
# cosine to 1e-8. Clip safety net optimizer_max_grad_norm=32.0 carried over
# from lr=8e-4. NO warmstart — single continuous schedule.
# See src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_gb32_lr1p13e3.yaml.

#SBATCH -J sfno_zgplev_group_clone_v11_gb32_lr1p13e3
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
# GB=32 v11_gb32 control wallclock = 6h 16m (50 epochs at ~7m 25s/epoch).
# Identical per-step compute here (lr does not affect step time). 10 h
# budget = 60% headroom over expected.
#SBATCH -t 10:00:00
#SBATCH -o logs/sfno_zgplev_group_clone_v11_gb32_lr1p13e3_%j.out
#SBATCH -e logs/sfno_zgplev_group_clone_v11_gb32_lr1p13e3_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=c561-004

# ============================================================================
# Required env vars (defaults match the v11 zgplev subset):
#   OUTPUT_ROOT — packager-output (or symlink subset) root.
#                 Default: $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full_v11
#   EXP_DIR     — experiment output dir. **Use a fresh dir** for this run.
#                 Default: $SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_v11_gb32_lr1p13e3
# ============================================================================

set -euo pipefail

REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
SFNO_JOB_NAME="sfno_zgplev_group_clone_v11_gb32_lr1p13e3"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full_v11}"
EXP_DIR="${EXP_DIR:-$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_v11_gb32_lr1p13e3}"
MAIL_TO="${MAIL_TO:-zhixingliu@uchicago.edu}"
source "$REPO_ROOT/src/sfno_training/slurm_helpers.sh"
sfno_install_status_trap

source "$REPO_ROOT/.venv/bin/activate"
source "$REPO_ROOT/src/sfno_training/env_nccl.sh"

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the dataset (subset) root}"
: "${EXP_DIR:?Set EXP_DIR to the experiment output dir}"

# Refuse to write into a directory that already has a checkpoint or run/0
# tree (memory: protect-prior-runs). Set RESUME=1 to bypass when intentionally
# continuing a prior run.
RUN0_DIR="$EXP_DIR/plasim_sim52_zgplev_group_clone_v11_gb32_lr1p13e3/0"
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
FULL_TPL="$REPO_ROOT/src/sfno_training/config/plasim_sim52_zgplev_group_clone_v11_gb32_lr1p13e3.yaml"
FULL_YAML="$EXP_DIR/plasim_sim52_zgplev_group_clone_v11_gb32_lr1p13e3.rendered.yaml"
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
    --config plasim_sim52_zgplev_group_clone_v11_gb32_lr1p13e3 \
    --template "$FULL_TPL" \
    --amp-mode bf16 \
    --checkpointing-level 2 \
    --log "$EXP_DIR/preflight_log.txt"

# 2) Train. world_size=4, GB=32 from yaml → per-rank=8 via
#    train_plasim._resolve_batch_sizes.
RESUME_CKPT_DIR="$EXP_DIR/plasim_sim52_zgplev_group_clone_v11_gb32_lr1p13e3/0/training_checkpoints"
RESUME_SENTINEL="$RESUME_CKPT_DIR/ckpt_mp0_v0.tar"
if [ -f "$RESUME_SENTINEL" ]; then
    LATEST_CKPT="$(ls -t "$RESUME_CKPT_DIR"/ckpt_mp0_v*.tar 2>/dev/null | head -1)"
    echo "INFO resume: ckpt_mp0_v0.tar present — trainer will resume from latest by mtime: $LATEST_CKPT"
    ls -lt "$RESUME_CKPT_DIR"/ckpt_mp0_v*.tar
else
    echo "INFO resume: no ckpt_mp0_v0.tar at $RESUME_CKPT_DIR — starting from scratch"
fi
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m sfno_training.train_plasim \
    --yaml_config "$FULL_YAML" \
    --config plasim_sim52_zgplev_group_clone_v11_gb32_lr1p13e3 \
    --run_num 0 \
    --multistep_count 1 \
    --amp_mode bf16 \
    --checkpointing_level 2

# 3) Post-training NaN/Inf scan (no-op if missing).
if [ -f scripts/scan_for_nans.py ]; then
    python scripts/scan_for_nans.py \
        --exp-dir "$EXP_DIR/plasim_sim52_zgplev_group_clone_v11_gb32_lr1p13e3/0" \
        --slurm-logs "logs/sfno_zgplev_group_clone_v11_gb32_lr1p13e3_${SLURM_JOB_ID}.out" "logs/sfno_zgplev_group_clone_v11_gb32_lr1p13e3_${SLURM_JOB_ID}.err" \
        || true
fi

echo "group clone v11 complete: $EXP_DIR/plasim_sim52_zgplev_group_clone_v11_gb32_lr1p13e3/0/"
