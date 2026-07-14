#!/bin/bash

# Group-recipe clone — direct 5410 batch-size match (GB=32). Spine is
# v10 base (group_clone) with the single delta `batch_size: 32` (per-rank
# 8 at world_size=4), matching 5410's actual training (4 GPUs × per-rank
# 8). Everything else locked: input perturb sigma=0.05, betas (0.9,
# 0.999), no grad clip, lr=1e-4, 50 epochs, 5-epoch warmup, cosine to
# 1e-8. NO warmstart — single continuous schedule. See
# src/sfno_training/config/plasim_sim52_zgplev_group_clone_gb32.yaml.

#SBATCH -J sfno_zgplev_group_clone_gb32
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
# GB=32 → ~4,550 steps/epoch (50-epoch total ≈ 227,500 optimizer steps),
# 4× fewer than v10 base GB=8. Per-step time will rise (4× per-rank work,
# 2 → 8) but not 4×; net wallclock should land well below the 17 h GB=8
# budget. Budget 12 h to leave headroom for the larger per-step compute
# without blowing past Stampede3's typical h100 queue limits.
#SBATCH -t 12:00:00
#SBATCH -o logs/sfno_zgplev_group_clone_gb32_%j.out
#SBATCH -e logs/sfno_zgplev_group_clone_gb32_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=c561-004

# ============================================================================
# Required env vars (defaults match the full v10 zgplev subset):
#   OUTPUT_ROOT — packager-output (or symlink subset) root.
#                 Default: $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full
#   EXP_DIR     — experiment output dir. **Use a fresh dir** (memory:
#                 feedback_protect_prior_runs.md — never resume into a prior
#                 production run dir).
#                 Default: $SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_gb32
# ============================================================================

set -euo pipefail

REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
SFNO_JOB_NAME="sfno_zgplev_group_clone_gb32"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full}"
EXP_DIR="${EXP_DIR:-$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_group_clone_gb32}"
MAIL_TO="${MAIL_TO:-zhixingliu@uchicago.edu}"
source "$REPO_ROOT/src/sfno_training/slurm_helpers.sh"
sfno_install_status_trap

source "$REPO_ROOT/.venv/bin/activate"
source "$REPO_ROOT/src/sfno_training/env_nccl.sh"

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the dataset (subset) root}"
: "${EXP_DIR:?Set EXP_DIR to the experiment output dir}"

# Refuse to write into a directory that already has a checkpoint or run/0
# tree (memory: feedback_protect_prior_runs.md). Set RESUME=1 to bypass
# this guard when intentionally continuing a prior run; the resume block
# below picks up epoch/iters/optimizer state from the latest ckpt_mp0_v*.tar.
RUN0_DIR="$EXP_DIR/plasim_sim52_zgplev_group_clone_gb32/0"
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
FULL_TPL="$REPO_ROOT/src/sfno_training/config/plasim_sim52_zgplev_group_clone_gb32.yaml"
FULL_YAML="$EXP_DIR/plasim_sim52_zgplev_group_clone_gb32.rendered.yaml"
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
    --config plasim_sim52_zgplev_group_clone_gb32 \
    --template "$FULL_TPL" \
    --amp-mode bf16 \
    --checkpointing-level 2 \
    --log "$EXP_DIR/preflight_log.txt"

# 2) Train. world_size=4, GB=32 from yaml → per-rank=8 via
#    train_plasim._resolve_batch_sizes.
RESUME_CKPT_DIR="$EXP_DIR/plasim_sim52_zgplev_group_clone_gb32/0/training_checkpoints"
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
    --config plasim_sim52_zgplev_group_clone_gb32 \
    --run_num 0 \
    --multistep_count 1 \
    --amp_mode bf16 \
    --checkpointing_level 2

# 3) Post-training NaN/Inf scan (no-op if missing).
if [ -f scripts/scan_for_nans.py ]; then
    python scripts/scan_for_nans.py \
        --exp-dir "$EXP_DIR/plasim_sim52_zgplev_group_clone_gb32/0" \
        --slurm-logs "logs/sfno_zgplev_group_clone_gb32_${SLURM_JOB_ID}.out" "logs/sfno_zgplev_group_clone_gb32_${SLURM_JOB_ID}.err" \
        || true
fi

echo "group clone gb32 complete: $EXP_DIR/plasim_sim52_zgplev_group_clone_gb32/0/"
