#!/bin/bash

# GB16 @ lr=2.83e-4, 40-yr empirical probe for the gbhpo40 LR sweep.
#
# Trains plasim_sim52_zgplev_gbhpo40_gb16_lr2_83e-4.yaml on the 40-yr symlink
# farm sim52_zgplev_40yr_y12-51_20260511. lr=2.83e-4 sits ABOVE the linear-
# scaling endpoint from the GB8 group_clone anchor (linear would be 2e-4 at
# GB=16) and has no theoretical motivation under common large-batch scaling
# arguments. Explicitly user-authorised on 2026-05-11 as an empirical probe
# despite the theoretical objection. All other knobs match group_clone:
# β₂=0.999, no grad clip, input_noise σ=0.05, EMA decay 0.999, 50 epochs,
# 5-epoch warmup.

#SBATCH -J sfno_zgplev_gbhpo40_gb16_lr283e4
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
# Wallclock estimate (extrapolated from GB=16 prod-knob 100-yr run at
# 64.7 ms/step, 247 samples/s): 100-yr was ~10.7 h for 50 epochs; 40-yr
# is ~4.3 h training only. Budget 6 h to absorb overhead.
#SBATCH -t 06:00:00
#SBATCH -o logs/sfno_zgplev_gbhpo40_gb16_lr283e4_%j.out
#SBATCH -e logs/sfno_zgplev_gbhpo40_gb16_lr283e4_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=c561-004

# ============================================================================
# Required env vars (defaults match the gbhpo40 plan):
#   OUTPUT_ROOT — 40-yr subset root.
#                 Default: $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_40yr_y12-51_20260511
#   EXP_DIR     — experiment output dir. **Use a fresh dir** (do not point at
#                 the GB4 / GB16 / GB32 / group_clone prior runs).
#                 Default: $SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_gbhpo40_gb16_lr2_83e-4_20260511
# ============================================================================

set -euo pipefail

REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
SFNO_JOB_NAME="sfno_zgplev_gbhpo40_gb16_lr283e4"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_40yr_y12-51_20260511}"
EXP_DIR="${EXP_DIR:-$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_gbhpo40_gb16_lr2_83e-4_20260511}"
MAIL_TO="${MAIL_TO:-zhixingliu@uchicago.edu}"
source "$REPO_ROOT/src/sfno_training/slurm_helpers.sh"
sfno_install_status_trap

source "$REPO_ROOT/.venv/bin/activate"
source "$REPO_ROOT/src/sfno_training/env_nccl.sh"

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the dataset (subset) root}"
: "${EXP_DIR:?Set EXP_DIR to the experiment output dir}"

# Refuse to write into a directory that already has a checkpoint or run/0 tree
# (memory: feedback_protect_prior_runs.md). Set RESUME=1 to bypass.
RUN0_DIR="$EXP_DIR/plasim_sim52_zgplev_gbhpo40_gb16_lr2_83e-4/0"
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
FULL_TPL="$REPO_ROOT/src/sfno_training/config/plasim_sim52_zgplev_gbhpo40_gb16_lr2_83e-4.yaml"
FULL_YAML="$EXP_DIR/plasim_sim52_zgplev_gbhpo40_gb16_lr2_83e-4.rendered.yaml"
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
    --config plasim_sim52_zgplev_gbhpo40_gb16_lr2_83e-4 \
    --template "$FULL_TPL" \
    --amp-mode bf16 \
    --checkpointing-level 2 \
    --log "$EXP_DIR/preflight_log.txt"

# 2) Train. world_size=4, GB=16 from yaml → per-rank=4.
RESUME_CKPT_DIR="$EXP_DIR/plasim_sim52_zgplev_gbhpo40_gb16_lr2_83e-4/0/training_checkpoints"
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
    --config plasim_sim52_zgplev_gbhpo40_gb16_lr2_83e-4 \
    --run_num 0 \
    --multistep_count 1 \
    --amp_mode bf16 \
    --checkpointing_level 2

# 3) Post-training NaN/Inf scan (no-op if missing).
if [ -f scripts/scan_for_nans.py ]; then
    python scripts/scan_for_nans.py \
        --exp-dir "$EXP_DIR/plasim_sim52_zgplev_gbhpo40_gb16_lr2_83e-4/0" \
        --slurm-logs "logs/sfno_zgplev_gbhpo40_gb16_lr283e4_${SLURM_JOB_ID}.out" "logs/sfno_zgplev_gbhpo40_gb16_lr283e4_${SLURM_JOB_ID}.err" \
        || true
fi

echo "gbhpo40 gb16 lr2.83e-4 probe complete: $EXP_DIR/plasim_sim52_zgplev_gbhpo40_gb16_lr2_83e-4/0/"
