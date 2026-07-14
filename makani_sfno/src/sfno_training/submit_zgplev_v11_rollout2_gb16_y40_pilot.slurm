#!/bin/bash

# v11 rollout2 GB16 40y screening pilot — 2026-05-14.
#
# Trains plasim_sim52_zgplev_v11_rollout2_gb16_y40_pilot.yaml on the
# v11 40-yr symlink farm sim52_zgplev_40yr_y12-51_v11_20260514. Single
# delta vs gbhpo40_gb16_lr2_0e-4: --multistep_count 2 (and n_future: 1
# in the yaml). Everything else (lr=2e-4, GB=16, β₂=0.999, no clip,
# σ=0.05 input noise, EMA 0.999, 50 epochs) is identical to the v11
# no-clip recipe at GB16 scaling, so the pilot isolates the multistep
# axis cleanly.
#
# See docs/2026-05-14_v11_rollout2_gb16_y40_pilot_plan.md.
#
# Wallclock estimate:
#   - gbhpo40 GB16 40y (multistep=1) ran in ~4.3 h at 64.7 ms/step.
#   - multistep=2 ≈ 1.4× per-epoch with bf16 + checkpointing_level=2.
#   - Budget 8 h to absorb preflight + overhead + extra unrolled forward.

#SBATCH -J sfno_zgplev_v11_rollout2_gb16_y40_pilot
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH -t 08:00:00
#SBATCH -o logs/sfno_zgplev_v11_rollout2_gb16_y40_pilot_%j.out
#SBATCH -e logs/sfno_zgplev_v11_rollout2_gb16_y40_pilot_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=c561-004

# ============================================================================
# Required env vars (defaults match the v11 40y plan):
#   OUTPUT_ROOT — v11 40-yr symlink farm root.
#                 Default: $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_40yr_y12-51_v11_20260514
#   EXP_DIR     — experiment output dir. **Use a fresh dir.**
#                 Default: $SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_v11_rollout2_gb16_y40_pilot_20260514
# ============================================================================

set -euo pipefail

REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
SFNO_JOB_NAME="sfno_zgplev_v11_rollout2_gb16_y40_pilot"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_40yr_y12-51_v11_20260514}"
EXP_DIR="${EXP_DIR:-$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_v11_rollout2_gb16_y40_pilot_20260514}"
MAIL_TO="${MAIL_TO:-zhixingliu@uchicago.edu}"
source "$REPO_ROOT/src/sfno_training/slurm_helpers.sh"
sfno_install_status_trap

source "$REPO_ROOT/.venv/bin/activate"
source "$REPO_ROOT/src/sfno_training/env_nccl.sh"

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the dataset (subset) root}"
: "${EXP_DIR:?Set EXP_DIR to the experiment output dir}"

# Refuse to write into a directory that already has a checkpoint or run/0 tree
# (memory: feedback_protect_prior_runs.md). Set RESUME=1 to bypass.
RUN0_DIR="$EXP_DIR/plasim_sim52_zgplev_v11_rollout2_gb16_y40_pilot/0"
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
FULL_TPL="$REPO_ROOT/src/sfno_training/config/plasim_sim52_zgplev_v11_rollout2_gb16_y40_pilot.yaml"
FULL_YAML="$EXP_DIR/plasim_sim52_zgplev_v11_rollout2_gb16_y40_pilot.rendered.yaml"
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

# 1) Preflight — same bf16 + checkpointing_level=2 AS WELL AS multistep_count=2,
#    so the memory probe exercises the actual training graph (not the
#    multistep=1 stub the legacy preflight defaults to).
python scripts/preflight.py \
    --yaml_config "$FULL_YAML" \
    --config plasim_sim52_zgplev_v11_rollout2_gb16_y40_pilot \
    --template "$FULL_TPL" \
    --amp-mode bf16 \
    --checkpointing-level 2 \
    --multistep-count 2 \
    --log "$EXP_DIR/preflight_log.txt"

# 2) Train. world_size=4, GB=16 from yaml → per-rank=4. multistep_count=2
#    → two-step rollout loss; n_future=1 in yaml.
RESUME_CKPT_DIR="$EXP_DIR/plasim_sim52_zgplev_v11_rollout2_gb16_y40_pilot/0/training_checkpoints"
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
    --config plasim_sim52_zgplev_v11_rollout2_gb16_y40_pilot \
    --run_num 0 \
    --multistep_count 2 \
    --amp_mode bf16 \
    --checkpointing_level 2

# 3) Post-training NaN/Inf scan (no-op if missing).
if [ -f scripts/scan_for_nans.py ]; then
    python scripts/scan_for_nans.py \
        --exp-dir "$EXP_DIR/plasim_sim52_zgplev_v11_rollout2_gb16_y40_pilot/0" \
        --slurm-logs "logs/sfno_zgplev_v11_rollout2_gb16_y40_pilot_${SLURM_JOB_ID}.out" "logs/sfno_zgplev_v11_rollout2_gb16_y40_pilot_${SLURM_JOB_ID}.err" \
        || true
fi

echo "v11 rollout2 GB16 40y pilot complete: $EXP_DIR/plasim_sim52_zgplev_v11_rollout2_gb16_y40_pilot/0/"
