#!/bin/bash
#
# DSI one-epoch full-data gate. HARD GATE before any multi-window full
# training on DSI. Reference: docs/dsi_full_training_plan.md (v4) §8.4.
#
# What this measures: epoch wall on DSI's GPU using the **full** dataset,
# **full** SFNO config, bf16 + checkpointing_level=2 — the actual
# multi-window training path. Smoke (proto data, smoke YAML) doesn't
# measure this; without it we don't know whether one epoch fits in DSI's
# 12 h `general` window.
#
# Gate criterion: one full epoch + validation + checkpoint write
# completes well under 12 h (≤ 8 h leaves ~4 h headroom for jitter and
# validation). If it doesn't fit, do NOT launch full training on DSI
# until either batch size / amp / checkpointing is tuned to fit, or DSI
# grants higher walltime QoS.
#
# Implementation note: train_plasim.py does NOT accept --max_epochs as
# a CLI flag (verified: argument_parser.py has no such argument). The
# override goes via the rendered YAML — pre-render with max_epochs:1
# and scheduler_T_max:1 via sed before invoking the trainer.
#
# Required at submit time:
#   export DSI_PROJECT=/project/<your-group>/$USER
#   sbatch --export=ALL,DSI_PROJECT="$DSI_PROJECT" \
#       src/sfno_training/submit_zgplev_full.dsi_1epoch.slurm

#SBATCH -J sfno_zgplev_full_dsi_1epoch_gate
#SBATCH --partition=<DSI_GPU_PARTITION>
#SBATCH --gres=gpu:a100:1                    # confirm exact syntax in full-plan §5.2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A <DSI_ACCOUNT>
#SBATCH --time=12:00:00                      # DSI general cap
#SBATCH --qos=general
# NOTE: --requeue intentionally NOT SET. This is a one-shot timing gate;
# silent requeue would mask "doesn't fit in 12 h" — we want to SEE the
# timeout. Multi-window full training in submit_zgplev_full.dsi.slurm
# enables --requeue.
#SBATCH -o logs/sfno_zgplev_full_dsi_1epoch_%j.out
#SBATCH -e logs/sfno_zgplev_full_dsi_1epoch_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

REPO_ROOT="$HOME/AI-RES"

# DSI_PROJECT must be exported by caller — same contract as the
# multi-window full wrapper.
: "${DSI_PROJECT:?export DSI_PROJECT (e.g. /project/<group>/\$USER) before sbatch — see full-plan §5.1}"
test -d "$DSI_PROJECT" || {
    echo "DSI_PROJECT=$DSI_PROJECT is not a directory" >&2
    exit 1
}
test -w "$DSI_PROJECT" || {
    echo "DSI_PROJECT=$DSI_PROJECT not writable" >&2
    exit 1
}

# CUDA env (DSI uses /usr/local/cuda-*, not modules):
export CUDA_HOME=/usr/local/cuda-12.4         # SUBSTITUTE per phase-1 §3.7
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Default placement — caller can override OUTPUT_ROOT/EXP_DIR via
# `sbatch --export=ALL,EXP_DIR=...,OUTPUT_ROOT=...`. EXP_DIR uses a
# distinct `_1epoch_gate` suffix so the gate doesn't overwrite the
# multi-window EXP_DIR or pollute its checkpoint history.
export OUTPUT_ROOT="${OUTPUT_ROOT:-$DSI_PROJECT/AI-RES/data/makani/sim52_zgplev_full}"
export EXP_DIR="${EXP_DIR:-$DSI_PROJECT/AI-RES/runs/sfno_zgplev_full_dsi_1epoch_gate}"

mkdir -p "$REPO_ROOT/logs" "$EXP_DIR" "$EXP_DIR/training_checkpoints"

# Activate the project venv (must include torch + makani + physicsnemo).
source "$REPO_ROOT/.venv/bin/activate"

cd "$REPO_ROOT"

# Pre-render YAML with max_epochs=1 + scheduler_T_max=1 in addition to
# the standard {{OUTPUT_ROOT}} / {{EXP_DIR}} substitution. Both lines
# in the source YAML are at module indentation (4 spaces) per
# config/plasim_sim52_zgplev_full.yaml:110,121.
FULL_TPL="$REPO_ROOT/src/sfno_training/config/plasim_sim52_zgplev_full.yaml"
FULL_YAML="$EXP_DIR/plasim_sim52_zgplev_full.rendered.yaml"
sed -e "s|{{OUTPUT_ROOT}}|$OUTPUT_ROOT|g" \
    -e "s|{{EXP_DIR}}|$EXP_DIR|g" \
    -e "s|^[[:space:]]*max_epochs:.*|    max_epochs: 1|" \
    -e "s|^[[:space:]]*scheduler_T_max:.*|    scheduler_T_max: 1|" \
    "$FULL_TPL" > "$FULL_YAML"

# Sanity-check the override actually landed before launching:
grep -E '^[[:space:]]*(max_epochs|scheduler_T_max):' "$FULL_YAML"
# Expect: `    max_epochs: 1` and `    scheduler_T_max: 1`. If not,
# the regex didn't match and the gate would silently run the full
# 50-epoch schedule.
test "$(grep -cE '^[[:space:]]*max_epochs:[[:space:]]*1[[:space:]]*$' "$FULL_YAML")" -ge 1 || {
    echo "max_epochs override did not land in $FULL_YAML" >&2
    exit 1
}
test "$(grep -cE '^[[:space:]]*scheduler_T_max:[[:space:]]*1[[:space:]]*$' "$FULL_YAML")" -ge 1 || {
    echo "scheduler_T_max override did not land in $FULL_YAML" >&2
    exit 1
}

set -x

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -c "import sfno_training; print('sfno_training import:', sfno_training.__file__)"

# Train one epoch on the full dataset / full config. CLI flags match
# submit_zgplev_full.slurm:90-100. Skip preflight (memory probe is
# orthogonal to the wall-time gate; preflight is also already covered
# by the multi-window wrapper that runs after this gate passes).
python -m sfno_training.train_plasim \
    --yaml_config "$FULL_YAML" \
    --config plasim_sim52_zgplev_full \
    --run_num 0 \
    --batch_size 4 \
    --multistep_count 1 \
    --amp_mode bf16 \
    --checkpointing_level 2 \
    --disable_ddp

set +x

# Post-run wall-time summary so the gate criterion is visible without
# digging through the trainer log:
echo "=== one-epoch gate complete: $EXP_DIR/plasim_sim52_zgplev_full/0/ ==="
echo "Job runtime (squeue / sacct will report exact value):"
echo "  Started:  ${SLURM_JOB_START_TIME:-unknown}"
echo "  Ended:    $(date +%s)"
echo "Latest checkpoint:"
ls -lat "$EXP_DIR"/plasim_sim52_zgplev_full/0/training_checkpoints/ 2>/dev/null | head -5 || true
echo
echo "Gate criterion: total wall must be well under 12 h (≤ 8 h preferred)."
echo "If close to 12 h, do NOT launch submit_zgplev_full.dsi.slurm without"
echo "tuning batch_size / checkpointing / GPU class first — see full-plan §8.4."
