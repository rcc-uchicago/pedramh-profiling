#!/bin/bash

# Full-config DDP microbenchmark — I2 decision step for the DDP
# throughput fix (docs/2026-05-05_ddp_throughput_fix_plan.md §I2).
#
# Runs the production architecture (embed_dim=256, num_layers=12,
# scale_factor=1) for 2 epochs at a chosen GLOBAL batch size on a fresh
# EXP_DIR, with validation skipped, EMA enabled (production-shaped
# optimizer-step path), and checkpoint write disabled. Per-epoch sample
# budget is set to ≈1500 steady-state training steps so the measured
# window has signal; epoch 1 is treated as warmup (JIT, cudnn-benchmark
# autotune, DataLoader prefetch fill, NCCL bootstrap) and epoch 2 is
# the measurement window. At ~60 ms/step → ~3-5 min wall-clock per
# point including init.
#
# Required env:
#   GB           — global batch size to test (e.g. 32, 16, or 8).
# Optional env:
#   OUTPUT_ROOT  — packaged dataset root.
#                  Default: $SCRATCH/AI-RES/data/makani/sim52_zgplev_full
#   EXP_DIR      — fresh per-point experiment dir.
#                  Default: $SCRATCH/AI-RES/runs/sfno_zgplev_full_microbench/gb${GB}
#   TARGET_STEPS — steady-state training steps per epoch.
#                  Default: 1500
#   MAX_EPOCHS   — number of epochs to run (1 warmup + 1 measured).
#                  Default: 2
#
# Per-point pass criteria (I2):
#   * No OOM, no NCCL hang, no NaN/Inf in loss or grad norm.
#   * memory footprint ≤ 65 GB on H100 (≥15 GB safety margin).
#   * samples/sec ≥ 1.7× the GB=4 single-GPU/4-rank baseline.
#
# Stability check (per-point smoke; not a substitute for a real
# scan_for_nans.py):
#   grep -E "loss=nan|loss=inf|grad norm=nan|grad norm=inf" out.log
#   grep -E "RuntimeError|CUDA error|NaN" out.log err.log
#   Plus the per-epoch summary's "training loss" / "gradient norm"
#   lines (Makani's structured log_epoch output, not tqdm).
#
#SBATCH -J sfno_zgplev_full_microbench
#SBATCH -p h100                     # Same partition as submit_zgplev_full.slurm
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH -t 00:30:00                 # Single point: ~3-5 min; 2 h budget covers a 3-point sweep.
#SBATCH -o logs/sfno_zgplev_full_microbench_%j.out
#SBATCH -e logs/sfno_zgplev_full_microbench_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

REPO_ROOT="$HOME/AI-RES"
SFNO_JOB_NAME="sfno_zgplev_full_microbench"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/AI-RES/data/makani/sim52_zgplev_full}"
GB="${GB:?Set GB to the global batch size to microbench (e.g. 32, 16, 8)}"
EXP_DIR="${EXP_DIR:-$SCRATCH/AI-RES/runs/sfno_zgplev_full_microbench/gb${GB}}"
TARGET_STEPS="${TARGET_STEPS:-1500}"
MAX_EPOCHS="${MAX_EPOCHS:-2}"
N_TRAIN_SAMPLES=$(( GB * TARGET_STEPS ))
MAIL_TO="${MAIL_TO:-zhixingliu@uchicago.edu}"

source "$REPO_ROOT/src/sfno_training/slurm_helpers.sh"
sfno_install_status_trap

source "$REPO_ROOT/.venv/bin/activate"

# NCCL diagnostics + behavioural async-error abort (P5).
source "$REPO_ROOT/src/sfno_training/env_nccl.sh"

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the packaged dataset root}"

# Refuse to clobber a directory that already holds a checkpoint —
# train_plasim.py:171-176 would auto-resume from prior state and
# silently invalidate the measurement.
RESUME_SENTINEL="$EXP_DIR/plasim_sim52_zgplev_full/0/training_checkpoints/ckpt_mp0_v0.tar"
if [ -f "$RESUME_SENTINEL" ]; then
    echo "ERROR microbench: $EXP_DIR already contains a checkpoint." >&2
    echo "       auto-resume would invalidate the timing measurement." >&2
    echo "       Choose a fresh EXP_DIR or remove the checkpoint." >&2
    exit 1
fi

# Render the full YAML, then overlay the microbench-only knobs.
# YParams loads a single named config block; it does not natively merge
# overlays, so we rewrite the rendered YAML in place per point.
#   max_epochs:                <MAX_EPOCHS>
#   n_train_samples_per_epoch: <GB * TARGET_STEPS>
# (save_checkpoint is set on the CLI below — argparse default "legacy"
# overrides the YAML at train_plasim.py:110, so a YAML overlay alone
# would not take effect.)
FULL_TPL="$REPO_ROOT/src/sfno_training/config/plasim_sim52_zgplev_full.yaml"
FULL_YAML="$EXP_DIR/plasim_sim52_zgplev_full.rendered.yaml"
mkdir -p "$EXP_DIR" "$EXP_DIR/training_checkpoints"
sed -e "s|{{OUTPUT_ROOT}}|$OUTPUT_ROOT|g" \
    -e "s|{{EXP_DIR}}|$EXP_DIR|g" \
    "$FULL_TPL" > "$FULL_YAML"

export FULL_YAML GB N_TRAIN_SAMPLES MAX_EPOCHS

python - <<'PY'
"""Idempotent YAML overlay for the microbench."""
import os
import re

yaml_path = os.environ["FULL_YAML"]
n_train = int(os.environ["N_TRAIN_SAMPLES"])
max_epochs = int(os.environ["MAX_EPOCHS"])

with open(yaml_path) as f:
    text = f.read()


def replace_or_append(text: str, key: str, value) -> str:
    pattern = re.compile(rf"^(\s*){re.escape(key)}\s*:.*$", re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(rf"\g<1>{key}: {value}", text)
    # Append at the end of the (single) config block. The full YAML
    # has exactly one top-level mapping with a 4-space-indented body.
    return text.rstrip() + f"\n    {key}: {value}\n"


text = replace_or_append(text, "max_epochs", max_epochs)
text = replace_or_append(text, "n_train_samples_per_epoch", n_train)

with open(yaml_path, "w") as f:
    f.write(text)

print(
    f"microbench overlay applied → max_epochs={max_epochs}, "
    f"n_train_samples_per_epoch={n_train} (GB={os.environ['GB']}, "
    f"target steps/epoch≈{n_train // int(os.environ['GB'])})"
)
PY

cd "$REPO_ROOT"
mkdir -p logs

sfno_send_status_mail "BEGIN" 0 || true

set -x

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Cap thread fan-out per rank so 4 ranks don't oversubscribe the node.
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-64} / 4 ))

# Train. --skip_validation drops the validation passes (microbench
# measures step time, not loss). --save_checkpoint none disables
# checkpoint writes; the CLI default is "legacy" and would otherwise
# override any YAML setting (train_plasim.py:110).
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m sfno_training.train_plasim \
    --yaml_config "$FULL_YAML" \
    --config plasim_sim52_zgplev_full \
    --run_num 0 \
    --batch_size "$GB" \
    --multistep_count 1 \
    --amp_mode bf16 \
    --checkpointing_level 2 \
    --skip_validation \
    --save_checkpoint none

echo "microbench complete (GB=$GB): $EXP_DIR/plasim_sim52_zgplev_full/0/"
