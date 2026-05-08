#!/usr/bin/env bash
# I1 short-config DDP smoke sweep
# (docs/2026-05-05_ddp_throughput_fix_plan.md §I1).
#
# Submits one sbatch job per global-batch point, each with a fresh
# EXP_DIR so train_plasim.py's auto-resume (train_plasim.py:171-176)
# cannot contaminate timings between points. The slurm script reads
# the GB env var and threads it into --batch_size on torchrun.
#
# Usage:
#   scripts/run_zgplev_short_ddp_sweep.sh                # GB ∈ {4,8,16,32}
#   scripts/run_zgplev_short_ddp_sweep.sh "4 16"         # explicit subset
#
# Env overrides (optional):
#   REPO_ROOT     — default $HOME/AI-RES
#   OUTPUT_ROOT   — packaged short subset; default
#                   $SCRATCH/AI-RES/data/makani/sim52_zgplev_short
#   SWEEP_ROOT    — parent dir for per-GB EXP_DIRs; default
#                   $SCRATCH/AI-RES/runs/sfno_zgplev_short_ddp_sweep
#   SLURM_SCRIPT  — path to submit_zgplev_short_ddp.slurm
#
# Each point is launched as an independent SLURM job; they queue and run
# in parallel up to the partition limit. Per-point pass criterion (I1):
# ≥ 2 epochs complete, no OOM, no NCCL hang, no NaN.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/AI-RES}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/AI-RES/data/makani/sim52_zgplev_short}"
SWEEP_ROOT="${SWEEP_ROOT:-$SCRATCH/AI-RES/runs/sfno_zgplev_short_ddp_sweep}"
SLURM_SCRIPT="${SLURM_SCRIPT:-$REPO_ROOT/src/sfno_training/submit_zgplev_short_ddp.slurm}"

GB_LIST="${1:-4 8 16 32}"

if [ ! -d "$OUTPUT_ROOT" ]; then
    echo "ERROR: OUTPUT_ROOT=$OUTPUT_ROOT does not exist." >&2
    exit 1
fi
if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "ERROR: SLURM_SCRIPT=$SLURM_SCRIPT not found." >&2
    exit 1
fi

mkdir -p "$SWEEP_ROOT" "$REPO_ROOT/logs"

for GB in $GB_LIST; do
    EXP_DIR="$SWEEP_ROOT/gb${GB}"
    CKPT_DIR="$EXP_DIR/plasim_sim52_zgplev_short/0/training_checkpoints"
    if compgen -G "$CKPT_DIR/ckpt_mp0_v*.tar" > /dev/null; then
        echo "WARN gb=${GB}: $EXP_DIR already contains checkpoints — auto-resume would contaminate timing. Skipping."
        continue
    fi
    mkdir -p "$EXP_DIR"
    echo "submitting gb=${GB} → EXP_DIR=$EXP_DIR"
    EXP_DIR="$EXP_DIR" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    GB="$GB" \
        sbatch -J "sfno_zgplev_short_ddp_gb${GB}" "$SLURM_SCRIPT"
done

echo
echo "All sweep points submitted. Per-point out.log:"
for GB in $GB_LIST; do
    echo "  $SWEEP_ROOT/gb${GB}/plasim_sim52_zgplev_short/0/out.log"
done
