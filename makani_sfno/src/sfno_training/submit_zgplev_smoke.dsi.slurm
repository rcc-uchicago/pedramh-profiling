#!/bin/bash
#
# DSI parallel-copy smoke wrapper. Stampede3 sibling is
# `submit_zgplev_smoke.slurm`; this one runs on UChicago DSI.
# Reference: docs/dsi_smoke_backup_plan.md (v3) §4.F.
#
# Pattern: thin wrapper that exports DSI-specific env vars (CUDA, scratch
# path, mail relay, OUTPUT_ROOT/EXP_DIR) and then `bash`-invokes the
# Stampede3 smoke script. The Stampede3 script's `#SBATCH` lines are
# inert under `bash` — the script body runs normally. No copy-paste, no
# in-place mutation of the Stampede3 sibling.
#
# Required at submit time:
#   sbatch --export=ALL,DSI_SCRATCH=/scratch/$USER \
#       src/sfno_training/submit_zgplev_smoke.dsi.slurm
#
# Substitute placeholders below (`<DSI_GPU_PARTITION>`, `<DSI_ACCOUNT>`,
# `/usr/local/cuda-X.Y`) per phase-1 §3 reconnaissance answers.

#SBATCH -J sfno_zgplev_smoke_dsi
#SBATCH --partition=<DSI_GPU_PARTITION>      # confirm in smoke-plan §3.3
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A <DSI_ACCOUNT>                     # confirm in smoke-plan §3.6, omit if not required
#SBATCH -o logs/sfno_zgplev_smoke_dsi_%j.out
#SBATCH -e logs/sfno_zgplev_smoke_dsi_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# === Choose ONE QoS block based on the smoke-plan §3.5 answer ===
#
# OPTION A (preferred for smoke — short, no preemption):
#SBATCH --qos=protected
#SBATCH --time=02:00:00
#
# OPTION B (longer wall, may be preempted post-2026-05-07; needs requeue):
# #SBATCH --qos=general
# #SBATCH --time=12:00:00
# #SBATCH --signal=B:USR1@300
# #SBATCH --requeue

set -euo pipefail

REPO_ROOT="$HOME/AI-RES"

# Hard-fail fast if DSI scratch isn't set — `set -u` would otherwise expand
# the unset var to empty mid-pipeline. Caller is expected to export
# DSI_SCRATCH from the smoke-plan §3.1 answer (e.g. /scratch/$USER or
# /net/scratch/$USER). No placeholder default — a typo'd default would
# silently route writes to a non-existent path.
: "${DSI_SCRATCH:?Set DSI_SCRATCH from smoke-plan §3.1 (e.g. /scratch/\$USER) before sbatch}"
test -d "$DSI_SCRATCH" || {
    echo "DSI_SCRATCH=$DSI_SCRATCH is not a directory" >&2
    exit 1
}
test -w "$DSI_SCRATCH" || {
    echo "DSI_SCRATCH=$DSI_SCRATCH not writable" >&2
    exit 1
}

# CUDA env (DSI uses /usr/local/cuda-*, not modules — see smoke-plan §3.7):
export CUDA_HOME=/usr/local/cuda-12.4         # SUBSTITUTE per smoke-plan §3.7
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Force the proto dataset (works around Stampede3 script's wrong default at
# submit_zgplev_smoke.slurm:35, which omits the `_proto` suffix). The
# `${VAR:-...}` form lets a caller override via `--export=...,OUTPUT_ROOT=...`.
export OUTPUT_ROOT="${OUTPUT_ROOT:-$DSI_SCRATCH/AI-RES/data/makani/sim52_astro_64x128_zgplev_proto}"
export EXP_DIR="${EXP_DIR:-$DSI_SCRATCH/AI-RES/runs/sfno_zgplev_smoke_dsi}"
export MAIL_TO="zhixingliu@uchicago.edu"

# Override Stampede3 SMTP defaults from slurm_helpers.sh (env-driven, no
# edit needed). Mail is best-effort (smoke-plan §5).
export SFNO_MAIL_FROM="${USER}@uchicago.edu"
export SFNO_SMTP_RELAY="${SFNO_SMTP_RELAY:-smtp.uchicago.edu}"   # confirm with DSI

# Stampede3 script references $SCRATCH only via DEFAULT values for
# OUTPUT_ROOT / EXP_DIR; since this wrapper EXPORTS both above, the
# defaults are unused. Aliasing $SCRATCH would only obscure mistakes,
# so we don't.

mkdir -p "$REPO_ROOT/logs" "$EXP_DIR"

# Run the original Stampede3 smoke script as plain bash. Its `#SBATCH`
# directives are comments to bash; the script body runs normally.
# (`exec` would discard any traps installed in this wrapper — see
# full-plan §7. Smoke wrapper has no traps today, but stay consistent
# with the full wrapper.)
bash "$REPO_ROOT/src/sfno_training/submit_zgplev_smoke.slurm"
