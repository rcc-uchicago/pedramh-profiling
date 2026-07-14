#!/bin/bash
#
# DSI parallel-copy full-training wrapper. Stampede3 sibling is
# `submit_zgplev_full.slurm`; this one runs on UChicago DSI.
# Reference: docs/dsi_full_training_plan.md (v4) §7.
#
# Phase relationship: this depends on phase 1 (smoke plan) having
# succeeded — env, repo clone, and DSI infrastructure answers are
# inherited from there.
#
# Required at submit time:
#   export DSI_PROJECT=/project/<your-group>/$USER   # NOT a placeholder
#   sbatch --export=ALL,DSI_PROJECT="$DSI_PROJECT" \
#       src/sfno_training/submit_zgplev_full.dsi.slurm
#
# Substitute placeholders (`<DSI_GPU_PARTITION>`, `<DSI_ACCOUNT>`, the
# CUDA toolkit version) per phase-1 §3 + phase-2 §5 reconnaissance.

#SBATCH -J sfno_zgplev_full_dsi
#SBATCH --partition=<DSI_GPU_PARTITION>
#SBATCH --gres=gpu:a100:1                    # confirm exact syntax in full-plan §5.2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A <DSI_ACCOUNT>
#SBATCH --time=12:00:00                      # DSI general cap; tune per QoS
#SBATCH --qos=general                        # OR --qos=protected if available + sufficient
# NOTE: --signal=B:USR1@300 is INTENTIONALLY NOT SET. Today, Python's
# default SIGUSR1 disposition terminates immediately, so the signal would
# kill the trainer 5 min before SLURM would anyway, with no chance to
# flush a checkpoint. Re-enable only after a SIGUSR1 handler lands in
# train_plasim.py (Mitigation B in full-plan §8.3).
#SBATCH --requeue                            # auto-resubmit on preempt
#SBATCH -o logs/sfno_zgplev_full_dsi_%j.out
#SBATCH -e logs/sfno_zgplev_full_dsi_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

set -euo pipefail

REPO_ROOT="$HOME/AI-RES"

# DSI_PROJECT must be set from outside; no placeholder default — a typo'd
# /project/<group>/$USER would silently route writes to a non-existent
# path. Caller is expected to `export DSI_PROJECT=...` before sbatch
# (and pass it through via --export=ALL,DSI_PROJECT="$DSI_PROJECT").
: "${DSI_PROJECT:?export DSI_PROJECT (e.g. /project/<group>/\$USER) before sbatch — see full-plan §5.1}"
test -d "$DSI_PROJECT" || {
    echo "DSI_PROJECT=$DSI_PROJECT is not a directory" >&2
    exit 1
}
test -w "$DSI_PROJECT" || {
    echo "DSI_PROJECT=$DSI_PROJECT not writable" >&2
    exit 1
}
# DSI_SCRATCH not needed: per full-plan §5.1, data + EXP_DIR live on
# project storage (50 GB scratch default is too small for ~340 GB data).

# CUDA env (DSI uses /usr/local/cuda-*, not modules):
export CUDA_HOME=/usr/local/cuda-12.4         # SUBSTITUTE per phase-1 §3.7
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Per full-plan §5.1, both data and EXP_DIR live on project storage.
# Default-if-unset so callers can override at sbatch time via
# `--export=ALL,EXP_DIR=...,OUTPUT_ROOT=...` (used by the §8.4 one-epoch
# gate). Without `:-`, the unconditional assignment would clobber the
# caller-side export.
export OUTPUT_ROOT="${OUTPUT_ROOT:-$DSI_PROJECT/AI-RES/data/makani/sim52_zgplev_full}"
export EXP_DIR="${EXP_DIR:-$DSI_PROJECT/AI-RES/runs/sfno_zgplev_full_dsi}"
export MAIL_TO="zhixingliu@uchicago.edu"
export SFNO_MAIL_FROM="${USER}@uchicago.edu"
export SFNO_SMTP_RELAY="${SFNO_SMTP_RELAY:-smtp.uchicago.edu}"

# The original Stampede3 script references $SCRATCH only via DEFAULT
# values for OUTPUT_ROOT / EXP_DIR (submit_zgplev_full.slurm:47-48).
# Since this wrapper EXPORTS both, the defaults are unused — we do NOT
# alias $SCRATCH here, to avoid hiding scratch/project mistakes.

# scan_for_nans.py is invoked at the end of submit_zgplev_full.slurm
# with `logs/sfno_zgplev_full_${SLURM_JOB_ID}.{out,err}`, but this
# wrapper writes `logs/sfno_zgplev_full_dsi_${SLURM_JOB_ID}.{out,err}`
# — naming mismatch. The scan call is guarded by `|| true` so it won't
# fail the run; it just won't actually scan on DSI. Run scan_for_nans.py
# manually post-run if desired.

mkdir -p "$REPO_ROOT/logs" "$EXP_DIR"

# DO NOT `exec bash …`. Keep this shell alive so a future SIGUSR1
# forwarder (Mitigation C in full-plan §8.3) can be installed here.
# `bash <script>` (without `exec`) makes the wrapper the parent process.
# Trainer auto-resumes from latest checkpoint on each invocation
# (load_checkpoint=legacy at argument_parser.py:61).
bash "$REPO_ROOT/src/sfno_training/submit_zgplev_full.slurm"
