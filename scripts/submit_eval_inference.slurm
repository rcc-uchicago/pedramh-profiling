#!/bin/bash
#SBATCH -J sfno_eval_inf
#SBATCH -p amd-rtx                  # AMD-EPYC + NVIDIA RTX (CUDA works; near-zero queue vs h100)
#SBATCH --exclude=c572-002          # c572-002 has a broken GPU at index 2; NCCL probes all
                                    # physical GPUs via NVML during init regardless of
                                    # CUDA_VISIBLE_DEVICES, so it crashes with
                                    # `nvmlDeviceGetHandleByIndex(2) failed: Unknown Error`
                                    # (observed 2026-05-21 on jobs 3139860 and 3140076).
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -o logs/sfno_eval_inf_%j.out
#SBATCH -e logs/sfno_eval_inf_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# SFNO emulator inference (docs/sfno_eval_plan.md §G.2 row 1).
# 96 NWP ICs × 56 steps + 8 climate × ~1456 steps = ~17,016 forward passes.
# Estimated runtime ~1.5 h on one H100; 6 h wallclock leaves ample margin.
#
# Body extracted to scripts/eval_run_inference_inline.sh so it can be sourced
# from both this SLURM wrapper and the bundled-in-training helper
# (docs/2026-05-20_bundled_training_eval_plan.md §4.3).
#
# Required env vars: see eval_run_inference_inline.sh header. RUN_TAG,
# EVAL_SHA7, DATA_SHA7, TRAIN_SHA7 must be set by the caller (the prelude
# function does this).
# ============================================================================

REPO_ROOT="${REPO_ROOT:-$HOME/projects/SFNO_Climate_Emulator}"
source "$REPO_ROOT/scripts/eval_run_inference_inline.sh"
