#!/bin/bash

# torch.profiler capture run. Single-GPU on purpose: characterizes the
# production codepath independently of DDP overhead so the trace shows
# dataloader / preprocess / forward / backward gaps cleanly.
#
# Profile overhead is ~5–15% while the capture window is active; read
# the resulting traces as ratios, not as absolute step times.
#SBATCH -J sfno_zgplev_profile
#SBATCH -p amd-rtx                  # Stampede3 AMD+RTX GPU partition
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o logs/sfno_zgplev_profile_%j.out
#SBATCH -e logs/sfno_zgplev_profile_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Phase-1 efficiency profiling (docs/2026-05-04_phase1_efficiency_implementation_plan.md P6).
#
# Captures one Chrome trace via torch.profiler covering steps
# capture_range_start..capture_range_stop on rank 0. Stats also print to
# stdout via profiling.trace_handler(print_stats=True).
#
# Open the trace at chrome://tracing or perfetto.dev. Compare the
# inter-step gap distribution before and after P2 (persistent_workers,
# prefetch_factor) to measure dataloader savings.
# ============================================================================

set -euo pipefail

REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
SFNO_JOB_NAME="sfno_zgplev_profile"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_short}"
EXP_DIR="${EXP_DIR:-$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_profile}"
PROFILE_DIR="${PROFILE_DIR:-$WORK/SFNO_Climate_Emulator/profiles/$(date +%Y%m%d_%H%M%S)}"
MAIL_TO="${MAIL_TO:-zhixingliu@uchicago.edu}"
source "$REPO_ROOT/src/sfno_training/slurm_helpers.sh"
sfno_install_status_trap

# Activate the project venv (must include torch + makani + physicsnemo).
source "$REPO_ROOT/.venv/bin/activate"

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the dataset (subset) root}"
: "${EXP_DIR:?Set EXP_DIR to the experiment output dir}"

# Render the short YAML (the profile job runs against the short config).
SHORT_TPL="$REPO_ROOT/src/sfno_training/config/plasim_sim52_zgplev_short.yaml"
SHORT_YAML="$EXP_DIR/plasim_sim52_zgplev_short.rendered.yaml"
mkdir -p "$EXP_DIR" "$EXP_DIR/training_checkpoints"
mkdir -p "$PROFILE_DIR"
sed -e "s|{{OUTPUT_ROOT}}|$OUTPUT_ROOT|g" \
    -e "s|{{EXP_DIR}}|$EXP_DIR|g" \
    "$SHORT_TPL" > "$SHORT_YAML"

cd "$REPO_ROOT"
mkdir -p logs

sfno_send_status_mail "BEGIN" 0 || true

set -x

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m sfno_training.train_plasim \
    --yaml_config "$SHORT_YAML" \
    --config plasim_sim52_zgplev_short \
    --run_num 0 \
    --batch_size 4 \
    --multistep_count 1 \
    --amp_mode bf16 \
    --disable_ddp \
    --capture_ranks 0 \
    --capture_type torch \
    --capture_mode training \
    --capture_range_start 5 \
    --capture_range_stop 35 \
    --capture_prefix "$PROFILE_DIR/plasim_sfno"

echo "profile complete: traces under $PROFILE_DIR"
