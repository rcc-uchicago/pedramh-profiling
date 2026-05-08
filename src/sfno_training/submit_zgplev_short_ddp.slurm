#!/bin/bash

# DDP smoke for the Phase-1 multi-GPU launch path
# (docs/2026-05-04_phase1_efficiency_implementation_plan.md §P1).
#
# Identical configuration to submit_zgplev_short.slurm except:
#   * runs on the h100 partition (same as submit_zgplev_full.slurm)
#     so this actually validates the launch path the full job uses.
#   * launches via torchrun --nproc_per_node=4 (no --disable_ddp).
#   * adds --cpus-per-task=64, env_nccl source, and OMP_NUM_THREADS
#     cap (mirrors the full SLURM). The h100 partition does NOT
#     advertise GPU GRES, so --gpus-per-node is rejected at submit
#     time; -p h100 alone grants the node's full GPU complement.
#
# Sweep-driver hook: ``GB`` (env, default 4) is threaded into
# --batch_size on the torchrun line so a single .slurm can serve every
# point of the I1 short-config DDP sweep
# (docs/2026-05-05_ddp_throughput_fix_plan.md §I1). Each sweep point
# must export a unique EXP_DIR so the trainer's auto-resume at
# train_plasim.py:171-176 cannot contaminate timings — that is
# enforced by scripts/run_zgplev_short_ddp_sweep.sh.
#
# Pass criterion (30 min budget): ≥ 2 epochs complete with no NCCL
# hang and no NaN. If this clears, the full job is safe to resubmit.
#SBATCH -J sfno_zgplev_short_ddp
#SBATCH -p h100                     # Same partition as submit_zgplev_full.slurm
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH -t 00:30:00
#SBATCH -o logs/sfno_zgplev_short_ddp_%j.out
#SBATCH -e logs/sfno_zgplev_short_ddp_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

REPO_ROOT="$HOME/AI-RES"
SFNO_JOB_NAME="sfno_zgplev_short_ddp"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/AI-RES/data/makani/sim52_zgplev_short}"
EXP_DIR="${EXP_DIR:-$SCRATCH/AI-RES/runs/sfno_zgplev_short_ddp}"
MAIL_TO="${MAIL_TO:-zhixingliu@uchicago.edu}"
GB="${GB:-4}"                       # global batch (sweep override; default reproduces prior behaviour)
source "$REPO_ROOT/src/sfno_training/slurm_helpers.sh"
sfno_install_status_trap

source "$REPO_ROOT/.venv/bin/activate"

# NCCL diagnostics + behavioural async-error abort (P5).
source "$REPO_ROOT/src/sfno_training/env_nccl.sh"

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the dataset (subset) root}"
: "${EXP_DIR:?Set EXP_DIR to the experiment output dir}"

SHORT_TPL="$REPO_ROOT/src/sfno_training/config/plasim_sim52_zgplev_short.yaml"
SHORT_YAML="$EXP_DIR/plasim_sim52_zgplev_short.rendered.yaml"
mkdir -p "$EXP_DIR" "$EXP_DIR/training_checkpoints"
sed -e "s|{{OUTPUT_ROOT}}|$OUTPUT_ROOT|g" \
    -e "s|{{EXP_DIR}}|$EXP_DIR|g" \
    "$SHORT_TPL" > "$SHORT_YAML"

cd "$REPO_ROOT"
mkdir -p logs

sfno_send_status_mail "BEGIN" 0 || true

set -x

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-64} / 4 ))
python -c "import sfno_training; print('sfno_training import:', sfno_training.__file__)"

# 1) Preflight (single-GPU; cheap sanity check before torchrun).
python scripts/preflight.py \
    --yaml_config "$SHORT_YAML" \
    --config plasim_sim52_zgplev_short \
    --template "$SHORT_TPL" \
    --amp-mode bf16 \
    --log "$EXP_DIR/preflight_log.txt"

# 2) Train via torchrun. --batch_size is the GLOBAL batch
#    (train_plasim.py:_resolve_batch_sizes divides by data parallel
#    size before the DataLoader sees it). Default GB=4 yields per-rank=1
#    on 4 GPUs (reproduces prior behaviour); the I1 sweep wrapper
#    overrides GB ∈ {4,8,16,32} per point.
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m sfno_training.train_plasim \
    --yaml_config "$SHORT_YAML" \
    --config plasim_sim52_zgplev_short \
    --run_num 0 \
    --batch_size "$GB" \
    --multistep_count 1 \
    --amp_mode bf16

# 3) Post-training NaN/Inf scan (harmless if missing).
if [ -f scripts/scan_for_nans.py ]; then
    python scripts/scan_for_nans.py \
        --exp-dir "$EXP_DIR/plasim_sim52_zgplev_short/0" \
        --slurm-logs "logs/sfno_zgplev_short_ddp_${SLURM_JOB_ID}.out" "logs/sfno_zgplev_short_ddp_${SLURM_JOB_ID}.err" \
        || true
fi

echo "short_ddp complete: $EXP_DIR/plasim_sim52_zgplev_short/0/"
