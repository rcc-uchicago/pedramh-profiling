#!/bin/bash

# v10 zg sigma → zg_plev migration. v9 sibling stays in place for the
# frozen contract; see docs/plasim_zg_plev_migration_plan.md (v7).
#SBATCH -J sfno_zgplev_full
#SBATCH -p h100                     # Stampede3 H100 partition
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64          # tune to actual cores/node on h100
# Stampede3's h100 partition does not advertise GPU GRES; -p h100 selects GPU nodes.
# 7h15m: ~7 epochs (~4h25m) + preflight + ckpt load + ~1h15m bundled-eval
# tail (NWP-only inference ~30 min + climatology+score ~40 min + report+figures
# ~5 min). See docs/2026-05-20_bundled_training_eval_plan.md §4.5.
# Restore to ~48:45:00 for any from-scratch full run with bundled eval.
#SBATCH -t 07:15:00
#SBATCH -o logs/sfno_zgplev_full_%j.out
#SBATCH -e logs/sfno_zgplev_full_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Exclude c561-004 after job 3084458 hung on a 2-elem ALLREDUCE there
# (epoch 44 end-of-epoch loss reduce, NCCL watchdog SIGABRT after 600s).
# Drop this line once we have evidence the node is healthy again.
#SBATCH --exclude=c561-004

# ============================================================================
# Full SFNO emulator training (docs/sfno_full_training_plan.md v1.1).
#
# !! DO NOT SUBMIT until the full v10 zgplev packager output exists with
# !! the year ranges below — i.e. $SCRATCH/SFNO_Climate_Emulator/data/makani/
# !! sim52_astro_64x128_zgplev/ contains MOST.0011.h5 plus 0012-0111.
# !! The v10 packaging today only produces a `…_zgplev_proto` root with
# !! years 3-10; running this script against fewer than the planned 100
# !! train years would burn H100 hours on a model that cannot reach the
# !! intended generalization. Tiny / short / smoke runs on proto are the
# !! correct prototype-validation path (see submit_zgplev_tiny.slurm).
#
# Year ranges (group convention, half-open):
#   train: 12-111  (100 yrs)
#   valid: 11      (1 yr)
#   no test split
#
# Resubmitting picks up from the latest checkpoint automatically:
# train_plasim.py:171-176 sets params["resuming"]=True iff
# {EXP_DIR}/{config}/{run_num}/training_checkpoints/ckpt_mp0_v0.tar exists,
# and deterministic_trainer.py:266-285 then loads the freshest v* by mtime
# (checkpoint_helpers.get_latest_checkpoint_version). This script logs the
# detected resume state before launching torchrun so it's visible in *.out.
#
# Required env vars (defaults match the planned subset layout):
#   OUTPUT_ROOT — packager-output (or symlink subset) root.
#                 Default: $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full
#   EXP_DIR     — experiment output dir.
#                 Default: $SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_full
#
# Build the subset (one-time) before launching. --src is the v10 zgplev
# packaged root produced by plasim_makani_packager (NOT the v9 sigma-level
# sim52_astro_64x128/, which was the previous example here):
#
#   scripts/build_subset_dataset.py \
#       --src $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev \
#       --dst $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full \
#       --train-years 12-111 \
#       --valid-years 11
# ============================================================================

set -euo pipefail

# Captured BEFORE training so bundled_eval.sh can detect "EMA ckpt refreshed
# this run" via mtime > JOB_START_EPOCH (plan §4.4).
JOB_START_EPOCH=$(date +%s)
export JOB_START_EPOCH

REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
SFNO_JOB_NAME="sfno_zgplev_full"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full}"
EXP_DIR="${EXP_DIR:-$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_full}"
MAIL_TO="${MAIL_TO:-zhixingliu@uchicago.edu}"

# Run-dir derivation contract for bundled_eval.sh (plan §4.1).
CONFIG_NAME="plasim_sim52_zgplev_full"
RUN_NUM="${RUN_NUM:-0}"
RUN_DIR="$EXP_DIR/$CONFIG_NAME/$RUN_NUM"
export REPO_ROOT CONFIG_NAME RUN_NUM RUN_DIR

# Bundled-eval default: ON for this production submit. Override with
# BUNDLED_EVAL=0 sbatch ... to disable for one-off runs.
: "${BUNDLED_EVAL:=1}"
export BUNDLED_EVAL
echo "[bundled-eval] BUNDLED_EVAL=$BUNDLED_EVAL (set to 1 to enable, 0 to disable)"

source "$REPO_ROOT/src/sfno_training/slurm_helpers.sh"
sfno_install_status_trap

# Activate the project venv (must include torch + makani + physicsnemo).
source "$REPO_ROOT/.venv/bin/activate"

# NCCL diagnostics + behavioural async-error abort (P5).
source "$REPO_ROOT/src/sfno_training/env_nccl.sh"

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the dataset (subset) root}"
: "${EXP_DIR:?Set EXP_DIR to the experiment output dir}"

# Render the full YAML with concrete paths.
# FULL_TPL is env-overridable so callers (e.g. the v11 SST-surface variant)
# can point at an alternative YAML while keeping the same submit path.
FULL_TPL="${FULL_TPL:-$REPO_ROOT/src/sfno_training/config/plasim_sim52_zgplev_full.yaml}"
FULL_YAML="$EXP_DIR/plasim_sim52_zgplev_full.rendered.yaml"
mkdir -p "$EXP_DIR" "$EXP_DIR/training_checkpoints"
sed -e "s|{{OUTPUT_ROOT}}|$OUTPUT_ROOT|g" \
    -e "s|{{EXP_DIR}}|$EXP_DIR|g" \
    "$FULL_TPL" > "$FULL_YAML"

cd "$REPO_ROOT"
mkdir -p logs

sfno_send_status_mail "BEGIN" 0 || true

set -x

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Cap thread fan-out per rank so 4 ranks don't oversubscribe the node.
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-64} / 4 ))
python -c "import sfno_training; print('sfno_training import:', sfno_training.__file__)"

# 1) Preflight — same bf16 + checkpointing_level=2 as training so the
#    memory probe represents the actual training path
#    (docs/sfno_full_training_plan.md §A.2).
python scripts/preflight.py \
    --yaml_config "$FULL_YAML" \
    --config plasim_sim52_zgplev_full \
    --template "$FULL_TPL" \
    --amp-mode bf16 \
    --checkpointing-level 2 \
    --log "$EXP_DIR/preflight_log.txt"

# 2) Train. Multi-GPU DDP via torchrun (one process per visible GPU).
#    Global batch comes from the YAML (batch_size: 32 → per-rank=8 at
#    4-GPU DDP via train_plasim._resolve_batch_sizes). The CLI no longer
#    overrides --batch_size — the YAML is the single source of truth so
#    the I0 launch summary, the YAML, and the runtime all agree. To
#    sweep batch sizes, override via env (e.g. GB=16 microbench script)
#    rather than re-introducing a hard-coded CLI flag here.
#    --checkpointing_level 2 enables activation checkpointing
#    (default 0 in argument_parser.py:41; group convention is 2).
RESUME_CKPT_DIR="$RUN_DIR/training_checkpoints"
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
    --config plasim_sim52_zgplev_full \
    --run_num 0 \
    --multistep_count 1 \
    --amp_mode bf16 \
    --checkpointing_level 2

# 3) Post-training NaN/Inf scan (chunk 2 deliverable; harmless if missing).
if [ -f scripts/scan_for_nans.py ]; then
    python scripts/scan_for_nans.py \
        --exp-dir "$RUN_DIR" \
        --slurm-logs "logs/sfno_zgplev_full_${SLURM_JOB_ID}.out" "logs/sfno_zgplev_full_${SLURM_JOB_ID}.err" \
        || true
fi

echo "full complete: $RUN_DIR/"

# 4) Bundled eval (plan §4.4). Never exits non-zero on eval failure — eval
#    status flows through BUNDLED_EVAL_STATUS into the status mail body.
set +x
source "$REPO_ROOT/src/sfno_training/bundled_eval.sh"
bundled_eval_maybe_run
