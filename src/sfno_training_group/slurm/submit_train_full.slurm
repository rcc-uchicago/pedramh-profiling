#!/bin/bash
# Phase F.F — Multi-node DDP train of group SFNO sigma10 production model.
# Plan: docs/2026-05-10_phaseF_group_production_plan.md (v5).
#
# 50 epochs total; 15h walltime per segment → afterany-chained via
# scripts/submit_train_full_loop.sh. Each segment self-checks .done sentinel
# (epoch >= max_epochs) to avoid over-submission.
#
# CRITICAL: -N is FIXED at 4 in the directives below (sbatch directives don't
# expand env vars). Override via the wrapper script:
#   NODES=8 scripts/submit_train_full_loop.sh
# which passes `sbatch -N $NODES` on the command line.

#SBATCH -J sfno_group_full
#SBATCH -p h100
#SBATCH -N 4
#SBATCH -n 4
#SBATCH --cpus-per-task=16
#SBATCH -t 15:00:00
#SBATCH -o logs/sfno_group_full_%j.out
#SBATCH -e logs/sfno_group_full_%j.err

set -euo pipefail
REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
source "$REPO_ROOT/src/sfno_training_group/env_activate.sh"

# Required env (set by submit_train_full_loop.sh):
RUN_DIR="${RUN_DIR:?need RUN_DIR (e.g. \$EXP_DIR/SFNO/prod_<TS>)}"
RENDERED="${RENDERED:?need RENDERED (path to rendered YAML)}"
RUN_NUM="${RUN_NUM:?need RUN_NUM}"

mkdir -p "$RUN_DIR" "$REPO_ROOT/logs"

# --- Run-dir provenance sentinel (per feedback_protect_prior_runs.md) ---
SENTINEL="$RUN_DIR/.aires_provenance.json"
if [[ -e "$RUN_DIR/checkpoints" && ! -e "$SENTINEL" ]]; then
  echo "ERROR: $RUN_DIR has checkpoints but no provenance sentinel."
  echo "       Refusing to overwrite a prior run. Use a fresh RUN_NUM."
  exit 2
fi
# Write/refresh sentinel.
GIT_SHA=$(cd "$REPO_ROOT" && git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
YAML_SHA=$(sha256sum "$RENDERED" | awk '{print $1}')
cat > "$SENTINEL" <<EOF
{
  "git_sha": "$GIT_SHA",
  "yaml_sha256": "$YAML_SHA",
  "run_num": "$RUN_NUM",
  "started_or_continued_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "max_epochs": 50,
  "phase": "F.L"
}
EOF

# --- Early-exit if max_epochs reached ---
if [[ -e "$RUN_DIR/.done" ]]; then
  echo "[full] $RUN_DIR/.done present; max_epochs reached. Exiting cleanly."
  exit 0
fi

# --- Resume logic ---
# YAML has fresh_start: false. CLI --fresh_start overrides only on first segment
# (no ckpt_latest.tar present).
if [[ -e "$RUN_DIR/checkpoints/ckpt_latest.tar" ]]; then
  RESUME_FLAG=""
  echo "[full] Resuming from $RUN_DIR/checkpoints/ckpt_latest.tar"
else
  RESUME_FLAG="--fresh_start"
  echo "[full] First segment — fresh_start"
fi

# --- Multi-node torchrun rendezvous ---
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}')
RDZV_PORT=29500
echo "[full] rdzv: head=$HEAD_NODE ($HEAD_IP):$RDZV_PORT, nnodes=$SLURM_NNODES"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

cd "$GROUP_PANGU_ROOT"
srun torchrun \
  --nnodes=$SLURM_NNODES --nproc_per_node=1 --node_rank=$SLURM_NODEID \
  --rdzv_backend=c10d --rdzv_endpoint="$HEAD_IP:$RDZV_PORT" --rdzv_id=$SLURM_JOB_ID \
  --rdzv_conf=timeout=300 \
  "$GROUP_PANGU_ROOT/train.py" \
  --config SFNO --yaml_config "$RENDERED" --run_num "$RUN_NUM" \
  $RESUME_FLAG

# --- Post-train: bridge ckpt dir for long_inference; mark .done if at 50 epochs ---
if [[ -d "$RUN_DIR/checkpoints" && ! -e "$RUN_DIR/training_checkpoints" ]]; then
  ln -sfn "$RUN_DIR/checkpoints" "$RUN_DIR/training_checkpoints"
  echo "[full] Bridged checkpoints/ -> training_checkpoints/ at $RUN_DIR"
fi

PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m sfno_training_group.tools.check_train_complete \
  --run-dir "$RUN_DIR" --max-epochs 50

echo "[full] segment done."
