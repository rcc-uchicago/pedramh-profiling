#!/bin/bash
# Phase F.L — chain N submit_train_full.slurm segments (afterany).
#
# Each segment exits cleanly if `.done` sentinel exists (max_epochs reached).
# So over-submitting CHAIN_LEN is safe — surplus segments are no-ops.
#
# Required env:
#   EXP_DIR    parent run dir (e.g. $SCRATCH/SFNO_Climate_Emulator/runs/sfno_group_sigma10_full)
#   RUN_NUM    suffix (e.g. prod_20260510_1234)
# Optional:
#   NODES      multi-node count (default 4); passed to sbatch -N $NODES
#   CHAIN_LEN  segments to chain (default 5); each is 15h max -> 75h total budget
#   RENDERED   rendered YAML (default: $EXP_DIR/SFNO/$RUN_NUM/rendered.yaml)
#
# Output: prints job IDs of all submitted segments.

set -euo pipefail

EXP_DIR="${EXP_DIR:?need EXP_DIR}"
RUN_NUM="${RUN_NUM:?need RUN_NUM}"
NODES="${NODES:-4}"
CHAIN_LEN="${CHAIN_LEN:-5}"

RUN_DIR="$EXP_DIR/SFNO/$RUN_NUM"
RENDERED="${RENDERED:-$RUN_DIR/rendered.yaml}"

if [[ ! -f "$RENDERED" ]]; then
  echo "ERROR: rendered YAML not found at $RENDERED"
  echo "       Render it first via sfno_training_group.tools.render_yaml"
  exit 2
fi

REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
SLURM_PATH="$REPO_ROOT/src/sfno_training_group/slurm/submit_train_full.slurm"

mkdir -p "$RUN_DIR" "$REPO_ROOT/logs"

EXPORT_LIST="ALL,RUN_DIR=$RUN_DIR,RENDERED=$RENDERED,RUN_NUM=$RUN_NUM"

JOBIDS=()
JOBID=$(sbatch --parsable -N $NODES -n $NODES \
  --export="$EXPORT_LIST" "$SLURM_PATH" \
  2>/dev/null | grep -oE '^[0-9]+$' | tail -1)
echo "[chain] segment 1: jobid=$JOBID"
JOBIDS+=("$JOBID")

for i in $(seq 2 "$CHAIN_LEN"); do
  JOBID=$(sbatch --parsable --dependency=afterany:$JOBID -N $NODES -n $NODES \
    --export="$EXPORT_LIST" "$SLURM_PATH" \
    2>/dev/null | grep -oE '^[0-9]+$' | tail -1)
  echo "[chain] segment $i: jobid=$JOBID (afterany prev)"
  JOBIDS+=("$JOBID")
done

cat <<EOF

[chain] submitted CHAIN_LEN=$CHAIN_LEN segments on $NODES h100 nodes each.
  RUN_DIR  = $RUN_DIR
  RENDERED = $RENDERED
  jobids   = ${JOBIDS[*]}

Watch:    squeue -j $(IFS=,; echo "${JOBIDS[*]}")
Cancel:   scancel ${JOBIDS[*]}
Sentinel: when epoch >= max_epochs, $RUN_DIR/.done is created and remaining
          segments exit cleanly without re-running torchrun.
EOF
