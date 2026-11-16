#!/bin/bash
# rank_env.sh -- PALS -> torch-style rank env for the SQG multi-node smoke (sqg_multisim.py), ALCF Polaris.
#
# Copied from makani_sfno/polaris/polaris_rank_env.sh (itself vendored from physicsnemo_ai_rossby), so the SQG tree
# stays launchable on its own; same mapping, plus ONE line: each rank sees exactly one GPU. One visible device per rank
# keeps JAX (the solver) and torch (the trainer) on the same device without any per-framework device selection.
# GPU_ORDER=default (the default): local rank r -> GPU r, as makani production. GPU_ORDER=reverse: r -> 3 - r, the
# NUMA-local GPU (polaris_pbs_notes.md §1: dev0->NUMA3 ... dev3->NUMA0); makani measured it +0.88% slower for 1-node DDP
# and 7% faster for 4-node h2w2 (makani_bench_report.md §5i), so it stays opt-in until an SQG A/B says otherwise.
#
#   mpiexec -n 8 --ppn 4 --hostfile "$PBS_NODEFILE" --cpu-bind depth -d 8 --label --line-buffer \
#       bash SQG/analysis/opt/multisim/rank_env.sh "$PY" SQG/analysis/opt/multisim/sqg_multisim.py ...
# MASTER_ADDR / MASTER_PORT must already be exported by the job script.

set -uo pipefail

: "${PMI_RANK:?not running under PALS mpiexec — PMI_RANK unset}"
: "${PMI_SIZE:?not running under PALS mpiexec — PMI_SIZE unset}"
: "${MASTER_ADDR:?export MASTER_ADDR in the job script before mpiexec}"
: "${MASTER_PORT:?export MASTER_PORT in the job script before mpiexec}"

export RANK="${PMI_RANK}"
export WORLD_SIZE="${PMI_SIZE}"
export LOCAL_RANK="${PMI_LOCAL_RANK:-$(( PMI_RANK % ${GPUS_PER_NODE:-4} ))}"
if [ "${GPU_ORDER:-default}" = "reverse" ]; then
    export CUDA_VISIBLE_DEVICES="$(( ${GPUS_PER_NODE:-4} - 1 - LOCAL_RANK ))"
else
    export CUDA_VISIBLE_DEVICES="${LOCAL_RANK}"
fi

exec "$@"
