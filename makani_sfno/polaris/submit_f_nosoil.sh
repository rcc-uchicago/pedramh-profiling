#!/bin/bash
# Port F production run: e3sm_alldata_nosoil.yaml, MULTI-NODE, on `capacity`.
#   bash polaris/submit_f_nosoil.sh <NODES 1-4> <EPOCHS> <WALLTIME hh:mm:ss> [scratch|warm]
#
# Recipe = prod1n_b32_sgdr's config.json exactly (LR 2e-3, beta2 0.95, grad clip 32,
# CosineAnnealingWarmRestarts T0=20 Tmult=1, warmup 3) at GLOBAL BATCH 32, so the
# optimizer trajectory is the base's and nodes buy only wall-clock:
#   LOCAL_BATCH = 32 / (4 x NODES)  -> 1 node x 8, 2 x 4, 4 x 2.
# EPOCHS should be 3 + 20k so the run ends on a cosine-cycle boundary.
# warm = start from 7646690's sliced checkpoint (surgical proof: val 0.01427, 1.11x
# the base, STRONG tier) with fresh optimizer/scheduler/counters/loss.
#
# FABRIC: inherits polaris_makani_multinode_scaling.pbs's measured CXI stack
# (aws-ofi-nccl v1.6.0, NCCL_PROTO=Simple, rendezvous block). This wrapper REFUSES
# to run if any fabric override is set in its environment -- a -v OFI_PLUGIN= is
# how the 128-node run silently went over TCP -- and the launcher's watchdog kills
# the ranks if NCCL selects anything but cxi.
set -euo pipefail
NODES="${1:?NODES}"; EPOCHS="${2:?EPOCHS}"; WALL="${3:?WALLTIME}"; ROUTE="${4:-warm}"
M=/eagle/projects/lighthouse-uchicago/members/mehta5
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for v in OFI_PLUGIN OFI_LIBFABRIC NCCL_PROTO CXI_RDZV OFI_NCCL_PROGRESS_MODEL NCCL_NET FI_PROVIDER; do
    if [ -n "${!v:-}" ]; then echo "ERROR FABRIC_OVERRIDE_REFUSED: ${v}=${!v} is set; unset it"; exit 2; fi
done
case "${NODES}" in 1|2|4) ;; *) echo "ERROR NODES must be 1, 2 or 4 (global batch 32)"; exit 2;; esac
LB=$(( 32 / (4 * NODES) ))
if [ $(( (EPOCHS - 3) % 20 )) -ne 0 ]; then echo "ERROR EPOCHS must be 3 + 20k (cycle boundary)"; exit 2; fi

RUN_NUM="f_nosoil_${NODES}n_b32_e${EPOCHS}_${ROUTE}"
SLICED="${M}/runs/makani_probe/surgical/7646690/best_ckpt_mp0_nosoil.tar"
VARS="TARGET_NODES=${NODES},HPAR=1,WPAR=1,LOCAL_BATCH=${LB},FULL=1,EPOCHS=${EPOCHS},EVAL_SAMPLES=512,WANDB=1"
VARS="${VARS},RUN_NUM=${RUN_NUM},LR=2.0E-3,BETA2=0.95,MAX_GRAD_NORM=32"
VARS="${VARS},SCHED=CosineAnnealingWarmRestarts,SCHED_T0=20,SCHED_TMULT=1,SCHED_MIN_LR=1.0E-6"
VARS="${VARS},WARMUP_EPOCHS=3,LR_START=0.01,CKPT_VERSIONS=250,EMA=1,EMA_DECAY=0.9995"
VARS="${VARS},CONFIG_YAML=e3sm_alldata_nosoil.yaml,PACK=${M}/data/e3sm_makani_alldata_production"
VARS="${VARS},MAKANI_SCALING_CSV=${M}/bench/makani_production.csv"
if [ "${ROUTE}" = "warm" ]; then
    [ -f "${SLICED}" ] || { echo "ERROR SLICED_CKPT_MISSING: ${SLICED}"; exit 2; }
    VARS="${VARS},PRETRAINED_CKPT=${SLICED},LOAD_OPTIMIZER=0,LOAD_SCHEDULER=0,LOAD_COUNTERS=0,LOAD_LOSS=0,OVERRIDE_LR=1"
fi
# select = NODES + 1 spare: the launcher's GPU preflight runs on the first NODES healthy.
SELECT=$(( NODES + 1 ))

PROV="${M}/runs/makani_mn_scaling/${RUN_NUM}.warmstart_provenance.txt"
{
    echo "run=${RUN_NUM}  submitted=$(date -u +%FT%TZ)  by=submit_f_nosoil.sh  git=$(git -C "${HERE}" rev-parse --short HEAD)"
    echo "route=${ROUTE}  nodes=${NODES} (+1 spare)  local_batch=${LB}  global_batch=32  epochs=${EPOCHS}"
    echo "base recipe = prod1n_b32_sgdr/config.json; the ONLY intended differences:"
    echo "  channel set 101 -> 99 (SOILWATER_10CM, TSOI_10CM dropped; port F)"
    echo "  EMA on (decay 0.9995; shadow only, raw trajectory unchanged)"
    echo "  node count ${NODES} (global batch held at 32)"
    [ "${ROUTE}" = "warm" ] && echo "  init = sliced prod1n_b32_sgdr best (7646690), fresh optimizer/sched/counters/loss"
    echo "vars: ${VARS}"
} > "${PROV}"
cat "${PROV}"

OUT=$(cd "${HERE}" && qsub -q capacity -l "select=${SELECT}:system=polaris" -l "walltime=${WALL}" \
        -v "${VARS}" polaris/polaris_makani_multinode_scaling.pbs 2>&1)
echo "${OUT}"
[[ "${OUT}" == *".polaris-pbs"* ]] && echo "F_QUEUED jobid=${OUT%%.*} run=${RUN_NUM}" || { echo "ERROR SUBMIT_REFUSED"; exit 2; }
