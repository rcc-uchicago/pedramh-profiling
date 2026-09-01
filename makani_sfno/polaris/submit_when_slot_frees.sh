#!/bin/bash
# Submit one makani scaling arm as soon as the single debug-scaling slot frees.
#
# Generalises submit_spatial_shapes_when_slot_frees.sh, which hard-coded its
# arms. Same constraint and same reason: `debug-scaling` allows one job per
# user in Q/H, and `-W depend=afterany:` is REFUSED here rather than parked in
# H ("would exceed queue generic's per-user limit", measured 2026-09-01 against
# 7580122), so arms cannot be chained at submission time.
#
# It retries **qsub**, never `qstat`: CLAUDE.md forbids a qstat poll loop on a
# shared login node, and a failed qsub is itself the signal that the slot is
# busy. Any refusal that is NOT the per-user limit stops the loop rather than
# burning 48 retries on a real error (a typo'd -v, a missing path).
#
# Usage:
#   nohup bash polaris/submit_when_slot_frees.sh <tag> "<extra -v vars>" [select] [walltime] &
# e.g.
#   nohup bash polaris/submit_when_slot_frees.sh ddp4n_lb2 \
#         "TARGET_NODES=4,HPAR=1,WPAR=1,LOCAL_BATCH=2" 5 00:50:00 &
#
# The common half of the -v payload (pack, config, plugin pins, flight
# recorder, CSV routing) is filled in here so an arm cannot silently drop the
# AUTO progress-model pin -- which lives in no script and kills the job with
# ENOSYS if forgotten.
#
# Log: $MEMBER_ROOT/polaris_logs/makani_arm_queue.log
# PASS token: ARM_QUEUED <tag> jobid=<id>
set -u

TAG="${1:?usage: submit_when_slot_frees.sh <tag> \"<vars>\" [select] [walltime]}"
EXTRA="${2:?missing -v payload}"
SELECT="${3:-5}"
WALLTIME="${4:-00:50:00}"

MEMBER_ROOT=/eagle/projects/lighthouse-uchicago/members/mehta5
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${MEMBER_ROOT}/polaris_logs/makani_arm_queue.log"
TRACE="${MEMBER_ROOT}/runs/makani_mn_scaling/nccl_trace"

MAX_TRIES=48          # 48 x 5 min = 4 h
SLEEP_S=300

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "${LOG}"; }

COMMON="STEPS=60,EPOCHS=2"
COMMON="${COMMON},MAKANI_SCALING_CSV=${MEMBER_ROOT}/bench/makani_spatial.csv"
COMMON="${COMMON},CONFIG_YAML=e3sm_alldata_full.yaml"
COMMON="${COMMON},PACK=${MEMBER_ROOT}/data/e3sm_makani_alldata_production"
COMMON="${COMMON},OFI_PLUGIN=${MEMBER_ROOT}/sw/aws-ofi-nccl-1.21.1/lib"
COMMON="${COMMON},OFI_NCCL_PROGRESS_MODEL=AUTO,NCCL_PROTO=Simple"
COMMON="${COMMON},TORCH_NCCL_TRACE_BUFFER_SIZE=2000,TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
COMMON="${COMMON},TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800"
COMMON="${COMMON},TORCH_NCCL_DEBUG_INFO_TEMP_FILE=${TRACE}/${TAG}_rank_"

mkdir -p "${TRACE}" "$(dirname "${LOG}")"
log "waiting for a slot: ${TAG} [${EXTRA}] select=${SELECT} walltime=${WALLTIME}"

for ((i = 1; i <= MAX_TRIES; i++)); do
    out=$(cd "${HERE}" && qsub -q debug-scaling \
              -l "select=${SELECT}:system=polaris" -l "walltime=${WALLTIME}" \
              -v "${EXTRA},${COMMON}" \
              polaris/polaris_makani_multinode_scaling.pbs 2>&1)
    if [[ "${out}" == *".polaris-pbs"* ]]; then
        log "ARM_QUEUED ${TAG} jobid=${out%%.*} (try ${i})"
        exit 0
    fi
    if [[ "${out}" != *"per-user limit"* ]]; then
        log "ERROR ARM_REFUSED ${TAG}: ${out}"
        exit 2
    fi
    log "slot busy for ${TAG} (try ${i}/${MAX_TRIES}); sleeping ${SLEEP_S}s"
    sleep "${SLEEP_S}"
done
log "ERROR ARM_GAVE_UP ${TAG} after ${MAX_TRIES} tries"
exit 3
