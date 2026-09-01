#!/bin/bash
# Submit the remaining spatial-shape arms (h2w4, then h2w2) as the single
# debug-scaling slot frees up.
#
# WHY A HELPER: `debug-scaling` allows one job per user in Q/H, and a
# `-W depend=afterany:` job is REFUSED here rather than parked in H
# ("would exceed queue generic's per-user limit", measured 2026-09-01 against
# 7580122) -- so the arms cannot be chained at submission time.
#
# It retries **qsub**, never `qstat`: CLAUDE.md forbids a qstat poll loop on a
# shared login node. A failed qsub is the signal that the slot is still busy.
#
# Prereg: makani_bench_report.md §7b. All arms are global batch 32 on 4 nodes,
# which makes per-GPU work identical across shapes -- topology is the only
# variable. Rows go to bench/makani_spatial.csv (EPOCHS=2 rows must not sit in
# the single-epoch scaling table, §0 trap 1).
#
# Usage:  nohup bash polaris/submit_spatial_shapes_when_slot_frees.sh &
# PASS tokens in the log: SPATIAL_ARM_QUEUED per arm, SPATIAL_CHAIN_DONE at end.
set -u

MEMBER_ROOT=/eagle/projects/lighthouse-uchicago/members/mehta5
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${MEMBER_ROOT}/polaris_logs/makani_spatial_chain.log"
TRACE="${MEMBER_ROOT}/runs/makani_mn_scaling/nccl_trace"
CSV="${MEMBER_ROOT}/bench/makani_spatial.csv"
PACK="${MEMBER_ROOT}/data/e3sm_makani_alldata_production"
PLUGIN="${MEMBER_ROOT}/sw/aws-ofi-nccl-1.21.1/lib"

MAX_TRIES=48          # 48 x 5 min = 4 h, enough for two 50-min arms plus queue wait
SLEEP_S=300

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "${LOG}"; }

# arm = "HPAR WPAR LOCAL_BATCH tag"
submit_arm() {
    local hpar="$1" wpar="$2" lb="$3" tag="$4"
    local vars="TARGET_NODES=4,HPAR=${hpar},WPAR=${wpar},LOCAL_BATCH=${lb},STEPS=60,EPOCHS=2"
    vars="${vars},MAKANI_SCALING_CSV=${CSV},CONFIG_YAML=e3sm_alldata_full.yaml,PACK=${PACK}"
    vars="${vars},OFI_PLUGIN=${PLUGIN},OFI_NCCL_PROGRESS_MODEL=AUTO,NCCL_PROTO=Simple"
    vars="${vars},TORCH_NCCL_TRACE_BUFFER_SIZE=2000,TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
    vars="${vars},TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800"
    vars="${vars},TORCH_NCCL_DEBUG_INFO_TEMP_FILE=${TRACE}/${tag}_rank_"

    local i out
    for ((i = 1; i <= MAX_TRIES; i++)); do
        out=$(cd "${HERE}" && qsub -q debug-scaling \
                  -l select=5:system=polaris -l walltime=00:50:00 \
                  -v "${vars}" \
                  polaris/polaris_makani_multinode_scaling.pbs 2>&1)
        if [[ "${out}" == *".polaris-pbs"* ]]; then
            log "SPATIAL_ARM_QUEUED ${tag} jobid=${out%%.*} (try ${i})"
            return 0
        fi
        # A busy slot is the expected failure; anything else should stop the chain
        # rather than burn 48 retries on a real error (a typo'd -v, a bad path).
        if [[ "${out}" != *"per-user limit"* ]]; then
            log "ERROR SPATIAL_ARM_REFUSED ${tag}: ${out}"
            return 2
        fi
        log "slot busy for ${tag} (try ${i}/${MAX_TRIES}); sleeping ${SLEEP_S}s"
        sleep "${SLEEP_S}"
    done
    log "ERROR SPATIAL_ARM_GAVE_UP ${tag} after ${MAX_TRIES} tries"
    return 3
}

mkdir -p "${TRACE}" "$(dirname "${LOG}")"
log "chain start: arm B h2w4 (LOCAL_BATCH=16), then arm C h2w2 (LOCAL_BATCH=8)"

submit_arm 2 4 16 h2w4 || { log "ERROR chain stopped at arm B"; exit 1; }
submit_arm 2 2 8  h2w2 || { log "ERROR chain stopped at arm C"; exit 1; }

log "SPATIAL_CHAIN_DONE both arms queued"
