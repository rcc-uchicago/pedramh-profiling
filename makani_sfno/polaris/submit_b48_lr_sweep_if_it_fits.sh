#!/bin/bash
# Submit the batch-48 LR sweep, but ONLY if the batch-48 memory probe (7585983)
# proved batch 48 fits on one node.
#
# WHY GATED: 12 samples/GPU sits in the untested gap between two measured points
# -- 8 samples = 16.04 GB, 16 samples = OOM on a 39.49 GiB card. Memory scales
# SUPERLINEARLY there (doubling samples took ~2.5x, not 2x), so 12 is a genuine
# coin-flip and three arms would be wasted if it OOMs.
#
# ARMS -- chosen to bracket both scaling rules from the batch-32 winner (2.0e-3):
#   2.0e-3  optimum does NOT move with batch
#   3.0e-3  optimum tracks batch LINEARLY (2e-3 x 48/32)
#   4.5e-3  optimum climbs faster than linear
# sqrt-scaling predicts 2.45e-3, which arms 1 and 2 straddle.
#
# QUEUE: `preemptable` -- max_run 10 per project against debug's 1, so all three
# run CONCURRENTLY (~35 min wall instead of ~105 serial). Each arm is short, so
# preemption costs a rerun, not a run, and no resume is needed. If preemptable
# refuses or is slow to start, fall back to:
#   bash polaris/submit_when_slot_frees.sh <tag> "<vars>" 2 00:55:00 <csv> debug-scaling
#
# Log: $MEMBER_ROOT/polaris_logs/makani_b48_sweep.log
# PASS token per arm: B48_ARM_QUEUED <lr> jobid=<id>
set -u

MEMBER_ROOT=/eagle/projects/lighthouse-uchicago/members/mehta5
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${MEMBER_ROOT}/polaris_logs/makani_b48_sweep.log"
PROBE_O="${HERE}/makani_mn_scaling.o7585983"
PROBE_LOG="${MEMBER_ROOT}/runs/makani_mn_scaling/b48_mem_probe.log"
CSV="${MEMBER_ROOT}/bench/makani_lrsweep_b48.csv"

MAX_TRIES=24        # 24 x 2 min = 48 min
SLEEP_S=120

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "${LOG}"; }
mkdir -p "$(dirname "${LOG}")"
log "gate: waiting on the batch-48 memory probe (7585983)"

for ((i = 1; i <= MAX_TRIES; i++)); do
    if [ -f "${PROBE_O}" ] && grep -q "OutOfMemoryError" "${PROBE_O}" 2>/dev/null; then
        log "ABORT batch 48 OOMs on one node -- submitting NOTHING. The LR sweep at this"
        log "      batch is moot until LOCAL_BATCH is reduced; record the OOM as the third"
        log "      point on the memory curve (8 -> 16.04 GB, 12 -> OOM, 16 -> OOM)."
        exit 4
    fi
    # a written epoch summary with a memory footprint means the step completed => it fits
    if [ -f "${PROBE_LOG}" ] && grep -q "memory footprint" "${PROBE_LOG}" 2>/dev/null; then
        MEM=$(grep "memory footprint" "${PROBE_LOG}" | tail -1 | sed 's/.*: //')
        log "GATE PASSED batch 48 fits -- peak ${MEM} GB on a 39.49 GiB card"
        break
    fi
    log "probe still running (try ${i}/${MAX_TRIES})"
    [ "${i}" -eq "${MAX_TRIES}" ] && { log "ERROR gave up waiting on the probe"; exit 3; }
    sleep "${SLEEP_S}"
done

COMMON="TARGET_NODES=1,HPAR=1,WPAR=1,LOCAL_BATCH=12,FULL=1,EPOCHS=3,EVAL_SAMPLES=512,WANDB=1"
COMMON="${COMMON},SCHED=CosineAnnealingWarmRestarts,SCHED_T0=2,SCHED_TMULT=1,SCHED_MIN_LR=1.0E-6"
COMMON="${COMMON},WARMUP_EPOCHS=1,LR_START=0.01,CKPT_VERSIONS=5"
COMMON="${COMMON},MAKANI_SCALING_CSV=${CSV},CONFIG_YAML=e3sm_alldata_full.yaml"
COMMON="${COMMON},PACK=${MEMBER_ROOT}/data/e3sm_makani_alldata_production"
COMMON="${COMMON},OFI_PLUGIN=${MEMBER_ROOT}/sw/aws-ofi-nccl-1.21.1/lib"
COMMON="${COMMON},OFI_NCCL_PROGRESS_MODEL=AUTO,NCCL_PROTO=Simple"

for LR in 2.0E-3 3.0E-3 4.5E-3; do
    TAG="b48_lr${LR}"
    OUT=$(cd "${HERE}" && qsub -q preemptable \
            -l select=1:system=polaris -l walltime=00:55:00 \
            -v "${COMMON},LR=${LR},RUN_NUM=${TAG}" \
            polaris/polaris_makani_multinode_scaling.pbs 2>&1)
    if [[ "${OUT}" == *".polaris-pbs"* ]]; then
        log "B48_ARM_QUEUED ${LR} jobid=${OUT%%.*}"
    else
        log "ERROR arm ${LR} refused: ${OUT}"
    fi
done
log "B48_SWEEP_SUBMITTED"
