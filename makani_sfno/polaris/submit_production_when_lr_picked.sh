#!/bin/bash
# Wait for the 3-arm LR sweep, apply a PRE-REGISTERED selection rule, and submit
# the 1-node production run to `capacity`.
#
# The rule is fixed here BEFORE arms 2 and 3 exist, so the choice cannot be
# rationalised after seeing the numbers (repo method: prereg -> measure -> score).
#
#   1. DISQUALIFY an arm with a non-finite loss, or whose gradient norm RISES
#      from epoch 2 to epoch 3. A higher LR always descends faster early, so
#      raw loss alone is not valid evidence across LRs (ai-rossby's lesson);
#      grad-norm trajectory is what predicts tail instability.
#   2. Among survivors, take the lowest EPOCH-3 VALIDATION loss.
#   3. If the top two are within 5%, prefer the LOWER LR -- under warm restarts
#      this value is the peak of every cycle for 46 h, so ties break safe.
#   4. If NO arm survives (1), submit nothing and say so.
#
# Polls the trainer LOG FILES, never qstat (no poll loops on a login node).
#
# PRODUCTION SHAPE (settled by measurement, makani_bench_report.md §5):
#   1 node / 4 GPUs / pure DDP / global batch 32 / LOCAL_BATCH=8 / GPU_ORDER=default
#   675.7 s/epoch measured at production settings => 243 epochs ~= 45.6 h
#   243 = 3 warmup + 12 x 20-epoch cycles, so the run ENDS ON a cycle boundary
#        and every one of the 12 snapshots is an annealed, usable checkpoint.
#   332,424 optimizer updates = 1.6x upstream FCN3 pretrain-1's 208,320.
#
# ⚠ COST, stated rather than buried: `capacity` is max_run 1 PER PROJECT, so this
#   blocks every other lighthouse-uchicago member for up to 48 h. Abort with
#   `qdel <jobid>` -- but note cancelling destroys accrued eligible_time.
#
# Log: $MEMBER_ROOT/polaris_logs/makani_production_submit.log
# PASS token: PRODUCTION_QUEUED jobid=<id> lr=<lr>
set -u

MEMBER_ROOT=/eagle/projects/lighthouse-uchicago/members/mehta5
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${MEMBER_ROOT}/polaris_logs/makani_production_submit.log"
RUNS="${MEMBER_ROOT}/runs/makani_mn_scaling"

MAX_TRIES=36        # 36 x 5 min = 3 h
SLEEP_S=300

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "${LOG}"; }
mkdir -p "$(dirname "${LOG}")"
log "waiting for lr_4e4 / lr_1e3 / lr_2e3 to reach epoch 3"

for ((i = 1; i <= MAX_TRIES; i++)); do
    PICK=$("${HERE}/polaris/pick_lr.py" "${RUNS}" 2>&1)
    RC=$?
    if [ "${RC}" -eq 0 ]; then
        log "selection: ${PICK}"
        break
    fi
    log "not ready (try ${i}/${MAX_TRIES}): ${PICK}"
    [ "${i}" -eq "${MAX_TRIES}" ] && { log "ERROR GAVE_UP waiting for the sweep"; exit 3; }
    sleep "${SLEEP_S}"
done

LR=$(echo "${PICK}" | sed -n 's/.*WINNER_LR=\([^ ]*\).*/\1/p')
[ -z "${LR}" ] && { log "ERROR NO_WINNER -- no arm survived the stability filter; submitting nothing"; exit 4; }

VARS="TARGET_NODES=1,HPAR=1,WPAR=1,LOCAL_BATCH=8,FULL=1,EPOCHS=243,EVAL_SAMPLES=512,WANDB=1"
VARS="${VARS},RUN_NUM=prod1n_b32_sgdr,LR=${LR}"
VARS="${VARS},SCHED=CosineAnnealingWarmRestarts,SCHED_T0=20,SCHED_TMULT=1,SCHED_MIN_LR=1.0E-6"
VARS="${VARS},WARMUP_EPOCHS=3,LR_START=0.01,CKPT_VERSIONS=250"
VARS="${VARS},MAKANI_SCALING_CSV=${MEMBER_ROOT}/bench/makani_production.csv"
VARS="${VARS},CONFIG_YAML=e3sm_alldata_full.yaml"
VARS="${VARS},PACK=${MEMBER_ROOT}/data/e3sm_makani_alldata_production"
VARS="${VARS},OFI_PLUGIN=${MEMBER_ROOT}/sw/aws-ofi-nccl-1.21.1/lib"
VARS="${VARS},OFI_NCCL_PROGRESS_MODEL=AUTO,NCCL_PROTO=Simple"

OUT=$(cd "${HERE}" && qsub -q capacity \
        -l select=1:system=polaris -l walltime=48:00:00 \
        -v "${VARS}" \
        polaris/polaris_makani_multinode_scaling.pbs 2>&1)

if [[ "${OUT}" == *".polaris-pbs"* ]]; then
    log "PRODUCTION_QUEUED jobid=${OUT%%.*} lr=${LR} epochs=243 walltime=48h queue=capacity"
    log "  abort with: qdel ${OUT%%.*}   (destroys accrued eligible_time)"
else
    log "ERROR SUBMIT_REFUSED: ${OUT}"
    exit 2
fi
