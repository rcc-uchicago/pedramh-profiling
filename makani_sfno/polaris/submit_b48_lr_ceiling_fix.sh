#!/bin/bash
# Prereg §7c: can the batch-48 LR ceiling be raised by optimizer config alone?
#
# BACKGROUND (bench report §5j). The LR ceiling is (2e-3, 3e-3] and it did NOT
# move when batch went 32 -> 48: linear scaling predicts 3e-3 should be as safe
# at 48 as 2e-3 was at 32, and 3e-3 died (7586383). Failure is a ONE-EPOCH
# blow-up into an irreversible dead attractor (train 0.1056, valid ~0.107, grad
# norm ~0.008), reproduced 2/2 at LR 4.5e-3 (v1 + v2 arms).
#
# TWO CONFIG SUSPECTS, both isolated here:
#   optimizer_beta2      0.95 -- a LARGE-batch setting inherited from the
#                        batch-512 run (~20-step second-moment memory).
#   optimizer_max_grad_norm  32 -- but the measured grad norm never exceeds
#                        0.2995 in 64 production epochs, so the clip sits ~107x
#                        above the operating point and has NEVER engaged.
#                        makani's own default is 1.0 (makani/train.py:74-75).
#
# ARM E IS THE CONTROL and is not optional: LR 3.0e-3 has n=1 for the collapse,
# so without a same-conditions repeat, arms A-D cannot be read at all.
#
# QUEUE: `preemptable` -- 5 arms need max_run 10/project, and 3 h of walltime.
# `debug`/`debug-scaling` cap at 1 h, which does not fit 3 epochs of batch 48
# (~730-1100 s/epoch, and 5-way I/O contention makes it worse). The signal
# (collapse or not) lands at EPOCH 2, so a preemption after epoch 3 still scores.
#
# Log: $MEMBER_ROOT/polaris_logs/makani_b48_ceiling_fix.log
# PASS token per arm: B48FIX_ARM_QUEUED <tag> jobid=<id>
set -u

MEMBER_ROOT=/eagle/projects/lighthouse-uchicago/members/mehta5
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${MEMBER_ROOT}/polaris_logs/makani_b48_ceiling_fix.log"
CSV="${MEMBER_ROOT}/bench/makani_lrsweep_b48fix.csv"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "${LOG}"; }
mkdir -p "$(dirname "${LOG}")"

# Matched to the sweep these arms are compared against (7586382/3/4) in every
# respect except the two knobs under test. Do not "improve" anything here.
COMMON="TARGET_NODES=1,HPAR=1,WPAR=1,LOCAL_BATCH=12,FULL=1,EPOCHS=6,EVAL_SAMPLES=512,WANDB=1"
COMMON="${COMMON},SCHED=CosineAnnealingWarmRestarts,SCHED_T0=2,SCHED_TMULT=1,SCHED_MIN_LR=1.0E-6"
COMMON="${COMMON},WARMUP_EPOCHS=1,LR_START=0.01,CKPT_VERSIONS=5"
COMMON="${COMMON},MAKANI_SCALING_CSV=${CSV},CONFIG_YAML=e3sm_alldata_full.yaml"
COMMON="${COMMON},PACK=${MEMBER_ROOT}/data/e3sm_makani_alldata_production"
COMMON="${COMMON},OFI_PLUGIN=${MEMBER_ROOT}/sw/aws-ofi-nccl-1.21.1/lib"
COMMON="${COMMON},OFI_NCCL_PROGRESS_MODEL=AUTO,NCCL_PROTO=Simple"

# tag|LR|BETA2|MAX_GRAD_NORM
ARMS=(
    "b48fix_base|3.0E-3|0.95|32"      # E control  -- shipped settings, must collapse
    "b48fix_beta2|3.0E-3|0.999|32"    # A          -- beta2 alone
    "b48fix_clip|3.0E-3|0.95|1.0"     # B          -- clip alone
    "b48fix_both|3.0E-3|0.999|1.0"    # C          -- both
    "b48fix_push|4.5E-3|0.999|1.0"    # D          -- both, harder rung
)

log "SUBMIT prereg 7c -- 5 arms, batch 48, preemptable, 3 h each"
for spec in "${ARMS[@]}"; do
    IFS='|' read -r TAG LR B2 CLIP <<< "${spec}"
    # ⚠ PBS -v is LAST-WINS: the per-arm vars MUST come after ${COMMON}, or
    # COMMON's values silently override them. This cost a whole sweep once.
    EXTRA="LR=${LR},BETA2=${B2},MAX_GRAD_NORM=${CLIP},RUN_NUM=${TAG}"
    OUT=$(cd "${HERE}" && qsub -q preemptable \
            -l select=1:system=polaris -l walltime=03:00:00 \
            -v "${COMMON},${EXTRA}" \
            polaris/polaris_makani_multinode_scaling.pbs 2>&1)
    if [[ "${OUT}" == *".polaris-pbs"* ]]; then
        log "B48FIX_ARM_QUEUED ${TAG} lr=${LR} beta2=${B2} clip=${CLIP} jobid=${OUT%%.*}"
        echo "B48FIX_ARM_QUEUED ${TAG} lr=${LR} beta2=${B2} clip=${CLIP} jobid=${OUT%%.*}"
    else
        log "ERROR arm ${TAG} refused: ${OUT}"
        echo "ERROR arm ${TAG} refused: ${OUT}"
    fi
done
log "B48FIX_SWEEP_SUBMITTED"
echo "B48FIX_SWEEP_SUBMITTED"
