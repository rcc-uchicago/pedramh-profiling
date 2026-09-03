#!/bin/bash
# Prereg §7d: the BATCH-32 MIRROR of the §7c LR-ceiling test.
#
# Sibling of submit_b48_lr_ceiling_fix.sh. Exactly two things differ:
#   LOCAL_BATCH  12 -> 8   (global 48 -> 32)
#   arms         5 -> 4    (no `push` arm; see below)
# Everything else -- LR, scheduler, warmup, eval samples, pack, plugin -- is
# byte-identical to the b48 script on purpose. If you change one, change both.
#
# WHY LR 3.0e-3 AND NOT BATCH 32's OWN DEAD POINT (4.0e-3, job 7585996):
# matching the PROBE LR to §7c is what makes the two sets differ in BATCH ALONE.
# Matching the dead point instead would confound batch with LR. It also fills a
# real gap -- 3.0e-3 at batch 32 has never been run; it sits inside the untested
# (2e-3, 4e-3) bracket -- so `b32fix_base` is a NEW measurement that narrows the
# batch-32 ceiling, not a repeat.
#
# THE PAIR OF CONTROLS IS THE POINT. 7587738 (batch 48, shipped settings) vs
# b32fix_base (batch 32, shipped settings) is a matched-LR, matched-optimizer,
# batch-only contrast -- the batch-independence test that did NOT exist before.
# §5j had to infer it from two DIFFERENT LRs.
#
# No `push` arm: §7c D already probes how far the ceiling moves, and at batch 32
# the ceiling location is what b32fix_base is measuring. Four arms also keeps the
# project under preemptable's max_run 10 (5 from §7c + 4 = 9, one slot spare).
#
# Log: $MEMBER_ROOT/polaris_logs/makani_b32_ceiling_fix.log
# PASS token per arm: B32FIX_ARM_QUEUED <tag> jobid=<id>
set -u

MEMBER_ROOT=/eagle/projects/lighthouse-uchicago/members/mehta5
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${MEMBER_ROOT}/polaris_logs/makani_b32_ceiling_fix.log"
CSV="${MEMBER_ROOT}/bench/makani_lrsweep_b32fix.csv"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "${LOG}"; }
mkdir -p "$(dirname "${LOG}")"

# LOCAL_BATCH=8 is the ONLY line that differs from the b48 COMMON block.
COMMON="TARGET_NODES=1,HPAR=1,WPAR=1,LOCAL_BATCH=8,FULL=1,EPOCHS=6,EVAL_SAMPLES=512,WANDB=1"
COMMON="${COMMON},SCHED=CosineAnnealingWarmRestarts,SCHED_T0=2,SCHED_TMULT=1,SCHED_MIN_LR=1.0E-6"
COMMON="${COMMON},WARMUP_EPOCHS=1,LR_START=0.01,CKPT_VERSIONS=5"
COMMON="${COMMON},MAKANI_SCALING_CSV=${CSV},CONFIG_YAML=e3sm_alldata_full.yaml"
COMMON="${COMMON},PACK=${MEMBER_ROOT}/data/e3sm_makani_alldata_production"
COMMON="${COMMON},OFI_PLUGIN=${MEMBER_ROOT}/sw/aws-ofi-nccl-1.21.1/lib"
COMMON="${COMMON},OFI_NCCL_PROGRESS_MODEL=AUTO,NCCL_PROTO=Simple"

# tag|LR|BETA2|MAX_GRAD_NORM
ARMS=(
    "b32fix_base|3.0E-3|0.95|32"      # E32 control -- shipped settings
    "b32fix_beta2|3.0E-3|0.999|32"    # A32         -- beta2 alone
    "b32fix_clip|3.0E-3|0.95|1.0"     # B32         -- clip alone
    "b32fix_both|3.0E-3|0.999|1.0"    # C32         -- both
)

log "SUBMIT prereg 7d -- 4 arms, batch 32, preemptable, 3 h each"
for spec in "${ARMS[@]}"; do
    IFS='|' read -r TAG LR B2 CLIP <<< "${spec}"
    # ⚠ PBS -v is LAST-WINS: per-arm vars MUST come after ${COMMON}.
    EXTRA="LR=${LR},BETA2=${B2},MAX_GRAD_NORM=${CLIP},RUN_NUM=${TAG}"
    OUT=$(cd "${HERE}" && qsub -q preemptable \
            -l select=1:system=polaris -l walltime=03:00:00 \
            -v "${COMMON},${EXTRA}" \
            polaris/polaris_makani_multinode_scaling.pbs 2>&1)
    if [[ "${OUT}" == *".polaris-pbs"* ]]; then
        log "B32FIX_ARM_QUEUED ${TAG} lr=${LR} beta2=${B2} clip=${CLIP} jobid=${OUT%%.*}"
        echo "B32FIX_ARM_QUEUED ${TAG} lr=${LR} beta2=${B2} clip=${CLIP} jobid=${OUT%%.*}"
    else
        log "ERROR arm ${TAG} refused: ${OUT}"
        echo "ERROR arm ${TAG} refused: ${OUT}"
    fi
done
log "B32FIX_SWEEP_SUBMITTED"
echo "B32FIX_SWEEP_SUBMITTED"
