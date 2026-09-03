#!/bin/bash
# Resume the 1-node production run (7585080) when it hits its 48 h walltime.
#
# WHY THIS IS NEEDED: 7585080 requested walltime 48:00:00 and is running at
# ~714 s/epoch (63 epochs in 12:30). 243 epochs needs ~48.2 h, so it walls out
# roughly 2-3 epochs short of the end. `qalter -l walltime=72:00:00` was tried
# first and REFUSED by ALCF ("Exception in account_check hook", rc=32), so the
# limit cannot be raised in place.
#
# ⚠⚠ THE TRAP THIS SCRIPT EXISTS TO AVOID: the PBS script RE-RENDERS the config
# on every submission from CONFIG_YAML + the -v overrides. Resuming with a
# different (or missing) override set would silently continue the run under
# DIFFERENT HYPERPARAMETERS -- a scheduler reset, a different peak LR, a
# different warmup -- and the loss curve would look like a training bug rather
# than a submission bug. VARS below is copied verbatim from
# submit_production_when_lr_picked.sh:64-72 with LR pinned to the value that
# script picked (2.0E-3, confirmed against the rendered
# e3sm_mn_scaling.prod1n_b32_sgdr.yaml).
#
# ⚠ SECOND TRAP: resume is keyed ENTIRELY on RUN_NUM matching the existing
# expDir (pbs :208-215). If RUN_NUM did not match, makani would NOT error -- it
# would cheerfully start a NEW run from epoch 0 and burn 48 h of the project's
# only capacity slot. The gate below refuses to submit unless real checkpoints
# are on disk.
#
# Resume itself is proven: epochs 5-8 after a restart matched the uninterrupted
# reference byte-identically, across a warm-restart cycle boundary.
# Checkpoints are written EVERY epoch, so at most one epoch is lost.
#
# Submitted with `-W depend=afterany:<jobid>` so it fires automatically whether
# the parent walls out, completes, or dies. If the parent DID finish all 243
# epochs, this job resumes at epoch 243, finds max_epochs reached, and exits in
# minutes -- benign.
#
# ⚠ THIRD TRAP -- WHY THE DEFAULT QUEUE IS NOT `capacity`. Pre-queuing the
# dependent job on `capacity` is IMPOSSIBLE: its per-project limit counts the
# RUNNING parent, so qsub refuses with "would exceed queue capacity's
# per-project limit" (measured 2026-09-03). The slot only frees when the parent
# ends -- which is the exact moment this job needs to already be queued. So the
# pre-staged resume goes to `preemptable`, where the tail is safe: only ~3
# epochs remain, checkpoints are per-epoch, and a preemption costs a resubmit,
# not progress. Once the capacity slot IS free, prefer it:
#     QUEUE=capacity WALLTIME=48:00:00 bash polaris/submit_production_resume.sh
#
# Usage:  bash polaris/submit_production_resume.sh [parent_jobid]
#   QUEUE=<q>          default preemptable (see above)
#   WALLTIME=<hh:mm:ss> default 06:00:00 -- ~30 epochs of catch-up, far more than
#                      the ~3 the tail needs, and short enough to start promptly
#                      on preemptable (start latency there is load-dependent).
# PASS token: PRODUCTION_RESUME_QUEUED jobid=<id> depends_on=<parent>
set -u

MEMBER_ROOT=/eagle/projects/lighthouse-uchicago/members/mehta5
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${MEMBER_ROOT}/polaris_logs/makani_production_resume.log"
PARENT="${1:-7585080}"
RUN_NUM=prod1n_b32_sgdr
CKPT_DIR="${MEMBER_ROOT}/runs/makani_mn_scaling/e3sm_mn_scaling/${RUN_NUM}/training_checkpoints"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "${LOG}"; }
mkdir -p "$(dirname "${LOG}")"

# ---- gate: refuse to "resume" a run that is not on disk ---------------------
NCKPT=$(ls -1 "${CKPT_DIR}"/ckpt_mp0_v*.tar 2>/dev/null | wc -l)
if [ "${NCKPT}" -lt 1 ]; then
    echo "ERROR NO_CHECKPOINTS_TO_RESUME: ${CKPT_DIR}"
    echo "      Submitting anyway would start a NEW 243-epoch run from scratch."
    log   "ERROR NO_CHECKPOINTS_TO_RESUME: ${CKPT_DIR}"
    exit 3
fi
LAST=$(ls -1 "${CKPT_DIR}"/ckpt_mp0_v*.tar 2>/dev/null | sed 's/.*_v//; s/\.tar//' | sort -n | tail -1)
echo "gate OK: ${NCKPT} checkpoints on disk, latest epoch v${LAST}"
log "gate OK: ${NCKPT} checkpoints, latest v${LAST}, parent=${PARENT}"

# ---- VARS: verbatim from submit_production_when_lr_picked.sh:64-72 ----------
VARS="TARGET_NODES=1,HPAR=1,WPAR=1,LOCAL_BATCH=8,FULL=1,EPOCHS=243,EVAL_SAMPLES=512,WANDB=1"
VARS="${VARS},RUN_NUM=${RUN_NUM},LR=2.0E-3"
VARS="${VARS},SCHED=CosineAnnealingWarmRestarts,SCHED_T0=20,SCHED_TMULT=1,SCHED_MIN_LR=1.0E-6"
VARS="${VARS},WARMUP_EPOCHS=3,LR_START=0.01,CKPT_VERSIONS=250"
VARS="${VARS},MAKANI_SCALING_CSV=${MEMBER_ROOT}/bench/makani_production.csv"
VARS="${VARS},CONFIG_YAML=e3sm_alldata_full.yaml"
VARS="${VARS},PACK=${MEMBER_ROOT}/data/e3sm_makani_alldata_production"
VARS="${VARS},OFI_PLUGIN=${MEMBER_ROOT}/sw/aws-ofi-nccl-1.21.1/lib"
VARS="${VARS},OFI_NCCL_PROGRESS_MODEL=AUTO,NCCL_PROTO=Simple"

QUEUE="${QUEUE:-preemptable}"
WALLTIME="${WALLTIME:-06:00:00}"
echo "submitting to ${QUEUE}, walltime ${WALLTIME}, depend=afterany:${PARENT}"
OUT=$(cd "${HERE}" && qsub -q "${QUEUE}" \
        -l select=1:system=polaris -l walltime="${WALLTIME}" \
        -W depend=afterany:"${PARENT}" \
        -v "${VARS}" \
        polaris/polaris_makani_multinode_scaling.pbs 2>&1)

if [[ "${OUT}" == *".polaris-pbs"* ]]; then
    echo "PRODUCTION_RESUME_QUEUED jobid=${OUT%%.*} depends_on=${PARENT}"
    log  "PRODUCTION_RESUME_QUEUED jobid=${OUT%%.*} depends_on=${PARENT} resume_from=v${LAST}"
else
    echo "ERROR resume submission refused: ${OUT}"
    log  "ERROR resume submission refused: ${OUT}"
    exit 4
fi
