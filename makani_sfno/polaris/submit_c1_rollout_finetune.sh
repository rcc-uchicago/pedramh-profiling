#!/bin/bash
# C1 — rollout fine-tune (handoff §3 C1). Stage-2 of upstream's two-stage recipe.
#
# WHY. Every run so far trained with n_future 0: the model predicts one 6 h step
# from TRUTH and has never seen its own output. At inference you feed predictions
# back autoregressively, so error compounds into a regime it was never trained
# on. That is the leading explanation for "inference is worse than training
# suggests", and NO amount of further single-step training addresses it -- the
# failure is not in the metric being optimised. Upstream agrees: FCN3 pretrain-2
# exists "to get good autoregressive rollouts" and is a FINE-TUNE from
# pretrain-1's checkpoint, not a run from scratch.
#
# STARTS FROM THE 1-NODE CHECKPOINT, not prod128's. The handoff names
# prod128_alldata_v2 and warns that "a rollout fine-tune cannot repair a base
# model that never converged" -- prod128 had 8,500 weight updates. prod1n_b32_sgdr
# has 332,424 and its cycle minima have flattened, so C1 is far better positioned
# here than in the version of the plan that was written.
#
# QUEUE: `preemptable`, NOT `capacity`. ~24 epochs at roughly 1.5-2x the 683
# s/epoch single-step cost is 7-9 h, which fits easily in preemptable's 72 h; the
# capacity slot is not needed and should not be taken for this. Checkpoints are
# per-epoch, so a preemption costs a resubmit, not progress.
#
# ⚠ TIMING: this contends for I/O with the running production job, whose
# remaining walltime margin is ~1.6 h. Launch AFTER production lands, or accept
# ~0.3 h of that margin for one concurrent job. `7587821` covers the downside.
#
# Usage:
#   bash polaris/submit_c1_rollout_finetune.sh probe     # 2 epochs, batch 16, measure peak memory
#   bash polaris/submit_c1_rollout_finetune.sh full      # 24 epochs at the batch the probe cleared
#   LOCAL_BATCH=6 bash polaris/submit_c1_rollout_finetune.sh full
# PASS token: C1_QUEUED mode=<mode> jobid=<id>
set -u

MEMBER_ROOT=/eagle/projects/lighthouse-uchicago/members/mehta5
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${MEMBER_ROOT}/polaris_logs/makani_c1_rollout.log"
SRC=${MEMBER_ROOT}/runs/makani_mn_scaling/e3sm_mn_scaling/prod1n_b32_sgdr
CKPT=${SRC}/training_checkpoints/best_ckpt_mp0.tar
MODE="${1:-probe}"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "${LOG}"; }
mkdir -p "$(dirname "${LOG}")"

[ -f "${CKPT}" ] || { echo "ERROR NO_PRETRAINED_CKPT: ${CKPT}"; exit 3; }

# ⚠ MEMORY. n_future 1 keeps the graph for TWO forward passes. Measured fit
# (bench report §5g): peak_torch ~= 2.31 + 2.12 GB per sample/GPU, plus ~8 GB
# non-torch. At the production 8 samples/GPU the doubled graph lands near 44 GB
# on a 39.49 GiB card => expected OOM. Hence the probe defaults to 4.
case "${MODE}" in
    probe) LB="${LOCAL_BATCH:-4}"; EP=2;  WALL=03:00:00 ;;
    full)  LB="${LOCAL_BATCH:-4}"; EP=24; WALL=12:00:00 ;;
    *) echo "ERROR unknown mode '${MODE}' (probe|full)"; exit 2 ;;
esac
TAG="c1_rollout_${MODE}_b$((LB*4))"

# ⚠ RUN_NUM MUST BE NEW. `pretrained` and `resuming` are mutually exclusive
# (deterministic_trainer.py:237). If this expDir already holds checkpoints,
# resuming wins, PRETRAINED_CKPT is silently ignored, and you continue the old
# run instead of starting the fine-tune.
if [ -d "${MEMBER_ROOT}/runs/makani_mn_scaling/e3sm_mn_scaling/${TAG}" ]; then
    echo "ERROR EXPDIR_EXISTS: ${TAG} -- pick a new tag, or this RESUMES instead of fine-tuning"
    exit 4
fi

V="TARGET_NODES=1,HPAR=1,WPAR=1,LOCAL_BATCH=${LB},FULL=1,EPOCHS=${EP},EVAL_SAMPLES=512,WANDB=1"
V="${V},RUN_NUM=${TAG}"
# --- the fine-tune itself ---
V="${V},MULTISTEP=2"                 # => n_future 1. The ONLY place n_future can be set.
V="${V},PRETRAINED=1,PRETRAINED_CKPT=${CKPT}"
V="${V},LOAD_OPTIMIZER=0,LOAD_SCHEDULER=0,LOAD_COUNTERS=0,OVERRIDE_LR=1"
# --- schedule: upstream's pretrain-2 LR, plain cosine (no restarts for a short
#     fine-tune -- warm restarts exist to harvest snapshots over 243 epochs) ---
V="${V},LR=4.0E-4,SCHED=CosineAnnealingLR,SCHED_MIN_LR=1.0E-6,WARMUP_EPOCHS=1,LR_START=0.01"
V="${V},CKPT_VERSIONS=30"
V="${V},MAKANI_SCALING_CSV=${MEMBER_ROOT}/bench/makani_c1_rollout.csv"
V="${V},CONFIG_YAML=e3sm_alldata_full.yaml"
V="${V},PACK=${MEMBER_ROOT}/data/e3sm_makani_alldata_production"
V="${V},OFI_PLUGIN=${MEMBER_ROOT}/sw/aws-ofi-nccl-1.21.1/lib"
V="${V},OFI_NCCL_PROGRESS_MODEL=AUTO,NCCL_PROTO=Simple"

OUT=$(cd "${HERE}" && qsub -q preemptable \
        -l select=1:system=polaris -l walltime=${WALL} \
        -v "${V}" polaris/polaris_makani_multinode_scaling.pbs 2>&1)
if [[ "${OUT}" == *".polaris-pbs"* ]]; then
    echo "C1_QUEUED mode=${MODE} tag=${TAG} global_batch=$((LB*4)) epochs=${EP} jobid=${OUT%%.*}"
    log  "C1_QUEUED mode=${MODE} tag=${TAG} gb=$((LB*4)) ep=${EP} jobid=${OUT%%.*} ckpt=${CKPT}"
else
    echo "ERROR c1 submission refused: ${OUT}"; log "ERROR refused: ${OUT}"; exit 5
fi

echo "VERIFY once it starts (all three, in the log):"
echo "  'Loading pretrained checkpoint'   <- pretrained path taken, NOT resuming"
echo "  'multistep_count.*2' / n_future 1 <- the rollout horizon actually applied"
echo "  'peak torch memory [GB]'          <- + non-torch must stay under 39.49"
