#!/bin/bash
# Stage-1 training arms of polaris_makani_finetune_stability_handoff.md §3.
#
# Sibling of submit_nfuture_ladder.sh (NOT edited: its own header's rule, and the
# handoff §5). That script cannot express these arms: it hardcodes
# CKPT_VERSIONS=4, passes no SCHED_TMAX, and its TAG (nf4_prod_b16_r1) would
# collide with B and be skipped. Everything else below is its `prod` block,
# value for value, so the arms differ from B only where the handoff says.
#
# Every arm: warm start from A's best_ckpt (trap 3: fresh RUN_NUM), global batch 16,
# LR 4e-4, CosineAnnealingLR, 1-epoch warmup, LR_START 0.01, min 1e-6, 24 epochs,
# every epoch kept (CKPT_VERSIONS=25, ~1.8 GB each), LOAD_LOSS=0 (trap 1),
# MULTISTEP the only n_future handle (trap 2).
#
# SCHED_TMAX=22: makani composes SequentialLR([LinearLR warmup, Cosine]) with the
# milestone at the end of warmup and steps once per epoch, so epoch e >= 2 trains at
# cosine t = e - 2, and T_max 22 puts epoch 24 at eta_min. Gate S1a
# (lr_schedule_check.py) must be green before an arm is submitted.
#
#   arm      MULTISTEP  NODES x LOCAL  queue        purpose
#   lrcheck  5          1 x 2 (gb 8)   debug        S1a: 3 short epochs, SCHED_TMAX=1
#   anneal   5          2 x 2 (gb 16)  preemptable  T-anneal (depth 4)
#   d8       9          4 x 1 (gb 16)  preemptable  T-d8
#   d16      17         4 x 1 (gb 16)  preemptable  T-d16 (only if its probe fits)
#
# Usage:  bash polaris/submit_finetune_stability_arm.sh <arm> [rep]
#   env: WALLTIME=, QUEUE=, DEPEND=<jobid> (afterany: one arm at a time), SCHED_TMAX=
# PASS token: FINETUNE_ARM_QUEUED arm=<a> tag=<t> jobid=<id>
set -u

MEMBER_ROOT=/eagle/projects/lighthouse-uchicago/members/mehta5
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPROOT="${MEMBER_ROOT}/runs/makani_mn_scaling/e3sm_mn_scaling"
LOG="${MEMBER_ROOT}/polaris_logs/makani_finetune_stability.log"
CKPT="${EXPROOT}/prod1n_b32_sgdr/training_checkpoints/best_ckpt_mp0.tar"

ARM="${1:?usage: submit_finetune_stability_arm.sh <lrcheck|anneal|d8|d16> [rep]}"
REP="${2:-1}"
FULLFLAG=1; ST=60; EVAL=512; EP=24; CKV=25; TMAX_DEFAULT=22
case "${ARM}" in
  lrcheck) MS=5;  NODES=1; LB=2; EP=3; FULLFLAG=0; ST=20; EVAL=32; CKV=5; TMAX_DEFAULT=1
           WALL="${WALLTIME:-01:00:00}"; Q="${QUEUE:-debug}" ;;
  anneal)  MS=5;  NODES=2; LB=2; WALL="${WALLTIME:-12:00:00}"; Q="${QUEUE:-preemptable}" ;;
  d8)      MS=9;  NODES=4; LB=1; WALL="${WALLTIME:-24:00:00}"; Q="${QUEUE:-preemptable}" ;;
  d16)     MS=17; NODES=4; LB=1; WALL="${WALLTIME:-48:00:00}"; Q="${QUEUE:-preemptable}" ;;
  *) echo "ERROR unknown arm '${ARM}' (lrcheck|anneal|d8|d16)"; exit 2 ;;
esac
TMAX="${SCHED_TMAX:-${TMAX_DEFAULT}}"
NF=$((MS - 1))
GB=$((LB * 4 * NODES))
TAG="fs_${ARM}_nf${NF}_b${GB}_r${REP}"

mkdir -p "$(dirname "${LOG}")"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "${LOG}"; }
[ -f "${CKPT}" ] || { echo "ERROR NO_CHECKPOINT: ${CKPT}"; exit 3; }
if [ -d "${EXPROOT}/${TAG}" ]; then
    echo "ERROR TAG_EXISTS ${TAG}: expDir exists (trap 3 -- resuming would win over pretrained); pass a new rep"
    exit 4
fi

V="TARGET_NODES=${NODES},HPAR=1,WPAR=1,LOCAL_BATCH=${LB},FULL=${FULLFLAG},EPOCHS=${EP},STEPS=${ST}"
V="${V},EVAL_SAMPLES=${EVAL},WANDB=0,RUN_NUM=${TAG}"
V="${V},MULTISTEP=${MS}"
V="${V},PRETRAINED=1,PRETRAINED_CKPT=${CKPT}"
V="${V},LOAD_OPTIMIZER=0,LOAD_SCHEDULER=0,LOAD_COUNTERS=0,LOAD_LOSS=0,OVERRIDE_LR=1"
V="${V},LR=4.0E-4,SCHED=CosineAnnealingLR,SCHED_MIN_LR=1.0E-6,WARMUP_EPOCHS=1,LR_START=0.01"
V="${V},SCHED_TMAX=${TMAX}"
V="${V},CKPT_VERSIONS=${CKV}"
V="${V},MAKANI_SCALING_CSV=${MEMBER_ROOT}/bench/makani_finetune_stability.csv"
V="${V},CONFIG_YAML=e3sm_alldata_full.yaml"
V="${V},PACK=${MEMBER_ROOT}/data/e3sm_makani_alldata_production"
V="${V},OFI_NCCL_PROGRESS_MODEL=AUTO,NCCL_PROTO=Simple"

DEP=()
[ -n "${DEPEND:-}" ] && DEP=(-W "depend=afterany:${DEPEND}")
OUT=$(cd "${HERE}" && qsub -q "${Q}" "${DEP[@]}" \
        -l select=${NODES}:system=polaris -l place=scatter \
        -l walltime="${WALL}" -l filesystems=home:eagle \
        -v "${V}" polaris/polaris_makani_multinode_scaling.pbs 2>&1)
if [[ "${OUT}" == *".polaris-pbs"* ]]; then
    echo "FINETUNE_ARM_QUEUED arm=${ARM} tag=${TAG} n_future=${NF} nodes=${NODES} global_batch=${GB}" \
         "sched_tmax=${TMAX} epochs=${EP} queue=${Q} walltime=${WALL} depend=${DEPEND:-none} jobid=${OUT%%.*}"
    log "FINETUNE_ARM_QUEUED arm=${ARM} tag=${TAG} nodes=${NODES} gb=${GB} tmax=${TMAX} q=${Q} jobid=${OUT%%.*}"
else
    echo "ERROR FINETUNE_ARM_REFUSED arm=${ARM}: ${OUT}"
    log "ERROR arm=${ARM}: ${OUT}"
    exit 1
fi
