#!/bin/bash
# Stage-1 fine-tune arms on a CHANNEL-SUBSET base: port F (99 out, soil-free) or
# port G (77 out, the ACE2-EAMv3 set). polaris_makani_f_finetune_handoff.md §2.3.
#
# Sibling of submit_finetune_stability_arm.sh, which is NOT edited: it is the record
# of the cancelled A-based arms (handoff §5). The recipe is its, value for value
# (global batch 16, LR 4e-4, CosineAnnealingLR, 1-epoch warmup, LR_START 0.01,
# min 1e-6, SCHED_TMAX 22, 24 epochs, CKPT_VERSIONS 25, fresh optimizer/scheduler/
# counters/loss, a new RUN_NUM per arm). What differs:
#   - the start: PRETRAINED_CKPT=<the base's checkpoint>, required, no default.
#     Handoff default for F: the raw best_ckpt_mp0.tar at the end of F.
#   - config by model; PACK taken from the BASE RUN's config.json (its
#     global_means_path), never chosen here, so an arm normalizes with the stats
#     its base learned under (F and G: production pack, 2015-2044; H: the 2020-2044 view).
#   - finetune_base_check.py refuses a base whose N_out_channels / channel_names
#     are not the config's (an A checkpoint for an F arm, an F one for a G arm).
#   - fabric: nothing passed; the harness's CXI stack (v1.6.0 + HPE rendezvous
#     block) is inherited and overrides in the environment are refused, as in
#     submit_f_nosoil.sh. The old launcher's AUTO/Simple pins predate that fix.
#   - tags fs<MODEL>_<arm>_nf<NF>_b<GB>_r<rep>, e.g. fsF_anneal_nf4_b16_r1.
#   - anneal_dryair is refused for G: G has no TMQ, and the dry-air fix
#     (mass_fix.DryAirFix) needs it. It would also raise in-job, after the wait.
#
#   arm      MULTISTEP  NODES x LOCAL  queue        purpose
#   lrcheck  5          1 x 2 (gb 8)   debug        gate F4: 3 short epochs, SCHED_TMAX=1
#   anneal   5          2 x 2 (gb 16)  preemptable  T-anneal (depth 4)
#   anneal_dryair  as anneal + CONSERVE_DRY_AIR=1   DIAGNOSTIC; F only; jesswan before any default
#   d8       9          4 x 1 (gb 16)  preemptable  T-d8
#   d16      17         4 x 1 (gb 16)  preemptable  T-d16 (blocked: OOM 7650263 unsharded)
#
# Usage:  PRETRAINED_CKPT=<ckpt> bash polaris/submit_subset_finetune_arm.sh <F|G> <arm> [rep]
#   env: WALLTIME=, QUEUE=, DEPEND=<jobid> (afterany: one arm at a time), SCHED_TMAX=,
#        DRY_RUN=1 (run every check, print the qsub, submit nothing)
# PASS token: FINETUNE_ARM_QUEUED model=<m> arm=<a> tag=<t> jobid=<id>  (or FINETUNE_ARM_DRY_RUN_OK)
set -u

MEMBER_ROOT=/eagle/projects/lighthouse-uchicago/members/mehta5
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPROOT="${MEMBER_ROOT}/runs/makani_mn_scaling/e3sm_mn_scaling"
LOG="${MEMBER_ROOT}/polaris_logs/makani_finetune_stability.log"

MODEL="${1:?usage: PRETRAINED_CKPT=<ckpt> submit_subset_finetune_arm.sh <F|G> <arm> [rep]}"
ARM="${2:?usage: PRETRAINED_CKPT=<ckpt> submit_subset_finetune_arm.sh <F|G> <lrcheck|anneal|anneal_dryair|d8|d16> [rep]}"
REP="${3:-1}"
CKPT="${PRETRAINED_CKPT:?PRETRAINED_CKPT=<base checkpoint> is required}"
case "${MODEL}" in
  F) CONFIG_YAML=e3sm_alldata_nosoil.yaml ;;
  G) CONFIG_YAML=e3sm_alldata_ace2vars.yaml ;;
  H) CONFIG_YAML=e3sm_alldata_ace2vars.yaml ;;   # G's channels; its PACK (the 2020-2044 view) comes from the base run
  *) echo "ERROR unknown model '${MODEL}' (F|G|H)"; exit 2 ;;
esac
for v in OFI_PLUGIN OFI_LIBFABRIC NCCL_PROTO CXI_RDZV OFI_NCCL_PROGRESS_MODEL NCCL_NET FI_PROVIDER; do
    if [ -n "${!v:-}" ]; then echo "ERROR FABRIC_OVERRIDE_REFUSED: ${v}=${!v} is set; unset it"; exit 2; fi
done

FULLFLAG=1; ST=60; EVAL=512; EP=24; CKV=25; TMAX_DEFAULT=22
case "${ARM}" in
  lrcheck) MS=5;  NODES=1; LB=2; EP=3; FULLFLAG=0; ST=20; EVAL=32; CKV=5; TMAX_DEFAULT=1
           WALL="${WALLTIME:-01:00:00}"; Q="${QUEUE:-debug}" ;;
  anneal)  MS=5;  NODES=2; LB=2; WALL="${WALLTIME:-12:00:00}"; Q="${QUEUE:-preemptable}" ;;
  anneal_dryair) MS=5; NODES=2; LB=2; DRYAIR=1
           WALL="${WALLTIME:-12:00:00}"; Q="${QUEUE:-preemptable}" ;;
  d8)      MS=9;  NODES=4; LB=1; WALL="${WALLTIME:-24:00:00}"; Q="${QUEUE:-preemptable}" ;;
  d16)     MS=17; NODES=4; LB=1; WALL="${WALLTIME:-48:00:00}"; Q="${QUEUE:-preemptable}" ;;
  *) echo "ERROR unknown arm '${ARM}' (lrcheck|anneal|anneal_dryair|d8|d16)"; exit 2 ;;
esac
if [ "${DRYAIR:-0}" = "1" ] && { [ "${MODEL}" = "G" ] || [ "${MODEL}" = "H" ]; }; then
    echo "ERROR DRYAIR_NEEDS_TMQ: port ${MODEL} dropped TMQ; the dry-air fix has no column water on ${MODEL}"
    exit 2
fi
TMAX="${SCHED_TMAX:-${TMAX_DEFAULT}}"
NF=$((MS - 1))
GB=$((LB * 4 * NODES))
TAG="fs${MODEL}_${ARM}_nf${NF}_b${GB}_r${REP}"

# The base must be a run of THIS config's channel set; its pack is the arm's pack.
CHK=$(python3 "${HERE}/polaris/finetune_base_check.py" "${CKPT}" \
        "${HERE}/polaris/${CONFIG_YAML}" "$(basename "${CONFIG_YAML}" .yaml)")
echo "${CHK}"
[[ "${CHK}" == FINETUNE_BASE_OK* ]] || exit 3
PACK="${CHK#*pack=}"; PACK="${PACK%% *}"
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
[ "${DRYAIR:-0}" = "1" ] && V="${V},CONSERVE_DRY_AIR=1"
V="${V},CKPT_VERSIONS=${CKV}"
V="${V},MAKANI_SCALING_CSV=${MEMBER_ROOT}/bench/makani_finetune_stability.csv"
V="${V},CONFIG_YAML=${CONFIG_YAML}"
V="${V},PACK=${PACK}"

DEP=()
[ -n "${DEPEND:-}" ] && DEP=(-W "depend=afterany:${DEPEND}")
QSUB=(qsub -q "${Q}" "${DEP[@]}" -l "select=${NODES}:system=polaris" -l place=scatter
      -l "walltime=${WALL}" -l filesystems=home:eagle -v "${V}"
      polaris/polaris_makani_multinode_scaling.pbs)
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN (cd ${HERE} &&) ${QSUB[*]}"
    echo "FINETUNE_ARM_DRY_RUN_OK model=${MODEL} arm=${ARM} tag=${TAG} pack=${PACK}"
    exit 0
fi
mkdir -p "$(dirname "${LOG}")"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "${LOG}"; }
OUT=$(cd "${HERE}" && "${QSUB[@]}" 2>&1)
if [[ "${OUT}" == *".polaris-pbs"* ]]; then
    echo "FINETUNE_ARM_QUEUED model=${MODEL} arm=${ARM} tag=${TAG} n_future=${NF} nodes=${NODES}" \
         "global_batch=${GB} dry_air=${DRYAIR:-0} sched_tmax=${TMAX} epochs=${EP} queue=${Q}" \
         "walltime=${WALL} depend=${DEPEND:-none} pack=${PACK} jobid=${OUT%%.*}"
    log "FINETUNE_ARM_QUEUED model=${MODEL} arm=${ARM} tag=${TAG} ckpt=${CKPT} q=${Q} jobid=${OUT%%.*}"
else
    echo "ERROR FINETUNE_ARM_REFUSED model=${MODEL} arm=${ARM}: ${OUT}"
    exit 1
fi
