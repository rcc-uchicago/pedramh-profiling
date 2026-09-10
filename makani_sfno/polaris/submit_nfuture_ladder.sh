#!/bin/bash
# n_future ladder -- warm-started fine-tunes at rollout depths 1 / 3 / 4.
#
# Sibling of submit_c1_rollout_finetune.sh, which hardcodes MULTISTEP=2 and is a
# proven artifact for C1; it is deliberately NOT edited (CLAUDE.md #7's rule,
# applied to the arm axis). This script parameterises MULTISTEP and SEED.
#
# Design and decision rules are pre-registered:
#   docs/2026-09-10_nfuture_ladder_prereg.md      (formal)
#   docs/2026-09-10_nfuture_plan_in_plain_english.md
#
# ⚠ THREE TRAPS, ALL MEASURED, ALL INHERITED FROM C1:
#  1. LOAD_LOSS=0 is REQUIRED. LossHandler carries running statistics whose
#     SHAPE depends on n_future; restoring an n_future=0 loss state into an
#     n_future=3 model raises a size-mismatch RuntimeError at construction
#     (job 7590350 died there; job 7603090 hit the same thing while SCORING).
#     Do NOT work around it with strict_restore: false -- that would silently
#     skip real mismatches in the model weights too.
#  2. A config-side `n_future` does NOTHING. train.py:119 does
#     params["n_future"] = args.multistep_count - 1, overwriting the config.
#     -v MULTISTEP is the only handle.
#  3. `pretrained` and `resuming` are mutually exclusive
#     (deterministic_trainer.py:237 gates on `pretrained and not resuming`), so
#     every arm needs a NEW RUN_NUM. If the target expDir already holds
#     checkpoints, resuming wins and PRETRAINED_CKPT is silently ignored.
#
# ⚠ BATCH IS HELD FIXED ACROSS SCIENCE ARMS. Deeper rollouts force a smaller
# batch, so if batch moved with depth the two would be confounded and no arm
# would be interpretable. Prereg threat T3: global batch 8 (LOCAL_BATCH=2) for
# every science arm INCLUDING the n_future=1 control. `probe` mode is the one
# exception -- it deliberately runs at the predicted MAXIMUM batch, because
# testing the memory model is its whole purpose.
#
# Memory model (handoff C2), calibrated at n_future 0 and 1 ONLY:
#     GiB ~= 10.31 + 2.12 * (n_future+1) * samples_per_GPU     card = 39.49
#     => samples_per_GPU <= 13.76 / (n_future+1)
#        n_future=3 -> 3   n_future=4 -> 2
# Using it at 3/4 is EXTRAPOLATION. That is what `probe` mode exists to check.
#
# Usage:
#   bash polaris/submit_nfuture_ladder.sh probe 4        # n_future=3, max batch
#   bash polaris/submit_nfuture_ladder.sh probe 5        # n_future=4, max batch
#   bash polaris/submit_nfuture_ladder.sh proxy 4 1      # n_future=3, seed 1
#   SEEDS="1 2 3" bash polaris/submit_nfuture_ladder.sh proxy 4
# PASS token: NFUTURE_QUEUED mode=<m> n_future=<n> seed=<s> jobid=<id>
set -u

MEMBER_ROOT=/eagle/projects/lighthouse-uchicago/members/mehta5
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPROOT="${MEMBER_ROOT}/runs/makani_mn_scaling/e3sm_mn_scaling"
LOG="${MEMBER_ROOT}/polaris_logs/makani_nfuture_ladder.log"
CKPT="${EXPROOT}/prod1n_b32_sgdr/training_checkpoints/best_ckpt_mp0.tar"

MODE="${1:?usage: submit_nfuture_ladder.sh <probe|proxy> <multistep> [seed]}"
MS="${2:?multistep (= n_future + 1); 4 => n_future 3, 5 => n_future 4}"
NF=$((MS - 1))

mkdir -p "$(dirname "${LOG}")"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "${LOG}"; }
[ -f "${CKPT}" ] || { echo "ERROR NO_CHECKPOINT: ${CKPT}"; exit 3; }

# Predicted max samples/GPU from the memory model above:
#     b <= (39.49 - 10.31) / (2.12 * MS) = 13.76 / MS
# The 2.12 coefficient is already absorbed into 13.76 -- do NOT divide by it
# again. Integer arithmetic at 100x: 1376 / (100 * MS).
#     MS=4 (n_future 3) -> 1376/400 = 3
#     MS=5 (n_future 4) -> 1376/500 = 2
MAXB=$(( 1376 / (100 * MS) ))
[ "${MAXB}" -lt 1 ] && MAXB=1

case "${MODE}" in
  # Memory probe: predicted MAX batch, a short truncated epoch. 60 steps is
  # plenty -- allocator peak lands in the first few steps, and FULL=0 sets
  # n_train_samples_per_epoch = STEPS * GLOBAL_BATCH exactly.
  probe) LB="${LOCAL_BATCH:-${MAXB}}"; EP=1; ST="${STEPS:-60}"; FULLFLAG=0
         WALL="${WALLTIME:-01:00:00}"; Q="${QUEUE:-debug}" ;;
  # Science arm: batch HELD at 8 global (prereg T3), one full pass of data.
  proxy) LB="${LOCAL_BATCH:-2}"; EP="${EPOCHS:-1}"; ST=60; FULLFLAG=1
         WALL="${WALLTIME:-01:00:00}"; Q="${QUEUE:-debug}" ;;
  *) echo "ERROR unknown mode '${MODE}' (probe|proxy)"; exit 2 ;;
esac

# ⚠ THERE IS NO SEED KNOB, AND ASKING FOR ONE WOULD BE A SILENT NO-OP.
# Verified 2026-09-10: makani has no global training seed -- every `seed=333` in
# the tree belongs to a noise module, a DALI loader or drop-path, none of which
# this fork uses. Our sampler is constructed as
#   DistributedSampler(..., shuffle=True)         plasim_trainer.py:104-110
# with NO seed argument, so torch defaults to seed 0, and nothing calls
# set_epoch(), so the shuffle is identical every epoch. Every arm additionally
# restores the SAME pretrained checkpoint, so there is no random init either.
#
# ⇒ Replication here varies only GPU nondeterminism (cuDNN algo choice, atomic
#   reduction order). REP does NOT set a seed; it only gives the arm a distinct
#   expDir. Two REPs of one config therefore MEASURE that nondeterminism rather
#   than assuming it -- which is the honest version of what "3 seeds" was
#   reaching for.
#
# The replication that actually matters for ranking arms is over INITIAL
# CONDITIONS at evaluation time, not over training runs. See the prereg.
REPS="${REPS:-${3:-1}}"
for SEED in ${REPS}; do
    TAG="nf${NF}_${MODE}_b$((LB*4))_r${SEED}"
    if [ -d "${EXPROOT}/${TAG}" ]; then
        echo "  SKIP ${TAG}: expDir exists (trap 3 -- resuming would win over pretrained)"
        continue
    fi

    V="TARGET_NODES=1,HPAR=1,WPAR=1,LOCAL_BATCH=${LB},FULL=${FULLFLAG},EPOCHS=${EP},STEPS=${ST}"
    V="${V},EVAL_SAMPLES=512,WANDB=0,RUN_NUM=${TAG}"
    V="${V},MULTISTEP=${MS}"                      # trap 2: the ONLY n_future handle
    V="${V},PRETRAINED=1,PRETRAINED_CKPT=${CKPT}"
    V="${V},LOAD_OPTIMIZER=0,LOAD_SCHEDULER=0,LOAD_COUNTERS=0,LOAD_LOSS=0,OVERRIDE_LR=1"
    # NOTE: no SEED here on purpose -- the PBS renderer has no `seed` key
    # (_bools at :451-459 plus the explicit keys list), so -v SEED=... would be
    # accepted by qsub and then silently dropped. Passing dead knobs is how this
    # project has lost days.
    # LR 4.0E-4 is upstream's pretrain-2 value and what C1 used -- held fixed so
    # the ladder varies depth alone.
    V="${V},LR=4.0E-4,SCHED=CosineAnnealingLR,SCHED_MIN_LR=1.0E-6,WARMUP_EPOCHS=0,LR_START=0.01"
    V="${V},CKPT_VERSIONS=4"
    V="${V},MAKANI_SCALING_CSV=${MEMBER_ROOT}/bench/makani_nfuture_ladder.csv"
    V="${V},CONFIG_YAML=e3sm_alldata_full.yaml"
    V="${V},PACK=${MEMBER_ROOT}/data/e3sm_makani_alldata_production"
    V="${V},OFI_PLUGIN=${MEMBER_ROOT}/sw/aws-ofi-nccl-1.21.1/lib"
    V="${V},OFI_NCCL_PROGRESS_MODEL=AUTO,NCCL_PROTO=Simple"

    OUT=$(cd "${HERE}" && qsub -q "${Q}" \
            -l select=1:system=polaris -l walltime="${WALL}" -l filesystems=home:eagle \
            -v "${V}" polaris/polaris_makani_multinode_scaling.pbs 2>&1)
    if [[ "${OUT}" == *".polaris-pbs"* ]]; then
        echo "  NFUTURE_QUEUED mode=${MODE} n_future=${NF} seed=${SEED} tag=${TAG}" \
             "global_batch=$((LB*4)) predicted_GiB=$(awk -v m=${MS} -v b=${LB} \
               'BEGIN{printf "%.1f", 10.31 + 2.12*m*b}') jobid=${OUT%%.*}"
        log "NFUTURE_QUEUED n_future=${NF} seed=${SEED} tag=${TAG} gb=$((LB*4)) jobid=${OUT%%.*}"
    else
        echo "  ERROR arm n_future=${NF} seed=${SEED} refused: ${OUT}"
        log "ERROR n_future=${NF} seed=${SEED}: ${OUT}"
    fi
done
echo "NFUTURE_SUBMITTED mode=${MODE} n_future=${NF}"
echo "Read peak memory: grep 'peak torch memory' \$MEMBER_ROOT/runs/makani_mn_scaling/nf${NF}_${MODE}_*.log"
