#!/bin/bash
# Score a checkpoint at several ROLLOUT LENGTHS, so "did inference improve?"
# becomes a curve against lead time instead of one number that cannot show it.
#
# WHY THE DEFAULT METRIC CANNOT ANSWER THE QUESTION. Validation ships at
# `valid_autoreg_steps: 3`, and `rollout_length = valid_autoreg_steps + 1`
# (deterministic_trainer.py:169), so every validation loss quoted in this
# project -- including production's 0.01284 -- is a **4-step, 24-forecast-hour**
# score. Error compounding, which `n_future: 0` fails at and C1 exists to fix,
# is negligible at 24 h and dominant at 5-15 days. Comparing C1 against the base
# model at the default would very likely show nothing, or show C1 slightly
# WORSE, and neither would mean C1 failed.
#
# ⚠⚠ COST SCALES WITH batch x (VALID_AUTOREG + 1). In eval mode the fork sets
# the DATASET's n_future to valid_autoreg_steps (plasim_trainer.py:140-142), so
# each sample carries VALID_AUTOREG+1 target frames of 101x180x360 float32 =
# 26.2 MiB each. At the production LOCAL_BATCH=8 and VALID_AUTOREG=40 that is
# 8.0 GiB of target tensors per batch of HOST memory before prefetch, and 512
# GiB of reads for 512 samples. Hence LOCAL_BATCH=2 and EVAL_SAMPLES=512 below.
# Validation runs under no_grad, so a small batch costs almost nothing.
#
# ⚠ THE VA=3 ARM IS A CONTROL, not padding. It must reproduce the run's already
# known validation loss (0.01284 for prod1n_b32_sgdr). If it does not, the
# scoring path is wrong and the longer arms mean nothing. Same discipline that
# made the 9-arm LR factorial readable.
#
# ⚠ Comparing ACROSS rollout lengths is not over identical initial conditions:
# a longer rollout needs more trailing frames, so the set of valid start indices
# shrinks. Compare MODELS at a FIXED length; read the curve across lengths as
# indicative only.
#
# Usage:
#   bash polaris/submit_rollout_scorecard.sh <run_name> [rollout lengths...]
#   bash polaris/submit_rollout_scorecard.sh prod1n_b32_sgdr 3 10 20
#   bash polaris/submit_rollout_scorecard.sh c1_rollout_full_b16 3 10 20
# PASS token per arm: SCORECARD_QUEUED <run> va=<n> jobid=<id>
set -u

MEMBER_ROOT=/eagle/projects/lighthouse-uchicago/members/mehta5
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPROOT="${MEMBER_ROOT}/runs/makani_mn_scaling/e3sm_mn_scaling"
LOG="${MEMBER_ROOT}/polaris_logs/makani_rollout_scorecard.log"
SRC_RUN="${1:?usage: submit_rollout_scorecard.sh <run_name> [rollout lengths...]}"; shift
VAS=("${@:-3 10 20}"); [ $# -eq 0 ] && VAS=(3 10 20)

QUEUE="${QUEUE:-preemptable}"     # NOT debug: 1 h is too tight once VA rises
WALL="${WALLTIME:-02:00:00}"      # qalter cannot extend walltime on this system
LB="${LOCAL_BATCH:-2}"
NEVAL="${EVAL_SAMPLES:-512}"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "${LOG}"; }
mkdir -p "$(dirname "${LOG}")"

CKPT="${EXPROOT}/${SRC_RUN}/training_checkpoints/best_ckpt_mp0.tar"
[ -f "${CKPT}" ] || { echo "ERROR NO_CHECKPOINT: ${CKPT}"; exit 3; }
echo "scoring ${SRC_RUN}  best_ckpt  at rollout lengths: ${VAS[*]}"

for VA in "${VAS[@]}"; do
    TAG="score_${SRC_RUN}_va${VA}"
    DST="${EXPROOT}/${TAG}"
    if [ -d "${DST}" ]; then
        echo "  SKIP ${TAG}: expDir exists (delete it to rescore)"; continue
    fi
    # ⚠ SEED AS v0. makani gates `resuming` on EXACTLY ckpt_mp0_v0.tar
    # (train.py:101-105); any other name gives resuming=False and the job
    # TRAINS FROM SCRATCH with no error. And the restore picks the newest file
    # by mtime, so a lone v0 resolves to this checkpoint whatever epoch it holds.
    mkdir -p "${DST}/training_checkpoints" || exit 4
    cp "${CKPT}" "${DST}/training_checkpoints/ckpt_mp0_v0.tar" || exit 4
    for f in config.json metadata.json global_means.npy global_stds.npy; do
        [ -f "${EXPROOT}/${SRC_RUN}/${f}" ] && cp "${EXPROOT}/${SRC_RUN}/${f}" "${DST}/${f}"
    done

    # ⚠ WANDB=0 is mandatory for a seeded dir: with wandb on AND resuming=True,
    # Driver._init_wandb reads <expDir>/wandb/makani_restart.yaml, which a seeded
    # dir has not got, and every rank dies at construction.
    V="TARGET_NODES=1,HPAR=1,WPAR=1,LOCAL_BATCH=${LB},FULL=0,EPOCHS=1,STEPS=1"
    V="${V},SKIP_TRAIN=1,WANDB=0,EVAL_SAMPLES=${NEVAL},VALID_AUTOREG=${VA}"
    V="${V},RUN_NUM=${TAG},MAKANI_SCALING_CSV=${MEMBER_ROOT}/bench/makani_scorecard.csv"
    V="${V},CONFIG_YAML=e3sm_alldata_full.yaml"
    V="${V},PACK=${MEMBER_ROOT}/data/e3sm_makani_alldata_production"
    V="${V},OFI_PLUGIN=${MEMBER_ROOT}/sw/aws-ofi-nccl-1.21.1/lib"
    V="${V},OFI_NCCL_PROGRESS_MODEL=AUTO,NCCL_PROTO=Simple"

    OUT=$(cd "${HERE}" && qsub -q "${QUEUE}" \
            -l select=1:system=polaris -l walltime="${WALL}" -l filesystems=home:eagle \
            -v "${V}" polaris/polaris_makani_multinode_scaling.pbs 2>&1)
    if [[ "${OUT}" == *".polaris-pbs"* ]]; then
        echo "  SCORECARD_QUEUED ${SRC_RUN} va=${VA} jobid=${OUT%%.*}"
        log  "SCORECARD_QUEUED ${SRC_RUN} va=${VA} jobid=${OUT%%.*} ckpt=${CKPT}"
    else
        echo "  ERROR arm va=${VA} refused: ${OUT}"; log "ERROR va=${VA}: ${OUT}"
    fi
done
echo "SCORECARD_SUBMITTED"
echo "Read: 'validation loss' in \$MEMBER_ROOT/runs/makani_mn_scaling/score_${SRC_RUN}_va*.log"
echo "CHECK THE va=3 CONTROL FIRST -- it must reproduce the run's known number."
