#!/bin/bash
# Fork the RUNNING production run at a fixed checkpoint and A/B batch 32 vs 48
# from IDENTICAL weights, to test whether more samples per update helps LATE in
# training -- the regime the early sweeps could not reach.
#
# WHY THIS EXPERIMENT. At equal DATA the early arms say batch 48 is ~10% worse
# (bench report §5j/§7). But at equal UPDATES batch 48 is BETTER -- -55.1% at
# 1824 steps, -12.3% at 2736, -3.1% at 4104 -- i.e. more samples per update does
# help, just sub-linearly and with a fast-decaying benefit. Those numbers cover
# 5,472 of 332,424 planned steps (1.6%), and gradient noise matters LEAST early
# and MOST late. So the decaying curve cannot be extrapolated into the regime
# that actually matters. This forks at epoch ~70 to sample that regime directly.
#
# ⚠⚠ DOES NOT TOUCH PRODUCTION. Both arms get a fresh RUN_NUM and a fresh expDir;
# the checkpoint is COPIED, not linked or moved. Reusing production's RUN_NUM
# would put two jobs in one expDir and corrupt the run -- never do that.
#
# The b32 arm is not redundant with production: it runs under the SAME I/O
# contention as the b48 arm, and it doubles as a fidelity check that the fork
# reproduces production's own trajectory.
#
# SCHEDULE IS PINNED TO PRODUCTION'S (lr 2.0E-3, warmup 3, T_0=20, min_lr 1e-6).
# The scheduler state comes back from the checkpoint, so a different T_0 or
# warmup here would make the restored state inconsistent and the comparison
# meaningless. Do not "simplify" these.
#
# Comparison is at EQUAL EPOCHS = equal data (43,776 samples/epoch either way),
# which is the deployment-relevant axis. Equal-step numbers fall out of the same
# logs (b32 1368 steps/epoch, b48 912).
#
# Usage: bash polaris/submit_batch_fork_ab.sh [n_epochs] [queue]
# PASS token: FORK_ARM_QUEUED <tag> jobid=<id>
set -u

MEMBER_ROOT=/eagle/projects/lighthouse-uchicago/members/mehta5
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${MEMBER_ROOT}/polaris_logs/makani_batch_fork_ab.log"
SRC="${MEMBER_ROOT}/runs/makani_mn_scaling/e3sm_mn_scaling/prod1n_b32_sgdr"
DSTROOT="${MEMBER_ROOT}/runs/makani_mn_scaling/e3sm_mn_scaling"
NEPOCH="${1:-3}"
QUEUE="${2:-debug}"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "${LOG}"; }
mkdir -p "$(dirname "${LOG}")"

# ---- pin ONE checkpoint version; production keeps writing while we work ------
# Second-newest, not newest: the newest may still be mid-write.
VERS=$(ls -1 "${SRC}/training_checkpoints"/ckpt_mp0_v*.tar 2>/dev/null \
        | sed 's/.*_v//; s/\.tar//' | sort -n | tail -2 | head -1)
if [ -z "${VERS}" ]; then echo "ERROR NO_SOURCE_CHECKPOINT: ${SRC}"; exit 3; fi
CKPT="${SRC}/training_checkpoints/ckpt_mp0_v${VERS}.tar"
END=$(( VERS + NEPOCH ))
echo "forking from epoch ${VERS}; each arm runs to max_epochs=${END} (${NEPOCH} epochs)"
log "fork source: ${CKPT} -> max_epochs=${END}"

COMMON="TARGET_NODES=1,HPAR=1,WPAR=1,FULL=1,EVAL_SAMPLES=512,WANDB=1"
COMMON="${COMMON},LR=2.0E-3,SCHED=CosineAnnealingWarmRestarts,SCHED_T0=20,SCHED_TMULT=1"
COMMON="${COMMON},SCHED_MIN_LR=1.0E-6,WARMUP_EPOCHS=3,LR_START=0.01,CKPT_VERSIONS=250"
COMMON="${COMMON},EPOCHS=${END}"
COMMON="${COMMON},MAKANI_SCALING_CSV=${MEMBER_ROOT}/bench/makani_batch_fork.csv"
COMMON="${COMMON},CONFIG_YAML=e3sm_alldata_full.yaml"
COMMON="${COMMON},PACK=${MEMBER_ROOT}/data/e3sm_makani_alldata_production"
COMMON="${COMMON},OFI_PLUGIN=${MEMBER_ROOT}/sw/aws-ofi-nccl-1.21.1/lib"
COMMON="${COMMON},OFI_NCCL_PROGRESS_MODEL=AUTO,NCCL_PROTO=Simple"

# tag|LOCAL_BATCH  (global = 4x)
for spec in "fork_ep${VERS}_b32|8" "fork_ep${VERS}_b48|12"; do
    IFS='|' read -r TAG LB <<< "${spec}"
    DST="${DSTROOT}/${TAG}"
    if [ -d "${DST}" ]; then
        echo "ERROR FORK_DIR_EXISTS: ${DST} -- refusing to reuse (would resume, not fork)"
        log   "ERROR FORK_DIR_EXISTS: ${DST}"; exit 4
    fi
    mkdir -p "${DST}/training_checkpoints" || exit 4
    # ⚠⚠ THE SEEDED FILE MUST BE NAMED v0, NOT ITS TRUE VERSION. makani gates
    # `resuming` on the existence of EXACTLY `ckpt_mp0_v0.tar`
    # (makani/train.py:101-105 -- `checkpoint_version=0`, hardcoded), NOT on the
    # newest file. Seeding this dir as ckpt_mp0_v71.tar gave `resuming = False`
    # and the arm silently began training FROM SCRATCH -- no error, no warning,
    # and the loss curve looks like a fresh run rather than a broken fork
    # (measured 2026-09-03, jobs 7588049/7588050, killed).
    # The epoch is NOT taken from the filename: restore_from_checkpoint reads the
    # counters stored inside the tar, and get_latest_checkpoint_version
    # (checkpoint_helpers.py:33-42) picks by MTIME over the glob, so a lone v0
    # resolves correctly and the run continues at the checkpoint's real epoch.
    cp "${CKPT}" "${DST}/training_checkpoints/ckpt_mp0_v0.tar" || exit 4
    for f in config.json metadata.json global_means.npy global_stds.npy; do
        [ -f "${SRC}/${f}" ] && cp "${SRC}/${f}" "${DST}/${f}"
    done
    # ⚠ -v is LAST-WINS: per-arm after COMMON.
    EXTRA="LOCAL_BATCH=${LB},RUN_NUM=${TAG}"
    OUT=$(cd "${HERE}" && qsub -q "${QUEUE}" \
            -l select=1:system=polaris -l walltime=01:00:00 \
            -v "${COMMON},${EXTRA}" \
            polaris/polaris_makani_multinode_scaling.pbs 2>&1)
    if [[ "${OUT}" == *".polaris-pbs"* ]]; then
        echo "FORK_ARM_QUEUED ${TAG} global_batch=$((LB*4)) jobid=${OUT%%.*}"
        log  "FORK_ARM_QUEUED ${TAG} global_batch=$((LB*4)) jobid=${OUT%%.*}"
    else
        echo "ERROR arm ${TAG} refused: ${OUT}"; log "ERROR arm ${TAG} refused: ${OUT}"
    fi
done
echo "BATCH_FORK_SUBMITTED"
