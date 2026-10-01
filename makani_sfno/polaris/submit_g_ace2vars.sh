#!/bin/bash
# Port G production run: e3sm_alldata_ace2vars.yaml (83-in / 77-out, the ACE2-EAMv3
# variable set; operator 2026-09-29).
#   [SPLIT=production|2020] bash polaris/submit_g_ace2vars.sh <QUEUE> <NODES 1|2|4> <EPOCHS> <WALLTIME hh:mm:ss> [scratch|warm]
#
# SPLIT (operator 2026-09-30: "the validation year change was supposed to be a separate
# test ... separate the split into H, where it keeps G's removed channels, but changed the
# train years"):
#   production (default) = PORT G: the production pack, train 2015-2044, A's/F's stats.
#                          The variable set is the ONLY change from F.       runs g_ace2vars_*
#   2020                 = PORT H: G's channel set on the 2020-2044 train view
#                          (polaris_pack_alldata_trainview.pbs, own stats).  runs h_ace2vars_train2020_*
#                          Launch it as polaris/submit_h_train2020.sh. Valid 2045-47 and test
#                          2048-49 are the same in both, so G vs H isolates the train years.
#
# Sibling of submit_f_nosoil.sh (not edited). Recipe = F's = prod1n_b32_sgdr's
# (LR 2e-3, beta2 0.95, grad clip 32, CosineAnnealingWarmRestarts T0=20 Tmult=1,
# warmup 3, EMA 0.9995) at GLOBAL BATCH 32. What differs from F:
#   channel set 99 -> 77            (config; channel_subset_gate.py checks it in-job)
#   SPLIT=2020 only: train 2015-2044 -> 2020-2044, with the view's own stats
# EPOCHS = 3 + 20k (cycle boundary). Updates/epoch at batch 32: production 1368 (43,800
# samples, = A), view 1140 (36,500). 243 epochs = A's epoch count. A cost 46.3 node-h for
# 243 epochs at 1 node; G on the production pack should be close to that per epoch (the
# trunk dominates; the encoder/decoder width change is small) -- an estimate.
# warm = start from G_WARM_CKPT, A sliced to 83/77 by slice_checkpoint.py
# (--n-in 107 --n-out 101 --drop 2 3 4 5 8 9 64..81), fresh optimizer. On SPLIT=production
# A's stats ARE G's stats (same pack), as for F's surgical transfer; on SPLIT=2020 A saw
# the excluded 2015-19 years and learned under other stats.
#
# QUEUE has no default on purpose: capacity is max_run 1 per PROJECT and
# preemptable start latency is load-dependent (CLAUDE.md cluster table). The
# operator picks. DRY_RUN=1 prints the qsub and the provenance and submits nothing.
# DEPEND=<jobid> chains it afterany (G runs after F: DEPEND=7660250).
#
# FABRIC: inherits polaris_makani_multinode_scaling.pbs's CXI stack (aws-ofi-nccl
# v1.6.0, NCCL_PROTO=Simple, HPE rendezvous block) and refuses fabric overrides in
# its environment, as submit_f_nosoil.sh does.
# PASS token: G_QUEUED jobid=<id> run=<run>  (or G_DRY_RUN_OK)
set -euo pipefail
Q="${1:?QUEUE}"; NODES="${2:?NODES}"; EPOCHS="${3:?EPOCHS}"; WALL="${4:?WALLTIME}"; ROUTE="${5:-scratch}"
M=/eagle/projects/lighthouse-uchicago/members/mehta5
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPLIT="${SPLIT:-production}"
case "${SPLIT}" in
    production) VIEW="${G_PACK:-${M}/data/e3sm_makani_alldata_production}"; TRAIN0=2015; PORT=G; RUN_PREFIX=g_ace2vars ;;
    2020)       VIEW="${G_PACK:-${M}/data/e3sm_makani_alldata_train2020_2044}"; TRAIN0=2020; PORT=H; RUN_PREFIX=h_ace2vars_train2020 ;;
    *) echo "ERROR SPLIT must be production or 2020"; exit 2 ;;
esac

for v in OFI_PLUGIN OFI_LIBFABRIC NCCL_PROTO CXI_RDZV OFI_NCCL_PROGRESS_MODEL NCCL_NET FI_PROVIDER; do
    if [ -n "${!v:-}" ]; then echo "ERROR FABRIC_OVERRIDE_REFUSED: ${v}=${!v} is set; unset it"; exit 2; fi
done
case "${Q}" in capacity|preemptable|prod|debug-scaling|debug) ;; *) echo "ERROR QUEUE '${Q}' unknown"; exit 2;; esac
case "${NODES}" in 1|2|4) ;; *) echo "ERROR NODES must be 1, 2 or 4 (global batch 32)"; exit 2;; esac
case "${ROUTE}" in scratch|warm) ;; *) echo "ERROR ROUTE must be scratch or warm"; exit 2;; esac
LB=$(( 32 / (4 * NODES) ))
if [ $(( (EPOCHS - 3) % 20 )) -ne 0 ]; then echo "ERROR EPOCHS must be 3 + 20k (cycle boundary)"; exit 2; fi

# The pack must exist and BE the split SPLIT names (the view: polaris_pack_alldata_trainview.pbs).
python3 - "${VIEW}" "${TRAIN0}" <<'PY' || exit 2
import json, os, sys
root, train0 = sys.argv[1], int(sys.argv[2])
p = os.path.join(root, "metadata", "data.json")
if not os.path.isfile(p) or not os.path.isfile(os.path.join(root, "stats", "global_stds.npy")):
    print("ERROR PACK_MISSING: %s (the 2020 view: run polaris_pack_alldata_trainview.pbs first)" % root)
    sys.exit(2)
a = json.load(open(p))["attrs"]
if a["train_years"] != list(range(train0, 2045)) or a["valid_years"] != [2045, 2046, 2047] \
        or a["test_years"] != [2048, 2049]:
    print("ERROR PACK_WRONG_SPLIT: want train %d..2044, got train %s..%s valid %s test %s" % (
        train0, a["train_years"][0], a["train_years"][-1], a["valid_years"], a["test_years"]))
    sys.exit(2)
print("pack: %s  train %d-2044 (%d) / valid 2045-2047 / test 2048-2049" % (root, train0, 2045 - train0))
PY

RUN_NUM="${RUN_PREFIX}_${NODES}n_b32_e${EPOCHS}_${ROUTE}"
if [ -d "${M}/runs/makani_mn_scaling/e3sm_mn_scaling/${RUN_NUM}" ]; then
    echo "ERROR RUN_EXISTS ${RUN_NUM}: resuming would win over a fresh start; resume it deliberately instead"
    exit 2
fi
VARS="TARGET_NODES=${NODES},HPAR=1,WPAR=1,LOCAL_BATCH=${LB},FULL=1,EPOCHS=${EPOCHS},EVAL_SAMPLES=512,WANDB=1"
VARS="${VARS},RUN_NUM=${RUN_NUM},LR=2.0E-3,BETA2=0.95,MAX_GRAD_NORM=32"
VARS="${VARS},SCHED=CosineAnnealingWarmRestarts,SCHED_T0=20,SCHED_TMULT=1,SCHED_MIN_LR=1.0E-6"
VARS="${VARS},WARMUP_EPOCHS=3,LR_START=0.01,CKPT_VERSIONS=250,EMA=1,EMA_DECAY=0.9995"
VARS="${VARS},CONFIG_YAML=e3sm_alldata_ace2vars.yaml,PACK=${VIEW}"
VARS="${VARS},MAKANI_SCALING_CSV=${M}/bench/makani_production.csv"
if [ "${ROUTE}" = "warm" ]; then
    : "${G_WARM_CKPT:?warm route needs G_WARM_CKPT=<A sliced to 83/77>}"
    [ -f "${G_WARM_CKPT}" ] || { echo "ERROR SLICED_CKPT_MISSING: ${G_WARM_CKPT}"; exit 2; }
    VARS="${VARS},PRETRAINED_CKPT=${G_WARM_CKPT},LOAD_OPTIMIZER=0,LOAD_SCHEDULER=0,LOAD_COUNTERS=0,LOAD_LOSS=0,OVERRIDE_LR=1"
fi
# select = NODES + SPARE (default 1): the launcher's GPU preflight runs on the first NODES
# healthy. The spare is charged like any node. capacity's resources_max.nodect is 4
# (qstat -Qf, 2026-09-29), so 4 training nodes + a spare is refused there: pass SPARE=0.
SPARE="${SPARE:-1}"
case "${SPARE}" in 0|1) ;; *) echo "ERROR SPARE must be 0 or 1"; exit 2;; esac
SELECT=$(( NODES + SPARE ))
if [ "${Q}" = "capacity" ] && [ "${SELECT}" -gt 4 ]; then
    echo "ERROR CAPACITY_NODECT: select=${SELECT} > 4 (capacity's max); use SPARE=0 or fewer NODES"
    exit 2
fi

PROV_TXT="$(
    echo "run=${RUN_NUM}  submitted=$(date -u +%FT%TZ)  by=submit_g_ace2vars.sh  git=$(git -C "${HERE}" rev-parse --short HEAD)"
    echo "queue=${Q}  route=${ROUTE}  nodes=${NODES} (+${SPARE} spare)  local_batch=${LB}  global_batch=32  epochs=${EPOCHS}"
    echo "base recipe = F's (submit_f_nosoil.sh) = prod1n_b32_sgdr/config.json; the ONLY intended differences from F:"
    echo "  channel set 99 -> 77 (U10 RHREFHT PSL TMQ Z3_l00..17 also dropped; port G, operator 2026-09-29)"
    if [ "${SPLIT}" = "2020" ]; then
        echo "  PORT H (SPLIT=2020): G's channels, train 2015-2044 -> 2020-2044, stats recomputed (PACK=${VIEW})"
    else
        echo "  PORT G (SPLIT=production): train 2015-2044, production stats (PACK=${VIEW}) -- same data and stats as A and F"
    fi
    echo "  route ${ROUTE}${G_WARM_CKPT:+ (init ${G_WARM_CKPT})}"
    echo "vars: ${VARS}"
)"
echo "${PROV_TXT}"
# DEPEND=<jobid>: start only after that job ends, whatever its exit (afterany) --
# the operator's order is "G after F" (2026-09-29), e.g. DEPEND=7660250.
DEP=()
[ -n "${DEPEND:-}" ] && DEP=(-W "depend=afterany:${DEPEND}")
QSUB=(qsub -q "${Q}" "${DEP[@]}" -l "select=${SELECT}:system=polaris" -l "walltime=${WALL}"
      -v "${VARS}" polaris/polaris_makani_multinode_scaling.pbs)
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN (cd ${HERE} &&) ${QSUB[*]}"
    echo "G_DRY_RUN_OK run=${RUN_NUM}"
    exit 0
fi
echo "${PROV_TXT}" > "${M}/runs/makani_mn_scaling/${RUN_NUM}.provenance.txt"
OUT=$(cd "${HERE}" && "${QSUB[@]}" 2>&1)
echo "${OUT}"
[[ "${OUT}" == *".polaris-pbs"* ]] && echo "G_QUEUED jobid=${OUT%%.*} run=${RUN_NUM} queue=${Q} depend=${DEPEND:-none}" || { echo "ERROR SUBMIT_REFUSED"; exit 2; }
