#!/bin/bash
# The first kernel-level profile of makani (TODO P1 item 8).
#
# WHY. Every makani measurement to date is SYSTEM-level: step time, throughput,
# weak-scaling efficiency, peak memory, I/O rate, placement, batch occupancy --
# 84 rows across 12 CSVs, and **zero** with `nsys` set. That tells us the run
# does 16.0 samples/second/GPU. It cannot tell us whether that is near the
# hardware limit or a factor of three off it, because nothing has ever measured
# WHERE THE TIME GOES.
#
# The precedent on the same A100s is not encouraging: PanguWeather's capture
# found **271 ms/rank-step, 47 percent of compute time, in `direct_copy` and
# `conj` -- kernels that compute nothing** (68 percent of compute in pointwise
# operations against 17 percent in GEMM). If makani carries anything similar,
# a real fraction of the 46-hour production run is recoverable WITHOUT changing
# what the model computes (DESIGN §4 still gates any such change on numerical
# equivalence).
#
# CONFIGURATION = THE PRODUCTION ONE. 1 node, 4 ranks, LOCAL_BATCH 8 (global
# batch 32), h1w1 pure data parallelism, 101-channel ALLDATA pack, 147.9 M
# parameters, `scale_factor` 3 (trunk 60x120). Profiling anything else would
# answer a question nobody asked.
#
# ⚠ FULL=0 ON PURPOSE (do not "fix" this). With FULL=0 an epoch is
# STEPS x GLOBAL_BATCH samples, so the job reaches the capture window in under a
# minute. FULL=1 would make it a 683-second epoch for no benefit -- the capture
# is 10 steps either way.
#
# ⚠ The capture window is steps 30-40 (`STEPS/2` to `STEPS/2 + 10`), which is
# past dataloader warmup. Trap 1 in the bench report exists because the FIRST
# steps of a run are not representative.
#
# ⚠ ITS CSV ROW IS TRUNCATED BY DESIGN and goes to a SEPARATE FILE. makani's
# CUDAProfiler calls `sys.exit(0)` at `capture_range_stop`, so the run ends
# mid-epoch: `total_train_s` is empty and the step average covers fewer steps,
# under profiler overhead besides. It must never be averaged with a clean row
# (CLAUDE.md #10), hence MAKANI_SCALING_CSV points at its own file and the
# `nsys` column marks it.
#
# WANDB=0: a profiled run's step_ms carries both profiler and diagnostics
# overhead and belongs in no comparison.
#
# Output: $MEMBER_ROOT/bench/nsys_makani_mn_nsys_prod_b32/rank_{0..3}.nsys-rep
# PASS token: MAKANI_NSYS_QUEUED jobid=<id>
set -u

MEMBER_ROOT=/eagle/projects/lighthouse-uchicago/members/mehta5
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${MEMBER_ROOT}/polaris_logs/makani_nsys_profile.log"
TAG="${TAG:-nsys_prod_b32}"
QUEUE="${QUEUE:-debug}"
WALL="${WALLTIME:-01:00:00}"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "${LOG}"; }
mkdir -p "$(dirname "${LOG}")"

V="TARGET_NODES=1,HPAR=1,WPAR=1,LOCAL_BATCH=8,STEPS=60,NSYS=1"
V="${V},FULL=0,EVAL_SAMPLES=8,WANDB=0,RUN_NUM=${TAG}"
V="${V},MAKANI_SCALING_CSV=${MEMBER_ROOT}/bench/makani_nsys.csv"
V="${V},CONFIG_YAML=e3sm_alldata_full.yaml"
V="${V},PACK=${MEMBER_ROOT}/data/e3sm_makani_alldata_production"
V="${V},OFI_PLUGIN=${MEMBER_ROOT}/sw/aws-ofi-nccl-1.21.1/lib"
V="${V},OFI_NCCL_PROGRESS_MODEL=AUTO,NCCL_PROTO=Simple"

OUT=$(cd "${HERE}" && qsub -q "${QUEUE}" \
        -l select=1:system=polaris -l walltime="${WALL}" -l filesystems=home:eagle \
        -v "${V}" polaris/polaris_makani_multinode_scaling.pbs 2>&1)
if [[ "${OUT}" == *".polaris-pbs"* ]]; then
    echo "MAKANI_NSYS_QUEUED jobid=${OUT%%.*} tag=${TAG} queue=${QUEUE}"
    log  "MAKANI_NSYS_QUEUED jobid=${OUT%%.*} tag=${TAG}"
else
    echo "ERROR nsys submission refused: ${OUT}"; log "ERROR refused: ${OUT}"; exit 3
fi
echo
echo "Expect in the log:  'nsys: <dir>  capture steps 30..40  (all 4 ranks)'"
echo "Then 4 reports:     ${MEMBER_ROOT}/bench/nsys_makani_mn_${TAG}/rank_{0,1,2,3}.nsys-rep"
echo "The job ENDS AT STEP 40 with rc=0 -- that is makani's CUDAProfiler calling"
echo "sys.exit(0), not a failure. Key on the .nsys-rep files existing."
