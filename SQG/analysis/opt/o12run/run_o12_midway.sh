#!/bin/bash
# run_o12_midway.sh -- Job 3A of the Midway3 port (MIDWAY3_INROW_PROMPT.md §6 Level 2): o1.2-TrainUViT_m16_tendency.py
# ITSELF on one Midway3 node (4 x H100 NVL, one rank per GPU), short, once per solver: ref, lean_u1, lean32ek_fast
# (same-job baseline: ref), then the full-schedule extrapolation. Beside run_o12_polaris.pbs (job 7835278): same
# stages, knobs, limits and tokens; Midway scheduler header and env block; the val store is the 20 MiB copy of
# Polaris' (no build_val_store.py: Midway has no taxonomy frames), unless O12_REF_DIR names taxonomy frames.
# Predictions + gates: SQG/midway3/PREDICTIONS_midway3_timing.md (written first).
#   sbatch SQG/analysis/opt/o12run/run_o12_midway.sh                                     (from the repo root)
# 1. test_o12run.py (O12_TEST_OK).  2. the val store ($SQG_ROOT/val_store/{theta.npy,attrs.json}).
# 3. o12_polaris.py x 3 arms, each O12_RUN_DONE, each under a timeout.  4. o12_extrapolate.py (O12_EXTRAPOLATE_OK).
# End token: O12_JOB_DONE failures=N. Logs + extrapolation -> SQG/analysis/results/midway3_o12_<jobid>/;
# checkpoints stay in $SQG_ROOT/sqg_profile/o12_<jobid>/.
#SBATCH --job-name=sqg-o12
#SBATCH --account=pi-pedramh
#SBATCH --partition=pedramh-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=SQG/analysis/logs/midway3_o12_%j.out
#SBATCH --error=SQG/analysis/logs/midway3_o12_%j.err
set -eo pipefail
FAILURES=0
DONE_PRINTED=0
fail() { echo "ERROR $*"; FAILURES=$((FAILURES + 1)); }
finish() {
    rc=$?
    if [ "$DONE_PRINTED" -eq 0 ]; then
        [ "$rc" -ne 0 ] && { echo "ERROR O12_JOB_EXIT rc=$rc"; FAILURES=$((FAILURES + 1)); }
        echo "O12_JOB_DONE failures=$FAILURES"
    fi
}
trap finish EXIT
cd "${SLURM_SUBMIT_DIR}"
REPO="${SLURM_SUBMIT_DIR}"

# ---- env bootstrap (the same block in every SQG Midway3 script; no module is loaded: CUDA comes from the wheels) --
SQG_ROOT="${SQG_ROOT:-/project/rcc/mehta5/sqg_midway3}"           # everything writable; never Weidong's tree
case "$SQG_ROOT" in /project/pedramh/weidong/*) echo "ERROR SQG_ROOT_IN_WEIDONG_TREE $SQG_ROOT"; exit 2 ;; esac
[ -w "$SQG_ROOT" ] || { echo "ERROR SQG_NO_ROOT: $SQG_ROOT (run bash SQG/midway3/setup_sqg_midway.sh)"; exit 2; }
PY="${SQG_PY:-/project/pedramh/weidong/miniconda3/envs/wdenv/bin/python}"   # Option A: wdenv, read only
[ -x "$PY" ] || { echo "ERROR SQG_ENV_MISSING: $PY"; exit 3; }
case ":${LD_LIBRARY_PATH:-}:" in *cuda*|*cufft*)                  # a CUDA module would change which cuFFT loads
    echo "ERROR SQG_CUDA_ON_LD_LIBRARY_PATH ${LD_LIBRARY_PATH} (submit from a shell without module load cuda)"; exit 3 ;; esac
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=8 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR="${SQG_ROOT}/tmp" WANDB_MODE=offline PYTHONUNBUFFERED=1
export PIP_CACHE_DIR="${SQG_ROOT}/pip_cache"
export SQG_ASQG_PY="${REPO}/SQG/fixed_ring/A-SQG_data.py"
export SQG_TRAINING_ROOT="${SQG_TRAINING_ROOT:-${SQG_ROOT}/AllTrainData}"   # the solver mkdirs here
WARP_PATH="${WARP_PATH:-${SQG_ROOT}/pylibs/warp-1.17.0}"
NVMATH_PATH="${NVMATH_PATH:-${SQG_ROOT}/pylibs/nvmath-1.0.0}"
JOB="${SLURM_JOB_ID}"
# ------------------------------------------------------------------------------------------------------------------
# o1.2 remaps CUDA_VISIBLE_DEVICES to its LOCAL_RANK's GPU only when it lists several devices (o1.2 L34-35): SLURM's
# list for --gres=gpu:4 must have 4 entries
IFS=, read -r -a CVD <<< "${CUDA_VISIBLE_DEVICES:-}"
[ "${#CVD[@]}" = "4" ] || { echo "ERROR O12_NOT_4_GPUS CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"; exit 5; }
HERE=SQG/analysis/opt/o12run
RUN="${SQG_ROOT}/sqg_profile/o12_${JOB}"
RES="${REPO}/SQG/analysis/results/midway3_o12_${JOB}"
VAL="${O12_VAL_STORE:-${SQG_ROOT}/val_store}"
[ -e "$RUN" ] && { echo "ERROR O12_RUN_DIR_EXISTS $RUN"; exit 5; }
mkdir -p "$RUN" "$RES" "$TMPDIR" "$SQG_TRAINING_ROOT"
(cd SQG/fixed_ring && sha256sum -c --quiet SHA256SUMS) || { echo "ERROR SQG_SOLVER_SHA_MISMATCH"; exit 4; }
sha256sum SQG/o1.2-TrainUViT_m16_tendency.py "$HERE"/*.py | tee "$RUN/o12_source.sha256"
T0=$(date +%s)
echo "=== run_o12_midway on $(hostname) job $JOB: $(date '+%F %T') ==="
echo "host=$(hostname) job=$JOB py=$PY run=$RUN root=$SQG_TRAINING_ROOT val=$VAL ref=${O12_REF_DIR:-<none>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" | tee "$RUN/ld_library_path.txt"
echo "co-tenants on this node: $(squeue -h -w "$(hostname -s)" -o '%i:%u:%b' | tr '\n' ' ')" | tee "$RUN/cotenants.txt"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv \
    | tee "$RUN/gpu.csv"
nvidia-smi topo -m > "$RUN/topo.txt" 2>&1 || true

# ---- env: libraries loaded + mathdx (no timing) ---------------------------------------------------------------------
"$PY" SQG/midway3/midway_env_extra.py --out "$RUN" --warp-path "$WARP_PATH" --nvmath-path "$NVMATH_PATH" \
    --expect-gpus 4 2>&1 | tee "$RUN/midway_env_extra.log" || fail O12_ENV
grep -q "^SQG_MIDWAY_ENV_OK" "$RUN/midway_env_extra.log" || fail O12_ENV_NO_TOKEN

# ---- 1. tests -------------------------------------------------------------------------------------------------------
"$PY" "$HERE/test_o12run.py" 2>&1 | tee "$RUN/test_o12run.log" || fail O12_TEST
grep -q "O12_TEST_OK" "$RUN/test_o12run.log" || fail O12_TEST_NO_TOKEN

# ---- 2. the val store (copied from Polaris o12tr_7853365) -----------------------------------------------------------
for f in theta.npy attrs.json; do
    [ -s "$VAL/$f" ] || fail "O12_VAL_STORE_MISSING $VAL/$f (copy Polaris' o12tr_7853365/val_store here)"
done
[ -e "$VAL/theta.npy" ] && sha256sum "$VAL"/theta.npy "$VAL"/attrs.json | tee "$RUN/val_store.sha256"

# ---- 3. o1.2, one arm per solver (timeouts keep the three inside the hour) -----------------------------------------
export O12_VAL_STORE="$VAL" O12_MEMBERS=1,2,3,4 \
       O12_BATCH=16 O12_START_PAIRS=600 O12_N_EPOCH=4 O12_STEPS_PER_EPOCH=30 O12_N_VAL=64 O12_BUF=16384
[ -n "${O12_REF_DIR:-}" ] && export O12_REF_DIR                    # unset: O12_BITWISE ... NO_REFERENCE (expected)
declare -A LIMIT=([ref]=1440 [lean_u1]=1020 [lean32ek_fast]=600)
LOGS=()
if [ "$FAILURES" = "0" ]; then
    for s in ref lean_u1 lean32ek_fast; do
        echo "=== arm $s: $(date '+%F %T') (limit ${LIMIT[$s]} s)"
        mkdir -p "$RUN/$s"
        set +e
        O12_SOLVER=$s O12_OUT="$RUN/$s" timeout --signal=TERM --kill-after=30 "${LIMIT[$s]}" \
            "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 --max_restarts=0 "$HERE/o12_polaris.py" \
            > "$RUN/$s.log" 2> "$RUN/$s.err"
        rc=$?
        set -e
        [ "$rc" = "0" ] || fail "O12_ARM_RC solver=$s rc=$rc (124 = timeout)"
        grep -q "O12_RUN_DONE solver=$s" "$RUN/$s.log" || { fail "O12_ARM_NO_TOKEN solver=$s"; tail -n 25 "$RUN/$s.err"; }
        grep -E "^(O12_|ERROR|epoch [0-9]+:|peak GPU|ConvNextUViT|buffer filled|strict fp32)" "$RUN/$s.log" \
            | grep -v "^O12_KNOB" || true
        LOGS+=(--log "$s=$RUN/$s.log")
    done
else
    echo "ERROR O12_ARMS_SKIPPED (env, tests or val store failed)"
fi

# ---- 4. extrapolation -----------------------------------------------------------------------------------------------
if [ "${#LOGS[@]}" -gt 0 ]; then
    "$PY" "$HERE/o12_extrapolate.py" "${LOGS[@]}" --out "$RES" 2>&1 | tee "$RUN/o12_extrapolate.log" \
        || fail O12_EXTRAPOLATE
    grep -q "O12_EXTRAPOLATE_OK" "$RUN/o12_extrapolate.log" || fail O12_EXTRAPOLATE_NOT_OK
fi

for f in "$RUN"/*.log "$RUN"/*.err "$RUN"/*.txt "$RUN"/gpu.csv "$RUN"/*.sha256 "$RUN"/midway_env_extra.json; do
    if [ -e "$f" ]; then cp "$f" "$RES"/ || fail "O12_COPY $f"; fi
done
for s in ref lean_u1 lean32ek_fast; do
    if [ -e "$RUN/$s/wandb_shim.jsonl" ]; then cp "$RUN/$s/wandb_shim.jsonl" "$RES/${s}_wandb_shim.jsonl" || true; fi
done
du -sh "$RUN"/*/model_ckpt 2>/dev/null || true
echo "=== end $(date '+%F %T') ($(( $(date +%s) - T0 )) s)"
DONE_PRINTED=1
echo "O12_JOB_DONE failures=$FAILURES"
