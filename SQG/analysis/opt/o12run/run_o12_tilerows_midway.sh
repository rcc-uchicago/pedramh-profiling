#!/bin/bash
# run_o12_tilerows_midway.sh -- Jobs 3B and 3C of the Midway3 port (MIDWAY3_INROW_PROMPT.md §6 Level 2): o1.2 ITSELF
# with the IF / tile-row / in-row solvers, on one Midway3 node (4 x H100 NVL), short, once per arm, O12_TR_GATE=1;
# then the full-schedule extrapolation. Beside run_o12_tilerows_polaris.pbs (jobs 7840909, 7853365): same stages,
# knobs, 480 s arm limit and tokens; Midway scheduler header and env block; the cuFFT 11.4.1 preload arm is dropped
# (Polaris-only toolkit path); the val store is the copy of Polaris' (as run_o12_midway.sh).
# Predictions + gates: SQG/midway3/PREDICTIONS_midway3_timing.md (written first).
#   sbatch SQG/analysis/opt/o12run/run_o12_tilerows_midway.sh                                  (3B: the 3 default arms)
#   sbatch --export=ALL,O12TR_ARMS=ref_b32+tilerows_inrow_if25_b32 SQG/analysis/opt/o12run/run_o12_tilerows_midway.sh
#                                                                                             (3C: batch 32)
# Arms: O12TR_ARMS ('+'- or space-separated) or the default "if32_25 tilerows_if25 tilerows_inrow_if25" (same-job
# baseline: tilerows_if25). Named variants: <solver>_b32 = O12_BATCH=32 (o1.2's own BATCH_SIZE; fits 94 GB);
# tilerows_inrow_if25_tf32 / _bf16 = O12_PREC (optional Run 3 only: ask the user first); tilerows_if25_scatter_t /
# _concat_t = O12_TR_EXPAND + O12_TR_TIMING. Limits: ref* 1440 s, lean_u1* 1020 s, lean32ek_fast* 600 s, else 480 s
# (Polaris' per-arm limits).
# 1. test_o12run.py + test_tilerows_advance.py + the model import.  2. the val store.  3. o12_polaris.py per arm.
# 4. o12_extrapolate.py. End token (every exit path): O12TR_JOB_DONE failures=N.
# Logs + extrapolation -> SQG/analysis/results/midway3_o12tr_<jobid>/; checkpoints stay in $SQG_ROOT/sqg_profile/.
#SBATCH --job-name=sqg-o12tr
#SBATCH --account=pi-pedramh
#SBATCH --partition=pedramh-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=SQG/analysis/logs/midway3_o12tr_%j.out
#SBATCH --error=SQG/analysis/logs/midway3_o12tr_%j.err
set -eo pipefail
FAILURES=0
DONE_PRINTED=0
fail() { echo "ERROR $*"; FAILURES=$((FAILURES + 1)); }
finish() {
    rc=$?
    if [ "$DONE_PRINTED" -eq 0 ]; then
        [ "$rc" -ne 0 ] && { echo "ERROR O12TR_JOB_EXIT rc=$rc"; FAILURES=$((FAILURES + 1)); }
        echo "O12TR_JOB_DONE failures=$FAILURES"
    fi
}
trap finish EXIT
cd "${SLURM_SUBMIT_DIR}"
REPO="${SLURM_SUBMIT_DIR}"
SRC="${O12_SRC:-${REPO}}"

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
export SQG_ASQG_PY="${SRC}/SQG/fixed_ring/A-SQG_data.py"
export SQG_TRAINING_ROOT="${SQG_TRAINING_ROOT:-${SQG_ROOT}/AllTrainData}"   # the solver mkdirs here
WARP_PATH="${WARP_PATH:-${SQG_ROOT}/pylibs/warp-1.17.0}"
NVMATH_PATH="${NVMATH_PATH:-${SQG_ROOT}/pylibs/nvmath-1.0.0}"
JOB="${SLURM_JOB_ID}"
# ------------------------------------------------------------------------------------------------------------------
# o1.2 remaps CUDA_VISIBLE_DEVICES to its LOCAL_RANK's GPU only when it lists several devices (o1.2 L34-35)
IFS=, read -r -a CVD <<< "${CUDA_VISIBLE_DEVICES:-}"
[ "${#CVD[@]}" = "4" ] || { echo "ERROR O12_NOT_4_GPUS CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"; exit 5; }
HERE=SQG/analysis/opt/o12run
RUN="${SQG_ROOT}/sqg_profile/o12tr_${JOB}"
RES="${REPO}/SQG/analysis/results/midway3_o12tr_${JOB}"
VAL="${O12_VAL_STORE:-${SQG_ROOT}/val_store}"
[ -e "$RUN" ] && { echo "ERROR O12_RUN_DIR_EXISTS $RUN"; exit 5; }
mkdir -p "$RUN" "$RES" "$TMPDIR" "$SQG_TRAINING_ROOT"
cd "$SRC"
(cd SQG/fixed_ring && sha256sum -c --quiet SHA256SUMS) || { echo "ERROR SQG_SOLVER_SHA_MISMATCH"; exit 4; }
if [ "$SRC" != "$REPO" ]; then
    sha256sum -c --quiet MANIFEST.sha256 || { echo "ERROR O12TR_SNAPSHOT_MANIFEST"; exit 4; }
    cp MANIFEST.sha256 GIT_HEAD "$RUN"/
fi
sha256sum SQG/o1.2-TrainUViT_m16_tendency.py "$HERE"/*.py | tee "$RUN/o12_source.sha256"
export O12_WARP_PATH="$WARP_PATH" O12_NVMATH_PATH="$NVMATH_PATH"
for d in "$O12_WARP_PATH/warp" "$O12_NVMATH_PATH/nvmath"; do
    [ -d "$d" ] || { echo "ERROR O12TR_LIB_MISSING $d"; exit 5; }
done
T0=$(date +%s)
echo "=== run_o12_tilerows_midway on $(hostname) job $JOB: $(date '+%F %T') src=$SRC ==="
echo "host=$(hostname) job=$JOB py=$PY run=$RUN root=$SQG_TRAINING_ROOT val=$VAL ref=${O12_REF_DIR:-<none>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" | tee "$RUN/ld_library_path.txt"
echo "co-tenants on this node: $(squeue -h -w "$(hostname -s)" -o '%i:%u:%b' | tr '\n' ' ')" | tee "$RUN/cotenants.txt"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv \
    | tee "$RUN/gpu.csv"
nvidia-smi topo -m > "$RUN/topo.txt" 2>&1 || true

# ---- env: libraries loaded + mathdx (no timing) ---------------------------------------------------------------------
"$PY" SQG/midway3/midway_env_extra.py --out "$RUN" --warp-path "$WARP_PATH" --nvmath-path "$NVMATH_PATH" \
    --expect-gpus 4 2>&1 | tee "$RUN/midway_env_extra.log" || fail O12TR_ENV
grep -q "^SQG_MIDWAY_ENV_OK" "$RUN/midway_env_extra.log" || fail O12TR_ENV_NO_TOKEN

# ---- 1. tests -------------------------------------------------------------------------------------------------------
"$PY" "$HERE/test_o12run.py" 2>&1 | tee "$RUN/test_o12run.log" || fail O12_TEST
grep -q "O12_TEST_OK" "$RUN/test_o12run.log" || fail O12_TEST_NO_TOKEN
"$PY" "$HERE/test_tilerows_advance.py" 2>&1 | tee "$RUN/test_tilerows_advance.log" || fail O12_TR_TEST
grep -q "O12_TR_TEST_OK" "$RUN/test_tilerows_advance.log" || fail O12_TR_TEST_NO_TOKEN
# o1.2 imports its model from its own directory (job 7840681: the snapshot lacked model_UViT.py, every arm died)
(cd SQG && "$PY" -c "import model_UViT; print('O12_MODEL_IMPORT_OK')") 2>&1 | tee "$RUN/model_import.log" \
    || fail O12_MODEL_IMPORT
grep -q "O12_MODEL_IMPORT_OK" "$RUN/model_import.log" || fail O12_MODEL_IMPORT_NO_TOKEN

# ---- 2. the val store (copied from Polaris o12tr_7853365) -----------------------------------------------------------
for f in theta.npy attrs.json; do
    [ -s "$VAL/$f" ] || fail "O12_VAL_STORE_MISSING $VAL/$f (copy Polaris' o12tr_7853365/val_store here)"
done
[ -e "$VAL/theta.npy" ] && sha256sum "$VAL"/theta.npy "$VAL"/attrs.json | tee "$RUN/val_store.sha256"

# ---- 3. o1.2, one arm per solver ------------------------------------------------------------------------------------
export O12_VAL_STORE="$VAL" O12_MEMBERS=1,2,3,4 \
       O12_BATCH=16 O12_START_PAIRS=600 O12_N_EPOCH=4 O12_STEPS_PER_EPOCH=30 O12_N_VAL=64 O12_BUF=16384 \
       O12_TR_GATE=1 O12_WARP_CACHE="$RUN/warp_cache"
[ -n "${O12_REF_DIR:-}" ] && export O12_REF_DIR                    # unset: O12_BITWISE ... NO_REFERENCE (expected)
ARMS_SPEC="${O12TR_ARMS:-if32_25 tilerows_if25 tilerows_inrow_if25}"
read -r -a ARMS <<< "${ARMS_SPEC//+/ }"
LOGS=()
if [ "$FAILURES" = "0" ]; then
    for a in "${ARMS[@]}"; do
        s="$a"; extra=""
        case "$a" in
            *_b32) s="${a%_b32}"; extra="O12_BATCH=32" ;;
            tilerows_if25_scatter_t) s=tilerows_if25; extra="O12_TR_EXPAND=scatter O12_TR_TIMING=1" ;;
            tilerows_if25_concat_t) s=tilerows_if25; extra="O12_TR_EXPAND=concat O12_TR_TIMING=1" ;;
            # the network's precision (mixprec/PREDICTIONS_mixprec.md, probe 2); the bare name is o1.2's ieee
            tilerows_inrow_if25_tf32) s=tilerows_inrow_if25; extra="O12_PREC=tf32" ;;
            tilerows_inrow_if25_bf16) s=tilerows_inrow_if25; extra="O12_PREC=bf16" ;;
        esac
        case "$s" in ref) LIMIT=1440 ;; lean_u1) LIMIT=1020 ;; lean32ek_fast) LIMIT=600 ;; *) LIMIT=480 ;; esac
        echo "=== arm $a (solver $s, env ${extra:-none}): $(date '+%F %T') (limit $LIMIT s)"
        mkdir -p "$RUN/$a"
        set +e
        env $extra O12_SOLVER=$s O12_OUT="$RUN/$a" timeout --signal=TERM --kill-after=30 "$LIMIT" \
            "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 --max_restarts=0 "$HERE/o12_polaris.py" \
            > "$RUN/$a.log" 2> "$RUN/$a.err"
        rc=$?
        set -e
        [ "$rc" = "0" ] || fail "O12_ARM_RC arm=$a rc=$rc (124 = timeout)"
        grep -q "O12_RUN_DONE solver=$s" "$RUN/$a.log" || { fail "O12_ARM_NO_TOKEN arm=$a"; tail -n 25 "$RUN/$a.err"; }
        grep -E "^(O12_|ERROR|epoch [0-9]+:|peak GPU|ConvNextUViT|buffer filled|strict fp32)" "$RUN/$a.log" \
            | grep -v "^O12_KNOB" || true
        LOGS+=(--log "$a=$RUN/$a.log")
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

for f in "$RUN"/*.log "$RUN"/*.err "$RUN"/*.txt "$RUN"/gpu.csv "$RUN"/*.sha256 "$RUN"/midway_env_extra.json \
         "$RUN"/MANIFEST.sha256 "$RUN"/GIT_HEAD; do
    if [ -e "$f" ]; then cp "$f" "$RES"/ || fail "O12TR_COPY $f"; fi
done
for a in "${ARMS[@]}"; do
    if [ -e "$RUN/$a/wandb_shim.jsonl" ]; then cp "$RUN/$a/wandb_shim.jsonl" "$RES/${a}_wandb_shim.jsonl" || true; fi
done
echo "=== end $(date '+%F %T') ($(( $(date +%s) - T0 )) s)"
DONE_PRINTED=1
echo "O12TR_JOB_DONE failures=$FAILURES"
