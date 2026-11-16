#!/bin/bash
# run_inrow_midway.sh -- Job 2 of the Midway3 port (MIDWAY3_INROW_PROMPT.md §6-7): the solver step on one H100 NVL,
# timed as Polaris timed it. Beside run_inrow_polaris.pbs: same stages and tokens; the Polaris-only cuFFT 11.4.1
# preload pass is dropped, seed 1 (run_round1_polaris.pbs's ref / lean / lean_u1 timing, job 7814300) is added.
# Predictions + gates: SQG/midway3/PREDICTIONS_midway3_timing.md, and inrow/PREDICTIONS_inrow.md (unchanged).
#   python3 SQG/analysis/opt/stability/snapshot_src.py <snap>                    (optional: freeze the code)
#   sbatch [--export=ALL,INROW_SRC=<snap>] SQG/analysis/opt/inrow/run_inrow_midway.sh      (from the repo root)
# env. midway_env_extra.py --expect-gpus 1 (libraries loaded, Warp mathdx = 1).
# 0. test_inrow_math.py + test_tilerows_math.py.  0b. seed1/run_seed1.py --budget-min 14 (SQG_INIT=random).
# 1. inrow_probe.py --tag prod (production cuFFT: deciding), block_dim 256 as on A100.  2. score_inrow.py, then
# SQG/midway3/score_midway3_timing.py --inrow (Level 1 only; information until the o1.2 jobs exist).
# End token (every exit path): IR_JOB_DONE failures=N. Results -> SQG/analysis/results/midway3_inrow_<jobid>/.
#SBATCH --job-name=sqg-inrow
#SBATCH --account=pi-pedramh
#SBATCH --partition=pedramh-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=SQG/analysis/logs/midway3_inrow_%j.out
#SBATCH --error=SQG/analysis/logs/midway3_inrow_%j.err
set -eo pipefail
FAILURES=0
DONE_PRINTED=0
fail() { echo "ERROR $*"; FAILURES=$((FAILURES + 1)); }
finish() {
    rc=$?
    if [ "$DONE_PRINTED" -eq 0 ]; then
        [ "$rc" -ne 0 ] && { echo "ERROR IR_JOB_EXIT rc=$rc"; FAILURES=$((FAILURES + 1)); }
        echo "IR_JOB_DONE failures=$FAILURES"
    fi
}
trap finish EXIT
cd "${SLURM_SUBMIT_DIR}"
REPO="${SLURM_SUBMIT_DIR}"
SRC="${INROW_SRC:-${REPO}}"

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
export SQG_INIT="${SQG_INIT:-random}"                              # seed 1 times the random state, as job 7814300
export WARP_CACHE_PATH="${SQG_ROOT}/tmp/warp_cache_${JOB}"
T0=$(date +%s)
OUT="${SQG_ROOT}/sqg_profile/inrow_${JOB}"
RES="${REPO}/SQG/analysis/results/midway3_inrow_${JOB}"
mkdir -p "$OUT" "$RES" "$TMPDIR" "$SQG_TRAINING_ROOT" "$WARP_CACHE_PATH"
cd "$SRC"
(cd SQG/fixed_ring && sha256sum -c --quiet SHA256SUMS) || { echo "ERROR SQG_SOLVER_SHA_MISMATCH"; exit 4; }
if [ "$SRC" != "$REPO" ]; then
    sha256sum -c --quiet MANIFEST.sha256 || { echo "ERROR IR_SNAPSHOT_MANIFEST"; exit 4; }
    cp MANIFEST.sha256 GIT_HEAD "$OUT"/
fi
HERE=SQG/analysis/opt/inrow
echo "=== run_inrow_midway on $(hostname) job $JOB: $(date '+%F %T') src=$SRC ==="
echo "host=$(hostname) job=$JOB py=$PY root=$SQG_ROOT training_root=$SQG_TRAINING_ROOT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" | tee "$OUT/ld_library_path.txt"
echo "co-tenants on this node: $(squeue -h -w "$(hostname -s)" -o '%i:%u:%b' | tr '\n' ' ')" | tee "$OUT/cotenants.txt"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv \
    | tee "$OUT/gpu.csv"
nvidia-smi topo -m > "$OUT/topo.txt" 2>&1 || true
sha256sum "$HERE"/*.py "$HERE"/PREDICTIONS_inrow.md SQG/analysis/opt/tilerows/tilerows_step.py \
    SQG/analysis/opt/seed1/*.py SQG/midway3/PREDICTIONS_midway3_timing.md | tee "$OUT/sha256.txt"
for d in "$WARP_PATH/warp" "$NVMATH_PATH/nvmath"; do
    [ -d "$d" ] || { echo "ERROR IR_LIB_MISSING $d"; exit 5; }
done
LABEL_TAIL="_frozen_ring_NU_B_v2_nu0.0002_B1_mu1_kf3.5_dkf0.5_seed1_N4096_dt5e-06_Tstart0_Tspin20_Tprod1100_save0.0025_pair0.02"
CKPT="${SQG_TRAINING_ROOT}/trainingdata_m01${LABEL_TAIL}/checkpoint/ckpt_t0100.00.npz"
[ -s "$CKPT" ] || { echo "ERROR IR_CKPT_NOT_STAGED $CKPT"; exit 5; }

# ---- env: libraries loaded + mathdx (no timing) ---------------------------------------------------------------------
"$PY" SQG/midway3/midway_env_extra.py --out "$OUT" --warp-path "$WARP_PATH" --nvmath-path "$NVMATH_PATH" \
    --expect-gpus 1 2>&1 | tee "$OUT/midway_env_extra.log" || fail IR_ENV
grep -q "^SQG_MIDWAY_ENV_OK" "$OUT/midway_env_extra.log" || fail IR_ENV_NO_TOKEN

# ---- 0. math tests --------------------------------------------------------------------------------------------------
"$PY" "$HERE/test_inrow_math.py" 2>&1 | tee "$OUT/test_inrow_math.log" || fail INROW_MATH_TEST
grep -q "INROW_MATH_TEST_OK" "$OUT/test_inrow_math.log" || fail INROW_MATH_TEST_NO_TOKEN
"$PY" SQG/analysis/opt/tilerows/test_tilerows_math.py 2>&1 | tee "$OUT/test_tilerows_math.log" || fail TR_MATH_TEST
grep -q "TILEROWS_MATH_TEST_OK" "$OUT/test_tilerows_math.log" || fail TR_MATH_TEST_NO_TOKEN
PRE_FAILURES=$FAILURES                                 # seed 1 and the probe are independent of each other

# ---- 0b. seed 1: ref / lean / lean_u1 / lean_split, float64 (G1-G3 + timing), as run_round1_polaris.pbs -------------
if [ "$PRE_FAILURES" = "0" ]; then
    echo "=== seed 1: $(date '+%F %T')"
    set +e
    timeout -k 60 1320 "$PY" SQG/analysis/opt/seed1/run_seed1.py --out "$OUT/seed1" --budget-min 14 \
        > "$OUT/seed1.log" 2>&1
    rc=$?
    set -e
    [ "$rc" = "0" ] || fail "IR_SEED1_RC rc=$rc (124 = timeout)"
    grep -q "SEED1_DONE" "$OUT/seed1.log" || fail IR_SEED1_NO_TOKEN
    grep -E '^(SEED1_(GATE_OK|BENCH|OPTSTEP|LONG|DONE)|ERROR |WARN )' "$OUT/seed1.log" || true
fi

# ---- 1. the probe, production cuFFT ---------------------------------------------------------------------------------
if [ "$PRE_FAILURES" = "0" ]; then
    echo "=== pass prod (production cuFFT): $(date '+%F %T')"
    set +e
    "$PY" "$HERE/inrow_probe.py" --out "$OUT" --tag prod --ckpt "1:${CKPT}" \
        --warp-path "$WARP_PATH" --nvmath-path "$NVMATH_PATH" --deadline $(( $(date +%s) + 30 * 60 )) \
        > "$OUT/inrow_probe_prod.log" 2>&1
    rc=$?
    set -e
    [ "$rc" = "0" ] || fail "IR_PROBE_RC tag=prod rc=$rc"
    grep -q "IR_PROBE_DONE failures=0" "$OUT/inrow_probe_prod.log" || fail "IR_PROBE_NOT_CLEAN tag=prod"
    grep -E "^(IR_|ERROR)" "$OUT/inrow_probe_prod.log" || true
    "$PY" "$HERE/score_inrow.py" "$OUT/inrow_probe_prod.json" 2>&1 | tee "$OUT/score_inrow.log" || fail IR_SCORE
else
    echo "ERROR IR_PROBE_SKIPPED (env or math tests failed)"
fi

[ -e "$OUT/seed1/seed1.json" ] && cp "$OUT/seed1/seed1.json" "$OUT"/
for f in "$OUT"/inrow_probe_*.json "$OUT"/seed1.json "$OUT"/midway_env_extra.json "$OUT"/*.log "$OUT"/*.txt \
         "$OUT"/gpu.csv "$OUT"/MANIFEST.sha256 "$OUT"/GIT_HEAD; do
    if [ -e "$f" ]; then cp "$f" "$RES"/ || fail "IR_COPY $f"; fi
done
# ---- 2. the Midway3 scorer (Level 1; Level 2-3 MISSING until Jobs 3A-3C) --------------------------------------------
"$PY" "${REPO}/SQG/midway3/score_midway3_timing.py" --inrow "$RES" --out "$RES/score_midway3_timing.json" \
    2>&1 | tee "$RES/score_midway3_timing.log" || fail IR_MIDWAY3_SCORE
echo "=== end $(date '+%F %T') ($(( $(date +%s) - T0 )) s)"
DONE_PRINTED=1
echo "IR_JOB_DONE failures=$FAILURES"
