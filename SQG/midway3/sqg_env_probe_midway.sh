#!/bin/bash
# sqg_env_probe_midway.sh -- Job 1 of the Midway3 port (MIDWAY3_INROW_PROMPT.md §4): the SQG env on one 4 x H100 NVL
# node. Beside SQG/polaris/sqg_env_probe.pbs; same probe, Midway scheduler header and env block.
#   bash SQG/midway3/setup_sqg_midway.sh                 (login node, once: pylibs + training root)
#   sbatch SQG/midway3/sqg_env_probe_midway.sh           (from the repo root)
# 1. SQG/polaris/sqg_env_probe.py (cluster-agnostic): SQG_ENV_OK + SQG_IMPORT_OK.
# 2. SQG/midway3/midway_env_extra.py: 4 x H100 in torch and jax, Warp 1.17.0 with cuFFTDx ("mathdx": 1) and a
#    4096-point wp.tile_fft kernel, nvmath's cuFFT version and the libraries mapped: SQG_MIDWAY_ENV_OK.
# PASS = "SQG_ENV_OK" + "SQG_IMPORT_OK" + "SQG_MIDWAY_ENV_OK" in the .out. Output: SQG/analysis/results/midway3_probe_<jobid>/
#SBATCH --job-name=sqg-env-probe
#SBATCH --account=pi-pedramh
#SBATCH --partition=pedramh-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=SQG/analysis/logs/midway3_env_probe_%j.out
#SBATCH --error=SQG/analysis/logs/midway3_env_probe_%j.err
set -eo pipefail
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
export SQG_INIT="${SQG_INIT:-random}"
OUT="${SQG_ROOT}/sqg_profile/probe_${JOB}"
RES="${REPO}/SQG/analysis/results/midway3_probe_${JOB}"
mkdir -p "$OUT" "$RES" "$TMPDIR" "$SQG_TRAINING_ROOT"
export WARP_CACHE_PATH="${SQG_ROOT}/tmp/warp_cache_${JOB}"
(cd SQG/fixed_ring && sha256sum -c --quiet SHA256SUMS) || { echo "ERROR SQG_SOLVER_SHA_MISMATCH"; exit 4; }
echo "=== sqg_env_probe_midway on $(hostname) job $JOB: $(date '+%F %T') ==="
echo "host=$(hostname) job=$JOB py=$PY root=$SQG_ROOT training_root=$SQG_TRAINING_ROOT init=$SQG_INIT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
echo "modules: $(module -t list 2>&1 | tr '\n' ' ')"
echo "co-tenants on this node: $(squeue -h -w "$(hostname -s)" -o '%i:%u:%b' | tr '\n' ' ')"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv \
    | tee "$OUT/gpu.csv"
nvidia-smi topo -m | tee "$OUT/topo.txt"
nvidia-smi | tee "$OUT/nvidia_smi.txt"

FAIL=0
"$PY" SQG/polaris/sqg_env_probe.py --out "$OUT" 2>&1 | tee "$OUT/sqg_env_probe.log" || FAIL=1
"$PY" SQG/midway3/midway_env_extra.py --out "$OUT" --warp-path "$WARP_PATH" --nvmath-path "$NVMATH_PATH" 2>&1 \
    | tee "$OUT/midway_env_extra.log" || FAIL=1
for f in sqg_env_probe.json midway_env_extra.json gpu.csv topo.txt nvidia_smi.txt sqg_env_probe.log midway_env_extra.log; do
    [ -e "$OUT/$f" ] && cp "$OUT/$f" "$RES"/
done
for t in SQG_ENV_OK SQG_IMPORT_OK SQG_MIDWAY_ENV_OK; do
    cat "$OUT"/*.log | grep -q "^$t" || { echo "ERROR SQG_MIDWAY_PROBE_MISSING_TOKEN $t"; FAIL=1; }
done
echo "=== end $(date '+%F %T') ==="
[ "$FAIL" = "0" ] && echo "SQG_MIDWAY_PROBE_DONE failures=0" || { echo "SQG_MIDWAY_PROBE_DONE failures=1"; exit 1; }
