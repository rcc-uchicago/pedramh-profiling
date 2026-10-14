#!/bin/bash
#SBATCH --account=pi-pedramh
#SBATCH --time=01:00:00
#SBATCH -p pedramh-gpu
#SBATCH --qos=pedramh-gpu
#SBATCH --constraint=H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o pangu_bench_%x_%j.out
#SBATCH -e pangu_bench_%x_%j.err
#
# PanguWeather SFNO CSV bench on Midway (pedramh-gpu, 4x H100 NVL) -- NO nsys.
# Sibling of midway_bench_nsys_e3sm_sfno.sh, for runs where the step time is
# the deliverable and a 44 MB report is not: env-var sweeps, scaling checks.
# Sibling of polaris_bench_e3sm_sfno.pbs -- per rule #7 that script is left
# untouched; this is the Midway version beside it.
#
# PASS = `PANGU_BENCH_OK` in the .out (a new bench CSV row). Key on the token.
#
# WARMUP defaults to 40 here, not 20. On this node the DDP all-reduce queue
# takes ~30 steps to saturate, so a 20-step warmup leaves fast fill steps in
# the median and under-reports step time (measured: 1.100 s reported vs ~1.17 s
# steady state). See PanguWeather/v2.0/bench_midway_notes.md.
#
# UNLIKE ACE2, this model is already instrumented: train.py calls
# cudaProfilerStart/Stop and emits the shared NVTX ranges when PANGU_NVTX=1, so
# --capture-range=cudaProfilerApi works and the window is exactly the measured
# steps. No hand-derived --delay/--duration is needed here.
#
# Knobs are PANGU_* by project convention. A stale S2S_* name errors out with
# LEGACY_BENCH_ENV -- do not carry S2S knobs across.
#
# HARDWARE CAVEAT for comparing against polaris_bench_report.md: midway3-0423 is
# H100 NVL with NVLink between GPU PAIRS only (GPU0<->1, GPU2<->3; cross-pair is
# PCIe + a NUMA hop -- measured 261 vs 18 GB/s by gpu_topology_check.py). A
# 4-GPU all-reduce crosses that boundary twice; Polaris A100 nodes do not. For
# ACE2 that cost +29% per step. Expect Pangu's NCCL share here to be HIGHER than
# the 10.5% measured on Polaris, for reasons that have nothing to do with the
# model.

set -eo pipefail

V20_DIR=/project/rcc/mehta5/pedramh-profiling/PanguWeather/v2.0
CONFIG_FILE="${PANGU_CONFIG:-${V20_DIR}/config/E3SM_SFNO_H5_MIDWAY.yaml}"
# The shared SFNO env, not the S2S venv: Pangu's sfnonet imports tensorly
# (factorizations), which the S2S venv lacks -- job 53539649 died on
# ModuleNotFoundError: tensorly. This env already carries torch_harmonics,
# tensorly, tltorch, h5py and netCDF4, so nothing needs installing into a
# venv that S2S and the Lightning port also depend on.
VENV=/project/pedramh/shared/conda/envs/py311_pip_sfno_cu129
RUN_NUM="${PANGU_RUN_NUM:-bench$(date +%s)}"

# --- env: Midway venv (the S2S/Pangu shared venv), NOT the Polaris conda ------
module load python/miniforge-25.3.0
eval "$(conda shell.bash hook)"
conda activate "${VENV}"

# Deliberately NO `module load cuda`: this env's torch is cu129 and a 12.6/12.8
# module would put mismatched CUDA libs ahead of torch's bundled ones.

export PYTHONPATH="${V20_DIR}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

unset NCCL_DEBUG
unset TORCH_DISTRIBUTED_DEBUG
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=0
ulimit -l unlimited

python -c "import torch, torch_harmonics, tensorly, h5py, netCDF4" 2>/dev/null || {
    echo "ERROR PANGU_ENV_NOT_ACTIVE python=$(command -v python) venv=${VENV}"; exit 1; }

# --- bench knobs (must match the CSV bench so trace and CSV describe one shape)
export PANGU_BENCH=1
export PANGU_BENCH_WARMUP="${PANGU_BENCH_WARMUP:-40}"
export PANGU_BENCH_STEPS="${PANGU_BENCH_STEPS:-40}"
export PANGU_NVTX=1
export PANGU_BENCH_CSV="${PANGU_BENCH_CSV:-${V20_DIR}/results/bench/pangu_sfno_midway.csv}"
mkdir -p "$(dirname "${PANGU_BENCH_CSV}")"

NPROC=$(nvidia-smi -L | wc -l)

echo "=== midway_bench_e3sm_sfno.sh: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODE=${SLURM_NODELIST}  NPROC=${NPROC}  run_num=${RUN_NUM}"
echo "config=${CONFIG_FILE}"
echo "knobs: warmup=${PANGU_BENCH_WARMUP} steps=${PANGU_BENCH_STEPS} nvtx=${PANGU_NVTX}"
nvidia-smi -L

cd "${V20_DIR}"

# rc is captured, not fatal: the gate is a NEW CSV row, never the exit code
# (CLAUDE.md #14).
rows_before=$( [ -f "${PANGU_BENCH_CSV}" ] && wc -l < "${PANGU_BENCH_CSV}" || echo 0 )
rc=0
torchrun \
    --standalone \
    --nproc_per_node="${NPROC}" \
    train.py \
    --yaml_config="${CONFIG_FILE}" \
    --run_num="${RUN_NUM}" \
    --config=SFNO \
    --fresh_start \
    --epochs 1 || rc=$?
echo "--- train rc=${rc} ---"

rows_after=$( [ -f "${PANGU_BENCH_CSV}" ] && wc -l < "${PANGU_BENCH_CSV}" || echo 0 )
if [ "${rows_after}" -le "${rows_before}" ]; then
    echo "ERROR PANGU_NO_BENCH_ROW ${PANGU_BENCH_CSV} gained no row (rc=${rc})"
    exit 1
fi
echo "bench CSV row: $(tail -1 "${PANGU_BENCH_CSV}")"
echo "PANGU_BENCH_OK n_gpus=${NPROC} run_num=${RUN_NUM} nccl_proto=${NCCL_PROTO:-<default>} nccl_algo=${NCCL_ALGO:-<default>}"
