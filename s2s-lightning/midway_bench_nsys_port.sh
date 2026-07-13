#!/bin/bash -l
#SBATCH --account=pi-pedramh
#SBATCH --time=00:45:00
#SBATCH -p pedramh-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o midway_bench_nsys_port_%x_%j.out
#SBATCH -e midway_bench_nsys_port_%x_%j.err

# ============================================================================
# Nsight Systems profile of the PORT (Lightning bench.py), matched 1:1 to the
# v2.0 baseline in v2.0/HPC_scripts/midway_bench_nsys.sh so the two .nsys-rep
# files are directly comparable.
#
# WHAT IS HELD IDENTICAL TO THE v2.0 BASELINE:
#   * config       : v2.0/config/exp2.yaml  §S2S         (same model/data/loss/ensemble)
#   * per-GPU batch : 2  (v2.0 ran exp2 batch_size=8 // world_size 4 = 2/GPU;
#                         bench.py treats batch_size as PER-GPU, so --batch_size 2)
#   * precision     : bf16  (S2S_PRECISION=bf16-mixed == v2.0's S2S_AMP_DTYPE=bf16)
#   * warmup/steps  : 20 / 80   (S2S_BENCH_WARMUP / S2S_BENCH_STEPS)
#   * torch.compile : OFF (eager) — same basis as the bench_report.md numbers
#   * nsys flags    : identical (--capture-range=cudaProfilerApi, same --trace set)
#   * GPUs          : 4, exclusive node
# The only intended difference vs the baseline is the model wrapper (Lightning).
#
# HOW TO COMPARE (see bench_methodology.md for the sqlite queries):
#   nsys stats nsys_port_bench_eager_<run>.nsys-rep      # kernel + NVTX summary
#   * NVTX: forward_loss / backward / optimizer durations should match the v2.0
#     trace within noise. CAVEAT (port analysis P2): the port's per-step NVTX
#     window opens at Lightning's on_train_batch_start, which fires AFTER the H2D
#     copy, so `step_med` excludes the transfer the v2.0 step included — compare
#     THROUGHPUT via `samples_per_s_wall` in the CSV, NOT step_med.
#   * Kernels: the top-kernel-by-time table (layer_norm / elementwise / GEMM /
#     ncclDevKernel / memcpy) should match the v2.0 breakdown — same model, same
#     ops. A divergence there is a real port regression.
#   * CSV: port_bench_results.csv (same schema as v2.0 bench_results.csv) —
#     compare samples_per_s_wall and peak_mem_gb_max_rank.
#
# Output: ${SLURM_SUBMIT_DIR}/nsys_port_bench_eager_<run>_rank0.nsys-rep (+ _rank1..3)
# Launch: srun with --ntasks-per-node=4 (Lightning's SLURM launcher needs ntasks == devices).
# ============================================================================

ulimit -l unlimited

module unload python
module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv     # LPORT_ENV: torch 2.6 + lightning 2.5

module load cuda/12.6

# Fail fast if the venv did not activate (else Lightning's DDP children die on import
# and nsys writes a junk profile). Port needs lightning in addition to torch/wandb.
python -c "import torch, lightning, wandb" 2>/dev/null || {
    echo "FATAL: venv not active or missing lightning (python=$(command -v python)); aborting."
    exit 1
}

unset NCCL_DEBUG
unset TORCH_DISTRIBUTED_DEBUG

export WANDB_MODE=offline
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Bench knobs — identical to midway_bench_nsys.sh.
export S2S_BENCH=1
export S2S_BENCH_WARMUP=20
export S2S_BENCH_STEPS=80
export S2S_NVTX=1                 # NVTX ranges + cudaProfilerStart/Stop window
export S2S_AMP_DTYPE=bf16         # CSV label
export S2S_PRECISION=bf16-mixed   # port's Trainer precision == v2.0 bf16 autocast
# torch.compile OFF (eager) — same basis as the v2.0 baseline. Do NOT enable here.

# Resolve repo layout from this script's location (pedramh-profiling/s2s-lightning/).
PORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../s2s-lightning
REPO_ROOT="$(cd "${PORT_DIR}/.." && pwd)"                  # .../pedramh-profiling
S2S_DIR="${REPO_ROOT}/s2s"                                 # regular S2S (shared, imported by the port)
export PYTHONPATH="${S2S_DIR}/v2.0:${PORT_DIR}"    # s2s/v2.0 -> utils,networks ; port dir -> data,modules,common
export S2S_BENCH_CSV="${SLURM_SUBMIT_DIR}/port_bench_results.csv"

config_file="${S2S_DIR}/v2.0/config/exp2.yaml"
run_num="port_bench_$(date +%s)"

echo "=== midway_bench_nsys_port: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODELIST=${SLURM_NODELIST}"
nvidia-smi -L
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "NUM_GPUS=${NUM_GPUS}  config=${config_file}  per-GPU batch=2  run_num=${run_num}"

NSYS_OUT="${SLURM_SUBMIT_DIR}/nsys_port_bench_eager_${run_num}"
echo "nsys output: ${NSYS_OUT}.nsys-rep"

# Lightning under SLURM uses the SLURM launcher, which REQUIRES --ntasks-per-node
# to equal the device count (the earlier ntasks-per-node=1 caused Lightning's
# "devices=4 does not match --ntasks-per-node=1" abort). srun starts the 4 ranks;
# nsys writes one rank-tagged report per rank — use the _rank0 report to compare
# against the v2.0 rank-0 baseline.
# (NCCL-free single-GPU alternative: set ntasks-per-node=1 + gres=gpu:1 above and
#  drop `srun`, running: python "${PORT_DIR}/bench.py" --devices 0 --strategy auto.)
srun nsys profile \
    --trace=cuda,nvtx,cudnn,cublas,osrt \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --output="${NSYS_OUT}_rank%q{SLURM_PROCID}" \
    --force-overwrite=true \
    --nic-metrics=true \
    python "${PORT_DIR}/bench.py" \
        --yaml_config "${config_file}" \
        --config S2S \
        --batch_size 2 \
        --devices 0 1 2 3 \
        --strategy ddp

echo "Profiles written: ${NSYS_OUT}_rank0.nsys-rep (+ _rank1..3)"
echo "Compare to the v2.0 baseline: nsys stats ${NSYS_OUT}_rank0.nsys-rep  (and diff port_bench_results.csv vs v2.0/HPC_scripts/bench_results.csv on samples_per_s_wall)"
