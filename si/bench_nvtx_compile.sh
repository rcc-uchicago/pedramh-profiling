#!/bin/bash
#SBATCH --account=pi-pedramh
#SBATCH --time=00:45:00
#SBATCH -p pedramh-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --mem=500G
#SBATCH -o snfo_nvtx_compile_%x_%j.out
#SBATCH -e snfo_nvtx_compile_%x_%j.err

# NVTX/nsys profiling run for the SNFO bench with torch.compile enabled.
#
# Identical to bench_nvtx.sh but with:
#   - SNFO_TORCH_COMPILE=1   compile the inner DiT
#   - SNFO_BENCH_WARMUP=40   absorb the per-rank compile cost (~30–60 s)
#   - output prefix snfo_nvtx_compile_*  so traces don't collide with the
#                                         eager baseline (snfo_nvtx_*)
#
# After the job:
#   nsys export --type=sqlite snfo_nvtx_compile_<jobid>_rank<N>.nsys-rep
#   python3 parse_nsys.py snfo_nvtx_compile_<jobid>_rank0.sqlite
# and diff against the eager baseline at snfo_nvtx_<jobid>_rank0.sqlite.

module load python/miniforge-25.3.0

eval "$(conda shell.bash hook)"
conda activate /project/pedramh/shared/anthonyz/venv

module unload cuda
module load cuda/12.6

unset NCCL_DEBUG
unset TORCH_DISTRIBUTED_DEBUG

export WANDB_MODE=offline
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- Bench knobs for profiling with torch.compile ---
export SNFO_NVTX=1
export SNFO_BENCH_WARMUP=40    # raised so compile cost lands inside warmup
export SNFO_BENCH_STEPS=20

export SNFO_PRECISION=bf16-mixed

# Compile the inner DiT.  "default" mode is the safest first pass — it skips
# CUDA graphs (which interact badly with DDP + stochastic interpolant) but
# still fuses kernels and removes Python overhead from the model forward.
# Switch to "reduce-overhead" later if "default" is stable.
export SNFO_TORCH_COMPILE=1
export SNFO_COMPILE_MODE=default

# Keep DDP optimizations from the previous successful run.
export SNFO_DDP_BUCKET_CAP_MB=200
export SNFO_DDP_BF16_COMPRESS=1
export SNFO_DDP_BUCKET_VIEW=1

# Distinct CSV so the compile rows live next to (not mixed with) the eager
# results.  Sanity-check row only — full bench numbers should come from
# bench_midway.sh, not the profiler.
export SNFO_BENCH_CSV="${SLURM_SUBMIT_DIR}/bench_nvtx_compile_results.csv"

config_file=configs/bench_midway.yaml

echo "=== snfo_nvtx_compile: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODELIST=${SLURM_NODELIST}"
nvidia-smi -L
which nsys
nsys --version

echo "config=${config_file}  csv=${SNFO_BENCH_CSV}"
echo "warmup=${SNFO_BENCH_WARMUP}  steps=${SNFO_BENCH_STEPS}"
echo "compile=${SNFO_TORCH_COMPILE}  mode=${SNFO_COMPILE_MODE}"

cd "${SLURM_SUBMIT_DIR}"

srun --export=ALL nsys profile \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop-shutdown \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --output="snfo_nvtx_compile_${SLURM_JOB_ID}_rank%q{SLURM_PROCID}" \
    python bench.py --config "${config_file}"
