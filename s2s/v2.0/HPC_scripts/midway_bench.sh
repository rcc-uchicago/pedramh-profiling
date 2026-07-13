#!/bin/bash -l
#SBATCH --account=pi-pedramh
#SBATCH --time=00:30:00
#SBATCH -p pedramh-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o midway_bench_%x_%j.out
#SBATCH -e midway_bench_%x_%j.err

# Wall-clock benchmark of the current 4-GPU DDP training shape.
# Patched via S2S_BENCH=1 in train.py: instrumented step loop with cuda.synchronize()
# bracketing, scaler-skip detection, and rank-0 CSV row + env side-car.
# No nsys, no NCCL_DEBUG, no TORCH_DISTRIBUTED_DEBUG — those distort timing.

ulimit -l unlimited

# Mamba activation (Midway): mamba shell hook must be eval'd in the batch shell
# before `mamba activate` works.
# `module purge` first: a Python module inherited from the submit environment can
# conflict with miniforge ("cannot be loaded due to a conflict"), silently leaving
# the system anaconda active and crashing every rank with ModuleNotFoundError.
module purge 2>/dev/null || true
module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv

module load cuda/12.6

# Fail fast if the venv did not actually activate.
python -c "import torch, wandb" 2>/dev/null || {
    echo "FATAL: venv not active (python=$(command -v python)); aborting before launch."
    exit 1
}

# Production training keeps these set; the bench must run without their overhead.
unset NCCL_DEBUG
unset TORCH_DISTRIBUTED_DEBUG

export WANDB_MODE=offline
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Avoid OMP oversubscription: 32 cpus / (4 ranks * 8 dataloader workers) = 1 each;
# leave a small intra-op pool for the loader transforms.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Bench knobs (defaults match the plan — change here if running variants).
export S2S_BENCH=1
export S2S_BENCH_WARMUP=20
export S2S_BENCH_STEPS=80

# --- Optimization variants (uncomment to measure a specific change) ---
# AMP dtype: bf16 removes GradScaler entirely and is native on H100.
# Verify no NaN by checking scaler_skips=0 in fp16 first (baseline does this).
export S2S_AMP_DTYPE=bf16

# torch.compile DISABLED: reduce-overhead (CUDA graphs) segfaults at DDP teardown on
# this model ("CUDA Graph is empty") and drops the CUPTI kernel table — see
# midway_bench_nsys.sh for the full note. The committed bench_results.csv baselines
# were collected without effective compilation, so they are unaffected. Re-enable
# (and restore the warmup-40 for Triton settling) only once it is graph-break-clean.
# export TORCH_COMPILE_MODE=reduce-overhead
# export S2S_BENCH_WARMUP=40   # use 40 (not 20) when re-enabling compile

# CSV / env side-car location. Resolves to absolute path so it doesn't depend on cwd.
export S2S_BENCH_CSV="${SLURM_SUBMIT_DIR}/bench_results.csv"

config_file=/project/pedramh/shared/S2S/v2.0/config/exp2.yaml
run_num="bench_$(date +%s)"

# Surface the yaml path to the trainer so its sha can be recorded in the env file.
export S2S_YAML="$(realpath ${config_file})"
export S2S_RUN_NUM="${run_num}"

echo "=== midway_bench: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODELIST=${SLURM_NODELIST}"
nvidia-smi -L

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "NUM_GPUS=${NUM_GPUS}  run_num=${run_num}  csv=${S2S_BENCH_CSV}"

torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    /project/pedramh/shared/S2S/v2.0/train.py \
    --yaml_config="${config_file}" \
    --run_num="${run_num}"
