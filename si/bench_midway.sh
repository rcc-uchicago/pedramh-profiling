#!/bin/bash
#SBATCH --account=pi-pedramh
#SBATCH --time=00:30:00
#SBATCH -p pedramh-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --mem=500G
#SBATCH -o si_bench_%x_%j.out
#SBATCH -e si_bench_%x_%j.err

# Wall-clock throughput benchmark for SI training.
# Measures 80 steps after a 20-step warm-up; writes one CSV row.
# No nsys, no NCCL_DEBUG, no TORCH_DISTRIBUTED_DEBUG — those distort timing.
#
# Important: bench.py sets accumulate_grad_batches=1 regardless of the
# YAML value.  bench_midway.yaml already sets this to 1; log it in any
# report that compares throughput numbers between bench and production.
#
# To profile with Nsight Systems:
#   1. Set SI_NVTX=1 below and raise SI_BENCH_WARMUP to 40.
#   2. Wrap the srun call with:
#        nsys profile \
#          --trace=cuda,nvtx,osrt \
#          --capture-range=cudaProfilerApi \
#          --output=si_bench_%j \
#          srun python bench.py ...
#   3. After the job: nsys export --type=sqlite si_bench_<job>.nsys-rep

module load python/miniforge-25.3.0

eval "$(conda shell.bash hook)"
conda activate /project/pedramh/shared/anthonyz/venv

module unload cuda
module load cuda/12.6

# Production training keeps these set; the bench must run without their overhead.
unset NCCL_DEBUG
unset TORCH_DISTRIBUTED_DEBUG

export WANDB_MODE=offline
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OMP threads: cpus-per-task / (num_data_workers + 1 training thread)
# With cpus-per-task=8 and num_data_workers=8 we'd starve OMP; keep it at 1.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- Bench knobs (edit here to run ablations) ---
export SI_BENCH_WARMUP=20
export SI_BENCH_STEPS=80

# Precision: bf16-mixed by default.  The H100 single-GPU smoke test
# (bench_test_h100_results.csv) peaked at 38.5 GB with bs=1 fp32, so the full
# bench at bs=4 per rank would risk OOM on 80 GB H100s.  bf16-mixed roughly
# halves activation memory and is the production-relevant precision anyway.
# bench.py reads SI_PRECISION and applies it on top of the YAML.
export SI_PRECISION=bf16-mixed
# export SI_PRECISION=32-true   # uncomment for an fp32 baseline (lower bs=2 first)
# export SI_PRECISION=16-mixed

# NVTX: set to 1 together with the nsys wrapper above.
# export SI_NVTX=1
# export SI_BENCH_WARMUP=40   # raise warmup when using torch.compile

# CSV output: absolute path so it survives cwd changes.
export SI_BENCH_CSV="${SLURM_SUBMIT_DIR}/bench_pedramh_node_results.csv"

config_file=configs/bench_midway.yaml

echo "=== si_bench: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODELIST=${SLURM_NODELIST}"
nvidia-smi -L

echo "config=${config_file}  csv=${SI_BENCH_CSV}"
echo "warmup=${SI_BENCH_WARMUP}  steps=${SI_BENCH_STEPS}"

cd "${SLURM_SUBMIT_DIR}"

srun --export=ALL python bench.py --config "${config_file}"
