#!/bin/bash
#SBATCH --account=pi-pedramh
#SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH -o si_bench_a100_%x_%j.out
#SBATCH -e si_bench_a100_%x_%j.err

# Single-GPU smoke test on the shared A100 (midway3-0294, CC 8.0).
# Uses a short warmup + step count so the job finishes in a few minutes.

module load python/miniforge-25.3.0
eval "$(conda shell.bash hook)"
conda activate /project/pedramh/shared/anthonyz/venv

module unload cuda
module load cuda/12.6

unset NCCL_DEBUG
unset TORCH_DISTRIBUTED_DEBUG

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

export SI_BENCH_WARMUP=5
export SI_BENCH_STEPS=20
export SI_BENCH_CSV="${SLURM_SUBMIT_DIR}/bench_test_a100_results.csv"

# A100 has 40 GB; at fp32 the model peaks ~39 GB with batch_size=1 and OOMs.
# bf16-mixed roughly halves activation memory and is supported natively on A100 (CC 8.0).
export SI_PRECISION=bf16-mixed

config_file=configs/bench_midway.yaml

echo "=== si_bench_test_a100: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODELIST=${SLURM_NODELIST}"
nvidia-smi -L

echo "config=${config_file}  csv=${SI_BENCH_CSV}"

cd "${SLURM_SUBMIT_DIR}"

srun --export=ALL python bench.py \
    --config "${config_file}" \
    --devices 0 \
    --batch_size 1
