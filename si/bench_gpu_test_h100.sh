#!/bin/bash
#SBATCH --account=pi-pedramh
#SBATCH --time=00:15:00
#SBATCH -p pedramh-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH -o si_bench_h100_%x_%j.out
#SBATCH -e si_bench_h100_%x_%j.err

# Single-GPU smoke test on pedramh H100 (CC 9.0).
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
export SI_BENCH_CSV="${SLURM_SUBMIT_DIR}/bench_test_h100_results.csv"

config_file=configs/bench_midway.yaml

echo "=== si_bench_test_h100: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODELIST=${SLURM_NODELIST}"
nvidia-smi -L

echo "config=${config_file}  csv=${SI_BENCH_CSV}"

cd "${SLURM_SUBMIT_DIR}"

srun --export=ALL python bench.py \
    --config "${config_file}" \
    --devices 0 \
    --batch_size 1
