#!/bin/bash
#SBATCH --account=pi-pedramh
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -o si_bench_validate_%x_%j.out
#SBATCH -e si_bench_validate_%x_%j.err

# CPU-only validation step in the bench dependency chain.
# Checks that bench_gpu_test.sh wrote a sane CSV row before
# allowing bench_midway.sh to consume a full GPU node.

module load python/miniforge-25.3.0

eval "$(conda shell.bash hook)"
conda activate /project/pedramh/shared/anthonyz/venv

export SI_BENCH_CSV="${SLURM_SUBMIT_DIR}/bench_test_results.csv"
export SI_BENCH_STEPS=20

cd "${SLURM_SUBMIT_DIR}"

echo "=== validate_bench: $(date -Iseconds) ==="
python validate_bench.py
