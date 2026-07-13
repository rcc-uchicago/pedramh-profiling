#!/bin/bash -l
#SBATCH --account=pi-pedramh
#SBATCH --time=00:15:00
#SBATCH -p pedramh-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1              # smoke test only needs one GPU
#SBATCH --cpus-per-task=4
#SBATCH -o smoke_d2h_midway_%j.out
#SBATCH -e smoke_d2h_midway_%j.err

# Smoke test: isolates .item() vs bulk numpy and sync vs async D2H transfer
# patterns from inference.py vs inference_optimized.py.
# Does NOT require a model checkpoint — pure PyTorch tensor operations only.
#
# For DSI cluster: adjust the partition name and paths above.

ulimit -l unlimited

module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv

module unload cuda
module load cuda/12.6

echo "=== midway_smoke_d2h: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODELIST=${SLURM_NODELIST}"
nvidia-smi -L

PYTHONPATH=/project/pedramh/shared/S2S/v2.0 \
python /project/pedramh/shared/S2S/v2.0/test/d2h_pattern_smoke.py
