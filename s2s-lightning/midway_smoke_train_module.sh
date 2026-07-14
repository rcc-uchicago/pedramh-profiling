#!/bin/bash -l
#SBATCH --account=pi-pedramh
#SBATCH --time=00:30:00
#SBATCH -p pedramh-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o smoke_train_module_%j.out
#SBATCH -e smoke_train_module_%j.err

# Phase-2 smoke: 2-step Lightning fit of TrainModule on the real HDF5 data.
# Env bootstrap mirrors v2.0/HPC_scripts/midway_smoke_d2h.sh EXACTLY.
# NOTE: do NOT `module purge` on Midway3 (strips SOFTPATH -> mamba/cuda break).

ulimit -l unlimited

module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv

module unload cuda
module load cuda/12.6

export WANDB_MODE=offline
# Reduce allocator fragmentation on the single-GPU smoke (large activations).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# s2s/v2.0 resolves `from utils...` / `from networks...`; the port dir resolves
# the ported `from data...` / `from modules...` packages (SI-style).
PORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../s2s-lightning
REPO_ROOT="$(cd "${PORT_DIR}/.." && pwd)"                  # .../pedramh-profiling
export PYTHONPATH="${REPO_ROOT}/s2s/v2.0:${PORT_DIR}"

echo "=== smoke_train_module: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODELIST=${SLURM_NODELIST}"
nvidia-smi -L

cd "${PORT_DIR}"
python "${PORT_DIR}/smoke_train_module.py"
