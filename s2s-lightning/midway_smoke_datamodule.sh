#!/bin/bash -l
#SBATCH --account=pi-pedramh
#SBATCH --time=00:15:00
#SBATCH -p pedramh-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1              # Phase 1 DataModule smoke needs one GPU node
#SBATCH --cpus-per-task=4
#SBATCH -o smoke_datamodule_%j.out
#SBATCH -e smoke_datamodule_%j.err

# Phase 1 smoke (S2S -> Lightning port): instantiate ClimateDataModule against
# the real HDF5 dataset, run setup("fit"), and pull one training batch.
# Reuses the existing S2S loaders (utils.data_loader_multifiles.get_data_loader);
# no model, no checkpoint. The commit gate keys on SMOKE_OK in the .out.

ulimit -l unlimited

# NOTE: do NOT `module purge` here — on Midway3 it strips SOFTPATH and breaks the
# python/miniforge + cuda modulefiles (observed: "set varName" Tcl error, then
# mamba not found). Mirror the proven v2.0/HPC_scripts/midway_smoke_d2h.sh flow.
module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv

module unload cuda 2>/dev/null
module load cuda/12.6

# Fail fast if the env is wrong (wrong python / missing deps) before touching data.
python -c "import torch, wandb" || { echo "FATAL: torch/wandb import failed"; exit 1; }

export WANDB_MODE=offline
# Resolve repo layout from this script's location (pedramh-profiling/s2s-lightning/).
PORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../s2s-lightning
REPO_ROOT="$(cd "${PORT_DIR}/.." && pwd)"                  # .../pedramh-profiling
# s2s/v2.0 resolves `from utils...`/`from networks...`; the port dir resolves `from data...`.
export PYTHONPATH="${REPO_ROOT}/s2s/v2.0:${PORT_DIR}"

echo "=== midway_smoke_datamodule: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODELIST=${SLURM_NODELIST}"
nvidia-smi -L

cd "${PORT_DIR}" && python smoke_datamodule.py
