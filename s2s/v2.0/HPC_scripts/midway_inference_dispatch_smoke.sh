#!/bin/bash -l
#SBATCH --account=pi-pedramh
#SBATCH --time=00:20:00
#SBATCH -p pedramh-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -o inference_dispatch_smoke_%N.out
#SBATCH -e inference_dispatch_smoke_%N.err

# Measures the GPU-idle time between consecutive autoregressive forward passes.
# Compares four dispatch patterns: list append, pre-allocated tensor,
# torch.compile, and CUDA Graph.
#
# If the gaps are large (10-50 ms) and CUDA Graph closes them: Python dispatch
# is the bottleneck — fixable in software.
# If gaps persist even with CUDA Graph: hardware (NUMA/PCIe) is the cause.
#
# Run on DSI with the same command to get a directly comparable result:
#   PYTHONPATH=/path/to/S2S/v2.0 python v2.0/test/inference_dispatch_smoke.py

ulimit -l unlimited

module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv

module unload cuda
module load cuda/12.6

echo "=== inference_dispatch_smoke: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODELIST=${SLURM_NODELIST}"
nvidia-smi -L

PYTHONPATH=/project/pedramh/shared/S2S/v2.0 \
python /project/pedramh/shared/S2S/v2.0/test/inference_dispatch_smoke.py
