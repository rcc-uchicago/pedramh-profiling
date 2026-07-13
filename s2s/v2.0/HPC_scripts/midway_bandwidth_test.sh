#!/bin/bash -l
#SBATCH --account=pi-pedramh
#SBATCH --time=00:20:00
#SBATCH -p pedramh-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o midway_bandwidth_%N.out
#SBATCH -e midway_bandwidth_%N.err

# PCIe H2D bandwidth test — measures single-GPU and concurrent 4-GPU transfer
# rates at the actual S2S inference tensor sizes.
#
# Run the same script on the DSI node (bare metal, same Python env) to get a
# directly comparable table. The contention delta section shows whether
# bandwidth drops under 4-GPU load, and by how much per GPU.
#
# Also prints nvidia-smi topo -m so the node topology is captured in the log.

ulimit -l unlimited

module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv

module unload cuda
module load cuda/12.6

echo "=== midway_bandwidth_test: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODELIST=${SLURM_NODELIST}"
nvidia-smi -L

echo ""
echo "=== GPU topology ==="
nvidia-smi topo -m

echo ""
echo "=== NUMA hardware ==="
numactl --hardware

echo ""
echo "=== Bandwidth test ==="
PYTHONPATH=/project/pedramh/shared/S2S/v2.0 \
python /project/pedramh/shared/S2S/v2.0/test/bandwidth_test.py
