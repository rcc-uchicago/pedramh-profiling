#!/bin/bash -l
#SBATCH --account=rcc-staff
#SBATCH --time=00:20:00
#SBATCH -p test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodelist=midway3-0602   # Intel Gold-6542Y, 1TB, H200 DLC
                                   # alternatives: midway3-0603/0604/0605/0606
#SBATCH -o bw_h200_intel_%N.out
#SBATCH -e bw_h200_intel_%N.err

ulimit -l unlimited

module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv

module unload cuda
module load cuda/12.6

echo "=== bandwidth_test H200 Intel: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODE=${SLURM_NODELIST}"
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
