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
#SBATCH --nodelist=midway3-0601   # AMD EPYC-9335, 768GB, H200 DLC
                                   # alternative: midway3-0600
#SBATCH -o bw_h200_amd_%N.out
#SBATCH -e bw_h200_amd_%N.err

# AMD EPYC-9335 (Genoa) uses a chiplet design: multiple CCDs each appearing
# as a separate NUMA node. A 32-core EPYC-9335 typically has 2 CCDs = 2+
# NUMA nodes, compared to Intel's 1-2 per socket. This makes numactl --hardware
# output especially important — GPU-to-NUMA affinity on EPYC may differ
# significantly from the Intel Gold nodes.

ulimit -l unlimited

module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv

module unload cuda
module load cuda/12.6

echo "=== bandwidth_test H200 AMD EPYC: $(date -Iseconds) ==="
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
