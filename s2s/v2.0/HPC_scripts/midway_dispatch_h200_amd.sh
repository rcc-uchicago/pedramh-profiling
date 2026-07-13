#!/bin/bash -l
#SBATCH --account=rcc-staff
#SBATCH --time=00:20:00
#SBATCH -p test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=midway3-0601   # AMD EPYC-9335, 768GB, H200 DLC
                                   # alternative: midway3-0600
#SBATCH -o dispatch_h200_amd_%N.out
#SBATCH -e dispatch_h200_amd_%N.err

ulimit -l unlimited

module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv

module unload cuda
module load cuda/12.6

echo "=== inference_dispatch_smoke H200 AMD EPYC: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODE=${SLURM_NODELIST}"
nvidia-smi -L

PYTHONPATH=/project/pedramh/shared/S2S/v2.0 \
python /project/pedramh/shared/S2S/v2.0/test/inference_dispatch_smoke.py
