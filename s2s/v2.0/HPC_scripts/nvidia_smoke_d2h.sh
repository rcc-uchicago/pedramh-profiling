#!/bin/bash -l
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1              # smoke test only needs one GPU
#SBATCH --cpus-per-task=4
#SBATCH -o smoke_d2h_nvidia_%j.out
#SBATCH -e smoke_d2h_nvidia_%j.err

# Smoke test: isolates .item() vs bulk numpy and sync vs async D2H transfer
# patterns from inference.py vs inference_optimized.py.
# Does NOT require a model checkpoint — pure PyTorch tensor operations only.

ulimit -l unlimited

module load apptainer

export NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "=== nvidia_smoke_d2h: $(date -Iseconds) ==="
echo "Node: $(hostname)   GPUs available: ${NUM_GPUS}   Job: ${SLURM_JOB_ID}"
nvidia-smi -L

SIF=/project/pedramh/shared/S2S/v2.0/containers/pytorch_25.10.sif
S2S=/project/pedramh/shared/S2S/v2.0

apptainer exec \
    --nv \
    --bind /lustre/fs01 \
    --bind /project \
    "${SIF}" \
    bash -c "
        PYTHONPATH=${S2S} \
        python ${S2S}/test/d2h_pattern_smoke.py
    "
