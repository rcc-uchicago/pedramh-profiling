#!/bin/bash -l
#SBATCH --account=rcc-staff
#SBATCH --time=01:00:00
#SBATCH -p test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodelist=midway3-0602   # Intel Gold-6542Y, 1TB, H200 DLC
                                   # alternatives: midway3-0603/0604/0605/0606
#SBATCH -o midway_infer_nsys_newinfer_%N_%j.out
#SBATCH -e midway_infer_nsys_newinfer_%N_%j.err

# File-swap diagnostic sibling of midway_infer_nsys_h200_intel.sh.
#
# Purpose:
#   Per CLAUDE.md, v2.0/inference.py is the actively-maintained variant
#   and v2.0/inference_optimized.py is the older fork. The production
#   nsys captures that drop CUPTI_ACTIVITY_KIND_KERNEL all target
#   inference_optimized.py. This script reruns the same nsys profile with
#   the ORIGINAL production flags (--trace-fork-before-exec=true, no
#   --sample=none) but points at inference.py instead -- so the only
#   variable between this run and the broken pre-flush production capture
#   is which inference script torchrun launches.
#
#   - If this captures KERNEL events -> the bug is specific to
#     inference_optimized.py and switching production to inference.py
#     sidesteps it entirely.
#   - If this also drops KERNEL events -> the bug is in shared code
#     (model construction, HDF5 loader, AMP path) and the noforktrace +
#     --sample=none direction in the other sibling script is correct
#     regardless of which inference file we use.
#
# --run_num="infer_nsys_h200_intel" matches the existing production
# Intel script so the same existing checkpoint at
#   results/S2S/infer_nsys_h200_intel/training_checkpoints/ckpt.tar
# is reused (no need for a stub).

ulimit -l unlimited

module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv

module unload cuda
module load cuda/12.9

echo "nsys version: $(nsys --version 2>&1 | head -1)"

export WANDB_MODE=offline

echo "=== midway_infer_nsys_newinfer H200 Intel: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODE=${SLURM_NODELIST}"
nvidia-smi -L
nvidia-smi topo -m

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "NUM_GPUS=${NUM_GPUS}"

config_file=/project/pedramh/shared/S2S/v2.0/config/exp2.yaml
NSYS_OUT="${SLURM_SUBMIT_DIR}/midway_h200_intel_4gpus_inference_newinfer_${SLURM_JOB_ID}"

echo "nsys output: ${NSYS_OUT}.nsys-rep"

# ORIGINAL production flags intentionally preserved here (with
# --trace-fork-before-exec=true and without --sample=none) so that the
# only difference from the broken pre-flush production capture is the
# target script (inference.py vs inference_optimized.py).
nsys profile \
    -w true \
    -t cuda,nvtx,cudnn \
    -o "${NSYS_OUT}" \
    --force-overwrite=true \
    --trace-fork-before-exec=true \
    torchrun \
        --standalone \
        --nproc_per_node="${NUM_GPUS}" \
        /project/pedramh/shared/S2S/v2.0/inference.py \
        --yaml_config="${config_file}" \
        --run_num="infer_nsys_h200_intel" \
        --async_save

echo "Profile written: ${NSYS_OUT}.nsys-rep"
echo "scp the .nsys-rep locally and run: nsys export --type=sqlite <file>.nsys-rep"
