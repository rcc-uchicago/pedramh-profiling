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
#SBATCH -o midway_infer_nsys_delay_%N_%j.out
#SBATCH -e midway_infer_nsys_delay_%N_%j.err

# Delay-based diagnostic sibling of midway_infer_nsys_h200_intel.sh.
#
# MOOT 2026-05-26 -- the cuDNN-autotune-burst hypothesis this script was
# written to test is RESOLVED (falsified, in the sense that it was never
# the right hypothesis to begin with). The broken-capture fingerprint
# (RUNTIME=4252 / MEMCPY=1960 / SYNC=1960, KERNEL missing) was actually
# inference[_optimized].py crashing at restore_checkpoint() on
# FileNotFoundError before launching any kernel; cuDNN autotune never ran
# because no forward pass ever started. Fixed in commit 56f73fe. See
# memory: project_midway_cupti_kernel_missing.md. The "Test" decision tree
# below should not be used as guidance for new investigations.
#
# Purpose (historical, pre-resolution):
#   The noforktrace + newinfer pair (job 50084384 / 50084385) produced
#   byte-identical CUPTI counts (RUNTIME=4252, MEMCPY=1960, SYNC=1960,
#   PROCESSES=692) and the same KERNEL/MEMSET/CUDA_EVENT/OVERHEAD-table-
#   missing fingerprint. So the breakage is deterministic, happens at
#   the same point in initialisation across both inference scripts, and
#   isn't affected by --trace-fork-before-exec or --sample=none.
#
#   Best remaining hypothesis (pre-resolution): cuDNN autotune during the
#   first forward pass. inference_optimized.py:403 and inference.py both
#   set torch.backends.cudnn.benchmark = True, which makes the first
#   forward burst dozens of trial kernel launches per conv layer
#   (including Hopper-specific wgmma / FA3 paths under bf16 autocast).
#   If CUPTI's activity-stream subscriber chokes during that burst, it
#   stays dead for the rest of the run -- exactly the symptom we see.
#
# Test (historical decision tree, DO NOT USE):
#   --delay=15 tells nsys to skip the first 15 s of capture. That covers
#   model construction + checkpoint load + NCCL init + the cuDNN
#   autotune burst on the first forward. Capture then begins clean for
#   the steady-state inference loop.
#
#   - KERNEL captured  -> autotune burst is the trigger. Production fix
#                         is either --delay (and accept losing init
#                         coverage) or set cudnn.benchmark=False in the
#                         inference scripts.
#   - KERNEL still missing -> not autotune; the trigger is something
#                             else that survives a 15 s delay.
#
# Note on "the only variable is --delay": the comment above used to claim
# that all other nsys flags matched the ORIGINAL production script so the
# only variable vs production was --delay. That invariant no longer holds
# against today's midway_infer_nsys_h200_intel.sh (the post-dedc9c0 revert
# restored the `_noforktrace` flavor of that script, which carries
# --sample=none AND lacks --trace-fork-before-exec=true). So if this
# script is ever rerun for a different reason, treat it as the
# (--delay=15 + --trace-fork-before-exec=true + no --sample=none)
# combination, not as "production + --delay".

ulimit -l unlimited

module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv

module unload cuda
module load cuda/12.9

echo "nsys version: $(nsys --version 2>&1 | head -1)"

export WANDB_MODE=offline

echo "=== midway_infer_nsys_delay H200 Intel: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODE=${SLURM_NODELIST}"
nvidia-smi -L
nvidia-smi topo -m

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "NUM_GPUS=${NUM_GPUS}"

config_file=/project/pedramh/shared/S2S/v2.0/config/exp2.yaml
NSYS_OUT="${SLURM_SUBMIT_DIR}/midway_h200_intel_4gpus_inference_delay_${SLURM_JOB_ID}"

echo "nsys output: ${NSYS_OUT}.nsys-rep"

# Original production flags + --delay=15 only.
nsys profile \
    -w true \
    -t cuda,nvtx,cudnn \
    -o "${NSYS_OUT}" \
    --force-overwrite=true \
    --trace-fork-before-exec=true \
    --delay=15 \
    torchrun \
        --standalone \
        --nproc_per_node="${NUM_GPUS}" \
        /project/pedramh/shared/S2S/v2.0/inference_optimized.py \
        --yaml_config="${config_file}" \
        --run_num="infer_nsys_h200_intel" \
        --async_save

echo "Profile written: ${NSYS_OUT}.nsys-rep"
echo "scp the .nsys-rep locally and run: nsys export --type=sqlite <file>.nsys-rep"
