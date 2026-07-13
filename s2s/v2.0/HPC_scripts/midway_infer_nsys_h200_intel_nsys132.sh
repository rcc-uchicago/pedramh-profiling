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
#SBATCH -o midway_infer_nsys_nsys132_%N_%j.out
#SBATCH -e midway_infer_nsys_nsys132_%N_%j.err

# nsys-binary-swap diagnostic sibling of midway_infer_nsys_h200_intel.sh.
#
# Purpose:
#   Three independent Intel production runs (noforktrace 50084384,
#   newinfer 50084385, delay 50102626) produced byte-identical CUPTI
#   counts -- RUNTIME=4252 / MEMCPY=1960 / SYNC=1960, KERNEL/MEMSET/
#   CUDA_EVENT/OVERHEAD all missing -- across very different nsys flag
#   combinations and across both inference scripts. Even --delay=15
#   produced the same ~3 s capture window, just shifted later in the
#   run -- proving the failure is "CUPTI dies after ~3 s of capture
#   regardless of when capture starts," not "CUPTI dies at a specific
#   point in the workload."
#
#   That pattern is consistent with a bug in nsys 2025.1.3 (which ships
#   with cuda/12.9 on Midway) + driver 535.216.03 + sustained
#   high-kernel-rate Hopper workloads.
#
#   CORRECTION (post-hoc, after job 50117688): the earlier kernel-diag
#   matrix did NOT test cuda-13.2/bin/nsys. It used
#   /software/cuda-12.9-el8-x86_64/bin/nsys (via `which nsys` after
#   `module load cuda/12.9`), and that binary reports
#   NVIDIA Nsight Systems version 2025.1.3.140-251335620677v0. The
#   cuda-13.2 nsys binary on Midway ALSO reports
#   2025.1.3.140-251335620677v0 -- same release, different build path,
#   not a version upgrade. So this script is a "different-build-of-
#   same-version" test, not a version-upgrade test. Job 50117688
#   confirmed it: byte-identical RUNTIME=4252 / MEMCPY=1960 / SYNC=1960
#   and same missing KERNEL/MEMSET/CUDA_EVENT/OVERHEAD as the
#   cuda-12.9 nsys captures.
#
#   What the kernel-diag matrix DID prove (per its own decision table):
#   cases A/B/C/D all captured KERNEL events with the cuda-12.9 nsys,
#   even in the full production layout (torchrun --nproc_per_node=4 +
#   --trace-fork-before-exec=true + -t cuda,nvtx,cudnn). So per
#   "all pass -> workload-specific" the bug is triggered by something
#   the real inference workload does that the 50-iter 1024x1024 matmul
#   probe doesn't.
#
# Test:
#   Invoke /software/cuda-13.2-el8-x86_64/bin/nsys with an absolute
#   path. Keep cuda/12.9 module loaded so the CUDA runtime the
#   workload links against stays unchanged -- the ONLY variable vs
#   the broken original production capture is the nsys binary path
#   (CUPTI bundle).
#
#   - KERNEL captured -> CUPTI bundle from cuda-13.2 fixes it.
#   - KERNEL missing  -> the bug isn't in the nsys binary; it's
#                        workload-specific. Next axis to vary is the
#                        workload itself (entry point, num_workers,
#                        AMP, kernel launch rate), or pull a genuinely
#                        newer nsys (>=2025.5) from NVIDIA's developer
#                        site.
#
# FURTHER CORRECTION (2026-05-26): the "workload-specific" framing above is
# now also moot. The actual cause of the broken fingerprint was
# inference[_optimized].py crashing at restore_checkpoint() on
# FileNotFoundError before launching any kernel; the 4252 / 1960 / 1960
# events are just the CUDA APIs from PanguModel_Plasim(...).to(device) up
# to the crash point. Fixed in commit 56f73fe by gating restore_checkpoint
# on os.path.isfile. See memory: project_midway_cupti_kernel_missing.md.
# The nsys-binary-swap test this script performs (cuda-12.9 vs cuda-13.2,
# both 2025.1.3.140) is still mechanically valid as a build-comparison
# exercise, but it does not have the diagnostic significance the original
# framing implied.

ulimit -l unlimited

module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv

module unload cuda
module load cuda/12.9

# Override only the nsys binary -- leave everything else the same.
NSYS=/software/cuda-13.2-el8-x86_64/bin/nsys
echo "nsys binary override: ${NSYS}"
echo "nsys version: $("${NSYS}" --version 2>&1 | head -1)"
echo "(for reference, default cuda/12.9 nsys would be: $(which nsys))"

export WANDB_MODE=offline

echo "=== midway_infer_nsys_nsys132 H200 Intel: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODE=${SLURM_NODELIST}"
nvidia-smi -L
nvidia-smi topo -m

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "NUM_GPUS=${NUM_GPUS}"

config_file=/project/pedramh/shared/S2S/v2.0/config/exp2.yaml
NSYS_OUT="${SLURM_SUBMIT_DIR}/midway_h200_intel_4gpus_inference_nsys132_${SLURM_JOB_ID}"

echo "nsys output: ${NSYS_OUT}.nsys-rep"

# Original production nsys flags intentionally preserved
# (--trace-fork-before-exec=true present, no --sample=none, no --delay,
# no --cuda-flush-interval) so the nsys binary swap is the only variable
# vs the broken pre-flush original capture.
"${NSYS}" profile \
    -w true \
    -t cuda,nvtx,cudnn \
    -o "${NSYS_OUT}" \
    --force-overwrite=true \
    --trace-fork-before-exec=true \
    torchrun \
        --standalone \
        --nproc_per_node="${NUM_GPUS}" \
        /project/pedramh/shared/S2S/v2.0/inference_optimized.py \
        --yaml_config="${config_file}" \
        --run_num="infer_nsys_h200_intel" \
        --async_save

echo "Profile written: ${NSYS_OUT}.nsys-rep"
echo "scp the .nsys-rep locally and run: nsys export --type=sqlite <file>.nsys-rep"
