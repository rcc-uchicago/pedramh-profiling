#!/bin/bash -l
#SBATCH --account=rcc-staff
#SBATCH --time=00:20:00
#SBATCH -p test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodelist=midway3-0603   # Intel Gold-6542Y, H200 DLC
                                   # alternatives: midway3-0604/0605/0606
                                   # (use whichever is idle; 0603 is where the
                                   # kernel-diag matrix passed end-to-end.)
#SBATCH -o midway_nsys_inference_short_intel_%N_%j.out
#SBATCH -e midway_nsys_inference_short_intel_%N_%j.err

# MOOT 2026-05-26 -- the CUPTI investigation this script was part of is
# RESOLVED. The broken-capture fingerprint that motivated it (RUNTIME=4252 /
# MEMCPY=1960 / SYNC=1960, KERNEL missing) was inference[_optimized].py
# crashing at restore_checkpoint() on FileNotFoundError before launching any
# kernel -- not a workload-internal CUPTI failure or a runtime-length issue.
# Fixed in commit 56f73fe by gating restore_checkpoint on os.path.isfile.
# See memory: project_midway_cupti_kernel_missing.md.
#
# The script itself is still useful as a bounded inference profile (it
# synthesises a stub checkpoint so the workload runs even when the real
# ckpt.tar is inaccessible, then caps the trace at 180 s). The diagnostic
# framing in the rest of this docstring is preserved as historical context.
# The SUMMARY/VERDICT block has been rewritten in neutral post-resolution
# language.
#
# Purpose (historical, pre-resolution):
#   The kernel-diag matrix (midway_nsys_kernel_diag_intel_4gpu.sh) showed
#   that all of the nsys-invocation factors -- trace flags, fork tracking,
#   torchrun launcher, multi-rank CUPTI -- pass on midway3-0603 with
#   driver 535.216.03 / nsys 2025.1.3 from cuda/12.9. Yet the production
#   midway_infer_nsys_h200_intel.sh still drops CUPTI_ACTIVITY_KIND_KERNEL
#   from the resulting sqlite. So the breakage has to be inside
#   inference_optimized.py itself.
#
#   This script runs the real inference under nsys but caps the collection
#   with --duration=180 so we get a bounded trace instead of waiting for the
#   full 15-day forecast loop. With nsys's default --kill behavior, the
#   python process is SIGTERM'd after 180 s so the job finishes quickly.
#   180 s leaves headroom over the ~30-60 s of model load + checkpoint load
#   + NCCL init before the first kernel actually launches.
#
# Reading the SUMMARY at the end:
#   KERNEL captured (N events)    -> CUPTI is fine on the real workload too;
#                                    the production breakage is somehow tied
#                                    to runtime *length* (CUPTI buffer drops
#                                    over long runs), not the workload itself.
#   KERNEL table missing          -> structural breakage inside
#                                    inference_optimized.py. Bisect next:
#                                    AMP autocast vs HDF5 DataLoader workers
#                                    vs the real model graph.
#   Everything zero / missing     -> CUPTI failed to initialize at all;
#                                    inspect nsys stderr -- usually a CUPTI
#                                    version mismatch or device-cap issue.

ulimit -l unlimited

module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv

module unload cuda
module load cuda/12.9

export WANDB_MODE=offline

echo "=== midway_nsys_inference_short Intel: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODE=${SLURM_NODELIST}"
echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo "nsys=$(which nsys)  $(nsys --version | head -1)"
nvidia-smi -L
echo

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "NUM_GPUS=${NUM_GPUS}"

config_file=/project/pedramh/shared/S2S/v2.0/config/exp2.yaml
RUN_NUM="infer_nsys_intel_diag"
# Jobid-suffix the nsys output so concurrent / repeated G submissions don't
# overwrite each other's .nsys-rep / .sqlite.
NSYS_OUT="${SLURM_SUBMIT_DIR}/midway_h200_intel_4gpus_inference_short_${SLURM_JOB_ID}"

# inference_optimized.py:412-416 hardcodes
#   checkpoint_path = ${cwd}/results/S2S/${run_num}/training_checkpoints/ckpt.tar
# and restore_checkpoint() does an unconditional torch.load on it. There's no
# real checkpoint on the user's setup, so we synthesize a minimal stub here:
# {'iters': 0, 'epoch': 0, 'model_state': {}}. That works because
# restore_checkpoint loads with strict=False -- an empty model_state just
# leaves the model at its random init, and the diagnostic doesn't care about
# weight values (CUDA kernel sequences depend on model topology, not weights).
# This is also written under an isolated --run_num so we don't collide with
# any real run dirs.
CKPT_DIR="${SLURM_SUBMIT_DIR}/results/S2S/${RUN_NUM}/training_checkpoints"
CKPT_FILE="${CKPT_DIR}/ckpt.tar"
mkdir -p "$CKPT_DIR"
# Write atomically: a concurrent G submission's [[ ! -f ]] check could see
# the file mid-write if we wrote directly to CKPT_FILE. tempfile-then-rename
# is atomic on the same filesystem, so concurrent torch.load can never see
# a partial file.
if [[ ! -f "$CKPT_FILE" ]]; then
    echo "Stub checkpoint missing; generating at ${CKPT_FILE}"
    TMP_CKPT="${CKPT_FILE}.tmp.${SLURM_JOB_ID}"
    python -c "import torch; torch.save({'iters': 0, 'epoch': 0, 'model_state': {}}, '${TMP_CKPT}')"
    mv -n "$TMP_CKPT" "$CKPT_FILE"
    rm -f "$TMP_CKPT"   # cleanup if mv -n lost the race (another G beat us)
else
    echo "Stub checkpoint already present: ${CKPT_FILE}"
fi
ls -lh "$CKPT_FILE"

echo "nsys output: ${NSYS_OUT}.nsys-rep"

# Same trace flags as the production midway_infer_nsys_h200_intel.sh, with
# --duration=180 added to bound the capture and --kill defaulting to SIGTERM
# so the python process is shut down once the trace closes.
nsys profile \
    -w true \
    -t cuda,nvtx,cudnn \
    -o "${NSYS_OUT}" \
    --force-overwrite=true \
    --trace-fork-before-exec=true \
    --duration=180 \
    torchrun \
        --standalone \
        --nproc_per_node="${NUM_GPUS}" \
        /project/pedramh/shared/S2S/v2.0/inference_optimized.py \
        --yaml_config="${config_file}" \
        --run_num="${RUN_NUM}" \
        --async_save

echo "Profile written: ${NSYS_OUT}.nsys-rep"

if [[ ! -s "${NSYS_OUT}.nsys-rep" ]]; then
    echo "ERROR: ${NSYS_OUT}.nsys-rep is missing or empty; nsys did not finalize." >&2
    exit 1
fi

nsys export --type=sqlite -o "${NSYS_OUT}.sqlite" --force-overwrite=true "${NSYS_OUT}.nsys-rep"

if [[ ! -s "${NSYS_OUT}.sqlite" ]]; then
    echo "ERROR: sqlite export failed; ${NSYS_OUT}.sqlite is missing or empty." >&2
    exit 1
fi

echo
echo "============================================"
echo "SUMMARY"
echo "============================================"
echo "CUPTI activity tables present in ${NSYS_OUT}.sqlite:"
sqlite3 "${NSYS_OUT}.sqlite" "SELECT name FROM sqlite_master WHERE name LIKE 'CUPTI_ACTIVITY%' ORDER BY name;"
echo

# Counts for the host-side tables we know are always present (RUNTIME hooks
# survive even when GPU-side activity is broken) and the GPU-side ones that
# the production captures are missing. Comparing these lets us tell
# "CUPTI completely failed" (everything zero) from "kernels dropped but host
# hooks fine" (the production pattern).
for tbl in CUPTI_ACTIVITY_KIND_RUNTIME \
           CUPTI_ACTIVITY_KIND_MEMCPY \
           CUPTI_ACTIVITY_KIND_SYNCHRONIZATION \
           CUPTI_ACTIVITY_KIND_KERNEL \
           CUPTI_ACTIVITY_KIND_MEMSET; do
    n=$(sqlite3 "${NSYS_OUT}.sqlite" "SELECT count(*) FROM ${tbl};" 2>/dev/null)
    if [[ -z "$n" ]]; then
        printf "  %-40s TABLE MISSING\n" "$tbl"
    else
        printf "  %-40s %s events\n" "$tbl" "$n"
    fi
done

echo
KERNEL_COUNT=$(sqlite3 "${NSYS_OUT}.sqlite" "SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL;" 2>/dev/null)
# Post-resolution verdict (see MOOT banner at top of file). The original
# "is the breakage structural in inference_optimized.py" / "is it duration"
# branches were pre-resolution framing; both possibilities are now closed.
if [[ -z "$KERNEL_COUNT" ]]; then
    echo "VERDICT: KERNEL table missing. The workload did not launch any kernels."
    echo "         Inspect .err: pre-fix this was FileNotFoundError at"
    echo "         restore_checkpoint(); other crashes during model init can"
    echo "         produce the same fingerprint. See memory:"
    echo "         project_midway_cupti_kernel_missing.md."
elif [[ "$KERNEL_COUNT" -eq 0 ]]; then
    echo "VERDICT: KERNEL table present but empty. CUPTI subscribed but recorded"
    echo "         no kernels; unusual state, inspect nsys stderr above."
else
    echo "VERDICT: KERNEL captured (${KERNEL_COUNT} events) on a bounded 180s run."
    echo "         Capture is healthy."
fi
