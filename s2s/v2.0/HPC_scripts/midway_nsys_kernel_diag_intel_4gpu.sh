#!/bin/bash -l
#SBATCH --account=rcc-staff
#SBATCH --time=00:30:00
#SBATCH -p test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodelist=midway3-0603   # Intel Gold-6542Y, H200 DLC
                                   # alternatives if 0603 isn't fully idle:
                                   # midway3-0604/0605/0606 (also Intel H200,
                                   # also need to be empty for --exclusive to
                                   # schedule). Edit this line if 0603 is busy.
#SBATCH -o midway_nsys_kernel_diag_4gpu_%N.out
#SBATCH -e midway_nsys_kernel_diag_4gpu_%N.err

# Full-node variant of midway_nsys_kernel_diag_intel.sh — runs A/B/C/D
# instead of A/B/C-with-D-skipped. Use this when an Intel H200 node is
# fully idle and you want to also test case D (torchrun nproc=4), which
# matches the production midway_infer_nsys_h200_*.sh layout that's been
# dropping CUPTI_ACTIVITY_KIND_KERNEL.
#
# Decision table for the SUMMARY block at the end of the .out:
#
#   A passes, B fails  -> -t cudnn or --trace-fork-before-exec breaks it
#   B passes, C fails  -> torchrun launcher around nsys breaks CUPTI attach
#   C passes, D fails  -> multi-rank contention on CUPTI
#   all pass           -> issue is workload-specific (real model, runtime length)

ulimit -l unlimited

module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv

module unload cuda
module load cuda/12.9

echo "=== midway_nsys_kernel_diag 4gpu H200 Intel: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODE=${SLURM_NODELIST}"
echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo "nsys=$(which nsys)  $(nsys --version | head -1)"

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "NUM_GPUS=${NUM_GPUS}"
echo

OUTDIR="${SLURM_SUBMIT_DIR}/nsys_diag_4gpu_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"
PROBE="${OUTDIR}/probe.py"

cat > "$PROBE" <<'PY'
import os, torch
lr = int(os.environ.get('LOCAL_RANK', 0))
if 'LOCAL_RANK' in os.environ:
    import torch.distributed as dist
    torch.cuda.set_device(lr)
    dist.init_process_group('nccl')
x = torch.randn(1024, 1024, device=f'cuda:{lr}')
for _ in range(50):
    x = x @ x
torch.cuda.synchronize()
print(f'rank={lr} done', flush=True)
PY

declare -A RESULT

run_case() {
    local label=$1
    shift
    echo
    echo "############################################"
    echo "### CASE $label"
    echo "### cmd: $*"
    echo "############################################"
    "$@" 2>&1 | tail -20
    local rep="${OUTDIR}/${label}.nsys-rep"
    local db="${OUTDIR}/${label}.sqlite"
    nsys export --type=sqlite -o "$db" --force-overwrite=true "$rep" >/dev/null 2>&1
    local n
    n=$(sqlite3 "$db" "SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL;" 2>/dev/null)
    if [[ -z "$n" ]]; then
        RESULT[$label]="TABLE_MISSING"
    else
        RESULT[$label]="$n events"
    fi
    echo ">>> CASE $label: KERNEL = ${RESULT[$label]}"
}

# A: minimal — bare python, -t cuda only. Known-good baseline.
run_case A_bare_cuda \
    nsys profile -t cuda \
        -o "${OUTDIR}/A_bare_cuda" --force-overwrite=true \
        python "$PROBE"

# B: production trace flags + fork tracking, still bare python.
run_case B_full_flags \
    nsys profile -t cuda,nvtx,cudnn --trace-fork-before-exec=true \
        -o "${OUTDIR}/B_full_flags" --force-overwrite=true \
        python "$PROBE"

# C: full flags + torchrun, single rank. Adds torchrun launcher.
run_case C_torchrun_1 \
    nsys profile -t cuda,nvtx,cudnn --trace-fork-before-exec=true \
        -o "${OUTDIR}/C_torchrun_1" --force-overwrite=true \
        torchrun --standalone --nproc_per_node=1 "$PROBE"

# D: full flags + torchrun, 4 ranks. Matches the production broken config.
# This variant of the script is run with --gres=gpu:4 --exclusive precisely
# so D can run; abort loudly if NUM_GPUS is short, instead of silently
# skipping like the 1-GPU sibling script does.
if [[ "${NUM_GPUS}" -lt 4 ]]; then
    echo "ERROR: this is the 4-GPU variant of the diagnostic but only NUM_GPUS=${NUM_GPUS} visible." >&2
    echo "       Either rerun under --gres=gpu:4 --exclusive, or use" >&2
    echo "       midway_nsys_kernel_diag_intel.sh (the 1-GPU variant) instead." >&2
    exit 1
fi
run_case D_torchrun_4 \
    nsys profile -t cuda,nvtx,cudnn --trace-fork-before-exec=true \
        -o "${OUTDIR}/D_torchrun_4" --force-overwrite=true \
        torchrun --standalone --nproc_per_node=4 "$PROBE"

echo
echo "============================================"
echo "SUMMARY"
echo "============================================"
for label in A_bare_cuda B_full_flags C_torchrun_1 D_torchrun_4; do
    printf "  %-18s  KERNEL = %s\n" "$label" "${RESULT[$label]:-not run}"
done
echo
echo "Profiles + sqlite in: $OUTDIR"
