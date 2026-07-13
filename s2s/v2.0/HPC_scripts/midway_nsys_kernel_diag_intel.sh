#!/bin/bash -l
#SBATCH --account=rcc-staff
#SBATCH --time=00:30:00
#SBATCH -p test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1              # only case D needs 4; A/B/C run on 1 GPU.
                                   # To also run case D, override at submit time:
                                   #   sbatch --gres=gpu:4 --exclusive --mem=0 ...
                                   # on a fully-free node.
#SBATCH --mem=16G
#SBATCH --nodelist=midway3-0603   # Intel Gold-6542Y, H200 DLC
                                   # alternatives if 0603 has no free GPU:
                                   # midway3-0604/0605/0606 (also Intel H200).
                                   # (0602 is the original target but was full at
                                   # time of writing.) The script auto-skips case D
                                   # when fewer than 4 GPUs are visible.
#SBATCH -o midway_nsys_kernel_diag_%N.out
#SBATCH -e midway_nsys_kernel_diag_%N.err

# Why this exists:
#   The Midway nsys captures from midway_infer_nsys_h200_*.sh have no
#   CUPTI_ACTIVITY_KIND_KERNEL entries, only RUNTIME/MEMCPY/SYNC. A minimal
#   probe (bare python + -t cuda) on the same node with the same nsys binary
#   *does* capture KERNEL events. So the failure mode is triggered by
#   something the production script has and the probe doesn't.
#
# This script adds one factor at a time and reports whether KERNEL events
# survived in each case. Reading the result table at the end pins the cause:
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

echo "=== midway_nsys_kernel_diag H200 Intel: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODE=${SLURM_NODELIST}"
echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo "nsys=$(which nsys)  $(nsys --version | head -1)"

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "NUM_GPUS=${NUM_GPUS}"
echo

OUTDIR="${SLURM_SUBMIT_DIR}/nsys_diag_${SLURM_JOB_ID}"
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
# Requires 4 GPUs in this allocation; skip if not.
if [[ "${NUM_GPUS}" -ge 4 ]]; then
    run_case D_torchrun_4 \
        nsys profile -t cuda,nvtx,cudnn --trace-fork-before-exec=true \
            -o "${OUTDIR}/D_torchrun_4" --force-overwrite=true \
            torchrun --standalone --nproc_per_node=4 "$PROBE"
else
    RESULT[D_torchrun_4]="skipped (NUM_GPUS=${NUM_GPUS}, needs 4)"
    echo
    echo ">>> CASE D_torchrun_4: skipped (NUM_GPUS=${NUM_GPUS}, needs 4)"
fi

echo
echo "============================================"
echo "SUMMARY"
echo "============================================"
for label in A_bare_cuda B_full_flags C_torchrun_1 D_torchrun_4; do
    printf "  %-18s  KERNEL = %s\n" "$label" "${RESULT[$label]:-not run}"
done
echo
echo "Profiles + sqlite in: $OUTDIR"
