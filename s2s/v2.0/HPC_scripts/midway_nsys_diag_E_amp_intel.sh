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
#SBATCH --nodelist=midway3-0604   # Intel Gold-6542Y, H200 DLC.
                                   # Intentionally NOT 0603 -- this script is
                                   # meant to be submitted in parallel with
                                   # midway_nsys_inference_short_intel.sh
                                   # (case G), which pins 0603. Same --nodelist
                                   # + --exclusive on both would serialize them.
                                   # Alternatives if 0604 is full: 0605/0606.
#SBATCH -o midway_nsys_diag_E_amp_intel_%N_%j.out
#SBATCH -e midway_nsys_diag_E_amp_intel_%N_%j.err

# Case E in the production-breakage bisect.
#
# Background:
#   midway_nsys_kernel_diag_intel_4gpu.sh proved that the nsys-invocation
#   factors (trace flags, fork tracking, torchrun launcher, multi-rank) all
#   capture CUPTI_ACTIVITY_KIND_KERNEL fine on midway3-0603 with the same
#   nsys binary and trace flags the production script uses. So the
#   production breakage has to come from something the real workload does
#   that the probe doesn't. The single most likely factor is AMP bfloat16
#   autocast (inference_optimized.py:123 wraps the whole inference loop in
#   torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16); the
#   probe doesn't autocast). H100/H200 + bf16 hits specialized wgmma /
#   Hopper kernels that some nsys/CUPTI builds handle badly.
#
# This script reruns the same matmul probe wrapped in AMP bfloat16 autocast,
# under the production trace flags + fork-before-exec, in three configurations:
#   E1 = bare python + AMP (1 GPU)
#   E2 = torchrun nproc=1 + AMP
#   E3 = torchrun nproc=4 + AMP  (matches the production layout exactly except
#                                  for the actual model)
#
# Interpreting the SUMMARY:
#   E1 fails              -> AMP autocast alone breaks kernel capture on this
#                            nsys/driver. Production fix: drop AMP or switch
#                            precision path.
#   E1 passes, E3 fails   -> AMP-x-multi-rank interaction. Less likely; look
#                            at NCCL bf16 collectives.
#   all pass              -> AMP isn't the trigger. The next bisect target
#                            is real-model-topology or data loader (case F).

ulimit -l unlimited

module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv

module unload cuda
module load cuda/12.9

echo "=== midway_nsys_diag_E_amp Intel: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODE=${SLURM_NODELIST}"
echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo "nsys=$(which nsys)  $(nsys --version | head -1)"

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "NUM_GPUS=${NUM_GPUS}"
echo

OUTDIR="${SLURM_SUBMIT_DIR}/nsys_diag_E_amp_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"
PROBE="${OUTDIR}/probe.py"

# Same probe as midway_nsys_kernel_diag_intel.sh BUT wrapped in
# torch.amp.autocast('cuda', dtype=torch.bfloat16). This is exactly the
# autocast invocation inference_optimized.py:123 uses; the only thing
# missing from the production setup is the actual model + data loader.
cat > "$PROBE" <<'PY'
import os, torch
lr = int(os.environ.get('LOCAL_RANK', 0))
if 'LOCAL_RANK' in os.environ:
    import torch.distributed as dist
    torch.cuda.set_device(lr)
    dist.init_process_group('nccl')
with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
    x = torch.randn(1024, 1024, device=f'cuda:{lr}')
    for _ in range(50):
        x = x @ x
torch.cuda.synchronize()
print(f'rank={lr} done (amp bf16)', flush=True)
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

# E1: bare python + AMP, 1 GPU. Tests whether AMP alone trips CUPTI.
run_case E1_bare_amp \
    nsys profile -t cuda,nvtx,cudnn --trace-fork-before-exec=true \
        -o "${OUTDIR}/E1_bare_amp" --force-overwrite=true \
        python "$PROBE"

# E2: torchrun nproc=1 + AMP. Adds the launcher.
run_case E2_torchrun_1_amp \
    nsys profile -t cuda,nvtx,cudnn --trace-fork-before-exec=true \
        -o "${OUTDIR}/E2_torchrun_1_amp" --force-overwrite=true \
        torchrun --standalone --nproc_per_node=1 "$PROBE"

# E3: torchrun nproc=4 + AMP. Matches the production layout exactly except
# for the actual model code.
if [[ "${NUM_GPUS}" -ge 4 ]]; then
    run_case E3_torchrun_4_amp \
        nsys profile -t cuda,nvtx,cudnn --trace-fork-before-exec=true \
            -o "${OUTDIR}/E3_torchrun_4_amp" --force-overwrite=true \
            torchrun --standalone --nproc_per_node=4 "$PROBE"
else
    RESULT[E3_torchrun_4_amp]="skipped (NUM_GPUS=${NUM_GPUS}, needs 4)"
    echo
    echo ">>> CASE E3_torchrun_4_amp: skipped (NUM_GPUS=${NUM_GPUS}, needs 4)"
fi

echo
echo "============================================"
echo "SUMMARY"
echo "============================================"
for label in E1_bare_amp E2_torchrun_1_amp E3_torchrun_4_amp; do
    printf "  %-22s  KERNEL = %s\n" "$label" "${RESULT[$label]:-not run}"
done
echo
echo "Profiles + sqlite in: $OUTDIR"
