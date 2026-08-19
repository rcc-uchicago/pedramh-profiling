#!/bin/bash
#SBATCH --account=pi-pedramh
#SBATCH --time=01:30:00
#SBATCH -p pedramh-gpu
#SBATCH --qos=pedramh-gpu
#SBATCH --constraint=H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o ace2_bench_nsys_%x_%j.out
#SBATCH -e ace2_bench_nsys_%x_%j.err
#
# ACE2 (ai2cm `fme`) nsys capture on 4x H100 (pedramh-gpu, midway3-0423).
#
# NOTE the A100-PCIE profile in bench_midway_notes.md was taken in the `test`
# partition before the 2026-08-19 restriction to pedramh-gpu. H100 numbers are
# not directly comparable to it -- and the ACE2_NSYS_DELAY/DURATION defaults
# were measured on A100, where startup and the training window sit later. Check
# them against a smoke's timeline on this node before trusting a capture.
#
# PASS = the line `ACE2_NSYS_OK` in the .out (a non-trivial .nsys-rep on disk
# AND a finite loss). Key on the token, not the exit code.
#
# TWO CAPTURE MODES
# -----------------
# ACE2_NVTX=1  -- the house mode. ace2_nvtx.py injects the shared NVTX range
#   names and calls cudaProfilerStart/Stop around the steady-state steps, so
#   this uses the same
#       --capture-range=cudaProfilerApi --capture-range-end=stop
#   flags as s2s/SI/port, the window is exactly the measured steps, and
#   parse_nsys.py produces a real phase breakdown. Prefer this.
#
# ACE2_NVTX=0 (default) -- legacy fallback for a stock fme, which ships NO
#   cudaProfilerApi, torch.profiler or NVTX in the SFNO lat-lon path. The house
#   capture-range flags would capture nothing, so a hand-derived time window is
#   the only option:
#     ACE2_NSYS_DELAY     seconds to skip before collecting (default 0)
#     ACE2_NSYS_DURATION  seconds to collect (default 0 = until exit)
#   Those must be re-derived per hardware -- the A100-era 45/110 opened AFTER
#   training had finished on H100 (the working H100 pair is 18/55). A window
#   capture also includes startup and validation, and nothing marks where warmup
#   ends. This mode exists so an uninstrumented tree can still be profiled; it is
#   not the one to reach for.
#
# osrt is dropped from --trace (the house scripts keep it) purely to keep an
# unwindowed report to a sane size; add it back for a windowed capture.
#
# HELD IDENTICAL to midway_smoke_train.sh so the two are comparable: env block,
# batch_size, model, loss, optimizer, AMP, normalization. The only differences
# are a longer epoch (512 samples, so the timeline has enough steady-state
# steps to average) and nsys itself.
#
# Eager only -- no torch.compile. The house rule is to profile eager because
# CUDA-graph kernels do not appear in CUPTI_ACTIVITY_KIND_KERNEL; fme does not
# compile by default, so nothing needs disabling here.

set -eo pipefail
# No `set -u`: the gxx_linux-64 activate.d hook dereferences CONDA_BUILD_SYSROOT.

ACE2_DIR=/project/rcc/mehta5/pedramh-profiling/ACE2_retrain
CONFIG="${ACE2_DIR}/config_midway.yaml"
FME_ENV=/project/rcc/mehta5/envs/fme

# --- env bootstrap (identical to midway_smoke_train.sh) --------------------
module load python/miniforge-25.3.0

eval "$(conda shell.bash hook)"
conda activate "${FME_ENV}"

module unload cuda
module load cuda/12.6

python -c "import torch, fme" 2>/dev/null || {
    echo "ERROR ACE2_ENV_NOT_ACTIVE python=$(command -v python) env=${FME_ENV}"
    exit 1
}
command -v nsys >/dev/null || { echo "ERROR ACE2_NSYS_MISSING after module load cuda/12.6"; exit 1; }

# --- bench-neutral env (must match midway_smoke_train.sh) ------------------
unset NCCL_DEBUG
unset TORCH_DISTRIBUTED_DEBUG

export WANDB_MODE=offline
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

NUM_GPUS=$(nvidia-smi -L | wc -l)
EXP_DIR="${ACE2_DIR}/outs/midway_nsys_${SLURM_JOB_ID}"
NSYS_OUT="${EXP_DIR}/nsys_ace2_eager_${SLURM_JOB_ID}"
mkdir -p "${EXP_DIR}"

ACE2_NVTX="${ACE2_NVTX:-0}"
export ACE2_NVTX
NSYS_DELAY="${ACE2_NSYS_DELAY:-0}"
NSYS_DURATION="${ACE2_NSYS_DURATION:-0}"
WINDOW_ARGS=()
[ "${NSYS_DELAY}" != "0" ]    && WINDOW_ARGS+=("--delay=${NSYS_DELAY}")
[ "${NSYS_DURATION}" != "0" ] && WINDOW_ARGS+=("--duration=${NSYS_DURATION}")

echo "=== midway_bench_nsys.sh: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODELIST=${SLURM_NODELIST}  NUM_GPUS=${NUM_GPUS}"
echo "capture: delay=${NSYS_DELAY}s duration=${NSYS_DURATION}s (0 = unbounded)"
nvidia-smi -L

OVERRIDES=(
    experiment_dir="${EXP_DIR}"
    max_epochs=1
    segment_epochs=1
    save_checkpoint=false
    checkpoint_save_epochs=null
    inference=null
    train_loader.batch_size=4
    train_loader.sample_with_replacement=512
    validation_loader.batch_size=4
    validation_loader.dataset.subset.stop_time=1996-01-05
    train_evaluation_samples=16
    log_train_every_n_batches=10
)

# Pre-flight before nsys attaches -- a config error inside a profiled run wastes
# the whole allocation and still writes a useless report.
python -m fme.ace.validate_config "${CONFIG}" --config_type train \
    --override "${OVERRIDES[@]}" \
    || { echo "ERROR ACE2_CONFIG_INVALID ${CONFIG}"; exit 1; }

# NOTE the `|| nsys_rc=$?`. When --duration expires, nsys SIGTERMs the target,
# so torchrun's elastic agent dies with SignalException(signal 15) and exits
# non-zero. That is the EXPECTED outcome of a windowed capture, not a failure --
# job 53524918 wrote a valid 217 MB report and was still marked FAILED because
# `set -e` aborted here before the gate could run. Rule #14: gate on the
# artifact, never on the exit code.
nsys_rc=0
# With ACE2_NVTX=1 the injector emits the shared range names AND calls
# cudaProfilerStart/Stop around the steady-state steps, so we can finally use
# the house capture-range flags instead of a hand-derived time window. Without
# it, fall back to --delay/--duration, which is all an uninstrumented fme allows.
if [ "${ACE2_NVTX}" = "1" ]; then
    CAPTURE_ARGS=(--capture-range=cudaProfilerApi --capture-range-end=stop)
    ENTRY=("${ACE2_DIR}/ace2_nvtx.py")
    echo "capture: cudaProfilerApi (NVTX enabled, warmup=${ACE2_NVTX_WARMUP:-20} steps=${ACE2_NVTX_STEPS:-80})"
else
    CAPTURE_ARGS=("${WINDOW_ARGS[@]}")
    ENTRY=(-m fme.ace.train)
fi

nsys profile \
    --trace=cuda,nvtx,cudnn,cublas \
    --cuda-memory-usage=true \
    --output="${NSYS_OUT}" \
    --force-overwrite=true \
    "${CAPTURE_ARGS[@]}" \
    torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
        "${ENTRY[@]}" "${CONFIG}" --override "${OVERRIDES[@]}" || nsys_rc=$?
if [ "${nsys_rc}" -ne 0 ]; then
    if [ "${NSYS_DURATION}" != "0" ]; then
        echo "note: rc=${nsys_rc} is expected -- nsys terminates the app when --duration ends"
    else
        echo "ERROR ACE2_NSYS_APP_FAILED rc=${nsys_rc} with no --duration set, so this is a real failure"
        exit 1
    fi
fi

# --- PASS gate ------------------------------------------------------------
REPORT="${NSYS_OUT}.nsys-rep"
[ -s "${REPORT}" ] || { echo "ERROR ACE2_NSYS_NO_REPORT expected ${REPORT}"; exit 1; }
rep_bytes=$(stat -c %s "${REPORT}")
[ "${rep_bytes}" -gt 1048576 ] || { echo "ERROR ACE2_NSYS_REPORT_TOO_SMALL ${rep_bytes} bytes -- capture window likely missed the run"; exit 1; }

LOG="${EXP_DIR}/out.log"
train_loss=$(awk -F'Train loss: ' '/Train loss: /{v=$2} END{print v}' "${LOG}" 2>/dev/null)
# An epoch-end loss only exists if the run got that far. A windowed capture is
# killed mid-epoch by design, so absence is fine -- but a NaN/Inf never is.
case "${train_loss}" in
    *[nN][aA][nN]*|*[iI][nN][fF]*)
        echo "ERROR ACE2_NSYS_LOSS_NOT_FINITE train=${train_loss} (report kept at ${REPORT})"
        exit 1
        ;;
    "") train_loss="<run windowed, no epoch-end loss>" ;;
esac

echo "ACE2_NSYS_OK train_loss=${train_loss} report_mb=$((rep_bytes / 1048576)) n_gpus=${NUM_GPUS}"
echo "report=${REPORT}"
echo "next: nsys export --type=sqlite ${REPORT}"
