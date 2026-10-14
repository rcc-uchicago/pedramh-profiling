#!/bin/bash
#SBATCH --account=bdiu-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --qos=bdiu-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:45:00
#SBATCH -o ace2_delta_nsys_%j.out
#SBATCH -e ace2_delta_nsys_%j.err
#
# ACE2 nsys profile on Delta/DeltaAI (GH200), producing a .sqlite to scp back.
#
# ⚠ UNTESTED ON DELTA. Written from ACE2_retrain/train.sh's SBATCH block; the
# account/partition/qos and the env activation almost certainly need adjusting.
# Everything adjustable is an env var -- see OVERRIDES below -- so you should not
# need to edit this file.
#
# WHY RUN IT: NCCL is expected to be a non-issue on GH200 (NVLink), but three
# ACE2 findings from Midway are architecture-level and should reproduce anywhere:
#   1. validation is ~61% of an epoch, ~52% of which is snapshot rendering whose
#      output is DISCARDED when wandb is off (may be worse on Grace ARM cores)
#   2. copies are ~28% of GPU kernel time, at the AMP/SHT boundary
#   3. ~9% launch-latency idle from ~2,900 tiny kernels per step per rank
# This capture measures all three on Delta so the two clusters are comparable.
#
# OVERRIDES (export before sbatch, or use --export):
#   ACE2_DIR       repo ACE2_retrain dir            (default: script's own dir)
#   ACE2_CONFIG    yaml to run                      (default: $ACE2_DIR/config_nsight.yaml)
#   ACE2_ACTIVATE  shell snippet to activate python (default: the train.sh pair)
#   ACE2_NSYS_BIN  nsys binary                      (default: whatever is on PATH)
#
# OUTPUT: one .sqlite next to the .nsys-rep. That is the file to scp -- it is
# self-contained and needs no nsys at the far end.

set -eo pipefail
# No `set -u`: conda's activate.d for the gxx toolchain dereferences an unset
# CONDA_BUILD_SYSROOT (documented in train.sh).

ACE2_DIR="${ACE2_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONFIG="${ACE2_CONFIG:-${ACE2_DIR}/config_nsight.yaml}"
ACTIVATE="${ACE2_ACTIVATE:-module load python/miniforge3_pytorch && source activate /scratch/midway3/krucker01/envs/fme}"

echo "=== delta_bench_nsys.sh: $(date -Iseconds) ==="
echo "host=$(hostname)  arch=$(uname -m)  job=${SLURM_JOB_ID:-none}"
eval "${ACTIVATE}" || { echo "ERROR ACE2_ACTIVATE_FAILED: ${ACTIVATE}"; exit 1; }

python -c "import torch, fme" 2>/dev/null || {
    echo "ERROR ACE2_ENV_NOT_ACTIVE python=$(command -v python)"
    echo "  set ACE2_ACTIVATE to the right activation for this cluster"; exit 1; }
[ -f "${CONFIG}" ] || { echo "ERROR ACE2_CONFIG_MISSING ${CONFIG} (set ACE2_CONFIG)"; exit 1; }

NSYS_BIN="${ACE2_NSYS_BIN:-$(command -v nsys || true)}"
[ -x "${NSYS_BIN}" ] || { echo "ERROR ACE2_NSYS_MISSING (set ACE2_NSYS_BIN)"; exit 1; }

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
unset NCCL_DEBUG
unset TORCH_DISTRIBUTED_DEBUG

# NVTX from ace2_nvtx.py: gives the shared range names AND cudaProfilerStart/Stop,
# so the capture window is exactly the measured steps and the phase breakdown is
# directly comparable to the Midway captures.
export ACE2_NVTX=1
export ACE2_NVTX_WARMUP="${ACE2_NVTX_WARMUP:-5}"
export ACE2_NVTX_STEPS="${ACE2_NVTX_STEPS:-1000000}"   # effectively never stop

# FULL TRACE by default. --capture-range=cudaProfilerApi stops collecting when
# the measured steps end, which would put VALIDATION outside the capture -- and
# validation is the main thing we want to see (61% of an epoch on Midway, ~52%
# of that rendering snapshots that get discarded). Set ACE2_NSYS_WINDOWED=1 for
# a training-only capture instead.
CAPTURE_ARGS=()
if [ "${ACE2_NSYS_WINDOWED:-0}" = "1" ]; then
    CAPTURE_ARGS=(--capture-range=cudaProfilerApi --capture-range-end=stop)
    export ACE2_NVTX_WARMUP="${ACE2_NVTX_WARMUP:-10}"
    export ACE2_NVTX_STEPS="${ACE2_NVTX_STEPS:-30}"
    echo "capture: windowed (training only)"
else
    echo "capture: FULL RUN (startup + training + validation)"
fi

NUM_GPUS=$(nvidia-smi -L | wc -l)
OUT_DIR="${ACE2_DIR}/outs/delta_nsys_${SLURM_JOB_ID:-manual}"
NSYS_OUT="${OUT_DIR}/nsys_ace2_delta_${SLURM_JOB_ID:-manual}"
mkdir -p "${OUT_DIR}"

echo "gpus=${NUM_GPUS}  config=${CONFIG}"
nvidia-smi topo -m 2>/dev/null | head -7    # so the report carries its own topology

OVERRIDES=(
    experiment_dir="${OUT_DIR}/run"
    max_epochs=1
    segment_epochs=1
    save_checkpoint=false
    checkpoint_save_epochs=null
    inference=null
    train_loader.batch_size="${ACE2_BATCH_SIZE:-4}"
    train_loader.sample_with_replacement="${ACE2_SAMPLES:-128}"
    validation_loader.batch_size="${ACE2_BATCH_SIZE:-4}"
    train_evaluation_samples=16
    log_train_every_n_batches=1
    # Bound validation so the trace stays a sane size -- the config validates
    # over 1996-1997 (~2900 samples) by default, which would dwarf the run.
    validation_loader.dataset.subset.stop_time="${ACE2_VAL_STOP:-1996-01-05}"
    # fme's GlobalTimer category breakdown only ever reaches wandb; this puts it
    # on disk so train/valid/inference split is readable without a profiler.
    logging.metrics_log_dir="${OUT_DIR}/metrics"
)

python -m fme.ace.validate_config "${CONFIG}" --config_type train \
    --override "${OVERRIDES[@]}" \
    || { echo "ERROR ACE2_CONFIG_INVALID ${CONFIG}"; exit 1; }

rc=0
"${NSYS_BIN}" profile \
    --trace=cuda,nvtx,cudnn,cublas \
    --cuda-memory-usage=true \
    "${CAPTURE_ARGS[@]}" \
    --output="${NSYS_OUT}" \
    --force-overwrite=true \
    torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
        "${ACE2_DIR}/ace2_nvtx.py" "${CONFIG}" --override "${OVERRIDES[@]}" || rc=$?
echo "--- nsys rc=${rc} (non-zero can be the capture-range stop; gate is the artifact) ---"

REPORT="${NSYS_OUT}.nsys-rep"
[ -s "${REPORT}" ] || { echo "ERROR ACE2_NSYS_NO_REPORT ${REPORT}"; exit 1; }

# Export here so the far end needs no nsys installed.
"${NSYS_BIN}" export --type=sqlite --force-overwrite=true \
    --output="${NSYS_OUT}.sqlite" "${REPORT}" || {
        echo "ERROR ACE2_SQLITE_EXPORT_FAILED -- scp the .nsys-rep instead"; exit 1; }

# --- bundle everything into ONE file, with a manifest ------------------------
# The run scatters artifacts across three directories: the trace, fme's python
# log, the GlobalTimer metrics, the resolved config, and the scheduler's own
# .out/.err. Sending them one at a time is how a set arrives incomplete and
# nobody notices until the analysis is half done. One tarball plus per-file
# checksums makes "did it all get here, intact?" answerable at the far end.
JOB="${SLURM_JOB_ID:-manual}"
STAGE="${OUT_DIR}/bundle"
BUNDLE="${ACE2_DIR}/outs/ace2_delta_${JOB}.tar.gz"
mkdir -p "${STAGE}"

cp "${NSYS_OUT}.sqlite"         "${STAGE}/" 2>/dev/null
cp "${OUT_DIR}/run/out.log"     "${STAGE}/fme_out.log" 2>/dev/null
cp "${OUT_DIR}/run/config.yaml" "${STAGE}/resolved_config.yaml" 2>/dev/null
[ -d "${OUT_DIR}/metrics" ] && cp -r "${OUT_DIR}/metrics" "${STAGE}/metrics" 2>/dev/null
# SLURM is still appending to these, so the copy stops here: the final
# ACE2_DELTA_NSYS_OK line will be absent from the captured slurm.out.
cp "${SLURM_SUBMIT_DIR:-.}/ace2_delta_nsys_${JOB}.out" "${STAGE}/slurm.out" 2>/dev/null
cp "${SLURM_SUBMIT_DIR:-.}/ace2_delta_nsys_${JOB}.err" "${STAGE}/slurm.err" 2>/dev/null
# .nsys-rep is hundreds of MB and the .sqlite is derived from it; ship it only on
# request (the raw report is needed to open the timeline in the Nsight GUI).
[ "${ACE2_BUNDLE_REP:-0}" = "1" ] && cp "${REPORT}" "${STAGE}/" 2>/dev/null

MAN="${STAGE}/MANIFEST.txt"
{
  echo "ACE2 Delta profiling bundle"
  echo "job_id      : ${JOB}"
  echo "host        : $(hostname)"
  echo "created_utc : $(date -u -Iseconds)"
  echo "nodelist    : ${SLURM_JOB_NODELIST:-<none>}"
  echo "gpus        : ${NUM_GPUS} x $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  echo "config      : ${CONFIG}"
  echo "batch_size  : ${ACE2_BATCH_SIZE:-4}   samples: ${ACE2_SAMPLES:-128}   windowed: ${ACE2_NSYS_WINDOWED:-0}"
  echo "git_commit  : $(git -C "${ACE2_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "python      : $(command -v python)"
  echo "versions    : $(python -c 'import torch,fme;print("torch",torch.__version__,"nccl",".".join(map(str,torch.cuda.nccl.version())),"fme",fme.__version__)' 2>/dev/null || echo unknown)"
  echo
  echo "topology:"
  nvidia-smi topo -m 2>/dev/null | head -7 | sed 's/^/  /'
  echo
  echo "files (sha256  bytes  name):"
  ( cd "${STAGE}" && find . -type f ! -name MANIFEST.txt -print0 \
      | xargs -0 -r sha256sum 2>/dev/null \
      | while read -r sum path; do printf "  %s  %s  %s\n" "${sum}" "$(stat -c %s "${path}")" "${path#./}"; done )
} > "${MAN}"

tar -czf "${BUNDLE}" -C "${STAGE}" . || {
    echo "ERROR ACE2_BUNDLE_FAILED -- artifacts remain in ${OUT_DIR}"; exit 1; }
cp "${MAN}" "${ACE2_DIR}/outs/ace2_delta_${JOB}.MANIFEST.txt" 2>/dev/null

BSUM=$(sha256sum "${BUNDLE}" | cut -d" " -f1)
echo "ACE2_DELTA_NSYS_OK rep_mb=$(( $(stat -c %s "${REPORT}") / 1048576 )) sqlite_mb=$(( $(stat -c %s "${NSYS_OUT}.sqlite") / 1048576 )) gpus=${NUM_GPUS}"
echo "ACE2_DELTA_BUNDLE_OK files=$(tar -tzf "${BUNDLE}" | grep -c .) size_mb=$(( $(stat -c %s "${BUNDLE}") / 1048576 ))"
echo "SCP THIS: ${BUNDLE}"
echo "sha256:   ${BSUM}"
echo "verify at the far end:  sha256sum ace2_delta_${JOB}.tar.gz   # expect ${BSUM}"
echo "manifest without extracting: ${ACE2_DIR}/outs/ace2_delta_${JOB}.MANIFEST.txt"
