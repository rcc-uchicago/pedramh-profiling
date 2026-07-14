#!/bin/bash
#SBATCH --account=rcc-staff
#SBATCH --time=00:30:00
#SBATCH --partition=test
#SBATCH --constraint=H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -o si_optim_%x_%j.out
#SBATCH -e si_optim_%x_%j.err

# =============================================================================
# bench_optim_sweep.sh — A/B sweep of the remaining optimization knobs.
#
# Each remaining optimization is tested one lever at a time against a clean
# baseline, exactly as the bench notes recommend ("change one lever, measure,
# repeat") — never all at once, so a regression (e.g. torch.compile silently
# falling back to eager, or bf16 gradient compression destabilising the loss)
# is attributable.
#
# DUAL MODE
# ─────────
#   Login node (driver):   bash bench_optim_sweep.sh
#       Submits the whole sweep — one 4-GPU job per config — to any free
#       H100 node in the test partition (--constraint=H100).
#       Each job writes its own CSV: bench_optim_<tag>_results.csv
#
#   Under SLURM (job):     sbatch invokes this file with RUN_TAG set
#       Runs a single bench config and appends one CSV row. You normally do
#       not call this directly; the driver does it for you. To run one config
#       by hand:  sbatch --export=ALL,RUN_TAG=foo,SI_TORCH_COMPILE=1 bench_optim_sweep.sh
#       Add SI_NVTX=1 to also emit per-rank profiles
#       (si_optim_<tag>_<jobid>_rank<N>.nsys-rep); the CSV row is still
#       written, with ~5% profiler overhead, so use a distinct RUN_TAG.
#       NB: do NOT combine SI_NVTX=1 with SI_TORCH_COMPILE=1 — nsys tracing
#       the compiled DDP backward segfaults (exit 139). Profile eager.
#
# WHAT THE SWEEP COVERS (env / CLI controllable today; see bench.py)
#   baseline         current best config (bucket=200, bf16-compress, bf16-mixed,
#                    bs=4). This is the clean wall-clock the notes only PROJECTED
#                    at ~21.8 samples/s from the profile — measure it for real.
#   preopt_ddp       25 MB buckets, no bf16 compress: reproduces the pre-DDP-fix
#                    ~928 ms/step baseline as the A/B reference point.
#   compile          torch.compile(default) on the DiT (notes item #1; warmup→40).
#                    MEASURED: -38% step, +62% throughput, -10 GiB peak vs baseline.
#   bucket400        bucket_cap_mb=400 ablation (notes item #5).
#   batch6           batch_size 4→6 to use the NVL headroom (notes item #3).
#   compile_batch6   torch.compile + batch_size 6 — the expected best combo.
#
# (max-autotune was dropped: it captures CUDA graphs, which crash on this model's
#  reused outputs, and its autotuning OOMs the host across 4 ranks. Re-add as
#  SI_COMPILE_MODE=max-autotune-no-cudagraphs if you want the kernel tuning.)
#
# NOT COVERED HERE — these need a one-line code change before they can be benched
# (out of scope for a test script; left as TODOs so the sweep stays honest):
#   • persistent_workers / prefetch_factor  → data/amip_new.py:182 DataLoader(...)
#                                              (bias.py:186 already sets both)
#   • fuse assemble_input / assemble_forcing → common/utils.py (the ~370 torch.cat/step)
#   • log sum(p.numel()) + NCCL byte volume  → bench.py (grounds the NCCL-floor analysis)
#   • bf16_compress_hook numerical check     → a 1k-step LOSS A/B, not a throughput run
#
# READING RESULTS
#   Compare the per-tag CSVs; the columns that matter:
#     step_med, samples_per_s, samples_per_s_wall, data_idle_frac, peak_mem_gb_max_rank
#   e.g.:  column -t -s, bench_optim_*_results.csv | less -S
#   GPU utilisation: each run also drops gpu_util_<tag>_<jobid>.csv (nvidia-smi,
#   1 Hz, all GPUs). Coarse ("a kernel ran"); cross-check against data_idle_frac.
# =============================================================================

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# ----------------------------------------------------------------------------
# DRIVER MODE — run on a login node, submits the sweep.
# ----------------------------------------------------------------------------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    cd "$(dirname "${SELF}")"
    echo "=== Submitting optimization sweep from $(pwd) ==="

    submit() {
        local tag="$1"; local extra="${2:-}"
        local exports="RUN_TAG=${tag}"
        [[ -n "${extra}" ]] && exports="${exports},${extra}"
        local jid
        jid=$(sbatch --parsable --job-name="${tag}" --export="ALL,${exports}" "${SELF}")
        printf "  [%-16s] job %-10s  %s\n" "${tag}" "${jid}" "${extra:-(defaults)}"
        SUBMITTED+=("${jid}")
    }

    SUBMITTED=()
    submit baseline
    submit preopt_ddp       "SI_DDP_BUCKET_CAP_MB=25,SI_DDP_BF16_COMPRESS=0"
    submit compile          "SI_TORCH_COMPILE=1,SI_BENCH_WARMUP=40"
    submit bucket400        "SI_DDP_BUCKET_CAP_MB=400"
    submit batch6           "SI_BENCH_BS=6"
    submit compile_batch6   "SI_TORCH_COMPILE=1,SI_BENCH_WARMUP=40,SI_BENCH_BS=6"

    echo ""
    echo "  Submitted ${#SUBMITTED[@]} jobs (any free H100 node; each waits for 4 free GPUs)."
    echo "  Monitor:    squeue -u \${USER}"
    echo "  Cancel all: scancel ${SUBMITTED[*]}"
    echo "  Results:    bench_optim_<tag>_results.csv  (one row each)"
    exit 0
fi

# ----------------------------------------------------------------------------
# JOB MODE — one config, run by SLURM. Knobs arrive via sbatch --export and are
# read straight from the environment by bench.py; we only set the per-run CSV,
# the launch environment, and an optional batch-size override.
# ----------------------------------------------------------------------------
RUN_TAG="${RUN_TAG:-adhoc}"

module load python/miniforge-25.3.0
eval "$(conda shell.bash hook)"
conda activate /project/pedramh/shared/anthonyz/venv
module unload cuda
module load cuda/12.6

# Production keeps these set; the bench must run without their overhead.
unset NCCL_DEBUG
unset TORCH_DISTRIBUTED_DEBUG

export WANDB_MODE=offline
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Defaults that the driver may override via --export (respect inherited values).
export SI_BENCH_WARMUP="${SI_BENCH_WARMUP:-20}"
if [[ "${SI_NVTX:-0}" == "1" ]]; then
    export SI_BENCH_STEPS="${SI_BENCH_STEPS:-20}"   # smaller trace when profiling
else
    export SI_BENCH_STEPS="${SI_BENCH_STEPS:-80}"
fi
export SI_PRECISION="${SI_PRECISION:-bf16-mixed}"

# Safety net: torch.compile needs a longer warm-up to absorb the JIT compile of
# the first measured step, or the first 1–2 measurements are inflated.
if [[ "${SI_TORCH_COMPILE:-0}" == "1" && "${SI_BENCH_WARMUP}" -lt 40 ]]; then
    export SI_BENCH_WARMUP=40
fi

# One CSV per config so concurrent/serial runs never clobber each other.
export SI_BENCH_CSV="${SLURM_SUBMIT_DIR}/bench_optim_${RUN_TAG}_results.csv"

# Optional batch-size override (bench.py exposes --batch_size).
BS_ARG=""
[[ -n "${SI_BENCH_BS:-}" ]] && BS_ARG="--batch_size ${SI_BENCH_BS}"

config_file=configs/bench_midway.yaml

echo "=== si_optim_sweep [${RUN_TAG}]: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODELIST=${SLURM_NODELIST}"
nvidia-smi -L
echo "csv=${SI_BENCH_CSV}"
echo "warmup=${SI_BENCH_WARMUP} steps=${SI_BENCH_STEPS} precision=${SI_PRECISION}"
echo "DDP: bucket_cap_mb=${SI_DDP_BUCKET_CAP_MB:-200(default)} bf16_compress=${SI_DDP_BF16_COMPRESS:-1(default)} bucket_view=${SI_DDP_BUCKET_VIEW:-1(default)}"
echo "compile=${SI_TORCH_COMPILE:-0(default)} mode=${SI_COMPILE_MODE:-default} batch_size=${SI_BENCH_BS:-yaml(4)}"

cd "${SLURM_SUBMIT_DIR}"

if [[ "${SI_NVTX:-0}" == "1" ]]; then
    # Profiling run: per-rank .nsys-rep, capturing only the measured steps
    # (bench.py brackets them with cudaProfilerStart/Stop). Same flags as
    # bench_nvtx_compile.sh. Convert after with: nsys export --type=sqlite <file>
    which nsys; nsys --version
    srun --export=ALL nsys profile \
        --trace=cuda,nvtx,osrt \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop-shutdown \
        --cuda-memory-usage=true \
        --force-overwrite=true \
        --output="si_optim_${RUN_TAG}_${SLURM_JOB_ID}_rank%q{SLURM_PROCID}" \
        python bench.py --config "${config_file}" ${BS_ARG}
else
    # Background GPU-utilisation sampler (all GPUs, 1 Hz) → one CSV per run.
    # nvidia-smi GPU-Util is coarse ("a kernel ran this interval"), so for this
    # compute-bound job it reads high; use it to watch dataloader-idle dips and
    # cross-check data_idle_frac in the bench CSV.
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used \
        --format=csv,nounits -l 1 > "gpu_util_${RUN_TAG}_${SLURM_JOB_ID}.csv" &
    SMI_PID=$!
    srun --export=ALL python bench.py --config "${config_file}" ${BS_ARG}
    kill "${SMI_PID}" 2>/dev/null
fi
