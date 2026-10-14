#!/bin/bash
#SBATCH --account=pi-pedramh
#SBATCH --time=01:30:00
#SBATCH -p pedramh-gpu
#SBATCH --qos=pedramh-gpu
#SBATCH --constraint=H100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o ace2_bench_nsys_2node_%x_%j.out
#SBATCH -e ace2_bench_nsys_2node_%x_%j.err
#
# ACE2 8-GPU (2 node x 4) nsys capture on H200. This is the run that answers what
# the single-node A100 profile could not: whether the 40.6% NCCL share is
# PCIe-bound gradient all-reduce, and what it becomes when the all-reduce has
# to cross a node boundary as well.
#
# PASS = `ACE2_NSYS_2NODE_OK` in the .out.
#
# ONE REPORT PER NODE. nsys traces a process tree, and here each node runs its
# own launcher, so nsys wraps the launcher on each node and emits
# nsys_ace2_2node_<jobid>_node<N>.nsys-rep. Analyse them together -- a
# single-node report cannot show inter-node imbalance, which is the whole point.
#
# HOMOGENEOUS NODES ARE REQUIRED HERE, hence --constraint=H200&gold-6542Y
# (5 nodes: 48-core gold-6542Y/1T) rather than the bare H200 the smoke uses,
# which also admits 64-core epyc-9335 boxes and currently lands on one of each.
# The single-node profile showed NCCL ring kernels spin while waiting for peers
# -- largest AllReduce instance 4.16 s against an 11.2 ms median -- so a slower
# partner node is recorded as communication cost that does not exist.
#
# 8 GPUs here is 2 nodes x 4: no node on this cluster has more than 4 GPUs. If
# the AI2 reference run is an 8-GPU HGX box, its 8 GPUs share NVLink/NVSwitch
# while ours cross InfiniBand between the two groups of 4. NCCL was already the
# largest bucket single-node, so that is where the two will diverge -- this
# capture is the measurement of exactly that gap.
#
# Same no-instrumentation caveat as the single-node script: fme has no
# cudaProfilerApi/NVTX in the SFNO path, so this is a time-windowed capture, not
# a --capture-range=cudaProfilerApi one, and parse_nsys.py yields nothing.
#   ACE2_NSYS_DELAY     seconds to skip before collecting (default 45)
#   ACE2_NSYS_DURATION  seconds to collect (default 110)
# Those defaults are measured from the single-node H200-less runs (startup ~50 s
# warm, ~80 s cold). CHECK THEM against the smoke's timeline on H200 before
# trusting a capture -- faster GPUs move the training window earlier.
#
# HELD IDENTICAL to midway_smoke_train_2node.sh: env, model, loss, optimizer,
# AMP, normalization, batch size. Only the epoch length and nsys differ.

# ############################################################################
# PARKED as of 2026-08-19: runs are restricted to pedramh-gpu, which has
# exactly ONE node (midway3-0423). --nodes=2 CANNOT be satisfied there, so
# sbatch rejects this script outright rather than queueing. That rejection is
# deliberate -- it is better than silently taking nodes from another partition.
#
# The 8-GPU results already captured (jobs 53483666/667/668 on 2x4 H200) stand;
# see bench_midway_notes.md. To run this again you must consciously go back to
# a multi-node partition, e.g.
#     sbatch --account=rcc-staff -p test --qos=test --constraint=H200 <this>
# ############################################################################

set -eo pipefail
# No `set -u`: the gxx_linux-64 activate.d hook dereferences CONDA_BUILD_SYSROOT.

ACE2_DIR=/project/rcc/mehta5/pedramh-profiling/ACE2_retrain
CONFIG="${ACE2_DIR}/config_midway.yaml"
FME_ENV=/project/rcc/mehta5/envs/fme
BATCH_SIZE="${ACE2_BATCH_SIZE:-16}"

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

unset NCCL_DEBUG
unset TORCH_DISTRIBUTED_DEBUG

export WANDB_MODE=offline
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NCCL_SOCKET_IFNAME=^lo,docker0
[ -n "${ACE2_NCCL_DEBUG}" ] && export NCCL_DEBUG="${ACE2_NCCL_DEBUG}"

nodes=( $(scontrol show hostnames "${SLURM_JOB_NODELIST}") )
head_node="${nodes[0]}"
head_ip=$(srun --nodes=1 --ntasks=1 -w "${head_node}" hostname -I | awk '{print $1}')
NUM_GPUS=$(nvidia-smi -L | wc -l)
WORLD_SIZE=$(( SLURM_NNODES * NUM_GPUS ))
EXP_DIR="${ACE2_DIR}/outs/midway_nsys2n_${SLURM_JOB_ID}"
NSYS_OUT="${EXP_DIR}/nsys_ace2_2node_${SLURM_JOB_ID}"
mkdir -p "${EXP_DIR}"

NSYS_DELAY="${ACE2_NSYS_DELAY:-45}"
NSYS_DURATION="${ACE2_NSYS_DURATION:-110}"
WINDOW_ARGS=()
[ "${NSYS_DELAY}" != "0" ]    && WINDOW_ARGS+=("--delay=${NSYS_DELAY}")
[ "${NSYS_DURATION}" != "0" ] && WINDOW_ARGS+=("--duration=${NSYS_DURATION}")

echo "=== midway_bench_nsys_2node.sh: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODES=${SLURM_NNODES} (${SLURM_JOB_NODELIST})"
echo "head=${head_node} (${head_ip})  gpus/node=${NUM_GPUS}  world=${WORLD_SIZE}  batch=${BATCH_SIZE}"
echo "capture: delay=${NSYS_DELAY}s duration=${NSYS_DURATION}s (0 = unbounded)"
# --cpus-per-task must be repeated here: this srun overrides --ntasks, which
# starts a fresh step, and without it the step is bound to a single core and
# `nproc` reports a misleading 2 that has nothing to do with what training gets.
srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" \
     --cpus-per-task="${SLURM_CPUS_PER_TASK}" bash -c \
    'echo "  $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x$(nvidia-smi -L | wc -l), $(nproc) cores"'

if [ $(( BATCH_SIZE % WORLD_SIZE )) -ne 0 ]; then
    echo "ERROR ACE2_BATCH_NOT_DIVISIBLE batch=${BATCH_SIZE} world=${WORLD_SIZE}"
    exit 1
fi

OVERRIDES=(
    experiment_dir="${EXP_DIR}"
    max_epochs=1
    segment_epochs=1
    save_checkpoint=false
    checkpoint_save_epochs=null
    inference=null
    train_loader.batch_size="${BATCH_SIZE}"
    train_loader.sample_with_replacement=1024
    validation_loader.batch_size="${BATCH_SIZE}"
    validation_loader.dataset.subset.stop_time=1996-01-05
    train_evaluation_samples=16
    log_train_every_n_batches=10
)

python -m fme.ace.validate_config "${CONFIG}" --config_type train \
    --override "${OVERRIDES[@]}" \
    || { echo "ERROR ACE2_CONFIG_INVALID ${CONFIG}"; exit 1; }

srun nsys profile \
    --trace=cuda,nvtx,cudnn,cublas \
    --cuda-memory-usage=true \
    --output="${NSYS_OUT}_node%q{SLURM_NODEID}" \
    --force-overwrite=true \
    "${WINDOW_ARGS[@]}" \
    python -m torch.distributed.run \
        --nnodes "${SLURM_NNODES}" \
        --nproc_per_node "${NUM_GPUS}" \
        --rdzv_id "${SLURM_JOB_ID}" \
        --rdzv_backend c10d \
        --rdzv_endpoint "${head_ip}:29500" \
        -m fme.ace.train "${CONFIG}" --override "${OVERRIDES[@]}"

# --- PASS gate ------------------------------------------------------------
n_reports=$(ls "${NSYS_OUT}"_node*.nsys-rep 2>/dev/null | wc -l)
[ "${n_reports}" -eq "${SLURM_NNODES}" ] || {
    echo "ERROR ACE2_NSYS_2NODE_REPORT_COUNT got ${n_reports}, expected ${SLURM_NNODES} (one per node)"
    exit 1
}
total_mb=0
for r in "${NSYS_OUT}"_node*.nsys-rep; do
    b=$(stat -c %s "$r")
    [ "${b}" -gt 1048576 ] || { echo "ERROR ACE2_NSYS_2NODE_REPORT_TOO_SMALL $r = ${b} bytes -- window likely missed the run"; exit 1; }
    total_mb=$(( total_mb + b / 1048576 ))
done

LOG="${EXP_DIR}/out.log"
train_loss=$(awk -F'Train loss: ' '/Train loss: /{v=$2} END{print v}' "${LOG}" 2>/dev/null)
case "${train_loss}" in
    ""|*[nN][aA][nN]*|*[iI][nN][fF]*)
        echo "ERROR ACE2_NSYS_2NODE_LOSS_NOT_FINITE train=${train_loss:-<none>} (reports kept in ${EXP_DIR})"
        exit 1
        ;;
esac

echo "ACE2_NSYS_2NODE_OK train_loss=${train_loss} reports=${n_reports} total_mb=${total_mb} world=${WORLD_SIZE} batch=${BATCH_SIZE}"
echo "reports=${EXP_DIR}"
