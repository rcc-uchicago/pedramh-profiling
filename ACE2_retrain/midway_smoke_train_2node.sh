#!/bin/bash
#SBATCH --account=rcc-staff
#SBATCH --time=01:00:00
#SBATCH -p test
#SBATCH --qos=test
#SBATCH --constraint=H200
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o ace2_smoke_2node_%x_%j.out
#SBATCH -e ace2_smoke_2node_%x_%j.err
#
# ACE2 8-GPU (2 node x 4) bring-up smoke on H200. Sibling of midway_smoke_train.sh,
# which is single-node; that script is left untouched.
#
# PASS = `ACE2_SMOKE_2NODE_OK` in the .out. Key on the token, not the exit code.
#
# WHY THIS IS A SEPARATE SCRIPT, NOT A FLAG
# The launcher shape genuinely differs. Single-node uses
# `torchrun --standalone`, which CANNOT span nodes -- it binds rendezvous to
# localhost. Multi-node needs one launcher per node sharing a c10d rendezvous,
# which is the shape the Delta train.sh already proved for this codebase:
#     srun python -m torch.distributed.run --nnodes N --rdzv_backend c10d ...
# with --ntasks-per-node=1 so srun starts ONE launcher per node and
# torch.distributed.run forks the 4 local ranks itself. Setting
# --ntasks-per-node=4 here would start 4 launchers per node = 16 ranks.
#
# WHY 2 NODES: this cluster has NO node with more than 4 GPUs (checked across
# every partition), so 8 GPUs necessarily means 2 nodes x 4. That is a real
# difference from the AI2 reference run this is meant to compare against: if
# theirs is an 8-GPU HGX box, all 8 of its GPUs share NVLink/NVSwitch, whereas
# here the two groups of 4 are joined by an InfiniBand hop. Since the
# single-node profile already found NCCL to be the largest bucket (40.6% of GPU
# kernel time), that topology gap is exactly where our number will diverge from
# theirs -- report it alongside any comparison rather than as a footnote.
#
# NODE HOMOGENEITY: bare --constraint=H200 admits both H200 flavours in `test`
# (epyc-9335/64-core/768G and gold-6542Y/48-core/1T) and will currently land on
# one of each. For a functional smoke that is fine. For a TIMING number you
# intend to compare against AI2, use `--constraint="H200&gold-6542Y"` -- the
# profile showed NCCL ring kernels spin while waiting for peers (largest
# AllReduce instance 4.16 s vs an 11.2 ms median), so a slower partner node is
# recorded as communication cost that does not exist. Measured with
# `sbatch --test-only` on 2026-08-18: bare H200 starts ~18 h out, homogeneous
# gold-6542Y ~30 h. The script prints each node's GPU model and core count --
# check that line before trusting a timing.
#
# BATCH SIZE: fme requires batch_size % world_size == 0. World size here is 8,
# so the default is the PRODUCTION batch_size of 16 (2/rank) -- the first time
# it is exercised at all; the A100 runs had to drop to 4 for 40 GB. Override
# with ACE2_BATCH_SIZE=8 for 1/rank, which is the like-for-like weak-scaling
# comparison against the 4-GPU A100 runs (also 1/rank).

set -eo pipefail
# No `set -u`: the gxx_linux-64 activate.d hook dereferences CONDA_BUILD_SYSROOT.

ACE2_DIR=/project/rcc/mehta5/pedramh-profiling/ACE2_retrain
CONFIG="${ACE2_DIR}/config_midway.yaml"
FME_ENV=/project/rcc/mehta5/envs/fme
BATCH_SIZE="${ACE2_BATCH_SIZE:-16}"

# --- env bootstrap (identical to the single-node scripts) ------------------
module load python/miniforge-25.3.0

eval "$(conda shell.bash hook)"
conda activate "${FME_ENV}"

module unload cuda
module load cuda/12.6

python -c "import torch, fme" 2>/dev/null || {
    echo "ERROR ACE2_ENV_NOT_ACTIVE python=$(command -v python) env=${FME_ENV}"
    exit 1
}

# --- bench-neutral env (must match the single-node scripts) ----------------
unset NCCL_DEBUG
unset TORCH_DISTRIBUTED_DEBUG

export WANDB_MODE=offline
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Inter-node only. Midway3 has ib0 (InfiniBand) plus bond0/ens*; excluding
# loopback and docker is enough for NCCL to pick a real interface, and leaving
# IB enabled (the default) is the point of running on 2 nodes at all. Set
# ACE2_NCCL_DEBUG=INFO to diagnose a hang -- it is NOT set by default because
# the house bench env unsets NCCL_DEBUG for timing neutrality.
export NCCL_SOCKET_IFNAME=^lo,docker0
[ -n "${ACE2_NCCL_DEBUG}" ] && export NCCL_DEBUG="${ACE2_NCCL_DEBUG}"

# --- rendezvous ------------------------------------------------------------
nodes=( $(scontrol show hostnames "${SLURM_JOB_NODELIST}") )
head_node="${nodes[0]}"
head_ip=$(srun --nodes=1 --ntasks=1 -w "${head_node}" hostname -I | awk '{print $1}')
NUM_GPUS=$(nvidia-smi -L | wc -l)
WORLD_SIZE=$(( SLURM_NNODES * NUM_GPUS ))
EXP_DIR="${ACE2_DIR}/outs/midway_smoke2n_${SLURM_JOB_ID}"

echo "=== midway_smoke_train_2node.sh: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODES=${SLURM_NNODES} (${SLURM_JOB_NODELIST})"
echo "head=${head_node} (${head_ip})  gpus/node=${NUM_GPUS}  world=${WORLD_SIZE}  batch=${BATCH_SIZE}"
# --cpus-per-task must be repeated here: this srun overrides --ntasks, which
# starts a fresh step, and without it the step is bound to a single core and
# `nproc` reports a misleading 2 that has nothing to do with what training gets.
srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" \
     --cpus-per-task="${SLURM_CPUS_PER_TASK}" bash -c \
    'echo "  $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x$(nvidia-smi -L | wc -l), $(nproc) cores"'

if [ $(( BATCH_SIZE % WORLD_SIZE )) -ne 0 ]; then
    echo "ERROR ACE2_BATCH_NOT_DIVISIBLE batch=${BATCH_SIZE} world=${WORLD_SIZE} -- fme requires batch_size % world_size == 0"
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
    train_loader.sample_with_replacement=128
    validation_loader.batch_size="${BATCH_SIZE}"
    validation_loader.dataset.subset.stop_time=1996-01-05
    train_evaluation_samples=16
    log_train_every_n_batches=1
)

python -m fme.ace.validate_config "${CONFIG}" --config_type train \
    --override "${OVERRIDES[@]}" \
    || { echo "ERROR ACE2_CONFIG_INVALID ${CONFIG}"; exit 1; }

srun python -m torch.distributed.run \
    --nnodes "${SLURM_NNODES}" \
    --nproc_per_node "${NUM_GPUS}" \
    --rdzv_id "${SLURM_JOB_ID}" \
    --rdzv_backend c10d \
    --rdzv_endpoint "${head_ip}:29500" \
    -m fme.ace.train "${CONFIG}" --override "${OVERRIDES[@]}"

# --- PASS gate: read the run's own log, not the exit code -----------------
LOG="${EXP_DIR}/out.log"
[ -f "${LOG}" ] || { echo "ERROR ACE2_SMOKE_2NODE_NO_LOG expected ${LOG}"; exit 1; }

train_loss=$(awk -F'Train loss: ' '/Train loss: /{v=$2} END{print v}' "${LOG}")
valid_loss=$(awk -F'Valid loss: ' '/Valid loss: /{v=$2} END{print v}' "${LOG}")

for v in "${train_loss}" "${valid_loss}"; do
    case "${v}" in
        ""|*[nN][aA][nN]*|*[iI][nN][fF]*)
            echo "ERROR ACE2_SMOKE_2NODE_LOSS_NOT_FINITE train=${train_loss:-<none>} valid=${valid_loss:-<none>}"
            exit 1
            ;;
    esac
done

echo "ACE2_SMOKE_2NODE_OK train_loss=${train_loss} valid_loss=${valid_loss} world=${WORLD_SIZE} batch=${BATCH_SIZE}"
echo "experiment_dir=${EXP_DIR}"
