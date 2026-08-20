#!/bin/bash
#SBATCH --account=pi-pedramh
#SBATCH --time=01:00:00
#SBATCH -p pedramh-gpu
#SBATCH --qos=pedramh-gpu
#SBATCH --constraint=H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o ace2_compile_%x_%j.out
#SBATCH -e ace2_compile_%x_%j.err
#
# Where should torch.compile be applied to cut ACE2's elementwise work?
#
# WHAT THE PROFILE SAYS THE CEILING IS. On 8x H200 elementwise/copy is 47.9% of
# GPU kernel time, but 58% of that bucket is `direct_copy`/`bfloat16_copy` --
# data movement from fme's dict<->tensor round trip (stacker.py:121) and AMP
# casts, which fusion does NOT remove. The reachable prize is the other ~20% of
# GPU time (add / unary / fill). Do not expect 47.9%; expect at most ~20%, minus
# whatever compile costs in launch overhead and graph breaks.
#
# ARMS (ACE2_COMPILE_TARGET, see ace2_compile_probe.py):
#   none normalizer network corrector all
#
# MEASUREMENT: median inter-step wall time over the LAST 30 steps, so
# torch.compile's warmup (which can be tens of seconds on the first steps) is
# excluded rather than averaged in. fme's own
# `training_samples_per_second_on_rank_0` is a CUMULATIVE average and is
# therefore useless for this -- it would bury the speedup under warmup.
#
# ALSO REPORTED: final train_loss per arm. Compare drift against the two
# measured floors in bench_midway_notes.md -- 2.5e-7 same-hardware, ~1e-5
# across GPU architectures. An arm that moves the loss well above 2.5e-7 on the
# same node is changing numerics, which is a DESIGN 4 matter and jesswan's call,
# not something to adopt on a speedup alone.
#
# Validation is shortened AND log_snapshots is off here (see the validation
# probe result: snapshots are ~52% of validation and their output is discarded
# when wandb is off). This probe measures the TRAINING step; validation is noise
# for it.
#
# PASS = `ACE2_COMPILE_PROBE_OK` plus the numbers on stdout.

set -eo pipefail
# No `set -u`: the gxx_linux-64 activate.d hook dereferences CONDA_BUILD_SYSROOT.

ACE2_DIR=/project/rcc/mehta5/pedramh-profiling/ACE2_retrain
CONFIG="${ACE2_DIR}/config_midway.yaml"
# ACE2_FME_ENV lets an arm run against a different env (e.g. the torch 2.8
# build) without editing this script, so the torch 2.7.1 baseline stays
# reproducible from the same file.
FME_ENV="${ACE2_FME_ENV:-/project/rcc/mehta5/envs/fme}"
TARGET="${ACE2_COMPILE_TARGET:-none}"
CMODE="${ACE2_COMPILE_MODE:-default}"

module load python/miniforge-25.3.0

eval "$(conda shell.bash hook)"
conda activate "${FME_ENV}"

module unload cuda
module load cuda/12.6

python -c "import torch, fme" 2>/dev/null || {
    echo "ERROR ACE2_ENV_NOT_ACTIVE python=$(command -v python) env=${FME_ENV}"
    exit 1
}

unset NCCL_DEBUG
unset TORCH_DISTRIBUTED_DEBUG
# sbatch --export=ALL carries the submitting shell's environment in, so an
# inherited NCCL tuning var silently changes what is being measured -- or kills
# the run. Job 53546805 died on `NCCL_ALGO=Tree` ("no algorithm/protocol
# available for AllGather with datatype ncclInt8"), and the 27% step-time
# outlier in 53546804 is consistent with the same pollution on a run where Tree
# merely worked badly rather than failing. A timing harness must define its own
# collective environment, not inherit one.
unset NCCL_ALGO NCCL_PROTO NCCL_NTHREADS NCCL_MAX_NCHANNELS NCCL_MIN_NCHANNELS
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_IB_DISABLE NCCL_NET_GDR_LEVEL
export WANDB_MODE=offline
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export ACE2_COMPILE_TARGET="${TARGET}"
export ACE2_COMPILE_MODE="${CMODE}"

NUM_GPUS=$(nvidia-smi -L | wc -l)
EXP_DIR="${ACE2_DIR}/outs/midway_compile_${TARGET}_${SLURM_JOB_ID}"

echo "=== midway_compile_probe.sh: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODE=${SLURM_NODELIST}  gpus=${NUM_GPUS}  target=${TARGET}  mode=${CMODE}"
echo "REP=${ACE2_REP:-1}  surviving NCCL_* env: $(env | grep -c '^NCCL_') var(s): $(env | grep '^NCCL_' | tr '\n' ' ')"
python -c "import torch, torch_harmonics; print(f'ENV torch={torch.__version__} torch_harmonics={torch_harmonics.__version__} env=${FME_ENV}')"

OVERRIDES=(
    experiment_dir="${EXP_DIR}"
    max_epochs=1
    segment_epochs=1
    save_checkpoint=false
    checkpoint_save_epochs=null
    inference=null
    train_loader.batch_size=4
    train_loader.sample_with_replacement=256
    validation_loader.batch_size=4
    validation_loader.dataset.subset.stop_time=1996-01-02
    validation_aggregator.log_snapshots=false
    train_evaluation_samples=8
    log_train_every_n_batches=1
)

python -m fme.ace.validate_config "${CONFIG}" --config_type train \
    --override "${OVERRIDES[@]}" \
    || { echo "ERROR ACE2_CONFIG_INVALID ${CONFIG}"; exit 1; }

torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
    "${ACE2_DIR}/ace2_compile_probe.py" "${CONFIG}" --override "${OVERRIDES[@]}"

LOG="${EXP_DIR}/out.log"
[ -f "${LOG}" ] || { echo "ERROR ACE2_COMPILE_NO_LOG expected ${LOG}"; exit 1; }

python3 - "${LOG}" "${TARGET}" "${CMODE}" <<'PYEOF'
import datetime
import re
import statistics
import sys

log, target, mode = sys.argv[1], sys.argv[2], sys.argv[3]
ts = lambda l: datetime.datetime.strptime(l[:23], "%Y-%m-%d %H:%M:%S,%f")

steps, loss = [], None
for line in open(log):
    if re.search(r"Step \d+: \{", line):
        steps.append(ts(line))
    elif "Train loss: " in line:
        loss = line.split("Train loss: ")[1].strip()

if len(steps) < 35:
    print(f"ERROR ACE2_COMPILE_TOO_FEW_STEPS got {len(steps)}")
    sys.exit(1)

deltas = [(b - a).total_seconds() for a, b in zip(steps, steps[1:])]
tail = deltas[-30:]                      # steady state, past compile warmup
med = statistics.median(tail)
warm = sum(deltas[: len(deltas) - 30])   # everything before the measured tail
print(f"COMPILE target={target} mode={mode} n_steps={len(steps)}")
print(f"COMPILE step_med_s={med:.4f} samples_per_s_per_rank={4/ (med*4):.3f}")
print(f"COMPILE warmup_s={warm:.1f} (steps before the measured tail)")
print(f"COMPILE train_loss={loss}")
print("ACE2_COMPILE_PROBE_OK")
PYEOF
