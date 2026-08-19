#!/bin/bash
#SBATCH --account=rcc-staff
#SBATCH --time=01:00:00
#SBATCH -p test
#SBATCH --qos=test
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o ace2_valprobe_%x_%j.out
#SBATCH -e ace2_valprobe_%x_%j.err
#
# Is ACE2's validation phase I/O-bound or aggregator-bound?
#
# WHY THIS EXISTS: the nsys captures showed validation runs at 3.3% GPU
# occupancy while consuming ~40% of a warm epoch's wall-clock (~80% of a cold
# one). That says "not GPU work" -- it does NOT say whether the CPU time is
# reading netCDF or running aggregator math. fme cannot answer it directly:
# the "data_loading" GlobalTimer category is only instrumented in the INFERENCE
# path, and log_durations() is never called from the training entrypoint.
#
# THE DISCRIMINATOR: hold the validation window fixed and vary only
# validation_loader.batch_size. Bytes read are identical; the number of batches
# changes 4x.
#   validation time scales with batch COUNT  -> per-batch overhead dominates
#                                               (aggregator, python, launches)
#   validation time stays FLAT               -> per-sample cost dominates
#                                               (I/O; GPU is only 3.3% busy)
#
# CONTROLLING FOR PAGE CACHE: the cold/warm split is the whole reason this is
# ambiguous (validation was 183 s cold vs ~45 s warm on identical work). So each
# run does TWO epochs and the comparison across arms uses EPOCH 2, which is warm
# in both. Epoch 1 is reported too -- it is the cold number, and the ratio
# between them is itself the I/O signal.
#
# RUN BOTH ARMS:
#   ACE2_VAL_BATCH=4  sbatch midway_validation_probe.sh
#   ACE2_VAL_BATCH=16 sbatch midway_validation_probe.sh
# Validation window is 64 samples, so the arms give 16 vs 4 batches -- a clean
# 4x with the same bytes. Training is deliberately trivial (4 steps/epoch): this
# probe measures validation, not throughput.
#
# SECOND DISCRIMINATOR (ACE2_VAL_WORKERS): the batch-size arm above separates
# per-BATCH overhead from per-SAMPLE cost, but loader I/O and per-sample
# aggregator math are BOTH per-sample, so it cannot tell those apart. Varying
# validation_loader.num_data_workers does:
#   validation time scales with workers -> loader-bound (dataloading issue)
#   validation time flat                -> aggregator-bound (main process)
#
# PASS = `ACE2_VALPROBE_OK` plus the per-epoch numbers on stdout.

set -eo pipefail
# No `set -u`: the gxx_linux-64 activate.d hook dereferences CONDA_BUILD_SYSROOT.

ACE2_DIR=/project/rcc/mehta5/pedramh-profiling/ACE2_retrain
CONFIG="${ACE2_DIR}/config_midway.yaml"
FME_ENV=/project/rcc/mehta5/envs/fme
VAL_BATCH="${ACE2_VAL_BATCH:-4}"
VAL_WORKERS="${ACE2_VAL_WORKERS:-8}"
# Arbitrary extra dotlist overrides, space separated, e.g.
#   ACE2_VAL_EXTRA="validation_aggregator.log_snapshots=false"
VAL_EXTRA="${ACE2_VAL_EXTRA:-}"

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
export WANDB_MODE=offline
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

NUM_GPUS=$(nvidia-smi -L | wc -l)
EXP_DIR="${ACE2_DIR}/outs/midway_valprobe_b${VAL_BATCH}_${SLURM_JOB_ID}"

echo "=== midway_validation_probe.sh: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODE=${SLURM_NODELIST}  gpus=${NUM_GPUS}  val_batch=${VAL_BATCH}  val_workers=${VAL_WORKERS}  extra=${VAL_EXTRA:-none}"
nvidia-smi -L | head -1

OVERRIDES=(
    experiment_dir="${EXP_DIR}"
    max_epochs=2
    segment_epochs=2
    save_checkpoint=false
    checkpoint_save_epochs=null
    inference=null
    train_loader.batch_size=4
    train_loader.sample_with_replacement=16
    validation_loader.batch_size="${VAL_BATCH}"
    validation_loader.num_data_workers="${VAL_WORKERS}"
    validation_loader.dataset.subset.stop_time=1996-01-17
    train_evaluation_samples=16
    log_train_every_n_batches=100
)
[ -n "${VAL_EXTRA}" ] && OVERRIDES+=( ${VAL_EXTRA} )

python -m fme.ace.validate_config "${CONFIG}" --config_type train \
    --override "${OVERRIDES[@]}" \
    || { echo "ERROR ACE2_CONFIG_INVALID ${CONFIG}"; exit 1; }

torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
    -m fme.ace.train "${CONFIG}" --override "${OVERRIDES[@]}"

# --- measurement: validation wall time per epoch, from the run's own log -----
# "Starting loop over validation data" -> "Time taken for epoch" brackets
# validation only; the post-epoch train evaluation happens inside
# train_one_epoch, BEFORE that marker, so it is excluded automatically.
LOG="${EXP_DIR}/out.log"
[ -f "${LOG}" ] || { echo "ERROR ACE2_VALPROBE_NO_LOG expected ${LOG}"; exit 1; }

python3 - "${LOG}" "${VAL_BATCH}" "${VAL_WORKERS}" <<'PYEOF'
import sys, re, datetime
log, vb, wk = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ts = lambda l: datetime.datetime.strptime(l[:23], "%Y-%m-%d %H:%M:%S,%f")
starts, ends = [], []
for line in open(log):
    if "Starting loop over validation data" in line:
        starts.append(ts(line))
    elif "Time taken for epoch" in line:
        ends.append(ts(line))
if not starts or len(ends) < len(starts):
    print(f"ERROR ACE2_VALPROBE_NO_MARKERS starts={len(starts)} ends={len(ends)}")
    sys.exit(1)
durs = [(e - s).total_seconds() for s, e in zip(starts, ends)]
n_samples = 64
n_batches = -(-n_samples // (vb))
print(f"VALPROBE val_batch={vb} workers={wk} n_val_samples~{n_samples} n_batches~{n_batches}")
for i, d in enumerate(durs, 1):
    tag = "cold" if i == 1 else "warm"
    print(f"VALPROBE epoch={i} ({tag}) validation_s={d:.2f} per_batch_s={d/n_batches:.2f} per_sample_s={d/n_samples:.3f}")
if len(durs) >= 2:
    print(f"VALPROBE cold_warm_ratio={durs[0]/durs[1]:.2f}x")
print("ACE2_VALPROBE_OK")
PYEOF
