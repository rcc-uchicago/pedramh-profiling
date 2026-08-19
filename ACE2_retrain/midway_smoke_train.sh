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
#SBATCH -o ace2_smoke_train_%x_%j.out
#SBATCH -e ace2_smoke_train_%x_%j.err
#
# ACE2 (ai2cm `fme`) 4-GPU bring-up smoke on Midway. Sibling of the Delta/NCSA
# train.sh, which is NOT runnable here (bdiu-dtai-gh account, ghx4/GH200
# partition, Slingshot hsn0 NCCL vars, an unreadable env). Per rule #7 that
# script is left untouched; this is the Midway sibling beside it.
#
# PROVES: the vendored ace_exp imports, the in-repo ERA5 copy loads, the SFNO
# stepper trains on 4 GPUs, and the loss is finite. It does NOT prove
# performance -- see midway_bench_nsys.sh.
#
# PASS = the line `ACE2_SMOKE_OK` in the .out (printed only after a finite train
# AND valid loss are read back out of the run's own out.log). Key on that token,
# never on the exit code: fme can exit 0 after a rank dies mid-epoch.
#
# HELD IDENTICAL to config_midway.yaml (and so to the Delta config): model,
# loss weights, optimizer, AMP, variable lists, normalization, seed,
# stepper_training.n_forward_steps.
# DELIBERATELY SHORTENED below via --override, so every deviation is visible
# here rather than hidden in a forked config:
#   batch_size 16 -> 4    kept at 4 for continuity with the 4x A100 baseline
#                         (1 sample/rank), NOT because 16 does not fit -- job
#                         53483666 proved 16 fits on H200 at world size 8.
#                         Changing it invalidates comparison with every earlier
#                         number, so change it deliberately, not casually.
#   epoch -> 64 samples   sample_with_replacement; a real epoch is ~10^5 samples
#   validation -> 4 days  the config validates over 1996-1997 (~2900 samples)
#   inference -> null     the config's inline inference is 7300 forward steps
#   checkpoints -> off    nothing here is worth resuming
#
# Launcher shape: torchrun-family (`python -m fme.ace.train` under
# torch.distributed), so --ntasks-per-node=1 + torchrun --nproc_per_node=4,
# the S2S shape -- NOT the srun-per-rank shape the Lightning models need.
#
# PARTITION: pedramh-gpu, per the 2026-08-19 restriction to that partition only.
# It is a ONE node partition (midway3-0423, 4x H100) that we share with nobody,
# so no --constraint is strictly needed; H100 is kept as documentation of what
# the numbers were taken on. NOTE this is H100, while the profile baseline in
# bench_midway_notes.md was taken on A100-PCIE in `test` -- step times are NOT
# comparable across that boundary, only shapes are.

set -eo pipefail
# Deliberately no `set -u`: the env's gxx_linux-64 activate.d hook dereferences
# an unset CONDA_BUILD_SYSROOT and would abort activation (same trap the Delta
# train.sh documents).

ACE2_DIR=/project/rcc/mehta5/pedramh-profiling/ACE2_retrain
CONFIG="${ACE2_DIR}/config_midway.yaml"
FME_ENV=/project/rcc/mehta5/envs/fme

# --- env bootstrap (conda, SI-style; do NOT `module purge` on Midway3 -- it
# --- strips SOFTPATH and breaks mamba/cuda) --------------------------------
module load python/miniforge-25.3.0

eval "$(conda shell.bash hook)"
conda activate "${FME_ENV}"

module unload cuda
module load cuda/12.6

python -c "import torch, fme" 2>/dev/null || {
    echo "ERROR ACE2_ENV_NOT_ACTIVE python=$(command -v python) env=${FME_ENV}"
    exit 1
}

# --- bench-neutral env (must match midway_bench_nsys.sh) -------------------
# Production training keeps these set; a timing run must not pay their overhead.
unset NCCL_DEBUG
unset TORCH_DISTRIBUTED_DEBUG

export WANDB_MODE=offline
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 32 cores / 4 ranks = 8 per rank, all 8 spent on loader workers -> 2 OMP
# threads each is the house setting; more oversubscribes the loaders.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

NUM_GPUS=$(nvidia-smi -L | wc -l)
EXP_DIR="${ACE2_DIR}/outs/midway_smoke_${SLURM_JOB_ID}"

echo "=== midway_smoke_train.sh: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODELIST=${SLURM_NODELIST}  NUM_GPUS=${NUM_GPUS}"
nvidia-smi -L

OVERRIDES=(
    experiment_dir="${EXP_DIR}"
    max_epochs=1
    segment_epochs=1
    save_checkpoint=false
    checkpoint_save_epochs=null
    inference=null
    train_loader.batch_size=4
    train_loader.sample_with_replacement=64
    validation_loader.batch_size=4
    validation_loader.dataset.subset.stop_time=1996-01-05
    train_evaluation_samples=16
    log_train_every_n_batches=1
)

# Pre-flight: pure dacite parsing, no data opened. fme parses strict, so one
# stale key anywhere aborts -- catch that in seconds rather than after four
# GPUs are allocated and a 2.39 TB dataset is opened.
python -m fme.ace.validate_config "${CONFIG}" --config_type train \
    --override "${OVERRIDES[@]}" \
    || { echo "ERROR ACE2_CONFIG_INVALID ${CONFIG}"; exit 1; }

torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
    -m fme.ace.train "${CONFIG}" --override "${OVERRIDES[@]}"

# --- PASS gate: read the run's own log, not the exit code -----------------
LOG="${EXP_DIR}/out.log"
[ -f "${LOG}" ] || { echo "ERROR ACE2_SMOKE_NO_LOG expected ${LOG}"; exit 1; }

train_loss=$(awk -F'Train loss: ' '/Train loss: /{v=$2} END{print v}' "${LOG}")
valid_loss=$(awk -F'Valid loss: ' '/Valid loss: /{v=$2} END{print v}' "${LOG}")

for v in "${train_loss}" "${valid_loss}"; do
    case "${v}" in
        ""|*[nN][aA][nN]*|*[iI][nN][fF]*)
            echo "ERROR ACE2_SMOKE_LOSS_NOT_FINITE train=${train_loss:-<none>} valid=${valid_loss:-<none>}"
            exit 1
            ;;
    esac
done

echo "ACE2_SMOKE_OK train_loss=${train_loss} valid_loss=${valid_loss} n_gpus=${NUM_GPUS}"
echo "experiment_dir=${EXP_DIR}"
