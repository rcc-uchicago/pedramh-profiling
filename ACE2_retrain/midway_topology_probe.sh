#!/bin/bash
#SBATCH --account=pi-pedramh
#SBATCH --time=00:30:00
#SBATCH -p pedramh-gpu
#SBATCH --qos=pedramh-gpu
#SBATCH --constraint=H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o ace2_topo_%x_%j.out
#SBATCH -e ace2_topo_%x_%j.err
#
# Does midway3-0423's split NVLink topology explain ACE2's 52% NCCL share?
#
# midway3-0423 is H100 NVL: GPU0<->GPU1 and GPU2<->GPU3 are NVLink (NV12), but
# ACROSS the pairs it is SYS -- PCIe plus a NUMA hop (GPU0/1 on NUMA 0, GPU2/3
# on NUMA 1). Documented in s2s/v2.0/bench_report.md footnote 6, reconfirmed by
# nvidia-smi topo -m in job 53537121.
#
# THE EXPERIMENT: run the identical 2-GPU job twice, changing ONLY which two
# GPUs are used.
#   ACE2_GPU_PAIR=0,1  -> both ranks on one NVLink pair
#   ACE2_GPU_PAIR=0,2  -> ranks on opposite pairs, all-reduce crosses SYS
# Same node, same data, same config, same per-rank batch. Any step-time
# difference is the interconnect and nothing else.
#
# WHY 2 GPUS AND batch_size=2: the 4-GPU baseline runs 1 sample/rank
# (batch_size 4 / 4 ranks). batch_size=2 on 2 ranks keeps that identical, so
# per-rank compute is unchanged between this probe and the baseline, and
# between the two arms. Only the gradient exchange path moves.
#
# This is the cheap alternative to testing on Polaris: ACE2's 2.39 TB dataset is
# not staged on eagle (and per polaris_pbs_notes.md even plain ERA5 is not), and
# a Polaris run would change GPU generation, CPU, filesystem and data all at
# once -- telling you "faster there" without telling you why.
#
# PASS = `ACE2_TOPO_OK` plus the numbers.

set -eo pipefail

ACE2_DIR=/project/rcc/mehta5/pedramh-profiling/ACE2_retrain
CONFIG="${ACE2_DIR}/config_midway.yaml"
FME_ENV=/project/rcc/mehta5/envs/fme
PAIR="${ACE2_GPU_PAIR:-0,1}"

module load python/miniforge-25.3.0
eval "$(conda shell.bash hook)"
conda activate "${FME_ENV}"
module unload cuda
module load cuda/12.6

python -c "import torch, fme" 2>/dev/null || {
    echo "ERROR ACE2_ENV_NOT_ACTIVE python=$(command -v python)"; exit 1; }

unset NCCL_DEBUG
unset TORCH_DISTRIBUTED_DEBUG
export WANDB_MODE=offline
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

export CUDA_VISIBLE_DEVICES="${PAIR}"
NUM_GPUS=2
EXP_DIR="${ACE2_DIR}/outs/midway_topo_$(echo "${PAIR}" | tr -d ',')_${SLURM_JOB_ID}"

echo "=== midway_topology_probe.sh: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODE=${SLURM_NODELIST}  pair=${PAIR}  ranks=${NUM_GPUS}"
echo "--- link between the two GPUs in use ---"
nvidia-smi topo -m 2>/dev/null | head -6

OVERRIDES=(
    experiment_dir="${EXP_DIR}"
    max_epochs=1
    segment_epochs=1
    save_checkpoint=false
    checkpoint_save_epochs=null
    inference=null
    train_loader.batch_size=2
    train_loader.sample_with_replacement=128
    validation_loader.batch_size=2
    validation_loader.dataset.subset.stop_time=1996-01-02
    validation_aggregator.log_snapshots=false
    train_evaluation_samples=4
    log_train_every_n_batches=1
)

python -m fme.ace.validate_config "${CONFIG}" --config_type train \
    --override "${OVERRIDES[@]}" \
    || { echo "ERROR ACE2_CONFIG_INVALID"; exit 1; }

torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
    -m fme.ace.train "${CONFIG}" --override "${OVERRIDES[@]}"

python3 - "${EXP_DIR}/out.log" "${PAIR}" <<'PYEOF'
import datetime, re, statistics, sys
log, pair = sys.argv[1], sys.argv[2]
ts = lambda l: datetime.datetime.strptime(l[:23], "%Y-%m-%d %H:%M:%S,%f")
steps = [ts(l) for l in open(log) if re.search(r"Step \d+: \{", l)]
if len(steps) < 35:
    print(f"ERROR ACE2_TOPO_TOO_FEW_STEPS {len(steps)}"); sys.exit(1)
d = [(b - a).total_seconds() for a, b in zip(steps, steps[1:])][-30:]
print(f"TOPO pair={pair} n_steps={len(steps)} step_med_s={statistics.median(d):.4f} "
      f"samples_per_s_per_rank={1/statistics.median(d):.3f}")
print("ACE2_TOPO_OK")
PYEOF
