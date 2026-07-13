#!/bin/bash -l
#SBATCH --account=pi-pedramh
#SBATCH --time=01:00:00
#SBATCH -p pedramh-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o midway_infer_nsys_%N.out
#SBATCH -e midway_infer_nsys_%N.err

# Nsight Systems inference profile — intentionally matches the DSI collection
# command as closely as possible so the two profiles are directly comparable:
#
#   nsys profile -w true -t cuda,nvtx,cudnn \
#     --force-overwrite=true \
#     torchrun --standalone --nproc_per_node=4 \
#     inference_optimized.py --yaml_config=exp2.yaml
#
# No container (bare metal conda env, same as DSI).
# No extra CUDA/NCCL env vars beyond what DSI had, so hardware topology
# is the only remaining variable between the two clusters.
# Exports to SQLite automatically for compare_nsys.py analysis.

ulimit -l unlimited

module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/pedramh/shared/S2S/v2.0/venv

module unload cuda
module load cuda/12.9

export WANDB_MODE=offline

echo "=== midway_infer_nsys: $(date -Iseconds) ==="
echo "JOB_ID=${SLURM_JOB_ID}  NODELIST=${SLURM_NODELIST}"
nvidia-smi -L
echo
echo "--- nvidia-smi topo -m ---"
nvidia-smi topo -m
echo
echo "--- numactl --hardware ---"
numactl --hardware 2>/dev/null || true
echo
echo "--- cpu / power state (dispatch-latency suspects) ---"
cpupower frequency-info 2>/dev/null | head -20 || true
cpupower idle-info     2>/dev/null | head -20 || true
cat /sys/module/intel_idle/parameters/max_cstate 2>/dev/null || true
echo
echo "--- nvidia IRQ steering ---"
grep -E "nvidia" /proc/interrupts 2>/dev/null | head -8 || true
echo
echo "--- driver / cuda versions ---"
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}')" 2>/dev/null || true
echo

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "NUM_GPUS=${NUM_GPUS}"

config_file=/project/pedramh/shared/S2S/v2.0/config/exp2.yaml
run_num="infer_nsys_$(date +%s)"
NSYS_OUT="${SLURM_SUBMIT_DIR}/midway_h100_4gpus_inference_${SLURM_JOB_ID}_$(date +%F)"

echo "nsys output: ${NSYS_OUT}.nsys-rep"

nsys profile \
    -w true \
    -t cuda,nvtx,cudnn \
    -o "${NSYS_OUT}" \
    --force-overwrite=true \
    torchrun \
        --standalone \
        --nproc_per_node="${NUM_GPUS}" \
        /project/pedramh/shared/S2S/v2.0/inference_optimized.py \
        --yaml_config="${config_file}" \
        --run_num="${run_num}" \
        --async_save




echo "Profile written: ${NSYS_OUT}.nsys-rep"
echo "scp the .nsys-rep locally and run: nsys export --type=sqlite <file>.nsys-rep"
