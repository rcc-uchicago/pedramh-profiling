#!/bin/bash
# profile_solver_midway.sh -- Job A: standalone profile of the A-SQG solver step on one Midway GPU.
#   sbatch SQG/analysis/profile_solver_midway.sh
# Phase 1: profile_solver.py --mode bench (timings, dispatch-blocking test, HLO counts, bitwise checks of the
#          donate / fixed-AB3-level variants) with nvidia-smi dmon logging alongside.
# Phase 2: nsys capture of the production pattern (12 x 40 steps + coarse, x5) for the ref and fixed variants,
#          then nsys stats CSVs and summarize_nsys.py.
# PASS = "PROFILE_OK bench" + 2x "NSYS_SUMMARY_OK" in the .out. Outputs: SQG/analysis/results/profile_solver_<jobid>/
#SBATCH --job-name=sqg-prof-solver
#SBATCH --account=pi-pedramh
#SBATCH --partition=pedramh-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=SQG/analysis/logs/prof_solver_%j.out
#SBATCH --error=SQG/analysis/logs/prof_solver_%j.err
set -eo pipefail

module load cuda/12.9                                       # nsys + NVTX/CUPTI (prepends LD_LIBRARY_PATH; the
                                                            # harness records which cuFFT/cudart JAX actually loads)
PY=/project/pedramh/weidong/miniconda3/envs/wdenv/bin/python   # Weidong's Midway JAX env (jax 0.4.28, torch 2.3.1)
export XLA_PYTHON_CLIENT_PREALLOCATE=false HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=8

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
OUT=SQG/analysis/results/profile_solver_${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
echo "=== profile_solver on $(hostname) job ${SLURM_JOB_ID:-local}: $(date '+%F %T') ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit --format=csv | tee "$OUT/gpu.csv"
echo "nsys: $(command -v nsys) $(nsys --version 2>&1 | head -1)"

# ---- phase 1: bench (+ GPU power/clock/util log) ----
nvidia-smi dmon -s pucvm -d 1 -i 0 > "$OUT/dmon_bench.txt" 2>&1 &
DMON=$!
"$PY" SQG/analysis/profile_solver.py --mode bench --out "$OUT"
kill "$DMON" 2>/dev/null || true

# ---- phase 2: nsys, ref and fixed ----
REPS=5
for V in ref fixed; do
    nsys profile -t cuda,nvtx,osrt --cuda-graph-trace=node \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        --force-overwrite=true -o "$OUT/nsys_$V" \
        "$PY" SQG/analysis/profile_solver.py --mode nsys --variant "$V" --reps "$REPS" --out "$OUT"
    nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_trace --format csv --force-export=true \
        --output "$OUT/nsys_$V" "$OUT/nsys_$V.nsys-rep" > /dev/null
    "$PY" SQG/analysis/summarize_nsys.py "$OUT/nsys_$V" --steps $((REPS * 12 * 40)) --out "$OUT/nsys_${V}_summary.json"
done
echo "=== end $(date '+%F %T') ==="
