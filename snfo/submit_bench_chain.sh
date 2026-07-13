#!/bin/bash
# Submit the bench pipeline as a SLURM dependency chain.
#
# Run bench_gpu_test_a100.sh or bench_gpu_test_h100.sh manually first, then
# call this script to chain validate → full bench on pi-pedramh.
#
# Step 1 — validate_bench.sh  : CPU-only CSV sanity check (5 min)
# Step 2 — bench_midway.sh    : full 4-GPU bench (pedramh-gpu, 30 min)
#
# Step 2 only runs if step 1 exits 0.
#
# Usage:
#   bash submit_bench_chain.sh

# Guard: this script must be run with bash on a login node, not submitted
# with sbatch (it has no SLURM headers and calls sbatch internally).
if [[ -n "${SLURM_JOB_ID}" ]]; then
    echo "ERROR: submit_bench_chain.sh must be run on a login node:"
    echo "  bash submit_bench_chain.sh"
    echo "Do not submit it with sbatch."
    exit 1
fi

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== Submitting bench chain from $(pwd) ==="

JOB1=$(sbatch --parsable validate_bench.sh)
echo "  [1] Validation submitted      : job ${JOB1}"

JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} bench_midway.sh)
echo "  [2] Full 4-GPU bench submitted: job ${JOB2} (after ${JOB1})"

echo ""
echo "  Chain: ${JOB1} → ${JOB2}"
echo "  Cancel all:  scancel ${JOB1} ${JOB2}"
echo "  Monitor:     squeue -u \${USER}"
