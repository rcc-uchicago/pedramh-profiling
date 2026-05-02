#!/bin/bash
#SBATCH -J pack_zgplev_loop
#SBATCH -p skx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o logs/packager_loop_%j.out

# Single-node parallel-loop wrapper for the packager. Used when the array
# size (109 in our v10 zgplev pack) exceeds Stampede3's MaxArraySize=100
# limit. Submits one skx job that runs all task indices in a bash-bg loop.
#
# Pass the same env vars submit.slurm expects (SIMS, POSTPROC_ROOT,
# BOUNDARY_ROOT, OUTPUT_ROOT, TRAIN_YEARS, VALID_YEARS, TEST_YEARS,
# SST_LAND_FILL_K, POSTPROC_SOURCE_DIR) plus N_TASKS and PARALLEL.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$HOME/AI-RES}"
[[ -d "$REPO_ROOT/src/plasim_makani_packager" ]] || REPO_ROOT="$HOME/AI-RES"

source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

: "${SIMS:?SIMS required}"
: "${POSTPROC_ROOT:?POSTPROC_ROOT required}"
: "${BOUNDARY_ROOT:?BOUNDARY_ROOT required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT required}"
: "${N_TASKS:?N_TASKS required (e.g. 109)}"
PARALLEL="${PARALLEL:-8}"

POSTPROC_SOURCE_DIR="${POSTPROC_SOURCE_DIR:-$REPO_ROOT/src/plasim_postprocessor}"
POSTPROC_SHA=$(git -C "$POSTPROC_SOURCE_DIR" rev-parse HEAD 2>/dev/null || echo unknown)

YEAR_FLAGS=()
[[ -n "${TRAIN_YEARS:-}" ]] && YEAR_FLAGS+=(--train-years $TRAIN_YEARS)
[[ -n "${VALID_YEARS:-}" ]] && YEAR_FLAGS+=(--valid-years $VALID_YEARS)
[[ -n "${TEST_YEARS:-}"  ]] && YEAR_FLAGS+=(--test-years  $TEST_YEARS)
[[ -n "${SST_LAND_FILL_K:-}" ]] && YEAR_FLAGS+=(--sst-land-fill-k "$SST_LAND_FILL_K")

cd "$REPO_ROOT"
mkdir -p logs

echo "[loop] N_TASKS=$N_TASKS PARALLEL=$PARALLEL OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[loop] flags: ${YEAR_FLAGS[*]}"
echo "[loop] start $(date -u +%FT%TZ)"

# Background-pool with manual concurrency cap. wait -n needs bash >= 4.3.
running=0
for ((idx=0; idx<N_TASKS; idx++)); do
    (
        python3 -m plasim_makani_packager.packager \
            --sims $SIMS \
            --postproc-root "$POSTPROC_ROOT" \
            --boundary-root "$BOUNDARY_ROOT" \
            --output-root "$OUTPUT_ROOT" \
            --postprocessor-git-sha "$POSTPROC_SHA" \
            "${YEAR_FLAGS[@]}" \
            --task-index "$idx" 2>&1 | sed "s|^|[task=$idx] |"
    ) &
    running=$((running + 1))
    if (( running >= PARALLEL )); then
        wait -n
        running=$((running - 1))
    fi
done
wait
echo "[loop] done $(date -u +%FT%TZ)"
