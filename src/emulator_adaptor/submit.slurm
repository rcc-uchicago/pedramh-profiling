#!/bin/bash
#SBATCH -J emu_adaptor
#SBATCH -p skx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH --array=0-0                 # REQUIRED — set after running --count-tasks (see below)
#SBATCH -o logs/adaptor_%A_%a.out

# ============================================================================
# REQUIRED — fill these in for your run. No defaults: an unedited submission
# fails fast on the checks below.
# ============================================================================
SIMS=""                # e.g. "30 31 32 33 34 35 36"
YEAR_START=            # e.g. 1
YEAR_END=              # e.g. 100
INPUT_ROOT=            # postprocess output root, e.g. $SCRATCH/AI-RES/data/postproc
OUTPUT_ROOT=           # boundary output root,    e.g. $SCRATCH/AI-RES/data/boundary

# Optional — default is "arithmetic" (rsdt = rst - rsut from PlaSim).
# Set to "astronomical" to use the analytic 6h-mean integral path instead.
RSDT_METHOD=

# ============================================================================
# SIZING THE ARRAY
# ----------------------------------------------------------------------------
# 1. Compute task count (does not require --input-root/--output-root):
#       SIMS="30 31" YEAR_START=1 YEAR_END=100
#       N=$(python3 adaptor.py --sims $SIMS --years $YEAR_START $YEAR_END --count-tasks)
# 2. Set #SBATCH --array=0-$((N-1)) above, OR submit with:
#       sbatch --array=0-$((N-1)) submit.slurm
# ============================================================================

# ============================================================================
# ENVIRONMENT — matches plasim_postprocessor's module stanza so the adaptor
# can be chained after the postprocessor in the same submission script.
# (LD_LIBRARY_PATH is a postprocessor-specific burn7 requirement and is
# harmless here; kept for uniformity.)
# ============================================================================
module purge
module load intel/24.0 cdo netcdf python/3.12.11
export LD_LIBRARY_PATH=/opt/apps/intel24/netcdf/4.9.2/x86_64/lib:/home1/09979/awikner/netcdf-4.2/lib:${LD_LIBRARY_PATH}

# ============================================================================
# FAIL FAST IF UNSET
# ============================================================================
: "${SIMS:?SIMS is required — edit submit.slurm}"
: "${YEAR_START:?YEAR_START is required}"
: "${YEAR_END:?YEAR_END is required}"
: "${INPUT_ROOT:?INPUT_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

# ============================================================================
# DISPATCH
# ============================================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
mkdir -p logs

RSDT_METHOD_FLAG=()
if [[ -n "$RSDT_METHOD" ]]; then
    RSDT_METHOD_FLAG=(--rsdt-method "$RSDT_METHOD")
fi

python3 "$SCRIPT_DIR/adaptor.py" \
    --sims $SIMS \
    --years "$YEAR_START" "$YEAR_END" \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    "${RSDT_METHOD_FLAG[@]}" \
    --task-index "$SLURM_ARRAY_TASK_ID"
