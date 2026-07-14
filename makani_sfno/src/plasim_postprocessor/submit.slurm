#!/bin/bash
#SBATCH -J plasim_postproc
#SBATCH -p skx                      # adjust queue as needed
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --array=0-0                 # REQUIRED — set after running --count-tasks (see below)
#SBATCH -o logs/postproc_%A_%a.out

# ============================================================================
# REQUIRED — fill these in for your run.
# Sample values are shown only as comments; this template intentionally has
# no defaults so an unedited submission fails fast.
# ============================================================================
SIMS=""                # e.g. "30 31 32 33 34 35 36"
YEAR_START=            # e.g. 1
YEAR_END=              # e.g. 100
INPUT_ROOT=            # e.g. /scratch/10000/amarchakitus/PLASIM/data
OUTPUT_ROOT=           # e.g. $SCRATCH/SFNO_Climate_Emulator/data/postproc

# Set WITH_SEA_ICE=1 only if the source sims were run with sea ice enabled.
# Leave empty (the default) otherwise — burn7 will fail if sic is requested on
# a sim whose PlaSim namelist had sea ice disabled.
WITH_SEA_ICE=

# ============================================================================
# SIZING THE ARRAY
# ----------------------------------------------------------------------------
# 1. Compute task count (does not require --input-root/--output-root):
#       SIMS="30 31" YEAR_START=1 YEAR_END=100
#       N=$(python3 plasim_postprocessor.py \
#               --sims $SIMS --years $YEAR_START $YEAR_END --count-tasks)
# 2. Set #SBATCH --array=0-$((N-1)) above, OR submit with:
#       sbatch --array=0-$((N-1)) submit.slurm
#
# The default --array=0-0 only processes the first (sim, year) pair as a
# smoke test. Resize before launching the real run.
# ============================================================================

# ============================================================================
# ENVIRONMENT (Stampede3-specific; verified by audit)
# ----------------------------------------------------------------------------
# - intel/24.0 is the toolchain cdo/2.3.0 needs.
# - python/3.12.11 is required (script uses 3.10+ union syntax / Path).
# - LD_LIBRARY_PATH adds the legacy netcdf C++ library that burn7 links against.
# ============================================================================
module purge
module load intel/24.0 cdo netcdf python/3.12.11
export LD_LIBRARY_PATH=/opt/apps/intel24/netcdf/4.9.2/x86_64/lib:/home1/09979/awikner/netcdf-4.2/lib:${LD_LIBRARY_PATH}

# ============================================================================
# FAIL FAST IF UNSET — keeps an unedited template from burning a queue slot
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

SEA_ICE_FLAG=()
if [[ -n "$WITH_SEA_ICE" ]]; then
    SEA_ICE_FLAG=(--with-sea-ice)
fi

python3 "$SCRIPT_DIR/plasim_postprocessor.py" \
    --sims $SIMS \
    --years "$YEAR_START" "$YEAR_END" \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    "${SEA_ICE_FLAG[@]}" \
    --task-index "$SLURM_ARRAY_TASK_ID"
