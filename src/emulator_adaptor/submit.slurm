#!/bin/bash
#SBATCH -J emu_adaptor
#SBATCH -p skx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH --array=0-0                 # REQUIRED — set after running --count-tasks (see below)
#SBATCH -o logs/adaptor_%A_%a.out

# ============================================================================
# REQUIRED — provide via the environment (e.g. `sbatch --export=ALL,SIMS=52,
# YEAR_START=1,YEAR_END=100,INPUT_ROOT=...,OUTPUT_ROOT=... submit.slurm`) or by
# editing the defaults below. The `${VAR:-}` form is intentional: it leaves any
# caller-exported value untouched so `sbatch --export=...` actually wins. An
# empty value still trips the fail-fast checks below.
# ============================================================================
SIMS="${SIMS:-}"                # e.g. "30 31 32 33 34 35 36"
YEAR_START="${YEAR_START:-}"    # e.g. 1
YEAR_END="${YEAR_END:-}"        # e.g. 100
INPUT_ROOT="${INPUT_ROOT:-}"    # postprocess output root, e.g. $SCRATCH/SFNO_Climate_Emulator/data/postproc
OUTPUT_ROOT="${OUTPUT_ROOT:-}"  # boundary output root,    e.g. $SCRATCH/SFNO_Climate_Emulator/data/boundary

# Optional — default is "arithmetic" (rsdt = rst - rsut from PlaSim).
# Set to "astronomical" to use the analytic 6h-mean integral path instead.
RSDT_METHOD="${RSDT_METHOD:-}"

# Optional — default is "surface" (PlaSim-faithful: ts over all non-land cells,
# preserves cold-ice signal). Set to "ocean_era5" for v10 back-compat regen
# (ERA5/CMIP convention: ice-covered ocean clamped to FREEZING_SEAWATER_K).
SST_MODE="${SST_MODE:-}"

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

# The adaptor needs python + xarray + numpy + pandas + netCDF4. Activate the
# project venv (same pattern as the packager submit script).
source "$HOME/projects/SFNO_Climate_Emulator/.venv/bin/activate"

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
# SLURM stages submit scripts into /var/spool/slurmd/jobXXX/ before execution,
# so the BASH_SOURCE-derived path lands outside the source tree. Match the
# packager's submit pattern: try BASH_SOURCE first, then SLURM_SUBMIT_DIR,
# then fall back to $HOME/projects/SFNO_Climate_Emulator/src/emulator_adaptor.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [[ ! -f "$SCRIPT_DIR/adaptor.py" ]]; then
    if [[ -f "${SLURM_SUBMIT_DIR:-}/src/emulator_adaptor/adaptor.py" ]]; then
        SCRIPT_DIR="$SLURM_SUBMIT_DIR/src/emulator_adaptor"
    else
        SCRIPT_DIR="$HOME/projects/SFNO_Climate_Emulator/src/emulator_adaptor"
    fi
fi
mkdir -p logs

RSDT_METHOD_FLAG=()
if [[ -n "$RSDT_METHOD" ]]; then
    RSDT_METHOD_FLAG=(--rsdt-method "$RSDT_METHOD")
fi

SST_MODE_FLAG=()
if [[ -n "$SST_MODE" ]]; then
    SST_MODE_FLAG=(--sst-mode "$SST_MODE")
fi

python3 "$SCRIPT_DIR/adaptor.py" \
    --sims $SIMS \
    --years "$YEAR_START" "$YEAR_END" \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    "${RSDT_METHOD_FLAG[@]}" \
    "${SST_MODE_FLAG[@]}" \
    --task-index "$SLURM_ARRAY_TASK_ID"
