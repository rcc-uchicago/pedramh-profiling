#!/bin/bash
#SBATCH -J plasim_makani_pack
#SBATCH -p skx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH --array=0-0                 # REQUIRED — set after running --count-tasks (see below)
#SBATCH -o logs/packager_%A_%a.out

# ============================================================================
# REQUIRED — provide via the environment (e.g. `sbatch --export=ALL,SIMS=52,
# POSTPROC_ROOT=...,BOUNDARY_ROOT=...,OUTPUT_ROOT=... submit.slurm`) or by
# editing the defaults below. The `${VAR:-}` form is intentional: it leaves
# any caller-exported value untouched so `sbatch --export=...` actually wins.
# An empty value still trips the fail-fast checks below.
# ============================================================================
SIMS="${SIMS:-}"                                # e.g. "52"
POSTPROC_ROOT="${POSTPROC_ROOT:-}"              # postprocess output root, e.g. $SCRATCH/SFNO_Climate_Emulator/data/postproc
BOUNDARY_ROOT="${BOUNDARY_ROOT:-}"              # boundary output root,    e.g. $SCRATCH/SFNO_Climate_Emulator/data/boundary_astro
OUTPUT_ROOT="${OUTPUT_ROOT:-}"                  # packager output root,    e.g. $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev (v10 / zgplev)
POSTPROC_SOURCE_DIR="${POSTPROC_SOURCE_DIR:-}"  # local clone used to resolve the postprocessor git SHA (L7c)
                                                # default: $HOME/projects/SFNO_Climate_Emulator/src/plasim_postprocessor

# Optional — split ranges. Defaults match the v9 plan: train 3-100, valid
# 101-120, test 121-128 (warmup years 1-2 are always skipped).
TRAIN_YEARS="${TRAIN_YEARS:-}"                  # "3 100"
VALID_YEARS="${VALID_YEARS:-}"                  # "101 120"
TEST_YEARS="${TEST_YEARS:-}"                    # "121 128"

# Optional — restrict iteration to (sim, year) pairs with year in
# [YS_LO YS_HI]. Used to shard a 109-task pack across two SLURM arrays
# when MaxArraySize<109 (Stampede3 caps at 100).
YEAR_SLICE="${YEAR_SLICE:-}"                    # "111 128"

# Optional — sentinel used to fill NaN over land in adaptor sst.
SST_LAND_FILL_K="${SST_LAND_FILL_K:-}"          # "271.35"

# ============================================================================
# SIZING THE ARRAY
# ----------------------------------------------------------------------------
# 1. Compute task count:
#       SIMS="52" TRAIN_YEARS="3 100" VALID_YEARS="101 120" TEST_YEARS="121 128"
#       N=$(python3 -m plasim_makani_packager.packager \
#               --sims $SIMS --train-years $TRAIN_YEARS \
#               --valid-years $VALID_YEARS --test-years $TEST_YEARS \
#               --count-tasks)
# 2. Set #SBATCH --array=0-$((N-1)) above, OR submit with:
#       sbatch --array=0-$((N-1)) submit.slurm
# ============================================================================

# ============================================================================
# ENVIRONMENT
# ----------------------------------------------------------------------------
# The packager needs python + xarray + h5py + netCDF4. Activate the project
# venv (which also provides makani for the Phase 4b smoke) before dispatch.
# ============================================================================
source "$HOME/projects/SFNO_Climate_Emulator/.venv/bin/activate"

# ============================================================================
# FAIL FAST IF UNSET
# ============================================================================
: "${SIMS:?SIMS is required — edit submit.slurm}"
: "${POSTPROC_ROOT:?POSTPROC_ROOT is required}"
: "${BOUNDARY_ROOT:?BOUNDARY_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

# ============================================================================
# DISPATCH
# ============================================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
# SLURM stages submit scripts to a temp dir, so BASH_SOURCE-derived REPO_ROOT
# can land outside the source tree. Fall back to SLURM_SUBMIT_DIR (the dir
# sbatch was invoked from) when the resolved REPO_ROOT doesn't contain the
# package; final fallback is $HOME/projects/SFNO_Climate_Emulator (consistent with the .venv activate
# line above, which is already hardcoded to that path).
if [[ ! -d "$REPO_ROOT/src/plasim_makani_packager" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR:-$HOME/projects/SFNO_Climate_Emulator}"
fi
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
mkdir -p logs

YEAR_RANGE_FLAGS=()
if [[ -n "$TRAIN_YEARS" ]]; then YEAR_RANGE_FLAGS+=(--train-years $TRAIN_YEARS); fi
if [[ -n "$VALID_YEARS" ]]; then YEAR_RANGE_FLAGS+=(--valid-years $VALID_YEARS); fi
if [[ -n "$TEST_YEARS"  ]]; then YEAR_RANGE_FLAGS+=(--test-years  $TEST_YEARS);  fi

YEAR_SLICE_FLAG=()
if [[ -n "$YEAR_SLICE" ]]; then YEAR_SLICE_FLAG+=(--year-slice $YEAR_SLICE); fi

SST_FLAG=()
if [[ -n "$SST_LAND_FILL_K" ]]; then SST_FLAG=(--sst-land-fill-k "$SST_LAND_FILL_K"); fi

# Opt-in destructive overwrite: re-runs that need to replace existing H5
# outputs (e.g. ZG_PLEV_HPA subset migrations) set OVERWRITE=1 in the env.
# Default behavior stays non-destructive: packager.py:493 skips existing
# outputs, so a routine resubmit is idempotent.
OVERWRITE_FLAG=()
if [[ "${OVERWRITE:-}" == "1" ]]; then OVERWRITE_FLAG=(--overwrite); fi

# Resolve postprocessor git SHA (L7c). Defaults to a checkout under $HOME so
# the v10 file_attrs always carry a traceable postproc version. validate.py
# --mode files rejects 'unknown' in production.
POSTPROC_SOURCE_DIR="${POSTPROC_SOURCE_DIR:-$HOME/projects/SFNO_Climate_Emulator/src/plasim_postprocessor}"
POSTPROC_SHA=$(git -C "$POSTPROC_SOURCE_DIR" rev-parse HEAD 2>/dev/null || echo unknown)

python3 -m plasim_makani_packager.packager \
    --sims $SIMS \
    --postproc-root "$POSTPROC_ROOT" \
    --boundary-root "$BOUNDARY_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --postprocessor-git-sha "$POSTPROC_SHA" \
    "${YEAR_RANGE_FLAGS[@]}" \
    "${YEAR_SLICE_FLAG[@]}" \
    "${SST_FLAG[@]}" \
    "${OVERWRITE_FLAG[@]}" \
    --task-index "$SLURM_ARRAY_TASK_ID"
