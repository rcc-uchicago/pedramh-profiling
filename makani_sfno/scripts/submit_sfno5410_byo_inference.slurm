#!/bin/bash
#SBATCH -J 5410_byo_inf
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o slurm-byo-%j.out
#SBATCH -e slurm-byo-%j.err

# ============================================================================
# SFNO-5410 inference from a user-supplied (BYO) NetCDF initial condition.
# Runs on Stampede3 H100 with the packed Derecho env. See the user guide:
#   docs/2026-05-09_sfno_5410_byo_inference_user_guide.md
#
# Required env vars (no defaults):
#   IC_NC          — absolute path to the user's IC NetCDF.
#   INIT_DATETIME  — IC datetime, "YYYY-MM-DD_HH:MM:SS" (hour 00/06/12/18).
#   HORIZON_DAYS   — forecast horizon in days (max 365).
#   OUT_DIR        — directory for member NetCDFs + metadata.
#
# Optional env vars (with defaults):
#   NUM_MEMBERS        — ensemble size; default 1 (deterministic).
#   EPSILON_FACTOR     — IC perturbation magnitude; default 0.0.
#                        Required >0 when NUM_MEMBERS>1.
#   PERTURBATION_TYPE  — gaussian_noise | gaussian_noise_n_minus_1 | perlin_noise.
#                        Required when EPSILON_FACTOR>0.
#   RANDOM_SEED        — perturber seed; default unset (random).
#   OUT_PREFIX         — filename prefix; default "byo".
#   BOUNDARY_TEMPLATE_YEAR — sim52 template year; default 51 non-leap / 52 leap.
#   FORCE              — set to 1 to overwrite existing outputs.
#   PACKED_PYTHON      — packed Derecho env Python; default below.
#   DRIVER_PY          — absolute path to infer_sfno5410_byo_ic.py.
#                        Auto-resolves to the script's own dir; override only
#                        if you've copied the driver elsewhere.
#   LOG_DIR            — where SLURM .out/.err land. Default: $PWD/logs at
#                        submit time (sbatch's working dir).
# ============================================================================

set -euo pipefail

: "${PACKED_PYTHON:=/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_runtime/aires_env_20260509/unpacked/bin/python}"
: "${OUT_PREFIX:=byo}"

# Resolve the script's own directory (works for both repo-layout and the
# flat group-workspace copy).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${DRIVER_PY:-}" ]]; then
    if [[ -f "$HERE/infer_sfno5410_byo_ic.py" ]]; then
        DRIVER_PY="$HERE/infer_sfno5410_byo_ic.py"
    elif [[ -f "$HERE/../scripts/infer_sfno5410_byo_ic.py" ]]; then
        DRIVER_PY="$(cd "$HERE/../scripts" && pwd)/infer_sfno5410_byo_ic.py"
    else
        echo "FATAL: could not locate infer_sfno5410_byo_ic.py near $HERE; set DRIVER_PY" >&2
        exit 2
    fi
fi

# --- required-env check ---
required=( IC_NC INIT_DATETIME HORIZON_DAYS OUT_DIR )
for v in "${required[@]}"; do
    if [[ -z "${!v:-}" ]]; then
        echo "FATAL: required env var $v is unset" >&2
        echo "Required: ${required[*]}" >&2
        exit 2
    fi
done

mkdir -p "$OUT_DIR"

if [[ ! -x "$PACKED_PYTHON" ]]; then
    echo "FATAL: packed Derecho env interpreter missing at $PACKED_PYTHON" >&2
    exit 2
fi
if [[ ! -f "$IC_NC" ]]; then
    echo "FATAL: IC_NC=$IC_NC does not exist" >&2
    exit 2
fi
if [[ ! -f "$DRIVER_PY" ]]; then
    echo "FATAL: DRIVER_PY=$DRIVER_PY does not exist" >&2
    exit 2
fi

ARGS=(
    "$DRIVER_PY"
    --ic-nc          "$IC_NC"
    --init-datetime  "$INIT_DATETIME"
    --horizon-days   "$HORIZON_DAYS"
    --output-dir     "$OUT_DIR"
    --output-prefix  "$OUT_PREFIX"
)
[[ -n "${NUM_MEMBERS:-}" ]]            && ARGS+=( --num-members        "$NUM_MEMBERS" )
[[ -n "${EPSILON_FACTOR:-}" ]]         && ARGS+=( --epsilon-factor     "$EPSILON_FACTOR" )
[[ -n "${PERTURBATION_TYPE:-}" ]]      && ARGS+=( --perturbation-type  "$PERTURBATION_TYPE" )
[[ -n "${RANDOM_SEED:-}" ]]            && ARGS+=( --random-seed        "$RANDOM_SEED" )
[[ -n "${BOUNDARY_TEMPLATE_YEAR:-}" ]] && ARGS+=( --boundary-template-year "$BOUNDARY_TEMPLATE_YEAR" )
[[ "${FORCE:-0}" == "1" ]]             && ARGS+=( --force )

echo "[submit_sfno5410_byo_inference] python=$PACKED_PYTHON"
echo "[submit_sfno5410_byo_inference] repo_root=$REPO_ROOT"
echo "[submit_sfno5410_byo_inference] argv=${ARGS[*]}"

time "$PACKED_PYTHON" "${ARGS[@]}"
