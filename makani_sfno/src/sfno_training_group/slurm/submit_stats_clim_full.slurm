#!/bin/bash
# Phase F.C — recompute mean/std + daily climatology over 100 train years
# (years 12-111). Reads NATIVE frames only via manifest's n_timesteps_native.
#
# MUST run AFTER F.B1 (native conversion) and BEFORE F.B2 (padding).
# Output:
#   $DATA_DIR/data_train_mean.nc
#   $DATA_DIR/data_train_std.nc
#   $DATA_DIR/climatology.nc
# Each NetCDF carries provenance attrs: built_before_padding, native_timesteps_total,
# native_timesteps_by_year, manifest_skip_padded_used, git_sha, build_timestamp_utc.

#SBATCH -J group_stats_clim_full
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 06:00:00
#SBATCH -o logs/group_stats_clim_full_%j.out
#SBATCH -e logs/group_stats_clim_full_%j.err

set -euo pipefail
REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
source "$REPO_ROOT/src/sfno_training_group/env_activate.sh"

DATA_DIR="${DATA_DIR:-$SCRATCH/SFNO_Climate_Emulator/data/group_sfno/sim52_full}"

# Train years 12 through 111 (100 years).
TRAIN_YEARS=()
for y in $(seq 12 111); do TRAIN_YEARS+=("$y"); done

cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/logs"

echo "[stats_clim_full] data_dir=$DATA_DIR"
echo "[stats_clim_full] train years: ${#TRAIN_YEARS[@]} ($(echo ${TRAIN_YEARS[@]} | head -c 40)...)"

# Sanity: assert F.B2 has NOT yet run (built_before_padding must be true).
PADDED_PRESENT=$(python3 -c "
import json
m = json.load(open('$DATA_DIR/_v10_calendar_manifest.json'))
print(any('n_timesteps_padded' in y for y in m['years']))
")
if [[ "$PADDED_PRESENT" == "True" ]]; then
  echo "ERROR: manifest has padded years; F.C must run BEFORE F.B2."
  exit 2
fi

echo "=== F.C / 1: stats (mean + std) ==="
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m sfno_training_group.tools.build_group_stats_netcdf \
  --data-dir "$DATA_DIR" \
  --train-years "${TRAIN_YEARS[@]}"

echo "=== F.C / 2: daily climatology ==="
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m sfno_training_group.tools.build_group_climatology \
  --data-dir "$DATA_DIR" \
  --train-years "${TRAIN_YEARS[@]}"

echo "[stats_clim_full] done."
echo "Files:"
ls -la "$DATA_DIR"/data_train_mean.nc "$DATA_DIR"/data_train_std.nc "$DATA_DIR"/climatology.nc 2>/dev/null
