#!/bin/bash
# Phase F.D — Build init NCs for the 8 test years (121..128).
#
# Each init NC has time coord = [YYYY-01-01 00:00:00] (1 entry, EXACT match
# required by long_inference.py:1331).
#
# Output:
#   $EXP_DIR/init_year121.nc ... init_year128.nc

#SBATCH -J group_init_nc
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH -o logs/group_init_nc_%j.out
#SBATCH -e logs/group_init_nc_%j.err

set -euo pipefail
REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"

# Use the project .venv (h5py + xarray + numpy + cftime). The group conda env
# loads cuda/12.4 which isn't available on skx-dev; the init-NC builder
# doesn't need cuda anyway.
source "$REPO_ROOT/.venv/bin/activate"

EXP_DIR="${EXP_DIR:-$SCRATCH/SFNO_Climate_Emulator/runs/sfno_group_sigma10_full}"
TEST_SRC="${TEST_SRC:-$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev/test}"

mkdir -p "$EXP_DIR" "$REPO_ROOT/logs"

cd "$REPO_ROOT"
for YEAR in 121 122 123 124 125 126 127 128; do
  SRC="$TEST_SRC/MOST.$(printf '%04d' $YEAR).h5"
  OUT="$EXP_DIR/init_year${YEAR}.nc"
  if [[ ! -f "$SRC" ]]; then
    echo "ERROR: $SRC missing"
    exit 2
  fi
  if [[ -f "$OUT" ]]; then
    echo "[init_nc] YEAR=$YEAR: $OUT exists, skipping (idempotent)"
    continue
  fi
  echo "[init_nc] YEAR=$YEAR ..."
  PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m sfno_training_group.tools.build_init_nc_from_v10 \
    --src-h5 "$SRC" \
    --ic-idx 0 \
    --synthetic-init-dt "$(printf '%04d' $YEAR)-01-01 00:00:00" \
    --out "$OUT"
done

echo "[init_nc] done. 8 init NCs at $EXP_DIR/init_year121..128.nc"
ls -la "$EXP_DIR"/init_year*.nc | head -10
