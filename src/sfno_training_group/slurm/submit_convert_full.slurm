#!/bin/bash
# Phase F.B1 — sequential conversion of 101 v10 years (year 11 + years 12-111)
# into group per-timestep H5 layout. NO padding here (F.B2 runs after F.C).
#
# Sequential because:
#   - Single manifest written at end by convert_v10_to_group_h5.main()
#   - Avoiding cross-job manifest merge logic
#   - 101 × ~4 min/yr ~= 7 h, well under 15 h h100 walltime
#
# Output:
#   $DATA_DIR_FULL/<year>_<idx:04>.h5   (1455 files per year)
#   $DATA_DIR_FULL/_v10_calendar_manifest.json

#SBATCH -J group_convert_full
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 12:00:00
#SBATCH -o logs/group_convert_full_%j.out
#SBATCH -e logs/group_convert_full_%j.err

set -euo pipefail
REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"

source "$REPO_ROOT/src/sfno_training_group/env_activate.sh"

SRC="${SRC:-$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev}"
DST="${DST:-$SCRATCH/SFNO_Climate_Emulator/data/group_sfno/sim52_full}"

mkdir -p "$DST" "$REPO_ROOT/logs"

# 101 years total: validation year 11 + 100 train years (12-111).
# Test years 121-128 are NOT converted: long_inference single_ic mode remaps
# all data_dir reads to canonical years 11/12 (data_loader_multifiles.py:926-934).
YEARS=(11)
for y in $(seq 12 111); do YEARS+=("$y"); done
echo "[convert_full] converting ${#YEARS[@]} years: ${YEARS[*]}"

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m sfno_training_group.tools.convert_v10_to_group_h5 \
  --src "$SRC" \
  --dst "$DST" \
  --years "${YEARS[@]}" \
  --max-forecast-lead-steps 60

echo "[convert_full] done. Manifest at $DST/_v10_calendar_manifest.json"
echo "Manifest size: $(stat -c%s "$DST/_v10_calendar_manifest.json") bytes"
# Count files via find to avoid `ls | head` SIGPIPE (which trips set -e).
echo "Total .h5 files in $DST: $(find "$DST" -maxdepth 1 -name '*.h5' -type f | wc -l)"
