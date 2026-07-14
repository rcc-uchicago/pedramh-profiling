#!/bin/bash
# Phase F.B2 — Pad canonical years 11 (no_leap_year) and 12 (leap_year) AFTER stats.
#
# CRITICAL ORDERING: must run AFTER F.C (stats + climatology) so the padded
# frames don't contaminate stats / climatology.
#
# Year 11 idx [1455, 1460) <- year 12 idx [0, 5)   (donor: train year 12)
# Year 12 idx [1455, 1464) <- year 13 idx [0, 9)   (donor: train year 13)
#
# Padded frames are invisible to train (caps at idx 1453) and validation
# (caps at idx 1394). Only single_ic long_inference reads the padded zone.

#SBATCH -J group_pad_canonical
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:15:00
#SBATCH -o logs/group_pad_canonical_%j.out
#SBATCH -e logs/group_pad_canonical_%j.err

set -euo pipefail
REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
# Use .venv (no cuda needed for h5 copy). The group conda env loads cuda/12.4
# which isn't available on skx-dev; the pad job only uses h5py + numpy + cftime.
source "$REPO_ROOT/.venv/bin/activate"

export DATA_DIR="${DATA_DIR:-$SCRATCH/SFNO_Climate_Emulator/data/group_sfno/sim52_full}"
mkdir -p "$REPO_ROOT/logs"

echo "[pad_canonical] data_dir=$DATA_DIR"

# Sanity: F.C must already have written stats / climatology.
for f in data_train_mean.nc data_train_std.nc climatology.nc; do
  if [[ ! -f "$DATA_DIR/$f" ]]; then
    echo "ERROR: $f missing in $DATA_DIR. F.C (stats + climatology) must run BEFORE F.B2."
    exit 2
  fi
done

# Sanity: stats/climatology attrs say built_before_padding=true.
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
import xarray as xr, os, sys
data_dir = os.environ["DATA_DIR"]
for fn in ("data_train_mean.nc", "data_train_std.nc", "climatology.nc"):
    ds = xr.open_dataset(f"{data_dir}/{fn}")
    assert int(ds.attrs.get("built_before_padding", -1)) == 1, (
        f"{fn} attrs say built_before_padding != 1; refusing to pad after the fact. "
        f"This protects against out-of-order F.B2 -> F.C runs."
    )
    print(f"  {fn}: built_before_padding={int(ds.attrs['built_before_padding'])}, "
          f"native_total={int(ds.attrs['native_timesteps_total'])}")
    ds.close()
print("[pad_canonical] stats provenance OK; safe to pad.")
PY

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m sfno_training_group.tools.convert_v10_to_group_h5 \
  --dst "$DATA_DIR" \
  --pad-canonical-years-only

echo "[pad_canonical] done."
echo "Year 11 file count: $(find "$DATA_DIR" -maxdepth 1 -name '11_*.h5' | wc -l)"
echo "Year 12 file count: $(find "$DATA_DIR" -maxdepth 1 -name '12_*.h5' | wc -l)"
