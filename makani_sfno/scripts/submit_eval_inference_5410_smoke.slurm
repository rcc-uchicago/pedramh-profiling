#!/bin/bash
#SBATCH -J 5410_smoke
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o logs/5410_smoke_%j.out
#SBATCH -e logs/5410_smoke_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# 5410 Phase-1 1-IC NWP smoke — in-process orchestrator (v2.1, 2026-05-08).
#
# One IC: Y=121, s=0. Exercises the SAME in-process code path as the
# 96-IC production sweep (`scripts/submit_eval_inference_5410.slurm`),
# limited to the first IC via `--years 121 --limit-ics 1`. Validates:
#   - Stepper one-time construction (model + ckpt load)
#   - Stepper.reconfigure_for_ic (LP-004) for one IC
#   - All 4 post-reconfigure ValueError gates fire correctly
#   - LP-003 partial-horizon patch produces time-dim==K+1 NetCDF
#   - Output-dir hygiene (assert_output_dir_empty pre / _complete post)
#
# Pre-conditions (must be set up before submission):
#   * <RUN_ROOT>/inference/ic_source.json present (§3 P-7 gate must have run)
#   * <RUN_ROOT>/inference/ic_nc/121_0000.nc present
#   * <RUN_ROOT>/inference/SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y121.yaml
#       present with K=60 horizon (rebuild via build_5410_yaml_override.py
#       --year 121 --K 60 ...)
#   * <RUN_ROOT>/inference/SFNO/5410/checkpoints/ckpt_epoch_50.tar
#       (single-file symlink)
#   * Output dir <RUN_ROOT>/inference/upstream_raw EMPTY (or fresh RUN_ROOT)
#
# Required env vars (or accept defaults):
#   RUN_ROOT  — eval run root.
#                Default: /work2/.../results/sfno_eval_5410/20260507_phase1_gate
#   K         — forecast-leads horizon. Default 60.
# ============================================================================

set -euo pipefail

REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
UPSTREAM_REPO="/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0"
: "${RUN_ROOT:=/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval_5410/20260507_phase1_gate}"
: "${K:=60}"
SMOKE_Y=121
SMOKE_S=0

cd "$REPO_ROOT"
mkdir -p logs

# ---- venv -----------------------------------------------------------------
source "$REPO_ROOT/.venv/bin/activate"

# ---- distributed env (single-rank) ----------------------------------------
# Upstream long_inference.py reads WORLD_SIZE unconditionally
# (long_inference.py:1278) even in single-GPU runs. The dist init branch
# is skipped at world_size=1.
export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# AI-RES src on PYTHONPATH so the orchestrator's
# `from sfno_inference_5410.* import ...` resolves.
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# ---- pre-flight assertions -----------------------------------------------
echo "[preflight] RUN_ROOT=$RUN_ROOT K=$K SMOKE_Y=$SMOKE_Y SMOKE_S=$SMOKE_S"
test -f "$RUN_ROOT/inference/ic_source.json" \
    || { echo "FATAL: missing ic_source.json — §3 P-7 gate not run" >&2; exit 2; }
test -f "$RUN_ROOT/inference/ic_nc/${SMOKE_Y}_$(printf '%04d' $SMOKE_S).nc" \
    || { echo "FATAL: missing IC NC — run scripts/build_ic_nc_from_h5.py" >&2; exit 2; }

YAML="$RUN_ROOT/inference/SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y${SMOKE_Y}.yaml"
SHIM="$RUN_ROOT/inference/SFNO/5410/checkpoints/ckpt_epoch_50.tar"
test -f "$YAML" || { echo "FATAL: missing yaml override $YAML" >&2; exit 2; }
test -L "$SHIM" || { echo "FATAL: missing ckpt symlink shim $SHIM" >&2; exit 2; }

# ---- preflight: upstream long_inference.py is patched (LP-003 4+2 + LP-004) -----
ALLOC=$(grep -c 'min(next_year_jan1, self.params.final_datetime)' "$UPSTREAM_REPO/long_inference.py" || true)
CONT=$(grep -c 'current_datetime < self.params.final_datetime' "$UPSTREAM_REPO/long_inference.py" || true)
LP004=$(grep -c 'def reconfigure_for_ic' "$UPSTREAM_REPO/long_inference.py" || true)
if [[ "$ALLOC" -ne 4 || "$CONT" -ne 2 ]]; then
    echo "FATAL: upstream long_inference.py LP-003 patch incomplete: allocator markers=$ALLOC (expected 4), continuation markers=$CONT (expected 2)" >&2
    exit 2
fi
if [[ "$LP004" -ne 1 ]]; then
    echo "FATAL: upstream long_inference.py LP-004 patch missing: reconfigure_for_ic count=$LP004 (expected 1)" >&2
    exit 2
fi
echo "[preflight] upstream patches OK: LP-003 allocators=$ALLOC continuations=$CONT, LP-004 reconfigure_for_ic=$LP004"

# ---- preflight: smoke yaml has (K+1)*6 horizon + prediction_duration_days -
python -c "
from pathlib import Path
from sfno_inference_5410.preflight import assert_yaml_horizon
assert_yaml_horizon(Path('$YAML'), $K)
" || { echo "FATAL: yaml horizon preflight failed for $YAML" >&2; exit 2; }
echo "[preflight] yaml-horizon OK: ensemble_inference_hours=$(((K + 1) * 6)) prediction_duration_days=$(python -c 'print(('"$K"' + 1) * 6 / 24.0)')"

# ---- launch (in-process, --limit-ics 1 → just Y=121 s=0) ------------------
# The orchestrator does its own assert_output_dir_empty preflight + the
# postflight assert_output_dir_complete (which checks time-dim==K+1 +
# variable set). No need to invoke long_inference.py directly.
echo "[smoke] $(date) — starting 1-IC in-process smoke, forecast_K=$K raw_steps=$((K + 1))"
set -x
time python -u "$REPO_ROOT/scripts/eval_inference_5410.py" \
    --run-root "$RUN_ROOT" \
    --config-dir "$RUN_ROOT/inference" \
    --K "$K" \
    --years "$SMOKE_Y" \
    --limit-ics 1 \
    --launch
set +x

# ---- post-flight schema dump (§B.4 / §J.2) --------------------------------
SAVE_BASE="Y${SMOKE_Y}_s$(printf '%04d' $SMOKE_S)"
OUT_DIR="$RUN_ROOT/inference/upstream_raw"
NC_FILE=$(ls "$OUT_DIR"/${SAVE_BASE}_member*_y*.nc 2>/dev/null | head -1 || true)
if [[ -n "$NC_FILE" ]]; then
    echo "[post] ncdump -h $NC_FILE"
    ncdump -h "$NC_FILE" 2>&1 | head -100 \
        || python -c "import xarray as xr; ds = xr.open_dataset('$NC_FILE'); print(ds); print('---'); print(list(ds.data_vars)); ds.close()" \
        || echo "[post] no ncdump available; xarray fallback also failed"
fi

echo "[smoke] DONE forecast_K=$K raw_steps=$((K + 1))"
