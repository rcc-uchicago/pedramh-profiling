#!/bin/bash
#SBATCH -J 5410_inf96
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o logs/5410_inf96_%j.out
#SBATCH -e logs/5410_inf96_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# 5410 Phase-1 96-IC NWP eval — in-process orchestrator (v2.1, 2026-05-08).
#
# Builds upstream's Stepper ONCE in a single Python process, then loops
# over the 96 (Y, s) tuples calling Stepper.reconfigure_for_ic + predict
# per IC (LP-004). Mirrors the own-track architecture (scripts/eval_inference.py).
#
# Pre-2026-05-08 this submitted 96 separate subprocesses, paying ~95 s of
# cold-import + ckpt reload per IC (~4 h total). The new in-process
# orchestrator pays setup ONCE and finishes in ~50-80 min on 1×H100.
# Wallclock dropped from 4 h to 2 h (~1 h estimate + 1 h margin).
#
# Refactor plan: docs/2026-05-08_sfno_5410_inproc_orchestrator_plan.md (v2.1).
#
# Pre-conditions (must be set up before submission):
#   * <RUN_ROOT>/inference/ic_source.json present (§3 P-7 gate).
#   * All 96 IC NCs at <RUN_ROOT>/inference/ic_nc/<Y>_<ssss>.nc.
#   * All 8 per-Y yamls at <RUN_ROOT>/inference/SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y<Y>.yaml.
#   * Ckpt symlink shim at <RUN_ROOT>/inference/SFNO/5410/checkpoints/ckpt_epoch_50.tar.
#
# Required env (or accept defaults):
#   RUN_ROOT  — eval run root.
#               Default: /work2/.../results/sfno_eval_5410/20260507_phase1_gate
# ============================================================================

set -euo pipefail

REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
UPSTREAM_REPO="/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0"
: "${RUN_ROOT:=/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval_5410/20260507_phase1_gate}"
# Canonical eval-track forecast horizon: K=60 forecast leads (raw_steps=61, raw_hours=366).
# Per-IC final_datetime = init + (K+1)*6h.
: "${K:=60}"

YEARS=(121 122 123 124 125 126 127 128)
OFFSETS=(0 122 244 366 488 610 732 854 976 1098 1220 1342)

cd "$REPO_ROOT"
mkdir -p logs

# ---- venv -----------------------------------------------------------------
source "$REPO_ROOT/.venv/bin/activate"

# ---- distributed env (single-rank, propagates to each subprocess) ---------
export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# AI-RES src on PYTHONPATH so the orchestrator's
# `from sfno_inference_5410.* import ...` resolves.
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# ---- pre-flight assertions -----------------------------------------------
echo "[preflight] RUN_ROOT=$RUN_ROOT"
test -f "$RUN_ROOT/inference/ic_source.json" \
    || { echo "FATAL: missing ic_source.json — §3 P-7 gate not run" >&2; exit 2; }

# All 8 per-Y yamls + ckpt shim.
SHIM="$RUN_ROOT/inference/SFNO/5410/checkpoints/ckpt_epoch_50.tar"
test -L "$SHIM" || { echo "FATAL: missing ckpt symlink shim $SHIM" >&2; exit 2; }
for Y in "${YEARS[@]}"; do
    YAML="$RUN_ROOT/inference/SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y${Y}.yaml"
    test -f "$YAML" || { echo "FATAL: missing yaml override $YAML" >&2; exit 2; }
done

# All 96 IC NCs (12 offsets × 8 years).
for Y in "${YEARS[@]}"; do
    for S in "${OFFSETS[@]}"; do
        IC_NC="$RUN_ROOT/inference/ic_nc/${Y}_$(printf '%04d' $S).nc"
        test -f "$IC_NC" || { echo "FATAL: missing IC NC $IC_NC" >&2; exit 2; }
    done
done
echo "[preflight] all 96 IC NCs + 8 yamls + ckpt shim present"

# ---- preflight: upstream long_inference.py is patched (LP-003 4+2 + LP-004) -------
ALLOC=$(grep -c 'min(next_year_jan1, self.params.final_datetime)' "$UPSTREAM_REPO/long_inference.py" || true)
CONT=$(grep -c 'current_datetime < self.params.final_datetime' "$UPSTREAM_REPO/long_inference.py" || true)
LP004=$(grep -c 'def reconfigure_for_ic' "$UPSTREAM_REPO/long_inference.py" || true)
if [[ "$ALLOC" -ne 4 || "$CONT" -ne 2 ]]; then
    echo "FATAL: upstream long_inference.py LP-003 patch incomplete: allocator markers=$ALLOC (expected 4), continuation markers=$CONT (expected 2)" >&2
    echo "  reapply hunks per docs/2026-05-04_makani_local_patches.md" >&2
    exit 2
fi
if [[ "$LP004" -ne 1 ]]; then
    echo "FATAL: upstream long_inference.py LP-004 patch missing: reconfigure_for_ic count=$LP004 (expected 1)" >&2
    echo "  reapply hunk per docs/2026-05-04_makani_local_patches.md (Stepper.reconfigure_for_ic)" >&2
    exit 2
fi
echo "[preflight] upstream patches OK: LP-003 allocators=$ALLOC continuations=$CONT, LP-004 reconfigure_for_ic=$LP004"

# ---- preflight: each per-Y yaml has the (K+1)*6 horizon, not 8760/8784 ----
EXPECTED_HOURS=$(( (K + 1) * 6 ))
EXPECTED_DAYS=$(python -c "print((${K} + 1) * 6 / 24.0)")
for Y in "${YEARS[@]}"; do
    YAML="$RUN_ROOT/inference/SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y${Y}.yaml"
    PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
        python -c "
from pathlib import Path
from sfno_inference_5410.preflight import assert_yaml_horizon
assert_yaml_horizon(Path('$YAML'), $K)
" || { echo "FATAL: yaml horizon preflight failed for Y=$Y ($YAML)" >&2; exit 2; }
done
echo "[preflight] yaml-horizon OK for all ${#YEARS[@]} years (ensemble_inference_hours=$EXPECTED_HOURS, prediction_duration_days=$EXPECTED_DAYS)"

mkdir -p "$RUN_ROOT/inference/upstream_raw"

# ---- launch (in-process orchestrator: Stepper ONCE + 96 reconfigure_for_ic) ------
# The orchestrator does its own assert_output_dir_empty preflight + the
# postflight assert_output_dir_complete (exact filenames + time-dim==K+1
# + variable set check). Don't pre-clean upstream_raw here — the
# orchestrator's preflight is the gate.
echo "[launch] $(date) — starting 96-IC in-process sweep, forecast_K=$K raw_steps=$((K + 1)) raw_hours=$EXPECTED_HOURS"
set -x
time python -u "$REPO_ROOT/scripts/eval_inference_5410.py" \
    --run-root "$RUN_ROOT" \
    --config-dir "$RUN_ROOT/inference" \
    --K "$K" \
    --launch
set +x
echo "[launch] $(date) — sweep complete"

# ---- post-flight inventory ------------------------------------------------
OUT_DIR="$RUN_ROOT/inference/upstream_raw"
N_NC=$(find "$OUT_DIR" -maxdepth 1 -type f -name 'Y*_s*_member000_y*.nc' | wc -l)
echo "[post] upstream_raw NC count: $N_NC (expected 96)"
if [[ "$N_NC" -ne 96 ]]; then
    echo "[post] WARNING: expected 96 NCs, got $N_NC — check log for failed ICs" >&2
fi
du -sh "$OUT_DIR" 2>&1 | head -1

echo "[5410_inf96] DONE"
