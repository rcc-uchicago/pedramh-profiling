#!/bin/bash
#SBATCH -J 5410_score
# CPU-only job (score_5410.py + climatology adapter are numpy/xarray;
# no CUDA imports). `skx` is the long-running CPU queue (no 2 h cap).
#SBATCH -p skx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o logs/5410_score_%j.out
#SBATCH -e logs/5410_score_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# 5410 NWP scoring SLURM (per docs/2026-05-08_sfno_5410_scoring_plan.md v4.4).
#
# Reads RUN_ROOT/inference/upstream_raw/*.nc (96 raw outputs from a
# completed inference job) plus the Derecho per-timestep h5 truth and the
# 5410 climatology. Adapter writes inference NetCDFs in score_nwp's
# canonical schema to OUT_ROOT/inference/nwp/, then score_nwp.py
# produces scorecard CSV + bias maps + applies the §D.6 sanity gate.
#
# Required env (set by submit_eval_5410.sh):
#   RUN_ROOT, OUT_ROOT, RUN_TAG, EVAL_SHA7, GROUP_SHA7, MODEL_SHA7, CKPT,
#   TRUTH_H5_DIR, CLIM_SRC, K
# Optional:
#   FORCE=1   — delete prior adapted NCs at OUT_ROOT/inference/nwp/
# ============================================================================

set -euo pipefail
REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
cd "$REPO_ROOT"
mkdir -p logs

source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Required env (from submit_eval_5410.sh).
: "${RUN_ROOT:?RUN_ROOT required}"
: "${OUT_ROOT:?OUT_ROOT required}"
: "${RUN_TAG:?RUN_TAG required}"
: "${K:=60}"
: "${TRUTH_H5_DIR:=/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/h5/sigma_data}"
: "${CLIM_SRC:=/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/baselines/climatology_proleptic_5410.nc}"
: "${CKPT:=/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0/results/SFNO/5410/checkpoints/ckpt_epoch_50.tar}"
: "${EVAL_SHA7:=}"
: "${GROUP_SHA7:=5410-v2.0}"
: "${MODEL_SHA7:=ckpt_epoch_50}"

FORCE_ARG=()
[[ "${FORCE:-0}" == "1" ]] && FORCE_ARG+=(--force)

echo "[5410_score] $(date) — start (RUN_TAG=$RUN_TAG)"
set -x
time python -u "$REPO_ROOT/scripts/score_5410.py" \
    --run-root "$RUN_ROOT" \
    --out-root "$OUT_ROOT" \
    --run-tag "$RUN_TAG" \
    --K "$K" \
    --truth-h5-dir "$TRUTH_H5_DIR" \
    --clim-src "$CLIM_SRC" \
    --ckpt-path "$CKPT" \
    --eval-sha7 "$EVAL_SHA7" \
    --data-sha7 "$GROUP_SHA7" \
    --train-sha7 "$MODEL_SHA7" \
    "${FORCE_ARG[@]}"
set +x
echo "[5410_score] $(date) — done"
