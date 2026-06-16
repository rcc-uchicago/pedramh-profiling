#!/bin/bash
#SBATCH -J 5410_report
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o logs/5410_report_%j.out
#SBATCH -e logs/5410_report_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# 5410 report-rendering SLURM (per docs/2026-05-08_sfno_5410_scoring_plan.md v4.4).
#
# Invokes the existing scripts/render_eval_report.py (own-track renderer)
# with 5410-flavored provenance strings. The renderer reads the scorecard
# CSV + bias_maps_*.npy from OUT_ROOT/scores/ and writes OUT_ROOT/report.md.
#
# Required env (set by submit_eval_5410.sh):
#   OUT_ROOT, RUN_TAG, EVAL_SHA7, GROUP_SHA7, MODEL_SHA7, CKPT
# ============================================================================

set -euo pipefail
REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
cd "$REPO_ROOT"
mkdir -p logs

source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

: "${OUT_ROOT:?OUT_ROOT required}"
: "${RUN_TAG:?RUN_TAG required}"
: "${EVAL_SHA7:?EVAL_SHA7 required}"
: "${GROUP_SHA7:=5410-v2.0}"
: "${MODEL_SHA7:=ckpt_epoch_50}"
: "${CKPT:=/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0/results/SFNO/5410/checkpoints/ckpt_epoch_50.tar}"

echo "[5410_report] $(date) — start (RUN_TAG=$RUN_TAG)"
set -x
time python -u "$REPO_ROOT/scripts/render_eval_report.py" \
    --out-root "$OUT_ROOT" \
    --run-tag "$RUN_TAG" \
    --eval-sha7 "$EVAL_SHA7" \
    --data-sha7 "$GROUP_SHA7" \
    --train-sha7 "$MODEL_SHA7" \
    --ckpt-path "$CKPT"
set +x
echo "[5410_report] $(date) — done; report at $OUT_ROOT/report.md"
