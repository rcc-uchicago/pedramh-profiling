#!/bin/bash
#SBATCH -J 5410_inf_packed
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o logs/5410_inf_packed_%j.out
#SBATCH -e logs/5410_inf_packed_%j.err
#SBATCH --mail-user=zhixingliu@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# 5410 96-IC inference using the H100 + packed Derecho env path. This is the
# valid Stampede3 production runtime for the blocking emulator (settled
# 2026-05-09; matches Derecho z500 RMSE to roundoff). See
# docs/2026-05-09_sfno_5410_investigation_history.md for the trail.
#
# Required env (or accept defaults):
#   RUN_ROOT       — eval run root. Defaults to the script's hardcoded
#                    DEFAULT_RUN_ROOT (a fresh dated path is recommended).
#   K              — forecast leads per IC (default 60 = 15 days).
#   LIMIT_ICS      — set to 1 for a smoke run (1 IC instead of 96).
#   PACKED_PYTHON  — packed Derecho env interpreter. Default points at
#                    the unpacked aires_env_20260509 from
#                    /work2/.../artifacts/derecho_runtime/.
# ============================================================================

set -euo pipefail

REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
cd "$REPO_ROOT"
mkdir -p logs

: "${PACKED_PYTHON:=/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_runtime/aires_env_20260509/unpacked/bin/python}"
: "${K:=60}"

test -x "$PACKED_PYTHON" \
    || { echo "FATAL: packed Derecho env interpreter missing at $PACKED_PYTHON" >&2; exit 2; }

ARGS=( "scripts/infer_sfno5410_blocking_h100_packed.py" "--K" "$K" )
[[ -n "${RUN_ROOT:-}" ]] && ARGS+=( "--run-root" "$RUN_ROOT" )
[[ -n "${LIMIT_ICS:-}" ]] && ARGS+=( "--limit-ics" "$LIMIT_ICS" )
[[ "${FORCE:-0}" == "1" ]] && ARGS+=( "--force" )

echo "[submit_eval_inference_5410_packed] python=$PACKED_PYTHON"
echo "[submit_eval_inference_5410_packed] argv=${ARGS[*]}"

time "$PACKED_PYTHON" "${ARGS[@]}"
