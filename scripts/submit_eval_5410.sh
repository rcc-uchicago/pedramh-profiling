#!/bin/bash
# ============================================================================
# submit_eval_5410.sh — top-level driver for the 5410 NWP eval pipeline.
#
# Per docs/2026-05-08_sfno_5410_scoring_plan.md (v4.4). Mirrors the own-track
# scripts/submit_eval.sh: chains 4 SLURM jobs with afterok dependencies.
#
# === Two roots (Codex round-3 fix; round-7 hoisted def order) ===
#   RUN_ROOT  = prepared inference root (ic_source.json, IC NCs, yamls, ckpt
#               shim, and (after JOB_INF) inference/upstream_raw/). MUST be
#               prepared by `scripts/build_5410_yaml_override.py
#               --all-years --K 60 ...` + IC NC builder + ic_source.json
#               setup BEFORE running this driver (unless SKIP_INF=1).
#   OUT_ROOT  = per-eval scoring root. Created fresh per RUN_TAG.
#               Holds inference/nwp/ (adapted), baselines/, scores/,
#               report.md, figures/, provenance.txt.
#
# === Skip flags ===
#   SCORE_ONLY=1       — alias for SKIP_INF=1 SKIP_REP=1 SKIP_FIG=1
#   SKIP_INF=1         — score against an existing RUN_ROOT/inference/upstream_raw
#   SKIP_SCO=1         — skip scoring; e.g., re-render an existing scorecard
#   SKIP_REP=1         — skip report
#   SKIP_FIG=1         — skip figures
#   FORCE=1            — delete prior adapted NCs at OUT_ROOT/inference/nwp
#   BLOCKER_JOB_ID=NNN — chain inference after a non-pipeline job
#
# === Final artifacts on success ===
#   $OUT_ROOT/report.md   $OUT_ROOT/figures/   $OUT_ROOT/scores/
# ============================================================================

set -euo pipefail
REPO_ROOT="$HOME/projects/SFNO_Climate_Emulator"
cd "$REPO_ROOT"
mkdir -p logs

# === required env (defaults shown) =========================================
: "${UPSTREAM_REPO:=/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0}"
: "${RUN_ROOT:=/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval_5410/20260507_phase1_gate}"
: "${CKPT:=$UPSTREAM_REPO/results/SFNO/5410/checkpoints/ckpt_epoch_50.tar}"
: "${TRUTH_H5_DIR:=/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/h5/sigma_data}"
: "${CLIM_SRC:=/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/baselines/climatology_proleptic_5410.nc}"
: "${K:=60}"

# === SCORE_ONLY alias normalization (Codex round-6 fix #4) =================
if [[ "${SCORE_ONLY:-0}" == "1" ]]; then
    SKIP_INF=1
    SKIP_REP=1
    SKIP_FIG=1
fi

# === Compute SHAs / RUN_TAG / OUT_ROOT FIRST (Codex round-7 fix #1) ========
EVAL_SHA7=$(git rev-parse --short=7 HEAD)
GROUP_SHA7="$(git -C "$UPSTREAM_REPO" rev-parse --short=7 HEAD 2>/dev/null || echo 5410-v2.0)"
MODEL_SHA7="$(basename "$CKPT" .tar)"
DATE_STR=$(date +%Y%m%d)
: "${RUN_TAG:=${DATE_STR}_eval-${EVAL_SHA7}_5410-${GROUP_SHA7}_${MODEL_SHA7}}"
: "${OUT_ROOT:=$WORK2/SFNO_Climate_Emulator/results/sfno_eval_5410/$RUN_TAG}"
mkdir -p "$OUT_ROOT" logs

count_matching_files() {
    local dir="$1"
    local pattern="$2"
    if [[ ! -d "$dir" ]]; then
        printf '0\n'
        return 0
    fi
    find "$dir" -maxdepth 1 -type f -name "$pattern" | wc -l
}

# === RUN_ROOT precondition check (skip if SKIP_INF=1) ======================
if [[ "${SKIP_INF:-0}" != "1" ]]; then
    test -f "$RUN_ROOT/inference/ic_source.json" \
        || { echo "FATAL: $RUN_ROOT/inference/ic_source.json missing — RUN_ROOT not prepared" >&2; exit 2; }
    test -L "$RUN_ROOT/inference/SFNO/5410/checkpoints/ckpt_epoch_50.tar" \
        || { echo "FATAL: ckpt symlink shim missing under $RUN_ROOT" >&2; exit 2; }
    for Y in 121 122 123 124 125 126 127 128; do
        test -f "$RUN_ROOT/inference/SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y${Y}.yaml" \
            || { echo "FATAL: per-Y yaml missing for Y=$Y under $RUN_ROOT" >&2; exit 2; }
    done

    # Codex round-5 fix #3: refuse to launch inference into a populated
    # upstream_raw. Saves queue position vs failing inside the orchestrator.
    if [[ -d "$RUN_ROOT/inference/upstream_raw" ]]; then
        n_nc=$(count_matching_files "$RUN_ROOT/inference/upstream_raw" 'Y*_member*_y*.nc')
        if [[ "$n_nc" -gt 0 ]]; then
            echo "FATAL: $RUN_ROOT/inference/upstream_raw is non-empty ($n_nc prior NetCDFs)." >&2
            echo "  Either:" >&2
            echo "    (a) Pass SKIP_INF=1 to score against the existing outputs." >&2
            echo "    (b) Backup/delete the existing upstream_raw before launch." >&2
            echo "    (c) Use a fresh RUN_ROOT." >&2
            exit 2
        fi
    fi
else
    BAD_PRE_FIX_RUN_ROOT="/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval_5410/20260507_phase1_gate"
    if [[ "$RUN_ROOT" == "$BAD_PRE_FIX_RUN_ROOT" && "${ALLOW_BAD_5410_RAW:-0}" != "1" ]]; then
        echo "FATAL: SKIP_INF=1 points at known pre-fix 5410 raw outputs:" >&2
        echo "  $RUN_ROOT" >&2
        echo "Those NetCDFs used nc_bc_offset=18 and produce invalid ACC/RMSE." >&2
        echo "Use a fresh post-fix RUN_ROOT, rerun inference, or set ALLOW_BAD_5410_RAW=1 only for forensic debugging." >&2
        exit 2
    fi
    n_nc=$(count_matching_files "$RUN_ROOT/inference/upstream_raw" 'Y*_member*_y*.nc')
    [[ "$n_nc" -eq 96 ]] \
        || { echo "FATAL: SKIP_INF=1 but $RUN_ROOT/inference/upstream_raw has $n_nc files (expect 96)" >&2; exit 2; }
fi

# === OUT_ROOT/inference/nwp rerun safety (Codex round-7 fix #3) ============
# FORCE=1 actively deletes prior adapted NCs (was a soft bypass in v4.3).
if [[ "${SKIP_SCO:-0}" != "1" ]]; then
    if [[ -d "$OUT_ROOT/inference/nwp" ]]; then
        n_adapted=$(count_matching_files "$OUT_ROOT/inference/nwp" '*.nc')
        if [[ "$n_adapted" -gt 0 ]]; then
            if [[ "${FORCE:-0}" == "1" ]]; then
                echo "[driver] FORCE=1: deleting $n_adapted prior adapted NCs at $OUT_ROOT/inference/nwp/"
                rm -f "$OUT_ROOT/inference/nwp/"*.nc
            else
                echo "FATAL: $OUT_ROOT/inference/nwp is non-empty ($n_adapted prior adapted NCs)." >&2
                echo "  score_nwp.py would silently include these in the scorecard." >&2
                echo "  Either:" >&2
                echo "    (a) Pass FORCE=1 to delete the prior set and rebuild from raw." >&2
                echo "    (b) Delete \$OUT_ROOT/inference/nwp/*.nc manually." >&2
                echo "    (c) Use a fresh OUT_ROOT (different RUN_TAG)." >&2
                exit 2
            fi
        fi
    fi
fi

# === provenance.txt ========================================================
cat > "$OUT_ROOT/provenance.txt" <<EOF
RUN_TAG=$RUN_TAG
RUN_ROOT=$RUN_ROOT
OUT_ROOT=$OUT_ROOT
EVAL_SHA7=$EVAL_SHA7
GROUP_SHA7=$GROUP_SHA7
MODEL_SHA7=$MODEL_SHA7
CKPT=$CKPT
UPSTREAM_REPO=$UPSTREAM_REPO
TRUTH_H5_DIR=$TRUTH_H5_DIR
CLIM_SRC=$CLIM_SRC
K=$K
SKIP_INF=${SKIP_INF:-0}
SKIP_SCO=${SKIP_SCO:-0}
SKIP_REP=${SKIP_REP:-0}
SKIP_FIG=${SKIP_FIG:-0}
FORCE=${FORCE:-0}
DATE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

cat <<EOF
[submit_eval_5410]
  RUN_ROOT     = $RUN_ROOT
  OUT_ROOT     = $OUT_ROOT
  RUN_TAG      = $RUN_TAG
  EVAL_SHA7    = $EVAL_SHA7
  GROUP_SHA7   = $GROUP_SHA7
  MODEL_SHA7   = $MODEL_SHA7
  CKPT         = $CKPT
  K            = $K
  SKIP         = INF=${SKIP_INF:-0} SCO=${SKIP_SCO:-0} REP=${SKIP_REP:-0} FIG=${SKIP_FIG:-0}
  FORCE        = ${FORCE:-0}
EOF

export RUN_ROOT OUT_ROOT EVAL_SHA7 GROUP_SHA7 MODEL_SHA7 RUN_TAG \
       CKPT TRUTH_H5_DIR CLIM_SRC K UPSTREAM_REPO

# === conditional afterok chain via prev_job accumulator (round-3 fix #2) ===
prev_job=""

submit_sbatch() {
    local output job_id
    output="$(sbatch "$@")"
    printf '%s\n' "$output" >&2
    job_id="$(printf '%s\n' "$output" | awk '/^[0-9]+(;|$)/ {sub(/;.*/, "", $1); print $1; exit}')"
    if [[ -z "$job_id" ]]; then
        echo "ERROR: could not parse sbatch job id" >&2
        return 1
    fi
    printf '%s\n' "$job_id"
}

submit_with_dep() {
    local slurm="$1"
    local args=()
    if [[ -n "$prev_job" ]]; then
        args+=("--dependency=afterok:$prev_job")
    fi
    args+=(--parsable "$slurm")
    submit_sbatch "${args[@]}"
}

if [[ "${SKIP_INF:-0}" != "1" ]]; then
    if [[ -n "${BLOCKER_JOB_ID:-}" ]]; then
        prev_job="$BLOCKER_JOB_ID"
    fi
    JOB_INF=$(submit_with_dep scripts/submit_eval_inference_5410.slurm)
    echo "[submit_eval_5410] inference job: $JOB_INF (deps: ${prev_job:-none})"
    prev_job="$JOB_INF"
fi

if [[ "${SKIP_SCO:-0}" != "1" ]]; then
    JOB_SCO=$(submit_with_dep scripts/submit_eval_score_5410.slurm)
    echo "[submit_eval_5410] scoring   job: $JOB_SCO (deps: ${prev_job:-none})"
    prev_job="$JOB_SCO"
fi

if [[ "${SKIP_REP:-0}" != "1" ]]; then
    JOB_REP=$(submit_with_dep scripts/submit_eval_report_5410.slurm)
    echo "[submit_eval_5410] report    job: $JOB_REP (deps: ${prev_job:-none})"
    prev_job="$JOB_REP"
fi

if [[ "${SKIP_FIG:-0}" != "1" ]]; then
    JOB_FIG=$(submit_with_dep scripts/submit_eval_figures_5410.slurm)
    echo "[submit_eval_5410] figures   job: $JOB_FIG (deps: ${prev_job:-none})"
fi

echo
echo "Final artifacts on success:"
echo "  $OUT_ROOT/report.md"
echo "  $OUT_ROOT/figures/"
echo "  $OUT_ROOT/scores/"
