#!/bin/bash
# submit_eval.sh — chain the three eval SLURM jobs with afterok dependencies.
#
# Per docs/sfno_eval_plan.md §G.3.
#
# Required env vars:
#   RUN_DIR      (default: v10 zgplev — $SCRATCH/AI-RES/runs/sfno_zgplev_full/plasim_sim52_zgplev_full/0)
#                For v9 sigma evals, override:
#                  RUN_DIR=$SCRATCH/AI-RES/runs/sfno_full/plasim_sim52_full/0
#   CKPT         (default: $RUN_DIR/training_checkpoints/best_ckpt_ema_mp0.tar
#                 when present, else best_ckpt_mp0.tar. EMA-best is canonical
#                 for EMA-enabled runs — see the eval-sfno-own skill.)
#   MODE         ('nwp' (default) or 'climate' — passed to inference job)
#   TEST_HOLDOUT (default: v10 — $SCRATCH/AI-RES/data/makani/sim52_zgplev_full/test_holdout)
#   TRAIN_DIR    (default: v10 — $SCRATCH/AI-RES/data/makani/sim52_zgplev_full/train)
#   PACKAGER_TEST_SRC (default: v10 — $SCRATCH/.../sim52_astro_64x128_zgplev/test;
#                used to read DATA_SHA7 and as auto-build src for TEST_HOLDOUT)
#
# Auto-derived:
#   EVAL_SHA7   — git SHA of AI-RES at submit time (--short=7).
#   DATA_SHA7   — read from any test h5 file's packager_git_sha attr.
#   TRAIN_SHA7  — preferred: $RUN_DIR/train_code_sha.txt (§G.5);
#                  fallback: grep "git hash:" $RUN_DIR/out.log;
#                  fallback: literal "unknown".
#   RUN_TAG     — composed from the SHAs and ckpt basename. Default template
#                 collapses redundant fields:
#                   - drops `_train-<sha>` when TRAIN_SHA7 == EVAL_SHA7
#                   - drops `_ckpt-<name>` when CKPT_BASENAME == "best_ckpt_mp0"
#                 Set FULL_RUN_TAG=1 to force the legacy 4-SHA + ckpt template.
#                 The full provenance is always written to
#                 $OUT_ROOT/provenance.txt regardless of which form is used.
#   OUT_ROOT    — $WORK2/AI-RES/results/sfno_eval/$RUN_TAG.
#
# Usage:
#   scripts/submit_eval.sh
#   MODE=climate scripts/submit_eval.sh

set -euo pipefail

REPO_ROOT="$HOME/AI-RES"
cd "$REPO_ROOT"

: "${RUN_DIR:=$SCRATCH/AI-RES/runs/sfno_zgplev_full/plasim_sim52_zgplev_full/0}"
# Prefer EMA-best (canonical for EMA-enabled runs); fall back to raw-best
# when EMA isn't available (legacy / EMA-disabled runs).
if [ -z "${CKPT:-}" ]; then
    if [ -s "$RUN_DIR/training_checkpoints/best_ckpt_ema_mp0.tar" ]; then
        CKPT="$RUN_DIR/training_checkpoints/best_ckpt_ema_mp0.tar"
    else
        CKPT="$RUN_DIR/training_checkpoints/best_ckpt_mp0.tar"
    fi
fi
: "${MODE:=nwp}"
: "${TEST_HOLDOUT:=$SCRATCH/AI-RES/data/makani/sim52_zgplev_full/test_holdout}"
: "${TRAIN_DIR:=$SCRATCH/AI-RES/data/makani/sim52_zgplev_full/train}"
: "${PACKAGER_TEST_SRC:=$SCRATCH/AI-RES/data/makani/sim52_astro_64x128_zgplev/test}"

# --- EVAL_SHA7 ---
if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git is required" >&2
    exit 2
fi
EVAL_SHA7="$(git rev-parse --short=7 HEAD)"

# --- DATA_SHA7 ---
TEST_FILE_FOR_SHA="$PACKAGER_TEST_SRC/MOST.0121.h5"
if [ ! -f "$TEST_FILE_FOR_SHA" ]; then
    echo "WARNING: $TEST_FILE_FOR_SHA not found; setting DATA_SHA7=unknown" >&2
    DATA_SHA7="unknown"
else
    DATA_SHA7="$(.venv/bin/python -c "
import h5py, sys
with h5py.File(sys.argv[1], 'r') as f:
    sha = f.attrs.get('packager_git_sha', b'unknown')
    if isinstance(sha, bytes):
        sha = sha.decode('utf-8')
    print(sha[:7])
" "$TEST_FILE_FOR_SHA")"
fi

# --- TRAIN_SHA7 ---
if [ -f "$RUN_DIR/train_code_sha.txt" ]; then
    TRAIN_SHA7="$(head -c 7 "$RUN_DIR/train_code_sha.txt")"
elif [ -f "$RUN_DIR/out.log" ]; then
    TRAIN_SHA7="$(grep -m1 "git hash:" "$RUN_DIR/out.log" | sed -E "s/.*git hash: b?'?([0-9a-f]{7}).*/\1/")"
    [ -z "$TRAIN_SHA7" ] && TRAIN_SHA7="unknown"
else
    TRAIN_SHA7="unknown"
fi

# --- RUN_TAG ---
# Auto-derive the training-run family name from RUN_DIR so two evals against
# different training runs at the same EVAL_SHA + DATA_SHA on the same day land
# in distinct OUT_ROOTs. The family name is the second-from-top directory under
# $SCRATCH/AI-RES/runs/, e.g.:
#   $SCRATCH/AI-RES/runs/sfno_zgplev_group_clone_v11/plasim_sim52_..._v11/0
#                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ family
TRAIN_FAMILY="$(basename "$(dirname "$(dirname "$RUN_DIR")")")"
DATE_STR="$(date +%Y%m%d)"
CKPT_BASENAME="$(basename "$CKPT" .tar)"
if [ "${FULL_RUN_TAG:-0}" = "1" ]; then
    : "${RUN_TAG:=${DATE_STR}_eval-${EVAL_SHA7}_data-${DATA_SHA7}_train-${TRAIN_SHA7}_family-${TRAIN_FAMILY}_ckpt-${CKPT_BASENAME}}"
else
    # Collapse redundant SHA fields, but ALWAYS include the train family so
    # two training runs (e.g. group_clone_v11 vs gbhpo40_gb16) at the same
    # eval/data/train SHA never share a RUN_TAG. The 2026-05-12 v11 ↔ gbhpo40
    # collision motivated this — see docs/2026-05-02_ema_implementation_plan.md
    # rollout notes and the eval-sfno-own skill §RUN_TAG-collision-guard.
    # Full SHAs are still recorded in provenance.txt below.
    _RT="${DATE_STR}_eval-${EVAL_SHA7}_data-${DATA_SHA7}"
    if [ "$TRAIN_SHA7" != "$EVAL_SHA7" ]; then
        _RT="${_RT}_train-${TRAIN_SHA7}"
    fi
    _RT="${_RT}_family-${TRAIN_FAMILY}"
    if [ "$CKPT_BASENAME" != "best_ckpt_mp0" ]; then
        _RT="${_RT}_ckpt-${CKPT_BASENAME}"
    fi
    : "${RUN_TAG:=${_RT}}"
    unset _RT
fi

# --- OUT_ROOT ---
: "${OUT_ROOT:=$WORK2/AI-RES/results/sfno_eval/$RUN_TAG}"

# Collision guard: if $OUT_ROOT/provenance.txt already exists and records a
# different CKPT than the one we're about to evaluate, abort with a loud
# message and require the user to pass an explicit RUN_TAG override (or
# clear/move the existing dir). This prevents the failure mode where two
# chained eval submissions race to the same OUT_ROOT and the second job's
# scoring stage reads stale inference NCs from the first — exactly what
# happened on 2026-05-12 between the v11 EMA chain and the gbhpo40 chain.
if [ -f "$OUT_ROOT/provenance.txt" ]; then
    existing_ckpt="$(grep -m1 '^CKPT=' "$OUT_ROOT/provenance.txt" | cut -d= -f2-)"
    if [ -n "$existing_ckpt" ] && [ "$existing_ckpt" != "$CKPT" ]; then
        cat >&2 <<EOF
[submit_eval] FATAL: $OUT_ROOT/provenance.txt already records a different CKPT.
  existing : $existing_ckpt
  requested: $CKPT

This RUN_TAG is reserved by a prior eval chain for a different checkpoint.
To proceed, either:
  - Pass RUN_TAG=<unique-name> on the command line, OR
  - Move/remove the existing $OUT_ROOT directory.

(This guard exists because the 2026-05-12 v11 EMA / gbhpo40 chain collision
silently mixed inference + scorecard data across runs. See the eval-sfno-own
skill §RUN_TAG-collision-guard.)
EOF
        exit 3
    fi
fi

mkdir -p "$OUT_ROOT" logs

# --- Provenance sidecar (always full, regardless of RUN_TAG form) ---
cat > "$OUT_ROOT/provenance.txt" <<EOF
RUN_TAG=$RUN_TAG
EVAL_SHA7=$EVAL_SHA7
DATA_SHA7=$DATA_SHA7
TRAIN_SHA7=$TRAIN_SHA7
TRAIN_FAMILY=$TRAIN_FAMILY
CKPT=$CKPT
CKPT_BASENAME=$CKPT_BASENAME
RUN_DIR=$RUN_DIR
MODE=$MODE
TEST_HOLDOUT=$TEST_HOLDOUT
TRAIN_DIR=$TRAIN_DIR
PACKAGER_TEST_SRC=$PACKAGER_TEST_SRC
DATE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

# Echo the resolved provenance.
cat <<EOF
[submit_eval]
  RUN_DIR           = $RUN_DIR
  CKPT              = $CKPT
  MODE              = $MODE
  TEST_HOLDOUT      = $TEST_HOLDOUT
  TRAIN_DIR         = $TRAIN_DIR
  PACKAGER_TEST_SRC = $PACKAGER_TEST_SRC
  EVAL_SHA7         = $EVAL_SHA7
  DATA_SHA7         = $DATA_SHA7
  TRAIN_SHA7        = $TRAIN_SHA7
  TRAIN_FAMILY      = $TRAIN_FAMILY
  RUN_TAG           = $RUN_TAG
  OUT_ROOT          = $OUT_ROOT
EOF

# BENCHMARK_5410_OUT_ROOT — group SFNO-5410 results to overlay in the report
# scorecard table and figures. Defaults to the H100 + packed-env settled valid
# 96-IC run; set to empty to disable the overlay entirely (figures and report
# render own-only with a loud warning when missing).
: "${BENCHMARK_5410_OUT_ROOT:=/work2/11114/zhixingliu/stampede3/AI-RES/results/sfno_eval_5410/20260509_blocking_96ic_h100_packed_derecho_env_valid}"

export RUN_DIR CKPT MODE TEST_HOLDOUT TRAIN_DIR PACKAGER_TEST_SRC \
       EVAL_SHA7 DATA_SHA7 TRAIN_SHA7 RUN_TAG OUT_ROOT \
       BENCHMARK_5410_OUT_ROOT

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

# Optional: chain the inference job after a prior SLURM job completes
# successfully (e.g. wait for training to finish before evaluating). Set
# BLOCKER_JOB_ID to that job id; downstream score/report jobs will inherit
# the wait via their own afterok-on-inference dependency.
INF_DEP_ARGS=()
if [[ -n "${BLOCKER_JOB_ID:-}" ]]; then
    INF_DEP_ARGS+=(--dependency=afterok:"$BLOCKER_JOB_ID")
    echo "[submit_eval] inference will wait for afterok:$BLOCKER_JOB_ID"
fi

JOB_INF="$(submit_sbatch --parsable "${INF_DEP_ARGS[@]}" scripts/submit_eval_inference.slurm)"
echo "[submit_eval] inference   job: $JOB_INF"

JOB_SCO="$(submit_sbatch --parsable --dependency=afterok:$JOB_INF scripts/submit_eval_score.slurm)"
echo "[submit_eval] scoring     job: $JOB_SCO  (afterok:$JOB_INF)"

JOB_REP="$(submit_sbatch --parsable --dependency=afterok:$JOB_SCO scripts/submit_eval_report.slurm)"
echo "[submit_eval] report      job: $JOB_REP  (afterok:$JOB_SCO)"

JOB_FIG="$(submit_sbatch --parsable --dependency=afterok:$JOB_REP scripts/submit_eval_figures.slurm)"
echo "[submit_eval] figures     job: $JOB_FIG  (afterok:$JOB_REP)"

echo
echo "Final artifacts on success:"
echo "  $OUT_ROOT/report.md"
echo "  $OUT_ROOT/figures/"
