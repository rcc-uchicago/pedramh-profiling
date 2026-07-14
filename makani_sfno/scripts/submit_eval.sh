#!/bin/bash
# submit_eval.sh — chain the four eval SLURM jobs with afterok dependencies.
#
# Per docs/sfno_eval_plan.md §G.3.
# Env-resolution + collision-guard logic lives in scripts/submit_eval_prelude.sh
# as the function `submit_eval_compute_env` (see plan §4.2). This script
# sources that prelude, calls it, and on success proceeds with the existing
# 4-job sbatch chain.
#
# Required env vars:
#   RUN_DIR      (default: v10 zgplev — $SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_full/plasim_sim52_zgplev_full/0)
#                For v9 sigma evals, override:
#                  RUN_DIR=$SCRATCH/SFNO_Climate_Emulator/runs/sfno_full/plasim_sim52_full/0
#   CKPT         (default: $RUN_DIR/training_checkpoints/best_ckpt_ema_mp0.tar
#                 when present, else best_ckpt_mp0.tar)
#   MODE         ('nwp' (default) or 'climate')
#   TEST_HOLDOUT (default: v10 — $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full/test_holdout)
#   TRAIN_DIR    (default: v10 — $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full/train)
#   PACKAGER_TEST_SRC (default: v10 — $SCRATCH/.../sim52_astro_64x128_zgplev/test)
#   ALLOW_RERUN  (set to 1 to reuse an existing $OUT_ROOT; default 0)
#                **Behaviour change (2026-05-20):** any existing $OUT_ROOT now
#                requires ALLOW_RERUN=1, not just CKPT-path mismatch.
#                See docs/2026-05-20_bundled_training_eval_plan.md §4.7.
#
# Usage:
#   scripts/submit_eval.sh
#   MODE=climate scripts/submit_eval.sh
#   ALLOW_RERUN=1 RUN_TAG=<existing> scripts/submit_eval.sh   # re-eval into existing dir

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/projects/SFNO_Climate_Emulator}"
cd "$REPO_ROOT"

# Resolve env (RUN_DIR, CKPT, RUN_TAG, OUT_ROOT, SHAs, provenance.txt, ...).
source "$REPO_ROOT/scripts/submit_eval_prelude.sh"
if ! submit_eval_compute_env; then
    rc=$?
    echo "[submit_eval] submit_eval_compute_env returned $rc — aborting" >&2
    exit "$rc"
fi

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
