#!/bin/bash
# bundled_eval.sh — run the full eval pipeline inline after training, on the
# same SLURM allocation, to avoid paying the h100 queue wait twice.
#
# Spec: docs/2026-05-20_bundled_training_eval_plan.md §4.4.
#
# Pre-conditions (caller sets these):
#   REPO_ROOT          (defaults to $HOME/projects/SFNO_Climate_Emulator)
#   BUNDLED_EVAL       0|1 — opt-in flag (see plan §4.8 for per-submit defaults)
#   JOB_START_EPOCH    epoch seconds, captured BEFORE the training step
#   RUN_DIR            $EXP_DIR/$CONFIG_NAME/$RUN_NUM (per plan §4.1)
#
# Post-conditions:
#   BUNDLED_EVAL_STATUS  string status code (read by status-mail body)
#     OK
#     SKIP_DISABLED
#     SKIP_NO_RUN_DIR
#     SKIP_NO_NEW_CKPT
#     FAIL_PRELUDE_<n>
#     FAIL_INFERENCE | FAIL_SCORE | FAIL_REPORT | FAIL_FIGURES
#   logs/bundled_eval_status_${SLURM_JOB_ID}.txt   (always; breadcrumb)
#   $OUT_ROOT/bundled_eval_status.txt              (only after prelude succeeds)
#
# This helper NEVER returns non-zero on eval failure. Training success must
# remain visible via the training SLURM job's exit code.

_bundled_log_fallback() {
    local repo="${REPO_ROOT:-$HOME/projects/SFNO_Climate_Emulator}"
    local f="$repo/logs/bundled_eval_status_${SLURM_JOB_ID:-noslurm}.txt"
    mkdir -p "$repo/logs" 2>/dev/null || true
    echo "$1" >> "$f"
}

bundled_eval_maybe_run() {
    local REPO_ROOT="${REPO_ROOT:-$HOME/projects/SFNO_Climate_Emulator}"
    local ema ema_mtime rc
    _bundled_log_fallback "ts=$(date -u +%Y-%m-%dT%H:%M:%SZ) job=${SLURM_JOB_ID:-noslurm}"

    if [ "${BUNDLED_EVAL:-0}" != "1" ]; then
        echo "[bundled-eval] BUNDLED_EVAL=${BUNDLED_EVAL:-0} — skip"
        BUNDLED_EVAL_STATUS="SKIP_DISABLED"
        export BUNDLED_EVAL_STATUS
        _bundled_log_fallback "status=$BUNDLED_EVAL_STATUS"
        return 0
    fi

    if [ -z "${RUN_DIR:-}" ]; then
        echo "[bundled-eval] RUN_DIR unset (submit script must export it) — skip"
        BUNDLED_EVAL_STATUS="SKIP_NO_RUN_DIR"
        export BUNDLED_EVAL_STATUS
        _bundled_log_fallback "status=$BUNDLED_EVAL_STATUS"
        return 0
    fi

    ema="$RUN_DIR/training_checkpoints/best_ckpt_ema_mp0.tar"
    ema_mtime=$(stat -c %Y "$ema" 2>/dev/null || echo 0)
    if [ "$ema_mtime" -le "${JOB_START_EPOCH:-0}" ]; then
        echo "[bundled-eval] EMA ckpt not refreshed this run (ema_mtime=$ema_mtime job_start=${JOB_START_EPOCH:-0}) — skip"
        BUNDLED_EVAL_STATUS="SKIP_NO_NEW_CKPT"
        export BUNDLED_EVAL_STATUS
        _bundled_log_fallback "status=$BUNDLED_EVAL_STATUS ema=$ema mtime=$ema_mtime job_start=${JOB_START_EPOCH:-0}"
        return 0
    fi

    # Errexit-safe: capture rc, do not let set -e abort us mid-helper.
    # Per plan §4.4, each helper-invoked command uses
    #   if cmd; then rc=0; else rc=$?; fi
    # because `cmd; rc=$?` under set -e exits before rc is captured.
    source "$REPO_ROOT/scripts/submit_eval_prelude.sh"
    if submit_eval_compute_env; then rc=0; else rc=$?; fi
    if [ "$rc" -ne 0 ]; then
        echo "[bundled-eval] submit_eval_compute_env returned $rc — skip"
        BUNDLED_EVAL_STATUS="FAIL_PRELUDE_$rc"
        export BUNDLED_EVAL_STATUS
        _bundled_log_fallback "status=$BUNDLED_EVAL_STATUS"
        return 0
    fi

    # From here, $OUT_ROOT exists and the per-OUT_ROOT status file is canonical.
    local status_file="$OUT_ROOT/bundled_eval_status.txt"
    : > "$status_file"
    echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status_file"
    echo "slurm_job_id=${SLURM_JOB_ID:-noslurm}" >> "$status_file"
    echo "run_dir=$RUN_DIR" >> "$status_file"
    echo "ema_mtime=$ema_mtime" >> "$status_file"
    echo "job_start_epoch=${JOB_START_EPOCH:-0}" >> "$status_file"
    _bundled_log_fallback "status=STARTED out_root=$OUT_ROOT"

    if bash "$REPO_ROOT/scripts/eval_run_inference_inline.sh"; then rc=0; else rc=$?; fi
    echo "inference_rc=$rc" >> "$status_file"
    if [ "$rc" -ne 0 ]; then
        BUNDLED_EVAL_STATUS="FAIL_INFERENCE"
        export BUNDLED_EVAL_STATUS
        _bundled_log_fallback "status=$BUNDLED_EVAL_STATUS out_root=$OUT_ROOT"
        return 0
    fi

    if bash "$REPO_ROOT/scripts/eval_run_score_inline.sh"; then rc=0; else rc=$?; fi
    echo "score_rc=$rc" >> "$status_file"
    if [ "$rc" -ne 0 ]; then
        BUNDLED_EVAL_STATUS="FAIL_SCORE"
        export BUNDLED_EVAL_STATUS
        _bundled_log_fallback "status=$BUNDLED_EVAL_STATUS out_root=$OUT_ROOT"
        return 0
    fi

    if bash "$REPO_ROOT/scripts/eval_run_report_inline.sh"; then rc=0; else rc=$?; fi
    echo "report_rc=$rc" >> "$status_file"
    if [ "$rc" -ne 0 ]; then
        BUNDLED_EVAL_STATUS="FAIL_REPORT"
        export BUNDLED_EVAL_STATUS
        _bundled_log_fallback "status=$BUNDLED_EVAL_STATUS out_root=$OUT_ROOT"
        return 0
    fi

    if bash "$REPO_ROOT/scripts/eval_run_figures_inline.sh"; then rc=0; else rc=$?; fi
    echo "figures_rc=$rc" >> "$status_file"
    if [ "$rc" -ne 0 ]; then
        BUNDLED_EVAL_STATUS="FAIL_FIGURES"
        export BUNDLED_EVAL_STATUS
        _bundled_log_fallback "status=$BUNDLED_EVAL_STATUS out_root=$OUT_ROOT"
        return 0
    fi

    echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status_file"
    BUNDLED_EVAL_STATUS="OK"
    export BUNDLED_EVAL_STATUS
    _bundled_log_fallback "status=OK out_root=$OUT_ROOT"
    return 0
}
