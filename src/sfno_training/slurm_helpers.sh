#!/bin/bash

# Shared helpers for the single-rank SFNO Slurm launchers.

sfno_job_id() {
    printf '%s\n' "${SLURM_JOB_ID:-manual}"
}

sfno_job_name() {
    printf '%s\n' "${SFNO_JOB_NAME:-${SLURM_JOB_NAME:-sfno_job}}"
}

sfno_log_dir() {
    printf '%s\n' "${SFNO_LOG_DIR:-${REPO_ROOT:-$HOME/AI-RES}/logs}"
}

sfno_mail_log() {
    printf '%s/%s_%s.mail.log\n' "$(sfno_log_dir)" "$(sfno_job_name)" "$(sfno_job_id)"
}

sfno_send_status_mail() {
    local status="$1"
    local rc="${2:-0}"
    local job_name
    local job_id
    local log_dir
    local mail_log
    local subject
    local body_file
    local sent=1

    job_name="$(sfno_job_name)"
    job_id="$(sfno_job_id)"
    log_dir="$(sfno_log_dir)"
    mail_log="$(sfno_mail_log)"
    subject="${job_name} ${status}: job ${job_id}"

    mkdir -p "$log_dir"

    if [[ -z "${MAIL_TO:-}" ]]; then
        printf '[%s] MAIL_TO is empty; not sending %s mail\n' "$(date -Is)" "$status" >> "$mail_log"
        return 1
    fi

    body_file="$(mktemp "${TMPDIR:-/tmp}/${job_name}_${status}.XXXXXX")"
    {
        printf 'Job: %s\n' "$job_name"
        printf 'Job ID: %s\n' "$job_id"
        printf 'Status: %s\n' "$status"
        printf 'Exit code: %s\n' "$rc"
        printf 'Host: %s\n' "$(hostname -f 2>/dev/null || hostname)"
        printf 'Workdir: %s\n' "$(pwd)"
        printf 'Output: %s/logs/%s_%s.out\n' "${REPO_ROOT:-$HOME/AI-RES}" "$job_name" "$job_id"
        printf 'Error: %s/logs/%s_%s.err\n' "${REPO_ROOT:-$HOME/AI-RES}" "$job_name" "$job_id"
        printf 'Mail log: %s\n' "$mail_log"
        printf 'Time: %s\n' "$(date -Is)"
    } > "$body_file"

    {
        printf '[%s] Sending %s mail to %s; subject=%s\n' "$(date -Is)" "$status" "$MAIL_TO" "$subject"
        printf '[%s] PATH=%s\n' "$(date -Is)" "$PATH"
    } >> "$mail_log"

    if command -v mail >/dev/null 2>&1; then
        printf '[%s] Trying mail: %s\n' "$(date -Is)" "$(command -v mail)" >> "$mail_log"
        if mail -s "$subject" "$MAIL_TO" < "$body_file" >> "$mail_log" 2>&1; then
            printf '[%s] mail exited 0\n' "$(date -Is)" >> "$mail_log"
            sent=0
        else
            printf '[%s] mail failed with rc=%s\n' "$(date -Is)" "$?" >> "$mail_log"
        fi
    else
        printf '[%s] mail command not found\n' "$(date -Is)" >> "$mail_log"
    fi

    if [[ "$sent" -ne 0 ]] && command -v sendmail >/dev/null 2>&1; then
        printf '[%s] Trying sendmail: %s\n' "$(date -Is)" "$(command -v sendmail)" >> "$mail_log"
        if {
            printf 'To: %s\n' "$MAIL_TO"
            printf 'Subject: %s\n' "$subject"
            printf '\n'
            cat "$body_file"
        } | sendmail -t -oi >> "$mail_log" 2>&1; then
            printf '[%s] sendmail exited 0\n' "$(date -Is)" >> "$mail_log"
            sent=0
        else
            printf '[%s] sendmail failed with rc=%s\n' "$(date -Is)" "$?" >> "$mail_log"
        fi
    elif [[ "$sent" -ne 0 ]]; then
        printf '[%s] sendmail command not found\n' "$(date -Is)" >> "$mail_log"
    fi

    if [[ "$sent" -ne 0 ]] && command -v python >/dev/null 2>&1; then
        printf '[%s] Trying python SMTP relay: %s via %s\n' \
            "$(date -Is)" "$(command -v python)" "${SFNO_SMTP_RELAY:-129.114.112.1}" >> "$mail_log"
        if SFNO_MAIL_TO="$MAIL_TO" \
            SFNO_MAIL_SUBJECT="$subject" \
            SFNO_MAIL_BODY_FILE="$body_file" \
            SFNO_MAIL_FROM="${SFNO_MAIL_FROM:-${USER:-zhixingliu}@stampede3.tacc.utexas.edu}" \
            SFNO_SMTP_RELAY="${SFNO_SMTP_RELAY:-129.114.112.1}" \
            python - <<'PY' >> "$mail_log" 2>&1
import email.message
import os
import smtplib

message = email.message.EmailMessage()
message["From"] = os.environ["SFNO_MAIL_FROM"]
message["To"] = os.environ["SFNO_MAIL_TO"]
message["Subject"] = os.environ["SFNO_MAIL_SUBJECT"]
with open(os.environ["SFNO_MAIL_BODY_FILE"], encoding="utf-8") as handle:
    message.set_content(handle.read())

with smtplib.SMTP(os.environ["SFNO_SMTP_RELAY"], 25, timeout=20) as smtp:
    smtp.send_message(message)
PY
        then
            printf '[%s] python SMTP relay exited 0\n' "$(date -Is)" >> "$mail_log"
            sent=0
        else
            printf '[%s] python SMTP relay failed with rc=%s\n' "$(date -Is)" "$?" >> "$mail_log"
        fi
    elif [[ "$sent" -ne 0 ]]; then
        printf '[%s] python command not found\n' "$(date -Is)" >> "$mail_log"
    fi

    rm -f "$body_file"

    if [[ "$sent" -ne 0 ]]; then
        printf '[%s] WARNING: all mail commands failed for %s\n' "$(date -Is)" "$status" >> "$mail_log"
        return 1
    fi
    return 0
}

sfno_on_exit() {
    local rc=$?
    set +x
    if [[ "$rc" -eq 0 ]]; then
        sfno_send_status_mail "END" "$rc" || true
    else
        sfno_send_status_mail "FAIL" "$rc" || true
        sfno_send_status_mail "END_FAILED" "$rc" || true
    fi
    exit "$rc"
}

sfno_install_status_trap() {
    trap sfno_on_exit EXIT
}
