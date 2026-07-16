#!/bin/bash
# ============================================================================
# polaris_submit_chain.sh — submit N copies of a PBS script as a dependency chain,
# so a multi-day training run needs NO human monitoring or daily resubmission.
#
#     bash polaris_submit_chain.sh <count> <script.pbs> [extra qsub args...]
#
#     # Pangu full training, ~8.5 days => 3 x 72 h links:
#     cd PanguWeather/v2.0
#     bash ../../polaris_submit_chain.sh 3 HPC_scripts/polaris_train_e3sm_sfno_alldata_full.pbs
#
#     # with wandb live logging:
#     bash ../../polaris_submit_chain.sh 3 HPC_scripts/polaris_train_e3sm_sfno_alldata_full.pbs \
#         -v WANDB_MODE=online
#
# WHY: `#PBS -r y` only auto-requeues a PREEMPTED job. A job that reaches its walltime
# is killed, full stop — so a single submission of an 8-day run delivers exactly one
# 72 h slice and then waits for a human. This helper pre-submits the whole timeline:
# link i+1 carries `-W depend=afterany:<link i>` and starts only when link i has
# terminated for good (afterany fires on completion, walltime kill, OR final deletion —
# and NOT while a preempted link is merely requeued, since a requeue keeps the job id
# alive). Every link re-runs the same script from the top and resumes from the run's
# checkpoint — that is the launchers' existing contract (`-r y` requires it) — and a
# link that starts after training already completed simply resumes a finished run and
# exits within minutes. Over-provisioning the chain is therefore cheap; a too-short
# chain just means submitting another one later (the run dir is stable).
#
# RUN IT FROM THE DIRECTORY THE SCRIPT EXPECTS (the same place you would qsub from —
# e.g. PanguWeather/v2.0/ or physicsnemo_sfno/); qsub inherits this directory as
# $PBS_O_WORKDIR for every link.
#
# Limits (queried from PBS 2026-07-16): preemptable allows walltime <= 72:00:00,
# max 20 queued and 10 running jobs per user — so count <= 19 leaves room for other work.
#
# PASS = one "link i: <jobid>" line per link, and `qstat -u $USER` showing link 1
# Q/R and the rest H (held on the dependency). To cancel the whole chain:
#     qdel <every printed jobid>     (deleting link 1 alone RELEASES link 2 — afterany
#                                     fires on deletion too — so delete LAST link first,
#                                     or all in one qdel command.)
# ============================================================================
set -uo pipefail

if [ $# -lt 2 ]; then
    echo "usage: bash polaris_submit_chain.sh <count> <script.pbs> [extra qsub args...]"
    exit 2
fi
count="$1"; script="$2"; shift 2

case "${count}" in (*[!0-9]*|'') echo "ERROR CHAIN_COUNT_NOT_A_NUMBER: '${count}'"; exit 2;; esac
if [ "${count}" -lt 1 ] || [ "${count}" -gt 19 ]; then
    echo "ERROR CHAIN_COUNT_OUT_OF_RANGE: ${count} (1..19; preemptable caps 20 queued/user)"
    exit 2
fi
if [ ! -f "${script}" ]; then
    echo "ERROR CHAIN_SCRIPT_NOT_FOUND: ${script} (run this from the script's submission dir)"
    exit 2
fi

prev=""
ids=()
for i in $(seq 1 "${count}"); do
    if [ -z "${prev}" ]; then
        id=$(qsub "$@" "${script}")
    else
        id=$(qsub "$@" -W depend=afterany:"${prev}" "${script}")
    fi
    rc=$?
    if [ ${rc} -ne 0 ] || [ -z "${id}" ]; then
        echo "ERROR CHAIN_QSUB_FAILED at link ${i} (rc=${rc})."
        if [ ${#ids[@]} -gt 0 ]; then
            echo "  Links already submitted (delete LAST first, or all at once):"
            echo "    qdel ${ids[*]}"
        fi
        exit 2
    fi
    echo "link ${i}: ${id}${prev:+  (afterany ${prev})}"
    ids+=("${id}")
    prev="${id}"
done
echo "CHAIN_SUBMITTED (${count} links). Cancel the WHOLE chain with:"
echo "    qdel ${ids[*]}"
