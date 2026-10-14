#!/bin/bash
# Drive the ai-rossby weak-scaling ladder: A/B/C/8n x 3 INTERLEAVED reps.
#
# WHY A DRIVER AT ALL.  Polaris allows a user ONE job per queue in Q/H state
# (measured 2026-08-27), so a whole `-W depend=` chain cannot be pre-submitted.
# Each link is submitted only once the previous one has left the queue. The
# driver retries `qsub` — it never polls `qstat` (CLAUDE.md forbids a qstat loop
# on a login node) — and each link also carries `-W depend=afterany:<prev>` so
# ORDER is enforced by PBS, not by the retry timing.
#
# INTERLEAVED, NOT BATCHED: A,B,C,8n,A,B,C,8n,A,B,C,8n. Rule #16 — two runs of
# an identical config once measured 42.2% vs 37.4% for the same quantity
# (CHANGELOG §4.4c), so three reps of one arm back-to-back cannot support a
# scaling claim. This is a validity requirement, not tidiness.
#
# ONE CONFIG FOR THE WHOLE TABLE (rule #4): NCCL_ALGO=Ring on every arm, OMP=1,
# per-batch TSV off, 60 steps, one store. Ring is REQUIRED, not a tuning choice:
# the default (tree) all-reduce silently corrupts and hangs above ~1 GB on this
# stack, and this model reduces 4.73 GB (jobs 7569817 vs 7569818).
#
# ⚠ Arm A rep 1 is submitted SEPARATELY (7569856) — it is already running under
# this config. Start this driver with the remaining 11 links.
#
# Usage:  nohup bash polaris/run_ai_rossby_ladder.sh [<jobid to start after>] &
set -u

AR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER=polaris/polaris_ai_rossby_multinode_scaling.pbs
DEP="${1:-}"
GAP="${GAP:-300}"
MAX_TRIES="${MAX_TRIES:-60}"

cd "${AR_ROOT}" || exit 2

# arm | queue | select | extra -v   ('-' = none)
#   C and 8n request ONE SPARE NODE and prune sick ones: three distinct nodes
#   with zombie GPU state killed four makani runs on 2026-08-26, and a
#   fast-failing node returns straight to the free pool for the retry.
#   Mandatory at >=8 nodes (mistakes ledger #4).
LADDER=(
  "B|debug|2|-"
  "C|debug-scaling|5|TARGET_NODES=4"
  "8n|debug-scaling|9|TARGET_NODES=8"
)
REPS="${REPS:-3}"

submit() {   # submit <arm> <queue> <select> <extra> <rep>
    local arm="$1" q="$2" sel="$3" extra="$4" rep="$5"
    local vars="NCCL_ALGO=Ring,REP=${rep}"
    [ "${extra}" != "-" ] && vars="${vars},${extra}"
    local depend=()
    [ -n "${DEP}" ] && depend=( -W "depend=afterany:${DEP}" )
    local i out rc
    for i in $(seq 1 "${MAX_TRIES}"); do
        out=$(qsub -q "${q}" -l "select=${sel}:system=polaris" \
                   "${depend[@]}" -v "${vars}" "${LAUNCHER}" 2>&1)
        rc=$?
        if [ ${rc} -eq 0 ]; then
            DEP="${out%%.*}"
            echo "[$(date -Iseconds)] arm ${arm} rep ${rep} -> ${out}"
            return 0
        fi
        echo "[$(date -Iseconds)] arm ${arm} rep ${rep} queue full (try ${i}): ${out}"
        sleep "${GAP}"
    done
    echo "ERROR LADDER_SUBMIT_FAILED arm=${arm} rep=${rep} after ${MAX_TRIES} tries"
    return 1
}

echo "=== ai-rossby ladder: A/B/C/8n x ${REPS} interleaved, NCCL_ALGO=Ring ==="
echo "    starting after job ${DEP:-<none>}"
for rep in $(seq 1 "${REPS}"); do
    # Arm A joins from rep 2 onward: rep 1 was submitted by hand (7569856).
    if [ "${rep}" -gt 1 ]; then
        submit "A" debug 1 "-" "${rep}" || exit 1
    fi
    for entry in "${LADDER[@]}"; do
        IFS='|' read -r arm q sel extra <<< "${entry}"
        submit "${arm}" "${q}" "${sel}" "${extra}" "${rep}" || exit 1
    done
done
echo "LADDER_SUBMITTED all links queued; last job ${DEP}"
echo "  PASS per arm = AI_ROSSBY_MN_SCALING_OK + a row in"
echo "  \$MEMBER_ROOT/bench/ai_rossby_multinode_scaling.csv"
