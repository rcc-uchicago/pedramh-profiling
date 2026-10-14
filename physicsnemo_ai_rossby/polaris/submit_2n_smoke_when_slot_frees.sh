#!/bin/bash
# One-shot helper: get the 2-node smoke queued once a `debug` slot frees.
#
# WHY IT EXISTS.  Polaris limits a user to ONE job per queue in Q/H state
# (measured 2026-08-27: `debug-scaling` Q + `debug` H is accepted, a second
# `debug` job is refused with "would exceed queue generic's per-user limit of
# jobs in 'Q' state").  The 4-node probe and the 1-node smoke already occupy
# both slots, so the 2-node smoke cannot be submitted until the 1-node one
# leaves the queue.
#
# It retries `qsub` — it does NOT poll `qstat` (CLAUDE.md forbids a qstat loop on
# a login node).  One sleeping shell, at most MAX_TRIES short-lived qsub calls
# 15 minutes apart, then it gives up and says so rather than looping forever.
# The submission carries `-W depend=afterany:$DEP`, so even if a slot frees early
# the 2-node arm still cannot overlap the 1-node one and contend for nodes.
#
# Usage:  bash polaris/submit_2n_smoke_when_slot_frees.sh <dependency-jobid>
set -u

DEP="${1:?usage: $0 <jobid the 2n smoke must follow>}"
AR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAX_TRIES="${MAX_TRIES:-10}"
GAP="${GAP:-900}"

cd "${AR_ROOT}" || exit 2
for i in $(seq 1 "${MAX_TRIES}"); do
    out=$(qsub -W "depend=afterany:${DEP}" -l select=2:system=polaris \
              polaris/polaris_ai_rossby_multinode_scaling.pbs 2>&1)
    rc=$?
    echo "[$(date -Iseconds)] try ${i}/${MAX_TRIES}: rc=${rc} ${out}"
    if [ ${rc} -eq 0 ]; then
        echo "SMOKE_2N_QUEUED ${out}"
        exit 0
    fi
    sleep "${GAP}"
done
echo "ERROR SMOKE_2N_NOT_QUEUED after ${MAX_TRIES} tries — submit it by hand:"
echo "  cd ${AR_ROOT} && qsub -l select=2:system=polaris polaris/polaris_ai_rossby_multinode_scaling.pbs"
exit 1
