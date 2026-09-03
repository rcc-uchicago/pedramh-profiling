#!/bin/bash
# ============================================================================
# run_ace2_ladder.sh — submit the ACE2 weak-scaling ladder as ONE dependency chain.
#
#     bash ACE2_retrain/polaris/run_ace2_ladder.sh [REPS] [LOCAL_BATCH]
#
# Defaults: 3 reps, LOCAL_BATCH=2.
#
# WHY A CHAIN AND NOT PARALLEL SUBMISSION
#   ⚠ **Two ACE2 arms running at once would contend for the SAME Lustre OST.**
#   The whole 2.4 TB store is `lmm_stripe_count: 1`, and the app-free probe (job
#   7587664) measured per-reader throughput collapsing past 8 concurrent readers:
#   21.4 -> 20.2 (8) -> 13.8 (16) -> 10.7 (32) MB/s. Overlapping two rungs would
#   put each one's I/O inside the other's measurement. Serial is not tidiness; it
#   is the experiment.
#
# ⚠ MEASURED 2026-09-02: A SINGLE CHAIN IS NOT SUBMITTABLE, and the handoff's
#   note that dependent jobs "sit in H and do not count" is WRONG for these
#   queues. A second `debug` link held on `-W depend=afterany:` is refused with
#       qsub: would exceed queue generic's per-user limit of jobs in 'Q' state
#   so held-on-dependency jobs DO count against `max_queued = [u:PBS_GENERIC=1]`.
#   `debug` and `debug-scaling` have SEPARATE limits, so the most that can be
#   pre-submitted is **one link per queue** — and this script therefore emits one
#   WAVE at a time: a `debug` rung plus a `debug-scaling` rung chained behind it,
#   which stays inside both limits and still never overlaps. Re-run it when the
#   wave drains; it picks up where the CSV left off.
#
# INTERLEAVED, NOT BATCHED: the order is 1n,2n,4n,8n, 1n,2n,4n,8n, ... — never
# all reps of one rung together. Two runs of an IDENTICAL config once measured
# 42.2% vs 37.4% for the same quantity (CHANGELOG §4.4c), so a rung whose reps
# are adjacent in time has its spread confounded with drift.
#
# WEAK SCALING: LOCAL_BATCH is held fixed and fme's GLOBAL `batch_size` is derived
# as `LOCAL_BATCH * 4 * nodes` by the launcher, so per-GPU work is constant.
# At LOCAL_BATCH=2 the 2-node rung is exactly the production global batch of 16 —
# which is also the smallest world size that FITS it (local batch 3 OOMs at
# 38.2 GiB of 39.49; jobs 7586496/7586506/7586526).
#
# ⚠ ≥8 NODES CARRIES A SPARE. `-l select=9 -v TARGET_NODES=8` prunes a sick node
# rather than losing the allocation: three different nodes handed to consecutive
# debug-scaling jobs have had a GPU stuck in 'CUDA-capable device(s) is/are busy
# or unavailable', and the preflight earned its keep on its first production use
# (48 healthy of 49).
#
# PASS = one `ACE2_POLARIS_TRAIN_OK nodes=N` per link, plus a row per link in
# $MEMBER_ROOT/bench/ace2_polaris_scaling.csv.
#
# To cancel: `qdel` every printed id IN ONE COMMAND (deleting a link alone
# RELEASES its successor — afterany fires on deletion too).
# ============================================================================
set -uo pipefail

REPS="${1:-3}"
LOCAL_BATCH="${2:-2}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="ACE2_retrain/polaris/polaris_ace2_train.pbs"

cd "${REPO}" || exit 2
[ -f "${SCRIPT}" ] || { echo "ERROR SCRIPT_MISSING: ${REPO}/${SCRIPT}"; exit 2; }

# rung: "<nodes>:<queue>:<select>:<target_nodes or ->"
RUNGS=(
    "1:debug:1:-"
    "2:debug:2:-"
    "4:debug-scaling:5:4"
    "8:debug-scaling:9:8"
)

echo "=== ACE2 weak-scaling ladder ==="
echo "  reps=${REPS}  local_batch=${LOCAL_BATCH}  (global batch = ${LOCAL_BATCH} x 4 x nodes)"
echo "  chain: each link waits for the previous one to terminate (afterany)"
echo

submit() {  # submit <rep> <rung-spec> [dependency]; echoes the job id or ""
    local rep="$1" rung="$2" dep="${3:-}"
    local nodes queue sel target vars id
    IFS=: read -r nodes queue sel target <<< "${rung}"

    # ⚠ ONE `-v` flag with comma-separated pairs, never several. PBS resolves
    # duplicate keys LAST-WINS, and a second -v is the documented way to
    # silently discard a per-arm override (handoff §1f).
    vars="LOCAL_BATCH=${LOCAL_BATCH},REP=${rep}"
    [ "${target}" != "-" ] && vars="${vars},TARGET_NODES=${target}"

    local args=(-q "${queue}" -l "select=${sel}:system=polaris" -v "${vars}")
    [ -n "${dep}" ] && args+=(-W "depend=afterany:${dep}")

    id="$(qsub "${args[@]}" "${SCRIPT}" 2>&1)"
    if [[ "${id}" != *.polaris-pbs* ]]; then
        printf "  rep %d  %dn  %-13s REJECTED: %s\n" "${rep}" "${nodes}" "${queue}" "${id}" >&2
        echo ""
        return 1
    fi
    printf "  rep %d  %dn  %-13s select=%-2s %-26s -> %s\n" \
        "${rep}" "${nodes}" "${queue}" "${sel}" \
        "$( [ "${target}" != "-" ] && echo "(spare: TARGET_NODES=${target})" || echo "")" \
        "${id%%.*}" >&2
    echo "${id}"
}

# ⚠ RESUME FROM THE CSV, do not blindly restart at rep 1. The first version of
# this script skipped only what was in flight, so re-running it after a wave
# drained re-submitted rungs that already had rows (it queued a third 1n and a
# second 4n when 2n and 8n were the missing ones). Count the rows already
# recorded for each rung at this LOCAL_BATCH and emit the SHORTEST rungs first,
# so the ladder fills out evenly rather than deepening one rung.
CSV="${ACE2_SCALING_CSV:-${MEMBER_ROOT:-/eagle/projects/lighthouse-uchicago/members/mehta5}/bench/ace2_polaris_scaling.csv}"
declare -A HAVE
for rung in "${RUNGS[@]}"; do
    nodes="${rung%%:*}"
    HAVE[${nodes}]=0
done
if [ -f "${CSV}" ]; then
    while IFS=, read -r jobid nodes ranks lb rest; do
        [ "${jobid}" = "jobid" ] && continue
        [ "${lb}" != "${LOCAL_BATCH}" ] && continue
        # a row with no n_steps is a failed/OOM arm and does not count as a rep
        case " ${!HAVE[*]} " in *" ${nodes} "*) HAVE[${nodes}]=$(( ${HAVE[${nodes}]} + 1 )) ;; esac
    done < <(awk -F, 'NR==1 || $10!=""' "${CSV}")
fi
echo "  rows already recorded at local_batch=${LOCAL_BATCH}:"
for rung in "${RUNGS[@]}"; do
    nodes="${rung%%:*}"
    printf "    %dn: %d/%d\n" "${nodes}" "${HAVE[${nodes}]}" "${REPS}"
done

# Emit in (fewest-rows-first, then node count) order so the gaps close first.
ORDER=()
for pass in $(seq 0 $(( REPS - 1 ))); do
    for rung in "${RUNGS[@]}"; do
        nodes="${rung%%:*}"
        [ "${HAVE[${nodes}]}" -gt "${pass}" ] && continue
        ORDER+=("$(( pass + 1 ))|${rung}")
    done
done
if [ ${#ORDER[@]} -eq 0 ]; then
    echo
    echo "  ladder COMPLETE at ${REPS} reps per rung — nothing to submit."
    exit 0
fi

# Skip anything already in flight in each queue -- re-running this script must be
# safe, and PBS will reject rather than queue a second one anyway.
# ⚠ USE `qselect`, NOT `qstat`. Both `qstat -u` AND `qstat -u -w` truncate the
# Job ID column with a trailing `*` (measured 2026-09-02), and feeding that to
# `-W depend=afterany:` fails with "illegal -W value". `qselect` prints full ids,
# one per line, which is exactly what a dependency needs.
_dbg="$(qselect -u "${USER}" -q debug 2>/dev/null)"
_scl="$(qselect -u "${USER}" -q debug-scaling 2>/dev/null)"
_inflight="$(printf '%s\n%s\n' "${_dbg}" "${_scl}" | sed '/^$/d')"
in_debug=$(echo "${_dbg}" | sed '/^$/d' | wc -l)
in_scale=$(echo "${_scl}" | sed '/^$/d' | wc -l)
echo "  already in flight: debug=${in_debug}  debug-scaling=${in_scale}"

ids=()
# ⚠ Chain this wave BEHIND anything already running, not alongside it. Without
# this the new link starts immediately and its I/O lands inside the in-flight
# arm's measurement -- the exact contention the serial design exists to prevent.
prev="$(echo "${_inflight}" | tail -n1)"
[ -n "${prev}" ] && echo "  chaining behind in-flight job ${prev%%.*}"
echo
for entry in "${ORDER[@]}"; do
    rep="${entry%%|*}"; rung="${entry#*|}"
    queue="$(echo "${rung}" | cut -d: -f2)"
    case "${queue}" in
        debug)         [ "${in_debug}" -gt 0 ] && continue; in_debug=1 ;;
        debug-scaling) [ "${in_scale}" -gt 0 ] && continue; in_scale=1 ;;
    esac
    id="$(submit "${rep}" "${rung}" "${prev}")" || break
    ids+=("${id}")
    prev="${id}"
    [ "${in_debug}" -gt 0 ] && [ "${in_scale}" -gt 0 ] && break
done

echo
echo "submitted ${#ids[@]} link(s) this wave — re-run this script when they drain"
echo "watch:   qstat -u \$USER"
echo "results: \${MEMBER_ROOT}/bench/ace2_polaris_scaling.csv"
[ ${#ids[@]} -gt 0 ] && echo "cancel:  qdel ${ids[*]}"
