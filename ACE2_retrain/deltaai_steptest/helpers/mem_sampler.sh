#!/bin/bash
# Page-cache / memory sampler for an ACE2 training job. Run in the BACKGROUND of the
# job script, on the compute node, before launching training:
#
#     bash mem_sampler.sh "${OUTDIR}/mem_samples.tsv" 60 &
#     SAMPLER=$!
#     ... srun/mpiexec training ...
#     kill ${SAMPLER}
#
# One TSV row per interval. Line the timestamps up with fme's epoch log lines
# ("Beginning epoch ...", "Time taken for epoch ...") and look at the epoch-1 -> 2
# boundary:
#   anon jumps + file squeezed + refault_file/events_max climbing  => memory pressure
#       evicting the data cache (re-reads / direct reclaim) -> the slowdown
#   file flat, no refaults, anon flat                              => not memory/cache
# n_python_procs = this user's python/dataloader processes on the node: a jump after epoch 1
# means worker pools that were not torn down. Works under Slurm or PBS, cgroup v1 or v2. Read-only; costs one `cat` per minute.
OUT="${1:-mem_samples.tsv}"; DT="${2:-60}"

# ---- find this job's memory cgroup ------------------------------------------
CG=""
if [ -f /proc/self/cgroup ]; then
    rel=$(awk -F: '$1=="0" {print $3}' /proc/self/cgroup)            # v2 unified
    [ -n "${rel}" ] && [ -f "/sys/fs/cgroup${rel}/memory.stat" ] && CG="/sys/fs/cgroup${rel}"
    if [ -z "${CG}" ]; then                                          # v1 memory controller
        rel=$(awk -F: '$2 ~ /(^|,)memory(,|$)/ {print $3}' /proc/self/cgroup)
        [ -n "${rel}" ] && [ -f "/sys/fs/cgroup/memory${rel}/memory.stat" ] && CG="/sys/fs/cgroup/memory${rel}"
    fi
    # the sampler may sit in a step/task child cgroup; walk up to the one with a real limit
    while [ -n "${CG}" ] && [ "${CG}" != "/sys/fs/cgroup" ]; do
        lim=$(cat "${CG}/memory.max" 2>/dev/null || cat "${CG}/memory.limit_in_bytes" 2>/dev/null)
        case "${lim}" in ""|max|9223372036854771712) CG=$(dirname "${CG}") ;; *) break ;; esac
    done
fi
echo "# cgroup=${CG:-none}  host=$(hostname)  interval=${DT}s" > "${OUT}"
printf 'time\tmeminfo_cached_gb\tmeminfo_avail_gb\tcg_limit_gb\tcg_current_gb\tcg_anon_gb\tcg_file_gb\tcg_refault_file\tcg_pgmajfault\tcg_events_high\tcg_events_max\tlustre_read_gb\tloadavg_1m\tn_python_procs\tn_cpus\n' >> "${OUT}"

g() { awk -v b="$1" 'BEGIN { if (b == "" || b == "max") print "NA"; else printf "%.2f", b / 1073741824 }'; }
while :; do
    cached=$(awk '/^Cached:/ {printf "%.2f", $2/1048576}' /proc/meminfo)
    avail=$(awk '/^MemAvailable:/ {printf "%.2f", $2/1048576}' /proc/meminfo)
    lim=""; cur=""; anon=""; file=""; ref=""; maj=""; eh=""; em=""
    if [ -n "${CG}" ]; then
        lim=$(cat "${CG}/memory.max" 2>/dev/null || cat "${CG}/memory.limit_in_bytes" 2>/dev/null)
        cur=$(cat "${CG}/memory.current" 2>/dev/null || cat "${CG}/memory.usage_in_bytes" 2>/dev/null)
        read -r anon file ref maj < <(awk '
            $1=="anon"||$1=="total_rss"   {a=$2}  $1=="file"||$1=="total_cache" {f=$2}
            $1=="workingset_refault_file"||$1=="workingset_refault" {r=$2}
            $1=="pgmajfault"||$1=="total_pgmajfault" {m=$2}
            END {print a+0, f+0, r+0, m+0}' "${CG}/memory.stat")
        read -r eh em < <(awk '$1=="high" {h=$2} $1=="max" {x=$2} END {print h+0, x+0}' "${CG}/memory.events" 2>/dev/null || echo "NA NA")
    fi
    lr=$(lctl get_param -n llite.*.stats 2>/dev/null | awk '$1=="read_bytes" {s+=$NF} END {if (NR) printf "%.1f", s/1073741824; else print "NA"}')
    la=$(awk '{print $1}' /proc/loadavg)
    np=$(ps -u "$(id -u)" -o comm= 2>/dev/null | awk '/python|pt_main|pt_data/ {n++} END {print n+0}')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "${cached}" "${avail}" \
        "$(g "${lim}")" "$(g "${cur}")" "$(g "${anon}")" "$(g "${file}")" "${ref:-NA}" "${maj:-NA}" \
        "${eh:-NA}" "${em:-NA}" "${lr:-NA}" "${la}" "${np}" "$(nproc)" >> "${OUT}"
    sleep "${DT}"
done
