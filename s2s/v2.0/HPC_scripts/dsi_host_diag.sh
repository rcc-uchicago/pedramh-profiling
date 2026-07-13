#!/usr/bin/env bash
#
# dsi_host_diag.sh — host diagnostic for the S2S CPU-to-GPU investigation.
#
# Captures the read-only §6 diagnostic block from v2.0/bench_report.md, 
#   1. the PyTorch / CUDA version line printed nothing (wrong interpreter / torch
#      not importable in that shell) — now tries python3 then python and shows the error;
#   2. numactl was not installed — now falls back to sysfs + lscpu for NUMA topology;
#   3. /proc/interrupts was dumped unfiltered — now filtered to the NVIDIA GPU IRQs
#      so we can see which CPUs field GPU completion interrupts;
#   4. the CPU frequency governor was not shown — it turned out to be "powersave",
#      a handoff-latency suspect, so it now gets its own section.
#
# Safe to run: read-only, normal user account, finishes in well under a minute.
# All output is echoed to the screen AND saved to a timestamped text file.
#
# Usage:
#   bash dsi_host_diag.sh                 # auto-named output file
#   bash dsi_host_diag.sh my_output.txt   # explicit output file
#
# Send the resulting .txt back for comparison against Midway's known-good baseline
# (driver 535.216.03 / CUDA 12.2 / PyTorch 2.6.0+cu124, intel_idle capped at C2).

# Output file: explicit arg, else hostname + timestamp.
host="$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown)"
OUT="${1:-dsi_host_diag_${host}_$(date +%Y%m%d_%H%M%S).txt}"

# ---- small helpers -------------------------------------------------------------
pf()  { printf '\n========== %s ==========\n' "$*"; }
# run a command line (pipes allowed), echoing it first; tolerate failure.
run() { printf '\n$ %s\n' "$*"; eval "$* 2>&1" || printf '(command failed, rc=%s)\n' "$?"; }
have() { command -v "$1" >/dev/null 2>&1; }

# Everything below runs inside main() 
main() {
  pf "Host identity, kernel, uptime"
  run "date"
  run "hostname"
  run "uname -a"
  run "uptime"
  run "who"          # who else is on the node (co-tenant contention was a candidate)

  pf "GPU identity: driver + driver-supported CUDA version"
  run "nvidia-smi | head -3"
  run "nvidia-smi --query-gpu=driver_version --format=csv"
  run "nvidia-smi --query-gpu=name,vbios_version --format=csv"  # exact H200 variant + firmware

  pf "GPU-to-GPU links + CPU/NUMA affinity"
  run "nvidia-smi topo -m"

  pf "GPU current state + co-tenant processes"
  run "nvidia-smi"
  run "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv"

  pf "PyTorch + the CUDA version it was built for"
  # The original one-liner produced no output on DSI. Try python3 then python,
  # and surface any import error instead of swallowing it.
  pybin=""
  have python3 && pybin=python3
  [ -z "$pybin" ] && have python && pybin=python
  if [ -n "$pybin" ]; then
    run "$pybin -c 'import torch; print(\"torch\", torch.__version__, \"| cuda\", torch.version.cuda, \"| cudnn\", torch.backends.cudnn.version())'"
  else
    echo "(no python3/python on PATH — activate the env used to run training, then re-run)"
  fi

  pf "CPU summary (lscpu)"
  run "lscpu | head -25"

  pf "NUMA topology"
  if have numactl; then
    run "numactl --hardware"
  else
    echo "(numactl not installed — falling back to lscpu + sysfs)"
    run "lscpu | grep -i numa"
    for n in /sys/devices/system/node/node*; do
      [ -e "$n" ] || continue
      printf '%s cpulist: %s\n' "$(basename "$n")" "$(cat "$n/cpulist" 2>/dev/null)"
    done
    echo "-- node distance matrix --"
    cat /sys/devices/system/node/node*/distance 2>/dev/null
  fi

  pf "CPU frequency governor (per-CPU values, summarized)"
  # 'powersave' here lets cores idle at min clock and pay a clock-up latency per dispatch.
  if ls /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null 2>&1; then
    cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c
  else
    echo "(scaling_governor not exposed in sysfs)"
  fi
  have cpupower && run "cpupower frequency-info | head -20"

  pf "CPU idle / C-states"
  run "cat /sys/module/intel_idle/parameters/max_cstate 2>/dev/null"   # Midway test partition = 2
  have cpupower && run "cpupower idle-info | head -40"

  pf "GPU completion IRQ steering (which CPUs field the GPU interrupts)"
  # Filtered to NVIDIA this time, not a blind head of /proc/interrupts.
  if grep -iqE 'nvidia' /proc/interrupts; then
    # column header (CPU0 CPU1 ...) then the nvidia lines
    head -1 /proc/interrupts
    grep -iE 'nvidia' /proc/interrupts
  else
    echo "(no 'nvidia' lines in /proc/interrupts — GPU may use MSI labelled differently;"
    echo " showing any line whose tail mentions gpu/nvidia)"
    awk 'NR==1 || tolower($0) ~ /nvidia|gpu/' /proc/interrupts
  fi

  pf "Kernel messages: NVIDIA / tpfottle / c-state warnings"
  run "dmesg 2>/dev/null | grep -iE 'nvidia|tpfottle|c-state|thermal' | tail -30"
  # dmesg is often root-only; note it if it produced nothing useful.

  pf "DONE"
  echo "Saved to: $OUT"
}

main 2>&1 | tee "$OUT"
