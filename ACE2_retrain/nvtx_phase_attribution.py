#!/usr/bin/env python3
"""Attribute GPU kernel time to NVTX phase ranges, correctly.

    python3 nvtx_phase_attribution.py <nsys.sqlite> [--kernel-regex RE] [--top N]

Answers "how much of kernel X's GPU time is spent in forward vs backward",
which is the question `parse_nsys.py` (per-range wall time) and
`kernel_census.py` (per-range launch counts) both leave open.

Two things make this join wrong if done naively, and BOTH bit this project:

1. **`correlationId` is unique per PROCESS, not per capture.** A multi-rank
   capture holds one sqlite for all ranks, so the bare
   `RUNTIME.correlationId = KERNEL.correlationId` join cross-products the
   ranks. On `nsys_pangu_sfno_7255503.sqlite` (4 ranks) it returns 459,088
   rows for 354,720 kernels -- **+29.4% phantom** -- matching the +30.8%
   measured independently on the Midway ACE2 capture. The guard is
   `KERNEL.globalPid = RUNTIME.globalTid & PID_MASK`: an nsys `globalTid`
   is `globalPid | tid` with the tid in the low 24 bits, so masking those
   off recovers the pid exactly (asserted at load).

2. **The launching thread is NOT the thread the NVTX range is on.** PyTorch's
   autograd engine launches from its own worker thread, so a range pushed on
   the main thread never contains the backward launches by thread identity.
   On rank 0 of that same capture, 62,680 of 88,680 launches come from the
   autograd worker while all 201 NVTX events are on the main thread --
   thread-scoped attribution credits `(outside)` for the whole of `backward`.
   Attribution is therefore scoped to the **process**: the phase windows are
   non-overlapping per rank (asserted), so a launch timestamp inside a
   window belongs to that phase whichever thread issued it.

Attribution is by **launch time** (the RUNTIME call's start), not by kernel
execution time: it credits the phase that *requested* the work, which is the
causal reading and the one that survives CUDA being async. `--by-exec` gives
the other reading as a sensitivity check; if the two disagree materially, the
CPU is running far enough ahead that neither is a clean phase attribution and
that is itself the finding.

Per CLAUDE.md #10 the NVTX range names are a cross-project contract -- this
script reads them, it must never rename them.
"""
import argparse
import bisect
import collections
import os
import re
import sqlite3
import sys

# An nsys globalTid packs the tid into the low 24 bits of the globalPid.
PID_MASK = ~0xFFFFFF

# The shared NVTX phase contract, imported rather than copied -- a fourth copy
# is how `unstack` drifted out of one of parse_nsys.py's two lists. The
# per-step `step_N` markers are deliberately NOT in it: they ENCLOSE these and
# would trip the overlap check in load_phases().
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_nsys import RANGE_NAMES as PHASES  # noqa: E402

# Ordered: first match wins. Collapses PyTorch's templated names to something
# a table can hold, keeping the dtype -- `direct_copy` over complex64 and over
# float are different lines in the profile and must not merge.
_LABELS = (
    (r'direct_copy_kernel_cuda', 'direct_copy'),
    (r'conj_kernel_cuda', 'conj'),
    (r'FusedAdam', 'FusedAdam'),
    (r'ncclDevKernel_(\w+?)_', r'nccl_\1'),
    (r'cutlass_\w*?(\w+gemm)\w*', r'cutlass_\1'),
    (r'bn_fw_tr', 'bn_fw'),
    (r'bn_bw', 'bn_bw'),
    (r'MulFunctor', 'MulFunctor'),
    (r'CUDAFunctor_add', 'CUDAFunctor_add'),
    (r'fft\w*', 'cufft'),
)


def short_label(name):
    """A compact `kernel<dtype>` label, dtype kept because it distinguishes rows."""
    base = None
    for pat, repl in _LABELS:
        m = re.search(pat, name)
        if m:
            base = m.expand(repl) if '\\' in repl else repl
            break
    if base is None:
        base = re.sub(r'^void\s+', '', name).split('<')[0].split('(')[0]
        base = base.rsplit('::', 1)[-1][:34] or '?'
    if 'c10::complex<float>' in name or 'complex<float>' in name:
        dt = 'complex64'
    elif 'c10::BFloat16' in name:
        dt = 'bf16'
    elif 'c10::Half' in name:
        dt = 'fp16'
    elif re.search(r'\bfloat\b', name):
        dt = 'float'
    else:
        dt = None
    # A vectorized TensorIterator kernel and the nocast fallback are the whole
    # bandwidth story in plan §0d -- never collapse them together.
    tag = 'vec' if 'vectorized_elementwise_kernel' in name else (
        'nocast' if 'gpu_kernel_impl_nocast' in name else None)
    parts = [p for p in (dt, tag) if p]
    return f"{base}<{','.join(parts)}>" if parts else base


def load_phases(cur, names=PHASES):
    """{globalPid: [(start, end, name), ...]} sorted, from the inline `text` column.

    The house ranges live in NVTX_EVENTS.text, NOT behind textId -> StringIds:
    on the Pangu Polaris captures the textId path holds only NCCL's own
    registered strings, which is why the naive join looked like the harness
    had emitted nothing.
    """
    qmarks = ','.join('?' * len(names))
    cur.execute(
        f"SELECT text, start, end, globalTid FROM NVTX_EVENTS "
        f"WHERE text IN ({qmarks}) AND end IS NOT NULL AND end > start",
        names)
    by_pid = collections.defaultdict(list)
    for text, start, end, gtid in cur.fetchall():
        by_pid[gtid & PID_MASK].append((start, end, text))
    for pid in by_pid:
        by_pid[pid].sort()
        prev_end = -1
        for start, end, text in by_pid[pid]:
            if start < prev_end:
                # Nested or overlapping ranges would double-count; the caller
                # must narrow `names` rather than get a silently wrong table.
                sys.exit(f"ERROR overlapping NVTX ranges for pid {pid:#x} at "
                         f"{text} — narrow --phases to one non-nested level")
            prev_end = end
    return dict(by_pid)


def _enclosing(intervals, ts):
    i = bisect.bisect_right(intervals, (ts, float('inf'), '')) - 1
    if i >= 0 and intervals[i][0] <= ts <= intervals[i][1]:
        return intervals[i][2]
    return '(outside)'


def attribute(cur, by_pid, by_exec=False):
    """(phase, label) -> [n_launches, gpu_ns], via the pid-guarded join."""
    cur.execute("""
        SELECT r.start, k.start, k.end - k.start, k.globalPid, s.value
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN CUPTI_ACTIVITY_KIND_KERNEL k
          ON k.correlationId = r.correlationId
         AND k.globalPid = (r.globalTid & ?)
        JOIN StringIds s ON s.id = k.demangledName
    """, (PID_MASK,))
    agg = collections.defaultdict(lambda: [0, 0])
    for launch_ts, exec_ts, dur, pid, name in cur.fetchall():
        intervals = by_pid.get(pid, ())
        phase = _enclosing(intervals, exec_ts if by_exec else launch_ts)
        cell = agg[(phase, short_label(name))]
        cell[0] += 1
        cell[1] += dur
    return agg


def report(agg, kernel_regex=None, top=12):
    total_ns = sum(v[1] for v in agg.values())
    total_n = sum(v[0] for v in agg.values())
    by_phase = collections.defaultdict(lambda: [0, 0])
    for (phase, _), (n, ns) in agg.items():
        by_phase[phase][0] += n
        by_phase[phase][1] += ns
    print(f"{total_n:,} launches, {total_ns / 1e9:.3f} s GPU kernel time "
          f"(all ranks, pid-guarded join)\n")
    print(f"{'phase':<16}{'launches':>10}{'% count':>9}{'gpu_s':>10}{'% time':>8}")
    print("  " + "-" * 51)
    for phase, (n, ns) in sorted(by_phase.items(), key=lambda kv: -kv[1][1]):
        print(f"{phase:<16}{n:>10,}{100 * n / total_n:>8.1f}%"
              f"{ns / 1e9:>10.3f}{100 * ns / total_ns:>7.1f}%")

    if kernel_regex is None:
        return
    rx = re.compile(kernel_regex)
    sel = {k: v for k, v in agg.items() if rx.search(k[1])}
    sel_ns = sum(v[1] for v in sel.values())
    print(f"\n--- kernels matching /{kernel_regex}/ : {sel_ns / 1e9:.3f} s = "
          f"{100 * sel_ns / total_ns:.1f}% of all GPU kernel time ---\n")
    print(f"{'phase':<16}{'kernel':<30}{'launches':>10}{'gpu_s':>10}"
          f"{'% of sel':>10}{'% all gpu':>11}")
    print("  " + "-" * 85)
    for (phase, lab), (n, ns) in sorted(sel.items(), key=lambda kv: -kv[1][1])[:top]:
        print(f"{phase:<16}{lab:<30}{n:>10,}{ns / 1e9:>10.3f}"
              f"{100 * ns / sel_ns:>9.1f}%{100 * ns / total_ns:>10.1f}%")
    roll = collections.defaultdict(int)
    for (phase, _), (_, ns) in sel.items():
        roll[phase] += ns
    print(f"\n{'phase rollup':<16}{'gpu_s':>10}{'% of sel':>10}")
    print("  " + "-" * 34)
    for phase, ns in sorted(roll.items(), key=lambda kv: -kv[1]):
        print(f"{phase:<16}{ns / 1e9:>10.3f}{100 * ns / sel_ns:>9.1f}%")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('sqlite')
    ap.add_argument('--kernel-regex', default=None,
                    help="e.g. 'direct_copy|conj' to split the plan §0d copies")
    ap.add_argument('--phases', default=None,
                    help='comma-separated NVTX names; default = the shared contract')
    ap.add_argument('--by-exec', action='store_true',
                    help='bucket by kernel execution time instead of launch time')
    ap.add_argument('--top', type=int, default=12)
    a = ap.parse_args(argv)
    # mode=ro: a capture is a primary artifact and is never written to.
    con = sqlite3.connect(f"file:{a.sqlite}?mode=ro", uri=True)
    names = tuple(a.phases.split(',')) if a.phases else PHASES
    by_pid = load_phases(con.cursor(), names)
    if not by_pid:
        sys.exit("ERROR no NVTX phase ranges found in NVTX_EVENTS.text — "
                 "check the harness emitted them and --trace=nvtx was passed")
    print(f"{len(by_pid)} rank(s); phase ranges: "
          + ', '.join(f"{t}={sum(1 for v in by_pid.values() for r in v if r[2] == t)}"
                      for t in names
                      if any(r[2] == t for v in by_pid.values() for r in v))
          + (" [by KERNEL EXEC time]" if a.by_exec else " [by LAUNCH time]") + "\n")
    report(attribute(con.cursor(), by_pid, a.by_exec), a.kernel_regex, a.top)


if __name__ == '__main__':
    main()
