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
from parse_nsys import RANGE_NAMES as PHASES, _table_exists  # noqa: E402

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


def _union_ns(intervals):
    """Total time covered by [start, end) intervals, counting overlap once.

    A phase's SUMMED kernel duration is not its GPU occupancy: NCCL runs on its
    own stream, so `backward` overlaps itself and its sum exceeds its span. This
    project has already had to retract one conclusion built on sum-vs-union, so
    both are always reported side by side.
    """
    total = 0
    cur_s = cur_e = None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            total += 0 if cur_e is None else cur_e - cur_s
            cur_s, cur_e = s, e
        elif e > cur_e:
            cur_e = e
    return total + (cur_e - cur_s if cur_e is not None else 0)


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
    spans = collections.defaultdict(list)
    for launch_ts, exec_ts, dur, pid, name in cur.fetchall():
        intervals = by_pid.get(pid, ())
        phase = _enclosing(intervals, exec_ts if by_exec else launch_ts)
        cell = agg[(phase, short_label(name))]
        cell[0] += 1
        cell[1] += dur
        spans[(pid, phase)].append((exec_ts, exec_ts + dur))
    # Union per (rank, phase), then summed over ranks so it is comparable with
    # the summed column. Per-rank first: two ranks' kernels are on different
    # devices and must never be unioned together.
    union = collections.Counter()
    for (pid, phase), iv in spans.items():
        union[phase] += _union_ns(iv)
    return agg, dict(union)


def report(agg, union=None, kernel_regex=None, top=12, rank_steps=None):
    total_ns = sum(v[1] for v in agg.values())
    total_n = sum(v[0] for v in agg.values())
    by_phase = collections.defaultdict(lambda: [0, 0])
    for (phase, _), (n, ns) in agg.items():
        by_phase[phase][0] += n
        by_phase[phase][1] += ns
    rs = rank_steps or 1
    per = f"/{rank_steps} rank-steps" if rank_steps else ""
    print(f"{total_n:,} launches, {total_ns / 1e9:.3f} s GPU kernel time "
          f"(all ranks, pid-guarded join){per}\n")
    hdr = (f"{'phase':<16}{'launches':>10}{'% count':>9}{'sum_ms/rs':>11}"
           f"{'% of sum':>10}")
    if union:
        hdr += f"{'UNION_ms/rs':>13}{'% of union':>12}{'overlap':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    total_u = sum(union.values()) if union else 0
    for phase, (n, ns) in sorted(by_phase.items(), key=lambda kv: -kv[1][1]):
        line = (f"{phase:<16}{n:>10,}{100 * n / total_n:>8.1f}%"
                f"{ns / 1e6 / rs:>11.2f}{100 * ns / total_ns:>9.1f}%")
        if union:
            u = union.get(phase, 0)
            line += (f"{u / 1e6 / rs:>13.2f}{100 * u / total_u:>11.1f}%"
                     f"{100 * (1 - u / ns) if ns else 0:>8.1f}%")
        print(line)
    if union:
        print("\n  sum counts a kernel on every stream it runs on; UNION counts "
              "wall-clock\n  occupancy (per rank, then summed). Quote the UNION "
              "against a step time.")

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


def per_step_report(cur, by_pid, top_outliers=3):
    """Per-rank-step GPU time, to expose a regime change or a one-step stall.

    Every aggregate in a capture is a mean over its steps; a single stalled step
    can move a sub-total by double digits while leaving the phase shares almost
    unchanged. Print the spread and name the outliers rather than trusting the
    mean (plan item 5).
    """
    rows = list(cur.execute("""
        SELECT r.start, k.end - k.start, k.globalPid, s.value
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN CUPTI_ACTIVITY_KIND_KERNEL k
          ON k.correlationId = r.correlationId
         AND k.globalPid = (r.globalTid & ?)
        JOIN StringIds s ON s.id = k.demangledName
    """, (PID_MASK,)))
    # index each rank's phase windows so a launch maps to (step, phase)
    steps = collections.defaultdict(list)
    for pid, iv in by_pid.items():
        n = -1
        for start, end, name in iv:
            if name in ('data_prep', 'preprocess'):
                n += 1
            steps[pid].append((start, end, n, name))
    per = collections.defaultdict(lambda: collections.Counter())
    for ts, dur, pid, name in rows:
        arr = steps.get(pid, ())
        i = bisect.bisect_right(arr, (ts, float('inf'), 0, '')) - 1
        if i < 0 or not (arr[i][0] <= ts <= arr[i][1]):
            continue
        _, _, n, phase = arr[i]
        per[(pid, n)][phase] += dur
        per[(pid, n)]['TOTAL'] += dur
        if 'nccl' in short_label(name):
            per[(pid, n)]['nccl'] += dur
    tot = sorted(v['TOTAL'] / 1e6 for v in per.values())
    med = tot[len(tot) // 2]
    print(f"\n  {len(per)} rank-steps: median {med:.2f} ms, "
          f"min {tot[0]:.2f}, max {tot[-1]:.2f}, spread {tot[-1] / tot[0]:.3f}x")
    print(f"  first vs median: {sorted(per.items(), key=lambda kv: kv[0][1])[0][1]['TOTAL'] / 1e6:.2f} "
          f"vs {med:.2f} ms  ⇒ "
          + ("NO warmup regime" if abs(sorted(per.items(), key=lambda kv: kv[0][1])[0][1]['TOTAL'] / 1e6 / med - 1) < 0.03
             else "WARMUP REGIME PRESENT"))
    worst = sorted(per.items(), key=lambda kv: -kv[1]['TOTAL'])[:top_outliers]
    print(f"  worst {top_outliers} rank-steps (GPU kernel sum):")
    for (pid, n), c in worst:
        print(f"    step index {n:>3} pid {pid:#x}: {c['TOTAL'] / 1e6:8.2f} ms "
              f"({c['TOTAL'] / 1e6 / med:.2f}x median), of which nccl "
              f"{c['nccl'] / 1e6:7.2f} ms")
    excl = {k: v for k, v in per.items() if k[1] != worst[0][0][1]}
    if excl:
        s_all = sum(v['TOTAL'] for v in per.values()) / len(per)
        s_ex = sum(v['TOTAL'] for v in excl.values()) / len(excl)
        n_all = sum(v['nccl'] for v in per.values()) / len(per)
        n_ex = sum(v['nccl'] for v in excl.values()) / len(excl)
        print(f"  excluding step index {worst[0][0][1]} (all ranks): "
              f"total {s_all / 1e6:.2f} -> {s_ex / 1e6:.2f} ms/rank-step "
              f"({100 * (s_ex / s_all - 1):+.1f}%), "
              f"nccl {n_all / 1e6:.2f} -> {n_ex / 1e6:.2f} ms "
              f"({100 * (n_ex / n_all - 1):+.1f}%)")


def memcpy_report(cur, by_pid, rank_steps=None):
    """Memcpy/memset bandwidth, bucketed by size against the L2 capacity.

    A memcpy's DRAM traffic is `2 x bytes` ONLY if the transfer misses L2 --
    read the source, write the destination. A transfer that fits in L2 can be
    serviced without touching DRAM twice, and applying the 2x rule to it yields
    a bandwidth ABOVE the device peak, which is how you know the rule does not
    apply there. So the population above L2 is reported separately and is the
    only one whose bandwidth may be quoted.
    """
    rs = rank_steps or 1
    if not _table_exists(cur, 'CUPTI_ACTIVITY_KIND_MEMCPY'):
        print("  no MEMCPY table in this capture")
        return
    peak = l2 = None
    if _table_exists(cur, 'TARGET_INFO_GPU'):
        cols = {r[1] for r in cur.execute("PRAGMA table_info(TARGET_INFO_GPU)")}
        want = [c for c in ('memoryBandwidth', 'l2CacheSize', 'name') if c in cols]
        rows = list(cur.execute(f"SELECT {','.join(want)} FROM TARGET_INFO_GPU"))
        if rows:
            d = dict(zip(want, rows[0]))
            peak, l2 = d.get('memoryBandwidth'), d.get('l2CacheSize')
            print(f"  device: {d.get('name', '?')}  peak {peak / 1e9:.0f} GB/s  "
                  f"L2 {l2 / 2**20:.0f} MiB   (read from the capture)")
    KINDS = {1: 'H2D', 2: 'D2H', 8: 'D2D', 10: 'P2P'}

    print(f"\n{'kind':<6}{'n':>8}{'ms/rank-step':>14}{'bytes/rank-step':>18}"
          f"{'GB/s DRAM':>12}{'% peak':>9}")
    print("  " + "-" * 65)
    for ck, n, ns, b in cur.execute(
            "SELECT copyKind, COUNT(*), SUM(end-start), SUM(bytes) "
            "FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY 1 ORDER BY 3 DESC"):
        mult = 2 if ck in (8,) else 1  # D2D touches DRAM twice; H2D/D2H once
        bw = mult * b / (ns / 1e9)
        print(f"{KINDS.get(ck, ck):<6}{n:>8,}{ns / 1e6 / rs:>14.2f}"
              f"{_human(b / rs):>18}{bw / 1e9:>12.0f}"
              f"{100 * bw / peak if peak else 0:>8.0f}%")

    print(f"\n  D2D by transfer size (the 2x-DRAM rule only holds above L2):")
    print(f"  {'size':>12}{'n':>8}{'GB':>10}{'s':>9}{'GB/s':>10}{'% peak':>9}  note")
    for size, n, ns, b in cur.execute(
            "SELECT bytes, COUNT(*), SUM(end-start), SUM(bytes) "
            "FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=8 "
            "GROUP BY 1 ORDER BY 4 DESC"):
        bw = 2 * b / (ns / 1e9)
        pct = 100 * bw / peak if peak else 0
        note = ('SUB-L2 — 2x rule INVALID' if l2 and size <= l2 else 'above L2')
        if pct > 100:
            note += ', >100% of peak PROVES it'
        print(f"  {_human(size):>12}{n:>8,}{b / 1e9:>10.2f}{ns / 1e9:>9.4f}"
              f"{bw / 1e9:>10.1f}{pct:>8.1f}%  {note}")

    if l2:
        row = list(cur.execute(
            "SELECT COUNT(*), SUM(end-start), SUM(bytes) "
            "FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=8 AND bytes > ?",
            (l2,)))[0]
        n, ns, b = row
        tot_b = list(cur.execute("SELECT SUM(bytes) FROM "
                                 "CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=8"))[0][0]
        bw = 2 * b / (ns / 1e9)
        print(f"\n  >>> QUOTABLE: D2D above L2 = {n:,} copies, {b / 1e9:.2f} GB "
              f"({100 * b / tot_b:.1f}% of all D2D bytes), {bw / 1e9:.0f} GB/s = "
              f"{100 * bw / peak:.0f}% of peak")

        print(f"\n  per device, above L2 only:")
        for pid, dev, n, ns, b in cur.execute(
                "SELECT globalPid, deviceId, COUNT(*), SUM(end-start), SUM(bytes) "
                "FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=8 AND bytes > ? "
                "GROUP BY 1,2 ORDER BY 2", (l2,)):
            bw = 2 * b / (ns / 1e9)
            print(f"    dev{dev}  n={n:>6,}  {b / 1e9:>8.2f} GB  {bw / 1e9:>7.0f} GB/s"
                  f"  {100 * bw / peak:>5.1f}% of peak")

    # Which stream? Concurrency changes what the number means.
    print(f"\n  streams: ", end='')
    print(', '.join(f"{KINDS.get(ck, ck)} on stream {sid} (n={n:,})" for ck, sid, n
                    in cur.execute("SELECT copyKind, streamId, COUNT(*) FROM "
                                   "CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY 1,2 "
                                   "ORDER BY 3 DESC LIMIT 4")))

    if _table_exists(cur, 'CUPTI_ACTIVITY_KIND_MEMSET'):
        n, ns, b = list(cur.execute(
            "SELECT COUNT(*), SUM(end-start), SUM(bytes) FROM "
            "CUPTI_ACTIVITY_KIND_MEMSET"))[0]
        print(f"\n  memset: n={n:,}  {ns / 1e6 / rs:.2f} ms/rank-step  "
              f"{_human((b or 0) / rs)}/rank-step  (NOT in the kernel total either)")

    # Phase attribution, launch-time, with the by-exec sensitivity spelled out.
    for by_exec in (False, True):
        agg = collections.Counter()
        for ts, ets, ns, pid, ck in cur.execute(
                "SELECT r.start, m.start, m.end-m.start, m.globalPid, m.copyKind "
                "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                "JOIN CUPTI_ACTIVITY_KIND_MEMCPY m "
                "  ON m.correlationId = r.correlationId "
                " AND m.globalPid = (r.globalTid & ?)", (PID_MASK,)):
            ph = _enclosing(by_pid.get(pid, ()), ets if by_exec else ts)
            agg[(ph, KINDS.get(ck, ck))] += ns
        tag = 'KERNEL-EXEC time' if by_exec else 'LAUNCH time'
        print(f"\n  memcpy by phase, bucketed on {tag}:")
        for (ph, k), ns in agg.most_common(6):
            print(f"    {ph:<14}{k:<5}{ns / 1e6 / rs:>8.2f} ms/rank-step")


def _human(b):
    for u in ('B', 'KB', 'MB', 'GB', 'TB'):
        if b < 1024 or u == 'TB':
            return f"{b:.2f} {u}" if u != 'B' else f"{b:.0f} B"
        b /= 1024.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('sqlite')
    ap.add_argument('--kernel-regex', default=None,
                    help="e.g. 'direct_copy|conj' to split the plan §0d copies")
    ap.add_argument('--phases', default=None,
                    help='comma-separated NVTX names; default = the shared contract')
    ap.add_argument('--by-exec', action='store_true',
                    help='bucket by kernel execution time instead of launch time')
    ap.add_argument('--memcpy', action='store_true',
                    help='memcpy/memset bandwidth, bucketed against L2 capacity')
    ap.add_argument('--per-step', action='store_true',
                    help='per-rank-step series: warmup regime and stall outliers')
    ap.add_argument('--rank-steps', type=int, default=None,
                    help='normalise to this many rank-steps (default: derived)')
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
    # rank-steps = phase-window count per rank, so every table is per-rank-step.
    anchor = next((n for n in ('data_prep', 'preprocess')
                   if any(r[2] == n for v in by_pid.values() for r in v)), None)
    rank_steps = a.rank_steps or (
        sum(1 for v in by_pid.values() for r in v if r[2] == anchor) if anchor else None)
    agg, union = attribute(con.cursor(), by_pid, a.by_exec)
    report(agg, union, a.kernel_regex, a.top, rank_steps)
    if a.per_step:
        per_step_report(con.cursor(), by_pid)
    if a.memcpy:
        print()
        memcpy_report(con.cursor(), by_pid, rank_steps)


if __name__ == '__main__':
    main()
