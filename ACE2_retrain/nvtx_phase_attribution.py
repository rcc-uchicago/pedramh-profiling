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

# Collectives with a distinguished ROOT, whose own time is ~0 regardless of skew.
# Parsed by name rather than pattern-matched, because rootedness does not follow
# from the substring: AllReduce and ReduceScatter are NOT rooted while Reduce is,
# and AllGather is NOT rooted while Gather is. A regex on `Reduce` gets AllReduce
# wrong, which silently empties the straggler ranking.
_ROOTED_OPS = frozenset(('broadcast', 'reduce', 'gather', 'scatter'))


def _is_rooted(kernel_name):
    m = re.search(r'ncclDevKernel_(\w+?)_', kernel_name) or \
        re.search(r'nccl\w*Kernel_?(\w+)', kernel_name)
    return bool(m) and m.group(1).lower() in _ROOTED_OPS

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
    """Per-rank-step GPU time, split COMPUTE vs NCCL, because they behave differently.

    Two distinct things inflate a step and they must not be conflated:

    * **compute warmup** — cudnn autotune, allocator growth, lazy init. Shows up in
      the non-NCCL kernels, decays over the first few steps, and is what a
      `bench_warmup` setting exists to skip.
    * **comms noise** — a rank arriving late makes every other rank *wait inside*
      an NCCL collective. Shows up only in NCCL, does not decay, and is not work.

    Judging warmup on the TOTAL confuses the two: on Pangu job 7255557 step 0 is
    **+6.7% on the median total** (685.09 vs 641.95 ms/rank-step), which clears a 3%
    threshold and reads as a warmup regime — but it is **+61.1% NCCL and only +0.5%
    compute**, so there is no compute warmup at all. Hence the verdict below keys on
    compute only, and comms is reported separately.
    (An earlier note here said "+20%". That was wrong: it compared ONE rank's step 0
    against the all-rank median. Fixed with the rest of this function.)
    """
    rows = list(cur.execute("""
        SELECT r.start, k.end - k.start, k.globalPid, s.value
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN CUPTI_ACTIVITY_KIND_KERNEL k
          ON k.correlationId = r.correlationId
         AND k.globalPid = (r.globalTid & ?)
        JOIN StringIds s ON s.id = k.demangledName
    """, (PID_MASK,)))
    steps = collections.defaultdict(list)
    for pid, iv in by_pid.items():
        n = -1
        for start, end, name in iv:
            if name in ('data_prep', 'preprocess'):
                n += 1
            steps[pid].append((start, end, n))
    comp = collections.Counter()
    comm = collections.Counter()
    for ts, dur, pid, name in rows:
        arr = steps.get(pid, ())
        i = bisect.bisect_right(arr, (ts, float('inf'), 0)) - 1
        if i < 0 or not (arr[i][0] <= ts <= arr[i][1]):
            continue
        n = arr[i][2]
        (comm if name.startswith('nccl') else comp)[n] += dur
    if not comp:
        print("  no attributable kernels for a per-step series")
        return
    nranks = max(1, len(by_pid))
    nsteps = max(comp) + 1

    def stats(counter):
        v = sorted(x / nranks / 1e6 for x in counter.values())
        return v[len(v) // 2], v[0], v[-1]

    c_med, c_min, c_max = stats(comp)
    n_med, n_min, n_max = stats(comm) if comm else (0, 0, 0)
    print(f"\n  {nsteps} steps x {nranks} ranks, ms/rank-step:")
    print(f"    {'COMPUTE (non-NCCL)':<20} median {c_med:8.2f}  min {c_min:8.2f}  "
          f"max {c_max:8.2f}  spread {c_max / c_min:5.3f}x")
    if comm:
        print(f"    {'NCCL':<20} median {n_med:8.2f}  min {n_min:8.2f}  "
              f"max {n_max:8.2f}  spread {n_max / n_min:5.1f}x")

    # Warmup verdict on COMPUTE only -- see the docstring.
    c0 = comp[0] / nranks / 1e6
    n0 = (comm[0] / nranks / 1e6) if comm else 0
    dc, dn = 100 * (c0 / c_med - 1), (100 * (n0 / n_med - 1) if n_med else 0)
    print(f"    step 0: compute {c0:.2f} ({dc:+.1f}% vs median), "
          f"nccl {n0:.2f} ({dn:+.1f}%)")
    print("    ⇒ " + ("COMPUTE WARMUP REGIME PRESENT — the warmup setting is too short"
                      if abs(dc) >= 3 else
                      "no compute warmup regime (the warmup setting was long enough)")
          + (f"; step 0's excess is COMMS, not warmup" if abs(dc) < 3 and dn > 10 else ""))

    worst = sorted(((comp[n] + comm[n], n) for n in comp), reverse=True)[:top_outliers]
    print(f"  worst {len(worst)} steps by total:")
    for tot, n in worst:
        print(f"    step {n:>3}: {tot / nranks / 1e6:8.2f} ms/rank-step  = compute "
              f"{comp[n] / nranks / 1e6:7.2f} + nccl {comm[n] / nranks / 1e6:7.2f}")
    if worst:
        drop = worst[0][1]
        keep = [n for n in comp if n != drop]
        f = lambda c, ks: sum(c[n] for n in ks) / len(ks) / nranks / 1e6
        allk = list(comp)
        print(f"  excluding step {drop}: compute {f(comp, allk):.2f} -> "
              f"{f(comp, keep):.2f} ms/rank-step "
              f"({100 * (f(comp, keep) / f(comp, allk) - 1):+.1f}%), "
              f"nccl {f(comm, allk):.2f} -> {f(comm, keep):.2f} "
              f"({100 * (f(comm, keep) / f(comm, allk) - 1) if f(comm, allk) else 0:+.1f}%)")
    print(f"\n  >>> COMPUTE-only total is the reproducible denominator: "
          f"{f(comp, allk):.2f} ms/rank-step. Shares taken against the FULL kernel\n"
          f"      total move run-to-run because NCCL wait sits in the denominator.")
    return {'compute_median_ms': c_med, 'nccl_median_ms': n_med,
            'compute_mean_ms': f(comp, allk), 'nccl_mean_ms': f(comm, allk),
            'step0_compute_pct': dc, 'step0_nccl_pct': dn,
            'compute_warmup': abs(dc) >= 3, 'n_steps': nsteps}


def per_rank_report(cur, by_pid, stall_factor=1.5):
    """Per-step, per-DEVICE NCCL vs compute — the direct straggler test.

    A collective makes every rank wait for the last arrival, so the straggler is
    the rank with the *lowest* NCCL time while the others are high: it did not
    wait, everyone waited for it. That signature is invisible in any aggregate
    that averages over ranks, which is why this is its own report.

    `stall_factor` counts steps whose mean NCCL exceeds this multiple of the
    per-step median — a stalled step is a balance event, not work.

    **Rooted collectives are excluded from the ranking, and that matters.** In a
    broadcast the ROOT does not wait by construction, so its time is near-zero on
    every step whether or not it is late: on both Pangu captures dev0's total
    `ncclBroadcast` is ~9 ms against 23-1884 ms for the non-roots. Ranking on
    all-NCCL therefore hands the root a spurious "always the straggler" credit.
    Arrival order is judged on the non-rooted collectives (all-reduce) only; the
    rooted total is reported separately so the root is visible rather than hidden.
    """
    steps = collections.defaultdict(list)
    for pid, iv in by_pid.items():
        n = -1
        for start, end, name in iv:
            if name in ('data_prep', 'preprocess'):
                n += 1
            steps[pid].append((start, end, n))
    comp = collections.defaultdict(int)
    comm = collections.defaultdict(int)      # non-rooted only -- see below
    rooted = collections.defaultdict(int)    # broadcast/gather: root is ~0 by design
    for ts, dur, pid, name in cur.execute("""
        SELECT r.start, k.end - k.start, k.globalPid, s.value
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN CUPTI_ACTIVITY_KIND_KERNEL k
          ON k.correlationId = r.correlationId
         AND k.globalPid = (r.globalTid & ?)
        JOIN StringIds s ON s.id = k.demangledName
    """, (PID_MASK,)):
        arr = steps.get(pid, ())
        i = bisect.bisect_right(arr, (ts, float('inf'), 0)) - 1
        if i < 0 or not (arr[i][0] <= ts <= arr[i][1]):
            continue
        key = (pid, arr[i][2])
        if not name.startswith('nccl'):
            comp[key] += dur
        elif _is_rooted(name):
            rooted[key] += dur
        else:
            comm[key] += dur
    if not comm:
        print("  no NCCL kernels — single-rank capture?")
        return
    pids = sorted({p for p, _ in comp})
    dev = {p: d for p, d in cur.execute(
        "SELECT DISTINCT globalPid, deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL")}
    nsteps = max(n for _, n in comp) + 1
    step_mean = {n: sum(comm[(p, n)] for p in pids) / len(pids)
                 for n in range(nsteps)}
    med = sorted(step_mean.values())[nsteps // 2]
    stalled = sorted(n for n in step_mean if step_mean[n] > stall_factor * med)
    vals = sorted(step_mean.values())
    print(f"\n  per-step mean NON-ROOTED NCCL: median {med / 1e6:.2f} ms, "
          f"min {vals[0] / 1e6:.2f}, max {vals[-1] / 1e6:.2f} "
          f"(spread {vals[-1] / max(vals[0], 1):.1f}x)")
    print(f"  **{len(stalled)} of {nsteps} steps > {stall_factor}x median**"
          + (f" -> {stalled}" if stalled else ""))
    if len(stalled) > 0.4 * nsteps:
        print(f"  ⚠ {100 * len(stalled) / nsteps:.0f}% of steps are 'stalled', so the "
              f"MEDIAN itself is near the stalled population — this count is\n"
              f"    unstable at this fraction. Quote the distribution above, not the count.")
    if rooted:
        r_tot = {p: sum(rooted[(p, n)] for n in range(nsteps)) / 1e6 for p in pids}
        lo = min(r_tot, key=r_tot.get)
        print(f"  rooted (broadcast) totals, EXCLUDED from the ranking: "
              + ', '.join(f"dev{dev.get(p, '?')}={r_tot[p]:.0f}ms" for p in pids)
              + f"  <- dev{dev.get(lo, '?')} is the root")

    hdr = f"  {'step':>5}" + ''.join(f"  dev{dev.get(p, '?')}_nccl" for p in pids)
    print(f"\n  NCCL ms per device, worst {min(5, len(stalled) or 1)} steps "
          f"(the LOWEST cell is the straggler — it never waited):")
    print(hdr)
    for n in sorted(stalled or [max(step_mean, key=step_mean.get)],
                    key=lambda k: -step_mean[k])[:5]:
        cells = [comm[(p, n)] / 1e6 for p in pids]
        lo = min(range(len(cells)), key=lambda i: cells[i])
        print(f"  {n:>5}" + ''.join(
            (f"  {c:>9.0f}*" if i == lo else f"  {c:>10.0f}")
            for i, c in enumerate(cells)) + f"   <- dev{dev.get(pids[lo], '?')}")

    # Which device is the straggler most often, across ALL stalled steps?
    tally = collections.Counter()
    for n in stalled:
        cells = {p: comm[(p, n)] for p in pids}
        tally[dev.get(min(cells, key=cells.get), '?')] += 1
    if tally:
        print(f"\n  straggler tally over the {len(stalled)} stalled steps: "
              + ', '.join(f"dev{d}={c}" for d, c in sorted(tally.items())))
        top, n_top = tally.most_common(1)[0]
        print(f"  >>> dev{top} is the straggler in {n_top}/{len(stalled)} "
              f"({100 * n_top / len(stalled):.0f}%) of stalled steps")

    print(f"\n  compute (non-NCCL) ms/step per device. ⚠ NOT a clean control: a "
          f"spinning NCCL kernel shares SMs with\n     the compute it overlaps, so a "
          f"WAITING rank's compute inflates too (~+5% observed).")
    print(f"  {'':>5}" + ''.join(f"  dev{dev.get(p, '?')}_comp" for p in pids))
    tot = [sum(comp[(p, n)] for n in range(nsteps)) / nsteps / 1e6 for p in pids]
    print(f"  {'mean':>5}" + ''.join(f"  {c:>10.2f}" for c in tot)
          + f"   spread {max(tot) / min(tot):.4f}x")


def stall_cause_report(cur, by_pid, phase='forward_loss', factor=3.0, top=6):
    """For each abnormally long PHASE window, what was the CPU actually doing?

    A rank that arrives late at a collective makes every other rank wait, and the
    GPU tables can only show you the *waiting* — never the cause. nsys's CPU
    sampling can: this joins COMPOSITE_EVENTS to SAMPLING_CALLCHAINS and prints
    the top leaf symbols inside the elongated window, on the thread that owns it.

    On both Pangu captures this is what identifies the step-30 stall as **CPython
    generation-2 garbage collection** (`gc_collect_main`, `visit_reachable`,
    `dict_traverse`, `func_traverse`) rather than a NUMA or affinity problem --
    which also explains why it recurs at the SAME iteration on two different
    nodes, since the gen-2 threshold is a function of allocation count, not of
    hardware. A NUMA lottery cannot reproduce an iteration index.
    """
    if not (_table_exists(cur, 'COMPOSITE_EVENTS')
            and _table_exists(cur, 'SAMPLING_CALLCHAINS')):
        print("  no CPU sampling in this capture — re-capture with "
              "`nsys profile --sample=process-tree` to get stall causes")
        return
    windows = []
    for pid, iv in by_pid.items():
        n = -1
        for start, end, name in iv:
            if name in ('data_prep', 'preprocess'):
                n += 1
            if name == phase:
                windows.append((pid, n, start, end, end - start))
    if not windows:
        print(f"  no `{phase}` ranges found")
        return
    med = sorted(w[4] for w in windows)[len(windows) // 2]
    hot = sorted((w for w in windows if w[4] > factor * med), key=lambda w: -w[4])
    dev = {p: d for p, d in cur.execute(
        "SELECT DISTINCT globalPid, deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL")}
    print(f"\n  `{phase}` median {med / 1e6:.1f} ms; "
          f"**{len(hot)} of {len(windows)} windows > {factor}x median**")
    for pid, n, start, end, dur in hot[:top]:
        print(f"\n  dev{dev.get(pid, '?')} step {n}: {phase} = {dur / 1e6:.1f} ms "
              f"({dur / med:.1f}x median) — top CPU leaf symbols:")
        best = None
        # list() first: reusing `cur` for the inner query would invalidate this
        # iteration and silently return no samples at all.
        tids = [r[0] for r in cur.execute(
            "SELECT DISTINCT globalTid FROM COMPOSITE_EVENTS "
            "WHERE (globalTid & ?) = ?", (PID_MASK, pid))]
        for tid in tids:
            rows = list(cur.execute("""
                SELECT s.value, COUNT(*) FROM COMPOSITE_EVENTS ce
                JOIN SAMPLING_CALLCHAINS sc ON sc.id = ce.id
                JOIN StringIds s ON s.id = sc.symbol
                WHERE ce.globalTid = ? AND ce.start BETWEEN ? AND ?
                  AND sc.stackDepth = 0
                GROUP BY 1 ORDER BY 2 DESC LIMIT 5""", (tid, start, end)))
            if rows and (best is None or rows[0][1] > best[0][1]):
                best = rows
        if not best:
            print("      (no samples in this window)")
            continue
        for sym, cnt in best:
            print(f"      {cnt:>5}  {sym[:64]}")
        if any('gc_' in s or 'traverse' in s or 'reachable' in s for s, _ in best):
            print("      ⇒ CPython GARBAGE COLLECTION. Output-neutral fix to try: "
                  "`gc.freeze()` after\n         model/optimizer construction, or "
                  "`gc.disable()` around the bench loop.")


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
    """SI units (1e3), NOT binary. The bandwidth figures divide bytes by 1e9, so a
    1024-based helper made one table mix '302.57 MiB' with '1279 GB/s' -- and it
    printed the D2D volume as 12.56 'GB' when 1e9-based it is 13.48 GB."""
    for u in ('B', 'kB', 'MB', 'GB', 'TB'):
        if b < 1000 or u == 'TB':
            return f"{b:.2f} {u}" if u != 'B' else f"{b:.0f} B"
        b /= 1000.0


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
    ap.add_argument('--per-rank', action='store_true',
                    help='per-step per-device NCCL: the direct straggler test')
    ap.add_argument('--stall-cause', action='store_true',
                    help='CPU leaf symbols inside abnormally long phase windows')
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
    if a.per_rank:
        per_rank_report(con.cursor(), by_pid)
    if a.stall_cause:
        stall_cause_report(con.cursor(), by_pid)
    if a.memcpy:
        print()
        memcpy_report(con.cursor(), by_pid, rank_steps)


if __name__ == '__main__':
    main()
