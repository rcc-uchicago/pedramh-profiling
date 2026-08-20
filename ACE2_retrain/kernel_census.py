"""Rank NVTX ranges by KERNEL LAUNCH COUNT, not by time.

    python3 kernel_census.py <nsys.sqlite>

Time-per-range answers "where does the GPU spend its time"; this answers "which
range *issues the most work*". They differ when a site launches many small
kernels: that costs little GPU time but occupies the launch pipeline, and a
time-sorted profile hides it.

**Whether that pattern is present is an empirical question, and on this project
it has so far been absent** -- see the closing note, which is printed from the
data rather than asserted. An earlier version of this docstring taught that ~9%
of training time was idle gaps between launches and that batching would recover
it; that reading is **refuted** (`PANGU_POLARIS_PROFILING_PLAN.md` §0a: GPU-busy
union is 95.6-96.5% on kernels alone and 98.5-98.6% counting memcpy/memset, so
there is no launch-latency headroom to recover). The tool is still useful for the
count ranking; it just must not be read as evidence for a conclusion it cannot
reach.

Attribution is delegated to `nvtx_phase_attribution.py` rather than re-derived,
because getting it right needs two things this file used to get wrong:

1. a **`globalPid` guard** on the `correlationId` join -- `correlationId` is
   unique per process, so on a multi-rank capture the bare join cross-products
   the ranks (+29.4% phantom rows on `nsys_pangu_sfno_7255503.sqlite`);
2. **process-scoped**, not thread-scoped, range lookup -- PyTorch's autograd
   engine launches from its own worker thread, so looking the range up on the
   launching thread put *the whole of `backward`* outside every range. This file
   reported `backward` as **203 launches and 0.0% of GPU time** where the truth
   is 250,880 and 72.6%.
"""
import collections
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nvtx_phase_attribution as npa  # noqa: E402


def census(cur, by_pid, rank_steps):
    """{range: (launches, gpu_ns)} — rolled up from the pid-guarded join."""
    agg, _ = npa.attribute(cur, by_pid)
    out = collections.defaultdict(lambda: [0, 0])
    for (phase, _label), (n, ns) in agg.items():
        out[phase][0] += n
        out[phase][1] += ns
    return out


SMALL_US = 10          # "tiny" = shorter than ~1 launch call (~8 us measured)
SKEW_BAR = 10.0        # pt of count-share minus time-share; see the note in report()


def small_kernel_census(cur, threshold_us=SMALL_US, top=5):
    """The tiny-kernel population, and the two numbers that decide whether it matters.

    The phase-level skew CANNOT see this population -- see report()'s note -- so
    the thesis has to be tested at the granularity where tiny kernels live. What
    settles it is not the count but the **prize**: how much GPU time perfect
    fusion would recover, against how deep the launch queue already is.
    """
    n, tot = cur.execute(
        "SELECT COUNT(*), SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchone()
    sn, st = cur.execute(
        "SELECT COUNT(*), SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE end - start < ?", (threshold_us * 1000,)).fetchone()
    rows = cur.execute("""
        SELECT s.value, COUNT(*), SUM(k.end - k.start)
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.demangledName
        WHERE k.end - k.start < ? GROUP BY 1 ORDER BY 2 DESC LIMIT ?""",
        (threshold_us * 1000, top)).fetchall()
    # Launch -> execute queue depth. A queue this deep cannot be starved by an
    # 8 us launch call, which is the direct refutation of the launch-latency
    # reading -- far better evidence than a GPU-busy percentage.
    depths = sorted(k - r for r, k in cur.execute(
        "SELECT r.start, k.start FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId = r.correlationId "
        "AND k.globalPid = (r.globalTid & ?)", (npa.PID_MASK,)))
    return dict(n=n, tot=tot, small_n=sn or 0, small_ns=st or 0, top=rows,
                depth_med=depths[len(depths) // 2] if depths else 0,
                depth_p25=depths[len(depths) // 4] if depths else 0)


def report(rows, rank_steps, small=None):
    total_n = sum(v[0] for v in rows.values())
    total_t = sum(v[1] for v in rows.values())
    print(f"{total_n:,} kernel launches over {rank_steps} rank-steps "
          f"({total_n / rank_steps:,.0f} per rank-step), "
          f"{total_t / 1e9:.3f} s GPU time\n")
    print(f"{'range':<26}{'launches':>11}{'per rank-step':>15}{'% count':>9}"
          f"{'% time':>8}{'avg us':>9}{'count-time':>12}")
    print("  " + "-" * 88)
    avg_all = total_t / total_n
    skew = []
    for r, (n, t) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
        pc, pt = 100 * n / total_n, 100 * t / total_t
        if r != '(outside)':          # not a code site; cannot be "batched"
            skew.append((pc - pt, r, pc, pt))
        print(f"{r:<26}{n:>11,}{n / rank_steps:>15,.0f}{pc:>8.1f}%"
              f"{pt:>7.1f}%{t / n / 1000:>9.1f}{pc - pt:>+11.1f} pt")
    if '(outside)' in rows:
        print(f"\n  ⚠ `(outside)` is present: {rows['(outside)'][0]:,} launches fall in no "
              f"NVTX range. That is an\n    instrumentation-coverage gap, not a code site — "
              f"it is excluded from the skew below.")

    print(f"\n  NOTE on `count-time`: it has a closed form, `skew_r = pc_r x (1 - avg_r/avg_all)`,"
          f"\n  so it is a COUNT-WEIGHTED RELATIVE mean-duration test, not an absolute one. Two"
          f"\n  consequences: a range holding {rows and 100 * min(v[0] for v in rows.values()) / total_n:.1f}%"
          f" of launches can never skew past that share no\n  matter how tiny its kernels are, and"
          f" `% time` here contains NCCL wait, which is not\n  reproducible run-to-run (§4.4c). "
          f"**The phase-level skew cannot see a tiny-kernel\n  population at all** — that is what "
          f"the section below is for.")
    worst = max(skew) if skew else (0, None, 0, 0)
    print(f"\n  largest phase skew: {worst[0]:+.1f} pt (`{worst[1]}`) vs a {SKEW_BAR:+.0f} pt bar"
          f" — but see below before concluding anything.")

    if not small:
        return
    sn, snt, n, tot = small['small_n'], small['small_ns'], small['n'], small['tot']
    pc, pt = 100 * sn / n, 100 * snt / tot
    print(f"\n  --- tiny kernels (< {SMALL_US} us, i.e. shorter than ~1 launch call) ---")
    print(f"  {sn:,} launches = **{pc:.1f}% of all launches** for **{pt:.2f}% of GPU time** "
          f"(skew {pc - pt:+.1f} pt),\n  {sn / rank_steps:,.0f} per rank-step. Top contributors:")
    for v, cnt, ns in small['top']:
        print(f"    {cnt:>7,}  {ns / cnt / 1000:>6.2f} us  {npa.short_label(v)[:52]}")
    print(f"\n  queue depth (launch -> execute): median "
          f"**{small['depth_med'] / 1e6:.0f} ms**, p25 {small['depth_p25'] / 1e6:.0f} ms")
    prize = 100 * snt / tot
    if prize >= 5:
        print(f"\n  BATCHING TARGET: perfectly fusing these would recover up to {prize:.1f}% of "
              f"GPU time.\n  Worth sizing properly.")
    else:
        print(f"\n  VERDICT: a tiny-kernel population EXISTS and is large by count "
              f"({pc:.1f}% of launches),\n  but batching is not worth funding — perfectly fusing "
              f"**all** of it recovers at most\n  **{prize:.2f}% of GPU time** "
              f"({snt / 1e9:.3f} s of {tot / 1e9:.1f} s), against a launch queue already\n  "
              f"**{small['depth_med'] / 1e6:.0f} ms deep** — the CPU runs roughly a third of a step "
              f"ahead, so an ~8 us\n  launch call cannot starve it. Do NOT read a small phase skew "
              f"as evidence for this;\n  the phase metric is blind here (see the NOTE above). The "
              f"launch-latency reading was\n  already retired on its own turf by "
              f"POLARIS_PROFILING_HANDOFF.md §6 dead ends 1-2\n  (stream/DDP-bucket dependency, "
              f"not launch starvation); this adds a Polaris queue-depth\n  datum and does not by "
              f"itself refute the ACE2/Midway measurement, which is a different\n  model on "
              f"different hardware.")


def main(path):
    # mode=ro: a capture is a primary artifact and is never written to.
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    by_pid = npa.load_phases(con.cursor())
    if not by_pid:
        sys.exit("ERROR no NVTX phase ranges in NVTX_EVENTS.text — was the harness "
                 "instrumented and --trace=nvtx passed?")
    # Normalise per RANK-STEP, not per step: counting distinct `step_%` starts
    # gives 156 on a 40-step x 4-rank capture (one step's range predates
    # cudaProfilerStart), and dividing 4 ranks' launches by that is wrong twice.
    anchor = next((n for n in ('data_prep', 'preprocess')
                   if any(r[2] == n for v in by_pid.values() for r in v)), None)
    if anchor is None:
        # Falling back to len(by_pid) would silently report the RANK count as the
        # rank-STEP count -- a 40x error with no warning. `ace2_nvtx.py:30` records
        # that ACE2 deliberately emits no `data_prep`, so this path is live.
        sys.exit("ERROR NO_STEP_ANCHOR: no `data_prep` or `preprocess` range, so "
                 "rank-steps cannot be\n  derived. Pass a per-rank-step count "
                 "explicitly rather than letting this guess.")
    rank_steps = sum(1 for v in by_pid.values() for r in v if r[2] == anchor)
    print(f"{len(by_pid)} rank(s), {rank_steps} rank-steps "
          f"(pid-guarded, process-scoped join)\n")
    report(census(con.cursor(), by_pid, rank_steps), rank_steps,
           small_kernel_census(con.cursor()))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: kernel_census.py <nsys.sqlite>")
    main(sys.argv[1])
