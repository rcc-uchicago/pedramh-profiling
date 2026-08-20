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


def report(rows, rank_steps):
    total_n = sum(v[0] for v in rows.values())
    total_t = sum(v[1] for v in rows.values())
    print(f"{total_n:,} kernel launches over {rank_steps} rank-steps "
          f"({total_n / rank_steps:,.0f} per rank-step), "
          f"{total_t / 1e9:.3f} s GPU time\n")
    print(f"{'range':<26}{'launches':>11}{'per rank-step':>15}{'% count':>9}"
          f"{'% time':>8}{'avg us':>9}{'count-time':>12}")
    print("  " + "-" * 88)
    skew = []
    for r, (n, t) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
        pc, pt = 100 * n / total_n, 100 * t / total_t
        skew.append((pc - pt, r, pc, pt))
        print(f"{r:<26}{n:>11,}{n / rank_steps:>15,.0f}{pc:>8.1f}%"
              f"{pt:>7.1f}%{t / n / 1000:>9.1f}{pc - pt:>+11.1f} pt")

    # The tool's thesis, tested rather than asserted.
    worst = max(skew) if skew else (0, None, 0, 0)
    print(f"\n  NOTE `% time` is a share of the kernel total, which CONTAINS NCCL wait and "
          f"is not\n  reproducible run-to-run (§4.4c: one such share moved 4.77 pt between "
          f"identical\n  configs). The `launches` and `per rank-step` columns are the durable "
          f"ones. The skew\n  below inherits that: it read +3.2 pt and +2.0 pt on the two "
          f"Pangu captures.")
    print()
    if worst[0] >= 10:
        print(f"  BATCHING TARGET: `{worst[1]}` issues {worst[2]:.1f}% of launches for "
              f"only {worst[3]:.1f}% of\n  GPU time (+{worst[0]:.1f} pt). That is many "
              f"small kernels -- fusing or batching\n  them buys launch-pipeline headroom "
              f"a time-sorted profile hides.")
    else:
        print(f"  NO batching target on this capture: the largest count-minus-time skew is "
              f"{worst[0]:+.1f} pt\n  (`{worst[1]}`), well under the +10 pt that would mark a "
              f"launch-bound site. Every range\n  issues roughly as much of the work as it "
              f"consumes of the time, so there is no\n  launch-pipeline headroom for batching "
              f"to recover here -- consistent with GPU-busy\n  of 98.5-98.6% (plan §0a). Do "
              f"not quote this tool as evidence for fusion.")


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
    rank_steps = sum(1 for v in by_pid.values() for r in v
                     if r[2] == anchor) if anchor else len(by_pid)
    print(f"{len(by_pid)} rank(s), {rank_steps} rank-steps "
          f"(pid-guarded, process-scoped join)\n")
    report(census(con.cursor(), by_pid, rank_steps), rank_steps)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: kernel_census.py <nsys.sqlite>")
    main(sys.argv[1])
