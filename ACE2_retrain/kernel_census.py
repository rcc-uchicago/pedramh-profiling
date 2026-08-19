"""Rank NVTX ranges by KERNEL LAUNCH COUNT, not by time.

When the goal is "fewer kernels so the GPU runs efficiently", time-per-range is
the wrong ranking. A site launching 2,000 kernels of 3 us each costs little GPU
time but wrecks the launch pipeline -- ACE2 issues ~7,222 launches/second on one
device, one every ~138 us, and ~9% of training time is idle gaps between them.
Those sites are invisible in a time-sorted profile and obvious here.

Attribution uses the RUNTIME -> KERNEL correlationId join, so a kernel is
credited to the range that LAUNCHED it. An NVTX range bounds CPU time and CUDA
is async, so the naive "kernels inside the range's time window" is wrong.

    python kernel_census.py <nsys.sqlite>
"""

import bisect
import collections
import re
import sqlite3
import sys

UNINFORMATIVE = {"aten::copy_", "aten::to", "aten::_to_copy"}


def main(path):
    db = sqlite3.connect(path)
    strip = lambda t: re.sub(r",\s*op_id\s*=\s*\d+", "", t).strip()
    by = collections.defaultdict(list)
    for t, s, e, g in db.execute(
        "SELECT text,start,end,globalTid FROM NVTX_EVENTS "
        "WHERE text IS NOT NULL AND end IS NOT NULL AND text NOT LIKE 'step_%'"
    ):
        by[g].append((s, e, strip(t)))
    for g in by:
        by[g].sort()

    def enclosing(ts, tid):
        arr = by.get(tid)
        if not arr:
            return "(outside)"
        i = bisect.bisect_right(arr, (ts, float("inf"), "")) - 1
        seen = 0
        while i >= 0 and seen < 5000:
            s, e, t = arr[i]
            if s <= ts <= e and t not in UNINFORMATIVE:
                return t
            i -= 1
            seen += 1
        return "(outside)"

    n_steps = len({s for (s,) in db.execute(
        "SELECT start FROM NVTX_EVENTS WHERE text LIKE 'step_%'")}) or 1
    count = collections.Counter()
    time_ns = collections.Counter()
    for rs, tid, dur in db.execute(
        "SELECT r.start, r.globalTid, k.end-k.start "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId = r.correlationId"
    ):
        e = enclosing(rs, tid)
        count[e] += 1
        time_ns[e] += dur

    total_n = sum(count.values())
    total_t = sum(time_ns.values())
    print(f"{total_n:,} kernel launches over ~{n_steps} steps "
          f"({total_n / n_steps:,.0f} per step), {total_t / 1e9:.1f} s GPU time\n")
    print(f"{'range':<26}{'launches':>11}{'per step':>10}{'% count':>9}"
          f"{'% time':>8}{'avg us':>9}")
    for r, n in count.most_common(16):
        t = time_ns[r]
        print(f"{r:<26}{n:>11,}{n / n_steps:>10,.0f}{100 * n / total_n:>8.1f}%"
              f"{100 * t / total_t:>7.1f}%{t / n / 1000:>9.1f}")
    print("\nHigh launch share + low time share = a batching target: many tiny")
    print("kernels, typically one per named variable. Fusing or batching those")
    print("buys launch-pipeline headroom that a time-sorted profile hides.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: kernel_census.py <nsys.sqlite>")
    main(sys.argv[1])
