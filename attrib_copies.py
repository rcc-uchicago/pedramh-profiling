#!/usr/bin/env python3
"""Attribute the `direct_copy`/`conj` kernels to their launching call sites.

    python3 attrib_copies.py <capture.sqlite> [--grid N] [--top K]

Plan item 8's remaining target: §4.8 measured the f32 / 66.4 MB copy population at
**31.50 sectors/request against an ideal of 4**, and §4.9 could not explain it from
source the way it explained the 377 MB weight. This walks

    KERNEL --correlationId--> RUNTIME --callchainId--> frames

and prints the call sites, ranked by how many launches each accounts for.

Two things this gets right that a naive query does not:

* **`correlationId` is per-PROCESS, not global.** Joining on it alone silently mixes
  ranks — the bug fixed once already in `nvtx_phase_attribution.py`. The guard is
  `k.globalPid = (r.globalTid & ~0xFFFFFF)`: `globalTid` is `globalPid | tid` with the
  thread id in the low 24 bits.
* **The backtrace table's name is discovered, not assumed.** `--cudabacktrace` creates
  its table only when enabled, so no capture we already own contains it and its name
  cannot be verified offline. Guessing wrong would look like "no call sites found",
  which is indistinguishable from a failed capture — so this reports which tables it
  considered.
"""
import argparse
import collections
import re
import sqlite3
import sys

PID_MASK = ~0xFFFFFF
KERNEL_LIKE = ("%direct_copy_kernel_cuda%", "%conj_kernel_cuda%")

# Dispatcher/interpreter boilerplate. Every copy stack passes through these, so
# leaving them in makes distinct call sites look identical for the first ~8 frames.
_BORING = re.compile(
    r"^(0x|c10::impl|at::_ops|torch::autograd::VariableType|cudaLaunchKernel|std::|"
    r"at::native::(copy_|copy_impl|copy_device_to_device|gpu_kernel)|"
    r"void at::native::gpu_kernel|_Py|Py[A-Z])")


def _signature(frames):
    return [f for f in frames if not _BORING.match(f)]


def _tables(cur):
    return {r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}


def find_callchain_table(cur):
    """Return (table, why) for the table holding --cudabacktrace frames.

    **Do NOT discriminate by id overlap.** Every *_CALLCHAINS table numbers its own
    chains from 1, so on a real capture `CUDA_CALLCHAINS`, `OSRT_CALLCHAINS` and
    `SAMPLING_CALLCHAINS` all contain ids 1..N and all overlap a sample 100% — the tie
    then breaks on set-iteration order. That silently selected OSRT_CALLCHAINS on job
    7551282 and produced confident nonsense (`ProcessGroupNCCL::Watchdog`,
    `_pickle_loads`) for kernel-launch call sites. No sample size fixes it.

    `CUDA_CALLCHAINS` is the table `--cudabacktrace` creates, so prefer it by name and
    corroborate on CARDINALITY: its distinct-id count should track the number of
    distinct callchainIds RUNTIME references.
    """
    tabs = _tables(cur)
    n_ref = cur.execute(
        "SELECT COUNT(DISTINCT callchainId) FROM CUPTI_ACTIVITY_KIND_RUNTIME "
        "WHERE callchainId IS NOT NULL AND callchainId != 0").fetchone()[0]
    if not n_ref:
        sys.exit("ERROR NO_CALLCHAINS: every RUNTIME row has callchainId 0 — the "
                 "capture ran without --cudabacktrace. (tick 17: the column exists "
                 "even when unused, so its presence proves nothing.)")

    def card(t):
        return cur.execute("SELECT COUNT(DISTINCT id) FROM %s" % t).fetchone()[0]

    if "CUDA_CALLCHAINS" in tabs:
        n = card("CUDA_CALLCHAINS")
        why = "named table for --cudabacktrace; %d ids vs %d referenced" % (n, n_ref)
        if n < 0.5 * n_ref:
            sys.exit("ERROR CUDA_CALLCHAINS_TOO_SMALL: %d ids for %d referenced "
                     "callchainIds — the capture is inconsistent; refusing to fall "
                     "back to a CPU-sampling table, which would look plausible and be "
                     "wrong." % (n, n_ref))
        return "CUDA_CALLCHAINS", why
    cands = [t for t in tabs if "CALLCHAIN" in t.upper()]
    if not cands:
        sys.exit("ERROR NO_CALLCHAIN_TABLE: none present.")
    best = min(cands, key=lambda t: abs(card(t) - n_ref))
    return best, ("no CUDA_CALLCHAINS; closest cardinality (%d vs %d referenced)"
                  % (card(best), n_ref))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--grid", type=int, default=None,
                    help="only kernels with this gridX (64800 = the §4.8 f32 row)")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    db = sqlite3.connect("file:%s?mode=ro" % a.sqlite, uri=True)  # captures are read-only
    cur = db.cursor()
    tbl, why = find_callchain_table(cur)
    print("callchain table: %s  (%s)" % (tbl, why))

    like = " OR ".join("s.value LIKE ?" for _ in KERNEL_LIKE)
    grid = " AND k.gridX = %d" % a.grid if a.grid else ""
    # The globalPid guard is the whole correctness story of this join.
    q = """
        SELECT k.gridX, s.value, r.callchainId, COUNT(*) n
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.demangledName
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
          ON r.correlationId = k.correlationId
         AND k.globalPid = (r.globalTid & {mask})
        WHERE ({like}){grid} AND r.callchainId IS NOT NULL AND r.callchainId != 0
        GROUP BY k.gridX, s.value, r.callchainId
        ORDER BY n DESC
    """.format(mask=PID_MASK, like=like, grid=grid)
    rows = cur.execute(q, KERNEL_LIKE).fetchall()
    if not rows:
        sys.exit("ERROR NO_ATTRIBUTED_LAUNCHES: no copy kernel joined to a callchain "
                 "(grid filter=%s)." % a.grid)

    # Group by the RESOLVED STACK, not by callchainId. nsys mints a fresh
    # callchainId per API CALL, so grouping on it yields exactly one launch per
    # "site" — job 7551282 printed "44800 launches across 44800 call sites", a
    # ratio of 1.0 that reads as "attribution is hopelessly smeared" when in truth
    # there were four stacks. A degenerate ratio is a bug signal, not a finding.
    frames_of = {}
    for ccid in {r[2] for r in rows}:
        frames_of[ccid] = cur.execute(
            "SELECT COALESCE(sy.value, '<unresolved>') FROM %s c "
            "LEFT JOIN StringIds sy ON sy.id = c.symbol "
            "WHERE c.id = ? ORDER BY c.stackDepth" % tbl, (ccid,)).fetchall()
    sites = collections.defaultdict(lambda: [0, None])
    for gridX, kname, ccid, n in rows:
        fr = [f[0] for f in frames_of[ccid]]
        key = (gridX, "c64" if "complex" in kname else "f32", tuple(_signature(fr)))
        sites[key][0] += n
        sites[key][1] = fr
    tot = sum(v[0] for v in sites.values())
    print("attributed launches: %d across %d distinct stacks\n" % (tot, len(sites)))
    for (gridX, dt, _), (n, fr) in sorted(sites.items(), key=lambda kv: -kv[1][0])[:a.top]:
        print("=" * 78)
        print("%d launches (%.1f%%)  grid=%d  %s" % (n, 100.0 * n / tot, gridX, dt))
        thread = ("autograd engine (BACKWARD)" if any("autograd::Engine" in f for f in fr)
                  else "main thread (forward)")
        print("  thread: %s" % thread)
        for f in _signature(fr)[:8]:
            print("    " + f[:88])
    db.close()


if __name__ == "__main__":
    main()
