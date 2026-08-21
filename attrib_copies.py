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
import sqlite3
import sys

PID_MASK = ~0xFFFFFF
KERNEL_LIKE = ("%direct_copy_kernel_cuda%", "%conj_kernel_cuda%")


def _tables(cur):
    return {r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}


def find_callchain_table(cur):
    """Return (table, id_col) for the table holding --cudabacktrace frames.

    Discovered by schema shape and then CONFIRMED by key overlap with the ids
    RUNTIME actually references — a table can look right and still be the wrong one
    (SAMPLING_CALLCHAINS has the same shape but holds periodic CPU samples).
    """
    cands = [t for t in _tables(cur) if "CALLCHAIN" in t.upper()]
    ids = {r[0] for r in cur.execute(
        "SELECT DISTINCT callchainId FROM CUPTI_ACTIVITY_KIND_RUNTIME "
        "WHERE callchainId IS NOT NULL AND callchainId != 0 LIMIT 500")}
    if not ids:
        sys.exit("ERROR NO_CALLCHAINS: every RUNTIME row has callchainId 0 — the "
                 "capture ran without --cudabacktrace. (tick 17: the column exists "
                 "even when unused, so its presence proves nothing.)")
    best = None
    for t in cands:
        cols = [r[1] for r in cur.execute("PRAGMA table_info(%s)" % t)]
        if "id" not in cols or "symbol" not in cols:
            continue
        have = {r[0] for r in cur.execute("SELECT DISTINCT id FROM %s" % t)}
        overlap = len(ids & have)
        if best is None or overlap > best[1]:
            best = (t, overlap, cols)
    if not best or best[1] == 0:
        sys.exit("ERROR NO_MATCHING_CALLCHAIN_TABLE: considered %s, none whose `id` "
                 "overlaps the callchainIds RUNTIME references." % (sorted(cands) or "none"))
    return best[0], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--grid", type=int, default=None,
                    help="only kernels with this gridX (64800 = the §4.8 f32 row)")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    db = sqlite3.connect("file:%s?mode=ro" % a.sqlite, uri=True)  # captures are read-only
    cur = db.cursor()
    tbl, _ = find_callchain_table(cur)
    print("callchain table: %s" % tbl)

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

    tot = sum(r[3] for r in rows)
    print("attributed launches: %d across %d distinct call sites\n" % (tot, len(rows)))
    for gridX, kname, ccid, n in rows[:a.top]:
        dt = "c64" if "complex" in kname else "f32"
        print("=" * 78)
        print("%d launches (%.1f%%)  grid=%d  %s" % (n, 100.0 * n / tot, gridX, dt))
        frames = cur.execute(
            "SELECT COALESCE(sy.value, '<unresolved>'), COALESCE(mo.value, '') "
            "FROM %s c LEFT JOIN StringIds sy ON sy.id = c.symbol "
            "LEFT JOIN StringIds mo ON mo.id = c.module "
            "WHERE c.id = ? ORDER BY c.stackDepth" % tbl, (ccid,)).fetchall()
        # Python frames are the point; interpreter internals are noise.
        for sym, mod in frames:
            keep = ".py" in mod or ".py" in sym or not sym.startswith(("_Py", "Py"))
            if keep:
                print("    %-52s %s" % (sym[:52], mod.split("/")[-1][:24]))
    db.close()


if __name__ == "__main__":
    main()
