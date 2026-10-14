#!/usr/bin/env python3
"""
Parse a Nsight Systems SQLite export and print a bench summary.

Usage (on Midway after the job, or locally after scp):
    python3 parse_nsys.py si_nvtx_<jobid>_rank<N>.sqlite

The .sqlite is produced from the .nsys-rep file via:
    nsys export --type=sqlite <file>.nsys-rep

Originally copied from Projects/S2S/v2.0/HPC_scripts/parse_nsys.py; extended
to include ACE2's `stack`/`unstack`/`normalize`/`denormalize` ranges
(emitted by ACE2_retrain/ace2_nvtx.py) plus `ema`. The four ACE2 names
exist to settle where the copy traffic comes from: the profile shows
copies at 28% of GPU kernel time, attributed to the dict<->tensor round
trip by code reading alone. `stack`/`unstack` measure that directly.

Per CLAUDE.md #10 the SHARED names are reused verbatim and never
renamed; only genuinely new phases get new names. Never remove a name --
a dropped range silently invalidates every prior comparison.
"""
import sqlite3
import statistics
import sys
from pathlib import Path

# statistics.fmean is 3.8+; the Polaris login node's default python3 is 3.6.15,
# and login-node re-analysis of an existing capture is the whole point of the
# no-GPU workflow. sum/len is what fmean computes, so 3.8+ output is unchanged.
_fmean = getattr(statistics, 'fmean', None) or (lambda v: sum(v) / len(v))

# The NVTX range contract (CLAUDE.md #10): SHARED names first, in contract
# order, then each project's own. Never rename, never remove -- a dropped name
# silently invalidates every prior comparison. ONE list, read by both the query
# and the print loop; `nvtx_phase_attribution.py` imports it rather than
# keeping a fourth copy.
#
# NOTE there are three other copies of this parser in the repo, and they do NOT
# agree: `s2s/v2.0/HPC_scripts/parse_nsys.py` carries only the first 8 names,
# and it is the one the Polaris PBS scripts and
# `physicsnemo_ai_rossby/polaris/bench_instrumentation_test.py` actually run.
# Adding a range here does not make that copy print it. See CHANGELOG 2026-08-20.
RANGE_NAMES = ('preprocess', 'data_prep', 'forward_loss',
               'backward', 'optimizer',
               'vae_encoder1', 'vae_encoder2', 'ema',
               'stack', 'unstack', 'normalize', 'denormalize',
               'amp_region', 'sht_fwd', 'sht_inv',
               'sfno_net', 'sfno_block', 'spectral_filter', 'sfno_mlp')


def _table_exists(cur, name):
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


def _col_exists(cur, table, col):
    cur.execute(f"PRAGMA table_info({table})")
    return any(row[1] == col for row in cur.fetchall())


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def table(rows, headers):
    if not rows:
        print("  (no data)")
        return
    widths = [max(len(str(h)), max(len(str(r[i])) for r in rows))
              for i, h in enumerate(headers)]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


def nvtx_summary(cur):
    section("NVTX Range Summary (measured steps only)")

    # nsys SQLite schema varies by version; try both common table names
    tbl = None
    for candidate in ("NVTX_EVENTS", "StringIds"):
        if _table_exists(cur, candidate):
            tbl = candidate
            break

    # Standard schema: NVTX_EVENTS with text and start/end in nanoseconds
    if not _table_exists(cur, "NVTX_EVENTS"):
        print("  NVTX_EVENTS table not found — was --trace=nvtx passed to nsys?")
        return

    # SI emits: preprocess, forward_loss (from train_module.py),
    # backward, optimizer (from bench_callback.py when SI_NVTX=1).
    # data_prep / vae_* are kept for backwards-compat with S2S traces.
    cur.execute(
        "SELECT text, (end - start) AS dur_ns FROM NVTX_EVENTS "
        f"WHERE text IN ({','.join('?' * len(RANGE_NAMES))}) "
        "AND end IS NOT NULL AND end > start ORDER BY text", RANGE_NAMES)
    rows = cur.fetchall()
    if not rows:
        print("  No bench NVTX ranges found "
              "(preprocess/forward_loss/backward/optimizer).")
        print("  Was SI_NVTX=1 set and did the capture range fire?")
        return

    from collections import defaultdict
    by_name = defaultdict(list)
    for name, dur_ns in rows:
        by_name[name].append(dur_ns / 1e6)  # → ms

    out = []
    # Both the query and this loop read RANGE_NAMES, so a new range cannot be
    # added to one and forgotten in the other -- which is exactly what had
    # happened to `unstack`: it was in the SQL, absent here, so its rows were
    # fetched and then silently never printed.
    for name in RANGE_NAMES:
        vals = by_name.get(name, [])
        if not vals:
            continue
        out.append((
            name,
            len(vals),
            f"{statistics.median(vals):.1f}",
            f"{_fmean(vals):.1f}",
            f"{min(vals):.1f}",
            f"{max(vals):.1f}",
        ))
    table(out, ["range", "n", "median_ms", "mean_ms", "min_ms", "max_ms"])

    # Step-level totals
    cur.execute("""
        SELECT text, (end - start) AS dur_ns
        FROM NVTX_EVENTS
        WHERE text LIKE 'step_%'
          AND end IS NOT NULL AND end > start
    """)
    step_rows = cur.fetchall()
    if step_rows:
        step_ms = [dur / 1e6 for _, dur in step_rows]
        print(f"\n  Step totals from NVTX ({len(step_ms)} steps):")
        print(f"    median {statistics.median(step_ms):.1f} ms  "
              f"mean {_fmean(step_ms):.1f} ms  "
              f"std {statistics.pstdev(step_ms):.1f} ms")


def kernel_summary(cur):
    section("Top 20 CUDA Kernels by Total GPU Time")

    # Table name differs across nsys versions
    for tbl in ("CUPTI_ACTIVITY_KIND_KERNEL", "GPU_METRICS"):
        if _table_exists(cur, tbl):
            break
    else:
        print("  Kernel activity table not found.")
        return

    name_col = "demangledName" if _col_exists(cur, tbl, "demangledName") else "shortName"
    dur = "(end - start)" if _col_exists(cur, tbl, "end") else "duration"
    cur.execute(f"""
        SELECT s.value AS kname,
               COUNT(*)                        AS launches,
               SUM({dur}) / 1e6               AS total_ms,
               AVG({dur}) / 1000.0            AS avg_us
        FROM {tbl} k
        JOIN StringIds s ON s.id = k.{name_col}
        GROUP BY kname
        ORDER BY total_ms DESC
        LIMIT 20
    """)
    rows = [(r[0][:60], r[1], f"{r[2]:.1f}", f"{r[3]:.1f}") for r in cur.fetchall()]
    table(rows, ["kernel (truncated to 60 chars)", "launches", "total_ms", "avg_us"])


def nccl_summary(cur):
    section("NCCL AllReduce Kernel Time (DDP gradient sync)")

    for tbl in ("CUPTI_ACTIVITY_KIND_KERNEL", "GPU_METRICS"):
        if _table_exists(cur, tbl):
            break
    else:
        print("  Kernel activity table not found.")
        return

    name_col = "demangledName" if _col_exists(cur, tbl, "demangledName") else "shortName"
    dur = "(end - start)" if _col_exists(cur, tbl, "end") else "duration"
    cur.execute(f"""
        SELECT s.value AS kname, {dur} / 1000.0 AS dur_us
        FROM {tbl} k
        JOIN StringIds s ON s.id = k.{name_col}
        WHERE s.value LIKE '%ncclKernel%AllReduce%'
           OR s.value LIKE '%nccl%all_reduce%'
           OR s.value LIKE '%AllReduce%'
    """)
    rows = cur.fetchall()
    if not rows:
        print("  No NCCL AllReduce kernels found in capture window.")
        return
    durations_us = [r[1] for r in rows]
    total_ms = sum(durations_us) / 1000.0
    print(f"  Calls: {len(durations_us)}")
    print(f"  Total time: {total_ms:.1f} ms  "
          f"avg: {_fmean(durations_us):.0f} µs  "
          f"max: {max(durations_us):.0f} µs")


def memcpy_summary(cur):
    section("Host→Device Memory Copies (data_prep H2D transfers)")

    for tbl in ("CUPTI_ACTIVITY_KIND_MEMCPY", "MEMCPY"):
        if _table_exists(cur, tbl):
            break
    else:
        print("  Memcpy activity table not found.")
        return

    # copyKind=1 is HtoD in CUPTI
    dur = "(end - start)" if _col_exists(cur, tbl, "end") else "duration"
    cur.execute(f"""
        SELECT COUNT(*) AS n,
               SUM({dur}) / 1e6   AS total_ms,
               AVG({dur}) / 1000.0 AS avg_us,
               SUM(bytes)  / 1e6   AS total_mb
        FROM {tbl}
        WHERE copyKind = 1
    """)
    row = cur.fetchone()
    if row and row[0]:
        print(f"  Transfers:  {row[0]}")
        print(f"  Total time: {row[1]:.1f} ms   avg per transfer: {row[2]:.0f} µs")
        print(f"  Total data: {row[3]:.1f} MB")
    else:
        print("  No HtoD transfers found (copyKind=1).")


def available_tables(cur):
    section("Available Tables in this SQLite (schema reference)")
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    names = [r[0] for r in cur.fetchall()]
    for i, name in enumerate(names):
        end = "  " if (i + 1) % 4 != 0 else "\n"
        print(f"  {name}", end=end)
    print()


def main():
    if len(sys.argv) != 2:
        print("Usage: python parse_nsys.py <file>.sqlite")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"\nNsight Systems analysis: {path.name}")

    # str(), not the Path: sqlite3.connect only grew os.PathLike support in
    # Python 3.7, and the Polaris login node's default python3 is 3.6.15 --
    # where this raised TypeError and made login-node re-analysis of a capture
    # (the whole no-GPU-needed workflow) impossible.
    con = sqlite3.connect(str(path))
    cur = con.cursor()

    available_tables(cur)
    nvtx_summary(cur)
    kernel_summary(cur)
    nccl_summary(cur)
    memcpy_summary(cur)

    con.close()


if __name__ == "__main__":
    main()
