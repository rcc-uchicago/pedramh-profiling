#!/usr/bin/env python3
"""
Cross-profile GPU utilization comparison for S2S inference benchmarks.

Usage:
    python compare_nsys.py <file1.sqlite> [<file2.sqlite> ...]

Each .sqlite file must be produced by exporting an nsys profile:
    nsys export --type=sqlite <file>.nsys-rep

Sections produced for each profile:
  1. Per-GPU utilization  — actual kernel-active time vs elapsed capture window.
     util% < 100% means the GPU was idle for that fraction of time.
  2. H2D bandwidth        — how fast host data reached each GPU (pinned PCIe throughput).
  3. NCCL collectives     — all-reduce / all-gather / broadcast kernel time (multi-GPU comms).
  4. Sync stall breakdown — how long the GPU spent blocked in cudaStreamSync /
                            cudaEventSync calls, split by CUPTI sync type.
  5. Idle gap histogram   — distribution of gaps *between* consecutive GPU kernels on GPU0,
                            showing whether idle time is many small gaps or a few large stalls.
"""

import sqlite3
import statistics
import sys
from pathlib import Path


def _table_exists(cur, name: str) -> bool:
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def section(title):
    """Print a visually distinct section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def tbl(rows, headers):
    """Print a left-aligned fixed-width table with column headers."""
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


# ---------------------------------------------------------------------------
# Analysis sections
# ---------------------------------------------------------------------------

def gpu_utilization(cur, label):
    """
    Per-device kernel statistics.

    active_ms  = sum of all kernel durations (GPU was doing real work)
    window_ms  = MAX(end) - MIN(start) across all kernels (elapsed wall time on GPU)
    util       = active_ms / window_ms — the fraction of the window the GPU was busy.

    If util is low (e.g. 15-40%), the GPU is spending most of its time idle between
    kernel launches, waiting for either CPU dispatch, inter-GPU barriers, or data.
    """
    section(f"Per-GPU Utilization — {label}")
    if not _table_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
        print("  CUPTI_ACTIVITY_KIND_KERNEL not found — GPU kernel activity was not captured.")
        print("  Likely cause: ptrace restrictions on this partition prevented nsys from")
        print("  attaching to torchrun worker processes. Try cuda/12.9 + --target-processes=all.")
        return
    cur.execute("""
        SELECT deviceId,
               COUNT(*)                        AS launches,
               SUM(end-start)/1e6             AS active_ms,
               (MAX(end)-MIN(start))/1e6      AS window_ms
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        GROUP BY deviceId
        ORDER BY deviceId
    """)
    rows = cur.fetchall()
    out = []
    for r in rows:
        util = r[2] / r[3] * 100 if r[3] else 0
        out.append((f"GPU{r[0]}", f"{r[1]:,}", f"{r[2]:.0f}", f"{r[3]:.0f}", f"{util:.1f}%"))
    tbl(out, ["device", "launches", "active_ms", "window_ms", "util"])


def h2d_bandwidth(cur, label):
    """
    Host-to-Device (H2D) PCIe transfer throughput per GPU, with source memory kind.

    copyKind=1 in CUPTI_ACTIVITY_KIND_MEMCPY means HtoD.
    srcKind distinguishes pageable (0) from pinned/page-locked (1) host memory.
    Pinned memory bypasses the OS page-fault mechanism and allows the DMA engine
    to transfer directly, achieving higher and more consistent bandwidth.
    Pageable transfers require a bounce through a staging buffer, roughly halving
    effective bandwidth on PCIe-attached GPUs.

    Low bandwidth across all GPUs suggests PCIe saturation or NUMA penalty.
    A large drop from 1-GPU to 4-GPU on the same node indicates PCIe contention
    because all GPUs share the same root complex lanes.
    """
    section(f"H2D Transfer Bandwidth — {label}")

    # Per-device bandwidth summary
    cur.execute("""
        SELECT deviceId, COUNT(*), SUM(end-start)/1e6, SUM(bytes)/1e9
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE copyKind=1
        GROUP BY deviceId
        ORDER BY deviceId
    """)
    rows = cur.fetchall()
    out = []
    for r in rows:
        # bandwidth = GB transferred / seconds spent transferring
        bw = r[3] / (r[2] / 1e3) if r[2] else 0
        out.append((f"GPU{r[0]}", f"{r[1]:,}", f"{r[2]:.0f}", f"{r[3]:.2f}", f"{bw:.1f}"))
    tbl(out, ["device", "transfers", "time_ms", "data_GB", "bw_GB/s"])

    # Source memory kind breakdown — tells us whether pin_memory=True was used
    print(f"\n  H2D source memory kind (all devices combined):")
    cur.execute("""
        SELECT e.label, COUNT(*), SUM(m.bytes)/1e9
        FROM CUPTI_ACTIVITY_KIND_MEMCPY m
        LEFT JOIN ENUM_CUDA_MEM_KIND e ON e.id = m.srcKind
        WHERE m.copyKind = 1
        GROUP BY m.srcKind
        ORDER BY COUNT(*) DESC
    """)
    rows = cur.fetchall()
    out2 = [(str(r[0]), f"{r[1]:,}", f"{r[2]:.2f}") for r in rows]
    tbl(out2, ["src_memory_kind", "transfers", "data_GB"])


def nccl_kernels(cur, label):
    """
    NCCL collective communication kernels (AllReduce, AllGather, Broadcast, etc.).

    These appear during DDP gradient sync (training) or tensor-parallel inference.
    For single-process data-parallel inference they should be absent. If present and
    large, inter-GPU communication is a bottleneck — compare avg_us against compute
    kernel avg to see whether comms dominate.
    """
    section(f"NCCL Collective Kernels — {label}")
    if not _table_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
        print("  Kernel table not available — skipped.")
        return
    cur.execute("""
        SELECT s.value,
               COUNT(*),
               SUM(end-start)/1e6,
               AVG(end-start)/1000.0,
               MAX(end-start)/1000.0
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.demangledName
        WHERE s.value LIKE '%nccl%'
           OR s.value LIKE '%Nccl%'
           OR s.value LIKE '%NCCL%'
        GROUP BY s.value
        ORDER BY SUM(end-start) DESC
        LIMIT 15
    """)
    rows = cur.fetchall()
    if not rows:
        print("  No NCCL kernels found")
        return
    out = [(r[0][:55], f"{r[1]:,}", f"{r[2]:.0f}", f"{r[3]:.0f}", f"{r[4]:.0f}")
           for r in rows]
    tbl(out, ["kernel (55 chars)", "count", "total_ms", "avg_us", "max_us"])


def sync_breakdown(cur, label):
    """
    Synchronization stall time broken down by CUPTI sync type.

    CUPTI records the wall time between a sync call being issued on the CPU
    and the GPU actually reaching that point. High 'Stream sync' or 'Context sync'
    totals mean the CPU is blocking frequently, waiting for the GPU to drain its
    work queue — a sign of fine-grained step-by-step dispatch (no pipelining).

    Sync types:
      Event sync        = cudaEventSynchronize()
      Stream wait sync  = cudaStreamWaitEvent() — non-blocking inter-stream dep
      Stream sync       = cudaStreamSynchronize() — CPU blocks on stream drain
      Context sync      = cudaDeviceSynchronize() — CPU blocks on all streams
    """
    section(f"Synchronisation Stall Breakdown — {label}")
    cur.execute("""
        SELECT e.label, COUNT(*), SUM(s.end-s.start)/1e6
        FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION s
        LEFT JOIN ENUM_CUPTI_SYNC_TYPE e ON e.id = s.syncType
        GROUP BY s.syncType
        ORDER BY SUM(s.end-s.start) DESC
    """)
    rows = cur.fetchall()
    out = [(str(r[0]), f"{r[1]:,}", f"{r[2]:.0f}") for r in rows]
    tbl(out, ["sync_type", "count", "total_ms"])


def idle_gap_distribution(cur, label):
    """
    Distribution of time gaps between consecutive GPU kernels on GPU0.

    A 'gap' is the time from when one kernel ends to when the next begins.
    Positive gaps are true idle periods — the GPU was not executing anything.

    What the histogram tells you:
      Many small gaps (<=10ms)  → normal dispatch latency; healthy pipelining.
      Many large gaps (>100ms)  → CPU-side stalls or serialised step dispatch;
                                  the GPU is sitting idle waiting for the CPU to
                                  queue the next batch of work.
      A few very large gaps (>1s) → data loading, checkpoint I/O, or barrier waits
                                    between inference steps.

    Cumulative idle in long gaps vs (window_ms - active_ms) shows how much of
    the GPU's idle time is explained by these stalls.
    """
    section(f"GPU0 Inter-Kernel Idle Gaps — {label}")
    if not _table_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
        print("  Kernel table not available — skipped.")
        return
    cur.execute("""
        SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE deviceId=0
        ORDER BY start
    """)
    kernels = cur.fetchall()
    if len(kernels) < 2:
        print("  Not enough kernels to compute gaps.")
        return

    # Compute gap between end of kernel[i] and start of kernel[i+1]
    gaps = [(kernels[i + 1][0] - kernels[i][1]) / 1e6
            for i in range(len(kernels) - 1)]
    # Negative gaps mean overlapping kernels on different streams; keep only positive idle
    pos = [g for g in gaps if g > 0]
    if not pos:
        print("  No positive gaps found.")
        return

    long = [g for g in pos if g > 10]
    print(f"  Total gaps measured : {len(pos):,}")
    print(f"  Median              : {statistics.median(pos):.3f} ms")
    print(f"  Mean                : {statistics.fmean(pos):.3f} ms")
    print(f"  Max                 : {max(pos):.1f} ms")
    print(f"  Gaps > 10 ms        : {len(long):,}  (cumulative idle: {sum(long):.0f} ms)")

    buckets = {"<=10ms": 0, "10-50ms": 0, "50-100ms": 0,
               "100-500ms": 0, "500ms-1s": 0, ">1s": 0}
    for g in pos:
        if   g >  1000: buckets[">1s"]        += 1
        elif g >   500: buckets["500ms-1s"]   += 1
        elif g >   100: buckets["100-500ms"]  += 1
        elif g >    50: buckets["50-100ms"]   += 1
        elif g >    10: buckets["10-50ms"]    += 1
        else:           buckets["<=10ms"]     += 1
    tbl([(k, f"{v:,}") for k, v in buckets.items()], ["gap_bucket", "count"])


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run(path: Path):
    """Load one SQLite export and print all analysis sections."""
    label = path.stem
    print(f"\n{'#'*60}")
    print(f"#  {label}")
    print(f"{'#'*60}")

    con = sqlite3.connect(path)
    cur = con.cursor()

    gpu_utilization(cur, label)
    h2d_bandwidth(cur, label)
    nccl_kernels(cur, label)
    sync_breakdown(cur, label)
    idle_gap_distribution(cur, label)

    con.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_nsys.py <file1.sqlite> [<file2.sqlite> ...]")
        sys.exit(1)
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.exists():
            print(f"File not found: {p}")
            continue
        run(p)


if __name__ == "__main__":
    main()
