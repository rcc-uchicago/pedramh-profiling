#!/usr/bin/env python3
"""Verify benchmark report numbers against raw nsys SQLite files.

Standalone analysis utility (predates the Lightning port; see the
``bench-instrumentation`` history). For each exported nsys ``.sqlite`` profile in
:data:`FILES` it prints GPU utilisation, H2D/D2H bandwidth, NCCL kernel counts,
transfer-size distributions and inter-kernel gap histograms, so the numbers in
the benchmark report can be re-derived from the raw CUPTI activity tables.

Note:
    The paths in :data:`FILES` are frozen pointers to nsys exports from an
    earlier capture campaign and are **not** the HDF5 dataset path. Several no
    longer resolve on this cluster; :func:`main` skips any that are missing
    (printing ``MISSING``) rather than failing. Repoint :data:`FILES` at a local
    ``.sqlite`` export before running.
"""
import sqlite3
import statistics
from pathlib import Path

FILES = {
    "DSI_1GPU": "/home/rmeht/Projects/S2S/dsi_h200_1gpu_inference.sqlite",
    "DSI_4GPU": "/home/rmeht/Projects/S2S/dsi_h200_4gpus_inference.sqlite",
    "NVIDIA_4GPU": "/home/rmeht/Projects/S2S/nvidia_h100_4gpus_inference.sqlite",
    # Post-fix Midway H200 test-partition captures (2026-05-29, standardized
    # bare-DSI nsys command, commit 3acb9b3; checkpoint guard 56f73fe means the
    # inference loop now runs and kernels are captured — earlier captures crashed
    # at restore_checkpoint and held only ~3-4k runtime events, no kernel table).
    "MIDWAY_INTEL_4GPU_POSTFIX": "/home/rmeht/Projects/S2S/test_partition_benchmarks/midway_h200_intel_4gpus_inference_50244406_2026-05-29.sqlite",
    "MIDWAY_AMD_4GPU_POSTFIX": "/home/rmeht/Projects/S2S/test_partition_benchmarks/midway_h200_amd_4gpus_inference_50244404_2026-05-29.sqlite",
    # pedramh-gpu H100 NVL inference, post-fix re-run (2026-05-31, job 50249569,
    # standardized midway_infer_nsys.sh). Fills the matrix's pedramh row.
    "PEDRAMH_H100_4GPU": "/home/rmeht/Projects/S2S/test_partition_benchmarks/midway_h100_4gpus_inference_50249569_2026-05-31.sqlite",
}


def gpu_util(con, label):
    """Print per-device GPU utilisation from the kernel activity table.

    Sums kernel active time over the captured window per ``deviceId`` and reports
    launches, active ms, window ms and the active/window utilisation percentage.

    Args:
        con: An open :class:`sqlite3.Connection` to the nsys export.
        label: Run label used in the printed header.
    """
    cur = con.cursor()
    cur.execute("""
        SELECT deviceId, COUNT(*) AS launches,
               SUM(end-start)/1e6 AS active_ms,
               (MAX(end)-MIN(start))/1e6 AS window_ms
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        GROUP BY deviceId ORDER BY deviceId
    """)
    rows = cur.fetchall()
    print(f"\n=== GPU UTILIZATION: {label} ===")
    print(f"  {'device':<8} {'launches':<10} {'active_ms':<12} {'window_ms':<12} {'util':<8}")
    for r in rows:
        util = r[2]/r[3]*100 if r[3] else 0
        print(f"  GPU{r[0]:<5} {r[1]:<10,} {r[2]:<12.1f} {r[3]:<12.1f} {util:<8.2f}%")


def h2d_bandwidth(con, label):
    """Print per-device host-to-device (H2D) copy bandwidth.

    Aggregates ``copyKind=1`` (host-to-device) memcpy activity per ``deviceId``
    and reports transfer count, total time, total GB and effective GB/s.

    Args:
        con: An open :class:`sqlite3.Connection` to the nsys export.
        label: Run label used in the printed header.
    """
    cur = con.cursor()
    cur.execute("""
        SELECT deviceId, COUNT(*), SUM(end-start)/1e6, SUM(bytes)/1e9
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE copyKind=1
        GROUP BY deviceId ORDER BY deviceId
    """)
    rows = cur.fetchall()
    print(f"\n=== H2D BANDWIDTH: {label} ===")
    print(f"  {'device':<8} {'transfers':<10} {'time_ms':<10} {'data_GB':<10} {'bw_GB/s':<10}")
    for r in rows:
        bw = r[3]/(r[2]/1e3) if r[2] else 0
        print(f"  GPU{r[0]:<5} {r[1]:<10,} {r[2]:<10.1f} {r[3]:<10.2f} {bw:<10.2f}")


def d2h_bandwidth(con, label):
    """Print per-device device-to-host (D2H) copy bandwidth.

    Aggregates ``copyKind=2`` (device-to-host) memcpy activity per ``deviceId``
    and reports transfer count, total time, total GB and effective GB/s -- the
    D2H counterpart of :func:`h2d_bandwidth`.

    Args:
        con: An open :class:`sqlite3.Connection` to the nsys export.
        label: Run label used in the printed header.
    """
    cur = con.cursor()
    cur.execute("""
        SELECT deviceId, COUNT(*), SUM(end-start)/1e6, SUM(bytes)/1e9
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE copyKind=2
        GROUP BY deviceId ORDER BY deviceId
    """)
    rows = cur.fetchall()
    print(f"\n=== D2H BANDWIDTH: {label} ===")
    for r in rows:
        bw = r[3]/(r[2]/1e3) if r[2] else 0
        print(f"  GPU{r[0]}  {r[1]:,} transfers  {r[2]:.1f} ms  {r[3]:.2f} GB  {bw:.2f} GB/s")


def gap_distribution(con, label, device_id=0):
    """Print the inter-kernel idle-gap histogram for one device.

    Orders kernels by start time, computes the positive gaps between consecutive
    kernels, and buckets them (``<=10ms`` .. ``>1s``) -- surfacing the long
    (> 500 ms) stalls the report attributes to I/O or barrier waits.

    Args:
        con: An open :class:`sqlite3.Connection` to the nsys export.
        label: Run label used in the printed header.
        device_id: The ``deviceId`` to analyse (default ``0``).
    """
    cur = con.cursor()
    cur.execute("""
        SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE deviceId=?
        ORDER BY start
    """, (device_id,))
    kernels = cur.fetchall()
    if len(kernels) < 2:
        return
    gaps = [(kernels[i+1][0] - kernels[i][1]) / 1e6 for i in range(len(kernels)-1)]
    pos = [g for g in gaps if g > 0]
    print(f"\n=== GPU{device_id} GAP DISTRIBUTION: {label} ===")
    print(f"  Total positive gaps: {len(pos):,}")
    buckets = {"<=10ms": 0, "10-50ms": 0, "50-100ms": 0,
               "100-500ms": 0, "500ms-1s": 0, ">1s": 0}
    bucket_sums = {k: 0.0 for k in buckets}
    for g in pos:
        if g > 1000:
            buckets[">1s"] += 1; bucket_sums[">1s"] += g
        elif g > 500:
            buckets["500ms-1s"] += 1; bucket_sums["500ms-1s"] += g
        elif g > 100:
            buckets["100-500ms"] += 1; bucket_sums["100-500ms"] += g
        elif g > 50:
            buckets["50-100ms"] += 1; bucket_sums["50-100ms"] += g
        elif g > 10:
            buckets["10-50ms"] += 1; bucket_sums["10-50ms"] += g
        else:
            buckets["<=10ms"] += 1; bucket_sums["<=10ms"] += g
    long_gaps = [g for g in pos if g > 10]
    for k in buckets:
        print(f"  {k:<14} count={buckets[k]:<6} sum_ms={bucket_sums[k]:.1f}")
    print(f"  Gaps > 10 ms: count={len(long_gaps)}  cumulative_ms={sum(long_gaps):.1f}")
    # Report uses "> 500ms (I/O or barrier stalls)" - sum 500ms-1s and >1s
    over_500 = sum(1 for g in pos if g > 500)
    print(f"  Gaps > 500 ms (combined): {over_500}")


def h2d_srckind(con, label):
    """Print H2D transfer volume grouped by source memory kind.

    Joins ``copyKind=1`` memcpys to ``ENUM_CUDA_MEM_KIND`` on ``srcKind`` to show
    whether H2D copies originate from pageable or pinned host memory (the pinned
    fraction is what ``pin_memory=True`` buys).

    Args:
        con: An open :class:`sqlite3.Connection` to the nsys export.
        label: Run label used in the printed header.
    """
    cur = con.cursor()
    cur.execute("""
        SELECT e.label, COUNT(*), SUM(m.bytes)/1e9
        FROM CUPTI_ACTIVITY_KIND_MEMCPY m
        LEFT JOIN ENUM_CUDA_MEM_KIND e ON e.id = m.srcKind
        WHERE m.copyKind = 1
        GROUP BY m.srcKind
        ORDER BY COUNT(*) DESC
    """)
    rows = cur.fetchall()
    print(f"\n=== H2D SRC MEM KIND: {label} ===")
    for r in rows:
        print(f"  {r[0]}: count={r[1]:,}  bytes={r[2]:.2f} GB")


def per_gpu_h2d_srckind(con, label):
    """Print H2D transfer volume by source memory kind, split per device.

    The per-``deviceId`` breakdown of :func:`h2d_srckind`, exposing whether a
    single GPU dominates the pageable-source H2D traffic.

    Args:
        con: An open :class:`sqlite3.Connection` to the nsys export.
        label: Run label used in the printed header.
    """
    cur = con.cursor()
    cur.execute("""
        SELECT m.deviceId, e.label, COUNT(*), SUM(m.bytes)/1e9
        FROM CUPTI_ACTIVITY_KIND_MEMCPY m
        LEFT JOIN ENUM_CUDA_MEM_KIND e ON e.id = m.srcKind
        WHERE m.copyKind = 1
        GROUP BY m.deviceId, m.srcKind
        ORDER BY m.deviceId, COUNT(*) DESC
    """)
    rows = cur.fetchall()
    print(f"\n=== H2D PER-GPU SRC KIND: {label} ===")
    for r in rows:
        print(f"  GPU{r[0]}  {r[1]}: count={r[2]:,}  bytes={r[3]:.2f} GB")


def nccl_kernels(con, label):
    """Print the count of NCCL collective kernels in the capture.

    Joins the kernel table to ``StringIds`` and counts demangled names matching
    ``nccl`` (any case) -- a quick check for whether DDP gradient all-reduce
    kernels are present in the profile.

    Args:
        con: An open :class:`sqlite3.Connection` to the nsys export.
        label: Run label used in the printed header.
    """
    cur = con.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.demangledName
        WHERE s.value LIKE '%nccl%' OR s.value LIKE '%Nccl%' OR s.value LIKE '%NCCL%'
    """)
    n = cur.fetchone()[0]
    print(f"\n=== NCCL KERNELS: {label} === count={n}")


def transfer_size_distribution(con, label):
    """Group transfers by size to identify ~165 MB tensors."""
    cur = con.cursor()
    cur.execute("""
        SELECT bytes, COUNT(*)
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE copyKind=1
        GROUP BY bytes
        ORDER BY bytes DESC
        LIMIT 20
    """)
    rows = cur.fetchall()
    print(f"\n=== H2D LARGEST TRANSFER SIZES: {label} ===")
    for r in rows:
        mb = r[0] / 1e6
        print(f"  size={r[0]} bytes ({mb:.2f} MB)  count={r[1]}")


def transfer_size_buckets(con, label):
    """Count H2D transfers > 100 MB per device."""
    cur = con.cursor()
    cur.execute("""
        SELECT deviceId, COUNT(*), SUM(bytes)/1e9
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE copyKind=1 AND bytes > 100000000
        GROUP BY deviceId ORDER BY deviceId
    """)
    rows = cur.fetchall()
    print(f"\n=== H2D LARGE (>100 MB) TRANSFERS PER GPU: {label} ===")
    for r in rows:
        print(f"  GPU{r[0]}: count={r[1]} sum={r[2]:.2f} GB")


def main():
    """Run every analysis section over each existing profile in :data:`FILES`.

    Iterates :data:`FILES`, skipping (and printing ``MISSING`` for) any path that
    does not resolve on this filesystem -- see the module docstring's note on the
    frozen paths -- and runs the full suite (utilisation, H2D/D2H bandwidth,
    source-kind breakdowns, NCCL count, transfer-size and per-device gap
    distributions) on each that does.
    """
    for label, path in FILES.items():
        if not Path(path).exists():
            print(f"MISSING: {path}")
            continue
        print(f"\n\n##############################")
        print(f"#  {label}: {path}")
        print(f"##############################")
        con = sqlite3.connect(path)
        gpu_util(con, label)
        h2d_bandwidth(con, label)
        d2h_bandwidth(con, label)
        h2d_srckind(con, label)
        per_gpu_h2d_srckind(con, label)
        nccl_kernels(con, label)
        transfer_size_distribution(con, label)
        transfer_size_buckets(con, label)
        for dev in range(4):
            gap_distribution(con, label, dev)
        con.close()


if __name__ == "__main__":
    main()
