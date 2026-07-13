#!/usr/bin/env python3
"""Adversarial review queries against the three SQLite profiles."""
import sqlite3
import statistics

FILES = {
    "DSI_1GPU": "/home/rmeht/Projects/S2S/dsi_h200_1gpu_inference.sqlite",
    "DSI_4GPU": "/home/rmeht/Projects/S2S/dsi_h200_4gpus_inference.sqlite",
    "NVIDIA_4GPU": "/home/rmeht/Projects/S2S/nvidia_h100_4gpus_inference.sqlite",
}


def gpu_util(cur):
    cur.execute("""
        SELECT deviceId, COUNT(*), SUM(end-start)/1e6, (MAX(end)-MIN(start))/1e6
        FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY deviceId ORDER BY deviceId
    """)
    return cur.fetchall()


def h2d_bw(cur):
    cur.execute("""
        SELECT deviceId, COUNT(*), SUM(end-start)/1e6, SUM(bytes)/1e9
        FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=1
        GROUP BY deviceId ORDER BY deviceId
    """)
    return cur.fetchall()


def src_kind(cur):
    cur.execute("""
        SELECT e.label, COUNT(*), SUM(m.bytes)/1e9, AVG(m.bytes)/1e6
        FROM CUPTI_ACTIVITY_KIND_MEMCPY m
        LEFT JOIN ENUM_CUDA_MEM_KIND e ON e.id = m.srcKind
        WHERE m.copyKind = 1
        GROUP BY m.srcKind ORDER BY COUNT(*) DESC
    """)
    return cur.fetchall()


def src_kind_per_dev(cur):
    cur.execute("""
        SELECT m.deviceId, e.label, COUNT(*), SUM(m.bytes)/1e9, AVG(m.bytes)/1e6
        FROM CUPTI_ACTIVITY_KIND_MEMCPY m
        LEFT JOIN ENUM_CUDA_MEM_KIND e ON e.id = m.srcKind
        WHERE m.copyKind = 1
        GROUP BY m.deviceId, m.srcKind
        ORDER BY m.deviceId, COUNT(*) DESC
    """)
    return cur.fetchall()


def large_pinned(cur):
    cur.execute("""
        SELECT m.deviceId, COUNT(*), AVG(m.bytes)/1e6, MIN(m.bytes)/1e6, MAX(m.bytes)/1e6
        FROM CUPTI_ACTIVITY_KIND_MEMCPY m
        WHERE m.copyKind = 1 AND m.bytes >= 100*1024*1024
        GROUP BY m.deviceId ORDER BY m.deviceId
    """)
    return cur.fetchall()


def gap_dist(cur, dev=0):
    cur.execute("""
        SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE deviceId=? ORDER BY start
    """, (dev,))
    kernels = cur.fetchall()
    if len(kernels) < 2:
        return None, None
    gaps = [(kernels[i+1][0] - kernels[i][1])/1e6 for i in range(len(kernels)-1)]
    pos = [g for g in gaps if g > 0]
    buckets = {"<=10ms": [], "10-50ms": [], "50-100ms": [],
               "100-500ms": [], "500ms-1s": [], ">1s": []}
    for g in pos:
        if   g >  1000: buckets[">1s"].append(g)
        elif g >   500: buckets["500ms-1s"].append(g)
        elif g >   100: buckets["100-500ms"].append(g)
        elif g >    50: buckets["50-100ms"].append(g)
        elif g >    10: buckets["10-50ms"].append(g)
        else:           buckets["<=10ms"].append(g)
    return buckets, pos


def main():
    for label, path in FILES.items():
        print(f"\n{'#'*70}\n# {label}: {path}\n{'#'*70}")
        con = sqlite3.connect(path)
        cur = con.cursor()

        print("\n-- gpu_util --")
        for r in gpu_util(cur):
            util = r[2]/r[3]*100 if r[3] else 0
            print(f"  GPU{r[0]}  launches={r[1]:,}  active_ms={r[2]:.1f}  window_ms={r[3]:.1f}  util={util:.2f}%")

        print("\n-- h2d_bandwidth --")
        for r in h2d_bw(cur):
            bw = r[3]/(r[2]/1e3) if r[2] else 0
            print(f"  GPU{r[0]}  transfers={r[1]:,}  time_ms={r[2]:.1f}  data_GB={r[3]:.2f}  bw_GBs={bw:.2f}")

        print("\n-- src_memory_kind (all devices) --")
        for r in src_kind(cur):
            print(f"  kind={r[0]}  transfers={r[1]:,}  data_GB={r[2]:.2f}  avg_MB={r[3]:.2f}")

        print("\n-- src_memory_kind per-device --")
        for r in src_kind_per_dev(cur):
            print(f"  GPU{r[0]}  kind={r[1]}  transfers={r[2]:,}  data_GB={r[3]:.2f}  avg_MB={r[4]:.2f}")

        print("\n-- large H2D transfers (>=100MB) per device --")
        for r in large_pinned(cur):
            print(f"  GPU{r[0]}  count={r[1]}  avg_MB={r[2]:.1f}  min_MB={r[3]:.1f}  max_MB={r[4]:.1f}")

        print("\n-- inter-kernel gap distribution on GPU0 --")
        buckets, pos = gap_dist(cur, 0)
        if buckets is None:
            print("  (no data)")
        else:
            for k, lst in buckets.items():
                tot = sum(lst)
                avg = (sum(lst)/len(lst)) if lst else 0
                print(f"  {k}: count={len(lst):,}  total_ms={tot:.1f}  avg_ms={avg:.3f}")
            long_gaps = [g for g in pos if g > 10]
            print(f"  TOTAL gaps>10ms idle: {sum(long_gaps):.1f} ms  count={len(long_gaps)}")
            b = buckets["10-50ms"]
            print(f"  Average of 10-50ms bucket: {(sum(b)/len(b)) if b else 0:.3f} ms")

        con.close()


if __name__ == "__main__":
    main()
