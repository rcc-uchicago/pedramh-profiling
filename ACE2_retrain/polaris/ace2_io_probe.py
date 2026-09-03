#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
"""App-free read probe for the ACE2 NetCDF: what does that ONE OST actually give us?

    python3 ace2_io_probe.py --config <config_polaris.yaml> [--readers 1,2,4,8,16,32]

WHY THIS EXISTS
---------------
The 1- and 2-node training arms demanded **357-512 MB/s** from a file that
`lfs getstripe` says lives on a SINGLE OST (`lmm_stripe_count: 1`), and held
`gpu_busy_frac` at 0.93-0.96. That was read as "the loader is not the
bottleneck". It does not support that conclusion, for one specific reason:

    **nobody has measured what that OST delivers.** 512 MB/s could be 15% of its
    ceiling or 90% of it, and the two lead to opposite decisions about the
    2.4 TB -> zarr conversion.

⚠ And the 2-node arm was a WEAKER loader test than the 1-node one, not a
stronger one: per-rank demand FELL from 101.5 to 64.0 MB/s because NCCL stretched
the step by 68% and handed the loader more time. Total demand rose only
406 -> 512 MB/s. So the ladder's own arms cannot answer this; they confound
"the OST kept up" with "the fabric slowed everything down".

WHAT IT MEASURES
----------------
The real access pattern, with no model, no CUDA and no fme: each reader process
draws random 3-timestep windows (`n_forward_steps=2` + 1 initial condition) and
reads every variable the config names, exactly as `XarrayDataset` does. Sweeping
the reader count finds the KNEE -- the point where adding readers stops adding
bandwidth -- and the training arms' demand is printed on the same axis.

The file is contiguous and uncompressed (verified: `chunks=None`,
`compression=None`), so a 3-timestep window of one variable is one ~778 KB
contiguous read at a random offset, and there is no chunk amplification or
decompression cost to model away.

PAGE CACHE: the file is 2.4 TB against 512 GB of node RAM, and the training arms
have touched under ~60 GB of it, so a uniformly random draw hits cache with
probability ~2%. Reported anyway (`--repeat` re-draws with a different seed) so
the effect can be seen rather than assumed.

PASS = ``ACE2_IO_PROBE_OK``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time


def _is_time_varying(dset, n_time: int) -> bool:
    """True when the leading dimension is time.

    ⚠ Not `ndim == 3`. The config's 56 variables are a mix: (time, lat, lon)
    fields, a (lat, lon) time-invariant field (`HGTsfc`), and at least one 1-D
    (time,) scalar series (`global_mean_co2`). Keying on rank instead of on the
    leading dimension is what made the first run of this probe die with
    `IndexError: tuple index out of range` (job 7586630).
    """
    return dset.ndim >= 1 and dset.shape[0] == n_time


def _window_bytes(dset, n_time: int, window: int) -> int:
    """Bytes one sample pulls from this variable."""
    n = dset.dtype.itemsize
    if _is_time_varying(dset, n_time):
        for s in dset.shape[1:]:      # empty for a (time,) series -> itemsize
            n *= int(s)
        return n * window
    for s in dset.shape:              # static: read once per sample
        n *= int(s)
    return n


def _read_worker(args):
    """One reader: draw `n` random windows, return (bytes, seconds, latencies)."""
    path, names, n_windows, window, seed, n_time = args
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    import h5py

    rng = random.Random(seed)
    nbytes = 0
    lat = []
    with h5py.File(path, "r") as f:
        # Resolve once; the per-window cost is what we are measuring.
        dsets = [(f[n], _is_time_varying(f[n], n_time)) for n in names if n in f]
        t0 = time.perf_counter()
        for _ in range(n_windows):
            t = rng.randrange(0, max(1, n_time - window))
            w0 = time.perf_counter()
            for d, tv in dsets:
                a = d[t:t + window] if tv else d[...]
                nbytes += a.nbytes
            lat.append(time.perf_counter() - w0)
        dt = time.perf_counter() - t0
    return nbytes, dt, lat


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", default="/eagle/projects/lighthouse-uchicago/ace2/"
                                     "ace_training/merged_ACE2_ERA5_final.nc")
    p.add_argument("--config", required=True, help="config_polaris.yaml (for the variable list)")
    p.add_argument("--readers", default="1,2,4,8,16,32")
    p.add_argument("--windows", type=int, default=24, help="random windows per reader")
    p.add_argument("--window", type=int, default=3, help="timesteps per sample")
    p.add_argument("--repeat", type=int, default=1, help="re-run each point with a new seed")
    args = p.parse_args(argv)

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    import h5py
    import yaml

    cfg = yaml.safe_load(open(args.config))
    step = cfg["stepper"]["step"]["config"]
    names = sorted(set(step["in_names"]) | set(step["out_names"]))

    with h5py.File(args.path, "r") as f:
        missing = [n for n in names if n not in f]
        if missing:
            print("ERROR ACE2_IO_PROBE_VARS_MISSING: %s" % missing)
            return 3
        # n_time from the longest leading dimension across the 3-D fields, not
        # from names[0] -- the variable list is sorted alphabetically and its
        # first entry is not guaranteed to be time-varying.
        n_time = max((f[n].shape[0] for n in names if f[n].ndim == 3), default=0)
        if n_time == 0:
            print("ERROR ACE2_IO_PROBE_NO_TIME_DIM: no 3-D field among %d variables"
                  % len(names))
            return 3
        sample_bytes = sum(_window_bytes(f[n], n_time, args.window) for n in names)
        n_static = sum(1 for n in names if not _is_time_varying(f[n], n_time))

    print("=== ACE2 single-OST read probe ===")
    print("  file        %s" % args.path)
    print("  variables   %d (%d time-invariant)   n_time %d"
          % (len(names), n_static, n_time))
    print("  per sample  %.2f MB (%d timesteps)" % (sample_bytes / 1e6, args.window))
    print("  windows/reader %d   repeats %d" % (args.windows, args.repeat))
    print()
    print("  %-8s %-10s %-12s %-12s %-10s %-10s %-8s" %
          ("readers", "MB/s", "per-reader", "med window", "p90 window", "cache hit",
           "scaling"))
    print("  (MB/s and latencies are the COLDEST repeat; warm repeats shown alongside)")

    # ⚠ SEEDS MUST BE GLOBALLY DISJOINT, and the headline must be the COLDEST
    # repeat, not the best.
    #
    # The first version of this probe did neither, and it fabricated its own
    # headline (job 7586642). Seeds were `1000*nr + 7*rep + i`, so the two
    # repeats overlapped by 0% at 1-4 readers, 12.5% at 8, 56% at 16 and 78% at
    # 32 -- the second repeat re-read what the first had just pulled into a
    # 497 GB page cache. Taking the *best* of the two then systematically
    # rewarded the warm repeat, and did so MORE at higher reader counts, which
    # manufactured exactly the linear scaling the probe was built to test
    # (an apparent 636.7 MB/s peak).
    #
    # The tell was internal and is now checked automatically: median window
    # latency collapsed 1.93 s -> 0.012 s (160x) while per-reader MB/s stayed
    # flat at ~20. Both cannot be true of the same reads.
    _seed = [0]

    def next_seed():
        _seed[0] += 1
        return _seed[0] * 7919

    reader_counts = [int(x) for x in args.readers.split(",") if x.strip()]
    rows = []
    base = None
    # A cold read of one window is ~2 s on this file (56 seeks on HDD-backed
    # Lustre); anything two orders of magnitude faster than that came from RAM.
    CACHE_MS = 100.0
    for nr in reader_counts:
        reps = []
        for _rep in range(args.repeat):
            work = [(args.path, names, args.windows, args.window,
                     next_seed(), n_time) for _ in range(nr)]
            t0 = time.perf_counter()
            with mp.Pool(nr) as pool:
                out = pool.map(_read_worker, work)
            wall = time.perf_counter() - t0
            total = sum(o[0] for o in out)
            lat = sorted(x for o in out for x in o[2])
            reps.append((total / 1e6 / wall, lat, wall))
        # COLDEST = first repeat. The others are printed so the cache effect is
        # visible rather than averaged away.
        mbs, lat, wall = reps[0]
        med = lat[len(lat) // 2]
        p90 = lat[min(len(lat) - 1, int(0.90 * len(lat)))]
        hit = sum(1 for x in lat if x * 1000 < CACHE_MS) / len(lat)
        if base is None:
            base = mbs
        warm = "  warm reps: " + ", ".join("%.1f" % r[0] for r in reps[1:]) if len(reps) > 1 else ""
        rows.append({"readers": nr, "mb_s": round(mbs, 1),
                     "per_reader_mb_s": round(mbs / nr, 1),
                     "median_window_s": round(med, 4),
                     "p90_window_s": round(p90, 4),
                     "cache_hit_frac": round(hit, 3),
                     "warm_repeats_mb_s": [round(r[0], 1) for r in reps[1:]]})
        print("  %-8d %-10.1f %-12.1f %-12.4f %-10.4f %-9.1f%% %-8.2fx%s"
              % (nr, mbs, mbs / nr, med, p90, 100 * hit, mbs / base, warm))

    # ---- self-consistency: does the latency agree with the throughput? ------
    # A per-reader rate derived from the median window must match the measured
    # one. When it does not, the aggregate is being set by something other than
    # the reads -- cache hits, or process startup dominating a short run.
    suspect = []
    for r in rows:
        if r["median_window_s"] <= 0:
            continue
        implied = (sample_bytes / 1e6) / r["median_window_s"]
        if implied > 4 * r["per_reader_mb_s"] or r["cache_hit_frac"] > 0.25:
            suspect.append((r["readers"], implied, r["per_reader_mb_s"],
                            r["cache_hit_frac"]))
    if suspect:
        print()
        print("  ⚠ WARN PROBE_INCONSISTENT -- these points are NOT clean OST measurements:")
        for nr, implied, measured, hit in suspect:
            print("      readers=%-3d median latency implies %.0f MB/s/reader but %.1f was "
                  "measured; %.0f%% of windows were cache hits"
                  % (nr, implied, measured, 100 * hit))
        print("      Re-run with a colder cache or fewer prior points before quoting a peak.")

    clean = [r for r in rows if r["cache_hit_frac"] <= 0.25]
    peak = max((r["mb_s"] for r in clean), default=0.0)
    print()
    if clean:
        print("  PEAK aggregate over CLEAN points: %.1f MB/s (at %d readers)"
              % (peak, max(clean, key=lambda r: r["mb_s"])["readers"]))
        if clean[-1] is rows[-1]:
            print("  ⚠ the largest reader count was still clean and still scaling -- this is a")
            print("    LOWER BOUND on the OST, not its ceiling. Extend --readers to find the knee.")
        else:
            print("  ⚠ the highest reader counts were discarded as cache-contaminated, so this")
            print("    is a LOWER BOUND on the OST, not its ceiling.")
    else:
        print("  ERROR ACE2_IO_PROBE_NO_CLEAN_POINTS: every point was cache-contaminated.")
        return 3

    # The whole point: put the training arms on the same axis.
    print()
    print("  --- what the training arms demanded of this same OST ---")
    demand = [("1 node, local batch 1", 357.0), ("1 node, local batch 2", 406.0),
              ("2 nodes, local batch 2", 512.3)]
    for tag, d in demand:
        print("    %-24s %6.1f MB/s  = %5.1f%% of the clean LOWER BOUND" % (tag, d, 100 * d / peak))
    print("    %-24s %6.1f MB/s  = %5.1f%% (linear from the 1-node point)"
          % ("4 nodes, local batch 2", 4 * 406.0, 100 * 4 * 406.0 / peak))
    print("    %-24s %6.1f MB/s  = %5.1f%%"
          % ("8 nodes, local batch 2", 8 * 406.0, 100 * 8 * 406.0 / peak))
    print()
    print("  ⚠ The 4-/8-node rows are LINEAR EXTRAPOLATIONS of per-rank demand and")
    print("    will be optimistic if the fabric stretches the step (it did at 2 nodes:")
    print("    per-rank demand fell 101.5 -> 64.0 MB/s). They bound the risk, not the cost.")

    print(json.dumps({"peak_mb_s": peak, "rows": rows}))
    print("ACE2_IO_PROBE_OK peak_mb_s=%.1f" % peak)
    return 0


if __name__ == "__main__":
    sys.exit(main())
