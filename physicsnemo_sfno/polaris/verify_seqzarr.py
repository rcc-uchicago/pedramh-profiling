#!/usr/bin/env python3
"""EXHAUSTIVELY verify a SeqZarr store against the E3SM h5 it was converted from.

Why this exists
---------------
`e3sm_h5_to_seqzarr.py`'s end-of-run gate prints CONVERT_OK after checking **one channel of
one sample** — 1 of ~10,000 channel-samples. It is deliberately cheap; this is the real
check. Its blind spots, each a plausible bug:

  * its probe channel is chosen as one with **no NAN_FILL entry**, so the fill — the only
    science decision in the converter — is never verified there.
  * `buf_p` is **reused across samples** and only sample 0 is probed, so a failed
    `read_direct` on sample 37 would leave sample 36's data in the buffer and still pass.

The opportunity this takes: a smoke store is ~64 samples / 1.4 GB, so every value can be
checked against its source. At the full 51,100 samples (~1.1 TB) you can only ever sample.
**Verify exhaustively while it is still cheap — that is the point of a smoke store.**

    python verify_seqzarr.py --store <path>.zarr                  # PASS = SEQZARR_VERIFIED
    python verify_seqzarr.py --store <path>.zarr --stride 500     # sample a big store

Exit 0 = every check passed. Any failure exits 1 and names the sample+channel.

What this CANNOT prove
----------------------
1. **That the fill VALUES are right.** NAN_FILL is imported from the converter, so this
   verifies the store against the converter's own intent. Change SST's fill to 0.0 in the
   dict and this still passes — by construction. A wrong constant is caught by review, or
   not at all. That is why the converter records nan_fill into the store's attrs.
2. **That a green smoke generalises.** A smoke store is ~16 days of January 2015. It cannot
   speak for 2016-2049 — and the three worst converter bugs found on 2026-07-15 were each
   invisible at exactly smoke scale (a frozen in-file year that is *correct* for 2015; a
   --max-samples default the smoke happens to pass explicitly; a zero-fill-on-interrupt trap
   a completed run never hits). Use --stride on the FULL store, per year.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e3sm_h5_to_seqzarr import (  # noqa: E402
    E3SM_ROOT, NAN_FILL, UNPREDICTED, EPOCH, sample_files, read_time_hours, year_of,
)

class Checks:
    def __init__(self):
        self.failed = []

    def report(self, name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
        if not ok:
            self.failed.append(name)
        return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--src", type=Path, default=E3SM_ROOT / "h5/plev_data")
    ap.add_argument("--years", nargs="+", type=int, default=[2015])
    ap.add_argument("--start-sample", type=int, default=None,
                    help="default: read from the store's own attrs if present, else 0")
    ap.add_argument("--samples", type=int, default=None, help="default: ALL (the point)")
    ap.add_argument("--stride", type=int, default=1,
                    help="verify every Nth sample. For the ~1.1 TB full store, where "
                         "exhaustive is hours; --stride 500 spans all 35 years.")
    args = ap.parse_args()

    z = zarr.open(str(args.store), mode="r")
    c = Checks()

    pred_names = list(z.attrs["channels_predicted"])
    unpred_names = list(z.attrs["channels_unpredicted"])
    T, cp, h, w = z["predicted"].shape

    # The file list comes from the store's OWN RECORD of what went into it. Re-deriving it
    # (what this did first) is circular: a store built from the wrong --start-sample, or a
    # random draw, then verifies as a faithful copy of whatever slice it happens to hold.
    # Job 7257705 is the proof — a random-sampled store was compared against a contiguous
    # slice and reported 5,665 bogus mismatches.
    recorded = z.attrs.get("source_files")
    if recorded:
        files = [args.src / n for n in recorded]
        if len(files) != T:
            print(f"  warn  store records {len(files)} source files but holds {T} samples")
    else:
        start = args.start_sample if args.start_sample is not None else z.attrs.get("start_sample", 0)
        print("  warn  store records no source_files (pre-2026-07-15 converter): re-deriving a\n"
              "        CONTIGUOUS slice from --years/--start-sample. This CANNOT detect a wrong\n"
              "        offset, and is simply wrong for a randomly-sampled store.")
        years = z.attrs.get("years") or args.years
        files = sample_files(args.src, years, start, T)
    idx = list(range(0, T, args.stride))
    if args.samples is not None:
        idx = idx[:args.samples]
    n = len(idx)
    print(f"store: {args.store}")
    print(f"  {T} samples, grid {h}x{w}, predicted={cp} unpredicted={len(unpred_names)}")
    print(f"  verifying {n}/{T} samples against {args.src}"
          f"{'  (EXHAUSTIVE)' if n == T else '  (PARTIAL — not a full verification)'}\n")

    # ---- 1. channel map: the two lists must partition the h5's keys exactly -----------
    with h5py.File(files[0], "r") as f:
        h5_names = sorted(k for k in f["input"].keys() if k != "time")
    c.report("channel_map/partition",
             sorted(pred_names + unpred_names) == h5_names,
             f"{len(pred_names)}+{len(unpred_names)} vs {len(h5_names)} h5 keys")
    c.report("channel_map/disjoint", not (set(pred_names) & set(unpred_names)))
    c.report("channel_map/no_dupes",
             len(set(pred_names)) == len(pred_names) and len(set(unpred_names)) == len(unpred_names))
    c.report("channel_map/unpredicted_is_the_declared_set",
             set(unpred_names) == set(UNPREDICTED))
    c.report("shape/counts", cp == len(pred_names) and z["unpredicted"].shape[1] == len(unpred_names))
    if c.failed:
        # Bail before the round-trip: it indexes the h5 by these names, so a bogus one
        # raises an uncaught KeyError and the run dies with a traceback instead of a report.
        print(f"\nERROR SEQZARR_VERIFY_FAILED ({len(c.failed)}): {', '.join(c.failed)}")
        print("  channel map is broken — skipping the round-trip (it would KeyError).")
        return 1

    # ---- 2. exhaustive round-trip + fill placement ------------------------------------
    # For every sample and every channel:
    #   where the source is finite -> the store must hold the SAME BITS
    #   where the source is NaN    -> the store must hold exactly NAN_FILL[name]
    # This subsumes the fill check AND the cross-sample buffer-reuse bug in one pass.
    mism_val, mism_fill, bad_nan = [], [], []
    for k, i in enumerate(idx):
        zp = z["predicted"][i]            # (cp,h,w) — one chunk
        zu = z["unpredicted"][i]
        with h5py.File(files[i], "r") as f:
            g = f["input"]
            for arr, names in ((zp, pred_names), (zu, unpred_names)):
                for ci, nm in enumerate(names):
                    ref = g[nm][()]
                    nan = np.isnan(ref)
                    fin = ~nan
                    if not np.array_equal(arr[ci][fin], ref[fin]):
                        d = np.abs(arr[ci][fin].astype(np.float64) - ref[fin].astype(np.float64))
                        mism_val.append((i, nm, float(d.max())))
                    if nan.any():
                        want = NAN_FILL.get(nm)
                        if want is None:
                            bad_nan.append((i, nm))
                        elif not np.all(arr[ci][nan] == np.float32(want)):
                            got = np.unique(arr[ci][nan])[:3]
                            mism_fill.append((i, nm, want, got))
        if k % 16 == 0:
            print(f"    …sample {k}/{n} (store index {i})", flush=True)

    c.report("roundtrip/values_bitwise_identical", not mism_val,
             "" if not mism_val else f"{len(mism_val)} channel-samples differ, first: "
                                     f"sample {mism_val[0][0]} ch {mism_val[0][1]} max|d|={mism_val[0][2]:.3e}")
    c.report("roundtrip/fill_value_and_placement", not mism_fill,
             "" if not mism_fill else f"first: sample {mism_fill[0][0]} ch {mism_fill[0][1]} "
                                      f"want {mism_fill[0][2]} got {mism_fill[0][3]}")
    c.report("roundtrip/every_nan_has_a_fill", not bad_nan,
             "" if not bad_nan else f"{len(bad_nan)}, first: {bad_nan[0]}")

    # ---- 3. no NaN survived, over EVERY sample (not just sample 0) --------------------
    finite = all(bool(np.isfinite(z["predicted"][i]).all()
                      and np.isfinite(z["unpredicted"][i]).all()) for i in idx)
    c.report("store/all_finite_every_sample", finite)

    # ---- 4. time axis: value AND spacing, not just monotonicity -----------------------
    t = z["time"][:].astype(np.int64)[idx]
    want_t = np.array([_safe_time(files[i]) for i in idx], dtype=np.int64)
    c.report("time/matches_source_file", np.array_equal(t, want_t),
             "" if np.array_equal(t, want_t) else f"first mismatch at {int(np.argmax(t != want_t))}")
    full_t = z["time"][:].astype(np.int64)
    d = np.unique(np.diff(full_t)) if T > 1 else np.array([6])
    if z.attrs.get("sampling_mode") == "random":
        # A random fixture skips around the archive by design: gaps are the point, and
        # demanding 6h spacing would fail every correct one. Monotonic is still required —
        # a non-monotonic axis would mean the year reconstruction is broken.
        c.report("time/monotonic (random fixture: gaps expected, 6h spacing not required)",
                 bool(np.all(d > 0)), f"{len(d)} distinct gaps, min {d.min()}h max {d.max()}h")
    else:
        c.report("time/uniform_6h_spacing_over_whole_store", d.tolist() == [6], f"diffs={d.tolist()}")

    # ---- 5. grid ---------------------------------------------------------------------
    lat, lon = z["latitude"][:], z["longitude"][:]
    c.report("grid/lat_ascending_matches_e3sm",
             bool(np.allclose(lat, np.linspace(-89.5, 89.5, h))),
             f"lat[0]={lat[0]:.2f} lat[-1]={lat[-1]:.2f}")
    # E3SM's longitudes are cell centres 0.5..359.5; the converter writes arange(360).
    lon_ok = bool(np.allclose(lon, np.arange(0.5, 360.0, 1.0)))
    c.report("grid/lon_matches_e3sm_cell_centres", lon_ok,
             "" if lon_ok else f"store lon starts {lon[0]:.1f}; E3SM cell centres start 0.5 "
                               "(0.5 deg offset — metadata only, the model reads the array)")

    # ---- 6. no dead metadata ----------------------------------------------------------
    # The converter used to write means_/stds_{predicted,unpredicted} copied from
    # normalize_*.npz. Nothing ever read them (the datapipe asks only for
    # time/predicted/unpredicted; train.py normalizes with BatchNorm2d) and the SST entry
    # was computed under a different fill convention, so it was dead AND wrong — it fooled
    # two independent auditors. It is gone. Warn, don't fail: a store written by the old
    # converter can still have perfect DATA, and failing it over dead metadata would train
    # people to ignore this tool.
    dead = [k for k in ("means_predicted", "stds_predicted",
                        "means_unpredicted", "stds_unpredicted") if k in set(z.array_keys())]
    if dead:
        print(f"  warn  store carries {len(dead)} vestigial stats array(s): {', '.join(dead)}")
        print("        Written by a pre-2026-07-15 converter. Nothing reads them and their SST")
        print("        entry is inconsistent with this store's own fill — do not trust them.")

    print()
    if c.failed:
        print(f"ERROR SEQZARR_VERIFY_FAILED ({len(c.failed)}): {', '.join(c.failed)}")
        return 1
    scope = "EXHAUSTIVE" if n == T else f"PARTIAL ({n}/{T}, stride {args.stride})"
    print(f"SEQZARR_VERIFIED ({scope}: {n} samples x {cp + len(unpred_names)} channels "
          f"= {n * (cp + len(unpred_names))} channel-samples, bitwise)")
    return 0


def _safe_time(p):
    # MUST pass the year, exactly as the converter does. Reading the raw in-file stamp here
    # would compare the store's (correct) year-offset axis against the archive's frozen-2015
    # label — failing every correct multi-year store, AND silently PASSING a legacy store
    # whose axis is off by a whole year, because both sides would derive from the same
    # frozen label. A check that agrees with the bug is worse than no check.
    with h5py.File(p, "r") as f:
        return read_time_hours(f, year=year_of(p), index_in_year=int(p.stem.split("_")[1]))


if __name__ == "__main__":
    sys.exit(main())
