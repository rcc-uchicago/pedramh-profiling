#!/usr/bin/env python
"""E3SM per-sample HDF5 -> SeqZarr store for PhysicsNeMo unified_recipe SFNO training.

Target: <root>/predicted (T,157,180,360) + unpredicted (T,5,180,360) + time (T,) i8 +
lat/lon, read by examples/weather/unified_recipe/seq_zarr_datapipe.py (SeqZarrSource does
zarr.open(store)["<array>"][time_idx]; axis 0 = time). Static / prescribed-forcing fields
go to "unpredicted"; the other 157 channels are "predicted". time is int hours-since-epoch
(DALI can't ingest datetime64/bytes).

This does exactly TWO things: a layout rewrite (per-sample files x per-variable datasets ->
time-major chunked arrays) and a NaN fill. The layout is mechanical; the fill is the only
judgment call in here, and it is spelled out in NAN_FILL below.

It deliberately writes **no normalization statistics** — see the note on NAN_FILL. Verify a
store with polaris/verify_seqzarr.py (bitwise, every sample x every channel).

Micro-tested on Polaris (base conda, zarr 2.18.7): 6-sample store -> max|zarr-h5|=0 + CONVERT_OK.
One full year is ~61 GB (1460*162*180*360*f4) -- convert subsets for a smoke.
Run inside the PBS job (compute node) via polaris/polaris_sfno_smoke.pbs.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import os
import zarr

# Default source archive. $E3SM_ROOT (exported by polaris_env.sh, overridable with
# POLARIS_E3SM_ROOT) wins, so the advertised knob actually works; the literal is the
# fallback for a bare run outside a PBS job.
E3SM_ROOT = Path(
    os.environ.get("E3SM_ROOT")
    or "/eagle/lighthouse-uchicago/members/jesswan/AI4SRM/data/"
       "E3SMv3_SSP245AMIP_CTL_SST0051_REST0101"
)
UNPREDICTED = ["PCT_GLACIER", "PCT_NATVEG", "PFTDATA_MASK", "TOPO", "sol_in"]
EPOCH = np.datetime64("2015-01-01T00:00:00")
# E3SM is a noleap calendar: 365 d x 24 h == 1460 samples x 6 h, EVERY year (measured:
# all 35 years hold exactly 1460 files; 51,100 total).
HOURS_PER_YEAR = 8760
SAMPLES_PER_YEAR = 1460
FIRST_YEAR = 2015

# E3SM masks land-only fields over ocean and ocean-only fields over land with NaN.
# Those NaNs MUST be filled before writing: SFNO would train on NaN otherwise (the
# store feeds the net directly; unified_recipe normalizes online and does not mask).
# Ocean-only (NaN over land): SST, ICE. Land-only (NaN over ocean): TOPO,
# PFTDATA_MASK, PCT_GLACIER, PCT_NATVEG, SOILWATER_10CM, TSOI_10CM.
#
# ⚠ THIS DICT IS THE ONLY SCIENCE DECISION IN THIS FILE. No automated check can tell you a
# fill VALUE is wrong — a wrong constant produces a perfectly self-consistent store. It is
# reviewed by humans or not at all, so it is recorded into the store's attrs. Measured
# facts to review it against (from the archive, not from a variable name):
#   SST        degC, ocean range [-1.80, 32.21]  -> -1.8 is the physical min, in-distribution
#   TSOI_10CM  KELVIN, land mean 268 K           -> 0.0 is 0 K over ocean: OUT of distribution
#   ICE        fraction [0, 1]                   -> 0.0 fine
# The TSOI_10CM fill is inherited and left alone deliberately: changing it changes what the
# model learns (DESIGN §1 — the science is frozen), and the E3SM npz stats were themselves
# computed under a 0-fill convention for that field. Flagged, not silently "fixed".
NAN_FILL = {
    "SST": -1.8,            # degC — freezing seawater (matches the makani packer)
    "ICE": 0.0,             # sea-ice fraction
    "SOILWATER_10CM": 0.0,
    "TSOI_10CM": 0.0,       # ⚠ Kelvin field; 0.0 == 0 K over ocean. See the box above.
    "TOPO": 0.0,            # m
    "PFTDATA_MASK": 0.0,
    "PCT_GLACIER": 0.0,
    "PCT_NATVEG": 0.0,
}


def sample_files(plev_dir, years, start, count=None, random_sample=None, seed=0):
    """The source files for a store.

    Contiguous by default. `random_sample=N` instead draws N files at random from the whole
    --years span (seeded, so it reproduces) — a small store that spans EVERY year rather
    than the first few days of the first one.

    Why that matters: the smoke store is 16 days of January 2015, and all three converter
    bugs found on 2026-07-15 were invisible at exactly that scale — the frozen in-file year
    is *correct* for 2015, and the archive's later years were simply never read. A random
    draw over 2015-2049 is the cheapest thing that can see them.

    ⚠ A random store is NOT TRAINABLE: SFNO learns t -> t+6h, and these samples are not
    consecutive. It is a correctness fixture only. The store records sampling_mode so the
    trainer's preflight can refuse it.
    """
    files = []
    for y in years:
        files.extend(sorted(plev_dir.glob(f"{y}_*.h5")))
    if random_sample is not None:
        rng = np.random.default_rng(seed)
        n = min(random_sample, len(files))
        pick = rng.choice(len(files), size=n, replace=False)
        # Sorted so the time axis stays monotonic (it just isn't uniformly spaced).
        return [files[i] for i in sorted(pick)]
    # count=None means ALL of --years. It used to default to 64 at the CLI, so the full
    # conversion script — which passes no --max-samples precisely because its comment
    # asserts that means "everything" — would have written 64 of 51,100 samples and
    # printed CONVERT_OK.
    return files[start:] if count is None else files[start:start + count]


def year_of(path):
    """The year from the FILENAME. See read_time_hours for why the file can't be trusted."""
    return int(path.name.split("_")[0])


def read_time_hours(f, year=None, index_in_year=None):
    """Hours since 2015-01-01, with the year taken from the FILENAME.

    ⚠ The archive stamps year 2015 into EVERY file. Measured:
        2016_0000.h5 -> '2015-01-01 00:00:00'      2030_0000.h5 -> '2015-01-01 00:00:00'
        2049_1459.h5 -> '2015-12-31 18:00:00'
    Month/day/hour DO track the sample index correctly; only the year is frozen. Trusting
    the in-file label therefore makes a multi-year store's time axis reset to 0 at every
    year boundary (which the monotonicity gate only catches AFTER the ~10 h write), and
    makes per-year stores carry identical duplicate axes while printing CONVERT_OK. The
    makani packer sidesteps this by synthesizing timestamps from file order entirely.

    Passing `year` reconstructs the true axis. `index_in_year` cross-checks that the
    in-file month/day still agrees with the file's position — so if the archive is ever
    re-stamped correctly, this fails loudly instead of double-counting the year.
    """
    raw = f["input/time"][()]
    if isinstance(raw, bytes):
        raw = raw.decode()
    t = np.datetime64(str(raw).replace(" ", "T"))
    in_year = int((t - EPOCH) / np.timedelta64(1, "h"))
    if index_in_year is not None and in_year != index_in_year * 6:
        raise RuntimeError(
            f"time label {str(raw)!r} is {in_year} h into its year but the filename says "
            f"sample {index_in_year} ({index_in_year * 6} h). The archive's in-file date no "
            f"longer matches file order — re-check the frozen-year assumption before trusting "
            f"any time axis built from it.")
    if year is None:
        return in_year
    return (year - FIRST_YEAR) * HOURS_PER_YEAR + in_year


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=E3SM_ROOT / "h5/plev_data")
    ap.add_argument("--out", type=Path, required=True, help="output .zarr dir")
    ap.add_argument("--years", nargs="+", type=int, default=[2015])
    ap.add_argument("--start-sample", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="default: ALL files in --years. The smoke passes 64 explicitly.")
    ap.add_argument("--random-sample", type=int, default=None, metavar="N",
                    help="draw N files at random from the whole --years span instead of a "
                         "contiguous slice. A small store that spans EVERY year — the cheap "
                         "way to validate the prep. NOT trainable (samples aren't consecutive).")
    ap.add_argument("--seed", type=int, default=0, help="seed for --random-sample")
    args = ap.parse_args()

    if args.random_sample is not None and args.max_samples is not None:
        print("ERROR --random-sample and --max-samples are mutually exclusive")
        return 1
    files = sample_files(args.src, args.years, args.start_sample, args.max_samples,
                         random_sample=args.random_sample, seed=args.seed)
    if not files:
        print(f"ERROR no h5 files under {args.src} for years {args.years}")
        return 1
    T = len(files)

    with h5py.File(files[0], "r") as f:
        names = sorted(k for k in f["input"].keys() if k != "time")
        h, w = f["input"][names[0]].shape
    missing = [v for v in UNPREDICTED if v not in names]
    if missing:
        print(f"ERROR expected unpredicted channels missing from h5: {missing}")
        return 1
    pred_names = [n for n in names if n not in UNPREDICTED]
    unpred_names = list(UNPREDICTED)
    cp, cu = len(pred_names), len(unpred_names)
    print(f"{T} samples, grid {h}x{w}, predicted={cp} unpredicted={cu}", flush=True)

    # NO normalization statistics are written. They used to be (means_/stds_predicted and
    # the unpredicted pair, copied from normalize_*.npz) and they were BOTH dead and wrong:
    #   * dead   — nothing reads them. seq_zarr_datapipe is asked for
    #              variables=["time","predicted","unpredicted"] (train.py:161/262);
    #              train.py:121-126 normalizes online with nn.BatchNorm2d(momentum=None,
    #              affine=False); model_packages.py:85-86 saves THAT batchnorm's running
    #              stats for inference. Upstream's own curate_era5.py writes none either,
    #              so they were never part of the SeqZarr contract — this converter invented
    #              them.
    #   * wrong  — the npz was computed under a land-fill convention of ~270 for SST, while
    #              this store fills -1.8. Measured on the 64-sample smoke: shipped
    #              (mean 109.963, std 123.908) vs the store's actual (8.54, 11.99). A
    #              consumer trusting them would squash SST to 0.097 sigma.
    # Dead-and-wrong metadata is worse than none: it fooled two independent auditors within
    # minutes. If a future consumer genuinely needs stats, compute them from the PACKED data
    # in-stream the way convert_e3sm_to_makani.py does — never copy the npz across a
    # different fill convention.
    root = zarr.open_group(str(args.out), mode="w")
    z_pred = root.create_dataset("predicted", shape=(T, cp, h, w), chunks=(1, cp, h, w), dtype="f4")
    z_unpred = root.create_dataset("unpredicted", shape=(T, cu, h, w), chunks=(1, cu, h, w), dtype="f4")
    z_time = root.create_dataset("time", shape=(T,), chunks=(T,), dtype="i8")
    root.create_dataset("latitude", shape=(h,), dtype="f4")[:] = np.linspace(-89.5, 89.5, h, dtype=np.float32)
    # Cell CENTRES 0.5..359.5, confirmed against boundary_data/TOPO.nc (lat -89.5..89.5
    # ascending, lon 0.5..359.5). This was arange(w) = 0..359, i.e. half a degree west.
    # train.py:447-451 reads this array straight into the inference model package, so the
    # error georegisters every downstream product — silently, forever.
    # NOTE latitude is ALREADY correct and ascending, matching the unflipped data (row 0 is
    # the Antarctic). Do not "align it with makani", which flips BOTH data and axis to suit
    # its own trainer's convention.
    root.create_dataset("longitude", shape=(w,), dtype="f4")[:] = np.arange(w, dtype=np.float32) + 0.5
    root.attrs["channels_predicted"] = pred_names
    root.attrs["channels_unpredicted"] = unpred_names
    root.attrs["source"] = str(args.src)
    root.attrs["time_units"] = "hours since 2015-01-01T00:00:00"
    # Provenance: what this store was ASKED for, so a verifier can check the slice that was
    # intended rather than re-deriving it from the store and agreeing with itself. Without
    # this, a store built from the wrong --start-sample verifies as perfect.
    root.attrs["years"] = [int(y) for y in args.years]
    root.attrs["start_sample"] = int(args.start_sample)
    root.attrs["max_samples"] = args.max_samples
    # A random store spans all years but its samples are NOT consecutive, so SFNO's
    # t -> t+6h target is meaningless on it. Recorded so the trainer preflight can refuse
    # it: a 2,000-sample random fixture would otherwise sail past a sample-count gate.
    root.attrs["sampling_mode"] = "random" if args.random_sample is not None else "contiguous"
    root.attrs["sampling_seed"] = args.seed if args.random_sample is not None else None
    # The FULL input list, in store order. This is the store's ground truth: it lets a
    # verifier compare against the files that actually went in, instead of re-deriving the
    # selection with the same code that built it and agreeing with itself. ~715 KB of JSON
    # at the full 51,100 samples — 0.00007% of a 1 TB store, and the only thing that makes
    # a wrong --start-sample or a wrong random draw detectable at all.
    root.attrs["source_files"] = [p.name for p in files]
    root.attrs["source_first_file"] = files[0].name
    root.attrs["source_last_file"] = files[-1].name
    root.attrs["nan_fill"] = {k: float(v) for k, v in NAN_FILL.items()}

    buf_p = np.empty((cp, h, w), dtype=np.float32)
    buf_u = np.empty((cu, h, w), dtype=np.float32)
    def _fill(buf, names_, where):
        """Replace E3SM land/ocean mask NaNs in-place; hard-fail on any unexpected NaN."""
        for c, n in enumerate(names_):
            if not np.isnan(buf[c]).any():
                continue
            if n not in NAN_FILL:
                raise RuntimeError(
                    f"{where} channel '{n}' has NaN but no NAN_FILL entry — refusing to "
                    f"write NaN into the training store")
            np.nan_to_num(buf[c], copy=False, nan=NAN_FILL[n])

    for i, path in enumerate(files):
        with h5py.File(path, "r") as f:
            g = f["input"]
            for c, n in enumerate(pred_names):
                g[n].read_direct(buf_p[c])
            for c, n in enumerate(unpred_names):
                g[n].read_direct(buf_u[c])
            # The year comes from the filename; the in-file stamp is frozen at 2015 for
            # every year of the archive. index_in_year cross-checks that the in-file
            # month/day still tracks file order.
            z_time[i] = read_time_hours(f, year=year_of(path),
                                        index_in_year=int(path.stem.split("_")[1]))
        _fill(buf_p, pred_names, "predicted")
        _fill(buf_u, unpred_names, "unpredicted")
        z_pred[i] = buf_p
        z_unpred[i] = buf_u
        if i % 50 == 0:
            print(f"  wrote sample {i}/{T} ({path.name})", flush=True)

    # --- validation ---------------------------------------------------------------------
    # This is a CHEAP end-of-run gate, not a verification: it reads 1 channel of 1 sample.
    # It is deliberately kept minimal because polaris/verify_seqzarr.py does the real job
    # (every sample x every channel, bitwise, plus fill placement). Do NOT read CONVERT_OK
    # as "the store is correct" — it means "the run finished and the shape is right".
    zr = zarr.open(str(args.out), mode="r")
    probe_c = next(c for c, n in enumerate(pred_names) if n not in NAN_FILL)
    with h5py.File(files[0], "r") as f:
        ref = f["input"][pred_names[probe_c]][()]
    err = float(np.max(np.abs(zr["predicted"][0, probe_c] - ref)))
    finite = bool(np.isfinite(zr["predicted"][0]).all() and np.isfinite(zr["unpredicted"][0]).all())
    # Every chunk must have been written. zarr pre-allocates the full (T,...) shape with
    # fill_value 0.0, so an unwritten sample reads back as an all-zero slab with NO error —
    # a preempted run leaves a full-shape store that trains silently on zeros. This is the
    # one check that distinguishes "complete" from "killed halfway".
    n_init, n_tot = zr["predicted"].nchunks_initialized, zr["predicted"].nchunks
    complete = n_init == n_tot
    # Monotonic, not uniform: a random fixture legitimately has gaps. Uniform 6h spacing is
    # checked by verify_seqzarr.py for contiguous stores only.
    monotonic = bool(np.all(np.diff(zr["time"][:]) > 0)) if T > 1 else True
    ok = (err == 0.0 and finite and complete and monotonic
          and zr["predicted"].shape == (T, cp, h, w)
          and zr["unpredicted"].shape == (T, cu, h, w) and zr["time"].shape == (T,))
    print(f"validation: max|zarr-h5| = {err:.3e} on predicted[0,{probe_c}] "
          f"({pred_names[probe_c]}); sample0 all-finite = {finite}; "
          f"chunks {n_init}/{n_tot}; time monotonic = {monotonic}", flush=True)
    if not finite:
        print("ERROR NaN/inf survived into the store (check NAN_FILL coverage)")
    if not complete:
        print(f"ERROR INCOMPLETE_STORE: {n_tot - n_init} of {n_tot} sample chunks were never "
              f"written — they would read back as all-zero slabs. Re-run the conversion.")
    if not monotonic:
        print("ERROR TIME_NOT_MONOTONIC: check the year-from-filename reconstruction")
    if ok:
        # The sentinel is written LAST and only on success. A consumer must require it:
        # every other property of a half-written store looks correct, including its shape.
        root.attrs["conversion_complete"] = True
        print("CONVERT_OK", flush=True)
        return 0
    print("ERROR converted store failed validation")
    return 1


if __name__ == "__main__":
    sys.exit(main())
