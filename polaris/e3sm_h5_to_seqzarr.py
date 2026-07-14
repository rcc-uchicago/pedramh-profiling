#!/usr/bin/env python
"""E3SM per-sample HDF5 -> SeqZarr store for PhysicsNeMo unified_recipe SFNO training.

Target: <root>/predicted (T,157,180,360) + unpredicted (T,5,180,360) + time (T,) i8 +
lat/lon + means/stds, read by examples/weather/unified_recipe/seq_zarr_datapipe.py
(SeqZarrSource does zarr.open(store)["<array>"][time_idx]; axis 0 = time). Static /
prescribed-forcing fields go to "unpredicted"; the other 157 channels are "predicted".
time is int hours-since-epoch (DALI can't ingest datetime64/bytes). std==0 clamped to 1.

Micro-tested on Polaris (base conda, zarr 2.18.7): 6-sample store -> max|zarr-h5|=0 + CONVERT_OK.
One full year is ~61 GB (1460*162*180*360*f4) -- convert subsets for a smoke.
Run inside the PBS job (compute node) via polaris/polaris_sfno_smoke.pbs.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import zarr

E3SM_ROOT = Path(
    "/eagle/lighthouse-uchicago/members/jesswan/AI4SRM/data/"
    "E3SMv3_SSP245AMIP_CTL_SST0051_REST0101"
)
UNPREDICTED = ["PCT_GLACIER", "PCT_NATVEG", "PFTDATA_MASK", "TOPO", "sol_in"]
EPOCH = np.datetime64("2015-01-01T00:00:00")

# E3SM masks land-only fields over ocean and ocean-only fields over land with NaN.
# Those NaNs MUST be filled before writing: SFNO would train on NaN otherwise (the
# store feeds the net directly; unified_recipe normalizes online and does not mask).
# Ocean-only (NaN over land): SST, ICE. Land-only (NaN over ocean): TOPO,
# PFTDATA_MASK, PCT_GLACIER, PCT_NATVEG, SOILWATER_10CM, TSOI_10CM.
NAN_FILL = {
    "SST": -1.8,            # degC — freezing seawater (matches the makani packer)
    "ICE": 0.0,             # sea-ice fraction
    "SOILWATER_10CM": 0.0,
    "TSOI_10CM": 0.0,
    "TOPO": 0.0,            # m
    "PFTDATA_MASK": 0.0,
    "PCT_GLACIER": 0.0,
    "PCT_NATVEG": 0.0,
}


def sample_files(plev_dir, years, start, count):
    files = []
    for y in years:
        files.extend(sorted(plev_dir.glob(f"{y}_*.h5")))
    return files[start:start + count]


def read_time_hours(f):
    raw = f["input/time"][()]
    if isinstance(raw, bytes):
        raw = raw.decode()
    t = np.datetime64(str(raw).replace(" ", "T"))
    return int((t - EPOCH) / np.timedelta64(1, "h"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=E3SM_ROOT / "h5/plev_data")
    ap.add_argument("--stats-root", type=Path, default=E3SM_ROOT)
    ap.add_argument("--out", type=Path, required=True, help="output .zarr dir")
    ap.add_argument("--years", nargs="+", type=int, default=[2015])
    ap.add_argument("--start-sample", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=64)
    args = ap.parse_args()

    files = sample_files(args.src, args.years, args.start_sample, args.max_samples)
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

    mean_npz = np.load(args.stats_root / "normalize_mean.npz")
    std_npz = np.load(args.stats_root / "normalize_std.npz")

    def stats(names_):
        mu = np.array([float(mean_npz[n][0]) for n in names_], dtype=np.float32)
        sd = np.array([float(std_npz[n][0]) for n in names_], dtype=np.float32)
        sd[sd == 0.0] = 1.0
        return mu.reshape(1, -1, 1, 1), sd.reshape(1, -1, 1, 1)

    mu_p, sd_p = stats(pred_names)
    mu_u, sd_u = stats(unpred_names)

    root = zarr.open_group(str(args.out), mode="w")
    z_pred = root.create_dataset("predicted", shape=(T, cp, h, w), chunks=(1, cp, h, w), dtype="f4")
    z_unpred = root.create_dataset("unpredicted", shape=(T, cu, h, w), chunks=(1, cu, h, w), dtype="f4")
    z_time = root.create_dataset("time", shape=(T,), chunks=(T,), dtype="i8")
    root.create_dataset("latitude", shape=(h,), dtype="f4")[:] = np.linspace(-89.5, 89.5, h, dtype=np.float32)
    root.create_dataset("longitude", shape=(w,), dtype="f4")[:] = np.arange(w, dtype=np.float32)
    root.create_dataset("means_predicted", shape=mu_p.shape, dtype="f4")[:] = mu_p
    root.create_dataset("stds_predicted", shape=sd_p.shape, dtype="f4")[:] = sd_p
    root.create_dataset("means_unpredicted", shape=mu_u.shape, dtype="f4")[:] = mu_u
    root.create_dataset("stds_unpredicted", shape=sd_u.shape, dtype="f4")[:] = sd_u
    root.attrs["channels_predicted"] = pred_names
    root.attrs["channels_unpredicted"] = unpred_names
    root.attrs["source"] = str(args.src)
    root.attrs["time_units"] = "hours since 2015-01-01T00:00:00"

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
            z_time[i] = read_time_hours(f)
        _fill(buf_p, pred_names, "predicted")
        _fill(buf_u, unpred_names, "unpredicted")
        z_pred[i] = buf_p
        z_unpred[i] = buf_u
        if i % 50 == 0:
            print(f"  wrote sample {i}/{T} ({path.name})", flush=True)

    # --- validation: exact round-trip on an UNFILLED channel + a hard NaN gate ---
    zr = zarr.open(str(args.out), mode="r")
    probe_c = next(c for c, n in enumerate(pred_names) if n not in NAN_FILL)
    with h5py.File(files[0], "r") as f:
        ref = f["input"][pred_names[probe_c]][()]
    err = float(np.max(np.abs(zr["predicted"][0, probe_c] - ref)))
    finite = bool(np.isfinite(zr["predicted"][0]).all() and np.isfinite(zr["unpredicted"][0]).all())
    ok = (err == 0.0 and finite
          and zr["predicted"].shape == (T, cp, h, w)
          and zr["unpredicted"].shape == (T, cu, h, w) and zr["time"].shape == (T,)
          and np.all(np.diff(zr["time"][:]) > 0))
    print(f"validation: max|zarr-h5| = {err:.3e} on predicted[0,{probe_c}] "
          f"({pred_names[probe_c]}); sample0 all-finite = {finite}", flush=True)
    if not finite:
        print("ERROR NaN/inf survived into the store (check NAN_FILL coverage)")
    if ok:
        print("CONVERT_OK", flush=True)
        return 0
    print("ERROR converted store failed validation")
    return 1


if __name__ == "__main__":
    sys.exit(main())
