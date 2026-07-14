#!/usr/bin/env python3
"""convert_e3sm_to_makani.py — E3SM per-sample h5 -> Makani "multifiles" dataset.

Packs the Polaris E3SMv3 SSP245-AMIP per-sample HDF5 archive into the exact
three-dataset contract consumed by src/sfno_training (PlasimForcingDataset /
PlasimTrainer), mirroring src/plasim_makani_packager/packager.py + stats.py +
metadata.py in one pass.

INPUT (verified 2026-07-14 on Polaris)
--------------------------------------
{e3sm-root}/h5/plev_data/{year}_{idx:04d}.h5   idx 0000..1459, 6-hourly,
    365-day year (years 2015..2049, 1460 samples/year each).
    Single group "input" with 162 float32 (180, 360) datasets named VAR or
    VAR_{plev} (plev = float hPa string), plus a scalar string "time".
    Grid: 1-degree equiangular; row 0 = lat -89.5 (ASCENDING, south first),
    col 0 = lon 0.5 (verified against boundary_data/TOPO.nc).
    NaNs: land-only fields (PFTDATA_MASK, TOPO, PCT_GLACIER, ...) are NaN
    over ocean; ocean-only fields (SST [deg C], ICE) are NaN over land.
    All atmosphere/state fields used below are NaN-free (verified).

OUTPUT (PlasimForcingDataset contract)
--------------------------------------
{output-root}/{split}/{year}.h5   for split in train/valid/test:
    /fields_state      (T, 52, 180, 360) float32, chunks (1, 52, 180, 360)
    /fields_diagnostic (T, 1,  180, 360) float32, chunks (1, 1, 180, 360)
    /forcing           (T, 6,  180, 360) float32, chunks (1, 6, 180, 360)
    /timestamp         (T,) int64   split-globally monotonic, step 21600 s
    /time_plasim       (T,) float64 (days since split start; schema parity)
    /channel_state (52,) /channel_diagnostic (1,) /channel_forcing (6,) ascii
    /lat (180,) float64 DESCENDING (89.5 .. -89.5)  /lon (360,) float64
    h5py dimension scales named exactly "timestamp", "channel_state",
    "channel_diagnostic", "channel_forcing", "lat", "lon" attached to dims
    0..3 of each 4D dataset (PlasimForcingDataset reads dims[i][name]).
    Data rows are FLIPPED to descending lat to match the repo convention
    (docs/2026-06-02_eval_inference_latitude_flip.md: fields_state row 0 =
    northernmost).
{output-root}/stats/
    global_means.npy          (1, 53, 1, 1) float32   state+diag, train split
    global_stds.npy           (1, 53, 1, 1) float32
    time_means.npy            (1, 53, 180, 360) float32
    forcing_global_means.npy  (1, 6, 1, 1) float32
    forcing_global_stds.npy   (1, 6, 1, 1) float32
    forcing_time_means.npy    (1, 6, 180, 360) float32
{output-root}/metadata/data.json   (same schema as plasim_makani_packager.metadata)

CHANNEL MAP (locked 52 state + 1 diag + 6 forcing = PlaSim contract shape)
--------------------------------------------------------------------------
state  : PS, TREFHT, then T/U/V/RELHUM/Z3 at the 10 lowest of the 18 pressure
         levels (TOA->surface: ~200,250,300,400,500,600,700,850,925,1000 hPa —
         these E3SM plevs match the PlaSim v10 ZG_PLEV_HPA list almost 1:1).
diag   : PRECT  (precip rate; loss-only channel, PlaSim pr_6h slot)
forcing: PFTDATA_MASK->lsm, TOPO->sg, PCT_GLACIER->z0 slot, SST->sst,
         sol_in->rsdt, ICE->sic.  NaN fills: SST -> -1.8 degC, all others -> 0.

Stats are computed from the PACKED train split (float64 sum/sumsq + per-pixel
time sums) — NOT from the dataset-level normalize_*.npz, whose SST entry is
inconsistent with a NaN->-1.8 fill.

USAGE (login node OK for a 1+1-year smoke pack; ~5-10 min/year, ~22 GB/year
full, or use --max-samples-per-year 400 for ~6 GB/year)
--------------------------------------------------------------------------
  python convert_e3sm_to_makani.py \
      --e3sm-root /eagle/lighthouse-uchicago/members/jesswan/AI4SRM/data/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101 \
      --output-root <OUT> \
      --train-years 2015 2015 --valid-years 2016 2016 --test-years 2017 2017 \
      [--max-samples-per-year 400] [--overwrite] [--validate]

Prints CONVERT_OK on success (with --validate, after a single-sample read-back).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np

STEP_SECONDS = 21600  # 6 hours
H, W = 180, 360

# Exact dataset-name plev suffixes (verified against 2015_0000.h5) and the
# nominal integer-hPa channel-name suffixes used in metadata/config.
PLEV_EXACT = [
    "200.99889546355382", "256.72368590525895", "302.21364012188303",
    "385.999023919911", "492.46857402252755", "608.6437744215842",
    "713.7046383204334", "849.6612491105952", "925.5197481473349",
    "998.4964394917621",
]
PLEV_NOMINAL = [200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# (channel_name, h5 dataset key under /input, nan_fill or None)
STATE_SPECS = (
    [("PS", "PS", None), ("TREFHT", "TREFHT", None)]
    + [(f"T{n}", f"T_{p}", None) for n, p in zip(PLEV_NOMINAL, PLEV_EXACT)]
    + [(f"U{n}", f"U_{p}", None) for n, p in zip(PLEV_NOMINAL, PLEV_EXACT)]
    + [(f"V{n}", f"V_{p}", None) for n, p in zip(PLEV_NOMINAL, PLEV_EXACT)]
    + [(f"RH{n}", f"RELHUM_{p}", None) for n, p in zip(PLEV_NOMINAL, PLEV_EXACT)]
    + [(f"Z{n}", f"Z3_{p}", None) for n, p in zip(PLEV_NOMINAL, PLEV_EXACT)]
)
DIAG_SPECS = [("PRECT", "PRECT", None)]
FORCING_SPECS = [
    ("lsm", "PFTDATA_MASK", 0.0),   # 1 on land, NaN over ocean -> 0
    ("topo", "TOPO", 0.0),          # m, NaN over ocean -> 0
    ("glacier", "PCT_GLACIER", 0.0),  # %, NaN over ocean -> 0
    ("sst", "SST", -1.8),           # deg C, NaN over land -> -1.8 (freezing seawater)
    ("solin", "sol_in", None),      # W/m2, NaN-free (rsdt analogue)
    ("ice", "ICE", 0.0),            # fraction, NaN over land -> 0
]
assert len(STATE_SPECS) == 52 and len(DIAG_SPECS) == 1 and len(FORCING_SPECS) == 6

STATE_CHANNELS = [n for n, _, _ in STATE_SPECS]
DIAG_CHANNELS = [n for n, _, _ in DIAG_SPECS]
FORCING_CHANNELS = [n for n, _, _ in FORCING_SPECS]
TARGET_CHANNELS = STATE_CHANNELS + DIAG_CHANNELS  # 53

LAT = np.arange(89.5, -90.0, -1.0)   # DESCENDING, matches flipped data rows
LON = np.arange(0.5, 360.0, 1.0)
assert LAT.shape == (H,) and LON.shape == (W,)


def _read_sample(path: str, specs) -> np.ndarray:
    """Read one per-sample file -> (C, H, W) float32, lat-flipped, NaN-filled."""
    out = np.empty((len(specs), H, W), dtype=np.float32)
    with h5py.File(path, "r") as f:
        g = f["input"]
        for i, (_name, key, fill) in enumerate(specs):
            arr = g[key][...]
            if arr.shape != (H, W):
                raise RuntimeError(f"{path}:{key} shape {arr.shape} != {(H, W)}")
            arr = arr[::-1, :]  # ascending (south-first) -> descending (north-first)
            if fill is not None:
                arr = np.where(np.isnan(arr), np.float32(fill), arr)
            elif np.isnan(arr).any():
                raise RuntimeError(f"{path}:{key} has unexpected NaNs (no fill configured)")
            out[i] = arr
    return out


def _year_files(e3sm_root: str, year: int, max_samples: int | None) -> list[str]:
    files = sorted(glob.glob(os.path.join(e3sm_root, "h5", "plev_data", f"{year}_*.h5")))
    if not files:
        raise RuntimeError(f"no input files for year {year} under {e3sm_root}/h5/plev_data")
    return files[:max_samples] if max_samples else files


def _write_year(out_path: str, e3sm_root: str, year: int, offset_seconds: int,
                max_samples: int | None, accum: dict | None) -> int:
    """Stream one year into out_path. Returns T. Updates stats accumulators."""
    files = _year_files(e3sm_root, year, max_samples)
    T = len(files)
    tmp = out_path + ".tmp"
    with h5py.File(tmp, "w") as f:
        ds_defs = [
            ("fields_state", 52, "channel_state", STATE_CHANNELS),
            ("fields_diagnostic", 1, "channel_diagnostic", DIAG_CHANNELS),
            ("forcing", 6, "channel_forcing", FORCING_CHANNELS),
        ]
        scales = {}
        for scale_name, data in (
            ("timestamp", np.int64(offset_seconds) + np.arange(T, dtype=np.int64) * STEP_SECONDS),
            ("lat", LAT.astype(np.float64)),
            ("lon", LON.astype(np.float64)),
            ("channel_state", np.array(STATE_CHANNELS, dtype=h5py.string_dtype("ascii"))),
            ("channel_diagnostic", np.array(DIAG_CHANNELS, dtype=h5py.string_dtype("ascii"))),
            ("channel_forcing", np.array(FORCING_CHANNELS, dtype=h5py.string_dtype("ascii"))),
        ):
            d = f.create_dataset(scale_name, data=data)
            d.make_scale(scale_name)
            scales[scale_name] = d
        f.create_dataset("time_plasim",
                         data=(np.arange(T, dtype=np.float64) * (STEP_SECONDS / 86400.0)
                               + offset_seconds / 86400.0))

        payloads = {}
        for dset_name, C, ch_scale, _names in ds_defs:
            p = f.create_dataset(dset_name, shape=(T, C, H, W), dtype="float32",
                                 chunks=(1, C, H, W))
            p.dims[0].attach_scale(scales["timestamp"])
            p.dims[1].attach_scale(scales[ch_scale])
            p.dims[2].attach_scale(scales["lat"])
            p.dims[3].attach_scale(scales["lon"])
            payloads[dset_name] = p

        f.attrs["source_root"] = e3sm_root
        f.attrs["year"] = year
        f.attrs["converter"] = "makani_sfno/polaris/convert_e3sm_to_makani.py"
        f.attrs["lat_order"] = "descending (row 0 = +89.5)"
        f.attrs["sst_units"] = "degC, land filled with -1.8"

        for t, path in enumerate(files):
            state = _read_sample(path, STATE_SPECS)
            diag = _read_sample(path, DIAG_SPECS)
            forc = _read_sample(path, FORCING_SPECS)
            payloads["fields_state"][t] = state
            payloads["fields_diagnostic"][t] = diag
            payloads["forcing"][t] = forc
            if accum is not None:
                tgt = np.concatenate([state, diag], axis=0).astype(np.float64)
                fo = forc.astype(np.float64)
                accum["n"] += H * W
                accum["sum_t"] += tgt.sum(axis=(1, 2))
                accum["sumsq_t"] += (tgt * tgt).sum(axis=(1, 2))
                accum["tsum_t"] += tgt
                accum["sum_f"] += fo.sum(axis=(1, 2))
                accum["sumsq_f"] += (fo * fo).sum(axis=(1, 2))
                accum["tsum_f"] += fo
                accum["t_count"] += 1
            if t % 200 == 0:
                print(f"  [{year}] {t}/{T}", flush=True)
    os.replace(tmp, out_path)
    print(f"wrote {out_path} (T={T})", flush=True)
    return T


def _write_stats(stats_dir: str, accum: dict) -> None:
    os.makedirs(stats_dir, exist_ok=True)
    n, tc = accum["n"], accum["t_count"]
    for tag, C, s, ss, ts in (
        ("", 53, accum["sum_t"], accum["sumsq_t"], accum["tsum_t"]),
        ("forcing_", 6, accum["sum_f"], accum["sumsq_f"], accum["tsum_f"]),
    ):
        mean = s / n
        var = np.maximum(ss / n - mean * mean, 0.0)
        std = np.maximum(np.sqrt(var), 1e-12)  # floor: avoid /0 for constant fields
        np.save(os.path.join(stats_dir, f"{tag}global_means.npy"),
                mean.astype(np.float32).reshape(1, C, 1, 1))
        np.save(os.path.join(stats_dir, f"{tag}global_stds.npy"),
                std.astype(np.float32).reshape(1, C, 1, 1))
        np.save(os.path.join(stats_dir, f"{tag}time_means.npy"),
                (ts / tc).astype(np.float32).reshape(1, C, H, W))
    print(f"wrote stats to {stats_dir} (n={n}, t_count={tc})", flush=True)


def _write_metadata(output_root: str, splits: dict) -> None:
    meta = {
        "dataset_name": "e3smv3-ssp245amip-plev10-180x360",
        "h5_path": "fields_state",
        "diagnostic_h5_path": "fields_diagnostic",
        "forcing_h5_path": "forcing",
        "dims": ["time", "channel", "lat", "lon"],
        "dhours": 6,
        "coords": {
            "grid_type": "equiangular",
            "lat": LAT.tolist(),
            "lon": LON.tolist(),
            "channel": TARGET_CHANNELS,
            "channel_state": STATE_CHANNELS,
            "channel_diagnostic": DIAG_CHANNELS,
            "channel_forcing": FORCING_CHANNELS,
        },
        "attrs": {
            "description": "E3SMv3 SSP245-AMIP plev subset packed into the "
                           "PlaSim/Makani three-dataset layout (patched trainer only)",
            "source_root": splits["source_root"],
            "train_years": splits["train"],
            "valid_years": splits["valid"],
            "test_years": splits["test"],
            "requires_patched_makani": True,
            "sst_land_fill_degC": -1.8,
            "plev_exact_hpa": PLEV_EXACT,
        },
    }
    meta_dir = os.path.join(output_root, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "data.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {meta_dir}/data.json", flush=True)


def _validate(output_root: str) -> None:
    """Single-sample read-back through the same access pattern the loader uses."""
    train_files = sorted(glob.glob(os.path.join(output_root, "train", "*.h5")))
    assert train_files, "no train files written"
    with h5py.File(train_files[0], "r") as f:
        st = f["fields_state"]
        assert st.shape[1:] == (52, H, W) and st.dtype == np.float32, st.shape
        assert f["fields_diagnostic"].shape[1:] == (1, H, W)
        assert f["forcing"].shape[1:] == (6, H, W)
        # dimension-scale access exactly as PlasimForcingDataset._get_stats_h5
        lat = st.dims[2]["lat"][...]
        lon = st.dims[3]["lon"][...]
        ts = st.dims[0]["timestamp"][...]
        assert lat[0] > lat[-1], "lat must be descending"
        assert lon.shape == (W,)
        d = np.diff(ts)
        assert d.size and np.all(d == STEP_SECONDS), "timestamps must step 21600 s"
        sample = st[0]
        assert np.isfinite(sample).all(), "NaN/inf in fields_state sample 0"
        assert np.isfinite(f["forcing"][0]).all(), "NaN/inf in forcing sample 0"
        assert np.isfinite(f["fields_diagnostic"][0]).all()
    for name, shape in (
        ("global_means.npy", (1, 53, 1, 1)), ("global_stds.npy", (1, 53, 1, 1)),
        ("time_means.npy", (1, 53, H, W)),
        ("forcing_global_means.npy", (1, 6, 1, 1)),
        ("forcing_global_stds.npy", (1, 6, 1, 1)),
        ("forcing_time_means.npy", (1, 6, H, W)),
    ):
        arr = np.load(os.path.join(output_root, "stats", name))
        assert arr.shape == shape and np.isfinite(arr).all(), (name, arr.shape)
    stds = np.load(os.path.join(output_root, "stats", "global_stds.npy")).ravel()
    print(f"validate: min target std = {stds.min():.3e} "
          f"(PRECT expected ~8e-8; PlaSim precedent trained fine)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--e3sm-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--train-years", type=int, nargs=2, default=[2015, 2015])
    p.add_argument("--valid-years", type=int, nargs=2, default=[2016, 2016])
    p.add_argument("--test-years", type=int, nargs=2, default=[2017, 2017])
    p.add_argument("--max-samples-per-year", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()

    split_years = {
        "train": list(range(args.train_years[0], args.train_years[1] + 1)),
        "valid": list(range(args.valid_years[0], args.valid_years[1] + 1)),
        "test": list(range(args.test_years[0], args.test_years[1] + 1)),
    }
    accum = {
        "n": 0, "t_count": 0,
        "sum_t": np.zeros(53), "sumsq_t": np.zeros(53), "tsum_t": np.zeros((53, H, W)),
        "sum_f": np.zeros(6), "sumsq_f": np.zeros(6), "tsum_f": np.zeros((6, H, W)),
    }
    skipped_train = []
    for split, years in split_years.items():
        os.makedirs(os.path.join(args.output_root, split), exist_ok=True)
        offset = 0
        for year in years:
            out_path = os.path.join(args.output_root, split, f"{year}.h5")
            if os.path.exists(out_path) and not args.overwrite:
                print(f"skip existing {out_path} (--overwrite to force)")
                if split == "train":
                    skipped_train.append(year)
                with h5py.File(out_path, "r") as f:
                    offset += f["fields_state"].shape[0] * STEP_SECONDS
                continue
            T = _write_year(out_path, args.e3sm_root, year, offset,
                            args.max_samples_per_year,
                            accum if split == "train" else None)
            offset += T * STEP_SECONDS

    if accum["t_count"] == 0:
        sys.exit("ERROR no train samples processed (all train files existed? "
                 "rerun with --overwrite to regenerate stats)")
    # A skipped train year never feeds the accumulator, so stats would be computed from
    # only the SUBSET of years that were (re)written — silently wrong normalization.
    if skipped_train:
        sys.exit(f"ERROR partial train stats: year(s) {skipped_train} were skipped as "
                 f"already-existing, so they are missing from the stats accumulator while "
                 f"other train years were written. Rerun with --overwrite to regenerate "
                 f"the whole train split + stats consistently.")
    _write_stats(os.path.join(args.output_root, "stats"), accum)
    _write_metadata(args.output_root, {**split_years, "source_root": args.e3sm_root})
    if args.validate:
        _validate(args.output_root)
    print("CONVERT_OK")


if __name__ == "__main__":
    main()
