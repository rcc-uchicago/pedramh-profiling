#!/usr/bin/env python3
"""convert_e3sm_to_makani_alldata.py — E3SM per-sample h5 -> Makani "multifiles",
108 of the 162 archive channels (100 state + 1 diag + 7 forcing; clouds excluded).

This is the ALL-DATA VARIANT of convert_e3sm_to_makani.py. It sits BESIDE the
locked 52/1/6 PlaSim-comparable path and never replaces it: different output
roots, different channel contract, different config (e3sm_alldata_*.yaml).
The 52/1/6 path exists for comparability with the group's PlaSim baseline —
keep using it for that; use THIS path when the model should see everything
the archive has.

CHANNEL MAP (the 108 kept channels; verified against 2015_0000.h5 2026-07-16)
------------------------------------------------------------------------------
state (100, predicted + fed back at rollout):
    PS, TREFHT, U10, RHREFHT, PSL, TMQ, FSNT, FSNTOA,
    SOILWATER_10CM (NaN over ocean -> 0.0), TSOI_10CM (NaN over ocean -> 270.0),
    then T/U/V/Z3/RELHUM at ALL 18 levels (TOA -> surface).
diag (1, loss-only):  PRECT
forcing (7, prescribed, never predicted):
    lsm<-PFTDATA_MASK, topo<-TOPO, glacier<-PCT_GLACIER, natveg<-PCT_NATVEG,
    sst<-SST (degC; NaN over land -> -1.8), solin<-sol_in, ice<-ICE.
    NaN fills: 0.0 for all land-only fields, -1.8 for SST.
The land prognostics (SOILWATER_10CM, TSOI_10CM) are STATE, not forcing: they
evolve with the simulation, so prescribing them at rollout would leak state.
SST/ICE are genuinely prescribed in an AMIP run; sol_in is astronomical;
the other forcings are static land masks. This split follows the group's own
Pangu categorization (PanguWeather/v2.0/config/E3SM_SFNO_H5_POLARIS.yaml:
land_variables vs {constant,varying}_boundary_variables), fills included.

LEVEL NAMING — read this before adding any "T850"-style alias
-------------------------------------------------------------
Channels are named {VAR}_l{00..17} by LEVEL INDEX (l00 = topmost, l17 = the
level nearest the surface), NOT by pressure. The archive's "plev_data" name
and the float suffixes (e.g. Z3_998.4964394917621) suggest isobaric surfaces,
but the levels are TERRAIN-FOLLOWING (sigma-like), measured on Polaris:
corr(Z3_998.5, TOPO) = 0.979 over land, Z3_998.5 - TOPO averages +15.4 m
(a real 1000 hPa surface would be underground over Tibet), and Z3_492 reaches
9,266 m where a true 500 hPa surface tops out near 6,000 m. The Pangu config
already documents this ("despite the name, plev_data h5 files hold SIGMA-level
fields"). KNOWN DEFECT elsewhere: the locked-path converter
(convert_e3sm_to_makani.py PLEV_NOMINAL) names its channels T850/Z500/...,
asserting isobaric surfaces the data does not have — those names are kept
there only because the 52/1/6 contract is frozen for PlaSim comparability.
Do not propagate them. The exact suffix and the NOMINAL-LABEL-ONLY hPa value
for each index are recorded in metadata/data.json (attrs.level_table).

CLOUD VARIABLES ARE EXCLUDED
----------------------------
CLDICE/CLDLIQ/CLOUD (3 vars x 18 levels = 54 channels) are NOT packed — the
science owner confirmed 2026-07-16 that they are excluded from every model, so
all three pipelines now agree on the same 108 of the archive's 162 channels.
This path is therefore "all YEARS + all LEVELS", not "all channels": what makes
it wider than the locked 52/1/6 path is 18 levels instead of 10 and the surface
fields the PlaSim contract has no slot for. The zero-variance story lives in
polaris_e3sm_variable_reference.md R5 (16 of the 54 dropped channels were
EXACTLY constant across all 35 years). The stats pass still records
attrs.zero_variance_channels — now expect ZERO entries; any hit is a finding.

OUTPUT (same PlasimForcingDataset contract as the locked path, wider):
    {output-root}/{split}/{year}.h5:
        /fields_state (T,100,180,360) /fields_diagnostic (T,1,180,360)
        /forcing (T,7,180,360), float32, chunks (1,C,180,360), plus the
        timestamp/lat/lon/channel_* dimension scales the loader requires.
    {output-root}/stats/*.npy  (target: 1x101, forcing: 1x7)
    {output-root}/metadata/data.json

Stats are computed in a SECOND PASS over the packed train files (unlike the
locked converter, which accumulates while writing and therefore ERRORs out if
any train year already exists). Reason: the full all-data pack is ~1.4 TB and
runs on the preemptable queue — it MUST be resumable, so skip-existing has to
be safe for every split, train included.

SIZE: 108 channels x 1460 x 180x360 float32 = 40.87 GB/year (~41 GB); full
35 years ~1.43 TB. Smoke subsets via --max-samples-per-year.

USAGE
-----
  python convert_e3sm_to_makani_alldata.py \
      --e3sm-root /eagle/lighthouse-uchicago/members/jesswan/AI4SRM/data/E3SMv3_SSP245AMIP_CTL_SST0051_REST0101 \
      --output-root <OUT> \
      --train-years 2015 2015 --valid-years 2016 2016 --test-years 2017 2017 \
      [--max-samples-per-year 96] [--overwrite] [--validate]

Prints CONVERT_ALLDATA_OK on success (with --validate, after a read-back).
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
STATS_CHUNK_T = 16  # samples per read in the stats pass (~1.3 GB float64 peak)
STD_FLOOR = 1e-12   # defensive /0 guard; no constant channels expected (clouds excluded)
ZERO_VAR_STD = 1e-9  # below this a channel is reported as zero-variance

# All 18 level suffixes, exactly as they appear in the h5 dataset names,
# ordered top-of-atmosphere -> surface (verified against 2015_0000.h5 and
# identical across all 8 upper-air variables). These floats are NOT pressures
# on the sphere — the levels are terrain-following; see the module docstring.
LEVEL_EXACT = [
    "4.714998332947841", "10.655023096474308", "19.235455601758737",
    "28.79458853709195", "50.11779996521295", "69.59908688413749",
    "96.46377266572703", "145.04282239200347", "200.99889546355382",
    "256.72368590525895", "302.21364012188303", "385.999023919911",
    "492.46857402252755", "608.6437744215842", "713.7046383204334",
    "849.6612491105952", "925.5197481473349", "998.4964394917621",
]
# Nominal hPa LABELS ONLY (the group's shorthand for the same 18 levels,
# PanguWeather E3SM configs `levels:`). Kept for the metadata level table so
# humans can orient; never used in channel names.
LEVEL_NOMINAL_LABEL = [5, 10, 20, 30, 50, 70, 100, 150,
                       200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
assert len(LEVEL_EXACT) == 18 and len(LEVEL_NOMINAL_LABEL) == 18

# The three cloud variables (CLDICE, CLDLIQ, CLOUD) are EXCLUDED from all models
# (science owner, 2026-07-16) — so this path packs 5 upper-air vars, not 8. That is
# 3 x 18 = 54 archive channels never packed, matching PanguWeather (commented out of
# upper_air_variables) and physicsnemo (EXCLUDED_VARS in e3sm_h5_to_seqzarr.py).
# See polaris_e3sm_variable_reference.md R5 for what is in them: 16 of the 54 are
# EXACTLY constant over all 35 years; the rest carry near-zero variance.
UPPER_AIR_VARS = ["T", "U", "V", "Z3", "RELHUM"]

# (channel_name, h5 dataset key under /input, nan_fill or None)
# Surface state first (mirrors the locked converter's PS,TREFHT-first order),
# then the 5 upper-air variables at all 18 levels. Land-prognostic fills follow
# the Pangu mask_fill precedent (SOILWATER_10CM: 0.0, TSOI_10CM: 270.0 K).
STATE_SPECS = (
    [
        ("PS", "PS", None), ("TREFHT", "TREFHT", None), ("U10", "U10", None),
        ("RHREFHT", "RHREFHT", None), ("PSL", "PSL", None), ("TMQ", "TMQ", None),
        ("FSNT", "FSNT", None), ("FSNTOA", "FSNTOA", None),
        ("SOILWATER_10CM", "SOILWATER_10CM", 0.0),  # kg/m2, NaN over ocean -> 0
        ("TSOI_10CM", "TSOI_10CM", 270.0),          # K, NaN over ocean -> 270
    ]
    + [
        (f"{var}_l{i:02d}", f"{var}_{lev}", None)
        for var in UPPER_AIR_VARS
        for i, lev in enumerate(LEVEL_EXACT)
    ]
)
DIAG_SPECS = [("PRECT", "PRECT", None)]
FORCING_SPECS = [
    ("lsm", "PFTDATA_MASK", 0.0),     # 1 on land, NaN over ocean -> 0
    ("topo", "TOPO", 0.0),            # m, NaN over ocean -> 0
    ("glacier", "PCT_GLACIER", 0.0),  # %, NaN over ocean -> 0
    ("natveg", "PCT_NATVEG", 0.0),    # %, NaN over ocean -> 0
    ("sst", "SST", -1.8),             # degC, NaN over land -> -1.8 (freezing seawater)
    ("solin", "sol_in", None),        # W/m2, NaN-free (rsdt analogue)
    ("ice", "ICE", 0.0),              # fraction, NaN over land -> 0
]
assert len(STATE_SPECS) == 100 and len(DIAG_SPECS) == 1 and len(FORCING_SPECS) == 7
# 100 + 1 + 7 = 108: every archive channel except the 54 cloud ones, exactly once.
assert len({k for _, k, _ in STATE_SPECS + DIAG_SPECS + FORCING_SPECS}) == 108

STATE_CHANNELS = [n for n, _, _ in STATE_SPECS]
DIAG_CHANNELS = [n for n, _, _ in DIAG_SPECS]
FORCING_CHANNELS = [n for n, _, _ in FORCING_SPECS]
TARGET_CHANNELS = STATE_CHANNELS + DIAG_CHANNELS
# DERIVED, never restated. These were literals (154, 155, 7) and the 2026-07-16 cloud
# exclusion silently invalidated them — the asserts above check STATE_SPECS, not these,
# so they would have written a correct 100-channel store advertising 154 in its metadata.
# Same duplication bug as physicsnemo's nr_predicted_variables=157. Today: 100, 101, 7.
N_STATE, N_TARGET, N_FORCING = len(STATE_CHANNELS), len(TARGET_CHANNELS), len(FORCING_CHANNELS)

LAT = np.arange(89.5, -90.0, -1.0)   # DESCENDING, matches flipped data rows
LON = np.arange(0.5, 360.0, 1.0)
assert LAT.shape == (H,) and LON.shape == (W,)


def _fill_stack(g, specs) -> np.ndarray:
    """(C, H, W) float32 from an open /input group: lat-flipped, NaN-filled."""
    out = np.empty((len(specs), H, W), dtype=np.float32)
    for i, (_name, key, fill) in enumerate(specs):
        arr = g[key][...]
        if arr.shape != (H, W):
            raise RuntimeError(f"{key} shape {arr.shape} != {(H, W)}")
        arr = arr[::-1, :]  # ascending (south-first) -> descending (north-first)
        if fill is not None:
            arr = np.where(np.isnan(arr), np.float32(fill), arr)
        elif np.isnan(arr).any():
            raise RuntimeError(f"{key} has unexpected NaNs (no fill configured)")
        out[i] = arr
    return out


def _read_sample(path: str):
    """One archive file -> (state, diag, forcing). Single open, one read per kept channel."""
    with h5py.File(path, "r") as f:
        g = f["input"]
        return (_fill_stack(g, STATE_SPECS),
                _fill_stack(g, DIAG_SPECS),
                _fill_stack(g, FORCING_SPECS))


def _year_files(e3sm_root: str, year: int, max_samples: int | None) -> list[str]:
    files = sorted(glob.glob(os.path.join(e3sm_root, "h5", "plev_data", f"{year}_*.h5")))
    if not files:
        raise RuntimeError(f"no input files for year {year} under {e3sm_root}/h5/plev_data")
    return files[:max_samples] if max_samples else files


def _write_year(out_path: str, e3sm_root: str, year: int, offset_seconds: int,
                max_samples: int | None) -> int:
    """Stream one year into out_path. Returns T. (Stats are a separate pass.)"""
    files = _year_files(e3sm_root, year, max_samples)
    T = len(files)
    tmp = out_path + ".tmp"
    with h5py.File(tmp, "w") as f:
        ds_defs = [
            ("fields_state", N_STATE, "channel_state", STATE_CHANNELS),
            ("fields_diagnostic", 1, "channel_diagnostic", DIAG_CHANNELS),
            ("forcing", N_FORCING, "channel_forcing", FORCING_CHANNELS),
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
        f.attrs["converter"] = "makani_sfno/polaris/convert_e3sm_to_makani_alldata.py"
        f.attrs["lat_order"] = "descending (row 0 = +89.5)"
        f.attrs["sst_units"] = "degC, land filled with -1.8"
        f.attrs["level_naming"] = ("VAR_lNN by level index, l00=top .. l17=near-surface; "
                                   "levels are terrain-following, NOT isobaric")

        for t, path in enumerate(files):
            state, diag, forc = _read_sample(path)
            payloads["fields_state"][t] = state
            payloads["fields_diagnostic"][t] = diag
            payloads["forcing"][t] = forc
            if t % 200 == 0:
                print(f"  [{year}] {t}/{T}", flush=True)
    os.replace(tmp, out_path)
    print(f"wrote {out_path} (T={T})", flush=True)
    return T


def _accumulate_stats_from_packed(train_dir: str) -> dict:
    """Second pass: float64 sum/sumsq/time-sum over the PACKED train files.

    Re-reading what was written makes the pack resumable (skip-existing is
    safe for train years) and guarantees the stats describe the exact bytes
    the trainer will read, fills and all.
    """
    files = sorted(glob.glob(os.path.join(train_dir, "*.h5")))
    if not files:
        raise RuntimeError(f"stats pass: no packed train files under {train_dir}")
    accum = {
        "n": 0, "t_count": 0,
        "sum_t": np.zeros(N_TARGET), "sumsq_t": np.zeros(N_TARGET),
        "tsum_t": np.zeros((N_TARGET, H, W)),
        "sum_f": np.zeros(N_FORCING), "sumsq_f": np.zeros(N_FORCING),
        "tsum_f": np.zeros((N_FORCING, H, W)),
    }
    for path in files:
        with h5py.File(path, "r") as f:
            st, dg, fo = f["fields_state"], f["fields_diagnostic"], f["forcing"]
            if st.shape[1] != N_STATE or fo.shape[1] != N_FORCING:
                raise RuntimeError(
                    f"stats pass: {path} has state/forcing channels "
                    f"{st.shape[1]}/{fo.shape[1]}, expected {N_STATE}/{N_FORCING} "
                    "(is this an old locked-contract pack in the alldata root?)")
            T = st.shape[0]
            for t0 in range(0, T, STATS_CHUNK_T):
                t1 = min(t0 + STATS_CHUNK_T, T)
                tgt = np.concatenate(
                    [st[t0:t1].astype(np.float64), dg[t0:t1].astype(np.float64)],
                    axis=1)
                frc = fo[t0:t1].astype(np.float64)
                nt = t1 - t0
                accum["n"] += nt * H * W
                accum["t_count"] += nt
                accum["sum_t"] += tgt.sum(axis=(0, 2, 3))
                accum["sumsq_t"] += (tgt * tgt).sum(axis=(0, 2, 3))
                accum["tsum_t"] += tgt.sum(axis=0)
                accum["sum_f"] += frc.sum(axis=(0, 2, 3))
                accum["sumsq_f"] += (frc * frc).sum(axis=(0, 2, 3))
                accum["tsum_f"] += frc.sum(axis=0)
        print(f"  stats <- {path}", flush=True)
    return accum


def _write_stats(stats_dir: str, accum: dict) -> list[str]:
    """Write the 6 stats .npy files; return zero-variance channel names."""
    os.makedirs(stats_dir, exist_ok=True)
    n, tc = accum["n"], accum["t_count"]
    zero_var: list[str] = []
    for tag, C, names, s, ss, ts in (
        ("", N_TARGET, TARGET_CHANNELS, accum["sum_t"], accum["sumsq_t"], accum["tsum_t"]),
        ("forcing_", N_FORCING, FORCING_CHANNELS, accum["sum_f"], accum["sumsq_f"], accum["tsum_f"]),
    ):
        mean = s / n
        var = np.maximum(ss / n - mean * mean, 0.0)
        std = np.maximum(np.sqrt(var), STD_FLOOR)
        zero_var += [names[i] for i in np.flatnonzero(std <= ZERO_VAR_STD)]
        np.save(os.path.join(stats_dir, f"{tag}global_means.npy"),
                mean.astype(np.float32).reshape(1, C, 1, 1))
        np.save(os.path.join(stats_dir, f"{tag}global_stds.npy"),
                std.astype(np.float32).reshape(1, C, 1, 1))
        np.save(os.path.join(stats_dir, f"{tag}time_means.npy"),
                (ts / tc).astype(np.float32).reshape(1, C, H, W))
    print(f"wrote stats to {stats_dir} (n={n}, t_count={tc})", flush=True)
    print(f"zero-variance channels ({len(zero_var)}; expected 0 now that the cloud "
          f"fields are excluded — any entry is a finding): {zero_var}", flush=True)
    return zero_var


def _write_metadata(output_root: str, splits: dict, zero_var: list[str]) -> None:
    level_table = [
        {"index": i, "channel_suffix": f"l{i:02d}", "exact_dataset_suffix": lev,
         "nominal_hpa_label": LEVEL_NOMINAL_LABEL[i]}
        for i, lev in enumerate(LEVEL_EXACT)
    ]
    meta = {
        # Channel count DERIVED (108 today) — the old hardcoded "all162" name is
        # exactly the restatement bug this file's N_* derivation exists to kill.
        "dataset_name": f"e3smv3-ssp245amip-alldata{N_TARGET + N_FORCING}-180x360",
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
            "description": f"E3SMv3 SSP245-AMIP, {N_TARGET + N_FORCING} of the 162 "
                           "archive channels (clouds excluded) packed into the "
                           "PlaSim/Makani three-dataset layout as "
                           f"{N_STATE} state + 1 diag + {N_FORCING} forcing "
                           "(patched trainer only)",
            "source_root": splits["source_root"],
            "train_years": splits["train"],
            "valid_years": splits["valid"],
            "test_years": splits["test"],
            "requires_patched_makani": True,
            "sst_land_fill_degC": -1.8,
            "soilwater_ocean_fill": 0.0,
            "tsoi_ocean_fill_K": 270.0,
            "level_coordinate": "terrain-following (sigma-like), NOT isobaric; "
                                "nominal_hpa_label is a human label only — "
                                "measured: corr(Z3_l17, TOPO)=0.979 over land",
            "level_table": level_table,
            "zero_variance_channels": zero_var,
        },
    }
    meta_dir = os.path.join(output_root, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "data.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {meta_dir}/data.json", flush=True)


def _validate(output_root: str) -> None:
    """Single-sample read-back through the loader's access pattern."""
    train_files = sorted(glob.glob(os.path.join(output_root, "train", "*.h5")))
    assert train_files, "no train files written"
    with h5py.File(train_files[0], "r") as f:
        st = f["fields_state"]
        assert st.shape[1:] == (N_STATE, H, W) and st.dtype == np.float32, st.shape
        assert f["fields_diagnostic"].shape[1:] == (1, H, W)
        assert f["forcing"].shape[1:] == (N_FORCING, H, W)
        lat = st.dims[2]["lat"][...]
        lon = st.dims[3]["lon"][...]
        ts = st.dims[0]["timestamp"][...]
        assert lat[0] > lat[-1], "lat must be descending"
        assert lon.shape == (W,)
        d = np.diff(ts)
        assert d.size and np.all(d == STEP_SECONDS), "timestamps must step 21600 s"
        assert np.isfinite(st[0]).all(), "NaN/inf in fields_state sample 0"
        assert np.isfinite(f["forcing"][0]).all(), "NaN/inf in forcing sample 0"
        assert np.isfinite(f["fields_diagnostic"][0]).all()
    for name, shape in (
        ("global_means.npy", (1, N_TARGET, 1, 1)),
        ("global_stds.npy", (1, N_TARGET, 1, 1)),
        ("time_means.npy", (1, N_TARGET, H, W)),
        ("forcing_global_means.npy", (1, N_FORCING, 1, 1)),
        ("forcing_global_stds.npy", (1, N_FORCING, 1, 1)),
        ("forcing_time_means.npy", (1, N_FORCING, H, W)),
    ):
        arr = np.load(os.path.join(output_root, "stats", name))
        assert arr.shape == shape and np.isfinite(arr).all(), (name, arr.shape)
    stds = np.load(os.path.join(output_root, "stats", "global_stds.npy")).ravel()
    n_floor = int((stds <= ZERO_VAR_STD).sum())
    print(f"validate: min target std = {stds.min():.3e}; {n_floor} channel(s) at "
          f"the floor (expected 0 — the constant cloud channels are excluded)")


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
    for split, years in split_years.items():
        os.makedirs(os.path.join(args.output_root, split), exist_ok=True)
        offset = 0
        for year in years:
            out_path = os.path.join(args.output_root, split, f"{year}.h5")
            if os.path.exists(out_path) and not args.overwrite:
                # Safe here (unlike the locked converter): stats are a second
                # pass over the packed files, so a skipped year still counts.
                print(f"skip existing {out_path} (--overwrite to force)")
                with h5py.File(out_path, "r") as f:
                    if f["fields_state"].shape[1] != N_STATE:
                        sys.exit(f"ERROR WRONG_CONTRACT: {out_path} has "
                                 f"{f['fields_state'].shape[1]} state channels, "
                                 f"expected {N_STATE}. This output root holds a "
                                 "different pack — pick a fresh --output-root.")
                    offset += f["fields_state"].shape[0] * STEP_SECONDS
                continue
            T = _write_year(out_path, args.e3sm_root, year, offset,
                            args.max_samples_per_year)
            offset += T * STEP_SECONDS

    print("stats pass over packed train split ...", flush=True)
    accum = _accumulate_stats_from_packed(os.path.join(args.output_root, "train"))
    zero_var = _write_stats(os.path.join(args.output_root, "stats"), accum)
    _write_metadata(args.output_root,
                    {**split_years, "source_root": args.e3sm_root}, zero_var)
    if args.validate:
        _validate(args.output_root)
    print("CONVERT_ALLDATA_OK")


if __name__ == "__main__":
    main()
