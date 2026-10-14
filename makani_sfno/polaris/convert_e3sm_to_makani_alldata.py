#!/usr/bin/env python3
"""Pack the E3SM archive into makani training files: the all-data set.

This packs 108 of the archive's 162 channels: 100 state, 1 diagnostic and 7 forcing.
It sits beside convert_e3sm_to_makani.py (the locked 52-channel set kept for PlaSim
comparisons) and does not replace it. It writes to its own output root and is read
by the e3sm_alldata_*.yaml configs. The input and output layout are described in
e3sm_makani_pack.py.

Channels:
    state (100): PS, TREFHT, U10, RHREFHT, PSL, TMQ, FSNT, FSNTOA, SOILWATER_10CM,
        TSOI_10CM, then T, U, V, Z3 and RELHUM at all 18 levels.
    diagnostic (1): PRECT. Used in the loss, never fed back.
    forcing (7): lsm (PFTDATA_MASK), topo (TOPO), glacier (PCT_GLACIER),
        natveg (PCT_NATVEG), sst (SST), solin (sol_in), ice (ICE).
    NaN fills: SOILWATER_10CM 0 and TSOI_10CM 270 K over ocean; SST -1.8 deg C over
    land; 0 for the other land-only forcing channels.

The soil channels are state, not forcing: they change as the simulation runs, so
giving them to the model during a rollout would hand it part of the answer. SST and
ICE are prescribed in this AMIP run, sol_in follows the Earth's orbit, and the other
forcing channels are fixed land maps. The split and fills follow the PanguWeather E3SM
config (PanguWeather/v2.0/config/E3SM_SFNO_H5_POLARIS.yaml).

The cloud variables CLDICE, CLDLIQ and CLOUD (3 x 18 levels = 54 channels) are left
out. The science owner excluded them from every model on 2026-07-16.

Levels: channels are named {VAR}_l00 to {VAR}_l17 by level number, l00 at the top and
l17 nearest the surface. The archive's dataset names carry pressure-like numbers
(Z3_998.4964394917621), but the levels follow the terrain. Measured on Polaris: over
land, Z3 at the lowest level correlates 0.979 with TOPO and sits 15.4 m above it on
average, and Z3_492 reaches 9,266 m, where a true 500 hPa surface tops out near
6,000 m. Do not name these channels by pressure (T850 and so on). The exact suffix
and a rounded hPa label for each level are in metadata/data.json (attrs.level_table).

Statistics are computed in a second pass over the packed train files. This makes the
pack resumable: a rerun skips years already on disk and still counts them. For
parallel packing (see polaris_pack_alldata_production.pbs), run each worker with
--skip-stats and then one --stats-only run, which first checks that every train year
is packed.

Size: 108 channels x 1460 samples x 180 x 360 float32 = 40.9 GB per year, about
1.4 TB for all 35 years. Use --max-samples-per-year for small test packs.

Usage:
    python convert_e3sm_to_makani_alldata.py --e3sm-root $E3SM_ROOT --output-root OUT \\
        --train-years 2015 2015 --valid-years 2016 2016 --test-years 2017 2017 \\
        [--max-samples-per-year 96] [--overwrite] [--validate]

Prints CONVERT_ALLDATA_OK when done, or CHUNK_ALLDATA_OK with --skip-stats.
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np

import e3sm_makani_pack as pack

CONVERTER = "makani_sfno/polaris/convert_e3sm_to_makani_alldata.py"
STATS_CHUNK_T = 16  # samples read at a time in the statistics pass (~1.3 GB peak)

# Level suffixes exactly as they appear in the archive's dataset names, top to
# bottom. They are not pressures; see the module docstring.
LEVEL_EXACT = [
    "4.714998332947841", "10.655023096474308", "19.235455601758737",
    "28.79458853709195", "50.11779996521295", "69.59908688413749",
    "96.46377266572703", "145.04282239200347", "200.99889546355382",
    "256.72368590525895", "302.21364012188303", "385.999023919911",
    "492.46857402252755", "608.6437744215842", "713.7046383204334",
    "849.6612491105952", "925.5197481473349", "998.4964394917621",
]
# Rounded hPa labels for the same levels (from the PanguWeather E3SM configs). Written
# to the metadata as labels only; never used in channel names.
LEVEL_NOMINAL_LABEL = [5, 10, 20, 30, 50, 70, 100, 150,
                       200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
assert len(LEVEL_EXACT) == 18 and len(LEVEL_NOMINAL_LABEL) == 18

UPPER_AIR_VARS = ["T", "U", "V", "Z3", "RELHUM"]  # clouds excluded

# (channel name, archive key, NaN fill or None)
STATE_SPECS = (
    [
        ("PS", "PS", None), ("TREFHT", "TREFHT", None), ("U10", "U10", None),
        ("RHREFHT", "RHREFHT", None), ("PSL", "PSL", None), ("TMQ", "TMQ", None),
        ("FSNT", "FSNT", None), ("FSNTOA", "FSNTOA", None),
        ("SOILWATER_10CM", "SOILWATER_10CM", 0.0),  # kg/m2, NaN over ocean
        ("TSOI_10CM", "TSOI_10CM", 270.0),          # K, NaN over ocean
    ]
    + [
        (f"{var}_l{i:02d}", f"{var}_{lev}", None)
        for var in UPPER_AIR_VARS
        for i, lev in enumerate(LEVEL_EXACT)
    ]
)
DIAG_SPECS = [("PRECT", "PRECT", None)]
FORCING_SPECS = [
    ("lsm", "PFTDATA_MASK", 0.0),     # 1 on land, NaN over ocean
    ("topo", "TOPO", 0.0),            # m, NaN over ocean
    ("glacier", "PCT_GLACIER", 0.0),  # %, NaN over ocean
    ("natveg", "PCT_NATVEG", 0.0),    # %, NaN over ocean
    ("sst", "SST", -1.8),             # deg C, NaN over land
    ("solin", "sol_in", None),        # W/m2, incoming sunlight at the top of the atmosphere
    ("ice", "ICE", 0.0),              # fraction, NaN over land
]
assert len(STATE_SPECS) == 100 and len(DIAG_SPECS) == 1 and len(FORCING_SPECS) == 7
# 100 + 1 + 7 = 108: every archive channel except the 54 cloud ones, each once.
assert len({k for _, k, _ in STATE_SPECS + DIAG_SPECS + FORCING_SPECS}) == 108
SPECS = (STATE_SPECS, DIAG_SPECS, FORCING_SPECS)

# Other scripts import these. They are derived from the lists above, never typed in:
# typed-in counts went stale once already, when the clouds were dropped.
STATE_CHANNELS = pack.channel_names(STATE_SPECS)
DIAG_CHANNELS = pack.channel_names(DIAG_SPECS)
FORCING_CHANNELS = pack.channel_names(FORCING_SPECS)
TARGET_CHANNELS = STATE_CHANNELS + DIAG_CHANNELS
N_STATE, N_TARGET, N_FORCING = len(STATE_CHANNELS), len(TARGET_CHANNELS), len(FORCING_CHANNELS)

LEVEL_NAMING = ("VAR_lNN by level index, l00=top .. l17=near-surface; "
                "levels are terrain-following, NOT isobaric")


def sum_packed_train(train_dir):
    """Add up the statistics over every packed train file, in float64.

    Reading back the packed files, rather than summing while writing, means the
    statistics describe exactly what the trainer will read, and a train year packed
    by an earlier run still counts.

    Returns:
        Running sums in the format of e3sm_makani_pack.new_sums.
    """
    files = sorted(glob.glob(os.path.join(train_dir, "*.h5")))
    if not files:
        raise RuntimeError(f"stats pass: no packed train files under {train_dir}")
    sums = pack.new_sums(N_TARGET, N_FORCING)
    for path in files:
        with h5py.File(path, "r") as f:
            state, diag, forcing = f["fields_state"], f["fields_diagnostic"], f["forcing"]
            if state.shape[1] != N_STATE or forcing.shape[1] != N_FORCING:
                raise RuntimeError(
                    f"stats pass: {path} has {state.shape[1]} state and {forcing.shape[1]} "
                    f"forcing channels, expected {N_STATE} and {N_FORCING}")
            n = state.shape[0]
            for t0 in range(0, n, STATS_CHUNK_T):
                t1 = min(t0 + STATS_CHUNK_T, n)
                target = np.concatenate(
                    [state[t0:t1].astype(np.float64), diag[t0:t1].astype(np.float64)], axis=1)
                frc = forcing[t0:t1].astype(np.float64)
                sums["n"] += (t1 - t0) * pack.H * pack.W
                sums["t_count"] += t1 - t0
                sums["sum_t"] += target.sum(axis=(0, 2, 3))
                sums["sumsq_t"] += (target * target).sum(axis=(0, 2, 3))
                sums["tsum_t"] += target.sum(axis=0)
                sums["sum_f"] += frc.sum(axis=(0, 2, 3))
                sums["sumsq_f"] += (frc * frc).sum(axis=(0, 2, 3))
                sums["tsum_f"] += frc.sum(axis=0)
        print(f"  stats <- {path}", flush=True)
    return sums


def main():
    parser = argparse.ArgumentParser(
        description="Pack E3SM into makani training files (all-data 108-channel set).")
    pack.add_common_args(parser)
    parser.add_argument("--skip-stats", action="store_true",
                        help="pack only; a later --stats-only run writes the statistics")
    parser.add_argument("--stats-only", action="store_true",
                        help="pack nothing; check every train year is packed, then "
                             "write the statistics and metadata")
    args = parser.parse_args()
    years = pack.split_years(args)

    if args.stats_only:
        have = sorted(int(os.path.basename(p)[:-3])
                      for p in glob.glob(os.path.join(args.output_root, "train", "*.h5")))
        if have != years["train"]:
            sys.exit(f"ERROR STATS_ONLY_INCOMPLETE: packed train years {have} != requested "
                     f"{years['train']}. Statistics from part of the train split would be "
                     f"wrong. Finish packing first.")
    else:
        pack.pack_splits(args, years, SPECS, CONVERTER, {"level_naming": LEVEL_NAMING})
        if args.skip_stats:
            print("CHUNK_ALLDATA_OK (packed; stats deferred to a --stats-only pass)")
            return

    print("stats pass over packed train split ...", flush=True)
    sums = sum_packed_train(os.path.join(args.output_root, "train"))
    constant = pack.write_stats(os.path.join(args.output_root, "stats"), sums,
                                TARGET_CHANNELS, FORCING_CHANNELS)
    # With the clouds left out, no channel should be constant; any name here is a finding.
    print(f"constant channels: {len(constant)} (expected 0) {constant}", flush=True)
    level_table = [
        {"index": i, "channel_suffix": f"l{i:02d}", "exact_dataset_suffix": lev,
         "nominal_hpa_label": LEVEL_NOMINAL_LABEL[i]}
        for i, lev in enumerate(LEVEL_EXACT)
    ]
    pack.write_metadata(
        args.output_root,
        dataset_name=f"e3smv3-ssp245amip-alldata{N_TARGET + N_FORCING}-180x360",
        description=f"E3SMv3 SSP245-AMIP, {N_TARGET + N_FORCING} of the 162 "
                    "archive channels (clouds excluded) packed into the "
                    "PlaSim/Makani three-dataset layout as "
                    f"{N_STATE} state + 1 diag + {N_FORCING} forcing "
                    "(patched trainer only)",
        years=years, e3sm_root=args.e3sm_root, specs=SPECS,
        extra_attrs={
            "soilwater_ocean_fill": 0.0,
            "tsoi_ocean_fill_K": 270.0,
            "level_coordinate": "terrain-following (sigma-like), NOT isobaric; "
                                "nominal_hpa_label is a human label only — "
                                "measured: corr(Z3_l17, TOPO)=0.979 over land",
            "level_table": level_table,
            "zero_variance_channels": constant,
        })
    if args.validate:
        pack.validate(args.output_root, years, SPECS, args.max_samples_per_year)
    print("CONVERT_ALLDATA_OK")


if __name__ == "__main__":
    main()
