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

The statistics pass also writes stats/time_diff_stds.npy: for each target channel,
the std of its change from one 6-hour step to the next. Steps never span two year
files. makani reads this file only when temp_diff_normalization is on; on 2026-10-14
no config turns it on.

Other options:
    --workers N        run the statistics pass on N processes, one year file each,
                       and add up the results. One process took 52 minutes for 30
                       train years (job 7565734).
    --time-diff-only   write only stats/time_diff_stds.npy and leave the other
                       statistics and the metadata untouched, so a pack in use can
                       gain it without its normalization changing.

Size: 108 channels x 1460 samples x 180 x 360 float32 = 40.9 GB per year, about
1.4 TB for all 35 years. Use --max-samples-per-year for small test packs.

Usage:
    python convert_e3sm_to_makani_alldata.py --e3sm-root $E3SM_ROOT --output-root OUT \\
        --train-years 2015 2015 --valid-years 2016 2016 --test-years 2017 2017 \\
        [--max-samples-per-year 96] [--overwrite] [--validate] [--workers N]

Prints CONVERT_ALLDATA_OK when done, CHUNK_ALLDATA_OK with --skip-stats, or
TIME_DIFF_STDS_OK with --time-diff-only.
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np

import e3sm_makani_pack as pack

CONVERTER = "makani_sfno/polaris/convert_e3sm_to_makani_alldata.py"
# Module-level so the tests can shrink the grid and the read size.
H, W = pack.H, pack.W
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


def _new_stats_accum():
    """Return zeroed running sums: those of e3sm_makani_pack.new_sums, plus the sum
    (sum_d), sum of squares (sumsq_d) and count (n_d) of the one-step changes."""
    sums = pack.new_sums(N_TARGET, N_FORCING, H, W)
    sums.update({"sum_d": np.zeros(N_TARGET), "sumsq_d": np.zeros(N_TARGET), "n_d": 0})
    return sums


def _combine_stats(parts):
    """Add up per-file running sums, in file order.

    Every entry is a sum and steps never span files, so the result matches the
    one-process pass up to float64 rounding.
    """
    sums = _new_stats_accum()
    for part in parts:
        for k in sums:
            sums[k] += part[k]
    return sums


def _diff_chunk(prev, target):
    """Return the one-step changes x(t+1) - x(t) for a chunk, and its last sample.

    Args:
        prev: Last sample of the previous chunk of the same file, or None at the
            start of a file. Sample 0 of one year does not follow the last sample
            of the year before in the trainer's indexing, so no pair spans files.
        target: The chunk, (t, C, H, W).

    Returns:
        (changes, last sample), to pass as prev for the next chunk.
    """
    seq = target if prev is None else np.concatenate([prev, target], axis=0)
    return np.diff(seq, axis=0), target[-1:]


def _accumulate_stats_file(path, accum=None):
    """Add one packed train file to the running sums, in float64.

    Args:
        path: A packed year file.
        accum: Running sums to add to; a new set if None. A process pool maps this
            function over files with accum None, so it must stay at module level.

    Returns:
        The running sums.
    """
    if accum is None:
        accum = _new_stats_accum()
    prev = None
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
            accum["n"] += (t1 - t0) * H * W
            accum["t_count"] += t1 - t0
            accum["sum_t"] += target.sum(axis=(0, 2, 3))
            accum["sumsq_t"] += (target * target).sum(axis=(0, 2, 3))
            accum["tsum_t"] += target.sum(axis=0)
            accum["sum_f"] += frc.sum(axis=(0, 2, 3))
            accum["sumsq_f"] += (frc * frc).sum(axis=(0, 2, 3))
            accum["tsum_f"] += frc.sum(axis=0)
            d, prev = _diff_chunk(prev, target)
            accum["sum_d"] += d.sum(axis=(0, 2, 3))
            accum["sumsq_d"] += (d * d).sum(axis=(0, 2, 3))
            accum["n_d"] += d.shape[0] * H * W
    print(f"  stats <- {path}", flush=True)
    return accum


def _accumulate_stats_from_packed(train_dir, workers=1):
    """Add up the statistics over every packed train file.

    Reading back the packed files, rather than summing while writing, means the
    statistics describe exactly what the trainer will read, and a train year packed
    by an earlier run still counts.

    Args:
        train_dir: {output_root}/train.
        workers: With more than 1, each process sums one file and the results are
            added up with _combine_stats. With 1, one running sum, in file order.

    Returns:
        The running sums.
    """
    files = sorted(glob.glob(os.path.join(train_dir, "*.h5")))
    if not files:
        raise RuntimeError(f"stats pass: no packed train files under {train_dir}")
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            return _combine_stats(pool.map(_accumulate_stats_file, files, chunksize=1))
    accum = _new_stats_accum()
    for path in files:
        _accumulate_stats_file(path, accum)
    return accum


def _accumulate_time_diff_file(path):
    """Return the float64 (sum, sum of squares, count) of one file's one-step changes.

    Used by --time-diff-only. A process pool maps it over files, so it must stay at
    module level.
    """
    s, ss, n = np.zeros(N_TARGET), np.zeros(N_TARGET), 0
    with h5py.File(path, "r") as f:
        state, diag = f["fields_state"], f["fields_diagnostic"]
        prev = None
        for t0 in range(0, state.shape[0], STATS_CHUNK_T):
            t1 = min(t0 + STATS_CHUNK_T, state.shape[0])
            target = np.concatenate(
                [state[t0:t1].astype(np.float64), diag[t0:t1].astype(np.float64)], axis=1)
            d, prev = _diff_chunk(prev, target)
            s += d.sum(axis=(0, 2, 3))
            ss += (d * d).sum(axis=(0, 2, 3))
            n += d.shape[0] * H * W
    print(f"  time-diff <- {path}", flush=True)
    return s, ss, n


def _time_diff_stds(s, ss, n):
    """Return the std of the one-step changes as (1, N_TARGET, 1, 1) float32.

    Same shape and channel order as global_stds.npy, because makani indexes both by
    the run's channel list. Physical units and no floor: makani applies its own floor
    of 1e-4, and a channel that hits it is worth reporting, not hiding.
    """
    mean = s / n
    var = np.maximum(ss / n - mean * mean, 0.0)
    return np.sqrt(var).astype(np.float32).reshape(1, N_TARGET, 1, 1)


def _require_complete_train(output_root, train_years):
    """Stop unless the packed train years are exactly train_years.

    Statistics from part of the train split would be wrong for the model.
    """
    have = sorted(int(os.path.basename(p)[:-3])
                  for p in glob.glob(os.path.join(output_root, "train", "*.h5")))
    if have != train_years:
        sys.exit(f"ERROR STATS_ONLY_INCOMPLETE: packed train years {have} != requested "
                 f"{train_years}. Statistics from part of the train split would be "
                 f"wrong. Finish packing first.")


def _write_stats(stats_dir, accum):
    """Write the six statistics files and, if there are step sums, time_diff_stds.npy.

    Returns:
        Names of channels whose std is at or below e3sm_makani_pack.ZERO_VAR_STD.
    """
    constant = pack.write_stats(stats_dir, accum, TARGET_CHANNELS, FORCING_CHANNELS)
    if accum.get("n_d"):
        np.save(os.path.join(stats_dir, "time_diff_stds.npy"),
                _time_diff_stds(accum["sum_d"], accum["sumsq_d"], accum["n_d"]))
    # With the clouds left out, no channel should be constant; any name here is a finding.
    print(f"constant channels: {len(constant)} (expected 0) {constant}", flush=True)
    return constant


def write_time_diff_only(args):
    """Write only stats/time_diff_stds.npy for a fully packed train split."""
    train_years = list(range(args.train_years[0], args.train_years[1] + 1))
    _require_complete_train(args.output_root, train_years)
    out = os.path.join(args.output_root, "stats", "time_diff_stds.npy")
    if os.path.exists(out) and not args.overwrite:
        sys.exit(f"ERROR TIME_DIFF_EXISTS: {out} (--overwrite to replace)")
    files = sorted(glob.glob(os.path.join(args.output_root, "train", "*.h5")))
    print(f"time-diff pass over {len(files)} train files, {args.workers} workers ...",
          flush=True)
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.workers) as pool:
            parts = pool.map(_accumulate_time_diff_file, files, chunksize=1)
    else:
        parts = [_accumulate_time_diff_file(p) for p in files]
    s = sum(part[0] for part in parts)
    ss = sum(part[1] for part in parts)
    n = sum(part[2] for part in parts)
    tmp = out + ".tmp.npy"
    np.save(tmp, _time_diff_stds(s, ss, n))
    os.replace(tmp, out)
    print(f"wrote {out} (n={n}, {len(files)} files)")
    print("TIME_DIFF_STDS_OK")


def main():
    parser = argparse.ArgumentParser(
        description="Pack E3SM into makani training files (all-data 108-channel set).")
    pack.add_common_args(parser)
    parser.add_argument("--skip-stats", action="store_true",
                        help="pack only; a later --stats-only run writes the statistics")
    parser.add_argument("--stats-only", action="store_true",
                        help="pack nothing; check every train year is packed, then "
                             "write the statistics and metadata")
    parser.add_argument("--time-diff-only", action="store_true",
                        help="write only stats/time_diff_stds.npy; leave the other "
                             "statistics and the metadata untouched")
    parser.add_argument("--workers", type=int, default=1,
                        help="processes for the statistics pass, one year file each")
    args = parser.parse_args()

    # Valid and test years play no part here, so they are not checked against train.
    if args.time_diff_only:
        write_time_diff_only(args)
        return

    years = pack.split_years(args)
    if args.stats_only:
        _require_complete_train(args.output_root, years["train"])
    else:
        pack.pack_splits(args, years, SPECS, CONVERTER, {"level_naming": LEVEL_NAMING})
        if args.skip_stats:
            print("CHUNK_ALLDATA_OK (packed; stats deferred to a --stats-only pass)")
            return

    print("stats pass over packed train split ...", flush=True)
    accum = _accumulate_stats_from_packed(os.path.join(args.output_root, "train"),
                                          workers=args.workers)
    constant = _write_stats(os.path.join(args.output_root, "stats"), accum)
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
