#!/usr/bin/env python3
"""Pack the E3SM archive into makani training files: the locked 52-channel set.

The locked set has the same shape as the group's PlaSim data (52 state, 1 diagnostic,
6 forcing channels), so results stay comparable with the PlaSim baseline. It must not
change. convert_e3sm_to_makani_alldata.py packs the wider set (100 state channels).
The input and output layout are described in e3sm_makani_pack.py.

Channels:
    state (52): PS, TREFHT, then T, U, V, RH (RELHUM) and Z (Z3) at the 10 lowest of
        the archive's 18 levels.
    diagnostic (1): PRECT, the precipitation rate. Used in the loss, never fed back.
    forcing (6): lsm (PFTDATA_MASK), topo (TOPO), glacier (PCT_GLACIER), sst (SST),
        solin (sol_in), ice (ICE).
    NaN fills: SST -1.8 (deg C, the freezing point of seawater) over land; 0 for the
    other forcing channels. The state and diagnostic channels have no NaN.

The channel names (T850, Z500, ...) say pressure levels, but the archive levels follow
the terrain (see convert_e3sm_to_makani_alldata.py). The names stay because the locked
set must not change.

Statistics come from the packed train years and are added up while each sample is
written. A train year that already exists on disk is skipped, so it is missing from
the sums; the script then stops with an error rather than write statistics from part
of the train split. Pass --overwrite to repack it.

A full pack does not fit in one 1-hour debug job, so it can be split across parallel
runs (see polaris_pack_e3sm_production.pbs):
    --chunk-accum-out FILE   pack this run's years and save their running sums to FILE
    --merge-accums FILES     add up the saved sums; write the statistics and metadata
The merged sums match a single run up to float64 rounding.

Usage:
    python convert_e3sm_to_makani.py --e3sm-root $E3SM_ROOT --output-root OUT \\
        --train-years 2015 2015 --valid-years 2016 2016 --test-years 2017 2017 \\
        [--max-samples-per-year 400] [--overwrite] [--validate]

Prints CONVERT_OK when done, or CHUNK_OK in chunk mode.
"""

import argparse
import os
import sys

import numpy as np

import e3sm_makani_pack as pack

CONVERTER = "makani_sfno/polaris/convert_e3sm_to_makani.py"

# Level suffixes exactly as they appear in the archive's dataset names, and the
# rounded values used in the channel names.
PLEV_EXACT = [
    "200.99889546355382", "256.72368590525895", "302.21364012188303",
    "385.999023919911", "492.46857402252755", "608.6437744215842",
    "713.7046383204334", "849.6612491105952", "925.5197481473349",
    "998.4964394917621",
]
PLEV_NOMINAL = [200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# (channel name, archive key, NaN fill or None)
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
    ("lsm", "PFTDATA_MASK", 0.0),     # 1 on land, NaN over ocean
    ("topo", "TOPO", 0.0),            # m, NaN over ocean
    ("glacier", "PCT_GLACIER", 0.0),  # %, NaN over ocean
    ("sst", "SST", -1.8),             # deg C, NaN over land
    ("solin", "sol_in", None),        # W/m2, incoming sunlight at the top of the atmosphere
    ("ice", "ICE", 0.0),              # fraction, NaN over land
]
assert len(STATE_SPECS) == 52 and len(DIAG_SPECS) == 1 and len(FORCING_SPECS) == 6
SPECS = (STATE_SPECS, DIAG_SPECS, FORCING_SPECS)

STATE_CHANNELS = pack.channel_names(STATE_SPECS)
DIAG_CHANNELS = pack.channel_names(DIAG_SPECS)
FORCING_CHANNELS = pack.channel_names(FORCING_SPECS)
TARGET_CHANNELS = STATE_CHANNELS + DIAG_CHANNELS  # 53

SUM_KEYS = ("n", "t_count", "sum_t", "sumsq_t", "tsum_t", "sum_f", "sumsq_f", "tsum_f")


def add_sample(sums, state, diag, forcing):
    """Add one packed sample to the running sums, in float64."""
    target = np.concatenate([state, diag], axis=0).astype(np.float64)
    forcing = forcing.astype(np.float64)
    sums["n"] += pack.H * pack.W
    sums["sum_t"] += target.sum(axis=(1, 2))
    sums["sumsq_t"] += (target * target).sum(axis=(1, 2))
    sums["tsum_t"] += target
    sums["sum_f"] += forcing.sum(axis=(1, 2))
    sums["sumsq_f"] += (forcing * forcing).sum(axis=(1, 2))
    sums["tsum_f"] += forcing
    sums["t_count"] += 1


def merge_sums(paths, train_years):
    """Add up the running sums saved by chunk runs.

    Args:
        paths: .npz files written with --chunk-accum-out.
        train_years: The full train split. The chunks must cover it exactly: a
            missing year would leave it out of the statistics, and a repeated year
            would count it twice.

    Returns:
        The combined sums.
    """
    covered, sums = [], None
    for path in paths:
        z = np.load(path)
        covered += [int(y) for y in z["packed_train_years"]]
        part = {k: z[k] for k in SUM_KEYS}
        if sums is None:
            sums = {k: (int(v) if v.ndim == 0 else v.astype(np.float64)) for k, v in part.items()}
        else:
            for k, v in part.items():
                sums[k] = sums[k] + (int(v) if v.ndim == 0 else v)
    if sorted(covered) != train_years:
        sys.exit(f"ERROR merge coverage mismatch: chunks cover {sorted(covered)} but "
                 f"--train-years requests {train_years}. A gap leaves a year out of the "
                 f"statistics; a repeat counts it twice.")
    return sums


def main():
    parser = argparse.ArgumentParser(
        description="Pack E3SM into makani training files (locked 52-channel set).")
    pack.add_common_args(parser)
    parser.add_argument("--chunk-accum-out", default=None,
                        help="pack this run's years, save their running sums here, "
                             "and write no statistics")
    parser.add_argument("--merge-accums", nargs="+", default=None,
                        help="sums saved by chunk runs: check they cover the train "
                             "years, then write statistics and metadata; pack nothing")
    parser.add_argument("--timestamp-offset-samples", type=int, default=0,
                        help="samples before this chunk's first train year (1460 per "
                             "earlier year), so timestamps run on across chunks")
    args = parser.parse_args()
    years = pack.split_years(args)

    if args.merge_accums:
        sums = merge_sums(args.merge_accums, years["train"])
    else:
        sums = pack.new_sums(len(TARGET_CHANNELS), len(FORCING_CHANNELS))
        packed, skipped = pack.pack_splits(
            args, years, SPECS, CONVERTER,
            train_offset_samples=args.timestamp_offset_samples,
            on_train_sample=lambda s, d, f: add_sample(sums, s, d, f))
        if args.chunk_accum_out is not None:
            if skipped:
                sys.exit(f"ERROR chunk skipped existing train year(s) {skipped}: they are "
                         f"in no saved sums. Rerun this chunk with --overwrite.")
            np.savez(args.chunk_accum_out,
                     packed_train_years=np.array(packed, dtype=np.int64), **sums)
            print(f"CHUNK_OK {args.chunk_accum_out} train_years={packed}")
            return
        if sums["t_count"] == 0:
            sys.exit("ERROR no train samples processed (did every train file already "
                     "exist? Rerun with --overwrite to rebuild the statistics.)")
        if skipped:
            sys.exit(f"ERROR partial train stats: year(s) {skipped} already existed and "
                     f"were skipped, so they are missing from the statistics. Rerun with "
                     f"--overwrite to repack the train split and statistics together.")

    pack.write_stats(os.path.join(args.output_root, "stats"), sums,
                     TARGET_CHANNELS, FORCING_CHANNELS)
    pack.write_metadata(
        args.output_root,
        dataset_name="e3smv3-ssp245amip-plev10-180x360",
        description="E3SMv3 SSP245-AMIP plev subset packed into the "
                    "PlaSim/Makani three-dataset layout (patched trainer only)",
        years=years, e3sm_root=args.e3sm_root, specs=SPECS,
        extra_attrs={"plev_exact_hpa": PLEV_EXACT})
    if args.validate:
        pack.validate(args.output_root, years, SPECS, args.max_samples_per_year)
    print("CONVERT_OK")


if __name__ == "__main__":
    main()
