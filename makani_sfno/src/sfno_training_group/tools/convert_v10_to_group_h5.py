"""Convert v10 zgplev packaged h5 → group PanguWeather v2.0 per-timestep layout.

Phase B.1 of the group-code training track plan v5.

Reads our v10 stacked h5 files at:
    <src>/{train,valid,test}/MOST.<YYYY>.h5
and emits, into a single flat output dir, one h5 file per timestep:
    <dst>/<year>_<idx:04>.h5
with per-level 2D datasets under /input/ as required by group's
data_loader_multifiles.GetDataset (lines 632-643, 73, 161).

Synthetic group calendar — output filenames use a chosen group year (1:1 with
v10's MOST.YYYY.h5 by default), NOT any source PlaSim timestamp.
v10 sources carry plasim_time_units = "days since 0017-08-01" which would
naively decode to year 17/18. We ignore that and emit synthetic year YYYY.

Outputs alongside the per-timestep h5 a manifest at
    <dst>/_v10_calendar_manifest.json
recording, per year:
- n_timesteps, synthetic_start_dt, synthetic_last_idx_dt
- last_train_init_dt, train_end_exclusive_dt          (= last_train_init + 6h)
- last_val_init_dt_for_max_lead_K, val_end_exclusive_dt_for_max_lead_K
- z0_temporal_std_mean (audit)

The render_yaml tool consumes the *_end_exclusive_dt fields when building the
training YAML's train_data_sets / validation_data_sets blocks (the group
loader's partition_date_range is exclusive on `end`).

The script will additionally try to apply ``lfs setstripe -c 1`` to the output
directory once (small-file Lustre layout) — failure is logged but non-fatal.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import cftime
import h5py
import numpy as np

# Repo's _h5_keys is in the same package; this file is a -m entrypoint.
from sfno_training_group.tools._h5_keys import (
    PLEVS_PA,
    SIGMA_LEVELS,
    ZG_HPA,
    h5_key,
)

logger = logging.getLogger("convert_v10_to_group_h5")

DATA_TIMEDELTA_HOURS = 6
N_LEVELS = 10

# Phase F padding contract (plan v5 §F.0). For long_inference single_ic mode:
# year 11 (no_leap_year) needs 1460 frames to span a non-leap synthetic year;
# year 12 (leap_year) needs 1464 frames to span a leap synthetic year.
# Source data has 1455 frames per year; pad with year-(N+1)'s first frames.
PAD_TARGETS = {
    11: {"n_native": 1455, "n_padded": 1460, "is_leap": False, "donor_year": 12},
    12: {"n_native": 1455, "n_padded": 1464, "is_leap": True,  "donor_year": 13},
}

# v10 channel manifest (must match what's stored in h5 attrs/datasets).
EXPECTED_STATE_CHANNELS: list[str] = (
    ["pl", "tas"]
    + [f"ta{i+1}" for i in range(N_LEVELS)]
    + [f"ua{i+1}" for i in range(N_LEVELS)]
    + [f"va{i+1}" for i in range(N_LEVELS)]
    + [f"hus{i+1}" for i in range(N_LEVELS)]
    + [f"zg{hpa}" for hpa in ZG_HPA]
)
EXPECTED_DIAGNOSTIC_CHANNELS: list[str] = ["pr_6h"]
EXPECTED_FORCING_CHANNELS: list[str] = ["lsm", "sg", "z0", "sst", "rsdt", "sic"]

# v10 split layout — converter searches for MOST.YYYY.h5 under each subdir.
V10_SPLIT_DIRS = ("train", "valid", "test")


def _read_v10_channel_lists(src_h5: h5py.File) -> tuple[list[str], list[str], list[str]]:
    def _decode(arr) -> list[str]:
        return [v.decode() if isinstance(v, bytes) else str(v) for v in arr[:]]

    return (
        _decode(src_h5["channel_state"]),
        _decode(src_h5["channel_diagnostic"]),
        _decode(src_h5["channel_forcing"]),
    )


def _assert_v10_channels(src_path: Path, src_h5: h5py.File) -> None:
    state, diag, forc = _read_v10_channel_lists(src_h5)
    if state != EXPECTED_STATE_CHANNELS:
        raise RuntimeError(
            f"{src_path}: channel_state mismatch.\n"
            f"  expected: {EXPECTED_STATE_CHANNELS}\n"
            f"  got:      {state}"
        )
    if diag != EXPECTED_DIAGNOSTIC_CHANNELS:
        raise RuntimeError(f"{src_path}: channel_diagnostic mismatch: {diag}")
    if forc != EXPECTED_FORCING_CHANNELS:
        raise RuntimeError(f"{src_path}: channel_forcing mismatch: {forc}")


def _find_v10_source(src_root: Path, year: int) -> Path:
    """Locate MOST.<YYYY>.h5 under one of train/valid/test subdirs."""
    for split in V10_SPLIT_DIRS:
        candidate = src_root / split / f"MOST.{year:04d}.h5"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"MOST.{year:04d}.h5 not found under any of {V10_SPLIT_DIRS} of {src_root}"
    )


def _try_lfs_setstripe(dst: Path) -> None:
    try:
        subprocess.run(
            ["lfs", "setstripe", "-c", "1", str(dst)],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Applied `lfs setstripe -c 1` to %s", dst)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        logger.warning("lfs setstripe skipped on %s: %s", dst, e)


def _write_per_timestep_h5(
    out_path: Path,
    *,
    fields_state: np.ndarray,       # (52, 64, 128) float32
    fields_diagnostic: np.ndarray,  # (1, 64, 128)  float32
    forcing: np.ndarray,            # (6, 64, 128)  float32
) -> None:
    """Write one timestep file with /input/<key> per the group format."""
    # Sanity (cheap): catch shape drift early.
    assert fields_state.shape == (52, 64, 128), fields_state.shape
    assert fields_diagnostic.shape == (1, 64, 128), fields_diagnostic.shape
    assert forcing.shape == (6, 64, 128), forcing.shape

    with h5py.File(out_path, "w") as f:
        g = f.create_group("input")

        # Surface state.
        g.create_dataset("pl", data=fields_state[0].astype(np.float32))
        g.create_dataset("tas", data=fields_state[1].astype(np.float32))

        # Upper-air state per (var, level).
        # state[2..11]   -> ta1..ta10  (sigma TOA->surface)
        # state[12..21]  -> ua1..ua10
        # state[22..31]  -> va1..va10
        # state[32..41]  -> hus1..hus10
        # state[42..51]  -> zg200..zg1000 (pressure ascending)
        for var, base_idx in (("ta", 2), ("ua", 12), ("va", 22), ("hus", 32)):
            for level_i in range(N_LEVELS):
                key = h5_key(var, level_i)
                g.create_dataset(key, data=fields_state[base_idx + level_i].astype(np.float32))
        for level_i in range(N_LEVELS):
            key = h5_key("zg", level_i)
            g.create_dataset(key, data=fields_state[42 + level_i].astype(np.float32))

        # Diagnostic.
        g.create_dataset("pr_6h", data=fields_diagnostic[0].astype(np.float32))

        # Forcing (constant + varying boundary, co-located in the same files).
        for fname, idx in (("lsm", 0), ("sg", 1), ("z0", 2),
                           ("sst", 3), ("rsdt", 4), ("sic", 5)):
            g.create_dataset(fname, data=forcing[idx].astype(np.float32))


def _synthetic_dt(year: int, idx: int) -> cftime.DatetimeProlepticGregorian:
    return cftime.DatetimeProlepticGregorian(year, 1, 1, 0, 0, 0, has_year_zero=True) \
           + timedelta(hours=DATA_TIMEDELTA_HOURS * idx)


def _fmt_dt(dt: cftime.datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def convert_year(
    src_path: Path,
    dst_dir: Path,
    *,
    year: int,
    max_forecast_lead_steps: int,
    n_timesteps_floor: int = 1452,
) -> dict:
    """Convert one v10 source file → per-timestep files in `dst_dir`. Returns manifest entry."""
    logger.info("Converting %s -> %s (group year %d)", src_path, dst_dir, year)
    with h5py.File(src_path, "r") as f:
        _assert_v10_channels(src_path, f)
        n_timesteps = int(f["fields_state"].shape[0])
        if n_timesteps < n_timesteps_floor:
            raise RuntimeError(
                f"{src_path}: only {n_timesteps} timesteps (< floor {n_timesteps_floor})."
            )

        # z0 audit (over time, then mean over space).
        z0 = f["forcing"][:, 2, :, :]  # (T, 64, 128)
        z0_temporal_std_mean = float(np.std(z0, axis=0).mean())

        # Stream timesteps.
        for idx in range(n_timesteps):
            out_path = dst_dir / f"{year}_{idx:04d}.h5"
            _write_per_timestep_h5(
                out_path,
                fields_state=f["fields_state"][idx],
                fields_diagnostic=f["fields_diagnostic"][idx],
                forcing=f["forcing"][idx],
            )
            if (idx + 1) % 100 == 0 or idx == n_timesteps - 1:
                logger.info("  %s year %d: wrote %d/%d", src_path.name, year, idx + 1, n_timesteps)

    last_idx = n_timesteps - 1
    last_train_init_idx = last_idx - 1                              # need t and t+1 for training
    last_val_init_idx = last_idx - max_forecast_lead_steps          # need t..t+K for validation

    if last_val_init_idx < 0:
        raise RuntimeError(
            f"year {year}: max_forecast_lead_steps={max_forecast_lead_steps} exceeds "
            f"available n_timesteps={n_timesteps}"
        )

    syn_start = _synthetic_dt(year, 0)
    syn_last_idx = _synthetic_dt(year, last_idx)
    last_train_init_dt = _synthetic_dt(year, last_train_init_idx)
    train_end_excl = _synthetic_dt(year, last_train_init_idx + 1)
    last_val_init_dt = _synthetic_dt(year, last_val_init_idx)
    val_end_excl = _synthetic_dt(year, last_val_init_idx + 1)

    return {
        "year": year,
        "n_timesteps": n_timesteps,
        # Phase F: stats/climatology readers slice files[:n_timesteps_native].
        # For freshly-converted years (no padding yet), n_native == n_timesteps.
        # F.B2 padding bumps n_timesteps to n_timesteps_padded but leaves
        # n_timesteps_native pinned at the pre-pad value.
        "n_timesteps_native": n_timesteps,
        "synthetic_start_dt": _fmt_dt(syn_start),
        "synthetic_last_idx_dt": _fmt_dt(syn_last_idx),
        "last_train_init_idx": last_train_init_idx,
        "last_train_init_dt": _fmt_dt(last_train_init_dt),
        "train_end_exclusive_dt": _fmt_dt(train_end_excl),
        "last_val_init_idx_for_max_lead_K": last_val_init_idx,
        "last_val_init_dt_for_max_lead_K": _fmt_dt(last_val_init_dt),
        "val_end_exclusive_dt_for_max_lead_K": _fmt_dt(val_end_excl),
        "z0_temporal_std_mean": z0_temporal_std_mean,
        "src_path": str(src_path),
    }


def pad_canonical_year(
    dst_dir: Path,
    *,
    year: int,
    n_native: int,
    n_padded: int,
    donor_year: int,
) -> dict:
    """Copy donor_year idx [0, n_padded - n_native) to year idx [n_native, n_padded).

    Bit-identical copy via shutil.copyfile (h5 file-level copy preserves all keys
    and dtypes; T.6 verifies). Returns pad_source manifest fragment.
    """
    import shutil
    n_pad = n_padded - n_native
    if n_pad <= 0:
        raise ValueError(f"year {year}: n_padded ({n_padded}) <= n_native ({n_native})")

    pad_source: list[dict] = []
    for i in range(n_pad):
        dst_idx = n_native + i
        donor_idx = i
        donor_path = dst_dir / f"{donor_year}_{donor_idx:04d}.h5"
        dst_path = dst_dir / f"{year}_{dst_idx:04d}.h5"
        if not donor_path.is_file():
            raise FileNotFoundError(
                f"pad year {year}: missing donor {donor_path}. "
                f"Convert donor year {donor_year} natively first."
            )
        # Idempotent: overwrite if already padded (e.g., re-running F.B2).
        shutil.copyfile(donor_path, dst_path)
        pad_source.append({
            "dst_idx": dst_idx,
            "src_year": donor_year,
            "src_idx": donor_idx,
        })
        logger.info("  pad: %s (year %d idx %d) <- %s (year %d idx %d)",
                    dst_path.name, year, dst_idx, donor_path.name, donor_year, donor_idx)
    return {
        "n_timesteps_padded": n_padded,
        "is_leap": PAD_TARGETS[year]["is_leap"],
        "pad_source": pad_source,
    }


def apply_pad_pass(dst_dir: Path, manifest_path: Path, *, only_years: list[int] | None = None) -> None:
    """F.B2 entry point: load manifest, pad year 11 + year 12, rewrite manifest.

    Idempotent. If `only_years` is set, restrict to that subset (used by tests).
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Cannot pad without manifest: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    eligible = list(PAD_TARGETS.keys()) if only_years is None else only_years
    by_year = {entry["year"]: entry for entry in manifest["years"]}

    for year in eligible:
        if year not in PAD_TARGETS:
            raise ValueError(f"year {year} not in PAD_TARGETS {list(PAD_TARGETS)}")
        if year not in by_year:
            raise RuntimeError(f"year {year} not in manifest; convert it natively first")
        spec = PAD_TARGETS[year]
        entry = by_year[year]
        if entry["n_timesteps_native"] != spec["n_native"]:
            raise RuntimeError(
                f"year {year}: manifest n_timesteps_native={entry['n_timesteps_native']}, "
                f"expected {spec['n_native']} per PAD_TARGETS"
            )
        donor_year = spec["donor_year"]
        if donor_year not in by_year:
            raise RuntimeError(f"year {year}: donor year {donor_year} missing from manifest")
        if by_year[donor_year]["n_timesteps_native"] < (spec["n_padded"] - spec["n_native"]):
            raise RuntimeError(
                f"year {year}: donor {donor_year} has only "
                f"{by_year[donor_year]['n_timesteps_native']} native frames; "
                f"need at least {spec['n_padded'] - spec['n_native']}"
            )
        pad_meta = pad_canonical_year(
            dst_dir, year=year,
            n_native=spec["n_native"],
            n_padded=spec["n_padded"],
            donor_year=donor_year,
        )
        # n_timesteps reflects the padded total now.
        entry["n_timesteps"] = pad_meta["n_timesteps_padded"]
        entry.update(pad_meta)
        logger.info("Year %d: padded %d -> %d frames (donor %d, %d frames)",
                    year, spec["n_native"], spec["n_padded"], donor_year,
                    spec["n_padded"] - spec["n_native"])

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Rewrote manifest with pad metadata: %s", manifest_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=None,
                        help="v10 packaged root (contains train/, valid/, test/ subdirs). "
                             "Required unless --pad-canonical-years-only is set.")
    parser.add_argument("--dst", required=True, type=Path,
                        help="Flat output dir for group-format per-timestep h5.")
    parser.add_argument("--years", type=int, nargs="+",
                        help="List of v10 years to convert (1:1 to group-calendar years). "
                             "Required unless --pad-canonical-years-only is set.")
    parser.add_argument("--max-forecast-lead-steps", type=int, default=60,
                        help="K used for last_val_init computation (6h units; default 60 = 15d).")
    parser.add_argument("--pad-canonical-years", action="store_true",
                        help="After native conversion of --years, also pad year 11 to 1460 "
                             "frames (donor year 12) and year 12 to 1464 (donor year 13). "
                             "Manifest is updated. Phase F.B2 entry point.")
    parser.add_argument("--pad-canonical-years-only", action="store_true",
                        help="Skip native conversion; only run F.B2 padding pass against an "
                             "existing manifest. Use after F.C stats/clim are written.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.pad_canonical_years_only and args.pad_canonical_years:
        raise SystemExit("Pass either --pad-canonical-years OR --pad-canonical-years-only, not both")
    if not args.pad_canonical_years_only:
        if not args.years:
            raise SystemExit("--years is required (unless --pad-canonical-years-only)")
        if args.src is None:
            raise SystemExit("--src is required (unless --pad-canonical-years-only)")

    args.dst.mkdir(parents=True, exist_ok=True)
    _try_lfs_setstripe(args.dst)
    manifest_path = args.dst / "_v10_calendar_manifest.json"

    if args.pad_canonical_years_only:
        apply_pad_pass(args.dst, manifest_path)
        return 0

    manifest_years: list[dict] = []
    for year in args.years:
        src_path = _find_v10_source(args.src, year)
        entry = convert_year(
            src_path, args.dst,
            year=year,
            max_forecast_lead_steps=args.max_forecast_lead_steps,
        )
        manifest_years.append(entry)

    manifest = {
        "calendar": "proleptic_gregorian",
        "has_year_zero": True,
        "data_timedelta_hours": DATA_TIMEDELTA_HOURS,
        "max_forecast_lead_steps": args.max_forecast_lead_steps,
        "src_root": str(args.src),
        "dst": str(args.dst),
        "expected_state_channels": EXPECTED_STATE_CHANNELS,
        "expected_diagnostic_channels": EXPECTED_DIAGNOSTIC_CHANNELS,
        "expected_forcing_channels": EXPECTED_FORCING_CHANNELS,
        "sigma_levels_pl_native": list(SIGMA_LEVELS),
        "zg_levels_pa": list(PLEVS_PA),
        "years": manifest_years,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest: %s", manifest_path)

    if args.pad_canonical_years:
        apply_pad_pass(args.dst, manifest_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
