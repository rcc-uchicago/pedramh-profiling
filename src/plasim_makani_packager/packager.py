#!/usr/bin/env python3
"""packager.py — PlaSim MOST + adaptor boundary → Makani three-dataset HDF5.

Inputs
------
{postproc-root}/sim{NN}/MOST.{YYYY:04d}.nc
    Produced by src/plasim_postprocessor/. 6-hourly, 64×128 (Gauss-Legendre).
    All state + diagnostic vars + static/dynamic masks (lsm, sg, z0, sic).

{boundary-root}/sim{NN}/boundary.{YYYY:04d}.nc
    Produced by src/emulator_adaptor/. Carries the emulator-contract
    prescribed fields: sst (land=NaN), rsdt (astronomical), sic (clipped).

Outputs
-------
{output-root}/{split}/MOST.{YYYY:04d}.h5 — one per (sim, year). Contents:
    /fields_state      (T, 52, 64, 128) float32
    /fields_diagnostic (T,  1, 64, 128) float32
    /forcing           (T,  6, 64, 128) float32
    /timestamp         (T,) int64    -- dataset-globally monotonic seconds
    /time_plasim       (T,) float64  -- raw days-since from source NetCDF
    /channel_state       (52,) ASCII
    /channel_diagnostic  (1,)  ASCII
    /channel_forcing     (6,)  ASCII
    /lat               (64,)  float64
    /lon               (128,) float64

{output-root}/validation/sic_clip_report_{YYYY:04d}.json — quantify-only.

Splits (locked by plan v9): train 3–100 · valid 101–120 · test 121–128.
Years 1 and 2 are warmup and skipped.

CLI
---
packager.py --postproc-root ... --boundary-root ... --output-root ...
            --sims 52 --train-years 3 100 --valid-years 101 120
            --test-years 121 128 --sst-land-fill-k 271.35
            [--task-index N] [--count-tasks] [--overwrite] [--dry-run] [-v]

With --count-tasks only, the script prints len(tasks) and exits.
Enumeration is deterministic: sorted(sims) × sorted(all years across splits).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

from plasim_makani_packager.channels import (
    DIAGNOSTIC_CHANNELS,
    FORCING_CHANNELS,
    STATE_CHANNELS,
)

logger = logging.getLogger("plasim_makani_packager")

# Locked constants ----------------------------------------------------------
STEP_SECONDS: int = 21600  # 6 hours
DEFAULT_SST_LAND_FILL_K: float = 271.35
COORD_TOLERANCE: float = 1e-6
SIC_CLIP_HARD_FAIL_TOL: float = 1e-6  # adaptor must clip only, nothing else

SPLITS: tuple[str, ...] = ("train", "valid", "test")
WARMUP_YEARS: frozenset[int] = frozenset({1, 2})


# ---------------------------------------------------------------------------
# Stacking
# ---------------------------------------------------------------------------
def _stack_fields_state(ds: xr.Dataset) -> np.ndarray:
    """Stack 52 state channels into (T, 52, H, W) float32.

    Order: [pl, tas, ta1..ta10, ua1..ua10, va1..va10, hus1..hus10, zg1..zg10]
    Sigma ordering: lev[0] (TOA) → ta1, lev[9] (surface) → ta10.
    """
    T = ds.sizes["time"]
    H = ds.sizes["lat"]
    W = ds.sizes["lon"]
    out = np.empty((T, 52, H, W), dtype=np.float32)
    out[:, 0] = ds["pl"].values
    out[:, 1] = ds["tas"].values
    col = 2
    for var in ("ta", "ua", "va", "hus", "zg"):
        arr = ds[var].values  # (T, 10, H, W)
        if arr.shape != (T, 10, H, W):
            raise RuntimeError(f"{var} has shape {arr.shape}, expected {(T, 10, H, W)}")
        out[:, col : col + 10] = arr
        col += 10
    assert col == 52
    return out


def _stack_fields_diagnostic(ds: xr.Dataset) -> np.ndarray:
    T = ds.sizes["time"]
    H = ds.sizes["lat"]
    W = ds.sizes["lon"]
    out = np.empty((T, 1, H, W), dtype=np.float32)
    out[:, 0] = ds["pr_6h"].values
    return out


def _stack_forcing(
    most_ds: xr.Dataset,
    boundary_ds: xr.Dataset,
    sst_land_fill_k: float,
) -> tuple[np.ndarray, float]:
    """Stack 6 forcing channels into (T, 6, H, W).

    lsm, sg     : from MOST. Globally static (same value every time step, every year).
    z0          : from MOST. Time-varying prescribed forcing — land-static,
                  ocean-dynamic (Charnock/sea-state roughness, 1.5e-5..1e-3 m).
    sst         : from boundary adaptor, NaN-over-land filled with sst_land_fill_k.
    rsdt        : from boundary adaptor.
    sic         : from boundary adaptor (already clipped to [0, 1]).

    Returns (forcing, sst_land_fill_fraction).
    """
    T = most_ds.sizes["time"]
    H = most_ds.sizes["lat"]
    W = most_ds.sizes["lon"]
    out = np.empty((T, 6, H, W), dtype=np.float32)
    out[:, 0] = most_ds["lsm"].values
    out[:, 1] = most_ds["sg"].values
    out[:, 2] = most_ds["z0"].values

    sst = boundary_ds["sst"].values.astype(np.float32, copy=False)
    nan_mask = np.isnan(sst)
    sst_filled = np.where(nan_mask, np.float32(sst_land_fill_k), sst)
    sst_land_fill_fraction = float(nan_mask.sum()) / float(nan_mask.size)
    out[:, 3] = sst_filled
    out[:, 4] = boundary_ds["rsdt"].values
    out[:, 5] = boundary_ds["sic"].values
    return out, sst_land_fill_fraction


# ---------------------------------------------------------------------------
# Validation (pre-write)
# ---------------------------------------------------------------------------
def _cross_validate_coords(most_ds: xr.Dataset, boundary_ds: xr.Dataset) -> None:
    """lat / lon must be byte-identical; time must tick the same values."""
    for name in ("lat", "lon"):
        m = most_ds[name].values
        b = boundary_ds[name].values
        if m.shape != b.shape:
            raise RuntimeError(
                f"coord '{name}' shape mismatch: MOST={m.shape} boundary={b.shape}"
            )
        if not np.array_equal(m, b):
            raise RuntimeError(f"coord '{name}' values differ between MOST and boundary")
    m_t = most_ds["time"].values
    b_t = boundary_ds["time"].values
    if m_t.shape != b_t.shape:
        raise RuntimeError(
            f"coord 'time' shape mismatch: MOST={m_t.shape} boundary={b_t.shape}"
        )
    if not np.allclose(m_t, b_t, atol=COORD_TOLERANCE, rtol=0.0):
        raise RuntimeError("coord 'time' values differ between MOST and boundary")


def _validate_sic_clipping(
    most_ds: xr.Dataset, boundary_ds: xr.Dataset
) -> dict:
    """Plan v9 fix: split into hard-fail (adaptor must clip only, nothing else)
    and quantify (raw MOST-vs-adaptor stats, report only)."""
    most_sic = most_ds["sic"].values.astype(np.float32, copy=False)
    adaptor_sic = boundary_ds["sic"].values.astype(np.float32, copy=False)

    if most_sic.shape != adaptor_sic.shape:
        raise RuntimeError(
            f"sic shape mismatch: MOST={most_sic.shape} adaptor={adaptor_sic.shape}"
        )

    most_nan = np.isnan(most_sic)
    adaptor_nan = np.isnan(adaptor_sic)
    if not np.array_equal(most_nan, adaptor_nan):
        raise RuntimeError(
            "NaN-mask parity broken between MOST.sic and adaptor.sic "
            f"(MOST NaN={most_nan.sum()}, adaptor NaN={adaptor_nan.sum()})"
        )

    # Hard gate: adaptor must equal np.clip(MOST.sic, 0, 1) cell-wise.
    clipped = np.where(most_nan, np.nan, np.clip(most_sic, 0.0, 1.0)).astype(
        np.float32
    )
    diff_clip = np.where(
        most_nan, 0.0, np.abs(adaptor_sic - clipped)
    )
    max_clip_diff = float(np.nanmax(diff_clip))
    if max_clip_diff > SIC_CLIP_HARD_FAIL_TOL:
        raise RuntimeError(
            f"adaptor.sic differs from np.clip(MOST.sic, 0, 1) by "
            f"max_abs_diff={max_clip_diff:.3e} > tol {SIC_CLIP_HARD_FAIL_TOL:.1e}. "
            f"Adaptor is expected to clip and nothing else."
        )

    # Quantify only (report, no hard-fail). Legit > 1e-3 whenever PlaSim
    # produced out-of-range sic that clip corrected.
    raw_diff = np.where(most_nan, 0.0, np.abs(most_sic - adaptor_sic))
    out_of_range = np.where(most_nan, False, (most_sic < 0.0) | (most_sic > 1.0))
    report = {
        "max_abs_diff": float(np.nanmax(raw_diff)),
        "mean_abs_diff": float(np.nanmean(raw_diff)),
        "fraction_cells_changed_by_clip": float(out_of_range.mean()),
        "adaptor_vs_clip_max_abs_diff": max_clip_diff,
        "n_nan": int(most_nan.sum()),
        "n_cells": int(most_sic.size),
    }
    return report


def _assert_rsdt_method(boundary_ds: xr.Dataset) -> None:
    method = boundary_ds.attrs.get("rsdt_method", "")
    if method != "astronomical":
        raise RuntimeError(
            f"boundary.attrs['rsdt_method']='{method}', expected 'astronomical'"
        )


# ---------------------------------------------------------------------------
# Timestamp synthesis (plan v9: dataset-globally monotonic seconds)
# ---------------------------------------------------------------------------
_T_CACHE: dict[tuple[str, int, int], int] = {}


def _get_T_for_year(postproc_root: Path, sim: int, year: int) -> int:
    # Key includes postproc_root to avoid cross-test contamination when
    # tests invoke the packager against different synthetic roots.
    key = (str(postproc_root.resolve()), sim, year)
    if key not in _T_CACHE:
        most_path = postproc_root / f"sim{sim}" / f"MOST.{year:04d}.nc"
        if not most_path.exists():
            raise RuntimeError(
                f"cannot read T for sim{sim} year {year:04d}: missing {most_path}"
            )
        with xr.open_dataset(most_path, decode_times=False) as ds:
            _T_CACHE[key] = int(ds.sizes["time"])
    return _T_CACHE[key]


def _compute_split_offset(
    year: int,
    split: str,
    *,
    sim: int,
    postproc_root: Path,
    train_years: tuple[int, int],
    valid_years: tuple[int, int],
    test_years: tuple[int, int],
) -> int:
    """Seconds offset for this file's first timestamp, chosen so concatenating
    /timestamp across files in the same split (in year order) produces a
    strictly increasing series with uniform diff == STEP_SECONDS.

    Makani's MultifilesDataset enforces uniform dT across file boundaries
    (data_loader_multifiles.py:216-221). Offsetting per-year by
    sum_{y < year, y in split} T_y * STEP_SECONDS is the simplest scheme
    that satisfies it. Per-year T is cached in _T_CACHE so repeated calls
    across (year_i) amortize down to one read per year.
    """
    ranges = {"train": train_years, "valid": valid_years, "test": test_years}
    if split not in ranges:
        return 0
    lo, _hi = ranges[split]
    prior_lo = max(lo, 3)  # warmup years 1-2 never produced outputs

    offset_steps = 0
    for y_prior in range(prior_lo, year):
        offset_steps += _get_T_for_year(postproc_root, sim, y_prior)
    return int(offset_steps) * STEP_SECONDS


def _synthetic_timestamps(offset_seconds: int, T: int) -> np.ndarray:
    """(T,) int64 seconds starting at `offset_seconds`, step STEP_SECONDS."""
    return np.int64(offset_seconds) + np.arange(T, dtype=np.int64) * np.int64(
        STEP_SECONDS
    )


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------
def _resolve_git_sha(script_dir: Path) -> str:
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(script_dir), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return sha or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _write_h5(
    output_path: Path,
    *,
    fields_state: np.ndarray,
    fields_diagnostic: np.ndarray,
    forcing: np.ndarray,
    timestamp: np.ndarray,
    time_plasim: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    file_attrs: dict,
) -> None:
    tmp_path = output_path.with_suffix(".h5.tmp")

    _, _, H, W = fields_state.shape
    with h5py.File(tmp_path, "w") as f:
        # Channel / coord labels
        state_ds = f.create_dataset(
            "channel_state",
            data=np.array(STATE_CHANNELS, dtype=h5py.string_dtype("ascii")),
        )
        diag_ds = f.create_dataset(
            "channel_diagnostic",
            data=np.array(DIAGNOSTIC_CHANNELS, dtype=h5py.string_dtype("ascii")),
        )
        force_ds = f.create_dataset(
            "channel_forcing",
            data=np.array(FORCING_CHANNELS, dtype=h5py.string_dtype("ascii")),
        )
        lat_ds = f.create_dataset("lat", data=lat.astype(np.float64))
        lon_ds = f.create_dataset("lon", data=lon.astype(np.float64))
        ts_ds = f.create_dataset("timestamp", data=timestamp.astype(np.int64))
        tp_ds = f.create_dataset("time_plasim", data=time_plasim.astype(np.float64))

        # Dimension scales on the label axes
        for name, obj in (
            ("timestamp", ts_ds),
            ("channel_state", state_ds),
            ("channel_diagnostic", diag_ds),
            ("channel_forcing", force_ds),
            ("lat", lat_ds),
            ("lon", lon_ds),
        ):
            obj.make_scale(name)

        # 4D fields with chunking matched to (1, C, H, W).
        state_h5 = f.create_dataset(
            "fields_state",
            data=fields_state,
            dtype="float32",
            chunks=(1, 52, H, W),
        )
        diag_h5 = f.create_dataset(
            "fields_diagnostic",
            data=fields_diagnostic,
            dtype="float32",
            chunks=(1, 1, H, W),
        )
        force_h5 = f.create_dataset(
            "forcing",
            data=forcing,
            dtype="float32",
            chunks=(1, 6, H, W),
        )

        for payload, ch_scale in (
            (state_h5, state_ds),
            (diag_h5, diag_ds),
            (force_h5, force_ds),
        ):
            payload.dims[0].attach_scale(ts_ds)
            payload.dims[1].attach_scale(ch_scale)
            payload.dims[2].attach_scale(lat_ds)
            payload.dims[3].attach_scale(lon_ds)

        for key, val in file_attrs.items():
            f.attrs[key] = val

    tmp_path.replace(output_path)


# ---------------------------------------------------------------------------
# Split resolution + task enumeration
# ---------------------------------------------------------------------------
def resolve_split(
    year: int,
    *,
    train_years: tuple[int, int],
    valid_years: tuple[int, int],
    test_years: tuple[int, int],
) -> str | None:
    """Return 'train' | 'valid' | 'test' | None (skip)."""
    if year in WARMUP_YEARS:
        return None
    if train_years[0] <= year <= train_years[1]:
        return "train"
    if valid_years[0] <= year <= valid_years[1]:
        return "valid"
    if test_years[0] <= year <= test_years[1]:
        return "test"
    return None


def enumerate_tasks(
    sims: list[int],
    train_years: tuple[int, int],
    valid_years: tuple[int, int],
    test_years: tuple[int, int],
) -> list[tuple[int, int]]:
    years: set[int] = set()
    for lo, hi in (train_years, valid_years, test_years):
        years.update(range(lo, hi + 1))
    years -= WARMUP_YEARS
    return [(sim, y) for sim in sorted(sims) for y in sorted(years)]


# ---------------------------------------------------------------------------
# Per-task processing
# ---------------------------------------------------------------------------
def _validation_dir(output_root: Path) -> Path:
    return output_root / "validation"


def process_one(sim: int, year: int, opts: argparse.Namespace) -> None:
    most_path = Path(opts.postproc_root) / f"sim{sim}" / f"MOST.{year:04d}.nc"
    boundary_path = (
        Path(opts.boundary_root) / f"sim{sim}" / f"boundary.{year:04d}.nc"
    )

    split = resolve_split(
        year,
        train_years=tuple(opts.train_years),
        valid_years=tuple(opts.valid_years),
        test_years=tuple(opts.test_years),
    )
    if split is None:
        logger.info("[sim%s/%04d] skipping (not in any split)", sim, year)
        return

    output_path = Path(opts.output_root) / split / f"MOST.{year:04d}.h5"

    if not most_path.exists():
        raise RuntimeError(f"missing MOST input: {most_path}")
    if not boundary_path.exists():
        raise RuntimeError(f"missing boundary input: {boundary_path}")

    if output_path.exists() and not opts.overwrite:
        logger.info(
            "[sim%s/%04d] skipping %s (exists; pass --overwrite to force)",
            sim,
            year,
            output_path,
        )
        return

    if opts.dry_run:
        logger.info("[sim%s/%04d] dry-run: would produce %s", sim, year, output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _validation_dir(Path(opts.output_root)).mkdir(parents=True, exist_ok=True)

    logger.info(
        "[sim%s/%04d] reading %s + %s", sim, year, most_path, boundary_path
    )
    with xr.open_dataset(most_path, decode_times=False) as most_ds, xr.open_dataset(
        boundary_path, decode_times=False
    ) as boundary_ds:
        _cross_validate_coords(most_ds, boundary_ds)
        _assert_rsdt_method(boundary_ds)

        sic_report = _validate_sic_clipping(most_ds, boundary_ds)
        sic_report_path = (
            _validation_dir(Path(opts.output_root))
            / f"sic_clip_report_{year:04d}.json"
        )
        with sic_report_path.open("w") as fh:
            json.dump(
                {"sim": sim, "year": year, **sic_report},
                fh,
                indent=2,
                sort_keys=True,
            )

        fields_state = _stack_fields_state(most_ds)
        fields_diagnostic = _stack_fields_diagnostic(most_ds)
        forcing, sst_land_fill_fraction = _stack_forcing(
            most_ds, boundary_ds, opts.sst_land_fill_k
        )

        T = most_ds.sizes["time"]
        offset_seconds = _compute_split_offset(
            year,
            split,
            sim=sim,
            postproc_root=Path(opts.postproc_root),
            train_years=tuple(opts.train_years),
            valid_years=tuple(opts.valid_years),
            test_years=tuple(opts.test_years),
        )
        timestamp = _synthetic_timestamps(offset_seconds, T)
        time_plasim = most_ds["time"].values.astype(np.float64)

        plasim_time_units = str(most_ds["time"].attrs.get("units", ""))
        plasim_calendar = str(most_ds["time"].attrs.get("calendar", ""))

        file_attrs = {
            "rsdt_method": "astronomical",
            "source_postproc": str(most_path.resolve()),
            "source_boundary": str(boundary_path.resolve()),
            "packager_git_sha": _resolve_git_sha(Path(__file__).resolve().parent),
            "sst_land_fill_K": float(opts.sst_land_fill_k),
            "sst_land_fill_fraction": sst_land_fill_fraction,
            "plasim_time_units": plasim_time_units,
            "plasim_calendar": plasim_calendar,
            "split": split,
            "sim": sim,
            "year": year,
        }

        _write_h5(
            output_path,
            fields_state=fields_state,
            fields_diagnostic=fields_diagnostic,
            forcing=forcing,
            timestamp=timestamp,
            time_plasim=time_plasim,
            lat=most_ds["lat"].values,
            lon=most_ds["lon"].values,
            file_attrs=file_attrs,
        )

    logger.info(
        "[sim%s/%04d] wrote %s  (split=%s, T=%d, sst_land_fill_fraction=%.4f)",
        sim,
        year,
        output_path,
        split,
        T,
        sst_land_fill_fraction,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pack PlaSim postprocess + adaptor boundary into a "
        "Makani three-dataset HDF5 layout (per sim, per year).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--sims", required=True, type=int, nargs="+")
    p.add_argument(
        "--train-years", type=int, nargs=2, default=[3, 100], metavar=("START", "END")
    )
    p.add_argument(
        "--valid-years",
        type=int,
        nargs=2,
        default=[101, 120],
        metavar=("START", "END"),
    )
    p.add_argument(
        "--test-years",
        type=int,
        nargs=2,
        default=[121, 128],
        metavar=("START", "END"),
    )
    p.add_argument(
        "--postproc-root",
        type=Path,
        default=None,
        help="Root containing sim{NN}/MOST.{YYYY}.nc. Not required with --count-tasks.",
    )
    p.add_argument(
        "--boundary-root",
        type=Path,
        default=None,
        help="Root containing sim{NN}/boundary.{YYYY}.nc. Not required with --count-tasks.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root ({split}/MOST.{YYYY}.h5). Not required with --count-tasks.",
    )
    p.add_argument(
        "--sst-land-fill-k",
        type=float,
        default=DEFAULT_SST_LAND_FILL_K,
        help="Scalar K used to fill NaN (land) in adaptor sst.",
    )
    p.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="0-based index into the (sim, year) task list (for SLURM arrays).",
    )
    p.add_argument(
        "--year-slice",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="Restrict processing to (sim, year) pairs with year in [START, END] "
        "(inclusive). Classification into train/valid/test still uses the full "
        "--{train,valid,test}-years ranges, so the timestamp offset scheme is "
        "unaffected. Use to shard a single logical run across parallel workers.",
    )
    p.add_argument("--count-tasks", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    tasks = enumerate_tasks(
        args.sims,
        tuple(args.train_years),
        tuple(args.valid_years),
        tuple(args.test_years),
    )

    if args.count_tasks:
        print(len(tasks))
        return

    missing = [
        flag
        for flag, val in (
            ("--postproc-root", args.postproc_root),
            ("--boundary-root", args.boundary_root),
            ("--output-root", args.output_root),
        )
        if val is None
    ]
    if missing:
        sys.exit(
            f"error: {', '.join(missing)} required unless --count-tasks is set"
        )

    if args.year_slice is not None:
        ys_lo, ys_hi = args.year_slice
        tasks = [(s, y) for (s, y) in tasks if ys_lo <= y <= ys_hi]

    if args.task_index is not None:
        if not (0 <= args.task_index < len(tasks)):
            sys.exit(
                f"error: --task-index {args.task_index} out of range "
                f"[0, {len(tasks)})"
            )
        tasks = [tasks[args.task_index]]

    for sim, year in tasks:
        process_one(sim, year, args)


if __name__ == "__main__":
    main()
