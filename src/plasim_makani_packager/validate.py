#!/usr/bin/env python3
"""validate.py — Phase 4a (structural) + 4b (Makani smoke) validation.

Phase 4a runs in this script directly: inspects every HDF5 under
{output-root}/{split}/MOST.*.h5, checks shapes, finiteness, timestamp
monotonicity (within + across files), dim scales, channel lists, and
required file attributes. Stats files are also checked for shape /
nonzero stds.

Phase 4b is the full YParams → PlasimPreprocessor → PlasimSingleStepWrapper
smoke test. It depends on makani + torch and is implemented as a pytest
module under tests/plasim_makani_packager/test_multifile_loader_smoke.py.
`--mode makani_smoke` and `--mode full` invoke it via pytest.

CLI
---
validate.py --output-root {root} --mode {structural,makani_smoke,full} [-v]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

from plasim_makani_packager.channels import (
    DIAGNOSTIC_CHANNELS,
    FORCING_CHANNELS,
    STATE_CHANNELS,
    TARGET_CHANNELS,
)
from plasim_makani_packager.packager import STEP_SECONDS
from plasim_makani_packager.stats import (
    MIN_STD_EPSILON,
    STATIC_FORCING_NAMES,
)

logger = logging.getLogger("plasim_makani_packager.validate")

REQUIRED_FILE_ATTRS: tuple[str, ...] = (
    "plasim_time_units",
    "plasim_calendar",
    "rsdt_method",
    "sst_land_fill_K",
)
REQUIRED_DATASETS: tuple[str, ...] = (
    "fields_state",
    "fields_diagnostic",
    "forcing",
    "timestamp",
    "time_plasim",
    "channel_state",
    "channel_diagnostic",
    "channel_forcing",
    "lat",
    "lon",
)
SPLITS: tuple[str, ...] = ("train", "valid", "test")


class ValidationError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Per-file structural checks
# ---------------------------------------------------------------------------
def _bytes_to_str(arr: np.ndarray) -> list[str]:
    return [x.decode("ascii") if isinstance(x, bytes) else str(x) for x in arr]


def _check_dim_scales(dset: h5py.Dataset, scales: tuple[h5py.Dataset, ...]) -> None:
    if dset.ndim != 4:
        raise ValidationError(f"{dset.name}: expected 4D, got {dset.ndim}D")
    for axis, expected in enumerate(scales):
        attached = list(dset.dims[axis].items())
        if not attached:
            raise ValidationError(
                f"{dset.name}: no dim scale attached at axis {axis} "
                f"(expected '{expected.name}')"
            )
        if expected.name not in {d[1].name for d in attached}:
            raise ValidationError(
                f"{dset.name}: dim scale '{expected.name}' not attached at axis "
                f"{axis} (found {[d[1].name for d in attached]})"
            )


def _validate_file(path: Path, year: int) -> int:
    """Return T (number of time steps) of the validated file."""
    with h5py.File(path, "r") as f:
        for name in REQUIRED_DATASETS:
            if name not in f:
                raise ValidationError(f"{path.name}: missing dataset '{name}'")
        for attr in REQUIRED_FILE_ATTRS:
            if attr not in f.attrs:
                raise ValidationError(
                    f"{path.name}: missing file attr '{attr}'"
                )

        state = f["fields_state"]
        diag = f["fields_diagnostic"]
        forcing = f["forcing"]
        ts = f["timestamp"]
        tp = f["time_plasim"]

        T, C_s, H, W = state.shape
        if C_s != 52 or state.dtype != np.float32:
            raise ValidationError(
                f"{path.name}: fields_state shape/dtype mismatch "
                f"{state.shape} {state.dtype}"
            )
        if diag.shape != (T, 1, H, W) or diag.dtype != np.float32:
            raise ValidationError(
                f"{path.name}: fields_diagnostic shape/dtype mismatch "
                f"{diag.shape} {diag.dtype}"
            )
        if forcing.shape != (T, 6, H, W) or forcing.dtype != np.float32:
            raise ValidationError(
                f"{path.name}: forcing shape/dtype mismatch "
                f"{forcing.shape} {forcing.dtype}"
            )
        if (H, W) != (64, 128):
            raise ValidationError(
                f"{path.name}: expected 64x128 grid, got {H}x{W}"
            )
        if ts.shape != (T,) or ts.dtype != np.int64:
            raise ValidationError(
                f"{path.name}: timestamp shape/dtype {ts.shape} {ts.dtype}"
            )
        if tp.shape != (T,) or tp.dtype != np.float64:
            raise ValidationError(
                f"{path.name}: time_plasim shape/dtype {tp.shape} {tp.dtype}"
            )

        cs = _bytes_to_str(f["channel_state"][...])
        cd = _bytes_to_str(f["channel_diagnostic"][...])
        cf = _bytes_to_str(f["channel_forcing"][...])
        if cs != STATE_CHANNELS:
            raise ValidationError(
                f"{path.name}: channel_state != expected master list"
            )
        if cd != DIAGNOSTIC_CHANNELS:
            raise ValidationError(
                f"{path.name}: channel_diagnostic != expected master list"
            )
        if cf != FORCING_CHANNELS:
            raise ValidationError(
                f"{path.name}: channel_forcing != expected master list"
            )

        # Dim scales on all 4D datasets
        ts_scale = f["timestamp"]
        state_scale = f["channel_state"]
        diag_scale = f["channel_diagnostic"]
        force_scale = f["channel_forcing"]
        lat_scale = f["lat"]
        lon_scale = f["lon"]
        _check_dim_scales(
            state, (ts_scale, state_scale, lat_scale, lon_scale)
        )
        _check_dim_scales(
            diag, (ts_scale, diag_scale, lat_scale, lon_scale)
        )
        _check_dim_scales(
            forcing, (ts_scale, force_scale, lat_scale, lon_scale)
        )

        # Finite content (stream in small chunks to avoid loading full tensor)
        for dset, label in ((state, "fields_state"), (diag, "fields_diagnostic"), (forcing, "forcing")):
            arr = dset[...]
            if not np.isfinite(arr).all():
                bad = np.argwhere(~np.isfinite(arr))[:5]
                raise ValidationError(
                    f"{path.name}: non-finite values in {label}, first indices: "
                    f"{bad.tolist()}"
                )

        # Timestamp within-file: monotonic, diff == STEP_SECONDS
        ts_vals = ts[...]
        dts = np.diff(ts_vals)
        if not np.all(dts == STEP_SECONDS):
            uniq = np.unique(dts).tolist()
            raise ValidationError(
                f"{path.name}: timestamp diffs not all {STEP_SECONDS}: {uniq}"
            )

        # PlaSim time: strictly monotonic
        tp_vals = tp[...]
        if not np.all(np.diff(tp_vals) > 0):
            raise ValidationError(
                f"{path.name}: time_plasim is not strictly increasing"
            )

        attrs = {k: f.attrs[k] for k in REQUIRED_FILE_ATTRS}
        rsdt_method = attrs["rsdt_method"]
        if isinstance(rsdt_method, bytes):
            rsdt_method = rsdt_method.decode()
        if rsdt_method != "astronomical":
            raise ValidationError(
                f"{path.name}: rsdt_method='{rsdt_method}', expected 'astronomical'"
            )
        sst_fill = float(attrs["sst_land_fill_K"])
        if not (200.0 < sst_fill < 320.0):
            raise ValidationError(
                f"{path.name}: sst_land_fill_K={sst_fill} out of plausible range"
            )

        return int(T)


# ---------------------------------------------------------------------------
# Cross-file monotonicity
# ---------------------------------------------------------------------------
def _assert_cross_file_monotonic(files_in_order: list[Path]) -> None:
    last = None
    for path in files_in_order:
        with h5py.File(path, "r") as f:
            first = int(f["timestamp"][0])
            final = int(f["timestamp"][-1])
        if last is not None and first <= last:
            raise ValidationError(
                f"cross-file monotonicity broken at {path.name}: first={first} "
                f"<= previous_last={last}"
            )
        last = final


def _files_in_year_order(output_root: Path) -> list[Path]:
    pairs: list[tuple[int, Path]] = []
    for split in SPLITS:
        for p in (output_root / split).glob("MOST.*.h5"):
            year = int(p.stem.split(".")[1])
            pairs.append((year, p))
    pairs.sort(key=lambda p: p[0])
    return [p for _, p in pairs]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def _validate_stats(output_root: Path, epsilon: float) -> None:
    stats = output_root / "stats"
    expected = {
        "global_means.npy": (1, 53, 1, 1),
        "global_stds.npy": (1, 53, 1, 1),
        "time_means.npy": (1, 53, 64, 128),
        "forcing_global_means.npy": (1, 6, 1, 1),
        "forcing_global_stds.npy": (1, 6, 1, 1),
        "forcing_time_means.npy": (1, 6, 64, 128),
    }
    for name, shape in expected.items():
        path = stats / name
        if not path.exists():
            raise ValidationError(f"missing stats file: {path}")
        arr = np.load(path)
        if arr.shape != shape:
            raise ValidationError(
                f"{name}: shape {arr.shape}, expected {shape}"
            )
        if arr.dtype != np.float32:
            raise ValidationError(
                f"{name}: dtype {arr.dtype}, expected float32"
            )
        if not np.isfinite(arr).all():
            raise ValidationError(f"{name}: non-finite values present")

    std_tgt = np.load(stats / "global_stds.npy").ravel()
    for i, name in enumerate(TARGET_CHANNELS):
        if std_tgt[i] < epsilon:
            raise ValidationError(
                f"global_stds[{name}]={std_tgt[i]:.3e} < {epsilon:.1e}"
            )
    std_frc = np.load(stats / "forcing_global_stds.npy").ravel()
    for i, name in enumerate(FORCING_CHANNELS):
        if std_frc[i] < epsilon:
            raise ValidationError(
                f"forcing_global_stds[{name}]={std_frc[i]:.3e} < {epsilon:.1e}"
            )

    # Static forcing channels: time_means across space should match the
    # per-cell value from any single training file (spatial variability is
    # fine; temporal variability must be zero).
    forcing_time_means = np.load(stats / "forcing_time_means.npy")
    for i, name in enumerate(FORCING_CHANNELS):
        if name not in STATIC_FORCING_NAMES:
            continue
        # Confirm: choose any training file and check the first time step
        # matches the time-averaged value for this channel.
        train_files = sorted((output_root / "train").glob("MOST.*.h5"))
        if not train_files:
            continue
        with h5py.File(train_files[0], "r") as f:
            sample = f["forcing"][0, i]
        if not np.allclose(
            forcing_time_means[0, i], sample, atol=1e-5, rtol=0.0
        ):
            raise ValidationError(
                f"static forcing '{name}': time_means vs first-sample diff > 1e-5"
            )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def run_structural(output_root: Path, epsilon: float) -> None:
    ordered = _files_in_year_order(output_root)
    if not ordered:
        raise ValidationError(f"no MOST.*.h5 files under {output_root}/{{train,valid,test}}")

    for path in ordered:
        year = int(path.stem.split(".")[1])
        T = _validate_file(path, year)
        logger.info("  ok  %s  (T=%d)", path, T)

    _assert_cross_file_monotonic(ordered)
    logger.info("cross-file timestamps strictly increasing ✓")

    _validate_stats(output_root, epsilon)
    logger.info("stats ok ✓")


def run_makani_smoke() -> int:
    """Launch pytest on the smoke-test module."""
    import pytest  # imported lazily: structural mode doesn't need pytest / torch

    repo_root = Path(__file__).resolve().parents[2]
    test_path = repo_root / "tests" / "plasim_makani_packager" / "test_multifile_loader_smoke.py"
    if not test_path.exists():
        raise ValidationError(f"smoke test module not found: {test_path}")
    return pytest.main(["-x", "-q", str(test_path)])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument(
        "--mode",
        choices=("structural", "makani_smoke", "full"),
        default="structural",
    )
    p.add_argument("--epsilon", type=float, default=MIN_STD_EPSILON)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.mode in ("structural", "full"):
        try:
            run_structural(args.output_root, args.epsilon)
        except ValidationError as e:
            sys.exit(f"structural validation failed: {e}")
    if args.mode in ("makani_smoke", "full"):
        rc = run_makani_smoke()
        if rc != 0:
            sys.exit(f"pytest smoke returned {rc}")


if __name__ == "__main__":
    main()
