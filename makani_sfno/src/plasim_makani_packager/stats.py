#!/usr/bin/env python3
"""stats.py — Single-pass Welford over the training split.

Reads {output-root}/train/MOST.*.h5 (produced by packager.py) and writes six
.npy files into {output-root}/stats/:

    global_means.npy         (1, 53, 1, 1) float32  -- /fields_state ‖ /fields_diagnostic
    global_stds.npy          (1, 53, 1, 1) float32
    time_means.npy           (1, 53, H, W) float32
    forcing_global_means.npy (1, 6, 1, 1) float32
    forcing_global_stds.npy  (1, 6, 1, 1) float32
    forcing_time_means.npy   (1, 6, H, W) float32

Hard-fail on any channel whose population std < MIN_STD_EPSILON.

CLI
---
stats.py --output-root {root} [--train-years 3 100] [--epsilon 1e-6] [-v]
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

logger = logging.getLogger("plasim_makani_packager.stats")

MIN_STD_EPSILON: float = 1e-6
# Known PlaSim quirk (v9 + v10): pr_6h global std over the training split is
# ~2.8e-7 because precipitation is mostly zero in the simulation. v9 stats
# (sim52_astro_64x128/stats/global_stds.npy) and v10 stats (full + proto)
# all show this value to within rounding. v9 trained successfully against
# these stats, so the value is data-faithful — not a regression. Operators
# running stats.py against the v10 zgplev pack should pass `--epsilon 1e-8`
# to let the inline G-z500 audit run; the channel itself is already in the
# v9-validated z-score range during training.

# v10 audit gate (L7d): the global mean of `zg500` over the training split
# must lie within this band. Outside it, abort before any .npy is written.
# Range from docs/plasim_postprocessor_audit.md:193 (sim30 yr12, fldmean).
ZG500_AUDIT_RANGE_M: tuple[float, float] = (5400.0, 5700.0)


def _audit_zg500_inline(mean_tgt: np.ndarray) -> None:
    """Hard-fail if mean_tgt[zg500] is outside ZG500_AUDIT_RANGE_M.

    Runs before any .npy is written so a bad pack never produces stats
    artifacts. ``mean_tgt`` is the C=53 per-channel running mean from
    the Welford pass, indexed by ``TARGET_CHANNELS``. See plan §3.6.
    """
    try:
        idx = TARGET_CHANNELS.index("zg500")
    except ValueError as e:
        raise RuntimeError(
            "TARGET_CHANNELS does not contain 'zg500'; this build is not "
            "the v10 contract. Did you forget to update channels.py?"
        ) from e
    val = float(mean_tgt[idx])
    lo, hi = ZG500_AUDIT_RANGE_M
    if not (lo <= val <= hi):
        raise RuntimeError(
            f"zg500 audit (L7d) failed: mean_tgt[zg500] = {val:.2f} m is "
            f"outside [{lo:.0f}, {hi:.0f}] m. Likely cause: wrong lev_2 "
            f"value lookup, mis-ordered ZG_PLEV_HPA, or a postproc-side "
            f"unit change. Aborting before any .npy is written."
        )

# Forcing channels that are constant along the time axis (time_means equal the
# per-cell value in any file). Used by validate.py to confirm the invariance;
# NOT used to exempt these channels from the MIN_STD_EPSILON hard-fail —
# global std is taken over (T, H, W), so real PlaSim lsm/sg still have
# non-zero spatial std.
#
# z0 is intentionally excluded. Diagnostic (sim52, 98 train years) shows z0 is
# land-static but ocean-dynamic: pure land cells are bit-identical within and
# across years, while ocean cells vary 1.5e-5 → 1e-3 m consistent with
# Charnock/sea-state roughness. Top-1% highest-variability cells are 100%
# ocean. z0 is therefore classified as a time-varying prescribed forcing.
STATIC_FORCING_NAMES: frozenset[str] = frozenset({"lsm", "sg"})


# ---------------------------------------------------------------------------
# Chan's parallel Welford update (for per-channel global mean / variance)
# ---------------------------------------------------------------------------
def _combine(
    na: int,
    mean_a: np.ndarray,
    m2_a: np.ndarray,
    nb: int,
    mean_b: np.ndarray,
    m2_b: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    nc = na + nb
    if nc == 0:
        return 0, mean_a, m2_a
    delta = mean_b - mean_a
    mean_c = mean_a + delta * (nb / nc)
    m2_c = m2_a + m2_b + (delta ** 2) * na * nb / nc
    return nc, mean_c, m2_c


def _batch_stats(data: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """(T, C, H, W) → (n_per_channel, mean[C], M2[C]) in float64.

    n_per_channel = T * H * W, mean along (T, H, W), M2 = sum (x - mean)^2.
    """
    T, _, H, W = data.shape
    n = T * H * W
    data64 = data.astype(np.float64, copy=False)
    mean = data64.mean(axis=(0, 2, 3))
    diff = data64 - mean[None, :, None, None]
    m2 = (diff * diff).sum(axis=(0, 2, 3))
    return n, mean, m2


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def compute_stats(
    output_root: Path,
    train_years: tuple[int, int] = (3, 100),
    epsilon: float = MIN_STD_EPSILON,
) -> None:
    train_dir = output_root / "train"
    stats_dir = output_root / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    lo, hi = train_years
    files = sorted(
        p for p in train_dir.glob("MOST.*.h5")
        if lo <= int(p.stem.split(".")[1]) <= hi
    )
    if not files:
        raise RuntimeError(
            f"no training files found in {train_dir} for year range {train_years}"
        )
    logger.info("found %d training files in [%d, %d]", len(files), lo, hi)

    # Initialize running stats lazily (we need shape from the first file).
    H: int | None = None
    W: int | None = None

    n_tgt = 0
    mean_tgt = np.zeros(53, dtype=np.float64)
    m2_tgt = np.zeros(53, dtype=np.float64)
    time_sum_tgt: np.ndarray | None = None
    t_count_tgt = 0

    n_frc = 0
    mean_frc = np.zeros(6, dtype=np.float64)
    m2_frc = np.zeros(6, dtype=np.float64)
    time_sum_frc: np.ndarray | None = None
    t_count_frc = 0

    for path in files:
        with h5py.File(path, "r") as f:
            state = f["fields_state"][...]            # (T, 52, H, W) float32
            diag = f["fields_diagnostic"][...]        # (T, 1,  H, W)
            forcing = f["forcing"][...]               # (T, 6,  H, W)

        if state.shape[1] != 52 or diag.shape[1] != 1 or forcing.shape[1] != 6:
            raise RuntimeError(
                f"{path.name}: unexpected channel counts state={state.shape[1]} "
                f"diag={diag.shape[1]} forcing={forcing.shape[1]}"
            )

        tgt = np.concatenate([state, diag], axis=1)   # (T, 53, H, W) float32

        if H is None:
            _, _, H, W = tgt.shape
            time_sum_tgt = np.zeros((53, H, W), dtype=np.float64)
            time_sum_frc = np.zeros((6, H, W), dtype=np.float64)
        else:
            assert (H, W) == tgt.shape[2:]

        n_b, mean_b, m2_b = _batch_stats(tgt)
        n_tgt, mean_tgt, m2_tgt = _combine(n_tgt, mean_tgt, m2_tgt, n_b, mean_b, m2_b)
        time_sum_tgt += tgt.astype(np.float64).sum(axis=0)
        t_count_tgt += tgt.shape[0]

        n_b, mean_b, m2_b = _batch_stats(forcing)
        n_frc, mean_frc, m2_frc = _combine(n_frc, mean_frc, m2_frc, n_b, mean_b, m2_b)
        time_sum_frc += forcing.astype(np.float64).sum(axis=0)
        t_count_frc += forcing.shape[0]

        logger.info("  %s  (T=%d)", path.name, tgt.shape[0])

    assert time_sum_tgt is not None and time_sum_frc is not None
    assert H is not None and W is not None

    std_tgt = np.sqrt(m2_tgt / n_tgt)
    std_frc = np.sqrt(m2_frc / n_frc)
    time_mean_tgt = (time_sum_tgt / t_count_tgt).astype(np.float32)
    time_mean_frc = (time_sum_frc / t_count_frc).astype(np.float32)

    # Hard-fail on any channel with std < epsilon. The plan is explicit: no
    # per-channel exemptions — real PlaSim lsm/sg/z0 have non-zero spatial
    # std so they pass; if they don't, something is wrong with the data.
    errs: list[str] = []
    for i, name in enumerate(TARGET_CHANNELS):
        if std_tgt[i] < epsilon:
            errs.append(f"  target channel [{i}] '{name}': std={std_tgt[i]:.3e}")
    for i, name in enumerate(FORCING_CHANNELS):
        if std_frc[i] < epsilon:
            errs.append(f"  forcing channel [{i}] '{name}': std={std_frc[i]:.3e}")
    if errs:
        raise RuntimeError(
            f"std < {epsilon:.1e} on {len(errs)} channel(s):\n" + "\n".join(errs)
        )

    # v10 inline audit (§3.6, L7d): zg500 global mean must be within
    # the audit band before any stats .npy is written.
    _audit_zg500_inline(mean_tgt)

    # Write .npy in the shapes the loss + dataloader expect.
    def _save(name: str, arr: np.ndarray) -> None:
        path = stats_dir / name
        np.save(path, arr)
        logger.info("wrote %s  shape=%s dtype=%s", path, arr.shape, arr.dtype)

    _save(
        "global_means.npy",
        mean_tgt.astype(np.float32).reshape(1, 53, 1, 1),
    )
    _save(
        "global_stds.npy",
        std_tgt.astype(np.float32).reshape(1, 53, 1, 1),
    )
    _save(
        "time_means.npy",
        time_mean_tgt.reshape(1, 53, H, W),
    )
    _save(
        "forcing_global_means.npy",
        mean_frc.astype(np.float32).reshape(1, 6, 1, 1),
    )
    _save(
        "forcing_global_stds.npy",
        std_frc.astype(np.float32).reshape(1, 6, 1, 1),
    )
    _save(
        "forcing_time_means.npy",
        time_mean_frc.reshape(1, 6, H, W),
    )

    notes_path = stats_dir / "sst_land_sentinel_notes.txt"
    with notes_path.open("w") as fh:
        fh.write(
            "sst land-fill sentinel notes\n"
            "----------------------------\n"
            "The sst channel in /forcing has been land-filled with a constant "
            "271.35 K sentinel (see packager.py --sst-land-fill-k). As a result, "
            "global_means[sst] and global_stds[sst] reflect a mix of ocean + "
            "sentinel and are NOT representative of ocean sst alone. Downstream "
            "consumers that apply zscore with these stats will recover the "
            "ocean signal consistently with training, but any physical "
            "interpretation of the sst stats must account for the sentinel.\n"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument(
        "--train-years",
        type=int,
        nargs=2,
        default=[3, 100],
        metavar=("START", "END"),
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
    try:
        compute_stats(
            args.output_root,
            train_years=tuple(args.train_years),
            epsilon=args.epsilon,
        )
    except RuntimeError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
