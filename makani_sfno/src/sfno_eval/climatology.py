"""climatology — time-of-year-proleptic climatology from the training pool.

Implements docs/sfno_eval_plan.md §C.2.

Indexing rule (from v2.2 round-2 fix 2): we bin by ``(month, day,
hour_quarter)`` where ``hour_quarter ∈ {0, 6, 12, 18}``. Total bins =
366 days × 4 hour-slots = **1464** per channel. Most bins receive ~100
contributors (one per training year); the four Feb-29 bins receive only
~24 contributors (one per leap-year file). ``n_contributors[366, 4]``
is stored alongside ``mean``/``std`` so downstream consumers can weight
or skip low-N bins.

Why not sample-of-year (v2.1's approach): with variable file lengths
(1455 vs 1459), sample s=240 in a non-leap file does NOT land on the
same calendar slot as s=240 in a leap file — Feb 29 shifts everything
by one slot. Naive averaging would smear by ±0.5 day per leap year.

This module is **CPU-only**. The accumulator footprint is:
    (366, 4, 53, H, W) × fp32 × 2 (sum, sumsq) + (366, 4) × i32 (count)
≈ 12.4 GB at H=64, W=128 — fits comfortably in a Stampede3 ``skx``
node (191 GB).
"""
from __future__ import annotations

import logging
import re
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anchor parsing — matches scripts/trace_calendar_anchors.py
# ---------------------------------------------------------------------------

_ANCHOR_RE = re.compile(
    r"days since (\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})"
)


def parse_anchor(units: str) -> tuple[int, int, int, int, int, int]:
    """Parse 'days since YYYY-MM-DD HH:MM:SS' → (Y, M, D, h, m, s)."""
    if isinstance(units, bytes):
        units = units.decode("utf-8")
    m = _ANCHOR_RE.match(units)
    if m is None:
        raise ValueError(f"unparseable plasim_time_units: {units!r}")
    return tuple(int(g) for g in m.groups())  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Bin index
# ---------------------------------------------------------------------------

# 1-indexed month/day for clarity; we'll subtract 1 when indexing the
# accumulators. February has 29 valid days under the proleptic Gregorian
# calendar (the centennial-exception rule decides whether a *file* even
# contains Feb 29 samples, not the bin layout).
_DAYS_PER_MONTH_LEAP = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _day_of_year_366(month: int, day: int) -> int:
    """Map a (month, day) pair to a 0-indexed day-of-year in [0, 366).

    Uses a fixed 366-day calendar so Feb 29 always has a valid bin even
    when the contributing year is non-leap (in which case zero ICs
    contribute and ``n_contributors`` stays 0).
    """
    if not (1 <= month <= 12):
        raise ValueError(f"bad month: {month}")
    cum = sum(_DAYS_PER_MONTH_LEAP[:month - 1])
    return cum + (day - 1)


def calendar_bin(month: int, day: int, hour: int) -> tuple[int, int]:
    """Return ``(day_of_year_idx, hour_quarter_idx)``.

    ``day_of_year_idx`` is in ``[0, 366)`` and ``hour_quarter_idx`` is
    in ``{0, 1, 2, 3}`` mapping to ``{0, 6, 12, 18}`` UTC.
    """
    return _day_of_year_366(month, day), hour // 6


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------

class ClimatologyAccumulator:
    """Welford-style accumulator over (366, 4, n_chan, H, W).

    Two backing arrays: ``sum`` and ``sumsq`` (fp32). The count array
    ``n_contrib[366, 4]`` is shared across all channels (a single sample
    contributes to every channel of one bin).

    After all files have been ingested, call :meth:`finalize` to get
    ``mean[366, 4, n_chan, H, W]``, ``std[366, 4, n_chan, H, W]``, and
    ``n_contributors[366, 4]``.
    """

    def __init__(self, *, n_chan: int, H: int, W: int):
        self.n_chan = n_chan
        self.H = H
        self.W = W
        self.sum = np.zeros((366, 4, n_chan, H, W), dtype=np.float32)
        self.sumsq = np.zeros((366, 4, n_chan, H, W), dtype=np.float64)
        self.n_contrib = np.zeros((366, 4), dtype=np.int64)

    def update(
        self, doy_idx: int, hq_idx: int, sample: np.ndarray
    ) -> None:
        """Add one ``(n_chan, H, W)`` sample to bin ``(doy_idx, hq_idx)``."""
        if sample.shape != (self.n_chan, self.H, self.W):
            raise ValueError(
                f"sample shape {sample.shape} != ({self.n_chan}, {self.H}, {self.W})"
            )
        self.sum[doy_idx, hq_idx] += sample
        # Accumulate squares in fp64 for numerical stability across ~100 contributors.
        self.sumsq[doy_idx, hq_idx] += sample.astype(np.float64) ** 2
        self.n_contrib[doy_idx, hq_idx] += 1

    def finalize(self) -> dict[str, np.ndarray]:
        """Return mean / std / n_contributors arrays.

        Bins with zero contributors yield ``mean=0`` and ``std=0``;
        downstream code is expected to skip those bins via the
        ``n_contributors`` array.
        """
        n = self.n_contrib.astype(np.float64)
        # Reshape n for broadcast against (366, 4, n_chan, H, W).
        n_b = n[..., np.newaxis, np.newaxis, np.newaxis]
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(n_b > 0, self.sum / np.maximum(n_b, 1), 0.0).astype(np.float32)
            var = np.where(n_b > 1,
                           (self.sumsq - n_b * (mean.astype(np.float64) ** 2)) / np.maximum(n_b - 1, 1),
                           0.0)
            # Clamp tiny negatives from float-roundoff before sqrt.
            var = np.clip(var, 0.0, None)
            std = np.sqrt(var).astype(np.float32)
        return {
            "mean": mean,
            "std": std,
            "n_contributors": self.n_contrib.copy(),
        }


# ---------------------------------------------------------------------------
# File walker
# ---------------------------------------------------------------------------

def ingest_file(
    accumulator: ClimatologyAccumulator,
    h5_path: Path,
    *,
    state_key: str = "fields_state",
    diag_key: str | None = "fields_diagnostic",
) -> int:
    """Read one h5 file fully and update ``accumulator``.

    Returns the number of samples ingested (1455 or 1459 for production
    files).

    The combined channel field is ``state ‖ diagnostic`` (52 + 1 = 53)
    — same layout as the training-time target tensor. If
    ``diag_key`` is None (e.g. for tests), only the state channels are
    used and the accumulator must have been built with ``n_chan=52``.
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        units = f.attrs.get("plasim_time_units", "")
        anchor = parse_anchor(units)
        Y_anchor, M_anchor, D_anchor, h_anchor, mi_anchor, s_anchor = anchor

        time_plasim = f["time_plasim"][:]                # (N,) float64 days
        state = f[state_key][:]                          # (N, n_state, H, W)
        if diag_key is not None and diag_key in f:
            diag = f[diag_key][:]                        # (N, n_diag, H, W)
            field = np.concatenate([state, diag], axis=1)
        else:
            field = state

    n = int(time_plasim.shape[0])
    if field.shape[1] != accumulator.n_chan:
        raise RuntimeError(
            f"{h5_path}: field has {field.shape[1]} channels but accumulator expects {accumulator.n_chan}"
        )

    # Build absolute datetimes via cftime.DatetimeProlepticGregorian.
    import cftime
    base = cftime.DatetimeProlepticGregorian(
        Y_anchor, M_anchor, D_anchor, h_anchor, mi_anchor, s_anchor
    )

    for s in range(n):
        dt = base + timedelta(days=float(time_plasim[s]))
        doy_idx, hq_idx = calendar_bin(dt.month, dt.day, dt.hour)
        accumulator.update(doy_idx, hq_idx, field[s])

    return n


def build_climatology(
    train_files: Iterable[Path],
    *,
    n_chan: int = 53,
    H: int = 64,
    W: int = 128,
) -> dict[str, np.ndarray]:
    """Walk a sequence of training files and return mean/std/n_contributors."""
    accumulator = ClimatologyAccumulator(n_chan=n_chan, H=H, W=W)
    train_files = list(train_files)
    for i, path in enumerate(train_files):
        ingested = ingest_file(accumulator, path)
        logger.info("ingested %s (%d samples) [%d/%d]",
                    Path(path).name, ingested, i + 1, len(train_files))
    return accumulator.finalize()


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def lookup_clim_at(
    clim_mean: np.ndarray,
    clim_n: np.ndarray,
    *,
    month: int,
    day: int,
    hour: int,
) -> np.ndarray | None:
    """Return ``clim_mean[doy, hq]`` if ``n_contributors[doy, hq] > 0``, else None."""
    doy, hq = calendar_bin(month, day, hour)
    if clim_n[doy, hq] == 0:
        return None
    return clim_mean[doy, hq]
