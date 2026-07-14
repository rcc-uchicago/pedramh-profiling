"""Synthetic timestamp generation + per-split continuity.

Makani's MultifilesDataset enforces uniform dT both within and across
files (data_loader_multifiles.py:216). Our scheme achieves that by
assigning year N its starting offset = sum of prior-year T values in
the same split, so concatenated /timestamp across a split is strictly
uniform with diff == STEP_SECONDS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from plasim_makani_packager.packager import (
    STEP_SECONDS,
    _compute_split_offset,
    _synthetic_timestamps,
)


def test_step_seconds():
    assert STEP_SECONDS == 21600  # 6 h


def test_synthetic_timestamps_uniform_diff():
    ts = _synthetic_timestamps(offset_seconds=1_000_000, T=4)
    assert ts.dtype == np.int64
    assert ts.shape == (4,)
    assert ts[0] == 1_000_000
    assert np.all(np.diff(ts) == STEP_SECONDS)


def test_compute_split_offset_reads_prior_year_T(tmp_path: Path):
    """year 5 in train split (with train=3..100): offset = (T_3 + T_4) * STEP_SECONDS.

    The computation opens each prior year's MOST file and sums their time sizes.
    """
    sim = 52
    d = tmp_path / "postproc" / f"sim{sim}"
    d.mkdir(parents=True)

    def _write_most(year: int, T: int) -> None:
        ds = xr.Dataset(
            {"dummy": (("time",), np.zeros(T))},
            coords={"time": np.arange(T, dtype=np.float64)},
        )
        ds.to_netcdf(d / f"MOST.{year:04d}.nc")

    _write_most(3, 7)
    _write_most(4, 11)
    _write_most(5, 5)

    off_5 = _compute_split_offset(
        year=5,
        split="train",
        sim=sim,
        postproc_root=tmp_path / "postproc",
        train_years=(3, 100),
        valid_years=(101, 120),
        test_years=(121, 128),
    )
    assert off_5 == (7 + 11) * STEP_SECONDS

    off_3 = _compute_split_offset(
        year=3,
        split="train",
        sim=sim,
        postproc_root=tmp_path / "postproc",
        train_years=(3, 100),
        valid_years=(101, 120),
        test_years=(121, 128),
    )
    assert off_3 == 0

    # valid split starts fresh at 0 for its first year
    _write_most(101, 4)
    off_101 = _compute_split_offset(
        year=101,
        split="valid",
        sim=sim,
        postproc_root=tmp_path / "postproc",
        train_years=(3, 100),
        valid_years=(101, 120),
        test_years=(121, 128),
    )
    assert off_101 == 0


def test_split_concat_strictly_uniform():
    """Simulate a three-year train split with T = [7, 11, 5], stacked per
    the offset scheme above. The concatenation must have diff == STEP_SECONDS
    everywhere (intra AND inter file)."""
    T = [7, 11, 5]
    offsets = [0, 7 * STEP_SECONDS, (7 + 11) * STEP_SECONDS]
    concat = np.concatenate(
        [_synthetic_timestamps(off, t) for off, t in zip(offsets, T)]
    )
    assert np.all(np.diff(concat) == STEP_SECONDS)
