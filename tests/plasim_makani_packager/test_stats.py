"""stats.compute_stats + min-std epsilon hard-fail + static forcing exemption."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from plasim_makani_packager.channels import (
    DIAGNOSTIC_CHANNELS,
    FORCING_CHANNELS,
    STATE_CHANNELS,
)
from plasim_makani_packager.stats import STATIC_FORCING_NAMES, compute_stats


_STATIC_RNG = np.random.default_rng(42)  # SAME static field across all files


def _write_fake_h5(path: Path, *, T: int, H: int = 8, W: int = 16, seed: int) -> None:
    """Static forcing channels get a per-cell pattern identical across files
    (matches real PlaSim: topography doesn't change across sim-years).
    Varying channels get per-file i.i.d. noise."""
    rng = np.random.default_rng(seed)
    static_rng = np.random.default_rng(42)  # fixed: static fields shared across files

    with h5py.File(path, "w") as f:
        f.create_dataset(
            "fields_state",
            data=rng.standard_normal((T, 52, H, W)).astype(np.float32),
        )
        f.create_dataset(
            "fields_diagnostic",
            data=rng.random((T, 1, H, W)).astype(np.float32),
        )
        forcing = np.empty((T, 6, H, W), dtype=np.float32)
        for idx, name in enumerate(FORCING_CHANNELS):
            if name in STATIC_FORCING_NAMES:
                forcing[:, idx] = static_rng.standard_normal((H, W)).astype(np.float32)[None]
            else:
                forcing[:, idx] = rng.standard_normal((T, H, W)).astype(np.float32)
        f.create_dataset("forcing", data=forcing)


def test_shapes_dtypes_and_epsilon_passes(tmp_path: Path):
    train = tmp_path / "train"
    train.mkdir()
    for y in (3, 4, 5):
        _write_fake_h5(train / f"MOST.{y:04d}.h5", T=3, seed=y)

    compute_stats(tmp_path, train_years=(3, 5), epsilon=1e-6)

    stats = tmp_path / "stats"
    for name, shape in [
        ("global_means.npy", (1, 53, 1, 1)),
        ("global_stds.npy", (1, 53, 1, 1)),
        ("time_means.npy", (1, 53, 8, 16)),
        ("forcing_global_means.npy", (1, 6, 1, 1)),
        ("forcing_global_stds.npy", (1, 6, 1, 1)),
        ("forcing_time_means.npy", (1, 6, 8, 16)),
    ]:
        arr = np.load(stats / name)
        assert arr.shape == shape
        assert arr.dtype == np.float32
        assert np.isfinite(arr).all()

    # Global std is taken over (T, H, W). Static forcing has spatial
    # variation (per-cell pattern repeated along T), so its global std is
    # non-zero — real lsm/sg satisfies this; the plan (v9) does not
    # grant a per-channel exemption.
    frc_std = np.load(stats / "forcing_global_stds.npy").ravel()
    assert np.all(frc_std >= 1e-6)

    # Static channels' time_means should equal the first-time-step value
    # (temporal invariance check used by validate.run_structural).
    ftm = np.load(stats / "forcing_time_means.npy")
    for idx, name in enumerate(FORCING_CHANNELS):
        if name not in STATIC_FORCING_NAMES:
            continue
        sample_path = next((tmp_path / "train").glob("MOST.*.h5"))
        with h5py.File(sample_path, "r") as f:
            first = f["forcing"][0, idx]
        assert np.allclose(ftm[0, idx], first, atol=1e-5)


def test_hard_fail_on_zero_std_target_channel(tmp_path: Path):
    train = tmp_path / "train"
    train.mkdir()
    path = train / "MOST.0003.h5"
    with h5py.File(path, "w") as f:
        # All zeros in fields_state → any state channel has std=0.
        f.create_dataset(
            "fields_state", data=np.zeros((2, 52, 4, 4), dtype=np.float32)
        )
        f.create_dataset(
            "fields_diagnostic", data=np.zeros((2, 1, 4, 4), dtype=np.float32)
        )
        f.create_dataset(
            "forcing", data=np.zeros((2, 6, 4, 4), dtype=np.float32)
        )
    with pytest.raises(RuntimeError, match="std < "):
        compute_stats(tmp_path, train_years=(3, 3))
