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
    ZG_PLEV_HPA,
)
from plasim_makani_packager.stats import STATIC_FORCING_NAMES, compute_stats

# Reference altitude table — used to give synthetic zg{P} channels
# physically plausible per-level means so the v10 audit
# (zg500 ∈ [5400, 5700] m) passes by construction.
_ZG_PLEV_REFERENCE_M: dict[int, float] = {
    150: 13500.0, 200: 11700.0, 250: 10300.0, 300: 9100.0,
    400:  7100.0, 500:  5550.0, 600:  4200.0, 700: 3000.0,
    850:  1450.0, 925:   750.0,
}


_STATIC_RNG = np.random.default_rng(42)  # SAME static field across all files


def _write_fake_h5(path: Path, *, T: int, H: int = 8, W: int = 16, seed: int) -> None:
    """Static forcing channels get a per-cell pattern identical across files
    (matches real PlaSim: topography doesn't change across sim-years).
    Varying channels get per-file i.i.d. noise.

    zg{P} state channels (indices 42..51) are filled with values centred on
    the standard-atmosphere geopotential heights at each pressure (per
    docs/plasim_zg_plev_migration_plan.md §3.9), so the inline zg500 audit
    (`stats._audit_zg500_inline`) passes on synthetic data."""
    rng = np.random.default_rng(seed)
    static_rng = np.random.default_rng(42)  # fixed: static fields shared across files

    fields_state = rng.standard_normal((T, 52, H, W)).astype(np.float32)
    for k, hpa in enumerate(ZG_PLEV_HPA):
        fields_state[:, 42 + k] = (
            _ZG_PLEV_REFERENCE_M[int(hpa)]
            + rng.normal(0.0, 50.0, size=(T, H, W))
        ).astype(np.float32)

    with h5py.File(path, "w") as f:
        f.create_dataset("fields_state", data=fields_state)
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
