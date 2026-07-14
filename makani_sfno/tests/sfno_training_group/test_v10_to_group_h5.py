"""Tests for src/sfno_training_group/tools/convert_v10_to_group_h5.py"""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sfno_training_group.tools._h5_keys import (  # noqa: E402
    PLEVS_PA,
    SIGMA_LEVELS,
    ZG_HPA,
    h5_key,
    all_input_keys_for_smoke,
)
from sfno_training_group.tools.convert_v10_to_group_h5 import (  # noqa: E402
    EXPECTED_DIAGNOSTIC_CHANNELS,
    EXPECTED_FORCING_CHANNELS,
    EXPECTED_STATE_CHANNELS,
    convert_year,
    _synthetic_dt,
)


def _build_synthetic_v10_h5(path: Path, *, n_timesteps: int = 4, year: int = 12) -> Path:
    """Write a v10-shaped h5 with predictable per-channel constants."""
    path.parent.mkdir(parents=True, exist_ok=True)
    H, W = 64, 128
    fields_state = np.zeros((n_timesteps, 52, H, W), dtype=np.float32)
    fields_diagnostic = np.zeros((n_timesteps, 1, H, W), dtype=np.float32)
    forcing = np.zeros((n_timesteps, 6, H, W), dtype=np.float32)

    # Encode (channel_index, timestep) into the field value so the converter can
    # be checked exactly.
    for t in range(n_timesteps):
        for c in range(52):
            fields_state[t, c, :, :] = c * 1000 + t
        fields_diagnostic[t, 0, :, :] = 9000 + t
        for c in range(6):
            # z0 (idx 2) made time-varying to exercise the audit; others constant.
            if c == 2:
                forcing[t, c, :, :] = 1e-3 + 1e-4 * t
            else:
                forcing[t, c, :, :] = -1.0 - c

    with h5py.File(path, "w") as f:
        f.create_dataset("fields_state", data=fields_state)
        f.create_dataset("fields_diagnostic", data=fields_diagnostic)
        f.create_dataset("forcing", data=forcing)
        f.create_dataset("timestamp", data=np.arange(n_timesteps, dtype=np.int64) * 21600)
        f.create_dataset("time_plasim", data=np.arange(n_timesteps, dtype=np.float64) * 0.25)
        f.create_dataset(
            "channel_state",
            data=np.array(EXPECTED_STATE_CHANNELS, dtype=h5py.string_dtype("ascii")),
        )
        f.create_dataset(
            "channel_diagnostic",
            data=np.array(EXPECTED_DIAGNOSTIC_CHANNELS, dtype=h5py.string_dtype("ascii")),
        )
        f.create_dataset(
            "channel_forcing",
            data=np.array(EXPECTED_FORCING_CHANNELS, dtype=h5py.string_dtype("ascii")),
        )
        f.create_dataset("lat", data=np.linspace(87.86, -87.86, H))
        f.create_dataset("lon", data=np.linspace(0, 357.1875, W))
        f.attrs["plasim_calendar"] = "proleptic_gregorian"
        f.attrs["year"] = year
    return path


def _make_v10_split_layout(tmp_path: Path, *, year: int, n_timesteps: int) -> Path:
    """Create a v10 src tree with the year file under train/."""
    src_root = tmp_path / "v10_src"
    train_dir = src_root / "train"
    _build_synthetic_v10_h5(train_dir / f"MOST.{year:04d}.h5",
                            n_timesteps=n_timesteps, year=year)
    return src_root


def test_convert_year_writes_per_timestep_files_with_expected_keys(tmp_path: Path) -> None:
    src_root = _make_v10_split_layout(tmp_path, year=12, n_timesteps=4)
    dst = tmp_path / "group_out"
    dst.mkdir()

    src_path = src_root / "train" / "MOST.0012.h5"
    entry = convert_year(src_path, dst, year=12, max_forecast_lead_steps=2,
                         n_timesteps_floor=1)

    # File presence + naming (unpadded year, 4-digit idx).
    for idx in range(4):
        assert (dst / f"12_{idx:04d}.h5").is_file(), f"missing {idx}"

    # Manifest entry.
    assert entry["year"] == 12
    assert entry["n_timesteps"] == 4
    assert entry["last_train_init_idx"] == 2          # n - 2
    assert entry["last_val_init_idx_for_max_lead_K"] == 1  # n - 1 - K with K=2
    # Synthetic dates: idx 3 -> 0012-01-01 18:00:00; train end exclusive = idx 3 = 18:00.
    assert entry["last_train_init_dt"] == "0012-01-01 12:00:00"
    assert entry["train_end_exclusive_dt"] == "0012-01-01 18:00:00"
    # val: K=2 lead steps → last_val_init_idx = n-1-K = 4-1-2 = 1 → 0012-01-01 06:00:00.
    assert entry["last_val_init_dt_for_max_lead_K"] == "0012-01-01 06:00:00"
    assert entry["val_end_exclusive_dt_for_max_lead_K"] == "0012-01-01 12:00:00"


def test_convert_year_keys_match_canonical_smoke_set(tmp_path: Path) -> None:
    src_root = _make_v10_split_layout(tmp_path, year=12, n_timesteps=2)
    dst = tmp_path / "group_out"
    dst.mkdir()
    src_path = src_root / "train" / "MOST.0012.h5"
    convert_year(src_path, dst, year=12, max_forecast_lead_steps=1,
                 n_timesteps_floor=1)

    with h5py.File(dst / "12_0000.h5", "r") as f:
        keys = sorted(list(f["input"].keys()))
    expected = sorted(all_input_keys_for_smoke())
    assert keys == expected, f"set diff: {set(keys) ^ set(expected)}"
    assert len(expected) == 59


def test_channel_mapping_round_trip(tmp_path: Path) -> None:
    """Each per-level destination dataset == the corresponding source state slice."""
    src_root = _make_v10_split_layout(tmp_path, year=12, n_timesteps=2)
    dst = tmp_path / "group_out"
    dst.mkdir()
    src_path = src_root / "train" / "MOST.0012.h5"
    convert_year(src_path, dst, year=12, max_forecast_lead_steps=1,
                 n_timesteps_floor=1)

    # Read source and converted t=0 file.
    with h5py.File(src_path, "r") as src, h5py.File(dst / "12_0000.h5", "r") as out:
        # pl ← state[0]
        assert np.array_equal(out["input/pl"][:], src["fields_state"][0, 0])
        # tas ← state[1]
        assert np.array_equal(out["input/tas"][:], src["fields_state"][0, 1])
        # ta1 ← state[2] mapped to ta_<sigma[0]>
        for level_i in range(10):
            # ta
            assert np.array_equal(
                out[f"input/{h5_key('ta', level_i)}"][:],
                src["fields_state"][0, 2 + level_i],
            )
            # ua
            assert np.array_equal(
                out[f"input/{h5_key('ua', level_i)}"][:],
                src["fields_state"][0, 12 + level_i],
            )
            # va
            assert np.array_equal(
                out[f"input/{h5_key('va', level_i)}"][:],
                src["fields_state"][0, 22 + level_i],
            )
            # hus
            assert np.array_equal(
                out[f"input/{h5_key('hus', level_i)}"][:],
                src["fields_state"][0, 32 + level_i],
            )
            # zg
            assert np.array_equal(
                out[f"input/{h5_key('zg', level_i)}"][:],
                src["fields_state"][0, 42 + level_i],
            )
        # diagnostic
        assert np.array_equal(out["input/pr_6h"][:], src["fields_diagnostic"][0, 0])
        # forcing
        for fname, idx in (("lsm", 0), ("sg", 1), ("z0", 2),
                           ("sst", 3), ("rsdt", 4), ("sic", 5)):
            assert np.array_equal(out[f"input/{fname}"][:], src["forcing"][0, idx])


def test_z0_audit_recorded(tmp_path: Path) -> None:
    src_root = _make_v10_split_layout(tmp_path, year=12, n_timesteps=4)
    dst = tmp_path / "group_out"
    dst.mkdir()
    src_path = src_root / "train" / "MOST.0012.h5"
    entry = convert_year(src_path, dst, year=12, max_forecast_lead_steps=1,
                         n_timesteps_floor=1)
    # Synthetic z0 has temporal std (1e-4 step over 4 timesteps); audit ≈ std of [1e-3, 1.1e-3, 1.2e-3, 1.3e-3]
    z0_values = 1e-3 + 1e-4 * np.arange(4)
    expected_std = float(np.std(z0_values))
    # Spatial mean reduces a const slab to its scalar — we just check finite + > 0.
    assert entry["z0_temporal_std_mean"] > 0
    assert entry["z0_temporal_std_mean"] == pytest.approx(expected_std, rel=1e-5)


def test_dtype_preserved_as_float32(tmp_path: Path) -> None:
    src_root = _make_v10_split_layout(tmp_path, year=12, n_timesteps=2)
    dst = tmp_path / "group_out"
    dst.mkdir()
    src_path = src_root / "train" / "MOST.0012.h5"
    convert_year(src_path, dst, year=12, max_forecast_lead_steps=1,
                 n_timesteps_floor=1)
    with h5py.File(dst / "12_0000.h5", "r") as f:
        for key in all_input_keys_for_smoke():
            ds = f[f"input/{key}"]
            assert ds.dtype == np.float32, f"{key} dtype={ds.dtype}"
            assert ds.shape == (64, 128), f"{key} shape={ds.shape}"


def test_synthetic_dt_calendar() -> None:
    dt = _synthetic_dt(year=12, idx=0)
    assert dt.strftime("%Y-%m-%d %H:%M:%S") == "0012-01-01 00:00:00"
    dt = _synthetic_dt(year=121, idx=4)  # 4 * 6h = 1d
    assert dt.strftime("%Y-%m-%d %H:%M:%S") == "0121-01-02 00:00:00"


def test_n_timesteps_floor_enforced(tmp_path: Path) -> None:
    src_root = _make_v10_split_layout(tmp_path, year=12, n_timesteps=10)
    dst = tmp_path / "group_out"
    dst.mkdir()
    src_path = src_root / "train" / "MOST.0012.h5"
    with pytest.raises(RuntimeError, match="floor"):
        convert_year(src_path, dst, year=12, max_forecast_lead_steps=1, n_timesteps_floor=20)
