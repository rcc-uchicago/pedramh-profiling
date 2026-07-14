"""T.6 — converter --pad-canonical-years padding bit-identity (Phase F).

Verifies that:
- pad_canonical_year copies year-(N+1) idx [0..n_pad) into year-N idx
  [n_native..n_padded), bit-identical at the h5-file level.
- apply_pad_pass updates the manifest with n_timesteps_native, n_timesteps_padded,
  is_leap, pad_source.
- F.B2 contract: year 11 -> 1460 (donor 12), year 12 -> 1464 (donor 13).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sfno_training_group.tools.convert_v10_to_group_h5 import (  # noqa: E402
    PAD_TARGETS,
    apply_pad_pass,
    pad_canonical_year,
)


def _make_year(dst: Path, year: int, n_frames: int, *, fill_base: float) -> None:
    """Write n_frames synthetic <year>_<idx:04>.h5 files with a sentinel value
    so we can check bit-identity after padding."""
    for idx in range(n_frames):
        path = dst / f"{year}_{idx:04d}.h5"
        with h5py.File(path, "w") as f:
            grp = f.create_group("input")
            # One synthetic dataset is enough to verify bit-identity.
            arr = np.full((4, 8), fill_base + year * 1000 + idx, dtype=np.float32)
            grp.create_dataset("z0", data=arr)


def _bytes_of(path: Path) -> bytes:
    return path.read_bytes()


def test_pad_canonical_year_year11_donor_year12_bit_identical(tmp_path: Path) -> None:
    # Set up: native year 11 (1455 frames) + native year 12 (5+ frames as donor).
    # We use small synthetic counts to keep the test fast — pass n_native/n_padded explicitly.
    _make_year(tmp_path, year=11, n_frames=10, fill_base=1.0)
    _make_year(tmp_path, year=12, n_frames=10, fill_base=2.0)

    pad_meta = pad_canonical_year(
        tmp_path, year=11, n_native=10, n_padded=15, donor_year=12,
    )
    # Padded files should bit-equal donor files.
    for i in range(5):
        donor = tmp_path / f"12_{i:04d}.h5"
        padded = tmp_path / f"11_{10 + i:04d}.h5"
        assert padded.is_file(), f"missing padded file: {padded}"
        assert _bytes_of(padded) == _bytes_of(donor), \
            f"pad mismatch at dst_idx={10+i}, src year=12 idx={i}"
    # Manifest fragment.
    assert pad_meta["n_timesteps_padded"] == 15
    assert pad_meta["is_leap"] is False  # year 11 is no_leap_year
    assert len(pad_meta["pad_source"]) == 5
    assert pad_meta["pad_source"][0] == {"dst_idx": 10, "src_year": 12, "src_idx": 0}
    assert pad_meta["pad_source"][-1] == {"dst_idx": 14, "src_year": 12, "src_idx": 4}


def test_pad_canonical_year_year12_donor_year13_leap(tmp_path: Path) -> None:
    _make_year(tmp_path, year=12, n_frames=10, fill_base=2.0)
    _make_year(tmp_path, year=13, n_frames=10, fill_base=3.0)

    pad_meta = pad_canonical_year(
        tmp_path, year=12, n_native=10, n_padded=19, donor_year=13,
    )
    assert pad_meta["is_leap"] is True
    assert len(pad_meta["pad_source"]) == 9
    for i in range(9):
        donor = tmp_path / f"13_{i:04d}.h5"
        padded = tmp_path / f"12_{10 + i:04d}.h5"
        assert _bytes_of(padded) == _bytes_of(donor)


def test_apply_pad_pass_full_F_B2_contract(tmp_path: Path) -> None:
    """End-to-end: native conversion artifacts already exist for years 11, 12, 13;
    apply_pad_pass updates the manifest and writes padded files for years 11+12.
    """
    # Native frames (synthetic small for fast test).
    for year in (11, 12, 13):
        _make_year(tmp_path, year=year, n_frames=10, fill_base=float(year))

    # Manifest preconditions — apply_pad_pass requires n_timesteps_native fields.
    manifest = {
        "calendar": "proleptic_gregorian",
        "has_year_zero": True,
        "data_timedelta_hours": 6,
        "max_forecast_lead_steps": 60,
        "src_root": "/fake",
        "dst": str(tmp_path),
        "expected_state_channels": [],
        "expected_diagnostic_channels": [],
        "expected_forcing_channels": [],
        "sigma_levels_pl_native": [],
        "zg_levels_pa": [],
        "years": [
            {"year": 11, "n_timesteps": 10, "n_timesteps_native": 10},
            {"year": 12, "n_timesteps": 10, "n_timesteps_native": 10},
            {"year": 13, "n_timesteps": 10, "n_timesteps_native": 10},
        ],
    }
    manifest_path = tmp_path / "_v10_calendar_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    # Override PAD_TARGETS for the test scope to use the small-fixture sizes.
    # We still want to exercise the SAME control flow that production uses, so
    # we monkey-patch via the module's PAD_TARGETS dict.
    from sfno_training_group.tools import convert_v10_to_group_h5 as mod
    saved = dict(mod.PAD_TARGETS)
    try:
        mod.PAD_TARGETS = {
            11: {"n_native": 10, "n_padded": 15, "is_leap": False, "donor_year": 12},
            12: {"n_native": 10, "n_padded": 19, "is_leap": True, "donor_year": 13},
        }
        apply_pad_pass(tmp_path, manifest_path)
    finally:
        mod.PAD_TARGETS = saved

    # Re-read manifest; verify per-year metadata.
    out = json.loads(manifest_path.read_text())
    by_year = {y["year"]: y for y in out["years"]}
    assert by_year[11]["n_timesteps"] == 15
    assert by_year[11]["n_timesteps_native"] == 10
    assert by_year[11]["n_timesteps_padded"] == 15
    assert by_year[11]["is_leap"] is False
    assert len(by_year[11]["pad_source"]) == 5
    assert by_year[12]["n_timesteps"] == 19
    assert by_year[12]["n_timesteps_native"] == 10
    assert by_year[12]["n_timesteps_padded"] == 19
    assert by_year[12]["is_leap"] is True
    assert len(by_year[12]["pad_source"]) == 9
    # Year 13 is a donor only — should be unchanged.
    assert "n_timesteps_padded" not in by_year[13]
    assert by_year[13]["n_timesteps"] == 10

    # Spot check bit-identity (year 11 idx 11 == year 12 idx 1)
    assert _bytes_of(tmp_path / "11_0011.h5") == _bytes_of(tmp_path / "12_0001.h5")


def test_pad_targets_production_contract() -> None:
    """Lock down the production padding contract per plan v5 §F.0."""
    assert PAD_TARGETS[11] == {"n_native": 1455, "n_padded": 1460,
                                "is_leap": False, "donor_year": 12}
    assert PAD_TARGETS[12] == {"n_native": 1455, "n_padded": 1464,
                                "is_leap": True, "donor_year": 13}
    # Years 121-128 must NOT be in the pad targets (single_ic remap to 11/12).
    for year in (121, 122, 123, 124, 125, 126, 127, 128):
        assert year not in PAD_TARGETS, f"year {year} should not be padded"
