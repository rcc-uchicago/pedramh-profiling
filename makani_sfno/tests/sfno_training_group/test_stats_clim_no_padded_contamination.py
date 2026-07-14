"""T.8 — stats / climatology builders skip padded indices via manifest (Phase F.0.b).

Builds a fake data_dir with year 12 native (1455 sentinel-A frames) + 9 padded
sentinel-B frames at idx 1455..1463. Asserts:

- _accumulate (stats): ingests only the 1455 native frames — count per key
  matches 1455 * spatial_size; mean stays in physical (sentinel-A) range and
  excludes sentinel-B contribution.
- Provenance attrs on the resulting xr.Dataset reflect the skip (built_before_padding,
  native_timesteps_total, native_timesteps_by_year, manifest_skip_padded_used).

A single fast year (year 12) is used; the test fixture writes a tiny grid (4x8).
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

# Use small, fast fixture sizes via monkey-patching N_LEVELS / H / W where needed.
# The builders write the "real" key set (50 upper-air + 2 surface + 1 diag + 6 forcing)
# but only with one frame per timestep; we use small spatial dims and a small native
# count so the test runs in <10s.

from sfno_training_group.tools._h5_keys import h5_key  # noqa: E402

N_LEVELS = 10
# climatology builder hardcodes (H=64, W=128) for per-doy buffers. Stats builder
# is shape-agnostic, but using the production shape keeps both tests on the same
# fixture. Per-frame size = 64*128*4 = 32 KB; 10 frames * 59 keys = ~19 MB total
# — fast enough for the test.
H, W = 64, 128

ALL_KEYS: list[str] = []
for var in ("ta", "ua", "va", "hus"):
    for i in range(N_LEVELS):
        ALL_KEYS.append(h5_key(var, i))
for i in range(N_LEVELS):
    ALL_KEYS.append(h5_key("zg", i))
ALL_KEYS.extend(("pl", "tas", "pr_6h", "lsm", "sg", "z0", "sst", "rsdt", "sic"))


def _write_per_timestep(path: Path, value: float) -> None:
    """Write a single-timestep h5 with `value` filling every key."""
    with h5py.File(path, "w") as f:
        grp = f.create_group("input")
        for k in ALL_KEYS:
            arr = np.full((H, W), value, dtype=np.float32)
            grp.create_dataset(k, data=arr)


def _make_v10_src_h5(path: Path, lat: np.ndarray, lon: np.ndarray) -> None:
    """Stats builders read lat/lon from a v10 source path referenced in manifest.
    Build a minimal stub h5 with `/lat` and `/lon` datasets.
    """
    with h5py.File(path, "w") as f:
        f.create_dataset("lat", data=lat)
        f.create_dataset("lon", data=lon)


def _build_fixture(
    tmp_path: Path, *, n_native: int, n_padded: int,
    val_native: float, val_padded: float,
) -> Path:
    """Build a fixture data_dir with 1 year (year 12), n_native + (n_padded-n_native)
    files. Manifest carries n_timesteps_native=n_native, n_timesteps_padded=n_padded.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Native frames (year 12 idx 0..n_native-1) with val_native
    for idx in range(n_native):
        _write_per_timestep(data_dir / f"12_{idx:04d}.h5", val_native)
    # Padded frames with val_padded
    for idx in range(n_native, n_padded):
        _write_per_timestep(data_dir / f"12_{idx:04d}.h5", val_padded)

    # Stub v10 source for lat/lon.
    src_h5 = tmp_path / "MOST.0012.h5"
    _make_v10_src_h5(src_h5, lat=np.linspace(-89, 89, H), lon=np.linspace(0, 357, W))

    # Manifest with both n_timesteps_native AND n_timesteps_padded -> simulates
    # post-F.B2 state. The builder must skip padded indices.
    manifest = {
        "calendar": "proleptic_gregorian", "has_year_zero": True,
        "data_timedelta_hours": 6, "max_forecast_lead_steps": 60,
        "src_root": "/fake", "dst": str(data_dir),
        "expected_state_channels": [], "expected_diagnostic_channels": [],
        "expected_forcing_channels": [],
        "sigma_levels_pl_native": [], "zg_levels_pa": [],
        "years": [{
            "year": 12,
            "n_timesteps": n_padded,
            "n_timesteps_native": n_native,
            "n_timesteps_padded": n_padded,
            "is_leap": True,
            "src_path": str(src_h5),
        }],
    }
    (data_dir / "_v10_calendar_manifest.json").write_text(json.dumps(manifest))
    return data_dir


def test_stats_accumulator_skips_padded_frames(tmp_path: Path) -> None:
    """_accumulate must ingest exactly n_timesteps_native frames per year."""
    from sfno_training_group.tools import build_group_stats_netcdf as mod
    data_dir = _build_fixture(
        tmp_path,
        n_native=10, n_padded=15,         # 10 native + 5 padded for fast test
        val_native=42.0, val_padded=999.0,
    )
    accs, provenance = mod._accumulate(data_dir, [12])
    # Per key: count == n_native * H * W.
    expected_count = 10 * H * W
    for k, a in accs.items():
        assert a["count"] == expected_count, (
            f"key {k!r}: count={a['count']} != expected {expected_count} — "
            f"likely padded frames were ingested")
        # Per-key mean: 42.0 (val_native), NOT contaminated by 999.
        mean = a["sum"] / a["count"]
        assert abs(mean - 42.0) < 1e-3, f"key {k}: mean={mean} (contaminated?)"
    # Provenance.
    assert provenance == {
        "native_timesteps_by_year": {"12": 10},
        "native_timesteps_total": 10,
        "manifest_skip_padded_used": True,
        "built_before_padding": False,   # manifest had n_timesteps_padded set
    }


def test_stats_accumulator_no_manifest_falls_through(tmp_path: Path) -> None:
    """If manifest absent, treat all files as native (legacy / smoke contract)."""
    from sfno_training_group.tools import build_group_stats_netcdf as mod
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for idx in range(10):
        _write_per_timestep(data_dir / f"12_{idx:04d}.h5", 5.0)
    # NO manifest file written.
    accs, provenance = mod._accumulate(data_dir, [12])
    assert all(a["count"] == 10 * H * W for a in accs.values())
    assert provenance["built_before_padding"] is True
    assert provenance["manifest_skip_padded_used"] is False


def test_provenance_attrs_attached_to_xr_dataset(tmp_path: Path) -> None:
    from sfno_training_group.tools import build_group_stats_netcdf as mod
    import xarray as xr

    provenance = {
        "native_timesteps_by_year": {"12": 1455},
        "native_timesteps_total": 1455,
        "manifest_skip_padded_used": True,
        "built_before_padding": False,
    }
    ds = xr.Dataset()
    mod._attach_provenance(ds, provenance)
    assert ds.attrs["source"] == "recomputed_from_converted_h5"
    assert ds.attrs["native_timesteps_total"] == 1455
    assert json.loads(ds.attrs["native_timesteps_by_year"]) == {"12": 1455}
    assert int(ds.attrs["manifest_skip_padded_used"]) == 1
    assert int(ds.attrs["built_before_padding"]) == 0
    assert ds.attrs["git_sha"]
    assert ds.attrs["build_timestamp_utc"]


def test_climatology_accumulator_skips_padded_frames(tmp_path: Path) -> None:
    from sfno_training_group.tools import build_group_climatology as mod
    data_dir = _build_fixture(
        tmp_path,
        n_native=10, n_padded=15,
        val_native=7.0, val_padded=999.0,
    )
    sums, counts, provenance = mod._accumulate(data_dir, [12])
    total_count = int(counts.sum())
    assert total_count == 10, f"counts total {total_count} != 10 (padded frames ingested?)"
    # Each per-key sum should reflect val_native * H * W * counts[doy].
    # Spot check the ta_0 key.
    key = h5_key("ta", 0)
    expected_sum_total = 10 * H * W * 7.0
    actual_sum_total = sums[key].sum()
    assert abs(actual_sum_total - expected_sum_total) < 1e-3, (
        f"ta_0 sum total {actual_sum_total} != expected {expected_sum_total} "
        f"(contamination?)"
    )
    assert provenance["manifest_skip_padded_used"] is True
    assert provenance["built_before_padding"] is False
