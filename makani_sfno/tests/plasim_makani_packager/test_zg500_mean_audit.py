"""Pin the v10 L7d audit (zg500 global mean must lie in [5400, 5700] m).

Two layers (per docs/plasim_zg_plev_migration_plan.md §3.6, §3.7):

  - ``stats._audit_zg500_inline``: runs against ``mean_tgt`` *before* any
    .npy is written, so a bad pack never produces stats artifacts.

  - ``validate._validate_zg500_saved_mean``: defense in depth — re-loads
    ``stats/global_means.npy`` and re-checks the same range, in case the
    .npy was hand-edited or arrived from elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plasim_makani_packager.channels import TARGET_CHANNELS
from plasim_makani_packager.stats import (
    ZG500_AUDIT_RANGE_M,
    _audit_zg500_inline,
)
from plasim_makani_packager.validate import (
    ValidationError,
    _validate_zg500_saved_mean,
)


def _zg500_idx() -> int:
    return TARGET_CHANNELS.index("zg500")


# ---------------------------------------------------------------------------
# Inline audit (mean_tgt as a (53,) array)
# ---------------------------------------------------------------------------
def test_inline_audit_passes_inside_band():
    mean_tgt = np.zeros(53, dtype=np.float64)
    mean_tgt[_zg500_idx()] = 5550.0
    _audit_zg500_inline(mean_tgt)


def test_inline_audit_fails_below_band():
    mean_tgt = np.zeros(53, dtype=np.float64)
    mean_tgt[_zg500_idx()] = 4000.0
    with pytest.raises(RuntimeError) as exc_info:
        _audit_zg500_inline(mean_tgt)
    msg = str(exc_info.value)
    assert "zg500" in msg
    assert "4000" in msg
    assert "L7d" in msg


def test_inline_audit_fails_above_band():
    mean_tgt = np.zeros(53, dtype=np.float64)
    mean_tgt[_zg500_idx()] = 7000.0
    with pytest.raises(RuntimeError) as exc_info:
        _audit_zg500_inline(mean_tgt)
    assert "7000" in str(exc_info.value)


def test_inline_audit_band_edges_inclusive():
    lo, hi = ZG500_AUDIT_RANGE_M
    mean_tgt = np.zeros(53, dtype=np.float64)
    mean_tgt[_zg500_idx()] = lo
    _audit_zg500_inline(mean_tgt)  # must not raise
    mean_tgt[_zg500_idx()] = hi
    _audit_zg500_inline(mean_tgt)  # must not raise


# ---------------------------------------------------------------------------
# Defense-in-depth audit on the saved global_means.npy
# ---------------------------------------------------------------------------
def _write_fake_means(out_root: Path, zg500_value_m: float) -> None:
    means = np.zeros(53, dtype=np.float32)
    means[_zg500_idx()] = float(zg500_value_m)
    stats_dir = out_root / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    np.save(stats_dir / "global_means.npy", means.reshape(1, 53, 1, 1))


def test_saved_mean_audit_passes(tmp_path: Path):
    _write_fake_means(tmp_path, 5550.0)
    _validate_zg500_saved_mean(tmp_path)


def test_saved_mean_audit_fails_outside_band(tmp_path: Path):
    _write_fake_means(tmp_path, 4000.0)
    with pytest.raises(ValidationError) as exc_info:
        _validate_zg500_saved_mean(tmp_path)
    assert "zg500" in str(exc_info.value)


def test_saved_mean_audit_missing_file_errors(tmp_path: Path):
    with pytest.raises(ValidationError):
        _validate_zg500_saved_mean(tmp_path)
