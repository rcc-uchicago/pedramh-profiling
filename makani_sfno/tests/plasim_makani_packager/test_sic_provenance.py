"""_validate_sic_clipping: hard-fail boundaries and quantify-only report.

Rules from plan v9 §3:
- Hard-fail: adaptor.sic != np.clip(MOST.sic, 0, 1) anywhere (|diff| > 1e-6),
  or shape mismatch, or NaN-mask parity broken.
- Quantify only (report): raw MOST-vs-adaptor stats and out-of-range fraction.
"""

from __future__ import annotations

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from plasim_makani_packager.packager import _validate_sic_clipping


def _make(T: int, H: int, W: int, most_sic: np.ndarray, adaptor_sic: np.ndarray):
    coords = {
        "time": np.arange(T, dtype=np.float64),
        "lat": np.arange(H, dtype=np.float64),
        "lon": np.arange(W, dtype=np.float64),
    }
    most_ds = xr.Dataset(
        {"sic": (("time", "lat", "lon"), most_sic.astype(np.float32))},
        coords=coords,
    )
    bnd_ds = xr.Dataset(
        {"sic": (("time", "lat", "lon"), adaptor_sic.astype(np.float32))},
        coords=coords,
    )
    return most_ds, bnd_ds


def test_passes_on_clip_identity():
    rng = np.random.default_rng(0)
    most = rng.random((2, 3, 4)).astype(np.float32)
    adaptor = np.clip(most, 0.0, 1.0).astype(np.float32)
    m, b = _make(2, 3, 4, most, adaptor)
    report = _validate_sic_clipping(m, b)
    assert report["max_abs_diff"] == pytest.approx(0.0, abs=1e-6)
    assert report["fraction_cells_changed_by_clip"] == 0.0


def test_passes_and_quantifies_when_plasim_out_of_range():
    # PlaSim emits marginally-out-of-range sic at cell edges; adaptor clips.
    most = np.array([[-0.01, 0.0, 0.5, 1.2]], dtype=np.float32)[None, :, :]  # (1,1,4)
    adaptor = np.clip(most, 0.0, 1.0).astype(np.float32)
    m, b = _make(1, 1, 4, most, adaptor)
    report = _validate_sic_clipping(m, b)
    assert report["adaptor_vs_clip_max_abs_diff"] < 1e-6
    assert report["max_abs_diff"] > 1e-3  # quantify shows the raw clip magnitude
    assert report["fraction_cells_changed_by_clip"] == 0.5


def test_hard_fail_on_extra_perturbation():
    # Adaptor does clip + drift. Must hard-fail.
    most = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)[None, :, :]
    adaptor = most + 1e-3  # adaptor altered the value, not just clipped
    m, b = _make(1, 1, 4, most, adaptor)
    with pytest.raises(RuntimeError, match="differs from np.clip"):
        _validate_sic_clipping(m, b)


def test_hard_fail_on_shape_mismatch():
    most = np.zeros((1, 2, 4), dtype=np.float32)
    adaptor = np.zeros((1, 2, 3), dtype=np.float32)  # mismatched
    with pytest.raises(RuntimeError, match="shape mismatch"):
        _validate_sic_clipping(
            xr.Dataset({"sic": (("time", "lat", "lon"), most)}),
            xr.Dataset({"sic": (("time", "lat", "lon"), adaptor)}),
        )


def test_hard_fail_on_nan_mask_mismatch():
    most = np.array([[0.1, 0.2, np.nan, 0.4]], dtype=np.float32)[None, :, :]
    adaptor = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)[None, :, :]
    m, b = _make(1, 1, 4, most, adaptor)
    with pytest.raises(RuntimeError, match="NaN-mask parity"):
        _validate_sic_clipping(m, b)
