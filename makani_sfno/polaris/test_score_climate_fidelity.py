"""Tests for score_climate_fidelity.py's area-weighted bias/RMSE math, and
(2026-10-03 review) its contract-enforcement paths -- channel-order mismatch,
masked/fill-value cells, and grid mismatch must all be hard errors, and were
previously untested (main() was never called by any test)."""
from __future__ import annotations

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from score_climate_fidelity import area_weighted_bias_rmse, main  # noqa: E402


def _weights(nlat):
    lat = 90.0 - (np.arange(nlat) + 0.5) * 180.0 / nlat
    w = np.cos(np.deg2rad(lat))
    return w / w.sum()


def test_identical_fields_are_zero():
    rng = np.random.default_rng(0)
    true = rng.standard_normal((3, 8, 16))
    bias, rmse = area_weighted_bias_rmse(true.copy(), true, _weights(8))
    assert np.allclose(bias, 0.0)
    assert np.allclose(rmse, 0.0)


def test_uniform_offset_is_exact_bias_and_rmse():
    true = np.zeros((2, 8, 16))
    member = true + np.array([3.0, -2.0])[:, None, None]
    weights = _weights(8)
    bias, rmse = area_weighted_bias_rmse(member, true, weights)
    assert np.allclose(bias, [3.0, -2.0])
    assert np.allclose(rmse, [3.0, 2.0])  # constant offset: RMSE == |bias|


def test_area_weighting_upweights_the_pole_row_less():
    # A difference concentrated at the (low cos-lat weight) first row should
    # bias less than the same-magnitude difference concentrated at the equator.
    nlat = 8
    weights = _weights(nlat)
    true = np.zeros((1, nlat, 4))
    pole = true.copy()
    pole[0, 0, :] = 10.0
    equator = true.copy()
    equator[0, nlat // 2, :] = 10.0
    bias_pole, _ = area_weighted_bias_rmse(pole, true, weights)
    bias_eq, _ = area_weighted_bias_rmse(equator, true, weights)
    assert bias_pole[0] < bias_eq[0]
    assert weights[0] < weights[nlat // 2]  # sanity: pole row is lower-weighted


def test_bias_can_cancel_while_rmse_cannot():
    # A field that's +X over half the globe and -X over the other half has
    # zero area-mean bias but nonzero pattern RMSE -- bias alone would hide it.
    nlat, nlon = 8, 4
    weights = np.full(nlat, 1.0 / nlat)  # uniform weights for an exact 50/50 split
    true = np.zeros((1, nlat, nlon))
    member = true.copy()
    member[0, : nlat // 2, :] = 5.0
    member[0, nlat // 2:, :] = -5.0
    bias, rmse = area_weighted_bias_rmse(member, true, weights)
    assert np.allclose(bias, 0.0)
    assert np.allclose(rmse, 5.0)


# ---------------------------------------------------------------------------
# main() contract tests (2026-10-03 review: these paths were never exercised)
# ---------------------------------------------------------------------------

_NAMES = ["a", "b", "c"]
_NLAT, _NLON = 8, 4


def _write_true_npz(path, *, names=_NAMES, nlat=_NLAT, nlon=_NLON, seed=0):
    rng = np.random.default_rng(seed)
    time_mean = rng.standard_normal((len(names), nlat, nlon))
    lat = 90.0 - (np.arange(nlat) + 0.5) * 180.0 / nlat
    lon = np.arange(nlon) * 360.0 / nlon
    std = 2.0 + rng.random(len(names))
    np.savez(path, time_mean=time_mean, channel_names=np.array(names, dtype=object),
             global_means=np.zeros(len(names)), global_stds=std,
             lat=lat, lon=lon, years=np.array([2045]), n_samples=100)
    return time_mean, lat, lon, std


def _write_member_nc(path, *, time_mean, lat, lon, names=_NAMES):
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("channel", len(names))
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        v = ds.createVariable("channel", str, ("channel",))
        v[:] = np.array(names, dtype=object)
        ds.createVariable("lat", "f8", ("lat",))[:] = lat
        ds.createVariable("lon", "f8", ("lon",))[:] = lon
        ds.createVariable("time_mean", "f8", ("channel", "lat", "lon"))[:] = time_mean


def test_main_matching_member_scores_cleanly(tmp_path, capsys):
    true_mean, lat, lon, _ = _write_true_npz(tmp_path / "true.npz")
    member = true_mean + 1.0  # a known, uniform-per-channel offset
    _write_member_nc(tmp_path / "m00.nc", time_mean=member, lat=lat, lon=lon)
    out = tmp_path / "fidelity.csv"

    rc = main(["--true", str(tmp_path / "true.npz"), "--member", str(tmp_path / "m00.nc"),
              "--out", str(out)])
    assert rc == 0
    assert "CLIMATE_FIDELITY_OK" in capsys.readouterr().out
    rows = out.read_text().splitlines()
    assert len(rows) == 1 + len(_NAMES)  # header + one row per channel


def test_main_refuses_channel_order_mismatch(tmp_path):
    true_mean, lat, lon, _ = _write_true_npz(tmp_path / "true.npz")
    _write_member_nc(tmp_path / "m00.nc", time_mean=true_mean, lat=lat, lon=lon,
                     names=list(reversed(_NAMES)))

    with pytest.raises(ValueError, match="CHANNEL_MISMATCH"):
        main(["--true", str(tmp_path / "true.npz"), "--member", str(tmp_path / "m00.nc"),
             "--out", str(tmp_path / "out.csv")])


def test_main_refuses_masked_fill_value_cells(tmp_path):
    true_mean, lat, lon, _ = _write_true_npz(tmp_path / "true.npz")
    member = true_mean.copy()
    with netCDF4.Dataset(tmp_path / "m00.nc", "w", format="NETCDF4") as ds:
        ds.createDimension("channel", len(_NAMES))
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        v = ds.createVariable("channel", str, ("channel",))
        v[:] = np.array(_NAMES, dtype=object)
        ds.createVariable("lat", "f8", ("lat",))[:] = lat
        ds.createVariable("lon", "f8", ("lon",))[:] = lon
        tm = ds.createVariable("time_mean", "f8", ("channel", "lat", "lon"), fill_value=9.969e36)
        tm[:] = member
        tm[0, 0, 0] = np.ma.masked  # one truncated/blown-up cell

    with pytest.raises(ValueError, match="NON_FINITE_TIME_MEAN"):
        main(["--true", str(tmp_path / "true.npz"), "--member", str(tmp_path / "m00.nc"),
             "--out", str(tmp_path / "out.csv")])


def test_main_refuses_grid_mismatch(tmp_path):
    true_mean, lat, lon, _ = _write_true_npz(tmp_path / "true.npz")
    _write_member_nc(tmp_path / "m00.nc", time_mean=true_mean, lat=lat[::-1], lon=lon)

    with pytest.raises(ValueError, match="GRID_MISMATCH"):
        main(["--true", str(tmp_path / "true.npz"), "--member", str(tmp_path / "m00.nc"),
             "--out", str(tmp_path / "out.csv")])
