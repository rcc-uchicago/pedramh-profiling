"""Unit tests for the BYO-IC NetCDF reader (sfno_inference_5410.byo_ic).

Tier 1: schema validation. No GPU, no upstream model — pure xarray/numpy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from sfno_inference_5410.byo_ic import (
    EXPECTED_LAT,
    EXPECTED_LEV,
    EXPECTED_LON,
    EXPECTED_PLEV,
    SURFACE_VARS,
    UPPER_AIR_VARS,
    DIAGNOSTIC_VARS,
    ZG_VAR,
    stack_for_model,
    validate_byo_ic,
)


def _make_valid_ic(tmp_path: Path, with_time: bool = False) -> Path:
    """Build a minimum-viable valid IC NetCDF and return its path."""
    rng = np.random.default_rng(0)
    lat = np.linspace(-87.86, 87.86, EXPECTED_LAT)
    lon = np.linspace(0, 357.1875, EXPECTED_LON)
    lev = np.linspace(0.05, 0.98, EXPECTED_LEV)
    plev = np.array([20000, 25000, 30000, 40000, 50000, 60000, 70000, 85000, 92500, 100000])

    surf_shape = (EXPECTED_LAT, EXPECTED_LON)
    ua_shape = (EXPECTED_LEV, EXPECTED_LAT, EXPECTED_LON)
    zg_shape = (EXPECTED_PLEV, EXPECTED_LAT, EXPECTED_LON)

    if with_time:
        surf_dims = ("time", "lat", "lon")
        ua_dims = ("time", "lev", "lat", "lon")
        zg_dims = ("time", "plev", "lat", "lon")
        surf_shape = (1, *surf_shape)
        ua_shape = (1, *ua_shape)
        zg_shape = (1, *zg_shape)
    else:
        surf_dims = ("lat", "lon")
        ua_dims = ("lev", "lat", "lon")
        zg_dims = ("plev", "lat", "lon")

    data_vars = {}
    for v in (*SURFACE_VARS, *DIAGNOSTIC_VARS):
        data_vars[v] = (surf_dims, rng.standard_normal(surf_shape).astype(np.float32))
    for v in UPPER_AIR_VARS:
        data_vars[v] = (ua_dims, rng.standard_normal(ua_shape).astype(np.float32))
    data_vars[ZG_VAR] = (zg_dims, rng.standard_normal(zg_shape).astype(np.float32))

    coords = {"lat": lat, "lon": lon, "lev": lev, "plev": plev}
    if with_time:
        coords["time"] = [np.datetime64("0001-01-01")]

    ds = xr.Dataset(data_vars, coords=coords)
    out = tmp_path / ("ic_with_time.nc" if with_time else "ic_no_time.nc")
    ds.to_netcdf(out)
    return out


def test_validate_byo_ic_happy_no_time(tmp_path):
    nc = _make_valid_ic(tmp_path, with_time=False)
    out = validate_byo_ic(nc)
    assert set(out.keys()) == set((*SURFACE_VARS, *UPPER_AIR_VARS, *DIAGNOSTIC_VARS, ZG_VAR))
    for v in (*SURFACE_VARS, *DIAGNOSTIC_VARS):
        assert out[v].shape == (EXPECTED_LAT, EXPECTED_LON)
        assert out[v].dtype == np.float32
    for v in UPPER_AIR_VARS:
        assert out[v].shape == (EXPECTED_LEV, EXPECTED_LAT, EXPECTED_LON)
    assert out[ZG_VAR].shape == (EXPECTED_PLEV, EXPECTED_LAT, EXPECTED_LON)


def test_validate_byo_ic_happy_with_time(tmp_path):
    nc = _make_valid_ic(tmp_path, with_time=True)
    out = validate_byo_ic(nc)
    # Time dim should have been squeezed away.
    for v in (*SURFACE_VARS, *DIAGNOSTIC_VARS):
        assert out[v].shape == (EXPECTED_LAT, EXPECTED_LON)
    for v in UPPER_AIR_VARS:
        assert out[v].shape == (EXPECTED_LEV, EXPECTED_LAT, EXPECTED_LON)


def test_validate_byo_ic_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        validate_byo_ic(tmp_path / "no_such.nc")


def test_validate_byo_ic_missing_variable(tmp_path):
    nc = _make_valid_ic(tmp_path, with_time=False)
    with xr.open_dataset(nc) as ds:
        ds = ds.drop_vars(["zg"])
    nc2 = tmp_path / "no_zg.nc"
    ds.to_netcdf(nc2)
    with pytest.raises(ValueError, match="missing required variables.*zg"):
        validate_byo_ic(nc2)


def test_validate_byo_ic_wrong_grid(tmp_path):
    rng = np.random.default_rng(1)
    bad_lat = 32  # wrong
    lat = np.linspace(-87.86, 87.86, bad_lat)
    lon = np.linspace(0, 357.1875, EXPECTED_LON)
    lev = np.linspace(0.05, 0.98, EXPECTED_LEV)
    plev = np.array([20000, 25000, 30000, 40000, 50000, 60000, 70000, 85000, 92500, 100000])

    data_vars = {}
    for v in (*SURFACE_VARS, *DIAGNOSTIC_VARS):
        data_vars[v] = (("lat", "lon"), rng.standard_normal((bad_lat, EXPECTED_LON)).astype(np.float32))
    for v in UPPER_AIR_VARS:
        data_vars[v] = (("lev", "lat", "lon"), rng.standard_normal((EXPECTED_LEV, bad_lat, EXPECTED_LON)).astype(np.float32))
    data_vars[ZG_VAR] = (("plev", "lat", "lon"), rng.standard_normal((EXPECTED_PLEV, bad_lat, EXPECTED_LON)).astype(np.float32))

    ds = xr.Dataset(data_vars, coords={"lat": lat, "lon": lon, "lev": lev, "plev": plev})
    nc = tmp_path / "bad_grid.nc"
    ds.to_netcdf(nc)
    with pytest.raises(ValueError, match="dim lat has size 32"):
        validate_byo_ic(nc)


def test_validate_byo_ic_multi_timestep_rejected(tmp_path):
    rng = np.random.default_rng(2)
    n_time = 3
    lat = np.linspace(-87.86, 87.86, EXPECTED_LAT)
    lon = np.linspace(0, 357.1875, EXPECTED_LON)
    lev = np.linspace(0.05, 0.98, EXPECTED_LEV)
    plev = np.array([20000, 25000, 30000, 40000, 50000, 60000, 70000, 85000, 92500, 100000])

    data_vars = {
        "pl": (("time", "lat", "lon"), rng.standard_normal((n_time, EXPECTED_LAT, EXPECTED_LON)).astype(np.float32)),
    }
    # Add the rest as no-time so only pl trips the multi-time check first.
    for v in ("tas", "pr_6h"):
        data_vars[v] = (("lat", "lon"), rng.standard_normal((EXPECTED_LAT, EXPECTED_LON)).astype(np.float32))
    for v in UPPER_AIR_VARS:
        data_vars[v] = (("lev", "lat", "lon"), rng.standard_normal((EXPECTED_LEV, EXPECTED_LAT, EXPECTED_LON)).astype(np.float32))
    data_vars[ZG_VAR] = (("plev", "lat", "lon"), rng.standard_normal((EXPECTED_PLEV, EXPECTED_LAT, EXPECTED_LON)).astype(np.float32))

    ds = xr.Dataset(
        data_vars,
        coords={
            "time": np.arange(n_time, dtype=np.int32),
            "lat": lat, "lon": lon, "lev": lev, "plev": plev,
        },
    )
    nc = tmp_path / "multi_time.nc"
    ds.to_netcdf(nc)
    with pytest.raises(ValueError, match="time dim has size 3"):
        validate_byo_ic(nc)


def test_validate_byo_ic_wrong_lev_count(tmp_path):
    rng = np.random.default_rng(3)
    bad_lev = 5
    lat = np.linspace(-87.86, 87.86, EXPECTED_LAT)
    lon = np.linspace(0, 357.1875, EXPECTED_LON)
    lev = np.linspace(0.05, 0.98, bad_lev)
    plev = np.array([20000, 25000, 30000, 40000, 50000, 60000, 70000, 85000, 92500, 100000])
    data_vars = {}
    for v in (*SURFACE_VARS, *DIAGNOSTIC_VARS):
        data_vars[v] = (("lat", "lon"), rng.standard_normal((EXPECTED_LAT, EXPECTED_LON)).astype(np.float32))
    for v in UPPER_AIR_VARS:
        data_vars[v] = (("lev", "lat", "lon"), rng.standard_normal((bad_lev, EXPECTED_LAT, EXPECTED_LON)).astype(np.float32))
    data_vars[ZG_VAR] = (("plev", "lat", "lon"), rng.standard_normal((EXPECTED_PLEV, EXPECTED_LAT, EXPECTED_LON)).astype(np.float32))
    ds = xr.Dataset(data_vars, coords={"lat": lat, "lon": lon, "lev": lev, "plev": plev})
    nc = tmp_path / "bad_lev.nc"
    ds.to_netcdf(nc)
    with pytest.raises(ValueError, match="dim 'lev' has size 5"):
        validate_byo_ic(nc)


def test_stack_for_model_orders_match_caller(tmp_path):
    nc = _make_valid_ic(tmp_path, with_time=False)
    raw = validate_byo_ic(nc)
    surf, ua = stack_for_model(raw, ("pl", "tas"), ("ta", "ua", "va", "hus", "zg"))
    assert surf.shape == (2, EXPECTED_LAT, EXPECTED_LON)
    assert ua.shape == (5, EXPECTED_LEV, EXPECTED_LAT, EXPECTED_LON)
    # First-channel sanity: stacking should preserve the underlying values.
    np.testing.assert_array_equal(surf[0], raw["pl"])
    np.testing.assert_array_equal(ua[4], raw["zg"])
