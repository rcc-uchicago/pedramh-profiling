"""Channel list ordering + names."""

from __future__ import annotations

import numpy as np

from plasim_makani_packager.channels import (
    DIAGNOSTIC_CHANNELS,
    FORCING_CHANNELS,
    STATE_CHANNELS,
    TARGET_CHANNELS,
)
from plasim_makani_packager.packager import (
    _stack_fields_diagnostic,
    _stack_fields_state,
    _stack_forcing,
)


def test_state_order():
    assert len(STATE_CHANNELS) == 52
    assert STATE_CHANNELS[0] == "pl"
    assert STATE_CHANNELS[1] == "tas"
    # Sigma ordering: ta1 (TOA) at index 2, ta10 (surface) at index 11
    assert STATE_CHANNELS[2] == "ta1"
    assert STATE_CHANNELS[11] == "ta10"
    assert STATE_CHANNELS[12] == "ua1"
    assert STATE_CHANNELS[21] == "ua10"
    assert STATE_CHANNELS[22] == "va1"
    assert STATE_CHANNELS[31] == "va10"
    assert STATE_CHANNELS[32] == "hus1"
    assert STATE_CHANNELS[41] == "hus10"
    assert STATE_CHANNELS[42] == "zg1"
    assert STATE_CHANNELS[51] == "zg10"


def test_diagnostic():
    assert DIAGNOSTIC_CHANNELS == ["pr_6h"]


def test_forcing_order():
    # Static first, then varying. Matches adaptor output contract.
    assert FORCING_CHANNELS == ["lsm", "sg", "z0", "sst", "rsdt", "sic"]


def test_target_equals_state_then_diagnostic():
    assert TARGET_CHANNELS == STATE_CHANNELS + DIAGNOSTIC_CHANNELS
    assert len(TARGET_CHANNELS) == 53


def _fake_most(T: int = 3, H: int = 4, W: int = 8):
    """Minimal xarray-like dataset built from numpy arrays (stack helpers
    only access .values / .sizes)."""
    import xarray as xr  # lazy import; test works without torch/makani

    lev = np.linspace(0.05, 0.98, 10, dtype=np.float32)
    time = np.arange(T, dtype=np.float64)
    lat = np.linspace(-80, 80, H, dtype=np.float64)
    lon = np.linspace(0, 360 - 1, W, dtype=np.float64)

    rng = np.random.default_rng(0)
    data_vars = {
        "pl": (("time", "lat", "lon"), rng.standard_normal((T, H, W), dtype=np.float32)),
        "tas": (("time", "lat", "lon"), rng.standard_normal((T, H, W), dtype=np.float32)),
        "pr_6h": (("time", "lat", "lon"), rng.standard_normal((T, H, W), dtype=np.float32)),
        "lsm": (("time", "lat", "lon"), rng.standard_normal((T, H, W), dtype=np.float32)),
        "sg": (("time", "lat", "lon"), rng.standard_normal((T, H, W), dtype=np.float32)),
        "z0": (("time", "lat", "lon"), rng.standard_normal((T, H, W), dtype=np.float32)),
        "sic": (("time", "lat", "lon"), np.clip(rng.random((T, H, W), dtype=np.float32), 0, 1)),
    }
    for v in ("ta", "ua", "va", "hus", "zg"):
        data_vars[v] = (
            ("time", "lev", "lat", "lon"),
            rng.standard_normal((T, 10, H, W), dtype=np.float32),
        )
    return xr.Dataset(
        data_vars,
        coords={"time": time, "lev": lev, "lat": lat, "lon": lon},
    )


def _fake_boundary(most_ds):
    import xarray as xr

    T, H, W = most_ds.sizes["time"], most_ds.sizes["lat"], most_ds.sizes["lon"]
    rng = np.random.default_rng(1)
    sst = rng.uniform(272, 305, size=(T, H, W)).astype(np.float32)
    # Punch ~20% land NaNs
    mask = rng.random((T, H, W)) < 0.2
    sst[mask] = np.nan
    return xr.Dataset(
        {
            "sst": (("time", "lat", "lon"), sst),
            "rsdt": (("time", "lat", "lon"), rng.uniform(0, 1400, size=(T, H, W)).astype(np.float32)),
            "sic": (("time", "lat", "lon"), most_ds["sic"].values.copy()),  # clip-identity
        },
        coords=most_ds.coords,
        attrs={"rsdt_method": "astronomical"},
    )


def test_stack_fields_state_shape_and_order():
    ds = _fake_most(T=3, H=4, W=8)
    arr = _stack_fields_state(ds)
    assert arr.shape == (3, 52, 4, 8)
    assert arr.dtype == np.float32
    # Channel 0 must be pl, channel 1 must be tas
    np.testing.assert_array_equal(arr[:, 0], ds["pl"].values)
    np.testing.assert_array_equal(arr[:, 1], ds["tas"].values)
    # ta1 at index 2 == lev[0] (TOA)
    np.testing.assert_array_equal(arr[:, 2], ds["ta"].values[:, 0])
    # ta10 at index 11 == lev[9] (surface)
    np.testing.assert_array_equal(arr[:, 11], ds["ta"].values[:, 9])
    # zg1 at index 42 == zg lev[0]
    np.testing.assert_array_equal(arr[:, 42], ds["zg"].values[:, 0])


def test_stack_fields_diagnostic():
    ds = _fake_most(T=3, H=4, W=8)
    arr = _stack_fields_diagnostic(ds)
    assert arr.shape == (3, 1, 4, 8)
    np.testing.assert_array_equal(arr[:, 0], ds["pr_6h"].values)


def test_stack_forcing_sst_land_fill_applied():
    most = _fake_most(T=2, H=3, W=4)
    bnd = _fake_boundary(most)
    forcing, sst_land_frac = _stack_forcing(most, bnd, sst_land_fill_k=271.35)
    assert forcing.shape == (2, 6, 3, 4)
    # Channels 0,1,2 passthrough from MOST
    np.testing.assert_array_equal(forcing[:, 0], most["lsm"].values)
    np.testing.assert_array_equal(forcing[:, 1], most["sg"].values)
    np.testing.assert_array_equal(forcing[:, 2], most["z0"].values)
    # Channel 3: sst, NaN filled with 271.35
    assert not np.isnan(forcing[:, 3]).any()
    nan_mask = np.isnan(bnd["sst"].values)
    assert np.all(forcing[:, 3][nan_mask] == np.float32(271.35))
    # Channel 5: sic passthrough (clip identity)
    np.testing.assert_array_equal(forcing[:, 5], bnd["sic"].values)
    # Fraction must match the NaN fraction
    assert sst_land_frac == float(nan_mask.sum()) / float(nan_mask.size)
