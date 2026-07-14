"""T.5 + T.10 — converter unit tests for the long_inference output -> 53-channel scorer NC.

T.5: assert channel mapping, dim names, lead range; reads from a synthetic
     long_inference-schema NetCDF + a synthetic MOST.0YYY.h5.
T.10: assert no NaN/Inf in prediction/truth over leads 1..60 when input is finite.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cftime
import h5py
import numpy as np
import pytest
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "convert_group_long_inference_to_aires_nc.py"


# ---------- fixtures ----------


def _build_synthetic_long_inference_nc(
    out_path: Path, *, year: int, n_time: int, H: int = 64, W: int = 128,
    n_lev: int = 10,
) -> None:
    """Fabricate a NetCDF in convert_ensemble_to_xarray's schema."""
    time_coord = xr.date_range(
        cftime.DatetimeProlepticGregorian(year, 1, 1, has_year_zero=True),
        cftime.DatetimeProlepticGregorian(year + 1, 1, 1, has_year_zero=True),
        freq="6h", inclusive="left", calendar="proleptic_gregorian", use_cftime=True,
    )[:n_time]

    rng = np.random.default_rng(seed=year)
    surface = lambda base: rng.normal(loc=base, scale=0.5, size=(1, n_time, H, W)).astype(np.float32)
    upper3d = lambda base: rng.normal(loc=base, scale=0.5, size=(1, n_time, n_lev, H, W)).astype(np.float32)

    ds = xr.Dataset(
        data_vars=dict(
            pl=(("ensemble_idx", "time", "lat", "lon"), surface(11.5)),       # ln(p_s) ~ ln(100k)
            tas=(("ensemble_idx", "time", "lat", "lon"), surface(280.0)),     # K
            pr_6h=(("ensemble_idx", "time", "lat", "lon"), surface(2.0)),     # mm/6h
            ta=(("ensemble_idx", "time", "lev", "lat", "lon"), upper3d(250.0)),
            ua=(("ensemble_idx", "time", "lev", "lat", "lon"), upper3d(10.0)),
            va=(("ensemble_idx", "time", "lev", "lat", "lon"), upper3d(0.0)),
            hus=(("ensemble_idx", "time", "lev", "lat", "lon"), upper3d(0.005)),
            zg=(("ensemble_idx", "time", "plev", "lat", "lon"), upper3d(5500.0)),
        ),
        coords=dict(
            ensemble_idx=("ensemble_idx", np.array([0])),
            time=("time", time_coord),
            lev=("lev", np.linspace(0.05, 0.98, n_lev)),
            plev=("plev", np.array([20000, 25000, 30000, 40000, 50000, 60000, 70000, 85000, 92500, 100000])),
            lat=("lat", np.linspace(-89, 89, H)),
            lon=("lon", np.linspace(0, 357, W)),
        ),
    )
    ds.to_netcdf(out_path)


def _build_synthetic_v10_test_h5(
    out_path: Path, *, n_t: int = 100, H: int = 64, W: int = 128,
) -> None:
    """Fabricate a v10 MOST.0YYY.h5 with enough fields for the converter's truth read."""
    rng = np.random.default_rng(seed=42)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("fields_state", data=rng.normal(size=(n_t, 52, H, W)).astype(np.float32))
        f.create_dataset("fields_diagnostic", data=rng.normal(size=(n_t, 1, H, W)).astype(np.float32))
        f.create_dataset("forcing", data=rng.normal(size=(n_t, 6, H, W)).astype(np.float32))
        f.create_dataset("lat", data=np.linspace(-89, 89, H).astype(np.float64))
        f.create_dataset("lon", data=np.linspace(0, 357, W).astype(np.float64))
        f.create_dataset("time_plasim", data=np.arange(n_t, dtype=np.float64))
        f.attrs["plasim_time_units"] = "days since 0121-01-01 00:00:00"


@pytest.fixture
def fixture_dir(tmp_path: Path) -> Path:
    """Build a fixture pair: long_inference NC (year 121, n_time=100) + MOST.0121.h5."""
    long_nc = tmp_path / "long_rollout_year121_member000_y0121.nc"
    _build_synthetic_long_inference_nc(long_nc, year=121, n_time=100)
    test_h5 = tmp_path / "MOST.0121.h5"
    _build_synthetic_v10_test_h5(test_h5, n_t=100)
    return tmp_path


def _run_converter(*, long_nc: Path, test_h5: Path, out: Path, max_leads: int = 60,
                   ic_global_idx: int = 0) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH),
         "--long-inference-nc", str(long_nc),
         "--test-h5", str(test_h5),
         "--ic-global-idx", str(ic_global_idx),
         "--max-output-leads", str(max_leads),
         "--out", str(out)],
        capture_output=True, text=True, env=env,
    )


# ---------- T.5 channel mapping / dims ----------


def test_converter_writes_53_channel_nc(fixture_dir: Path) -> None:
    out = fixture_dir / "MOST.0121_ic000.nc"
    res = _run_converter(
        long_nc=fixture_dir / "long_rollout_year121_member000_y0121.nc",
        test_h5=fixture_dir / "MOST.0121.h5",
        out=out, max_leads=10,
    )
    assert res.returncode == 0, f"converter failed:\nstdout={res.stdout}\nstderr={res.stderr}"
    assert out.is_file()

    ds = xr.open_dataset(out)
    assert ds["prediction"].dims == ("init_time", "lead_time", "channel", "lat", "lon")
    assert ds["truth"].dims == ("init_time", "lead_time", "channel", "lat", "lon")
    assert ds["init_state"].dims == ("init_time", "channel_ic", "lat", "lon")
    assert ds.sizes["channel"] == 53
    assert ds.sizes["channel_ic"] == 52
    assert ds.sizes["lead_time"] == 10
    # Channel ordering matches v10 contract.
    expected_channels = (
        ["pl", "tas"]
        + [f"ta{i+1}" for i in range(10)]
        + [f"ua{i+1}" for i in range(10)]
        + [f"va{i+1}" for i in range(10)]
        + [f"hus{i+1}" for i in range(10)]
        + [f"zg{p}" for p in (200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)]
        + ["pr_6h"]
    )
    assert list(ds["channel"].values) == expected_channels
    ds.close()


def test_converter_lead_time_in_hours(fixture_dir: Path) -> None:
    out = fixture_dir / "out.nc"
    res = _run_converter(
        long_nc=fixture_dir / "long_rollout_year121_member000_y0121.nc",
        test_h5=fixture_dir / "MOST.0121.h5",
        out=out, max_leads=10,
    )
    assert res.returncode == 0
    ds = xr.open_dataset(out)
    np.testing.assert_array_equal(ds["lead_time"].values, np.arange(1, 11) * 6)
    assert ds["lead_time"].attrs["units"] == "hours"
    ds.close()


def test_converter_pl_channel_carries_long_inference_pl(fixture_dir: Path) -> None:
    """Channel 0 of `prediction` must come from long_inference's pl variable."""
    out = fixture_dir / "out.nc"
    _run_converter(
        long_nc=fixture_dir / "long_rollout_year121_member000_y0121.nc",
        test_h5=fixture_dir / "MOST.0121.h5",
        out=out, max_leads=5,
    )
    long_inf = xr.open_dataset(fixture_dir / "long_rollout_year121_member000_y0121.nc")
    out_ds = xr.open_dataset(out)
    # prediction[0, lead k, ch 0] should equal long_inf pl[ensemble=0, time=k+1]
    for k in range(5):
        np.testing.assert_array_equal(
            out_ds["prediction"].values[0, k, 0],
            long_inf["pl"].values[0, k + 1],
        )
    out_ds.close()
    long_inf.close()


# ---------- T.10 K=60 finite output ----------


def test_converter_K60_no_nan_or_inf(fixture_dir: Path) -> None:
    """Produce K=60 leads from finite synthetic input; assert no NaN/Inf."""
    # The fixture only has n_time=100, so K can be up to 99. K=60 fits.
    out = fixture_dir / "out_K60.nc"
    res = _run_converter(
        long_nc=fixture_dir / "long_rollout_year121_member000_y0121.nc",
        test_h5=fixture_dir / "MOST.0121.h5",
        out=out, max_leads=60,
    )
    assert res.returncode == 0, f"K=60 converter failed:\nstdout={res.stdout}\nstderr={res.stderr}"
    ds = xr.open_dataset(out)
    assert ds.sizes["lead_time"] == 60
    assert np.isfinite(ds["prediction"].values).all(), "prediction has NaN/Inf"
    assert np.isfinite(ds["truth"].values).all(), "truth has NaN/Inf"
    assert np.isfinite(ds["init_state"].values).all(), "init_state has NaN/Inf"
    ds.close()


def test_converter_truncates_when_max_leads_exceeds_input(fixture_dir: Path) -> None:
    """If user passes K > available leads, converter truncates to n_time-1."""
    # fixture has n_time=100; K=200 should truncate to 99.
    out = fixture_dir / "out_truncated.nc"
    res = _run_converter(
        long_nc=fixture_dir / "long_rollout_year121_member000_y0121.nc",
        test_h5=fixture_dir / "MOST.0121.h5",
        out=out, max_leads=200,
    )
    assert res.returncode == 0
    ds = xr.open_dataset(out)
    assert ds.sizes["lead_time"] == 99   # n_time(100) - 1
    ds.close()
