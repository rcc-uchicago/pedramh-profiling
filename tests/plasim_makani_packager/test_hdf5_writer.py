"""End-to-end per-(sim, year) HDF5 write + structural validation.

Builds synthetic MOST.{YYYY}.nc + boundary.{YYYY}.nc, runs packager.process_one,
and asserts the written HDF5 has the expected datasets, dim scales, attrs,
and shapes. Also asserts validate.run_structural / validate._validate_file
pass on the output.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from plasim_makani_packager.channels import (
    DIAGNOSTIC_CHANNELS,
    FORCING_CHANNELS,
    STATE_CHANNELS,
)
from plasim_makani_packager.packager import process_one, STEP_SECONDS
from plasim_makani_packager.validate import _validate_file


# ---------------------------------------------------------------------------
# Synthetic fixture builders
# ---------------------------------------------------------------------------
def _make_most_file(path: Path, T: int, H: int = 64, W: int = 128) -> None:
    """Synthetic MOST file.

    lsm/sg/z0 are static across files (fixed seed) to match real PlaSim
    topography semantics — this lets validate.run_structural's static-
    invariance check on forcing_time_means pass after stats compute.
    """
    rng = np.random.default_rng(0)
    static_rng = np.random.default_rng(42)  # shared across test files
    lev = np.linspace(0.05, 0.98, 10, dtype=np.float32)
    time = np.arange(T, dtype=np.float64)
    lat = np.linspace(-80.0, 80.0, H, dtype=np.float64)
    lon = np.linspace(0.0, 357.1875, W, dtype=np.float64)

    lsm_static = (static_rng.random((H, W)) > 0.7).astype(np.float32)
    sg_static = static_rng.random((H, W)).astype(np.float32)
    z0_static = static_rng.random((H, W)).astype(np.float32)

    d: dict = {
        "pl": (("time", "lat", "lon"), rng.standard_normal((T, H, W)).astype(np.float32)),
        "tas": (("time", "lat", "lon"), (280 + 20 * rng.standard_normal((T, H, W))).astype(np.float32)),
        "pr_6h": (("time", "lat", "lon"), rng.random((T, H, W)).astype(np.float32)),
        "lsm": (("time", "lat", "lon"), np.broadcast_to(lsm_static, (T, H, W)).copy()),
        "sg": (("time", "lat", "lon"), np.broadcast_to(sg_static, (T, H, W)).copy()),
        "z0": (("time", "lat", "lon"), np.broadcast_to(z0_static, (T, H, W)).copy()),
        "sic": (("time", "lat", "lon"), np.clip(rng.random((T, H, W)).astype(np.float32), 0.0, 1.0)),
    }
    for v in ("ta", "ua", "va", "hus", "zg"):
        d[v] = (
            ("time", "lev", "lat", "lon"),
            rng.standard_normal((T, 10, H, W)).astype(np.float32),
        )
    ds = xr.Dataset(
        d, coords={"time": time, "lev": lev, "lat": lat, "lon": lon}
    )
    ds["time"].attrs = {
        "units": "days since 0006-08-25 00:00:00",
        "calendar": "proleptic_gregorian",
    }
    ds.to_netcdf(path)


def _make_boundary_file(path: Path, most_path: Path) -> None:
    src = xr.open_dataset(most_path, decode_times=False)
    try:
        T, H, W = src.sizes["time"], src.sizes["lat"], src.sizes["lon"]
        rng = np.random.default_rng(1)
        sst = (272 + 20 * rng.random((T, H, W))).astype(np.float32)
        land = src["lsm"].values > 0.5
        sst[land] = np.nan  # NaN-over-land convention
        rsdt = (rng.random((T, H, W)) * 1367.0).astype(np.float32)
        # sic is a clip-identity pass-through in this synthetic (no out-of-range input)
        sic = src["sic"].values.astype(np.float32).copy()

        out = xr.Dataset(
            {
                "sst": (("time", "lat", "lon"), sst),
                "rsdt": (("time", "lat", "lon"), rsdt),
                "sic": (("time", "lat", "lon"), sic),
            },
            coords=src.coords,
        )
        out.attrs["rsdt_method"] = "astronomical"
        out.to_netcdf(path)
    finally:
        src.close()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test_process_one_writes_valid_h5(tmp_path: Path):
    sim = 52
    year = 3  # first year of train split — offset = 0, no prior files needed
    postproc_root = tmp_path / "postproc"
    boundary_root = tmp_path / "boundary"
    output_root = tmp_path / "out"

    most_dir = postproc_root / f"sim{sim}"
    boundary_dir = boundary_root / f"sim{sim}"
    most_dir.mkdir(parents=True)
    boundary_dir.mkdir(parents=True)

    most_path = most_dir / f"MOST.{year:04d}.nc"
    boundary_path = boundary_dir / f"boundary.{year:04d}.nc"
    T = 5
    _make_most_file(most_path, T=T)
    _make_boundary_file(boundary_path, most_path)

    opts = Namespace(
        sims=[sim],
        train_years=[3, 100],
        valid_years=[101, 120],
        test_years=[121, 128],
        postproc_root=postproc_root,
        boundary_root=boundary_root,
        output_root=output_root,
        sst_land_fill_k=271.35,
        task_index=None,
        count_tasks=False,
        overwrite=False,
        dry_run=False,
        verbose=False,
    )

    process_one(sim, year, opts)

    out_path = output_root / "train" / f"MOST.{year:04d}.h5"
    assert out_path.exists()

    # structural validator passes
    assert _validate_file(out_path, year) == T

    # Spot-check datasets + dim scales attached correctly
    with h5py.File(out_path, "r") as f:
        assert f["fields_state"].shape == (T, 52, 64, 128)
        assert f["fields_diagnostic"].shape == (T, 1, 64, 128)
        assert f["forcing"].shape == (T, 6, 64, 128)
        assert f["timestamp"].dtype == np.int64
        assert f["time_plasim"].dtype == np.float64

        # /timestamp diff uniform
        ts = f["timestamp"][...]
        assert np.all(np.diff(ts) == STEP_SECONDS)

        # Channel labels match
        cs = [x.decode() for x in f["channel_state"][...]]
        assert cs == STATE_CHANNELS
        cd = [x.decode() for x in f["channel_diagnostic"][...]]
        assert cd == DIAGNOSTIC_CHANNELS
        cf = [x.decode() for x in f["channel_forcing"][...]]
        assert cf == FORCING_CHANNELS

        # File attrs present
        for name in ("rsdt_method", "sst_land_fill_K", "plasim_time_units", "plasim_calendar"):
            assert name in f.attrs
        assert f.attrs["rsdt_method"] in ("astronomical", b"astronomical")

    # sic report written
    report = output_root / "validation" / f"sic_clip_report_{year:04d}.json"
    assert report.exists()


def test_process_one_skips_warmup_year(tmp_path: Path):
    sim = 52
    opts = Namespace(
        sims=[sim],
        train_years=[3, 100],
        valid_years=[101, 120],
        test_years=[121, 128],
        postproc_root=tmp_path / "postproc",
        boundary_root=tmp_path / "boundary",
        output_root=tmp_path / "out",
        sst_land_fill_k=271.35,
        task_index=None,
        count_tasks=False,
        overwrite=False,
        dry_run=False,
        verbose=False,
    )
    # Year 1 and 2 are warmup: process_one must no-op without raising on missing inputs
    process_one(sim, 1, opts)
    process_one(sim, 2, opts)
    assert not (tmp_path / "out").exists()
