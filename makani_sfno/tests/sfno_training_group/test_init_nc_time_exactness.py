"""T.9 — init NC time-exactness (Phase F).

Verifies that ``build_init_nc_from_v10.py`` writes a NetCDF whose time coord
contains the requested init datetime EXACTLY. ``long_inference.py:1331`` does
``ds.get_index("time").get_loc(init_datetime)`` which raises KeyError on any
mismatch (incl. floating-point drift from cftime arithmetic).

Tests:
  - For YEAR=121 (non-leap): init NC time coord contains 0121-01-01 00:00:00.
  - For YEAR=124 (leap): init NC time coord contains 0124-01-01 00:00:00.
  - Init NC does NOT contain init_dt + 18h (would mean +offset got baked in).
  - get_loc succeeds for both via xarray.
"""

from __future__ import annotations

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


def _build_synthetic_v10_test_h5(out_path: Path, *, n_t: int = 10, H: int = 64, W: int = 128) -> None:
    """Minimal v10 source for the init-NC builder. Audit constraints:
      - mean(exp(pl)) ∈ [80000, 120000] Pa  → pl ≈ ln(100000) = 11.51
      - zg500 mean ∈ [5400, 5700] m         → state[46] ≈ 5500 m
    """
    rng = np.random.default_rng(seed=121)
    fields_state = rng.normal(size=(n_t, 52, H, W)).astype(np.float32)
    # Channel 0 = pl = ln(p_s); set to ~ln(100000) Pa.
    fields_state[:, 0] = np.log(100_000.0).astype(np.float32) + 0.01 * rng.normal(size=(n_t, H, W)).astype(np.float32)
    # zg200..zg1000 at state[42..51]; standard geopotential heights:
    zg_levels_m = [11700, 10300, 9100, 7100, 5500, 4200, 3000, 1500, 800, 100]
    for i, zg_m in enumerate(zg_levels_m):
        fields_state[:, 42 + i] = zg_m + 50 * rng.normal(size=(n_t, H, W)).astype(np.float32)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("fields_state", data=fields_state)
        f.create_dataset("fields_diagnostic", data=rng.normal(size=(n_t, 1, H, W)).astype(np.float32))
        f.create_dataset("forcing", data=rng.normal(size=(n_t, 6, H, W)).astype(np.float32))
        f.create_dataset("lat", data=np.linspace(-89, 89, H).astype(np.float64))
        f.create_dataset("lon", data=np.linspace(0, 357, W).astype(np.float64))
        f.create_dataset("time_plasim", data=np.arange(n_t, dtype=np.float64))
        # Channel manifests required by build_init_nc_from_v10.
        state_names = (
            ["pl", "tas"]
            + [f"ta{i+1}" for i in range(10)]
            + [f"ua{i+1}" for i in range(10)]
            + [f"va{i+1}" for i in range(10)]
            + [f"hus{i+1}" for i in range(10)]
            + [f"zg{p}" for p in (200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)]
        )
        f.create_dataset("channel_state",
                         data=np.array([s.encode() for s in state_names], dtype="S20"))
        f.create_dataset("channel_diagnostic",
                         data=np.array([b"pr_6h"], dtype="S20"))
        f.create_dataset("channel_forcing",
                         data=np.array([b"lsm", b"sg", b"z0", b"sst", b"rsdt", b"sic"], dtype="S20"))


def _run_init_nc_builder(*, src_h5: Path, init_dt: str, out: Path, ic_idx: int = 0) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}".rstrip(":")
    )
    return subprocess.run(
        [sys.executable, "-m", "sfno_training_group.tools.build_init_nc_from_v10",
         "--src-h5", str(src_h5),
         "--ic-idx", str(ic_idx),
         "--synthetic-init-dt", init_dt,
         "--out", str(out)],
        capture_output=True, text=True, env=env,
    )


@pytest.mark.parametrize("year", [121, 124])
def test_init_nc_contains_exact_init_datetime(tmp_path: Path, year: int) -> None:
    src_h5 = tmp_path / f"MOST.0{year:03d}.h5"
    _build_synthetic_v10_test_h5(src_h5)
    out = tmp_path / f"init_year{year}.nc"

    init_dt_str = f"{year:04d}-01-01 00:00:00"
    res = _run_init_nc_builder(src_h5=src_h5, init_dt=init_dt_str, out=out)
    assert res.returncode == 0, (
        f"build_init_nc_from_v10 failed for year {year}:\n"
        f"stdout={res.stdout}\nstderr={res.stderr}"
    )

    ds = xr.open_dataset(out, use_cftime=True)
    # Mirror long_inference.py:1330-1331: get_loc(init_datetime) MUST succeed exactly.
    target = cftime.DatetimeProlepticGregorian(year, 1, 1, 0, has_year_zero=True)
    try:
        idx = ds.get_index("time").get_loc(target)
    except KeyError as e:
        pytest.fail(
            f"YEAR={year}: init NC time coord does NOT contain {target} exactly. "
            f"long_inference.py:1331 will KeyError. Coord values: "
            f"{list(ds.get_index('time'))!r}\n"
            f"Original error: {e}"
        )
    assert isinstance(idx, (int, np.integer)), \
        f"get_loc returned non-int: {type(idx).__name__}={idx!r}"
    ds.close()


@pytest.mark.parametrize("year", [121, 124])
def test_init_nc_does_not_contain_offset_18h(tmp_path: Path, year: int) -> None:
    """Defensive: assert init NC time coord does NOT contain init_dt + 18h.

    The +18h `nc_bc_offset` only applies to the group H5 boundary/state read
    path in single_ic mode. If this offset accidentally leaked into the init NC
    builder, this test catches it.
    """
    src_h5 = tmp_path / f"MOST.0{year:03d}.h5"
    _build_synthetic_v10_test_h5(src_h5)
    out = tmp_path / f"init_year{year}.nc"
    init_dt_str = f"{year:04d}-01-01 00:00:00"
    _run_init_nc_builder(src_h5=src_h5, init_dt=init_dt_str, out=out)

    ds = xr.open_dataset(out, use_cftime=True)
    offset_target = cftime.DatetimeProlepticGregorian(year, 1, 1, 18, has_year_zero=True)
    with pytest.raises(KeyError):
        ds.get_index("time").get_loc(offset_target)
    ds.close()
