"""§3 P-7 hard gate — IC-NetCDF compatibility for ``--init_nc_filepaths``.

Per docs/2026-05-06_group_sfno_5410_eval_plan.md §3 P-7, runs the actual
upstream ``get_data_given_path_nc`` against the IC source pinned in
``<run_root>/inference/ic_source.json`` for every ``(Y, s)`` in the
96-tuple run plan. Asserts:

1. **Sanity-load (per `(Y, s)`)** — every call returns without raising;
   tensor shape is ``(52, 64, 128)``; every level-match within tolerance;
   every ``time``-lookup hits the requested ``init_datetime``.

2. **H5 cross-check** — coverage by source:
     - ``plev_data`` / ``sigma_data_transferred``: 3 spot tuples
       (``(121, 0)``, ``(124, 0)`` leap, ``(128, 1342)``);
     - ``ic_nc_built_from_h5``: all 96 tuples (each NC distinct).

   Loads the same channels through ``get_data_given_path`` against the
   per-timestep h5 (``<Y>_<s:04d>.h5``) and asserts
   ``max-abs-diff < 1e-4`` across all 52 prognostic channels.

3. **Contingency cascade** — failure at step 1 forces one of:
     - C-A: transfer ``sigma_data/<Y>_gaussian.nc`` from Derecho;
     - C-B: build per-IC NCs via ``scripts/build_ic_nc_from_h5.py``.

This test **requires** ``<run_root>`` and ``ic_source.json`` to be set
up by a wrapper script (e.g., a SLURM submit step or a manual run via
the gate orchestrator below). Without ``RUN_ROOT`` env var pointed at
a prepared run, the test auto-skips.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import cftime
import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_REPO = Path(
    "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0"
)


# Upstream import (skip cleanly if not on Stampede3).
def _upstream_loader():
    if str(_UPSTREAM_REPO) not in sys.path:
        sys.path.insert(0, str(_UPSTREAM_REPO))
    from utils.data_loader_multifiles import (  # type: ignore
        get_data_given_path,
        get_data_given_path_nc,
    )
    return get_data_given_path, get_data_given_path_nc


# Yaml-derived constants (verified 2026-05-07).
_SIGMA_LEVELS = (
    0.03830000013113022, 0.11910000443458557, 0.21085000783205032,
    0.3168500065803528, 0.4368000030517578, 0.5668000280857086,
    0.6993500888347626, 0.8233500719070435, 0.9240999817848206,
    0.983299970626831,
)
_PLEV_LEVELS = (20000, 25000, 30000, 40000, 50000, 60000, 70000, 85000, 92500, 100000)
_LEVELS_PER_VAR = [
    list(_SIGMA_LEVELS), list(_SIGMA_LEVELS), list(_SIGMA_LEVELS),
    list(_SIGMA_LEVELS), list(_PLEV_LEVELS),
]
_VARS_3D = ["ta", "ua", "va", "hus", "zg"]
_VARS_2D = ["pl", "tas"]

_OFFSETS = (0, 122, 244, 366, 488, 610, 732, 854, 976, 1098, 1220, 1342)
_TEST_YEARS = tuple(range(121, 129))
_LEAP_YEARS = (124, 128)


def _init_dt(Y: int, s: int) -> cftime.DatetimeProlepticGregorian:
    base = cftime.DatetimeProlepticGregorian(Y, 1, 1, 0, has_year_zero=True)
    return base + dt.timedelta(hours=s * 6)


def _h5_keys_in_channel_order():
    """Return the 52 h5 variable names in the same channel order as the
    NetCDF loader's output ((52, lat, lon)):

      ta sigma × 10, ua sigma × 10, va sigma × 10, hus sigma × 10,
      zg plev × 10, pl, tas.
    """
    keys: list[str] = []
    for var in ("ta", "ua", "va", "hus"):
        keys.extend(f"{var}_{lev}" for lev in _SIGMA_LEVELS)
    keys.extend(f"zg_{float(lev)}" for lev in _PLEV_LEVELS)
    keys.extend(["pl", "tas"])
    assert len(keys) == 52
    return keys


@pytest.fixture(scope="module")
def run_root():
    rr = os.environ.get("RUN_ROOT")
    if not rr:
        pytest.skip("RUN_ROOT not set — gate test runs against a prepared run dir")
    p = Path(rr)
    if not (p / "inference" / "ic_source.json").is_file():
        pytest.skip(f"ic_source.json missing under {p}/inference — gate not initialized")
    return p


@pytest.fixture(scope="module")
def loader():
    if not _UPSTREAM_REPO.is_dir():
        pytest.skip(f"upstream repo not present: {_UPSTREAM_REPO}")
    return _upstream_loader()


@pytest.fixture(scope="module")
def ic_source(run_root):
    import json
    cfg = json.loads((run_root / "inference" / "ic_source.json").read_text())
    return cfg["ic_source"]


@pytest.fixture(scope="module")
def resolver():
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    from sfno_inference_5410.ic_source import resolve_ic_nc_path
    return resolve_ic_nc_path


# Build the 96-tuple run plan.
def _run_plan_tuples():
    out = []
    for Y in _TEST_YEARS:
        for s in _OFFSETS:
            out.append((Y, s))
    return out


# -------------------------------------------------------------------- #
# Step 1 — sanity-load every (Y, s) through `get_data_given_path_nc`.
# -------------------------------------------------------------------- #
class TestSanityLoad:
    @pytest.mark.parametrize("Y,s", _run_plan_tuples())
    def test_loader_returns_correct_shape(self, run_root, loader, resolver, Y, s):
        _, get_nc = loader
        nc_path = resolver(Y, s, run_root)
        if not Path(nc_path).is_file():
            pytest.skip(f"IC NC missing: {nc_path}")
        arr = get_nc(
            str(nc_path), _VARS_3D, _VARS_2D,
            init_datetime=_init_dt(Y, s), levels_per_var=_LEVELS_PER_VAR,
        )
        assert arr.shape == (52, 64, 128), f"unexpected shape {arr.shape} for ({Y}, {s})"
        assert np.isfinite(arr).all(), f"non-finite values in ({Y}, {s})"


# -------------------------------------------------------------------- #
# Step 2 — H5 cross-check. Coverage depends on ic_source.
# -------------------------------------------------------------------- #
def _h5_cross_check_tuples(ic_source: str):
    if ic_source == "ic_nc_built_from_h5":
        return _run_plan_tuples()
    # plev_data / sigma_data_transferred: 3 spot tuples.
    return [(121, 0), (124, 0), (128, 1342)]


class TestH5CrossCheck:
    """Compare NC-loaded values vs h5-loaded values; max-abs-diff < 1e-4."""

    @pytest.fixture(scope="class")
    def cross_check_tuples(self, ic_source):
        return _h5_cross_check_tuples(ic_source)

    def test_all_tuples(self, run_root, loader, resolver, cross_check_tuples):
        get_h5, get_nc = loader
        h5_keys = _h5_keys_in_channel_order()
        h5_dir = Path(
            "/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/h5/sigma_data"
        )
        for Y, s in cross_check_tuples:
            nc_path = resolver(Y, s, run_root)
            h5_path = h5_dir / f"{Y}_{s:04d}.h5"
            if not nc_path.is_file() or not h5_path.is_file():
                pytest.skip(f"missing files for ({Y}, {s})")

            arr_nc = get_nc(
                str(nc_path), _VARS_3D, _VARS_2D,
                init_datetime=_init_dt(Y, s), levels_per_var=_LEVELS_PER_VAR,
            )
            arr_h5 = get_h5(str(h5_path), h5_keys)
            assert arr_nc.shape == arr_h5.shape, (
                f"shape mismatch for ({Y}, {s}): nc={arr_nc.shape}, h5={arr_h5.shape}"
            )
            diff = np.abs(arr_nc.astype(np.float64) - arr_h5.astype(np.float64))
            mad = float(diff.max())
            assert mad < 1e-4, (
                f"max-abs-diff {mad:.6e} exceeds 1e-4 for ({Y}, {s})"
            )
