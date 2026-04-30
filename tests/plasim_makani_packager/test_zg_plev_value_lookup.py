"""Pin §3.2's zg_plev value-lookup semantics.

Three cases (per docs/plasim_zg_plev_migration_plan.md §3.9):

  (a) ``lev_2 == [50, 100, ..., 1000]`` (canonical postproc order):
      packs successfully, with each ZG_PLEV_HPA value drawn from the
      lev_2 row of that hPa.

  (b) ``lev_2`` shuffled to a non-contiguous order containing all
      ZG_PLEV_HPA values: still packs successfully and yields zg500
      from the row whose lev_2 value is 500.

  (c) ``lev_2`` missing one of ZG_PLEV_HPA (e.g. drop 925):
      raises RuntimeError mentioning the missing value.
"""

from __future__ import annotations

import numpy as np
import pytest

from plasim_makani_packager.channels import ZG_PLEV_HPA
from plasim_makani_packager.packager import _stack_fields_state

xr = pytest.importorskip("xarray")


# ---------------------------------------------------------------------------
def _make_postproc_ds(
    *,
    T: int = 2,
    H: int = 4,
    W: int = 8,
    lev_2_values: tuple[int, ...] = (
        50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000,
    ),
    zg_plev: np.ndarray | None = None,
) -> xr.Dataset:
    """Minimal postproc-shaped dataset with a configurable ``lev_2``.

    ``zg_plev`` is filled with values keyed to the lev_2 hPa value for
    each row, so the value-lookup result can be verified by reading the
    row whose lev_2 == hPa.
    """
    rng = np.random.default_rng(0)
    lev = np.linspace(0.05, 0.98, 10, dtype=np.float32)
    time = np.arange(T, dtype=np.float64)
    lat = np.linspace(-80.0, 80.0, H, dtype=np.float64)
    lon = np.linspace(0.0, 357.1875, W, dtype=np.float64)

    if zg_plev is None:
        zg_plev = np.empty((T, len(lev_2_values), H, W), dtype=np.float32)
        for k, hpa in enumerate(lev_2_values):
            # Stamp each level with its hPa as a scalar — value-lookup
            # correctness is then verified by reading the resulting row.
            zg_plev[:, k] = float(hpa)

    data_vars = {
        "pl": (("time", "lat", "lon"), rng.standard_normal((T, H, W)).astype(np.float32)),
        "tas": (("time", "lat", "lon"), rng.standard_normal((T, H, W)).astype(np.float32)),
    }
    for v in ("ta", "ua", "va", "hus"):
        data_vars[v] = (
            ("time", "lev", "lat", "lon"),
            rng.standard_normal((T, 10, H, W)).astype(np.float32),
        )
    data_vars["zg_plev"] = (("time", "lev_2", "lat", "lon"), zg_plev)

    return xr.Dataset(
        data_vars,
        coords={
            "time": time,
            "lev": lev,
            "lev_2": np.array(lev_2_values, dtype=np.int32),
            "lat": lat,
            "lon": lon,
        },
    )


# ---------------------------------------------------------------------------
def test_canonical_lev_2_packs_correctly():
    ds = _make_postproc_ds()
    arr = _stack_fields_state(ds)
    # zg{P} is at index 42 + k where ZG_PLEV_HPA[k] == P
    for k, hpa in enumerate(ZG_PLEV_HPA):
        np.testing.assert_array_equal(
            arr[:, 42 + k], np.full(arr[:, 42 + k].shape, float(hpa), dtype=np.float32)
        )


def test_shuffled_lev_2_still_packs_by_value():
    """Postprocessor reorders lev_2 — packager must follow the values, not
    the indices. zg500 lookup must still resolve to the row where lev_2 == 500."""
    shuffled = (1000, 500, 50, 925, 100, 850, 700, 600, 400, 300, 250, 200, 150)
    ds = _make_postproc_ds(lev_2_values=shuffled)
    arr = _stack_fields_state(ds)
    # ZG_PLEV_HPA[5] == 500 → channel at index 47 must equal 500
    assert ZG_PLEV_HPA[5] == 500
    np.testing.assert_array_equal(
        arr[:, 47], np.full(arr[:, 47].shape, 500.0, dtype=np.float32)
    )
    # And at index 42 (zg150) must equal 150
    np.testing.assert_array_equal(
        arr[:, 42], np.full(arr[:, 42].shape, 150.0, dtype=np.float32)
    )


def test_missing_pressure_level_raises():
    """If lev_2 is missing one of ZG_PLEV_HPA (e.g. 925), packing must
    raise a clear RuntimeError naming the missing hPa."""
    # Drop 925 from the canonical list
    incomplete = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 1000)
    ds = _make_postproc_ds(lev_2_values=incomplete)
    with pytest.raises(RuntimeError) as exc_info:
        _stack_fields_state(ds)
    msg = str(exc_info.value)
    assert "925" in msg
    assert "missing" in msg.lower()
