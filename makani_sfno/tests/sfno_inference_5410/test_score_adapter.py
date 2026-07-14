"""Tests for src/sfno_inference_5410/score_adapter.py.

Per docs/2026-05-08_sfno_5410_scoring_plan.md (v4.4). Coverage:

  Schema:
    - canonical_channel_names: 53-channel order matches climatology.
    - h5 key map (pl, ta1, ta10, zg500, zg1000, pr_6h).
    - Adapter output shape (init_time=1, lead_time=K, channel=53|52, lat, lon).
    - lead_time = [6, 12, ..., 360] integer hours.
    - init_time calendar-equivalence (cftime fallback for pre-1582).
    - file_anchor regex-parseable + matches `_date_for_lead`.
    - ic_file = "0XYZ.h5" → score_nwp.py:139 strip yields clean year.
    - channel_ic = 52 channels, drops pr_6h.

  Truth alignment (Codex round-4 + round-5):
    - test_truth_at_lead_6h_equals_h5_at_s_plus_1: bit-exact for 53 channels.
    - test_truth_magnitude_bounds_at_lead_6h: tas / zg500 / pl / pr_6h / ua5.
    - test_prediction_at_lead_6h_equals_raw_time1: per-channel match.
    - test_prediction_minus_truth_bounds: triple-bounded mean/p99/max.

  Climatology compat:
    - write_compat_clim renames time_of_year -> doy.
    - Idempotent on doy-form input.

Tests skip cleanly on machines without the upstream tree / production
RUN_ROOT.
"""
from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path

import cftime
import numpy as np
import pytest


_RUN_ROOT = Path(
    "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval_5410/20260507_phase1_gate"
)
_TRUTH_H5_DIR = Path(
    "/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/h5/sigma_data"
)
_CLIM_SRC = Path(
    "/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/baselines/climatology_proleptic_5410.nc"
)
_K = 60


def _need_live_data():
    if not (_RUN_ROOT / "inference" / "upstream_raw" / "Y121_s0000_member000_y0121.nc").is_file():
        pytest.skip(f"no live raw NC at {_RUN_ROOT}")
    if not (_TRUTH_H5_DIR / "121_0000.h5").is_file():
        pytest.skip(f"no truth h5 at {_TRUTH_H5_DIR}")
    if not _CLIM_SRC.is_file():
        pytest.skip(f"no clim at {_CLIM_SRC}")


# ----------------------------------------------------------------------
# Schema tests
# ----------------------------------------------------------------------

def test_canonical_channel_names_count_and_order():
    from sfno_inference_5410.score_adapter import canonical_channel_names

    names = canonical_channel_names()
    assert len(names) == 53
    assert names[0] == "pl"
    assert names[1] == "tas"
    assert names[2:12] == [f"ta{k+1}" for k in range(10)]
    assert names[12:22] == [f"ua{k+1}" for k in range(10)]
    assert names[22:32] == [f"va{k+1}" for k in range(10)]
    assert names[32:42] == [f"hus{k+1}" for k in range(10)]
    assert names[42:52] == [
        "zg200", "zg250", "zg300", "zg400", "zg500",
        "zg600", "zg700", "zg850", "zg925", "zg1000",
    ]
    assert names[52] == "pr_6h"


def test_canonical_channel_names_match_climatology():
    """Live: assert our canonical list matches the actual clim file."""
    pytest.importorskip("xarray")
    if not _CLIM_SRC.is_file():
        pytest.skip(f"no clim at {_CLIM_SRC}")
    import xarray as xr
    from sfno_inference_5410.score_adapter import canonical_channel_names

    canonical = canonical_channel_names()
    with xr.open_dataset(_CLIM_SRC) as ds:
        got = list(map(str, ds["channel"].values))
    assert got == canonical, (
        f"clim channel coord != canonical:\n  got first: {got[:5]}\n"
        f"  canonical first: {canonical[:5]}"
    )


def test_h5_key_for_channel():
    from sfno_inference_5410.score_adapter import _h5_key_for_channel

    assert _h5_key_for_channel("pl") == "pl"
    assert _h5_key_for_channel("tas") == "tas"
    assert _h5_key_for_channel("pr_6h") == "pr_6h"
    assert _h5_key_for_channel("ta1") == "ta_0.03830000013113022"
    assert _h5_key_for_channel("ta10") == "ta_0.983299970626831"
    assert _h5_key_for_channel("ua5") == "ua_0.4368000030517578"
    assert _h5_key_for_channel("zg500") == "zg_50000.0"
    assert _h5_key_for_channel("zg1000") == "zg_100000.0"
    assert _h5_key_for_channel("zg200") == "zg_20000.0"
    with pytest.raises(ValueError):
        _h5_key_for_channel("bogus")
    with pytest.raises(ValueError):
        _h5_key_for_channel("ta11")
    with pytest.raises(ValueError):
        _h5_key_for_channel("zg999")  # not in canonical plev list


# ----------------------------------------------------------------------
# Adapter output shape + schema
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def adapted_one_ic(tmp_path_factory):
    _need_live_data()
    pytest.importorskip("xarray")
    pytest.importorskip("h5py")
    from sfno_inference_5410.score_adapter import adapt_5410_ic_to_score_nwp

    out_dir = tmp_path_factory.mktemp("score_adapt")
    out_nc = out_dir / "Y121_s0000.nc"
    adapt_5410_ic_to_score_nwp(
        raw_nc_path=_RUN_ROOT / "inference" / "upstream_raw"
                              / "Y121_s0000_member000_y0121.nc",
        truth_h5_dir=_TRUTH_H5_DIR,
        Y=121, s=0, K=_K,
        out_nc_path=out_nc,
        ckpt_path="dummy_ckpt.tar",
        eval_sha7="abc1234",
        data_sha7="5410-v2.0",
        train_sha7="ckpt_epoch_50",
        run_tag="test_run_tag",
    )
    return out_nc


def test_adapt_one_ic_shapes(adapted_one_ic):
    import xarray as xr
    with xr.open_dataset(adapted_one_ic, decode_times=True) as ds:
        assert ds["prediction"].shape == (1, 60, 53, 64, 128)
        assert ds["truth"].shape == (1, 60, 53, 64, 128)
        assert ds["init_state"].shape == (1, 52, 64, 128)
        assert ds["lead_time"].dtype == np.int64
        assert ds["lead_time"].values.tolist() == list(range(6, 6 * 61, 6))
        assert list(ds["channel"].values).count("pr_6h") == 1
        assert "pr_6h" not in list(ds["channel_ic"].values)
        assert len(list(ds["channel_ic"].values)) == 52


def test_adapt_one_ic_init_time_calendar_equivalence(adapted_one_ic):
    """Codex round-5 fix #2 + round-6 fix #3: cftime fallback for pre-1582
    + full cftime arithmetic (not (s*6) % 24)."""
    import xarray as xr

    Y, s = 121, 0
    expected = cftime.DatetimeProlepticGregorian(
        Y, 1, 1, 0, has_year_zero=True,
    ) + dt.timedelta(hours=6 * s)
    with xr.open_dataset(adapted_one_ic, decode_times=True) as ds:
        v = ds["init_time"].values[0]
    if hasattr(v, "year"):
        y, mo, d, h = v.year, v.month, v.day, v.hour
    else:
        import pandas as pd
        ts = pd.Timestamp(v)
        y, mo, d, h = ts.year, ts.month, ts.day, ts.hour
    assert (y, mo, d, h) == (
        expected.year, expected.month, expected.day, expected.hour,
    )


def test_adapt_one_ic_attrs(adapted_one_ic):
    """nc_writer.py-canonical attrs + 5410 specifics."""
    import xarray as xr
    with xr.open_dataset(adapted_one_ic) as ds:
        a = ds.attrs
    # Required attrs (mirrors nc_writer.py).
    for k in ("ckpt_path", "eval_sha7", "data_sha7", "train_sha7", "run_tag",
              "ic_file", "ic_sample_idx", "ic_global_idx",
              "file_anchor", "time_plasim_at_ic", "rollout_mode",
              "K", "dt_hours"):
        assert k in a, f"missing attr {k}"
    # 5410 specifics.
    assert a["ic_file"] == "0121.h5"           # round-2 fix
    assert a["truth_h5_file"] == "121_0000.h5"
    assert a["file_anchor"] == "0121-01-01 00:00:00"  # parseable
    assert float(a["time_plasim_at_ic"]) == 0.0       # s=0
    assert int(a["K"]) == 60
    assert int(a["dt_hours"]) == 6
    assert str(a["rollout_mode"]) == "nwp"


def test_file_anchor_parses(adapted_one_ic):
    """file_anchor matches score_nwp.py:92 regex AND _date_for_lead
    yields the verification time of `init + lead`."""
    import xarray as xr
    with xr.open_dataset(adapted_one_ic) as ds:
        anchor = str(ds.attrs["file_anchor"])
        t_at_ic = float(ds.attrs["time_plasim_at_ic"])
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})", anchor)
    assert m is not None, f"anchor {anchor!r} does not parse"
    Y, M, D, h, mi, s = (int(g) for g in m.groups())
    base = cftime.DatetimeProlepticGregorian(Y, M, D, h, mi, s)
    # Lead 6h on (Y=121, s=0): expected verification = init + 6h.
    expected = (
        cftime.DatetimeProlepticGregorian(121, 1, 1, 0, has_year_zero=True)
        + dt.timedelta(hours=6)
    )
    actual = base + dt.timedelta(days=t_at_ic) + dt.timedelta(hours=6)
    assert (actual.month, actual.day, actual.hour) == (
        expected.month, expected.day, expected.hour,
    )


def test_ic_file_extracts_clean_year(adapted_one_ic):
    """score_nwp.py:139 does
    ic_year = ic_file.replace('MOST.','').replace('.h5',''). For our
    ic_file='0121.h5' this yields '0121', not '121_0000'."""
    import xarray as xr
    with xr.open_dataset(adapted_one_ic) as ds:
        ic_file = str(ds.attrs["ic_file"])
    ic_year = ic_file.replace("MOST.", "").replace(".h5", "")
    assert ic_year == "0121"


# ----------------------------------------------------------------------
# Truth-side bit-exact tests (Codex round-4 fix)
# ----------------------------------------------------------------------

def test_truth_at_lead_6h_equals_h5_at_s_plus_1(adapted_one_ic):
    """For Y=121 s=0, adapted truth at lead=6h MUST bit-equal
    121_0001.h5/input/<channel> for every channel."""
    pytest.importorskip("h5py")
    import h5py
    import xarray as xr
    from sfno_inference_5410.score_adapter import (
        canonical_channel_names, _h5_key_for_channel,
    )

    with xr.open_dataset(adapted_one_ic) as ds:
        # truth shape: (init_time=1, lead_time=60, channel=53, lat, lon).
        # lead_time index 0 is lead=6h.
        truth_at_6h = ds["truth"].isel(init_time=0, lead_time=0).values
    with h5py.File(_TRUTH_H5_DIR / "121_0001.h5", "r") as f:
        inp = f["input"]
        for c, name in enumerate(canonical_channel_names()):
            h5_arr = inp[_h5_key_for_channel(name)][:]
            adapted_arr = truth_at_6h[c]
            max_abs = float(np.max(np.abs(adapted_arr - h5_arr)))
            assert max_abs == 0.0, (
                f"{name}: truth at lead=6h max_abs vs h5 = {max_abs}"
            )


def test_truth_sic_at_each_lead_equals_h5_input_sic(adapted_one_ic):
    """truth_sic[k_lead] MUST bit-equal {Y}_{s+k_lead+1:04d}.h5/input/sic
    for every lead. This is the alignment-check counterpart to
    test_truth_at_lead_6h_equals_h5_at_s_plus_1 (docs/2026-05-14_tas_no_ice
    _metric_plan.md §6.5)."""
    pytest.importorskip("h5py")
    import h5py
    import xarray as xr

    with xr.open_dataset(adapted_one_ic) as ds:
        assert "truth_sic" in ds.variables
        assert ds["truth_sic"].dims == ("init_time", "lead_time", "lat", "lon")
        sic_adapted = ds["truth_sic"].isel(init_time=0).values  # (K, H, W)
        K = ds.sizes["lead_time"]

    for k_lead in range(K):
        s_target = 0 + (k_lead + 1)
        truth_h5 = _TRUTH_H5_DIR / f"121_{s_target:04d}.h5"
        with h5py.File(truth_h5, "r") as f:
            h5_arr = np.asarray(f["input/sic"][...], dtype=np.float32)
        max_abs = float(np.max(np.abs(sic_adapted[k_lead] - h5_arr)))
        assert max_abs == 0.0, (
            f"lead={k_lead}: truth_sic vs 121_{s_target:04d}.h5/input/sic "
            f"max_abs = {max_abs}"
        )


def test_truth_sic_attrs(adapted_one_ic):
    """truth_sic should carry the documented attrs so downstream tooling
    can find the threshold convention."""
    import xarray as xr
    with xr.open_dataset(adapted_one_ic) as ds:
        attrs = ds["truth_sic"].attrs
    assert attrs.get("units") == "fraction"
    assert "0.15" in attrs.get("description", "")


def test_truth_magnitude_bounds_at_lead_6h(adapted_one_ic):
    """Codex round-2 strengthening: magnitude bounds catch wrong-channel /
    wrong-unit / wrong-pressure-level mistakes that bit-exact alone
    wouldn't."""
    import xarray as xr
    with xr.open_dataset(adapted_one_ic) as ds:
        truth_at_6h = ds["truth"].isel(init_time=0, lead_time=0)
        chs = list(map(str, ds["channel"].values))

    def get(name):
        return truth_at_6h.isel(channel=chs.index(name)).values

    tas = get("tas")
    assert 240 <= tas.mean() <= 310, f"tas mean {tas.mean()} not in [240,310] K"
    assert tas.min() >= 180, f"tas min {tas.min()} < 180 K"
    assert tas.max() <= 340, f"tas max {tas.max()} > 340 K"

    zg500 = get("zg500")
    assert 5400 <= zg500.mean() <= 5800, f"zg500 mean {zg500.mean()}"
    # Live Y121 s=0 has min=4697.6 (Antarctic), so 4500 floor.
    assert zg500.min() >= 4500
    assert zg500.max() <= 6200

    pl = get("pl")
    assert 11.4 <= pl.mean() <= 11.6, f"pl mean {pl.mean()}"

    pr_6h = get("pr_6h")
    assert pr_6h.min() >= 0
    assert 1e-5 <= pr_6h.mean() <= 5e-3, f"pr_6h mean {pr_6h.mean()}"
    assert np.percentile(pr_6h, 99) <= 0.02
    assert pr_6h.max() <= 0.05

    ua5 = get("ua5")
    assert abs(ua5.mean()) <= 50, f"ua5 |mean| {abs(ua5.mean())}"


# ----------------------------------------------------------------------
# Prediction-side tests (Codex round-4 fix #1 second half)
# ----------------------------------------------------------------------

def test_prediction_at_lead_6h_equals_raw_time1(adapted_one_ic):
    """Adapted prediction at lead=6h comes from the raw NC's time=1 slice
    (per-channel match through the flat / sigma / plev remap)."""
    import xarray as xr

    raw_nc = (
        _RUN_ROOT / "inference" / "upstream_raw"
        / "Y121_s0000_member000_y0121.nc"
    )
    with xr.open_dataset(adapted_one_ic) as ds, \
         xr.open_dataset(raw_nc, decode_times=False) as raw:
        chs = list(map(str, ds["channel"].values))
        pred = ds["prediction"].isel(init_time=0, lead_time=0)

        # tas (flat 2D).
        tas_pred = pred.isel(channel=chs.index("tas")).values
        tas_raw = raw["tas"].isel(time=1).values
        assert np.array_equal(tas_pred, tas_raw)

        # ta5 (sigma index 4 = 5th sigma level, 1-indexed).
        ta5_pred = pred.isel(channel=chs.index("ta5")).values
        ta5_raw = raw["ta"].isel(time=1, lev=4).values
        assert np.array_equal(ta5_pred, ta5_raw)

        # zg500 (plev=50000 Pa, by literal value not positional).
        zg500_pred = pred.isel(channel=chs.index("zg500")).values
        zg500_raw = raw["zg"].isel(time=1).sel(plev=50000).values
        assert np.array_equal(zg500_pred, zg500_raw)


def test_prediction_minus_truth_bounds(adapted_one_ic):
    """Codex round-5 fix #1: triple-bounded (mean/p99/max) forecast
    error bounds calibrated against live Y121 s=0 data.
    Live: tas (mean=1.42, p99=6.49, max=8.90), zg500 (19.5/66.4/87.1),
    ua5 (1.50/5.29/6.78). Bounds set with margin."""
    import xarray as xr

    with xr.open_dataset(adapted_one_ic) as ds:
        pred_at_6h = ds["prediction"].isel(init_time=0, lead_time=0)
        truth_at_6h = ds["truth"].isel(init_time=0, lead_time=0)
        chs = list(map(str, ds["channel"].values))
        diff = np.abs(pred_at_6h.values - truth_at_6h.values)

    def diff_for(name):
        return diff[chs.index(name)]

    # tas
    d = diff_for("tas")
    assert d.mean() <= 3.0, f"tas mean error {d.mean()} > 3 K"
    assert np.percentile(d, 99) <= 8.0, f"tas p99 {np.percentile(d, 99)} > 8 K"
    assert d.max() <= 20.0, f"tas max {d.max()} > 20 K"

    # zg500
    d = diff_for("zg500")
    assert d.mean() <= 30.0
    assert np.percentile(d, 99) <= 80.0
    assert d.max() <= 200.0

    # ua5
    d = diff_for("ua5")
    assert d.mean() <= 3.0
    assert np.percentile(d, 99) <= 8.0
    assert d.max() <= 20.0


# ----------------------------------------------------------------------
# IC state tests
# ----------------------------------------------------------------------

def test_init_state_matches_h5_sample_s_state_only(adapted_one_ic):
    """init_state[0, c, :, :] for c=0..51 == 121_0000.h5's state channels."""
    pytest.importorskip("h5py")
    import h5py
    import xarray as xr
    from sfno_inference_5410.score_adapter import (
        canonical_channel_names, _h5_key_for_channel,
    )

    with xr.open_dataset(adapted_one_ic) as ds:
        ic = ds["init_state"].isel(init_time=0).values  # (52, 64, 128)
        chs_ic = list(map(str, ds["channel_ic"].values))
        assert "pr_6h" not in chs_ic, "pr_6h must NOT appear in channel_ic"

    canonical = canonical_channel_names()
    assert chs_ic == canonical[:52]

    with h5py.File(_TRUTH_H5_DIR / "121_0000.h5", "r") as f:
        inp = f["input"]
        for c, name in enumerate(chs_ic):
            h5_arr = inp[_h5_key_for_channel(name)][:]
            assert np.array_equal(ic[c], h5_arr), f"{name}: IC state mismatch"


def test_pr_6h_in_truth_but_not_init_state(adapted_one_ic):
    """pr_6h is at channel index 52 (last) in truth/prediction, but
    channel_ic has only 52 channels and excludes pr_6h."""
    import xarray as xr
    with xr.open_dataset(adapted_one_ic) as ds:
        chs = list(map(str, ds["channel"].values))
        chs_ic = list(map(str, ds["channel_ic"].values))
        assert chs[52] == "pr_6h"
        assert "pr_6h" not in chs_ic
        # truth at pr_6h channel must be finite (read from input/pr_6h).
        pr = ds["truth"].isel(init_time=0, lead_time=0, channel=52).values
        assert np.isfinite(pr).all()


# ----------------------------------------------------------------------
# Climatology compat tests
# ----------------------------------------------------------------------

def test_compat_clim_renames_time_of_year_to_doy(tmp_path):
    """Synthesize a small time_of_year-form clim and verify the rename.

    Avoid loading the live 2.37 GB clim in this unit test (causes OOM
    when run after the adapter tests in the same pytest process). The
    live clim is exercised by the smoke run.
    """
    pytest.importorskip("xarray")
    import numpy as np
    import xarray as xr
    from sfno_inference_5410.score_climatology_compat import write_compat_clim

    src = tmp_path / "src.nc"
    n_chan = 5
    ds = xr.Dataset(
        data_vars=dict(
            mean=(("time_of_year", "hour_quarter", "channel", "lat", "lon"),
                  np.zeros((366, 4, n_chan, 4, 4), dtype=np.float32)),
            n_contributors=(("time_of_year", "hour_quarter"),
                            np.zeros((366, 4), dtype=np.int32)),
        ),
        coords=dict(channel=("channel", ["pl", "tas", "ta1", "zg500", "pr_6h"])),
    )
    ds.to_netcdf(src)
    dst = tmp_path / "compat" / "climatology_proleptic.nc"
    write_compat_clim(src, dst)
    assert dst.is_file() or dst.is_symlink()
    with xr.open_dataset(dst) as out:
        assert "doy" in out.dims
        assert out.sizes["doy"] == 366
        assert "n_contributors" in out.data_vars
        assert "channel" in out.coords
        assert out.sizes["channel"] == n_chan


def test_compat_clim_idempotent_on_doy_input(tmp_path):
    """If input already has 'doy' as a dim, write_compat_clim symlinks."""
    pytest.importorskip("xarray")
    import xarray as xr
    from sfno_inference_5410.score_climatology_compat import write_compat_clim

    # Synthesize a doy-form clim file.
    src = tmp_path / "src.nc"
    n_chan = 3
    ds = xr.Dataset(
        data_vars=dict(
            mean=(("doy", "hour_quarter", "channel", "lat", "lon"),
                  np.zeros((366, 4, n_chan, 4, 4), dtype=np.float32)),
            n_contributors=(("doy", "hour_quarter"),
                            np.zeros((366, 4), dtype=np.int32)),
        ),
        coords=dict(channel=("channel", ["a", "b", "c"])),
    )
    ds.to_netcdf(src)
    dst = tmp_path / "out.nc"
    write_compat_clim(src, dst)
    # Confirm dst is reachable and has 'doy'.
    with xr.open_dataset(dst) as out:
        assert "doy" in out.dims


# ----------------------------------------------------------------------
# Subset-mode test for assert_output_dir_complete
# ----------------------------------------------------------------------

def test_assert_output_dir_complete_subset_allows_extras(tmp_path):
    """Codex round-6 fix #1 / round-7 #4: smoke against existing 96-IC
    raw dir."""
    import xarray as xr
    from sfno_inference_5410.preflight import assert_output_dir_complete

    out_dir = tmp_path / "raw"
    out_dir.mkdir()
    # Plant 3 well-formed NCs but only target 1 in the plan.
    for Y, s in [(121, 0), (121, 122), (121, 244)]:
        ds = xr.Dataset(
            {var: (("time", "lat", "lon"),
                   np.zeros((61, 64, 128), dtype=np.float32))
             for var in ("pl", "tas", "pr_6h", "ta", "ua", "va", "hus", "zg")}
        )
        # Stub: ta/ua/va/hus/zg need extra dims. Make them simple to satisfy
        # the variable-set check (assert_output_dir_complete only checks
        # data_vars membership, not shapes — except via assert_output_time_dim).
        ds.to_netcdf(out_dir / f"Y{Y}_s{s:04d}_member000_y{Y:04d}.nc")

    plan = [{"Y": 121, "s": 0, "save_basename": "Y121_s0000"}]

    # exact mode: should fail because 2 extras present.
    with pytest.raises(ValueError, match="extra"):
        assert_output_dir_complete(out_dir, plan, K=60, mode="exact")

    # subset mode: should pass (1 expected file present, extras ignored).
    assert_output_dir_complete(out_dir, plan, K=60, mode="subset")
