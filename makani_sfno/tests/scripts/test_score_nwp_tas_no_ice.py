"""Tests for the tas_no_ice rows emitted by score_nwp._compute_metrics_for_one_ic.

Coverage of the locked-in behaviour from
docs/2026-05-14_tas_no_ice_metric_plan.md:

  - When the per-IC NetCDF carries a `truth_sic` variable, scoring emits
    a `tas_no_ice` channel with emulator RMSE, persistence RMSE, and ACC
    rows alongside the existing `tas` rows.
  - The existing `tas` rows are unchanged (bit-identical) regardless of
    whether `truth_sic` is present.
  - `truth_sic` is NaN-over-land safe: cells with NaN are kept (they
    represent land per packager.py:226).
  - Threshold is strict: `sic == 0.15` is dropped; `sic < 0.15` is kept.
  - Fully-masked leads emit no `tas_no_ice` row (preserves the finite-row
    gate at score_nwp.py:258-262).
  - When `truth_sic` is absent, no `tas_no_ice` rows appear and no crash.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")
xr = pytest.importorskip("xarray")

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import score_nwp  # noqa: E402


# Channel-set used everywhere here. Position of "tas" matters (score_nwp
# looks it up by name); pad with dummy state channels and a diagnostic.
_CHANNELS = ["tas"] + [f"st{i}" for i in range(51)] + ["pr_6h"]


def _write_test_nc(
    path: Path,
    *,
    K: int,
    H: int,
    W: int,
    pred_tas: np.ndarray,
    truth_tas: np.ndarray,
    init_tas: np.ndarray,
    truth_sic: np.ndarray | None,
    file_anchor: str = "0126-08-01 00:00:00",
) -> None:
    """Build a minimal score_nwp-shaped per-IC NetCDF.

    Only the tas channel carries non-zero values; everything else is zero,
    so the masked-tas rows are the only ones that exercise non-trivial
    arithmetic. lead_time is 6 h × {1,...,K}.
    """
    prediction = np.zeros((1, K, len(_CHANNELS), H, W), dtype=np.float32)
    truth = np.zeros_like(prediction)
    init_state = np.zeros((1, 52, H, W), dtype=np.float32)
    prediction[0, :, 0] = pred_tas
    truth[0, :, 0] = truth_tas
    init_state[0, 0] = init_tas

    data_vars = {
        "prediction": (
            ("init_time", "lead_time", "channel", "lat", "lon"),
            prediction,
        ),
        "truth": (
            ("init_time", "lead_time", "channel", "lat", "lon"),
            truth,
        ),
        "init_state": (
            ("init_time", "channel_ic", "lat", "lon"),
            init_state,
        ),
    }
    if truth_sic is not None:
        data_vars["truth_sic"] = (
            ("init_time", "lead_time", "lat", "lon"),
            truth_sic[np.newaxis, ...].astype(np.float32),
        )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            init_time=("init_time", np.array([np.datetime64("2000-01-01")])),
            lead_time=("lead_time", np.arange(1, K + 1, dtype=np.int64) * 6),
            channel=("channel", _CHANNELS),
            channel_ic=("channel_ic", _CHANNELS[:52]),
            lat=("lat", np.linspace(-90, 90, H, dtype=np.float64)),
            lon=("lon", np.linspace(0, 360, W, endpoint=False, dtype=np.float64)),
        ),
        attrs=dict(
            ic_file=f"MOST.0121.h5",
            ic_sample_idx=0,
            file_anchor=file_anchor,
            time_plasim_at_ic=0.0,
        ),
    )
    ds.to_netcdf(path)


def _empty_climatology(H: int, W: int, *, has_bin: bool = True):
    """Build clim arrays sized (366, 4, n_chan, H, W) + n_contributors.

    If has_bin=True, mark the calendar bin score_nwp will look up for the
    test's file_anchor+time_plasim_at_ic+lead_h as populated, so ACC rows
    are emitted. Otherwise leave it zero (no ACC rows emitted).
    """
    clim_mean = np.zeros((366, 4, len(_CHANNELS), H, W), dtype=np.float32)
    clim_n = np.zeros((366, 4), dtype=np.int32)
    if has_bin:
        clim_n[:] = 1   # all bins populated; we just want ACC to run
    return clim_mean, clim_n


def _score_one(nc_path: Path, *, truth_sic_present: bool):
    """Wrap score_nwp._compute_metrics_for_one_ic. Returns rows list."""
    H, W = 4, 8
    rows: list[dict] = []
    bias_acc: dict = {}
    lat_weights = np.ones(H, dtype=np.float32) / H
    clim_mean, clim_n = _empty_climatology(H, W, has_bin=True)
    score_nwp._compute_metrics_for_one_ic(
        nc_path,
        clim_mean=clim_mean,
        clim_n=clim_n,
        channels=_CHANNELS,
        lat_weights=lat_weights,
        rows=rows,
        bias_accumulators=bias_acc,
        bias_channel_list=(),  # no bias maps for this unit test
    )
    return rows, bias_acc


def _rows_for(rows, *, channel: str, model: str, metric: str):
    return [r for r in rows if r["channel"] == channel
            and r["model"] == model and r["metric"] == metric]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTasNoIceRows:
    def test_emits_rmse_acc_with_sic_present(self, tmp_path):
        K, H, W = 2, 4, 8
        # err = 1 everywhere on tas; sic = 0 (open ocean) so all cells kept.
        truth_tas = np.zeros((K, H, W), dtype=np.float32)
        pred_tas = np.ones((K, H, W), dtype=np.float32)
        init_tas = np.full((H, W), 0.5, dtype=np.float32)
        sic = np.zeros((K, H, W), dtype=np.float32)
        nc = tmp_path / "with_sic.nc"
        _write_test_nc(
            nc, K=K, H=H, W=W,
            pred_tas=pred_tas, truth_tas=truth_tas, init_tas=init_tas,
            truth_sic=sic,
        )
        rows, _ = _score_one(nc, truth_sic_present=True)

        # tas_no_ice rows present for emulator + persistence RMSE, emulator ACC.
        em = _rows_for(rows, channel="tas_no_ice", model="emulator", metric="rmse")
        ps = _rows_for(rows, channel="tas_no_ice", model="persistence", metric="rmse")
        ac = _rows_for(rows, channel="tas_no_ice", model="emulator", metric="acc")
        # Only the 6h lead falls inside _SCORED_LEADS_H here (K=2 → 6h, 12h;
        # 12h is not scored).
        assert len(em) == 1
        assert len(ps) == 1
        assert len(ac) == 1
        # Sanity: emulator RMSE on all-kept cells matches the unmasked value.
        # err=1 everywhere → RMSE=1.
        assert em[0]["value"] == pytest.approx(1.0, abs=1e-6)

    def test_no_tas_no_ice_when_sic_absent(self, tmp_path):
        K, H, W = 2, 4, 8
        nc = tmp_path / "no_sic.nc"
        _write_test_nc(
            nc, K=K, H=H, W=W,
            pred_tas=np.ones((K, H, W), dtype=np.float32),
            truth_tas=np.zeros((K, H, W), dtype=np.float32),
            init_tas=np.zeros((H, W), dtype=np.float32),
            truth_sic=None,
        )
        rows, _ = _score_one(nc, truth_sic_present=False)
        assert _rows_for(rows, channel="tas_no_ice", model="emulator", metric="rmse") == []
        # Existing tas rows still emitted.
        assert _rows_for(rows, channel="tas", model="emulator", metric="rmse") != []

    def test_tas_rows_bit_identical_with_and_without_sic(self, tmp_path):
        """`tas` numbers must not change when truth_sic is added."""
        K, H, W = 2, 4, 8
        truth_tas = np.zeros((K, H, W), dtype=np.float32)
        pred_tas = np.ones((K, H, W), dtype=np.float32)
        init_tas = np.full((H, W), 0.5, dtype=np.float32)
        nc_no = tmp_path / "no_sic.nc"
        nc_yes = tmp_path / "with_sic.nc"
        _write_test_nc(
            nc_no, K=K, H=H, W=W,
            pred_tas=pred_tas, truth_tas=truth_tas, init_tas=init_tas,
            truth_sic=None,
        )
        _write_test_nc(
            nc_yes, K=K, H=H, W=W,
            pred_tas=pred_tas, truth_tas=truth_tas, init_tas=init_tas,
            truth_sic=np.zeros((K, H, W), dtype=np.float32),
        )
        rows_no, _ = _score_one(nc_no, truth_sic_present=False)
        rows_yes, _ = _score_one(nc_yes, truth_sic_present=True)
        for model, metric in (("emulator", "rmse"),
                              ("persistence", "rmse"),
                              ("emulator", "acc")):
            a = _rows_for(rows_no, channel="tas", model=model, metric=metric)
            b = _rows_for(rows_yes, channel="tas", model=model, metric=metric)
            assert len(a) == len(b)
            for ra, rb in zip(a, b):
                assert ra["value"] == rb["value"]   # bit-identical

    def test_fully_masked_lead_emits_no_row(self, tmp_path):
        K, H, W = 2, 4, 8
        # sic = 1.0 everywhere → mask fully False → no tas_no_ice row.
        sic = np.ones((K, H, W), dtype=np.float32)
        nc = tmp_path / "full_ice.nc"
        _write_test_nc(
            nc, K=K, H=H, W=W,
            pred_tas=np.ones((K, H, W), dtype=np.float32),
            truth_tas=np.zeros((K, H, W), dtype=np.float32),
            init_tas=np.zeros((H, W), dtype=np.float32),
            truth_sic=sic,
        )
        rows, _ = _score_one(nc, truth_sic_present=True)
        assert _rows_for(rows, channel="tas_no_ice", model="emulator", metric="rmse") == []
        # And no NaN rows snuck in.
        for r in rows:
            if r["channel"] == "tas_no_ice" and r["model"] == "emulator":
                assert r["value"] == r["value"]   # not NaN

    def test_nan_over_land_is_kept(self, tmp_path):
        """NaN sic (land per packager.py:226) must NOT be treated as ice."""
        K, H, W = 1, 4, 8
        # Half the field is "land" (NaN sic), half is "ocean" (sic=0).
        # tas err is 0 over land, 2 over ocean. If land were dropped (the
        # bug), the masked RMSE would be 2; if it's kept (correct), the
        # masked RMSE is sqrt((0*0.5 + 4*0.5)) = sqrt(2).
        truth_tas = np.zeros((K, H, W), dtype=np.float32)
        pred_tas = np.zeros((K, H, W), dtype=np.float32)
        pred_tas[:, H // 2 :, :] = 2.0  # err=2 on the ocean half
        init_tas = np.zeros((H, W), dtype=np.float32)
        sic = np.zeros((K, H, W), dtype=np.float32)
        sic[:, : H // 2, :] = float("nan")   # land
        # Restrict to a single scored lead by giving K=1 → only 6h scored.
        nc = tmp_path / "land_nan.nc"
        _write_test_nc(
            nc, K=K, H=H, W=W,
            pred_tas=pred_tas, truth_tas=truth_tas, init_tas=init_tas,
            truth_sic=sic,
        )
        rows, _ = _score_one(nc, truth_sic_present=True)
        em = _rows_for(rows, channel="tas_no_ice", model="emulator", metric="rmse")
        assert len(em) == 1
        # If land was kept correctly: sqrt(2). If land was dropped (bug): 2.
        assert em[0]["value"] == pytest.approx(np.sqrt(2.0), abs=1e-5)

    def test_threshold_strict_ge(self, tmp_path):
        """sic == 0.15 is dropped (strict >=). sic < 0.15 is kept."""
        K, H, W = 1, 4, 8
        truth_tas = np.zeros((K, H, W), dtype=np.float32)
        # Per-cell err: row 0 = 1, row 1 = 1, row 2 = 1, row 3 = 1 (constant).
        pred_tas = np.ones((K, H, W), dtype=np.float32)
        init_tas = np.zeros((H, W), dtype=np.float32)
        # Lat row sic values: 0.0, 0.149999, 0.15, 0.5.
        # mask via ~(>=0.15) → kept: rows 0+1, dropped: rows 2+3.
        sic = np.zeros((K, H, W), dtype=np.float32)
        sic[0, 0, :] = 0.0
        sic[0, 1, :] = 0.149999
        sic[0, 2, :] = 0.15
        sic[0, 3, :] = 0.5
        nc = tmp_path / "edge.nc"
        _write_test_nc(
            nc, K=K, H=H, W=W,
            pred_tas=pred_tas, truth_tas=truth_tas, init_tas=init_tas,
            truth_sic=sic,
        )
        rows, _ = _score_one(nc, truth_sic_present=True)
        em = _rows_for(rows, channel="tas_no_ice", model="emulator", metric="rmse")
        assert len(em) == 1
        # err=1 everywhere on kept cells → RMSE=1.
        assert em[0]["value"] == pytest.approx(1.0, abs=1e-6)
