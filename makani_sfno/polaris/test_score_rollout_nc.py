#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
"""Tests for score_rollout_nc.py -- the K=56 read-out and its silent-failure guards.

    python makani_sfno/polaris/test_score_rollout_nc.py   # PASS = K56_SCORE_TESTS_OK
    pytest -q makani_sfno/polaris/test_score_rollout_nc.py

⚠ Needs a compute node: importing the scorer pulls in torch, and CLAUDE.md #3
forbids that on a Polaris login node.

Two things are being pinned, and only one of them is arithmetic.

1. **The decision rule.** `classify_regime` encodes
   `docs/2026-09-20_k56_readout_prereg.md` §3, fixed before job 7633207's output
   was scored. Every branch has a test here, so moving a threshold after seeing
   the numbers breaks a test rather than quietly changing the answer.
2. **The quadrature.** A Gauss-Legendre-weighted number on our equiangular grid
   is plausible and wrong -- change G measured the polar row over-weighted 1.50x,
   and the only guard downstream is a shape check that 180-vs-180 passes.
   `test_uses_equiangular_weights_not_gauss` fails if the scorer ever drifts back.

The three synthetic forecasts are the three reference regimes of prereg §3:
pred == truth (perfect), pred == climatology (fully blurred: NRMSE 1, VR 0), and
a sign-flipped anomaly (NRMSE 2, VR 1, ACC -1).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import score_rollout_nc as S  # noqa: E402

# Small enough to be instant, big enough that the two quadratures disagree
# visibly at H=8 (polar weight 0.0381 equiangular vs 0.0506 Gauss-Legendre).
H, W, C = 8, 16, 3
K = 56  # must reach 336 h, the pre-registered anchor
CHANNELS = ["PS", "T_l00", "PRECT"]


def _grid():
    """Return (lat, lon) for a cell-centred equiangular grid, lat DESCENDING."""
    lat = 90.0 - (np.arange(H) + 0.5) * (180.0 / H)
    lon = (np.arange(W) + 0.5) * (360.0 / W)
    return lat, lon


def _clim_field(rng):
    """A fixed per-cell climatology, the array `time_means.npy` stands in for."""
    return rng.normal(size=(C, H, W)) * 10.0 + 250.0


def _write_nc(path: Path, pred, truth, lead_h):
    """Write one rollout NetCDF in `nc_writer`'s schema (the subset the scorer reads)."""
    import xarray as xr

    lat, lon = _grid()
    ds = xr.Dataset(
        data_vars=dict(
            prediction=(("init_time", "lead_time", "channel", "lat", "lon"),
                        pred[np.newaxis, ...].astype(np.float32)),
            truth=(("init_time", "lead_time", "channel", "lat", "lon"),
                   truth[np.newaxis, ...].astype(np.float32)),
        ),
        coords=dict(init_time=("init_time", np.array([np.int64(0)])),
                    lead_time=("lead_time", np.asarray(lead_h, dtype=np.int64)),
                    channel=("channel", CHANNELS),
                    lat=("lat", lat), lon=("lon", lon)),
        attrs=dict(ic_file="2048.h5", ic_sample_idx=0, K=len(lead_h), dt_hours=6),
    )
    ds["lead_time"].attrs["units"] = "hours"
    ds.to_netcdf(path, format="NETCDF4")


def _make_case(tmp: Path, kind: str, n_ic: int = 2, n_lead: int = K):
    """Write `n_ic` NetCDFs of one regime plus the matching climatology .npy.

    `kind` is `perfect`, `climatology` (pred == clim), or `flipped`
    (pred - clim == -(truth - clim)).
    """
    rng = np.random.default_rng(0)
    clim = _clim_field(rng)
    lead_h = (np.arange(1, n_lead + 1) * 6).astype(np.int64)
    nc_dir = tmp / "nwp"
    nc_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_ic):
        anom = rng.normal(size=(n_lead, C, H, W)) * 3.0
        truth = clim[np.newaxis, ...] + anom
        if kind == "perfect":
            pred = truth.copy()
        elif kind == "climatology":
            pred = np.broadcast_to(clim, truth.shape).copy()
        elif kind == "flipped":
            pred = clim[np.newaxis, ...] - anom
        else:
            raise ValueError(kind)
        _write_nc(nc_dir / f"2048_ic{i:03d}.nc", pred, truth, lead_h)

    clim_path = tmp / "time_means.npy"
    np.save(clim_path, clim[np.newaxis, ...].astype(np.float32))
    return nc_dir, clim_path


def _run_main(nc_dir: Path, clim_path: Path, out_dir: Path, extra=()) -> int:
    """Invoke the scorer's CLI in-process; returns its exit code."""
    argv = ["score_rollout_nc.py", "--nc-dir", str(nc_dir), "--time-means", str(clim_path),
            "--out-dir", str(out_dir), *extra]
    old, sys.argv = sys.argv, argv
    try:
        return S.main()
    finally:
        sys.argv = old


def _readout(out_dir: Path) -> dict:
    return json.loads((out_dir / "k56_readout.json").read_text())


# ---------------------------------------------------------------------------
# The three reference regimes
# ---------------------------------------------------------------------------

def test_perfect_forecast_scores_zero_error():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        nc_dir, clim = _make_case(tmp, "perfect")
        out = tmp / "scores"
        assert _run_main(nc_dir, clim, out) == 0
        r = _readout(out)["readout"]
        assert abs(r["nrmse336"]) < 1e-6, r["nrmse336"]
        assert abs(r["acc336"] - 1.0) < 1e-6, r["acc336"]
        assert abs(r["vr336"] - 1.0) < 1e-6, r["vr336"]


def test_climatology_forecast_reads_as_blurring():
    """pred == climatology is the mode-averaging limit: NRMSE 1, VR 0, ACC 0 => CRPS."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        nc_dir, clim = _make_case(tmp, "climatology")
        out = tmp / "scores"
        assert _run_main(nc_dir, clim, out) == 0
        blob = _readout(out)
        r, v = blob["readout"], blob["verdict"]
        assert abs(r["nrmse336"] - 1.0) < 1e-5, r["nrmse336"]
        assert abs(r["vr336"]) < 1e-6, r["vr336"]
        assert abs(r["acc336"]) < 1e-6, r["acc336"]
        assert v["branch"] == "CRPS", v


def test_sign_flipped_anomaly_reads_as_drift():
    """Anti-correlated at full amplitude: ACC -1, NRMSE 2 -- past the drift threshold."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        nc_dir, clim = _make_case(tmp, "flipped")
        out = tmp / "scores"
        assert _run_main(nc_dir, clim, out) == 0
        blob = _readout(out)
        r, v = blob["readout"], blob["verdict"]
        # 1e-4, not 1e-6: the NetCDF stores float32, so `pred - truth == -2*anom`
        # holds only to f32 rounding at a climatology offset of ~250.
        assert abs(r["acc336"] + 1.0) < 1e-4, r["acc336"]
        assert abs(r["nrmse336"] - 2.0) < 1e-4, r["nrmse336"]
        assert abs(r["vr336"] - 1.0) < 1e-4, r["vr336"]
        assert v["branch"] == "DRIFT_FIRST", v


# ---------------------------------------------------------------------------
# Silent-failure guards
# ---------------------------------------------------------------------------

def test_uses_equiangular_weights_not_gauss():
    """A latitude-dependent error must be weighted by cos(lat), not by GL nodes.

    This is change G's defect in miniature. Both weight vectors have length H, so
    the only downstream guard -- a shape check -- passes either way; the number
    just comes out wrong, worst at the pole.
    """
    import torch
    from sfno_eval import metrics as M

    err = np.tile(np.arange(H, dtype=np.float64)[:, None], (1, W))
    pred = torch.from_numpy(err[None, ...])
    truth = torch.zeros_like(pred)

    w_eq = M.lat_weights(H, "equiangular").to(torch.float64)
    w_gl = M.lat_weights(H, "legendre-gauss").to(torch.float64)
    got = float(M.rmse_lat_weighted(pred, truth, w_eq)[0])

    lat = np.deg2rad(90.0 - (np.arange(H) + 0.5) * (180.0 / H))
    w_hand = np.cos(lat) / np.cos(lat).sum()
    expect = float(np.sqrt((w_hand * np.arange(H) ** 2).sum()))
    # 1e-6 relative, not exact: `lat_weights` returns float32 by construction
    # (both weight functions end in `.to(torch.float32)`), so a float64 hand
    # calculation agrees only to ~1e-8. That is the shared convention with the
    # PLaSim track and is 6 orders below the 1% the wrong quadrature costs.
    assert abs(got - expect) / expect < 1e-6, (got, expect)

    wrong = float(M.rmse_lat_weighted(pred, truth, w_gl)[0])
    assert abs(wrong - got) / got > 1e-3, (
        "the two quadratures agree here, so this test cannot catch the defect it "
        "exists for -- pick a grid where they differ"
    )


def test_channel_count_mismatch_fails_loud():
    """A 100-channel climatology against 101-channel NetCDFs must refuse, not average."""
    import pytest

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        nc_dir, clim_path = _make_case(tmp, "perfect", n_ic=1)
        short = np.load(clim_path)[:, :-1]          # drop the diagnostic channel
        np.save(clim_path, short)
        with pytest.raises(SystemExit) as exc:
            _run_main(nc_dir, clim_path, tmp / "scores")
        assert "CHANNEL_COUNT_MISMATCH" in str(exc.value)


def test_subset_model_scores_against_wider_pack_climatology_by_name():
    """Port F: 3-channel NetCDFs vs a 5-channel pack climatology with two channels
    inserted mid-list. Rows must be picked by name via metadata/data.json -- the
    `climatology` regime makes any row misalignment change the read-out."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        nc_dir, clim_path = _make_case(tmp, "climatology", n_ic=1)
        assert _run_main(nc_dir, clim_path, tmp / "ref") == 0
        clim = np.load(clim_path)[0]
        pack = tmp / "pack"
        (pack / "stats").mkdir(parents=True)
        (pack / "metadata").mkdir()
        names = ["PS", "SOIL", "TSOI", "T_l00", "PRECT"]
        wide = np.stack([clim[0], clim[0] * 0 + 1e3, clim[0] * 0 - 1e3, clim[1], clim[2]])
        np.save(pack / "stats" / "time_means.npy", wide[np.newaxis].astype(np.float32))
        (pack / "metadata" / "data.json").write_text(json.dumps({"coords": {"channel": names}}))
        assert _run_main(nc_dir, pack / "stats" / "time_means.npy", tmp / "sub") == 0
        ref, sub = _readout(tmp / "ref")["readout"], _readout(tmp / "sub")["readout"]
        for k, v in ref.items():
            if isinstance(v, float):
                assert np.isclose(sub[k], v, equal_nan=True), k


def test_wider_climatology_without_metadata_still_fails_loud():
    import pytest

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        nc_dir, clim_path = _make_case(tmp, "perfect", n_ic=1)
        wide = np.concatenate([np.load(clim_path)] * 2, axis=1)   # 6 channels, no metadata
        np.save(clim_path, wide)
        with pytest.raises(SystemExit) as exc:
            _run_main(nc_dir, clim_path, tmp / "scores")
        assert "CHANNEL_COUNT_MISMATCH" in str(exc.value)


def test_short_horizon_refuses_a_verdict():
    """A sweep that never reaches 336 h writes curves but must not read as a decision."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        nc_dir, clim = _make_case(tmp, "perfect", n_ic=1, n_lead=21)  # 126 h
        out = tmp / "scores"
        rc = _run_main(nc_dir, clim, out)
        assert rc == 3, rc
        v = _readout(out)["verdict"]
        assert v["branch"] == "INSUFFICIENT_HORIZON", v
        assert (out / "k56_metrics.h5").is_file()


def test_sigma_curve_is_written_for_every_lead():
    """Task 13 needs sigma(k) at all 56 depths, not just the scored anchors."""
    import h5py

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        nc_dir, clim = _make_case(tmp, "flipped", n_ic=2)
        out = tmp / "scores"
        _run_main(nc_dir, clim, out)
        with h5py.File(out / "k56_metrics.h5", "r") as f:
            assert f["rmse_mean"].shape == (K, C)
            assert list(f["lead_hours"][:]) == list(range(6, 6 * K + 1, 6))
            assert [c.decode() for c in f["channel"][:]] == CHANNELS
            assert np.isfinite(f["rmse_mean"][:]).all()


# ---------------------------------------------------------------------------
# The pre-registered rule itself (prereg §3) -- pure function, no I/O
# ---------------------------------------------------------------------------

def _readout_stub(**kw) -> dict:
    base = dict(available=True, nrmse336=1.2, nrmse126=0.8, vr336=0.9, vr126=0.95,
                acc336=0.4, acc126=0.88, slope_early_per_h=0.01, slope_late_per_h=0.005,
                r_slope=0.5, max_channel_nrmse336=1.4)
    base.update(kw)
    return base


def test_classify_crps_branch():
    v = S.classify_regime(_readout_stub(vr336=0.4, nrmse336=1.05, acc336=0.2))
    assert v["branch"] == "CRPS", v
    assert v["acc_consistent"] is True


def test_classify_crps_downgraded_when_acc_contradicts():
    """Blurring with ACC still 0.7 is not a coherent story -- say so, don't pick a side."""
    v = S.classify_regime(_readout_stub(vr336=0.4, nrmse336=1.05, acc336=0.7))
    assert v["branch"] == "AMBIGUOUS", v
    assert v["acc_consistent"] is False


def test_classify_nfuture8_branch():
    v = S.classify_regime(_readout_stub(vr336=0.95, nrmse336=1.35, r_slope=0.9, acc336=0.35))
    assert v["branch"] == "NFUTURE8", v


def test_classify_drift_branch_on_a_single_bad_channel():
    v = S.classify_regime(_readout_stub(nrmse336=1.4, max_channel_nrmse336=5.0))
    assert v["branch"] == "DRIFT_FIRST", v


def test_classify_ambiguous_between_the_bands():
    v = S.classify_regime(_readout_stub(vr336=0.72, nrmse336=1.22, r_slope=0.4))
    assert v["branch"] == "AMBIGUOUS", v


def test_classify_insufficient_horizon():
    v = S.classify_regime({"available": False, "missing_leads_h": [240, 336]})
    assert v["branch"] == "INSUFFICIENT_HORIZON", v


def test_readout_slope_is_a_ratio_of_finite_differences():
    """R_slope must read the pre-registered anchors, not a fitted window."""
    lead_h = np.arange(1, K + 1) * 6
    nrmse = np.zeros((K, 1))
    nrmse[:, 0] = np.linspace(0.0, 1.0, K)        # perfectly linear => R_slope 1
    vr = np.ones((K, 1))
    acc = np.zeros((K, 1))
    r = S.readout_from_curves(lead_h, nrmse, vr, acc)
    assert r["available"]
    assert abs(r["r_slope"] - 1.0) < 1e-9, r["r_slope"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok {fn.__name__}")
    print(f"K56_SCORE_TESTS_OK {len(fns)} tests")
