"""Tests for src/sfno_inference/nc_writer.py.

Coverage (per docs/sfno_eval_plan.md §B.4):

  - Round-trip: write a synthetic RolloutResult, reload via xarray,
    and check dims/coords/attrs match the §B.4 schema.
  - lead_time coord is ``[6, 12, ..., 6K]`` (NO lead 0).
  - channel_ic excludes pr_6h (the 53rd name).
  - global attrs preserve all three SHAs and the run-tag.
  - Variable-shape mismatch (lat length wrong) raises ValueError.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")
xr = pytest.importorskip("xarray")

from sfno_inference.nc_writer import write_rollout_nc, _parse_anchor_to_datetime64  # noqa: E402
from sfno_inference.rollout_driver import RolloutResult  # noqa: E402


_CHANNELS = [f"ch{i}" for i in range(52)] + ["pr_6h"]


def _make_result(K: int = 5, H: int = 4, W: int = 8) -> RolloutResult:
    return RolloutResult(
        prediction=torch.zeros(K, 53, H, W, dtype=torch.float32),
        truth=torch.ones(K, 53, H, W, dtype=torch.float32),
        init_state=torch.full((52, H, W), 0.5, dtype=torch.float32),
        K=K,
        ic_global_idx=12,
        ic_sample_idx=12,
        ic_file="MOST.0121.h5",
        file_anchor="0126-08-01 00:00:00",
        time_plasim_at_ic=3.0,  # 3 days after Aug 1
        rollout_mode="nwp",
    )


# ---------------------------------------------------------------------------
# _parse_anchor_to_datetime64
# ---------------------------------------------------------------------------

class TestParseAnchor:
    def test_basic(self):
        dt = _parse_anchor_to_datetime64("0126-08-01 00:00:00")
        assert dt == np.datetime64("0126-08-01T00:00:00", "s")

    def test_rejects_garbage(self):
        with pytest.raises(ValueError, match="unparseable"):
            _parse_anchor_to_datetime64("not a date")


# ---------------------------------------------------------------------------
# write_rollout_nc — round-trip
# ---------------------------------------------------------------------------

class TestWriteRollout:
    def test_round_trip_dims_and_coords(self, tmp_path):
        K, H, W = 5, 4, 8
        result = _make_result(K=K, H=H, W=W)
        out = tmp_path / "MOST.0121_ic000.nc"
        lat = np.linspace(90, -90, H)  # descending (North-first), per grid contract
        lon = np.linspace(0, 360, W, endpoint=False)
        write_rollout_nc(
            out,
            result=result,
            channel_names=_CHANNELS,
            lat=lat,
            lon=lon,
            ckpt_path="/scratch/best_ckpt_mp0.tar",
            eval_sha7="abc1234",
            data_sha7="58413cb",
            train_sha7="106d19d",
            run_tag="20260429_eval-abc1234_data-58413cb_train-106d19d_ckpt-best_ckpt_mp0",
        )
        ds = xr.open_dataset(out)
        try:
            assert ds.sizes == {
                "init_time": 1,
                "lead_time": K,
                "channel": 53,
                "channel_ic": 52,
                "lat": H,
                "lon": W,
            }
            # lead_time coords = [6, 12, ..., 6K] (NO lead 0).
            assert list(ds["lead_time"].values) == [6, 12, 18, 24, 30]
            # channel_ic drops the diagnostic.
            assert list(ds["channel_ic"].values) == _CHANNELS[:52]
            assert "pr_6h" not in list(ds["channel_ic"].values)
            assert "pr_6h" in list(ds["channel"].values)
        finally:
            ds.close()

    def test_global_attrs_preserved(self, tmp_path):
        K, H, W = 3, 4, 8
        result = _make_result(K=K, H=H, W=W)
        out = tmp_path / "out.nc"
        write_rollout_nc(
            out,
            result=result,
            channel_names=_CHANNELS,
            lat=np.linspace(90, -90, H),
            lon=np.linspace(0, 360, W, endpoint=False),
            ckpt_path="/scratch/best_ckpt_mp0.tar",
            eval_sha7="abc1234",
            data_sha7="58413cb",
            train_sha7="106d19d",
            run_tag="run-tag-here",
            rollout_mode="nwp",
        )
        ds = xr.open_dataset(out)
        try:
            assert ds.attrs["eval_sha7"] == "abc1234"
            assert ds.attrs["data_sha7"] == "58413cb"
            assert ds.attrs["train_sha7"] == "106d19d"
            assert ds.attrs["run_tag"] == "run-tag-here"
            assert ds.attrs["ic_file"] == "MOST.0121.h5"
            assert ds.attrs["ic_sample_idx"] == 12
            assert ds.attrs["file_anchor"] == "0126-08-01 00:00:00"
            assert ds.attrs["rollout_mode"] == "nwp"
            assert ds.attrs["K"] == K
            assert ds.attrs["dt_hours"] == 6
        finally:
            ds.close()

    def test_data_round_trip(self, tmp_path):
        K, H, W = 3, 4, 8
        result = _make_result(K=K, H=H, W=W)
        out = tmp_path / "out.nc"
        write_rollout_nc(
            out, result=result,
            channel_names=_CHANNELS,
            lat=np.linspace(90, -90, H),
            lon=np.linspace(0, 360, W, endpoint=False),
            ckpt_path="/x", eval_sha7="a", data_sha7="b", train_sha7="c", run_tag="t",
        )
        ds = xr.open_dataset(out)
        try:
            np.testing.assert_array_equal(
                ds["prediction"].values[0], np.zeros((K, 53, H, W), dtype=np.float32)
            )
            np.testing.assert_array_equal(
                ds["truth"].values[0], np.ones((K, 53, H, W), dtype=np.float32)
            )
            np.testing.assert_array_equal(
                ds["init_state"].values[0],
                np.full((52, H, W), 0.5, dtype=np.float32),
            )
        finally:
            ds.close()


# ---------------------------------------------------------------------------
# guard rails
# ---------------------------------------------------------------------------

class TestGuards:
    def test_rejects_lat_size_mismatch(self, tmp_path):
        result = _make_result(K=3, H=4, W=8)
        with pytest.raises(ValueError, match="lat/lon"):
            write_rollout_nc(
                tmp_path / "out.nc",
                result=result,
                channel_names=_CHANNELS,
                lat=np.linspace(-90, 90, 5),  # wrong length
                lon=np.linspace(0, 360, 8, endpoint=False),
                ckpt_path="/x", eval_sha7="a", data_sha7="b", train_sha7="c", run_tag="t",
            )

    def test_rejects_ascending_lat(self, tmp_path):
        # Data is North-first (descending); an ascending lat coordinate would
        # hemisphere-mislabel the file. The writer must fail loud.
        result = _make_result(K=3, H=4, W=8)
        with pytest.raises(ValueError, match="descending"):
            write_rollout_nc(
                tmp_path / "out.nc",
                result=result,
                channel_names=_CHANNELS,
                lat=np.linspace(-90, 90, 4),  # ascending => rejected
                lon=np.linspace(0, 360, 8, endpoint=False),
                ckpt_path="/x", eval_sha7="a", data_sha7="b", train_sha7="c", run_tag="t",
            )

    def test_rejects_channel_name_mismatch(self, tmp_path):
        result = _make_result(K=3, H=4, W=8)
        with pytest.raises(ValueError, match="channel_names"):
            write_rollout_nc(
                tmp_path / "out.nc",
                result=result,
                channel_names=_CHANNELS[:50],  # too few
                lat=np.linspace(-90, 90, 4),
                lon=np.linspace(0, 360, 8, endpoint=False),
                ckpt_path="/x", eval_sha7="a", data_sha7="b", train_sha7="c", run_tag="t",
            )

    def test_rejects_non_RolloutResult(self, tmp_path):
        with pytest.raises(TypeError, match="RolloutResult"):
            write_rollout_nc(
                tmp_path / "out.nc",
                result={"prediction": np.zeros((1,))},  # wrong type
                channel_names=_CHANNELS,
                lat=np.linspace(-90, 90, 4),
                lon=np.linspace(0, 360, 8, endpoint=False),
                ckpt_path="/x", eval_sha7="a", data_sha7="b", train_sha7="c", run_tag="t",
            )

    def test_truth_sic_absent_when_result_field_none(self, tmp_path):
        K, H, W = 3, 4, 8
        result = _make_result(K=K, H=H, W=W)
        assert result.truth_sic is None
        out = tmp_path / "no_sic.nc"
        write_rollout_nc(
            out, result=result, channel_names=_CHANNELS,
            lat=np.linspace(90, -90, H),
            lon=np.linspace(0, 360, W, endpoint=False),
            ckpt_path="/x", eval_sha7="a", data_sha7="b", train_sha7="c", run_tag="t",
        )
        ds = xr.open_dataset(out)
        try:
            assert "truth_sic" not in ds.variables
        finally:
            ds.close()

    def test_truth_sic_round_trip(self, tmp_path):
        K, H, W = 3, 4, 8
        # Hand-built sic: NaN over half (synthetic land), zeros over rest.
        sic = torch.zeros(K, H, W, dtype=torch.float32)
        sic[:, :H // 2, :] = float("nan")
        result = _make_result(K=K, H=H, W=W)
        result.truth_sic = sic
        out = tmp_path / "with_sic.nc"
        write_rollout_nc(
            out, result=result, channel_names=_CHANNELS,
            lat=np.linspace(90, -90, H),
            lon=np.linspace(0, 360, W, endpoint=False),
            ckpt_path="/x", eval_sha7="a", data_sha7="b", train_sha7="c", run_tag="t",
        )
        ds = xr.open_dataset(out)
        try:
            assert "truth_sic" in ds.variables
            arr = ds["truth_sic"].values
            assert arr.shape == (1, K, H, W)
            assert arr.dtype == np.float32
            # Land NaNs survive round-trip.
            assert np.all(np.isnan(arr[0, :, : H // 2, :]))
            assert np.all(arr[0, :, H // 2 :, :] == 0.0)
            assert ds["truth_sic"].attrs["units"] == "fraction"
        finally:
            ds.close()
