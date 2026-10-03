"""Test for build_true_climatology.py's streaming time-mean, against a direct
(non-streaming) numpy mean over the same tiny synthetic pack used by
tests/sfno_inference/test_climate_driver.py -- reuses its plain fixture-building
functions rather than duplicating a second synthetic dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("makani")
pytest.importorskip("h5py")

_TEST_DIR = Path(__file__).resolve().parents[1] / "tests" / "sfno_inference"
sys.path.insert(0, str(_TEST_DIR))
from test_climate_driver import C_OUT, FPY, _dataset, _make_pack, _year_dir  # noqa: E402

from build_true_climatology import compute_time_mean_z  # noqa: E402


def test_streaming_mean_matches_direct_mean(tmp_path):
    pack = _make_pack(tmp_path / "pack", {2043: "train", 2044: "train", 2045: "valid"})
    yd = _year_dir(pack, [2043, 2044, 2045], tmp_path / "years")
    dataset = _dataset(yd, pack)

    got, n_total = compute_time_mean_z(dataset, C_OUT)
    assert n_total == dataset.n_samples_total - 1

    direct = torch.zeros((C_OUT, got.shape[-2], got.shape[-1]), dtype=torch.float64)
    for idx in range(n_total):
        _, tar, _, _ = dataset[idx]
        direct += tar.reshape(tar.shape[-3:]).to(torch.float64)
    direct /= n_total

    assert torch.equal(got, direct)


def test_streaming_mean_covers_almost_the_whole_window(tmp_path):
    pack = _make_pack(tmp_path / "pack", {2043: "train", 2044: "train", 2045: "valid"})
    yd = _year_dir(pack, [2043, 2044, 2045], tmp_path / "years")
    dataset = _dataset(yd, pack)
    _, n_total = compute_time_mean_z(dataset, C_OUT)
    # Every frame but the very last one of the 3-year synthetic pack.
    assert n_total == 3 * FPY - 1
