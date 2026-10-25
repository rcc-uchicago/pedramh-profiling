"""Test for build_true_climatology.py's streaming time-mean.

The primary check compares against a ground truth computed independently of
`dataset[idx]` -- reading the synthetic pack's raw h5 arrays directly via
h5py -- so an index/frame-offset bug in `compute_time_mean_z` or in the
dataset's own indexing would actually be caught (a same-dataset-loop
comparison, used in an earlier version of this test, cannot catch that: see
the 2026-10-03 review that flagged it as tautological).

Reuses tests/sfno_inference/test_climate_driver.py's plain fixture-building
functions rather than duplicating a second synthetic dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("makani")
h5py = pytest.importorskip("h5py")

_TEST_DIR = Path(__file__).resolve().parents[1] / "tests" / "sfno_inference"
sys.path.insert(0, str(_TEST_DIR))
from test_climate_driver import C_OUT, FPY, _dataset, _make_pack, _year_dir  # noqa: E402

from build_true_climatology import compute_time_mean_z  # noqa: E402

_YEARS = {2043: "train", 2044: "train", 2045: "valid"}


def _ground_truth_mean_phys(pack, years):
    """Independent of PlasimForcingDataset: read the raw h5 arrays directly,
    concatenate state+diagnostic channels (the dataset's own out_channels
    order, confirmed elsewhere in this fixture), and average over every
    frame EXCEPT THE FIRST of the whole multi-year window -- the same window
    `compute_time_mean_z` is documented to cover.
    """
    splits = {2043: "train", 2044: "train", 2045: "valid"}
    frames = []
    for y in years:
        path = pack / splits[y] / f"{y}.h5"
        with h5py.File(path, "r") as f:
            state = f["fields_state"][:]        # (T, CS, H, W)
            diag = f["fields_diagnostic"][:]     # (T, 1, H, W)
        frames.append(np.concatenate([state, diag], axis=1))  # (T, C_OUT, H, W)
    full = np.concatenate(frames, axis=0)        # (n_years*T, C_OUT, H, W)
    return full[1:].astype(np.float64).mean(axis=0)  # drop frame 0, matches contract


def test_streaming_mean_matches_independent_ground_truth(tmp_path):
    pack = _make_pack(tmp_path / "pack", _YEARS)
    yd = _year_dir(pack, [2043, 2044, 2045], tmp_path / "years")
    dataset = _dataset(yd, pack)

    time_mean_z, n_total = compute_time_mean_z(dataset, C_OUT)
    mean = np.load(pack / "stats/global_means.npy").astype(np.float64).reshape(-1)
    std = np.load(pack / "stats/global_stds.npy").astype(np.float64).reshape(-1)
    got_phys = time_mean_z.numpy() * std[:, None, None] + mean[:, None, None]

    assert n_total == 3 * FPY - 1
    truth_phys = _ground_truth_mean_phys(pack, [2043, 2044, 2045])
    np.testing.assert_allclose(got_phys, truth_phys, rtol=1e-8, atol=1e-6)


def test_seeded_off_by_one_breaks_the_ground_truth_match(tmp_path, monkeypatch):
    """The same seeded-fault pattern used throughout test_climate_driver.py:
    shift the read window by one frame and confirm the ground-truth check
    above actually goes red -- proof the test isn't tautological anymore.

    Shifts BACKWARD (idx-1, clamped at 0): compute_time_mean_z itself only
    ever queries 0..n_total-1 (n_samples_total - 2), so every shifted index
    stays inside that already-valid range. Shifting forward instead would
    probe n_samples_total - 1, the absolute last sample, which has no next
    frame to serve as `tar` and raises on an empty tensor -- a dataset
    boundary quirk unrelated to the fault this test means to inject.
    """
    pack = _make_pack(tmp_path / "pack", _YEARS)
    yd = _year_dir(pack, [2043, 2044, 2045], tmp_path / "years")
    dataset = _dataset(yd, pack)

    import build_true_climatology as btc
    real_getitem = type(dataset).__getitem__

    def shifted_getitem(self, idx):
        return real_getitem(self, max(idx - 1, 0))

    monkeypatch.setattr(type(dataset), "__getitem__", shifted_getitem)
    time_mean_z, _ = btc.compute_time_mean_z(dataset, C_OUT)
    mean = np.load(pack / "stats/global_means.npy").astype(np.float64).reshape(-1)
    std = np.load(pack / "stats/global_stds.npy").astype(np.float64).reshape(-1)
    got_phys = time_mean_z.numpy() * std[:, None, None] + mean[:, None, None]

    truth_phys = _ground_truth_mean_phys(pack, [2043, 2044, 2045])
    assert not np.allclose(got_phys, truth_phys, rtol=1e-8, atol=1e-6)


def test_raises_on_empty_dataset():
    class _Empty:
        n_samples_total = 1

    with pytest.raises(ValueError, match="NO_FRAMES"):
        compute_time_mean_z(_Empty(), C_OUT)


def test_raises_on_channel_count_mismatch(tmp_path):
    pack = _make_pack(tmp_path / "pack", _YEARS)
    yd = _year_dir(pack, [2043, 2044, 2045], tmp_path / "years")
    dataset = _dataset(yd, pack)
    with pytest.raises(ValueError, match="CHANNEL_COUNT_MISMATCH"):
        compute_time_mean_z(dataset, C_OUT + 1)
