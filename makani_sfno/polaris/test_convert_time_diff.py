"""Tests for the converter's time_diff_stds accumulator (port B step 1, handoff §3).

Synthetic packed files at the real channel width on a 2x3 grid; STATS_CHUNK_T is
shrunk so chunk seams are exercised. Needs numpy + h5py only.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import convert_e3sm_to_makani_alldata as C  # noqa: E402


@pytest.fixture
def pack(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "H", 2)
    monkeypatch.setattr(C, "W", 3)
    monkeypatch.setattr(C, "STATS_CHUNK_T", 2)
    rng = np.random.default_rng(1)
    train = tmp_path / "train"
    train.mkdir()
    years = {}
    for y, T, offset in ((2015, 7, 0.0), (2016, 5, 1000.0)):
        # a large level jump between files: a pair spanning the boundary would
        # dominate the std, so the test can see it if it is (wrongly) counted
        x = offset + np.cumsum(rng.normal(size=(T, C.N_TARGET, 2, 3)), axis=0)
        x = x.astype(np.float32)
        with h5py.File(train / f"{y}.h5", "w") as f:
            f["fields_state"] = x[:, :C.N_STATE]
            f["fields_diagnostic"] = x[:, C.N_STATE:]
            f["forcing"] = np.zeros((T, C.N_FORCING, 2, 3), np.float32)
        years[y] = x.astype(np.float64)
    return tmp_path, years


def _reference(years):
    d = np.concatenate([np.diff(x, axis=0) for x in years.values()], axis=0)
    return d.std(axis=(0, 2, 3))


def test_per_file_diffs_never_span_files(pack):
    root, years = pack
    parts = [C._accumulate_time_diff_file(str(root / "train" / f"{y}.h5")) for y in years]
    got = C._time_diff_stds(sum(p[0] for p in parts), sum(p[1] for p in parts),
                            sum(p[2] for p in parts)).ravel()
    np.testing.assert_allclose(got, _reference(years), rtol=1e-5)
    # and the spanning version really would differ (the test has teeth)
    spanning = np.diff(np.concatenate(list(years.values())), axis=0).std(axis=(0, 2, 3))
    assert np.all(np.abs(spanning - got) / got > 1.0)


def test_stats_pass_writes_the_same_statistic(pack):
    root, years = pack
    accum = C._accumulate_stats_from_packed(str(root / "train"))
    assert accum["n_d"] == sum((x.shape[0] - 1) for x in years.values()) * 2 * 3
    got = C._time_diff_stds(accum["sum_d"], accum["sumsq_d"], accum["n_d"]).ravel()
    np.testing.assert_allclose(got, _reference(years), rtol=1e-5)
    stats = root / "stats"
    C._write_stats(str(stats), accum)
    arr = np.load(stats / "time_diff_stds.npy")
    assert arr.shape == np.load(stats / "global_stds.npy").shape == (1, C.N_TARGET, 1, 1)


def test_time_diff_only_guard_refuses_an_incomplete_split(pack):
    root, _ = pack
    with pytest.raises(SystemExit, match="STATS_ONLY_INCOMPLETE"):
        C._require_complete_train(str(root), [2015, 2016, 2017])
    C._require_complete_train(str(root), [2015, 2016])  # complete: no exit
