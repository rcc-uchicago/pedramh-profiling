"""Tests for the converter's parallel stats pass (`--stats-only --workers N`).

The 2020-2044 train view (polaris_pack_alldata_trainview.pbs) needs the full
stats pass inside one 1-h debug job; sequentially 30 years took 52 min
(7565734) before the time-diff sums were added. The pool path must give the
statistics the sequential path gives. Needs numpy + h5py only.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import convert_e3sm_to_makani_alldata as C  # noqa: E402

STATS = ("global_means.npy", "global_stds.npy", "time_means.npy", "time_diff_stds.npy",
         "forcing_global_means.npy", "forcing_global_stds.npy", "forcing_time_means.npy")


def _write_pack(train, rng, H, W, lengths):
    train.mkdir(parents=True)
    data = {}
    for y, T in lengths.items():
        x = (100.0 * (y - 2015) + np.cumsum(rng.normal(size=(T, C.N_TARGET, H, W)), axis=0))
        x = x.astype(np.float32)
        fo = rng.normal(size=(T, C.N_FORCING, H, W)).astype(np.float32)
        with h5py.File(train / f"{y}.h5", "w") as f:
            f["fields_state"] = x[:, :C.N_STATE]
            f["fields_diagnostic"] = x[:, C.N_STATE:]
            f["forcing"] = fo
        data[y] = x.astype(np.float64)
    return data


@pytest.fixture
def small(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "H", 2)
    monkeypatch.setattr(C, "W", 3)
    monkeypatch.setattr(C, "STATS_CHUNK_T", 2)
    data = _write_pack(tmp_path / "train", np.random.default_rng(3), 2, 3,
                       {2020: 7, 2021: 5, 2022: 4})
    return tmp_path / "train", data


def _assert_accum_close(a, b, rtol):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], (int, np.integer)):
            assert a[k] == b[k], k
        else:
            np.testing.assert_allclose(a[k], b[k], rtol=rtol, err_msg=k)


def test_per_file_partials_combine_to_the_sequential_pass(small):
    train, _ = small
    files = sorted(str(p) for p in train.glob("*.h5"))
    seq = C._accumulate_stats_from_packed(str(train), workers=1)
    comb = C._combine_stats([C._accumulate_stats_file(p) for p in files])
    _assert_accum_close(comb, seq, rtol=1e-12)


def test_combine_has_teeth_a_double_counted_file_is_caught(small):
    train, _ = small
    files = sorted(str(p) for p in train.glob("*.h5"))
    seq = C._accumulate_stats_from_packed(str(train), workers=1)
    bad = C._combine_stats([C._accumulate_stats_file(p) for p in files + files[:1]])
    with pytest.raises(AssertionError):
        _assert_accum_close(bad, seq, rtol=1e-12)


def test_sequential_pass_matches_numpy_reference(small):
    train, data = small
    accum = C._accumulate_stats_from_packed(str(train), workers=1)
    x = np.concatenate(list(data.values()), axis=0)
    n = accum["n"]
    np.testing.assert_allclose(accum["sum_t"] / n, x.mean(axis=(0, 2, 3)), rtol=1e-10)
    np.testing.assert_allclose(accum["tsum_t"] / accum["t_count"], x.mean(axis=0), rtol=1e-10)
    # one-step diffs restart at every file (2020 -> 2021 is a +100 jump that must not count)
    d = np.concatenate([np.diff(v, axis=0) for v in data.values()], axis=0)
    assert accum["n_d"] == d.shape[0] * 2 * 3


def test_pool_matches_sequential_at_real_shape(tmp_path):
    """The real spawn Pool, at the real 180x360 grid (children re-import the module,
    so a monkeypatched grid would not reach them). Written stats agree to float32."""
    train = tmp_path / "train"
    _write_pack(train, np.random.default_rng(5), C.H, C.W, {2020: 3, 2021: 2})
    out = {}
    for w in (1, 2):
        d = tmp_path / f"stats_w{w}"
        C._write_stats(str(d), C._accumulate_stats_from_packed(str(train), workers=w))
        out[w] = {name: np.load(d / name) for name in STATS}
    for name in STATS:
        np.testing.assert_allclose(out[2][name], out[1][name], rtol=1e-6, atol=1e-6,
                                   err_msg=name)
