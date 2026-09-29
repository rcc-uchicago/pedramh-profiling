"""rollout_driver changes for polaris_makani_ace2_ports_handoff.md.

* port F: a run trained on a channel subset of its pack carries full-width stats
  and must de-normalize with them indexed by ``out_channels``; full-width runs
  must take exactly the old path.
* port A: ``force_positive_names`` clamps at physical 0 after every step, and is
  OFF unless configured.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sfno_inference.rollout_driver import (  # noqa: E402
    _apply_force_positive,
    _force_positive_setup,
    _load_run_norm_stats,
)


def _stats(tmp_path, n):
    mean = np.arange(n, dtype=np.float32).reshape(1, n, 1, 1)
    std = (1.0 + np.arange(n, dtype=np.float32)).reshape(1, n, 1, 1)
    np.save(tmp_path / "m.npy", mean)
    np.save(tmp_path / "s.npy", std)
    return str(tmp_path / "m.npy"), str(tmp_path / "s.npy")


def test_full_width_stats_path_is_unchanged(tmp_path):
    m, s = _stats(tmp_path, 5)
    ep = SimpleNamespace(global_means_path=m, global_stds_path=s, N_out_channels=5,
                         out_channels=[0, 1, 2, 3, 4])
    b, sc = _load_run_norm_stats(ep, "cpu")
    assert b.flatten().tolist() == [0, 1, 2, 3, 4]
    assert sc.flatten().tolist() == [1, 2, 3, 4, 5]


def test_subset_stats_are_indexed_by_out_channels(tmp_path):
    m, s = _stats(tmp_path, 5)
    ep = SimpleNamespace(global_means_path=m, global_stds_path=s, N_out_channels=3,
                         out_channels=[0, 3, 4])
    b, sc = _load_run_norm_stats(ep, "cpu")
    assert b.flatten().tolist() == [0, 3, 4]
    assert sc.flatten().tolist() == [1, 4, 5]


def test_subset_without_out_channels_still_fails_loud(tmp_path):
    m, s = _stats(tmp_path, 5)
    ep = SimpleNamespace(global_means_path=m, global_stds_path=s, N_out_channels=3)
    with pytest.raises(RuntimeError, match="unexpected global_means shape"):
        _load_run_norm_stats(ep, "cpu")


def test_force_positive_is_off_by_default():
    ep = SimpleNamespace(channel_names=["a", "b"])
    idx, floor = _force_positive_setup(ep, torch.zeros(1, 2, 1, 1), torch.ones(1, 2, 1, 1))
    assert idx is None and floor is None
    x = torch.randn(1, 2, 3, 4)
    assert _apply_force_positive(x, idx, floor) is x


def test_force_positive_clamps_at_physical_zero_only_where_named():
    bias = torch.tensor([10.0, 5.0, 2.0]).reshape(1, 3, 1, 1)
    scale = torch.tensor([2.0, 1.0, 4.0]).reshape(1, 3, 1, 1)
    ep = SimpleNamespace(channel_names=["T", "PRECT", "Q"], force_positive_names=["PRECT", "Q"])
    idx, floor = _force_positive_setup(ep, bias, scale)
    phys = torch.tensor([-3.0, -1.0, 7.0]).reshape(1, 3, 1, 1).expand(1, 3, 2, 2)
    z = (phys - bias) / scale
    out = _apply_force_positive(z, idx, floor) * scale + bias
    assert torch.allclose(out[:, 0], phys[:, 0])            # T untouched, even negative
    assert torch.allclose(out[:, 1], torch.zeros(1, 2, 2))  # PRECT -1 -> 0
    assert torch.allclose(out[:, 2], phys[:, 2])            # Q already positive


def test_force_positive_unknown_name_fails_loud():
    ep = SimpleNamespace(channel_names=["T"], force_positive_names=["SOILWATER_10CM"])
    with pytest.raises(ValueError, match="force_positive_names"):
        _force_positive_setup(ep, torch.zeros(1, 1, 1, 1), torch.ones(1, 1, 1, 1))
