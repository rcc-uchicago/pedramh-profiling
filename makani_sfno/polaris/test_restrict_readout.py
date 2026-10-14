"""Tests for restrict_readout.restrict (port F baseline re-take, handoff §6a)."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from restrict_readout import max_abs_diff, restrict  # noqa: E402
from score_rollout_nc import readout_from_curves  # noqa: E402

LEADS = np.arange(6, 337, 6)
CHAN = ["A", "B", "SOIL", "TSOI", "C"]


def _curves():
    rng = np.random.default_rng(0)
    k, c = len(LEADS), len(CHAN)
    nrmse = np.cumsum(rng.uniform(0.0, 0.05, (k, c)), axis=0)
    nrmse[:, 2] *= 40.0  # the dropped channel is the outlier, as SOILWATER is
    vr = 1.0 + rng.normal(0, 0.01, (k, c))
    acc = 1.0 - nrmse / (nrmse.max() + 1)
    return nrmse, vr, acc


def test_empty_drop_equals_the_unrestricted_readout():
    nrmse, vr, acc = _curves()
    r, _, kept = restrict(LEADS, CHAN, nrmse, vr, acc, [])
    assert kept == CHAN
    assert max_abs_diff(r, readout_from_curves(LEADS, nrmse, vr, acc)) == 0.0


def test_drop_removes_exactly_those_columns():
    nrmse, vr, acc = _curves()
    r, _, kept = restrict(LEADS, CHAN, nrmse, vr, acc, ["SOIL", "TSOI"])
    assert kept == ["A", "B", "C"]
    keep = [0, 1, 4]
    ref = readout_from_curves(LEADS, nrmse[:, keep], vr[:, keep], acc[:, keep])
    assert max_abs_diff(r, ref) == 0.0
    # the worst-channel clause no longer sees the dropped outlier
    assert r["max_channel_nrmse336"] < nrmse[-1, 2]


def test_unknown_name_fails_loud():
    nrmse, vr, acc = _curves()
    with pytest.raises(ValueError, match="UNKNOWN_CHANNEL"):
        restrict(LEADS, CHAN, nrmse, vr, acc, ["SOILWATER_10CM"])
