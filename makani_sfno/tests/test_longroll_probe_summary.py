"""Unit tests for the long-rollout probe read-out (torch-free, numpy only).

Pins the three reductions `polaris/longroll_probe_summary.py` makes before printing:
where a median curve crosses a level, whether it is decelerating, and whether a
training-year arm is "better" broadly enough to flag.  Run anywhere with numpy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_POLARIS = Path(__file__).resolve().parents[1] / "polaris"
if str(_POLARIS) not in sys.path:
    sys.path.insert(0, str(_POLARIS))

import longroll_probe_summary as S  # noqa: E402


def _leads(k: int) -> np.ndarray:
    return 6 * np.arange(1, k + 1)


def test_crossing_hours_first_hit_and_never():
    lead_h = _leads(10)
    med = np.linspace(0.1, 1.0, 10)
    assert S.crossing_hours(lead_h, med, 0.5) == 6 * 5   # med[4] = 0.5 exactly
    assert S.crossing_hours(lead_h, med, 1.414) is None


def test_crossing_hours_ignores_nan():
    lead_h = _leads(4)
    med = np.array([np.nan, 0.5, 2.0, 2.5])
    assert S.crossing_hours(lead_h, med, 1.0) == 18


def test_slope_ratio_linear_is_one_and_saturating_below_one():
    lead_h = _leads(200)
    linear = 0.001 * lead_h
    assert S.slope_ratio(lead_h, linear) == pytest.approx(1.0)
    saturating = 1.0 - np.exp(-lead_h / 200.0)
    assert S.slope_ratio(lead_h, saturating) < 0.5


def test_blowup_and_worst_channels_sorted_worst_first():
    chan = ["a", "b", "c", "d"]
    last = np.array([0.5, 4.0, np.nan, 3.5])
    assert S.blowup_channels(last, chan) == [(4.0, "b"), (3.5, "d")]
    assert [c for _, c in S.worst_channels(last, chan, n=2)] == ["b", "d"]


def test_contamination_table_and_flag():
    lead_h = _leads(56)
    test = np.ones((56, 8))
    # Training arm 20 % better on 7 of 8 channels at every lead -> flag.
    train = np.ones((56, 8)) * 0.8
    train[:, 0] = 1.2
    rows = S.contamination_table(lead_h, train, test)
    assert [h for h, _, _ in rows] == [6, 24, 72, 168, 336]
    h, ratio, frac = rows[0]
    assert h == 6 and ratio == pytest.approx(0.8) and frac == pytest.approx(7 / 8)
    assert S.contamination_flag(rows) is True
    # Same size of gap but on only half the channels -> no flag: not broad.
    narrow = np.ones((56, 8))
    narrow[:, :4] = 0.7
    assert S.contamination_flag(S.contamination_table(lead_h, narrow, test)) is False
    # Broad but tiny gap -> no flag: not substantial.
    tiny = np.ones((56, 8)) * 0.97
    assert S.contamination_flag(S.contamination_table(lead_h, tiny, test)) is False


def test_contamination_table_skips_absent_leads():
    lead_h = _leads(10)   # only reaches 60 h
    rows = S.contamination_table(lead_h, np.ones((10, 3)), np.ones((10, 3)))
    assert [h for h, _, _ in rows] == [6, 24]
