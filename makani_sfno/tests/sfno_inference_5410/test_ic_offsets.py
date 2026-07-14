"""Tests for src/sfno_inference_5410/ic_offsets.py.

Coverage (per docs/2026-05-06_group_sfno_5410_eval_plan.md §H):

  - ``nwp_ic_offsets_5410(1460, K=60, n_ic=12, step=122)`` returns the
    pinned 12-element list ``[0, 122, ..., 1342]``;
  - identical for ``n_samples=1464`` (PlaSim leap year);
  - ``s + K < n_samples`` for all returned ``s`` in both cases;
  - the function raises if ``step × (n_ic-1) + K >= n_samples``.
"""
from __future__ import annotations

import pytest

from sfno_inference_5410.ic_offsets import nwp_ic_offsets_5410


_PINNED = [0, 122, 244, 366, 488, 610, 732, 854, 976, 1098, 1220, 1342]


class TestPinnedOffsets:
    def test_non_leap_year(self):
        assert nwp_ic_offsets_5410(1460, K=60, n_ic=12, step=122) == _PINNED

    def test_leap_year(self):
        assert nwp_ic_offsets_5410(1464, K=60, n_ic=12, step=122) == _PINNED

    def test_window_fits_non_leap(self):
        offsets = nwp_ic_offsets_5410(1460, K=60, n_ic=12, step=122)
        for s in offsets:
            assert s + 60 < 1460

    def test_window_fits_leap(self):
        offsets = nwp_ic_offsets_5410(1464, K=60, n_ic=12, step=122)
        for s in offsets:
            assert s + 60 < 1464


class TestOverrunDetection:
    def test_raises_when_window_overruns(self):
        # last_s = (12-1) * 122 = 1342. last_s + K = 1342 + 60 = 1402.
        # Set n_samples = 1402 → 1402 >= 1402 → must raise.
        with pytest.raises(ValueError, match="overruns year"):
            nwp_ic_offsets_5410(1402, K=60, n_ic=12, step=122)

    def test_raises_with_too_many_ics(self):
        # 13 ICs at stride 122: last_s = 1464. last_s + 60 = 1524 > 1460.
        with pytest.raises(ValueError, match="overruns year"):
            nwp_ic_offsets_5410(1460, K=60, n_ic=13, step=122)


class TestParameterization:
    def test_default_args(self):
        assert nwp_ic_offsets_5410(1460) == _PINNED

    def test_custom_step(self):
        assert nwp_ic_offsets_5410(2000, K=10, n_ic=5, step=100) == [
            0, 100, 200, 300, 400,
        ]
