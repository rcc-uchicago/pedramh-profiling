"""Tests for make_capped_weights.capped_weights (architect review §3)."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from make_capped_weights import capped_weights  # noqa: E402


def test_equal_ratios_reproduce_makani_constant():
    # makani 'constant' = ones / C; equal r must give exactly that
    w = capped_weights([2.0, 4.0, 8.0], [1.0, 2.0, 4.0], 30)
    np.testing.assert_allclose(w, np.full(3, 1 / 3))


def test_cap_bounds_the_slow_channel_and_prect_is_not_deleted():
    # Z3_l17-like (sigma/delta ~9518), PRECT-like (~1.04, tiny PHYSICAL delta), fast (0.73)
    sigma = [0.271, 8.3e-8 * 1.044, 307.0]
    delta = [0.271 / 9518, 8.3e-8, 422.0]
    w = capped_weights(sigma, delta, 30)
    rel = w / w.min()
    assert rel[0] == pytest.approx(30.0)          # capped, not 9518
    assert rel[1] == pytest.approx(1.044, rel=1e-3)  # PRECT keeps ~1x -- no physical clamp
    assert w.sum() == pytest.approx(1.0)


def test_degenerate_delta_fails_loud():
    with pytest.raises(ValueError, match="DEGENERATE_DELTA"):
        capped_weights([1.0], [0.0], 30)
