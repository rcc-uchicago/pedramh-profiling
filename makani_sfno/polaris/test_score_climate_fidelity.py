"""Tests for score_climate_fidelity.py's area-weighted bias/RMSE math."""
from __future__ import annotations

import numpy as np

from score_climate_fidelity import area_weighted_bias_rmse


def _weights(nlat):
    lat = 90.0 - (np.arange(nlat) + 0.5) * 180.0 / nlat
    w = np.cos(np.deg2rad(lat))
    return w / w.sum()


def test_identical_fields_are_zero():
    rng = np.random.default_rng(0)
    true = rng.standard_normal((3, 8, 16))
    bias, rmse = area_weighted_bias_rmse(true.copy(), true, _weights(8))
    assert np.allclose(bias, 0.0)
    assert np.allclose(rmse, 0.0)


def test_uniform_offset_is_exact_bias_and_rmse():
    true = np.zeros((2, 8, 16))
    member = true + np.array([3.0, -2.0])[:, None, None]
    weights = _weights(8)
    bias, rmse = area_weighted_bias_rmse(member, true, weights)
    assert np.allclose(bias, [3.0, -2.0])
    assert np.allclose(rmse, [3.0, 2.0])  # constant offset: RMSE == |bias|


def test_area_weighting_upweights_the_pole_row_less():
    # A difference concentrated at the (low cos-lat weight) first row should
    # bias less than the same-magnitude difference concentrated at the equator.
    nlat = 8
    weights = _weights(nlat)
    true = np.zeros((1, nlat, 4))
    pole = true.copy()
    pole[0, 0, :] = 10.0
    equator = true.copy()
    equator[0, nlat // 2, :] = 10.0
    bias_pole, _ = area_weighted_bias_rmse(pole, true, weights)
    bias_eq, _ = area_weighted_bias_rmse(equator, true, weights)
    assert bias_pole[0] < bias_eq[0]
    assert weights[0] < weights[nlat // 2]  # sanity: pole row is lower-weighted


def test_bias_can_cancel_while_rmse_cannot():
    # A field that's +X over half the globe and -X over the other half has
    # zero area-mean bias but nonzero pattern RMSE -- bias alone would hide it.
    nlat, nlon = 8, 4
    weights = np.full(nlat, 1.0 / nlat)  # uniform weights for an exact 50/50 split
    true = np.zeros((1, nlat, nlon))
    member = true.copy()
    member[0, : nlat // 2, :] = 5.0
    member[0, nlat // 2:, :] = -5.0
    bias, rmse = area_weighted_bias_rmse(member, true, weights)
    assert np.allclose(bias, 0.0)
    assert np.allclose(rmse, 5.0)
