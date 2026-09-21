"""Tests for the weighted combination and the probabilistic scores (tasks 13-14).

    pytest -q makani_sfno/tests/sfno_ensemble/test_combine_and_scores.py

The load-bearing one is `test_crps_matches_makani_unweighted`.  We compute CRPS
ourselves because makani's `skillspread` kernel ignores its `ensemble_weights`
argument, so there is nothing to reuse for a weighted ensemble -- but "ours" must
still mean the same thing as makani's, or a lagged-ensemble CRPS and a training-side
CRPS are different quantities wearing one name.  That test pins the uniform-weight
case against makani's own kernel and skips cleanly where makani is unavailable.

The rest guard properties that are easy to get subtly wrong and impossible to notice
downstream: that the weighted estimators reduce EXACTLY to the uniform ones, that the
spread debiasing matches `E/(E-1)` at uniform weights, and that a perfectly calibrated
synthetic ensemble really does score SSR = 1 and a flat rank histogram.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sfno_ensemble import scores as S  # noqa: E402
from sfno_ensemble.combine import (  # noqa: E402
    member_weights,
    weighted_mean,
    weighted_spread,
)

E, C, H, W = 6, 3, 8, 16


def _stack(seed=0, spread=1.0):
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=(C, H, W))
    members = truth[None] + rng.normal(size=(E, C, H, W)) * spread
    return members, truth


def _uniform():
    return np.full((E, C), 1.0 / E)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

def test_weights_are_per_channel_and_favour_the_accurate_depth():
    """A channel that degrades fast must down-weight old members more than one that does not."""
    K = 8
    sigma = np.ones((K, 2))
    sigma[:, 0] = np.linspace(1.0, 8.0, K)    # degrades fast
    sigma[:, 1] = 1.0                          # flat
    w = member_weights(sigma, depths=[1, 8])
    assert np.allclose(w.sum(axis=0), 1.0)
    assert w[0, 0] > w[1, 0], "fast-degrading channel must prefer the shallow member"
    assert np.isclose(w[0, 1], w[1, 1]), "flat channel must weight both equally"
    assert np.isclose(w[0, 0] / w[1, 0], 64.0)   # (8/1)^2


def test_weights_reject_a_degenerate_sigma():
    with pytest.raises(ValueError, match="finite and positive"):
        member_weights(np.array([[1.0], [0.0]]), depths=[1, 2])


# ---------------------------------------------------------------------------
# Combination
# ---------------------------------------------------------------------------

def test_weighted_mean_reduces_to_the_plain_mean():
    members, _ = _stack()
    assert np.allclose(weighted_mean(members, _uniform()), members.mean(axis=0))


def test_weighted_spread_reduces_to_the_sample_std():
    """At uniform weights the 1/(1 - sum w^2) correction must be exactly E/(E-1)."""
    members, _ = _stack()
    got = weighted_spread(members, _uniform())
    want = members.std(axis=0, ddof=1)
    assert np.allclose(got, want), np.abs(got - want).max()


def test_weighted_spread_is_larger_than_the_biased_form():
    """The correction must inflate, not deflate -- under-dispersion is the failure mode."""
    members, _ = _stack()
    w = np.abs(np.random.default_rng(1).normal(size=(E, C))) + 0.1
    w = w / w.sum(axis=0)
    biased = np.sqrt(np.einsum("ec,echw->chw", w, (members - weighted_mean(members, w)) ** 2))
    assert np.all(weighted_spread(members, w) > biased)


def test_a_degenerate_weight_vector_collapses_to_one_member():
    """w -> one-hot must give that member back exactly, and zero spread."""
    members, _ = _stack()
    w = np.zeros((E, C))
    w[2] = 1.0
    assert np.allclose(weighted_mean(members, w), members[2])


# ---------------------------------------------------------------------------
# CRPS -- the equivalence pin
# ---------------------------------------------------------------------------

def test_crps_matches_makani_unweighted():
    """Our weighted CRPS at uniform weights must equal makani's fair skillspread kernel.

    makani's `_crps_skillspread_kernel` computes `E|X-y| - 0.5 * espread` with the fair
    `(E - 1 + alpha)/(E(E-1))` normalisation at alpha=1; ours is the energy form with
    the `1/(1 - sum w^2)` correction. They are the same estimator written two ways, and
    if they ever stop being, this fails rather than our scorecard quietly drifting from
    the training-side number.
    """
    # Not importorskip: off a compute node torch raises ValueError (not ImportError)
    # hunting libcublas, which importorskip does not catch. CLAUDE.md #3 is the rule
    # that keeps this suite on a node where the import works; the skip is the graceful
    # path elsewhere, and polaris_e3sm_port_test.pbs is where it really runs.
    try:
        import torch
        from makani.utils.losses import crps_loss as kern
    except Exception as exc:           # noqa: BLE001 -- environment, not logic
        pytest.skip(f"makani/torch unavailable here ({type(exc).__name__}); "
                    "run via polaris/polaris_e3sm_port_test.pbs")

    members, truth = _stack(seed=3)
    lat_w = S.equiangular_weights(H)
    ours = S.weighted_crps(members, truth, _uniform(), lat_w, fair=True)

    f = torch.from_numpy(members)                       # (E, C, H, W)
    o = torch.from_numpy(truth)                         # (C, H, W)
    theirs_map = kern._crps_skillspread_kernel(
        o, torch.sort(f, dim=0).values, torch.ones_like(f), 1.0).numpy()
    theirs = S.area_mean(theirs_map, lat_w)

    assert np.allclose(ours, theirs, rtol=1e-10), np.abs(ours - theirs).max()


def test_crps_of_a_perfect_forecast_is_zero():
    _, truth = _stack()
    members = np.broadcast_to(truth, (E, C, H, W)).copy()
    lat_w = S.equiangular_weights(H)
    assert np.allclose(S.weighted_crps(members, truth, _uniform(), lat_w, fair=False), 0.0)


def test_crps_of_a_single_effective_member_is_its_mae():
    """A one-hot weight vector leaves no spread, so CRPS must collapse to MAE.

    This is exactly the degeneracy the CRPS arm's config warns about -- CRPS over one
    member is not a distributional score -- and it is what makes `crps_shallowest` a
    fair deterministic baseline in the driver.
    """
    members, truth = _stack(seed=5)
    lat_w = S.equiangular_weights(H)
    w = np.zeros((E, C))
    w[0] = 1.0
    got = S.weighted_crps(members, truth, w, lat_w, fair=False)
    want = S.area_mean(np.abs(members[0] - truth), lat_w)
    assert np.allclose(got, want)


def test_fair_crps_exceeds_the_biased_one():
    """The plain estimator is biased low -- it flatters a small ensemble."""
    members, truth = _stack(seed=7)
    lat_w = S.equiangular_weights(H)
    fair = S.weighted_crps(members, truth, _uniform(), lat_w, fair=True)
    biased = S.weighted_crps(members, truth, _uniform(), lat_w, fair=False)
    assert np.all(fair < biased), "fair correction enlarges the spread term, lowering CRPS"


def test_weighting_changes_crps_at_all():
    """Guard against the failure makani has: weights silently ignored."""
    members, truth = _stack(seed=11)
    lat_w = S.equiangular_weights(H)
    skewed = np.zeros((E, C))
    skewed[0] = 0.9
    skewed[1:] = 0.1 / (E - 1)
    assert not np.allclose(S.weighted_crps(members, truth, skewed, lat_w),
                           S.weighted_crps(members, truth, _uniform(), lat_w))


# ---------------------------------------------------------------------------
# Spread / SSR / rank histogram
# ---------------------------------------------------------------------------

def test_perfectly_calibrated_ensemble_scores_ssr_one():
    """Truth drawn from the same distribution as the members => SSR ~ 1.

    Built so the test has teeth: with a large grid and many trials the fair correction
    is what brings a finite ensemble to 1 rather than systematically below it.
    """
    rng = np.random.default_rng(0)
    e, c, h, w_ = 20, 1, 64, 128
    mu = rng.normal(size=(c, h, w_))
    members = mu[None] + rng.normal(size=(e, c, h, w_))
    truth = mu + rng.normal(size=(c, h, w_))
    lat_w = S.equiangular_weights(h)
    w_ens = np.full((e, c), 1.0 / e)
    ss = S.spread_skill(members, truth, members.mean(axis=0), w_ens, lat_w)
    assert 0.9 < float(ss["ssr"][0]) < 1.1, ss["ssr"]


def test_under_dispersed_ensemble_scores_ssr_below_one():
    rng = np.random.default_rng(1)
    e, c, h, w_ = 20, 1, 64, 128
    mu = rng.normal(size=(c, h, w_))
    members = mu[None] + rng.normal(size=(e, c, h, w_)) * 0.25   # too confident
    truth = mu + rng.normal(size=(c, h, w_))
    lat_w = S.equiangular_weights(h)
    ss = S.spread_skill(members, truth, members.mean(axis=0), np.full((e, c), 1.0 / e), lat_w)
    assert float(ss["ssr"][0]) < 0.5, ss["ssr"]


def test_effective_sample_size_of_uniform_weights_is_E():
    members, truth = _stack()
    lat_w = S.equiangular_weights(H)
    ss = S.spread_skill(members, truth, members.mean(axis=0), _uniform(), lat_w)
    assert np.allclose(ss["e_eff"], E)


def test_rank_histogram_is_flat_for_a_calibrated_ensemble():
    rng = np.random.default_rng(2)
    e, c, h, w_ = 9, 1, 64, 128
    members = rng.normal(size=(e, c, h, w_))
    truth = rng.normal(size=(c, h, w_))
    hist = S.rank_histogram(members, truth, S.equiangular_weights(h))
    assert hist.shape == (c, e + 1)
    assert np.isclose(hist.sum(), c)                      # area-weighted, sums to 1/channel
    assert np.allclose(hist[0], 1.0 / (e + 1), atol=0.02), hist[0]


def test_rank_histogram_is_u_shaped_when_under_dispersed():
    rng = np.random.default_rng(3)
    e, c, h, w_ = 9, 1, 64, 128
    members = rng.normal(size=(e, c, h, w_)) * 0.1
    truth = rng.normal(size=(c, h, w_))
    hist = S.rank_histogram(members, truth, S.equiangular_weights(h))[0]
    assert hist[0] + hist[-1] > 0.7, hist


# ---------------------------------------------------------------------------
# Quadrature -- the same guard the K=56 scorer carries
# ---------------------------------------------------------------------------

def test_equiangular_weights_are_exact_cell_areas():
    """cos(lat) is not an approximation for this grid -- verify against sin-edge areas."""
    for nlat in (8, 64, 180):
        i = np.arange(nlat)
        edges = np.deg2rad(90.0 - i * (180.0 / nlat)), np.deg2rad(90.0 - (i + 1) * (180.0 / nlat))
        exact = np.sin(edges[0]) - np.sin(edges[1])
        exact = exact / exact.sum()
        assert np.allclose(S.equiangular_weights(nlat), exact, atol=1e-12)


def test_equiangular_differs_from_gauss_legendre():
    """If these agreed, change G's defect would be untestable here."""
    gl = np.polynomial.legendre.leggauss(180)[1]
    gl = gl / gl.sum()
    eq = S.equiangular_weights(180)
    assert np.abs(gl - eq).max() > 1e-6
    assert gl[0] / eq[0] > 1.4, "Gauss-Legendre over-weights the polar row"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
