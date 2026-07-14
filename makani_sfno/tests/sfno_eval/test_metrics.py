"""Tests for src/sfno_eval/metrics.py.

Coverage (per docs/sfno_eval_plan.md §D.1, §D.2, §D.3, §B.5, and §H):

  - ``rmse_lat_weighted``: matches a hand-computed reference at 1e-6;
    rejects shape mismatches.
  - ``acc``: =+1 when pred == truth (after centring), =-1 when
    pred = -truth, ~0 on uncorrelated noise.
  - ``bias_map``: equals the IC-mean of the residual.
  - ``legendre_gauss_lat_weights``: weights sum to 1; nlat=64 returns 64
    weights; symmetric around the equator (weights[i] == weights[N-1-i]).
"""
from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from sfno_eval import metrics  # noqa: E402


# ---------------------------------------------------------------------------
# legendre_gauss_lat_weights
# ---------------------------------------------------------------------------

class TestLatWeights:
    def test_sums_to_one(self):
        w = metrics.legendre_gauss_lat_weights(64)
        assert torch.allclose(w.sum(), torch.tensor(1.0, dtype=w.dtype), atol=1e-6)

    def test_correct_length(self):
        for nlat in (16, 32, 64, 128):
            w = metrics.legendre_gauss_lat_weights(nlat)
            assert w.shape == (nlat,)

    def test_symmetric_about_equator(self):
        w = metrics.legendre_gauss_lat_weights(64)
        # nodes are symmetric about 0; weights mirror.
        assert torch.allclose(w, w.flip(0), atol=1e-6)


# ---------------------------------------------------------------------------
# rmse_lat_weighted
# ---------------------------------------------------------------------------

class TestRmseLatWeighted:
    def _hand_rmse(self, pred, truth, w):
        err2 = (pred - truth) ** 2
        err2_lon = err2.mean(axis=-1)
        err2_w = (err2_lon * w).sum(axis=-1)
        return np.sqrt(err2_w)

    def test_matches_hand_computation(self):
        rng = np.random.default_rng(0)
        H, W = 8, 16
        pred = torch.from_numpy(rng.standard_normal((3, 4, H, W), dtype=np.float32))
        truth = torch.from_numpy(rng.standard_normal((3, 4, H, W), dtype=np.float32))
        # uniform-ish latitude weights (sum to 1)
        w_np = np.ones(H, dtype=np.float32) / H
        w = torch.from_numpy(w_np)

        got = metrics.rmse_lat_weighted(pred, truth, w).numpy()
        want = self._hand_rmse(pred.numpy(), truth.numpy(), w_np)
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)

    def test_zero_for_perfect_prediction(self):
        H, W = 4, 8
        pred = torch.zeros(2, H, W)
        truth = pred.clone()
        w = torch.ones(H) / H
        out = metrics.rmse_lat_weighted(pred, truth, w)
        assert torch.allclose(out, torch.zeros(2))

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape"):
            metrics.rmse_lat_weighted(
                torch.zeros(4, 8), torch.zeros(5, 8), torch.ones(4) / 4
            )

    def test_rejects_lat_size_mismatch(self):
        with pytest.raises(ValueError, match="lat"):
            metrics.rmse_lat_weighted(
                torch.zeros(4, 8), torch.zeros(4, 8), torch.ones(5) / 5
            )


# ---------------------------------------------------------------------------
# acc
# ---------------------------------------------------------------------------

class TestAcc:
    def test_perfect_correlation(self):
        rng = np.random.default_rng(1)
        H, W = 4, 8
        truth_anom = torch.from_numpy(rng.standard_normal((H, W), dtype=np.float32))
        clim = torch.zeros(H, W)
        truth = truth_anom + clim
        pred = truth.clone()  # identical
        w = torch.ones(H) / H
        out = metrics.acc(pred, truth, clim, w)
        assert torch.allclose(out, torch.tensor(1.0), atol=1e-5)

    def test_anti_correlation(self):
        rng = np.random.default_rng(2)
        H, W = 4, 8
        truth_anom = torch.from_numpy(rng.standard_normal((H, W), dtype=np.float32))
        clim = torch.zeros(H, W)
        truth = truth_anom + clim
        pred = -truth_anom + clim  # exact negative anomaly
        w = torch.ones(H) / H
        out = metrics.acc(pred, truth, clim, w)
        assert torch.allclose(out, torch.tensor(-1.0), atol=1e-5)

    def test_uncorrelated_near_zero(self):
        """Independent random noise should give ACC ≈ 0 over a large field."""
        rng = np.random.default_rng(3)
        H, W = 64, 128
        truth = torch.from_numpy(rng.standard_normal((H, W), dtype=np.float32))
        pred = torch.from_numpy(rng.standard_normal((H, W), dtype=np.float32))
        clim = torch.zeros(H, W)
        w = metrics.legendre_gauss_lat_weights(H)
        out = metrics.acc(pred, truth, clim, w)
        assert abs(float(out)) < 0.05

    def test_handles_leading_dims(self):
        H, W = 4, 8
        n_lead = 6
        truth = torch.zeros(n_lead, H, W) + 1.0
        clim = torch.zeros(H, W)
        pred = truth.clone()
        w = torch.ones(H) / H
        out = metrics.acc(pred, truth, clim, w)
        assert out.shape == (n_lead,)
        # Non-zero anomaly identical pred/truth → ACC ≈ 1 elementwise
        assert torch.all(out > 0.99)


# ---------------------------------------------------------------------------
# bias_map
# ---------------------------------------------------------------------------

class TestBiasMap:
    def test_equals_mean_residual(self):
        rng = np.random.default_rng(4)
        n_ic, H, W = 7, 4, 8
        pred = torch.from_numpy(rng.standard_normal((n_ic, H, W), dtype=np.float32))
        truth = torch.from_numpy(rng.standard_normal((n_ic, H, W), dtype=np.float32))
        out = metrics.bias_map(pred, truth)
        want = (pred - truth).mean(dim=0)
        assert torch.allclose(out, want)

    def test_handles_extra_lead_dim(self):
        n_ic, n_lead, H, W = 5, 3, 4, 8
        pred = torch.ones(n_ic, n_lead, H, W) * 2.0
        truth = torch.ones(n_ic, n_lead, H, W) * 1.0
        out = metrics.bias_map(pred, truth)
        assert out.shape == (n_lead, H, W)
        assert torch.allclose(out, torch.ones(n_lead, H, W))


# ---------------------------------------------------------------------------
# cache_lat_weights
# ---------------------------------------------------------------------------

class TestCacheLatWeights:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "stats" / "lat_weights.npy"
        metrics.cache_lat_weights(path, nlat=64)
        assert path.is_file()
        loaded = np.load(path)
        assert loaded.shape == (64,)
        np.testing.assert_allclose(loaded.sum(), 1.0, atol=1e-6)

    def test_idempotent(self, tmp_path):
        path = tmp_path / "lat.npy"
        metrics.cache_lat_weights(path, nlat=32)
        metrics.cache_lat_weights(path, nlat=32)  # second run overwrites cleanly
        assert path.is_file()


# ---------------------------------------------------------------------------
# rmse_lat_weighted_masked
# ---------------------------------------------------------------------------

class TestRmseLatWeightedMasked:
    def test_identity_mask_matches_unmasked(self):
        rng = np.random.default_rng(10)
        H, W = 8, 16
        pred = torch.from_numpy(rng.standard_normal((H, W), dtype=np.float32))
        truth = torch.from_numpy(rng.standard_normal((H, W), dtype=np.float32))
        w = torch.from_numpy(np.ones(H, dtype=np.float32) / H)
        mask = torch.ones(H, W, dtype=torch.bool)
        got = metrics.rmse_lat_weighted_masked(pred, truth, w, mask)
        want = metrics.rmse_lat_weighted(pred, truth, w)
        assert torch.allclose(got, want, atol=1e-6, rtol=1e-6)

    def test_hand_3x4_case(self):
        # Construct a tiny case where the masked answer is computable by hand.
        # Use err = pred - truth = 1 everywhere; w_lat uniform 1/H => masked
        # RMSE = sqrt(sum_masked w_ij * 1) / sqrt(sum_masked w_ij) = 1.
        H, W = 3, 4
        pred = torch.zeros(H, W)
        truth = -torch.ones(H, W)            # err == 1 everywhere
        w = torch.ones(H) / H
        mask = torch.zeros(H, W, dtype=torch.bool)
        mask[0, :] = True                     # one full row kept
        mask[1, 2] = True                     # one extra cell kept
        got = metrics.rmse_lat_weighted_masked(pred, truth, w, mask)
        assert torch.allclose(got, torch.tensor(1.0), atol=1e-6)

    def test_all_false_returns_nan(self):
        H, W = 4, 8
        pred = torch.ones(H, W)
        truth = torch.zeros(H, W)
        w = torch.ones(H) / H
        mask = torch.zeros(H, W, dtype=torch.bool)
        out = metrics.rmse_lat_weighted_masked(pred, truth, w, mask)
        assert torch.isnan(out)

    def test_per_row_mask_renormalises(self):
        # Two rows: row 0 fully masked out, row 1 fully kept. err = 2 in row 1.
        # Expected RMSE = sqrt( (1*0 + 1*4) / (0 + 1) ) but renormalising by
        # the remaining lat-weight: only row 1's weight contributes.
        H, W = 2, 4
        pred = torch.zeros(H, W)
        truth = torch.zeros(H, W)
        truth[1, :] = -2.0                    # err = 2 in row 1
        w = torch.tensor([0.5, 0.5])
        mask = torch.zeros(H, W, dtype=torch.bool)
        mask[1, :] = True
        out = metrics.rmse_lat_weighted_masked(pred, truth, w, mask)
        assert torch.allclose(out, torch.tensor(2.0), atol=1e-6)

    def test_rejects_mask_shape_mismatch(self):
        with pytest.raises(ValueError, match="mask"):
            metrics.rmse_lat_weighted_masked(
                torch.zeros(4, 8), torch.zeros(4, 8),
                torch.ones(4) / 4, torch.ones(5, 8, dtype=torch.bool),
            )


# ---------------------------------------------------------------------------
# acc_masked
# ---------------------------------------------------------------------------

class TestAccMasked:
    def test_identity_mask_matches_unmasked(self):
        rng = np.random.default_rng(20)
        H, W = 8, 16
        pred = torch.from_numpy(rng.standard_normal((H, W), dtype=np.float32))
        truth = torch.from_numpy(rng.standard_normal((H, W), dtype=np.float32))
        clim = torch.from_numpy(rng.standard_normal((H, W), dtype=np.float32))
        w = torch.from_numpy(np.ones(H, dtype=np.float32) / H)
        mask = torch.ones(H, W, dtype=torch.bool)
        got = metrics.acc_masked(pred, truth, clim, w, mask)
        want = metrics.acc(pred, truth, clim, w)
        assert torch.allclose(got, want, atol=1e-6, rtol=1e-6)

    def test_masked_region_invariance(self):
        # Flipping sign of pred/truth anomalies inside the masked-OUT region
        # must not change ACC over the kept region.
        rng = np.random.default_rng(21)
        H, W = 6, 12
        clim = torch.zeros(H, W)
        truth = torch.from_numpy(rng.standard_normal((H, W), dtype=np.float32))
        pred = truth + 0.1 * torch.from_numpy(
            rng.standard_normal((H, W), dtype=np.float32)
        )
        w = torch.ones(H) / H
        # Keep the top half, mask out the bottom half.
        mask = torch.zeros(H, W, dtype=torch.bool)
        mask[: H // 2] = True
        base = metrics.acc_masked(pred, truth, clim, w, mask)
        # Garbage in the masked-out half: should leave acc unchanged.
        pred_pert = pred.clone()
        truth_pert = truth.clone()
        pred_pert[H // 2 :] = -100.0
        truth_pert[H // 2 :] = 100.0
        perturbed = metrics.acc_masked(pred_pert, truth_pert, clim, w, mask)
        assert torch.allclose(base, perturbed, atol=1e-6, rtol=1e-6)

    def test_all_false_returns_nan(self):
        H, W = 4, 8
        pred = torch.ones(H, W)
        truth = torch.zeros(H, W)
        clim = torch.zeros(H, W)
        w = torch.ones(H) / H
        mask = torch.zeros(H, W, dtype=torch.bool)
        out = metrics.acc_masked(pred, truth, clim, w, mask)
        assert torch.isnan(out)
