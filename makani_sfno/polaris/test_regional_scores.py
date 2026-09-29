"""Tests for regional_scores (port F land panel + acceptance rule)."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from regional_scores import BAR, LEADS_H, PANEL, land_rule, masked_metrics, region_weights  # noqa: E402


def test_all_ones_mask_equals_the_global_scorer():
    torch = pytest.importorskip("torch")
    from sfno_eval import metrics as M

    rng = np.random.default_rng(0)
    H, W = 8, 16
    p, t, c = (rng.normal(size=(3, H, W)) for _ in range(3))
    lw = M.lat_weights(H, "equiangular").to(torch.float64)
    r, a, at = masked_metrics(p, t, c, region_weights(lw.numpy(), np.ones((H, W), bool)))
    P, T, Cl = (torch.from_numpy(x) for x in (p, t, c))
    np.testing.assert_allclose(r, M.rmse_lat_weighted(P, T, lw).numpy(), rtol=1e-12)
    np.testing.assert_allclose(a, M.acc(P, T, Cl, lw).numpy(), rtol=1e-9)
    np.testing.assert_allclose(at, M.rmse_lat_weighted(T, Cl, lw).numpy(), rtol=1e-12)


def test_mask_ignores_off_region_cells():
    lw = np.full(4, 0.25)
    mask = np.zeros((4, 4), bool)
    mask[:2] = True
    p = np.zeros((1, 4, 4))
    t = np.zeros((1, 4, 4))
    t[:, 2:] = 100.0                      # error only off-mask
    r, _, _ = masked_metrics(p, t, np.zeros((1, 4, 4)), region_weights(lw, mask))
    assert r[0] == 0.0
    with pytest.raises(ValueError, match="EMPTY_REGION"):
        region_weights(lw, np.zeros((4, 4), bool))


def _table(val):
    return {("land", ch, h): {"nrmse": val, "acc": 0.5} for ch in PANEL for h in LEADS_H}


def test_rule_branches():
    base = _table(1.0)
    assert land_rule(base, _table(1.0 + BAR * 0.99), 0.97, 0.99)["verdict"] == "ACCEPT"
    worse = _table(1.0)
    worse[("land", "PRECT", 336)]["nrmse"] = 1.0 + 2 * BAR
    assert land_rule(base, worse, 0.97, 0.96)["verdict"] == "SOIL_FEEDBACK_MATTERED"
    assert land_rule(base, worse, 0.97, 0.98)["verdict"] == "REJECT"
    nan = _table(1.0)
    nan[("land", "TMQ", 126)]["nrmse"] = float("nan")
    assert land_rule(base, nan, 0.97, 0.96)["verdict"] != "ACCEPT"
