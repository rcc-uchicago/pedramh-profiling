"""Probabilistic scores for a weighted lagged ensemble (stage 5 / task 14).

CRPS, spread, spread-skill ratio and rank histogram, all latitude-weighted on the
**equiangular** E3SM grid.  Everything here is `(E, C, H, W)` numpy in physical units
-- the ensemble axis first, matching `combine.load_target_stack`.

WHY THIS IS NOT makani's `MetricsHandler`, which plan §4.2 proposed.  Two reasons,
both found by reading the code rather than the docs:

1. **makani's default CRPS kernel ignores ensemble weights.**  `_crps_skillspread_kernel`
   -- the `skillspread` type the CRPS arm's config selects, and upstream's choice for
   FCN3 -- reduces over the ensemble with a plain `torch.mean`; `weights` enters only
   through the NaN mask.  Only the `cdf` and `probability weighted moment` kernels
   consume them.  So for the weighted ensemble plan §4.4 asks for, there is nothing to
   reuse, and passing `ensemble_weights` to the obvious constructor would have
   silently produced an unweighted number.
2. `MetricsHandler` and the `Geometric*` wrappers call `comm.get_size("ensemble")`,
   so they need makani's process groups initialised.  That is a lot of training-loop
   machinery for an offline pass over NetCDFs.

What is preserved is the **definition**: `test_crps_matches_makani_unweighted` pins
the uniform-weight case of `fair_crps` against makani's own kernel, so our number and
a training-side number mean the same thing.  Where they disagree, that test fails.
"""
from __future__ import annotations

import numpy as np


def equiangular_weights(nlat: int) -> np.ndarray:
    """Cell areas for a cell-centred equiangular grid, normalised to sum to 1.

    Cell `i` is centred at `90 - (i+0.5)*180/nlat` and spans `Δ = 180/nlat`, so its
    area is `sin(lat+Δ/2) - sin(lat-Δ/2) = 2 sin(Δ/2) cos(lat)`; the constant drops out
    under normalisation, leaving exactly `cos(lat)`.  Identical to
    `sfno_eval.metrics.equiangular_lat_weights`, in numpy so this module stays free of
    the torch import (change G -- Gauss-Legendre over-weights the polar row by 1.50x
    here, and the only guard downstream is a shape check that 180-vs-180 passes).
    """
    lat = np.deg2rad(90.0 - (np.arange(nlat) + 0.5) * (180.0 / nlat))
    w = np.cos(lat)
    return w / w.sum()


def area_mean(field: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Latitude-weighted area mean over the trailing `(lat, lon)` axes."""
    return np.einsum("h,...h->...", w, field.mean(axis=-1))


def rmse(pred: np.ndarray, truth: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Latitude-weighted RMSE over `(..., lat, lon)`, reducing to the leading dims."""
    return np.sqrt(area_mean((pred - truth) ** 2, w))


def weighted_crps(stack: np.ndarray, truth: np.ndarray, w_ens: np.ndarray,
                  lat_w: np.ndarray, *, fair: bool = True) -> np.ndarray:
    """Weighted CRPS per channel, in the energy (NRG) form.

        CRPS = Σᵢ wᵢ|xᵢ − y| − ½ ΣᵢΣⱼ wᵢwⱼ|xᵢ − xⱼ|

    which is the standard ensemble estimator generalised to unequal member weights and
    reduces to it exactly at `wᵢ = 1/E`.  This form -- rather than makani's
    `skillspread` -- because that kernel does not read its weights at all (module
    docstring), and our members are deliberately not exchangeable.

    `fair=True` divides the spread term by `1 - Σwᵢ²`, the reliability-weight
    generalisation of the `E/(E-1)` fair correction.  The plain estimator is biased
    **low** at small ensembles, i.e. it flatters a 14-member ensemble, so the fair
    variant is the default and the honest one to report.

    Shapes: `stack` `(E, C, H, W)`, `truth` `(C, H, W)`, `w` `(E, C)`; returns `(C,)`.

    The pairwise term is accumulated in an `E`-long loop rather than materialised: the
    full `(E, E, C, H, W)` difference is 1.6 GB at E=14 and 16 channels, and 10 GB at
    101.
    """
    E = stack.shape[0]
    skill = np.einsum("ec,echw->chw", w_ens, np.abs(stack - truth))

    spread = np.zeros_like(skill)
    for i in range(E):
        # |x_i - X| against the whole stack at once: (E, C, H, W) per iteration.
        d = np.abs(stack[i] - stack)                       # (E, C, H, W)
        spread += w_ens[i][:, None, None] * np.einsum("ec,echw->chw", w_ens, d)

    if fair:
        spread = spread / (1.0 - (w_ens ** 2).sum(axis=0))[:, None, None]

    return area_mean(skill - 0.5 * spread, lat_w)


def spread_skill(stack: np.ndarray, truth: np.ndarray, mean: np.ndarray,
                 w_ens: np.ndarray, lat_w: np.ndarray) -> dict:
    """Ensemble spread, skill and their ratio, per channel -- makani's SSR convention.

    `spread` and `skill` are each the **square root of the area-integrated variance**,
    i.e. quadrature first, root second, which is what `GeometricSpread` and
    `GeometricSSR` do; rooting per cell first would give a different (smaller) number
    and would not be comparable with a training-side value.

    SSR carries makani's fair correction, `spread / sqrt(skill² − spread²/E_eff)`,
    where `E_eff = 1/Σw²` is the weights' effective sample size (`= E` when uniform).
    Without it a finite ensemble reads as under-dispersed purely by construction.

    **SSR ≈ 1 is calibrated; < 1 is under-dispersed, i.e. over-confident.**  A lagged
    ensemble shares one set of weights across all members, so it samples initial-
    condition and accumulated-rollout uncertainty and **not** model uncertainty --
    expect < 1, and read the number as how badly rather than whether (plan §4.5).
    """
    var = np.einsum("ec,echw->chw", w_ens, (stack - mean) ** 2)
    denom = 1.0 - (w_ens ** 2).sum(axis=0)
    spread_q = area_mean(var / denom[:, None, None], lat_w)
    skill_q = area_mean((mean - truth) ** 2, lat_w)
    e_eff = 1.0 / (w_ens ** 2).sum(axis=0)

    # NaN, not a clamp, where the fair correction turns the denominator non-positive.
    # makani clamps to `eps` because it needs a differentiable training-time number;
    # offline that turns an undefined ratio into a huge finite one -- a smoke reported
    # a median SSR of 599581 that read as a value rather than as "this is not defined
    # for this ensemble". The corrected term goes negative when the spread exceeds what
    # the ensemble-mean error can account for, which is itself the finding.
    fair_denom = skill_q - spread_q / e_eff
    ssr = np.where(fair_denom > 0, np.sqrt(spread_q / np.where(fair_denom > 0, fair_denom, 1.0)),
                   np.nan)
    return {"spread": np.sqrt(spread_q), "skill": np.sqrt(skill_q), "ssr": ssr,
            "e_eff": e_eff, "ssr_raw": np.sqrt(spread_q / np.clip(skill_q, 1e-30, None)),
            "n_undefined": int((fair_denom <= 0).sum())}


def rank_histogram(stack: np.ndarray, truth: np.ndarray, lat_w: np.ndarray) -> np.ndarray:
    """Talagrand rank histogram, `(C, E+1)`, area-weighted.

    The rank is how many members fall below the observation, so a calibrated ensemble
    is flat, U-shaped is under-dispersed and a slope is bias.

    ⚠ **Diagnostic only here, and the plan says so (§4.4).**  The statistic assumes
    exchangeable members; ours are ordered by construction -- depth 4 is systematically
    better than depth 56 -- so a lagged ensemble's histogram is U-shaped or sloped
    *by construction* and reading it as "under-dispersed" would be an error.  Restrict
    it to a fixed-depth sub-ensemble before drawing a calibration conclusion.
    """
    E, C = stack.shape[0], stack.shape[1]
    rank = (stack < truth[None]).sum(axis=0)                   # (C, H, W) in 0..E
    cell_w = np.broadcast_to(lat_w[:, None] / stack.shape[-1], rank.shape[-2:])
    out = np.zeros((C, E + 1))
    for r in range(E + 1):
        out[:, r] = ((rank == r) * cell_w).sum(axis=(-2, -1))
    return out


def member_rmse(stack: np.ndarray, truth: np.ndarray, lat_w: np.ndarray) -> np.ndarray:
    """RMSE of each member separately, `(E, C)` -- the ensemble's own baseline.

    Plan §4.3's first and bluntest check compares the combined field against the
    **shallowest** member, which is the best single forecast available for that target.
    An ensemble that does not beat it has bought nothing, and the weighting is wrong.
    """
    return rmse(stack, truth[None], lat_w)
