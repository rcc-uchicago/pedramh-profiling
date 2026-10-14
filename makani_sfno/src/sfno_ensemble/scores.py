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


# --------------------------------------------------------------------------------
# ACE2 eq 8 -- the framework this scorecard is supposed to follow.
#
# Watt-Meyer et al., "ACE2: Accurately learning subseasonal to decadal atmospheric
# variability and forced responses" (arXiv:2411.11268), §4.3 eq (8):
#
#     alpha = (1/C) sum_c sqrt( sum_{phi,lambda} w_{phi,lambda}
#                               ( MEAN_{t,ensemble}[ y_c - yhat_c ] )^2 )
#
# with eq (9) the single-variable form of the same quantity.  Three things about it
# are easy to read past, and all three change the answer:
#
# 1. **The overbar is INSIDE the square.**  The time- and ensemble-average is taken on
#    the SIGNED error field, and only then squared, area-weighted and rooted.  So this
#    is the RMS of the mean error -- a BIAS metric.  `rmse()` above squares per
#    snapshot first, which keeps the random component; the two differ by exactly the
#    variance of the error about its own time mean.  A construction that loses on
#    instantaneous RMSE can still win here, because averaging signed errors is what
#    cancels random error.  That is the whole reason the paper scores this way.
# 2. **y and yhat are NORMALIZED** (standard scaling, §4.3 "Data Normalization").  The
#    scaling mean cancels in the difference `y - yhat`, so only the per-channel sigma
#    is needed -- see `sigma_c` below.  Normalising is what makes step 3 legal.
# 3. **The channel reduction is the arithmetic mean `(1/C) sum_c`**, not a median.  A
#    median is what you are forced into when the channels are in disparate physical
#    units; once they are normalised the paper's plain mean is the right reduction,
#    and unlike a median it cannot hide a single blown-up channel.
#
# ACE2's own ensemble in this equation is a LAGGED one -- "an ensemble of eight 5-year
# long simulations, initialized at evenly spaced intervals", each contributing its
# prediction "for the corresponding time, from a simulation initialized at some
# previous time".  The paper combines them with a PLAIN, UNWEIGHTED average.  There is
# no inverse-variance weighting anywhere in the construction.
# --------------------------------------------------------------------------------


def time_mean_bias(pred_sum: np.ndarray, truth_sum: np.ndarray, n: int) -> np.ndarray:
    """The `(C, H, W)` time-and-ensemble-mean error field -- eq 8's overbar.

    Taken from running sums so a caller can stream targets without holding every
    field: `mean_t[yhat - y] = (sum_t yhat - sum_t y) / n`, and the ensemble average
    is already folded into `pred_sum` because truth is shared across a target's
    members (`combine.TargetStack`), so `mean_e[y - yhat] = y - mean_e[yhat]`.

    Sign convention follows the paper, `y - yhat` (truth minus prediction); it is
    squared immediately so the sign never reaches a reported number, but keeping it
    the paper's way round makes the per-channel bias maps readable as "model too low".
    """
    return (truth_sum - pred_sum) / float(n)


def bias_rms_per_channel(bias: np.ndarray, lat_w: np.ndarray,
                         sigma_c: np.ndarray | None = None) -> np.ndarray:
    """ACE2 eq **9**: global area-weighted RMS of a time-mean error field, `(C,)`.

    `bias` is `(C, H, W)` in physical units; `sigma_c` is the per-channel standard
    scaling that turns it into eq 8's normalised units.  Passing `sigma_c=None` leaves
    the result in physical units, which is eq 9 as written (it is applied per variable,
    so it never needs to be commensurable across channels) but is **not** summable into
    eq 8.
    """
    out = np.sqrt(area_mean(bias ** 2, lat_w))
    if sigma_c is None:
        return out
    sigma_c = np.asarray(sigma_c, dtype=np.float64)
    if np.any(~np.isfinite(sigma_c)) or np.any(sigma_c <= 0):
        raise ValueError(
            "sigma_c must be finite and positive; a zero or NaN normalisation makes "
            "one channel's contribution to alpha infinite and alpha meaningless."
        )
    return out / sigma_c


def ace2_alpha(bias: np.ndarray, lat_w: np.ndarray, sigma_c: np.ndarray,
               channel_weights: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """ACE2 eq **8**: `(alpha, per_channel)` from a time-mean error field.

    `channel_weights` is the paper's one documented deviation from a flat mean -- it
    "downweighted the contribution of q0 to the calculation of alpha by a factor of
    10, since our poor skill in predicting the time-mean of this variable otherwise
    dominated alpha".  Implemented as a weighted mean normalised by the weight sum, so
    the all-ones case is exactly `(1/C) sum_c` and a downweighted run stays on the same
    scale as an unweighted one.  Default is flat: a channel that dominates alpha is a
    finding, and it should have to be silenced on purpose.
    """
    per = bias_rms_per_channel(bias, lat_w, sigma_c)
    if channel_weights is None:
        return float(per.mean()), per
    cw = np.asarray(channel_weights, dtype=np.float64)
    if cw.shape != per.shape:
        raise ValueError(f"channel_weights {cw.shape} != n_channels {per.shape}")
    if np.any(cw < 0) or not cw.sum() > 0:
        raise ValueError("channel_weights must be non-negative with a positive sum")
    return float((per * cw).sum() / cw.sum()), per
