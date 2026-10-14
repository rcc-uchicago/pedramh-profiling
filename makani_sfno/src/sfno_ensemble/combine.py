"""Weighted combination of lagged members (stage 4 / task 13).

Per target, per channel, per grid cell: a weighted mean over the members with
`w_k ∝ 1/σ(k)²`, plus the weighted spread.  σ(k) is the error-vs-depth curve from
`polaris/score_rollout_nc.py`'s `k56_metrics.h5`, and using it is not a refinement
but a correction: the median channel's RMSE at depth 21 is **4.65x** its depth-1
value, so a plain mean gives a fourteen-day-old member the same say as a six-hour-old
one (plan §3.3).

⚠ **σ(k) must come from a DIFFERENT sample than the ensemble being weighted.**  The
shipped curve does -- it is the monthly K=56 sweep (24 ICs, non-overlapping) -- so the
weights are independent of the lagged members they combine.  Weighting an ensemble by
its own errors would fit the noise and flatter the result.

Reads are chunked over channels because the natural unit here is large: one target's
stack is `(14, 101, 180, 360)` float64 = 734 MB before any temporary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class TargetStack:
    """One target's members, their truth, and the weights that combine them.

    `members` is `(E, C, H, W)` in physical units, ordered shallowest-first to match
    `align_members_by_target`.  `truth` is `(C, H, W)` -- a single field, because every
    member of a target predicts the **same valid time**.  That redundancy is checked
    rather than assumed (`truth_max_disagreement`): all E copies come from the same h5
    samples through the same de-normalisation, so a non-zero value means the alignment
    put unrelated times in one bucket, and every metric below it would be meaningless.
    """

    target: int
    depths: tuple[int, ...]
    members: np.ndarray
    truth: np.ndarray
    truth_max_disagreement: float


def load_target_stack(members, *, channels: slice | None = None,
                      check_truth: bool = True) -> TargetStack:
    """Read one target's member fields out of the rollout NetCDFs.

    `members` is the shallowest-first list `align_members_by_target` produces.  Only
    the `(lead_index)` slab of each file is read, so the cost is E slab reads rather
    than E whole files.

    `check_truth` re-reads each member's truth slab to verify they agree.  It doubles
    the reads, which is why a caller sweeping many targets checks a sample rather than
    all of them -- but it is the one test that proves the alignment arithmetic against
    real data, so do not skip it entirely.
    """
    import xarray as xr

    sel = channels if channels is not None else slice(None)
    fields, truths = [], []
    for m in members:
        with xr.open_dataset(m.path, decode_timedelta=False) as ds:
            fields.append(np.asarray(ds["prediction"][0, m.lead_index, sel].values,
                                     dtype=np.float64))
            if check_truth or not truths:
                truths.append(np.asarray(ds["truth"][0, m.lead_index, sel].values,
                                         dtype=np.float64))

    stack = np.stack(fields)
    disagree = 0.0
    if len(truths) > 1:
        ref = truths[0]
        disagree = float(max(np.abs(t - ref).max() for t in truths[1:]))
    return TargetStack(
        target=members[0].start_global_idx + members[0].depth_steps,
        depths=tuple(m.depth_steps for m in members),
        members=stack,
        truth=truths[0],
        truth_max_disagreement=disagree,
    )


def member_weights(sigma: np.ndarray, depths: Sequence[int]) -> np.ndarray:
    """Return `(E, C)` weights `w_k ∝ 1/σ(k, c)²`, normalised over members per channel.

    `sigma` is `(K, C)` indexed by depth-1 -- the layout `k56_metrics.h5` stores as
    `rmse_mean`.  Weights are **per channel**: a channel whose error grows slowly with
    depth should keep its old members, and one that degrades fast should not, and a
    single shared weight vector cannot express both.

    Raises on a non-finite or non-positive σ rather than emitting an infinite weight,
    which would hand one member the whole ensemble.
    """
    sig = np.asarray(sigma, dtype=np.float64)[[d - 1 for d in depths], :]
    if not np.all(np.isfinite(sig)) or np.any(sig <= 0):
        bad = sorted({int(depths[i]) for i, _ in zip(*np.where(~(np.isfinite(sig) & (sig > 0))))})
        raise ValueError(
            f"sigma must be finite and positive; depths {bad} are not. A zero or NaN "
            "sigma gives one member infinite weight, which is not an ensemble."
        )
    w = 1.0 / (sig ** 2)
    return w / w.sum(axis=0, keepdims=True)


def weighted_mean(stack: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted ensemble mean, `(E, C, H, W)` x `(E, C)` -> `(C, H, W)`."""
    return np.einsum("ec,echw->chw", w, stack)


def weighted_spread(stack: np.ndarray, w: np.ndarray,
                    mean: np.ndarray | None = None) -> np.ndarray:
    """Per-cell weighted ensemble standard deviation, debiased for non-uniform weights.

    The bias correction is `1 / (1 - Σ w²)`, the reliability-weight generalisation of
    `E/(E-1)`: at uniform `w = 1/E` it reduces to exactly that, which is the factor
    makani's `GeometricSpread` applies (`sqrt(spread / (ens_size - 1))`).  Using the
    uncorrected form would understate the spread and so overstate confidence -- the
    wrong direction for a calibration diagnostic, where the model is already expected
    to be under-dispersed.
    """
    mu = weighted_mean(stack, w) if mean is None else mean
    var = np.einsum("ec,echw->chw", w, (stack - mu) ** 2)
    denom = 1.0 - (w ** 2).sum(axis=0)          # (C,)
    return np.sqrt(var / denom[:, None, None])


def combine_target(stack: TargetStack, sigma: np.ndarray) -> dict:
    """Combine one target: weighted mean, weighted spread, and the uniform comparison.

    The uniform-weight mean is carried alongside deliberately.  Stage 4's acceptance
    test is not "is the ensemble good" but "did the weighting earn its complexity" --
    if `w_k ∝ 1/σ(k)²` does not beat a plain mean, and the plain mean does not beat
    the shallowest single member, then the construction bought nothing (plan §4.3).
    """
    w = member_weights(sigma, stack.depths)
    mean_w = weighted_mean(stack.members, w)
    uniform = np.full_like(w, 1.0 / w.shape[0])
    return {
        "weights": w,
        "mean_weighted": mean_w,
        "mean_uniform": weighted_mean(stack.members, uniform),
        "spread_weighted": weighted_spread(stack.members, w, mean_w),
    }
