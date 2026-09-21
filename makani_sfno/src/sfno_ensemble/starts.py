"""Stagger-`d` start generation for the lagged ensemble (stage 1 / task 11).

The lagged ensemble's members are rollouts from **staggered start times** that all
span a common target, so the generator here is the exact opposite of
`sfno_inference.rollout_driver.nwp_ic_offsets`: that one spaces ICs
`(n_samples - K) // n_ic` apart -- roughly monthly -- precisely so the forecasts do
**not** overlap, which is right for a scorecard and useless here.  Nothing in this
module changes that path; it is additive, and the NWP sweep is untouched.
→ `docs/2026-09-10_lagged_ensemble_endtoend_plan.md` §3.1-§3.2.

⚠ Members from staggered starts are NOT the 243 on-disk checkpoint snapshots.
Those are a **checkpoint** ensemble -- different weights, same data, sampling model
uncertainty, which this construction does not.  The two are routinely blurred in
the surrounding docs; only the combination stage is shared.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LaggedSweepPlan:
    """What one stagger-`d` sweep costs and what it buys.

    `starts` are sample indices **within one holdout file** -- cross-file rollout is
    unsupported (`nwp_ic_offsets` raises on it) because the dataset's `_get_indices`
    would silently serve samples from the next file and break `time_plasim`
    provenance.

    `target_range` is the inclusive span of absolute sample indices that receive the
    **full** `members_per_target` complement; targets outside it are partially
    covered and are the reason a sweep is longer than the window it scores.

    ⚠ **The member COUNT is uniform across covered targets; the member DEPTHS are
    not.** A target's depths are `T - s` over the starts that span it, so they depend
    on `(T - first_start) mod stride`: there are exactly `stride` distinct depth
    patterns, of which only the on-lattice one is the plan's quoted `d, 2d, … K`.
    That is harmless for `w_k ∝ 1/σ(k)²`, which weights each member by its own depth,
    but it does mean two targets are not quite the same experiment -- so aggregating a
    spread or a CRPS across residue classes mixes constructions. `lattice_targets`
    is the subset that shares one depth pattern.
    """

    starts: list[int]
    K: int
    stride: int
    members_per_target: int
    target_range: tuple[int, int]
    first_start: int

    @property
    def n_rollouts(self) -> int:
        return len(self.starts)

    @property
    def targets(self) -> list[int]:
        """Every fully-covered target, in order."""
        lo, hi = self.target_range
        return list(range(lo, hi + 1))

    @property
    def n_covered_targets(self) -> int:
        lo, hi = self.target_range
        return max(0, hi - lo + 1)

    @property
    def lattice_depths(self) -> tuple[int, ...]:
        """`(d, 2d, … K)` -- the depths of a target sitting on the start lattice."""
        return tuple(self.stride * (i + 1) for i in range(self.members_per_target))

    @property
    def lattice_targets(self) -> list[int]:
        """Covered targets congruent to `first_start` mod `stride`.

        These all share `lattice_depths`, so metrics aggregate across them without
        mixing depth compositions. There are `1/stride` as many as `targets`, and the
        rollouts cost the same either way -- the choice is about comparability, not
        compute.
        """
        return [t for t in self.targets if (t - self.first_start) % self.stride == 0]

    def depths_for_target(self, target: int) -> tuple[int, ...]:
        """Depths at which `target` receives members, shallowest first."""
        return tuple(sorted(target - s for s in self.starts if s < target <= s + self.K))

    def summary(self) -> str:
        """One greppable line for a launcher log."""
        lo, hi = self.target_range
        return (f"lagged sweep: {self.n_rollouts} rollouts, stride {self.stride}, K={self.K} "
                f"-> {self.members_per_target} members/target over targets {lo}..{hi} "
                f"({self.n_covered_targets} fully covered, {len(self.lattice_targets)} "
                f"on-lattice at depths {self.lattice_depths[0]}..{self.lattice_depths[-1]})")


def plan_lagged_sweep(
    n_samples: int,
    *,
    K: int,
    stride: int,
    n_targets: int,
    first_start: int = 0,
) -> LaggedSweepPlan:
    """Plan a stagger-`stride` sweep covering `n_targets` consecutive targets.

    A rollout from start `s` of length `K` contributes to target `T` whenever
    `s < T <= s + K`, at depth `T - s`.  Equivalently `s` ranges over the `K` integers
    `[T-K, T-1]`, of which exactly `K // stride` lie on the start lattice when `stride`
    divides `K` -- so every target whose window is fully inside the start range gets
    the same member COUNT (see the class docstring on why the depths still vary).

    Count, derived from the two clipping conditions rather than assumed.  The lowest
    full target needs the first `M = K/stride` starts at or below `T-1`, giving
    `T >= first_start + K - stride + 1`; the highest needs the last `M` starts at or
    above `T-K`, giving `T <= last_start + stride`.  So a sweep of `n` rollouts covers
    `(n+1)*stride - K` targets, and covering `n_targets` needs
    `ceil((n_targets + K) / stride) - 1`.  At the plan's `K=56, stride=4, n_targets=32`
    that is **21** rollouts and 14 members per target.

    ⚠ COST.  Each rollout is a full K-step forecast: ~1.95 GB of NetCDF and ~77 s of
    A100 at the production 101-channel shape (measured, job 7633207).  32 targets at
    stride 4 is therefore 21 rollouts ≈ 41 GB and ~27 min.  A whole test year would be
    ~350 rollouts ≈ 685 GB -- check disk before widening the window; the 243 snapshots
    already hold 403 GiB and `max_checkpoints_to_keep` does not prune.

    Raises
    ------
    ValueError
        If the parameters cannot produce an overlapping sweep, or if the requested
        window does not fit inside the file.  Refusing is deliberate: a silently
        truncated sweep yields targets with unequal member counts, and a weighted
        mean over unequal ensembles is not comparable across targets.
    """
    if K < 1:
        raise ValueError(f"K={K} must be >= 1")
    if stride < 1:
        raise ValueError(f"stride={stride} must be >= 1")
    if n_targets < 1:
        raise ValueError(f"n_targets={n_targets} must be >= 1")
    if first_start < 0:
        raise ValueError(f"first_start={first_start} must be >= 0")
    if stride > K:
        raise ValueError(
            f"stride={stride} > K={K}: consecutive rollouts would not overlap, so no "
            "target receives more than one member and there is no ensemble to combine"
        )
    if K % stride:
        # Unequal member counts across targets -- see the docstring. Cheap to avoid
        # by choosing a divisor, expensive to discover in stage 4.
        raise ValueError(
            f"stride={stride} does not divide K={K}: targets would receive "
            f"{K // stride} or {K // stride + 1} members depending on their residue, "
            "and the per-depth weights would not line up across targets"
        )

    # ceil((n_targets + K) / stride) - 1, floored at the K/stride rollouts it takes
    # to give any target a full complement at all.
    n_rollouts = max(K // stride, -(-(n_targets + K) // stride) - 1)
    starts = [first_start + i * stride for i in range(n_rollouts)]

    # A rollout needs samples s .. s+K, so the last one must satisfy s + K < n_samples
    # -- the same bound nwp_ic_offsets enforces, for the same reason.
    last = starts[-1]
    if last + K >= n_samples:
        raise ValueError(
            f"sweep does not fit: {n_rollouts} rollouts from {first_start} at stride "
            f"{stride} end at start {last}, and {last} + K={K} >= n_samples={n_samples}. "
            f"Cross-file rollout is unsupported. Reduce n_targets (max here is "
            f"{max(0, (n_samples - 1 - K - first_start) // stride * stride - K + 2)}), "
            "lower first_start, or shorten K."
        )

    return LaggedSweepPlan(
        starts=starts,
        K=K,
        stride=stride,
        members_per_target=K // stride,
        target_range=(first_start + K - stride + 1, last + stride),
        first_start=first_start,
    )


def lagged_ic_offsets(
    n_samples: int,
    *,
    K: int = 56,
    stride: int = 4,
    n_targets: int = 32,
    first_start: int = 0,
) -> list[int]:
    """Start indices for a stagger-`stride` sweep -- the thin counterpart of `nwp_ic_offsets`.

    Defaults are the plan's worked example (§3.1): `K=56`, `stride=4` ⇒ 14 members per
    target at depths 4, 8, … 56.  See `plan_lagged_sweep` for the coverage arithmetic
    and the cost warning.
    """
    return plan_lagged_sweep(
        n_samples, K=K, stride=stride, n_targets=n_targets, first_start=first_start
    ).starts
