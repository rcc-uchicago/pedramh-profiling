"""Lagged-ensemble machinery: staggered starts, member alignment, depth weights.

Stages 1 and 3 of `docs/2026-09-10_lagged_ensemble_endtoend_plan.md` §3.2 (tasks 11
and 12), plus the weight formula stage 4 consumes.  A separate package rather than
additions to `sfno_inference/` because that tree is shared with the Stampede3
`eval-sfno-own` path and nothing here is wanted there (CLAUDE.md #7); the rollout
driver and the NWP scorecard are untouched.

Deliberately torch-free -- it manipulates indices and NetCDF metadata, never model
state -- so the tests run anywhere.

⚠ The members here come from **staggered start times**.  The 243 on-disk snapshots
(`ckpt_mp0_v0 … v242`) are a **checkpoint** ensemble, a different construction that
samples model uncertainty as this one does not.  Only the combination stage is
shared between them.
"""
from sfno_ensemble.alignment import (
    Member,
    RolloutIndex,
    align_members_by_target,
    coverage_report,
    index_rollout_dir,
    read_rollout_index,
    sigma_weights,
)
from sfno_ensemble.starts import LaggedSweepPlan, lagged_ic_offsets, plan_lagged_sweep

__all__ = [
    "LaggedSweepPlan",
    "Member",
    "RolloutIndex",
    "align_members_by_target",
    "coverage_report",
    "index_rollout_dir",
    "lagged_ic_offsets",
    "plan_lagged_sweep",
    "read_rollout_index",
    "sigma_weights",
]
