"""Tests for the lagged-ensemble staggered starts and member alignment (tasks 11-12).

    pytest -q makani_sfno/tests/sfno_ensemble/test_lagged_ensemble.py
    python  makani_sfno/tests/sfno_ensemble/test_lagged_ensemble.py   # LAGGED_ENSEMBLE_OK

Torch-free by construction (the package manipulates indices and NetCDF metadata,
never model state), so these run anywhere -- unlike the rest of the fork's suites,
which CLAUDE.md #3 keeps off the login node.  They ride along in
`polaris/polaris_e3sm_port_test.pbs` so one job covers every CPU correctness test.

What is actually being guarded, beyond the arithmetic:

* **Overlap.** `nwp_ic_offsets` spaces starts ~117 samples apart at K=56, so the
  existing 24-IC sweep contributes exactly ZERO lagged members. A generator that
  silently produced non-overlapping starts would yield one member per target and an
  "ensemble" whose spread is identically zero.
* **File boundaries.** An absolute sample index means different times in different
  holdout files. Aligning across them would average 2048 against 2049 under one
  target id, and nothing downstream would notice.
* **Duplicate members.** The same rollout counted twice shrinks the spread and
  inflates SSR -- a wrong answer to the calibration question the ensemble exists
  to ask.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sfno_ensemble import (  # noqa: E402
    Member,
    RolloutIndex,
    align_members_by_target,
    coverage_report,
    index_rollout_dir,
    lagged_ic_offsets,
    plan_lagged_sweep,
    read_rollout_index,
    sigma_weights,
)

N_SAMPLES = 1460  # one E3SM test year, 6-hourly noleap


# ---------------------------------------------------------------------------
# Task 11 -- staggered starts
# ---------------------------------------------------------------------------

def test_plan_matches_the_worked_example():
    """Plan §3.1: K=56, d=4 => 14 members per target, on-lattice depths 4, 8, … 56."""
    plan = plan_lagged_sweep(N_SAMPLES, K=56, stride=4, n_targets=32)
    assert plan.members_per_target == 14
    assert plan.lattice_depths == tuple(range(4, 57, 4))
    assert plan.starts == [4 * i for i in range(plan.n_rollouts)]
    assert plan.n_covered_targets >= 32


def test_rollout_count_is_not_over_provisioned():
    """Each surplus rollout is ~1.95 GB and ~77 s, so the count must be tight.

    Exactly `(n+1)*stride - K` targets are covered, so asking for one fewer rollout
    must drop below the request. The first draft of this arithmetic over-provisioned
    by 2 rollouts at these parameters.
    """
    plan = plan_lagged_sweep(N_SAMPLES, K=56, stride=4, n_targets=32)
    assert plan.n_rollouts == 21
    assert plan.n_covered_targets == 32
    shorter = plan.starts[:-1]
    covered = sum(1 for t in plan.targets
                  if sum(1 for s in shorter if s < t <= s + 56) == plan.members_per_target)
    assert covered < 32


def test_every_covered_target_gets_the_full_complement():
    """The coverage arithmetic must be exactly right, so verify it by brute force.

    A rollout from `s` covers target `T` iff `s < T <= s + K`. Counting that directly
    for every claimed target is the check the closed-form count could get wrong.
    """
    plan = plan_lagged_sweep(N_SAMPLES, K=56, stride=4, n_targets=32)
    lo, hi = plan.target_range
    for target in range(lo, hi + 1):
        n = sum(1 for s in plan.starts if s < target <= s + plan.K)
        assert n == plan.members_per_target, (target, n, plan.members_per_target)


def test_depths_vary_by_residue_class_but_only_in_stride_many_ways():
    """Member COUNT is uniform; the depth SET is not -- and the difference matters.

    A target's depths are `T - s`, so they depend on `(T - first_start) mod stride`.
    There are exactly `stride` patterns, and only the on-lattice one is the plan's
    quoted `d, 2d, … K`. Asserting uniformity here (the first draft did) would have
    been asserting something false.
    """
    plan = plan_lagged_sweep(N_SAMPLES, K=24, stride=6, n_targets=10)
    patterns = {plan.depths_for_target(t) for t in plan.targets}
    assert len(patterns) == plan.stride, patterns
    assert all(len(p) == plan.members_per_target for p in patterns)

    on_lattice = {plan.depths_for_target(t) for t in plan.lattice_targets}
    assert on_lattice == {plan.lattice_depths}
    assert len(plan.lattice_targets) == plan.n_covered_targets // plan.stride


def test_starts_overlap_unlike_the_monthly_generator():
    """The whole point of task 11: consecutive rollouts must share targets."""
    sys.path.insert(0, str(_SRC / "sfno_inference"))
    plan = plan_lagged_sweep(N_SAMPLES, K=56, stride=4, n_targets=32)
    a, b = plan.starts[0], plan.starts[1]
    assert b - a < plan.K, "consecutive lagged starts must be closer than K"
    # The monthly spacing at these parameters is (1460 - 56) // 12 = 117 > 56, i.e.
    # no overlap at all -- which is why a new generator exists.
    assert (N_SAMPLES - 56) // 12 > 56


def test_stride_must_divide_K():
    with pytest.raises(ValueError, match="does not divide"):
        plan_lagged_sweep(N_SAMPLES, K=56, stride=5, n_targets=8)


def test_stride_wider_than_K_is_refused():
    """d > K gives every target at most one member -- not an ensemble."""
    with pytest.raises(ValueError, match="would not overlap"):
        plan_lagged_sweep(N_SAMPLES, K=8, stride=16, n_targets=4)


def test_sweep_that_would_cross_a_file_boundary_is_refused():
    """Cross-file rollout is unsupported; a truncated sweep would be silent otherwise."""
    with pytest.raises(ValueError, match="does not fit"):
        plan_lagged_sweep(200, K=56, stride=4, n_targets=500)


def test_last_start_leaves_room_for_the_full_rollout():
    plan = plan_lagged_sweep(N_SAMPLES, K=56, stride=4, n_targets=200)
    assert plan.starts[-1] + plan.K < N_SAMPLES


def test_first_start_shifts_the_window():
    plan = plan_lagged_sweep(N_SAMPLES, K=56, stride=4, n_targets=16, first_start=400)
    assert plan.starts[0] == 400
    assert plan.target_range[0] == 400 + 56 - 4 + 1
    assert all((t - 400) % 4 == 0 for t in plan.lattice_targets)


def test_lagged_ic_offsets_is_the_plan_starts():
    assert lagged_ic_offsets(N_SAMPLES, K=56, stride=4, n_targets=32) == \
        plan_lagged_sweep(N_SAMPLES, K=56, stride=4, n_targets=32).starts


# ---------------------------------------------------------------------------
# Task 12 -- alignment by absolute target index
# ---------------------------------------------------------------------------

def _idx(start: int, K: int = 8, ic_file: str = "2048.h5", path: str | None = None):
    """A RolloutIndex as `read_rollout_index` would return it, without touching disk."""
    return RolloutIndex(
        path=Path(path or f"{ic_file[:4]}_s{start:05d}.nc"),
        start_global_idx=start,
        start_sample_idx=start,
        ic_file=ic_file,
        lead_hours=tuple(6 * (i + 1) for i in range(K)),
        dt_hours=6,
    )


def test_alignment_regroups_by_absolute_target():
    """One rollout feeds many targets; target T gets depth T - s from each."""
    table = align_members_by_target([_idx(0), _idx(2), _idx(4)], min_members=1)
    assert table[5] == sorted(table[5], key=lambda m: m.depth_steps)
    assert [(m.start_global_idx, m.depth_steps) for m in table[5]] == [(4, 1), (2, 3), (0, 5)]
    assert all(m.lead_hours == 6 * m.depth_steps for v in table.values() for m in v)


def test_alignment_matches_the_plan_arithmetic():
    """Members per fully-covered target must equal K // stride, end to end."""
    plan = plan_lagged_sweep(N_SAMPLES, K=24, stride=4, n_targets=12)
    table = align_members_by_target([_idx(s, K=24) for s in plan.starts], min_members=1)
    lo, hi = plan.target_range
    for target in range(lo, hi + 1):
        assert len(table[target]) == plan.members_per_target, target


def test_partially_covered_edges_are_dropped():
    """A 3-member target and a 14-member one are not comparable -- drop the edges."""
    plan = plan_lagged_sweep(N_SAMPLES, K=24, stride=4, n_targets=12)
    idxs = [_idx(s, K=24) for s in plan.starts]
    full = align_members_by_target(idxs, min_members=plan.members_per_target)
    assert sorted(full) == plan.targets
    assert coverage_report(full)["uniform"] is True
    # With min_members=1 the ragged edges reappear, so the filter is what removed them.
    assert len(align_members_by_target(idxs, min_members=1)) > len(full)


def test_cross_file_targets_are_refused():
    """Sample index 5 means different times in 2048.h5 and 2049.h5."""
    with pytest.raises(ValueError, match="different times"):
        align_members_by_target([_idx(0, ic_file="2048.h5"), _idx(2, ic_file="2049.h5")],
                                min_members=1)


def test_duplicate_member_is_refused():
    """The same rollout twice would double-count it and understate the spread."""
    with pytest.raises(ValueError, match="duplicate member"):
        align_members_by_target([_idx(0), _idx(0, path="a_copy.nc")], min_members=1)


def test_max_depth_truncates_without_shifting_shallow_members():
    table = align_members_by_target([_idx(0), _idx(2)], min_members=1, max_depth_steps=3)
    assert max(m.depth_steps for v in table.values() for m in v) == 3
    assert [(m.start_global_idx, m.depth_steps) for m in table[3]] == [(2, 1), (0, 3)]


def test_coverage_report_flags_non_uniform_tables():
    rep = coverage_report(align_members_by_target([_idx(0), _idx(2)], min_members=1))
    assert rep["uniform"] is False
    assert rep["n_targets"] == 10
    assert rep["depth_range"] == (1, 8)


# ---------------------------------------------------------------------------
# Depth weights (the formula stage 4 consumes)
# ---------------------------------------------------------------------------

def test_sigma_weights_favour_the_shallow_member():
    """w_k ~ 1/sigma^2: a member with twice the error gets a quarter the weight."""
    sigma = [1.0, 2.0, 3.0, 4.0]
    w = sigma_weights(sigma, depths=[1, 2])
    assert np.isclose(w.sum(), 1.0)
    assert np.isclose(w[0] / w[1], 4.0)


def test_sigma_weights_reject_a_degenerate_sigma():
    """A zero sigma would hand one member the entire ensemble."""
    with pytest.raises(ValueError, match="finite and positive"):
        sigma_weights([1.0, 0.0], depths=[1, 2])
    with pytest.raises(ValueError, match="finite and positive"):
        sigma_weights([1.0, float("nan")], depths=[1, 2])


def test_sigma_weights_index_by_depth_not_position():
    """sigma[k-1] is depth k -- an off-by-one here silently mis-weights every member."""
    sigma = [1.0, 10.0, 100.0]
    w = sigma_weights(sigma, depths=[1, 3])
    assert np.isclose(w[0] / w[1], 1e4)


# ---------------------------------------------------------------------------
# Round trip through a real NetCDF (nc_writer's schema)
# ---------------------------------------------------------------------------

def test_reads_provenance_from_a_real_netcdf(tmp_path):
    """read_rollout_index must agree with what nc_writer actually writes."""
    xr = pytest.importorskip("xarray")
    K = 4
    for start in (0, 2):
        ds = xr.Dataset(
            data_vars=dict(prediction=(("init_time", "lead_time", "channel", "lat", "lon"),
                                       np.zeros((1, K, 1, 2, 2), dtype=np.float32))),
            coords=dict(init_time=("init_time", np.array([np.int64(start)])),
                        lead_time=("lead_time", np.arange(1, K + 1, dtype=np.int64) * 6),
                        channel=("channel", ["PS"]),
                        lat=("lat", [45.0, -45.0]), lon=("lon", [0.0, 180.0])),
            attrs=dict(ic_file="2048.h5", ic_sample_idx=start, ic_global_idx=start,
                       rollout_mode="lagged", K=K, dt_hours=6),
        )
        ds["lead_time"].attrs["units"] = "hours"
        ds.to_netcdf(tmp_path / f"2048_s{start:05d}.nc", format="NETCDF4")

    idxs = index_rollout_dir(tmp_path)
    assert [i.start_global_idx for i in idxs] == [0, 2]
    assert idxs[0].lead_hours == (6, 12, 18, 24)
    table = align_members_by_target(idxs, min_members=2)
    assert [(m.start_global_idx, m.depth_steps) for m in table[4]] == [(2, 2), (0, 4)]


def test_netcdf_without_provenance_is_refused(tmp_path):
    xr = pytest.importorskip("xarray")
    ds = xr.Dataset(coords=dict(lead_time=("lead_time", np.array([6], dtype=np.int64))))
    ds.to_netcdf(tmp_path / "bare.nc", format="NETCDF4")
    with pytest.raises(ValueError, match="lacks provenance attrs"):
        read_rollout_index(tmp_path / "bare.nc")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
