"""Regroup rollout outputs by absolute target index (stage 3 / task 12).

A rollout from start `s` of length `K` predicts absolute sample indices `s+1 … s+K`
at depths `1 … K`, so **one rollout contributes a member to many targets** and the
sweep is regrouped rather than re-run (plan §3.1).  This module builds the
`(target, depth) -> prediction` table that stage 4 then combines with
`w_k ∝ 1/σ(k)²`.

Cheap on purpose: `read_rollout_index` opens each NetCDF for its attributes and
lead coordinate only and never touches `prediction`/`truth`, so indexing a
45 GB sweep costs milliseconds.  Stage 4 reads the fields, for one target at a
time, using the `(path, lead_index)` pairs this produces.

Labelling is **step-index**, not calendar (change E): the E3SM pack is on a noleap
365-day calendar with a split-cumulative day count, so a proleptic-Gregorian anchor
drifts a day per leap year crossed.  Absolute sample index is exact and is the
natural key for a lagged ensemble anyway.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class RolloutIndex:
    """The provenance of one rollout NetCDF -- everything but the fields.

    `start_global_idx` is the IC's index into the **concatenated** dataset
    (`ic_global_idx`), which is unique across holdout files; `start_sample_idx` is
    its index within `ic_file`.  Alignment keys on the global one: two files both
    have a sample 100, and they are different times.
    """

    path: Path
    start_global_idx: int
    start_sample_idx: int
    ic_file: str
    lead_hours: tuple[int, ...]
    dt_hours: int


@dataclass(frozen=True)
class Member:
    """One ensemble member for one target: where to read it, and how old it is."""

    path: Path
    lead_index: int
    depth_steps: int
    lead_hours: int
    start_global_idx: int
    ic_file: str


def read_rollout_index(path) -> RolloutIndex:
    """Read one rollout NetCDF's provenance without loading any field data."""
    import xarray as xr

    path = Path(path)
    # decode_timedelta=False keeps the lead coord as the integers on disk; current
    # xarray turns a `units = "hours"` attribute into timedelta64 and warns that the
    # default is about to move again.
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        lead = np.asarray(ds["lead_time"].values)
        if np.issubdtype(lead.dtype, np.timedelta64):
            lead = lead / np.timedelta64(1, "h")
        lead = tuple(int(x) for x in lead)
        attrs = ds.attrs
        missing = [k for k in ("ic_global_idx", "ic_sample_idx", "ic_file")
                   if k not in attrs]
        if missing:
            raise ValueError(
                f"{path.name} lacks provenance attrs {missing}; it was not written by "
                "nc_writer.write_rollout_nc and cannot be aligned by target index"
            )
        return RolloutIndex(
            path=path,
            start_global_idx=int(attrs["ic_global_idx"]),
            start_sample_idx=int(attrs["ic_sample_idx"]),
            ic_file=str(attrs["ic_file"]),
            lead_hours=lead,
            dt_hours=int(attrs.get("dt_hours", 6)),
        )


def index_rollout_dir(nc_dir, pattern: str = "*.nc") -> list[RolloutIndex]:
    """Index every rollout NetCDF in a directory, sorted by start index."""
    paths = sorted(Path(nc_dir).glob(pattern))
    if not paths:
        raise ValueError(f"no files matching {pattern!r} under {nc_dir}")
    return sorted((read_rollout_index(p) for p in paths),
                  key=lambda r: (r.ic_file, r.start_global_idx))


def align_members_by_target(
    indexes: Iterable[RolloutIndex],
    *,
    min_members: int = 2,
    max_depth_steps: int | None = None,
) -> dict[int, list[Member]]:
    """Group rollout outputs into `{absolute target index: [Member, …]}`.

    Target `T = start + k` for `k = 1 … K`, so a member's `depth_steps` is `k` and
    its `lead_index` is `k - 1`.  Members are returned **shallowest first**, which is
    the order stage 4's acceptance check needs: the ensemble must beat its own
    shallowest member, and if it does not the weighting is wrong (plan §4.3).

    Targets with fewer than `min_members` members are dropped -- those are the
    partially-covered edges of the sweep, and mixing a 3-member target with a
    14-member one makes a spread or CRPS incomparable across targets.  The default
    of 2 drops only targets that have no ensemble at all; pass
    `members_per_target` from the sweep plan to keep exclusively full ones.

    `max_depth_steps` truncates deep members -- useful once σ(k) says the oldest
    members carry almost no weight, since dropping them saves reads rather than
    accuracy.

    Raises
    ------
    ValueError
        On a target whose members disagree about `ic_file` (the starts crossed a
        file boundary, so the same index means two different times), or on a
        duplicated `(start, depth)` pair (the same rollout indexed twice, which
        would double-count a member and shrink the spread).
    """
    if min_members < 1:
        raise ValueError(f"min_members={min_members} must be >= 1")

    by_target: dict[int, list[Member]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()

    for idx in indexes:
        for lead_index, lead_h in enumerate(idx.lead_hours):
            depth = lead_index + 1
            if max_depth_steps is not None and depth > max_depth_steps:
                break
            key = (idx.start_global_idx, depth)
            if key in seen:
                raise ValueError(
                    f"duplicate member: start {idx.start_global_idx} depth {depth} "
                    f"appears twice ({idx.path.name}). The same rollout is indexed more "
                    "than once, which would double-count it and understate the spread."
                )
            seen.add(key)
            by_target[idx.start_global_idx + depth].append(
                Member(path=idx.path, lead_index=lead_index, depth_steps=depth,
                       lead_hours=int(lead_h), start_global_idx=idx.start_global_idx,
                       ic_file=idx.ic_file)
            )

    out: dict[int, list[Member]] = {}
    for target in sorted(by_target):
        members = by_target[target]
        files = {m.ic_file for m in members}
        if len(files) > 1:
            raise ValueError(
                f"target {target} draws members from {sorted(files)}: an absolute sample "
                "index means different times in different holdout files, so these are "
                "not the same target. Cross-file rollout is unsupported."
            )
        if len(members) >= min_members:
            out[target] = sorted(members, key=lambda m: m.depth_steps)
    return out


def coverage_report(table: dict[int, list[Member]]) -> dict:
    """Summarise an alignment table: how many targets, how deep, how uniform.

    Written for a launcher to print and a human to sanity-check before stage 4 reads
    45 GB.  `uniform` false means member counts differ across targets, which is legal
    but makes cross-target spread comparisons unsound (see `align_members_by_target`).
    """
    if not table:
        return {"n_targets": 0, "uniform": True, "member_counts": {}}
    counts = sorted({len(v) for v in table.values()})
    depths = sorted({m.depth_steps for v in table.values() for m in v})
    targets = sorted(table)
    return {
        "n_targets": len(table),
        "target_range": (targets[0], targets[-1]),
        "member_counts": {c: sum(1 for v in table.values() if len(v) == c) for c in counts},
        "uniform": len(counts) == 1,
        "depth_range": (depths[0], depths[-1]),
        "n_distinct_depths": len(depths),
    }


def sigma_weights(sigma: Sequence[float], depths: Sequence[int]) -> np.ndarray:
    """Return `w_k ∝ 1/σ(k)²` for the given depths, normalised to sum to 1 (plan §3.3).

    `sigma` is indexed by depth-1, i.e. `sigma[k-1]` is the RMSE at depth `k` -- the
    layout `polaris/score_rollout_nc.py` writes to `k56_metrics.h5` as `rmse_mean`,
    which is where the 56-depth curve comes from.  Per channel; a caller scoring 101
    channels calls this 101 times or vectorises over the channel axis itself.

    Uniform weighting is measurably wrong here: the median channel's RMSE at depth 21
    is 4.65x its depth-1 value, so a plain mean gives a two-week-old member the same
    say as a six-hour-old one.

    Raises on a non-finite or non-positive σ rather than emitting an infinite weight
    -- that would hand one member the entire ensemble.
    """
    sig = np.asarray([sigma[d - 1] for d in depths], dtype=np.float64)
    if not np.all(np.isfinite(sig)) or np.any(sig <= 0):
        bad = [int(d) for d, s in zip(depths, sig) if not np.isfinite(s) or s <= 0]
        raise ValueError(
            f"sigma must be finite and positive; depths {bad} are not. A zero or NaN "
            "sigma gives one member infinite weight, which is not an ensemble."
        )
    w = 1.0 / (sig ** 2)
    return w / w.sum()
