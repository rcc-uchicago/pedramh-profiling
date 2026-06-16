"""Preflight assertions for the explicit-K 5410 NWP eval pipeline.

Helpers shared by the orchestrator (`scripts/eval_inference_5410.py`),
the SLURMs (via small `python -c` invocations), and the test suite.

All helpers raise ``ValueError`` (not ``assert``) so they fail loud
under ``python -O`` as well as default mode.

Five preflight gates (per docs/2026-05-08_sfno_5410_explicit_K_horizon_plan.md):
  1. K is explicit (positive int, not bool).
  2. final_datetime == init_datetime + (K + 1) * 6h.
  3. yaml carries ensemble_inference_hours = (K+1)*6 AND
     prediction_duration_days = (K+1)*6/24, neither equal to year-long
     8760/8784h sentinel values.
  4. Output NetCDF time dim equals K + 1 (IC + K forecast leads).
  5. Upstream long_inference.py has been patched: exactly 4 allocator
     markers + 2 continuation markers (no partial / duplicate apply).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import cftime


# Sentinels that indicate a year-long override leaked through (the
# pre-fix values from `_ensemble_inference_hours_for_year`).
_YEAR_LONG_HOURS = (8760, 8784)

# Strict marker counts for the 6-hunk LP-003 partial-horizon patch.
_PATCH_ALLOCATOR_MARKER = "min(next_year_jan1, self.params.final_datetime)"
_PATCH_ALLOCATOR_COUNT = 4
_PATCH_CONTINUATION_MARKER = "current_datetime < self.params.final_datetime"
_PATCH_CONTINUATION_COUNT = 2

# LP-004 marker (single-hunk patch adding Stepper.reconfigure_for_ic).
_LP004_MARKER = "def reconfigure_for_ic"
_LP004_COUNT = 1
_BAD_NC_BC_OFFSET_MARKER = "params['nc_bc_offset'] = 18"

# Expected 8-variable schema in every output NetCDF (verified by smoke
# job 3097936). Drift here would indicate a partial / corrupt write.
_EXPECTED_OUTPUT_VARS = frozenset(
    {"pl", "tas", "pr_6h", "ta", "ua", "va", "hus", "zg"}
)


def assert_K_explicit(K: Any) -> None:
    """Raise ValueError unless K is a positive int (rejecting bool)."""
    if isinstance(K, bool) or not isinstance(K, int) or K < 1:
        raise ValueError(
            f"K must be a positive int (not bool), got {K!r} ({type(K).__name__})"
        )


def assert_final_datetime_matches(
    init_dt: cftime.datetime,
    final_dt: cftime.datetime,
    K: int,
    *,
    dt_hours: int = 6,
) -> None:
    """Raise ValueError unless final_dt == init_dt + (K + 1) * dt_hours."""
    assert_K_explicit(K)
    expected = init_dt + dt.timedelta(hours=(K + 1) * dt_hours)
    if final_dt != expected:
        raise ValueError(
            f"final_datetime mismatch: got {final_dt}, "
            f"expected init + (K+1)*{dt_hours}h = {expected} "
            f"(init={init_dt}, K={K})"
        )


def assert_yaml_horizon(yaml_path: Path, K: int, *, section: str = "SFNO") -> None:
    """Raise ValueError if the yaml's horizon keys don't match K.

    Asserts both:
      * ``<section>.ensemble_inference_hours == (K + 1) * 6``
      * ``<section>.prediction_duration_days == (K + 1) * 6 / 24``

    Also rejects ``ensemble_inference_hours in {8760, 8784}`` even if
    K=1459 / 1463 (the year-long sentinels would never be the true
    horizon for an eval-track partial-year run).
    """
    from ruamel.yaml import YAML

    assert_K_explicit(K)
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(yaml_path, "r") as f:
        doc = yaml.load(f)
    if section not in doc or not isinstance(doc[section], dict):
        raise ValueError(
            f"yaml {yaml_path} missing section {section!r}; "
            f"available top-level keys: {list(doc) if doc else []}"
        )
    sec = doc[section]
    expected_hours = (K + 1) * 6
    expected_days = (K + 1) * 6 / 24.0

    eih = sec.get("ensemble_inference_hours")
    if eih is None:
        raise ValueError(
            f"yaml {yaml_path}: {section}.ensemble_inference_hours is missing; "
            f"expected {expected_hours} for K={K}"
        )
    if eih in _YEAR_LONG_HOURS:
        raise ValueError(
            f"yaml {yaml_path}: {section}.ensemble_inference_hours == {eih} "
            f"(year-long sentinel); the year-long override leaked through. "
            f"Expected {expected_hours} for K={K}."
        )
    if int(eih) != expected_hours:
        raise ValueError(
            f"yaml {yaml_path}: {section}.ensemble_inference_hours == {eih}, "
            f"expected {expected_hours} for K={K}"
        )

    pdd = sec.get("prediction_duration_days")
    if pdd is None:
        raise ValueError(
            f"yaml {yaml_path}: {section}.prediction_duration_days is missing — "
            f"the BCS data loader (single_ic branch, "
            f"data_loader_multifiles.py:818-823) will collapse to "
            f"long_rollout_years=0 and the date range will be empty. "
            f"Expected {expected_days} for K={K}."
        )
    if abs(float(pdd) - expected_days) > 1e-9:
        raise ValueError(
            f"yaml {yaml_path}: {section}.prediction_duration_days == {pdd}, "
            f"expected {expected_days} for K={K}"
        )


def assert_upstream_patched(upstream_long_inference_path: Path) -> None:
    """Raise ValueError unless long_inference.py has exactly 4 allocator + 2 continuation markers.

    Strict counts guard against:
      * partially applied patch (e.g., only 1 of 4 allocator hunks landed)
      * accidental duplicate application after upstream resync
    """
    p = Path(upstream_long_inference_path)
    if not p.is_file():
        raise ValueError(f"upstream long_inference.py not found: {p}")
    text = p.read_text()
    alloc = text.count(_PATCH_ALLOCATOR_MARKER)
    cont = text.count(_PATCH_CONTINUATION_MARKER)
    if alloc != _PATCH_ALLOCATOR_COUNT or cont != _PATCH_CONTINUATION_COUNT:
        raise ValueError(
            f"upstream patch incomplete in {p}: "
            f"allocator markers={alloc} (expected {_PATCH_ALLOCATOR_COUNT}), "
            f"continuation markers={cont} (expected {_PATCH_CONTINUATION_COUNT}). "
            f"Reapply hunks per docs/2026-05-04_makani_local_patches.md."
        )


def assert_upstream_patched_lp004(upstream_long_inference_path: Path) -> None:
    """Raise ValueError unless long_inference.py has the LP-004 marker.

    LP-004 adds a single new method ``Stepper.reconfigure_for_ic``. Strict
    count: exactly 1 occurrence of ``def reconfigure_for_ic``. Any other
    count indicates partial application or accidental duplication.
    """
    p = Path(upstream_long_inference_path)
    if not p.is_file():
        raise ValueError(f"upstream long_inference.py not found: {p}")
    text = p.read_text()
    n = text.count(_LP004_MARKER)
    if n != _LP004_COUNT:
        raise ValueError(
            f"LP-004 patch marker count in {p}: got {n}, expected "
            f"{_LP004_COUNT}. Reapply the LP-004 hunk per "
            f"docs/2026-05-04_makani_local_patches.md (adds "
            f"Stepper.reconfigure_for_ic)."
        )


def assert_upstream_boundary_phase(upstream_long_inference_path: Path) -> None:
    """Raise ValueError if standalone long_inference still has the bad 18h BCS offset."""
    p = Path(upstream_long_inference_path)
    if not p.is_file():
        raise ValueError(f"upstream long_inference.py not found: {p}")
    text = p.read_text()
    if _BAD_NC_BC_OFFSET_MARKER in text:
        raise ValueError(
            f"upstream long_inference.py still contains {_BAD_NC_BC_OFFSET_MARKER!r}. "
            f"5410 NWP eval requires nc_bc_offset=0 so boundary forcing matches "
            f"the validation autoregression phase."
        )


def assert_yamls_share_static_arch(yaml_paths: list[Path]) -> None:
    """Cross-yaml invariant: every architecture/checkpoint/normalization/
    precision field matches across all per-Y yamls.

    Only val_year_start, val_year_end, leap_year, no_leap_year may differ.
    Delegates to ``upstream_hydration.assert_yamls_share_static_arch`` —
    this thin wrapper exists so the orchestrator and SLURMs can import a
    single ``preflight`` module for all gates.
    """
    from sfno_inference_5410.upstream_hydration import (
        assert_yamls_share_static_arch as _impl,
    )
    _impl(yaml_paths)


def assert_output_dir_empty(out_dir: Path) -> None:
    """Pre-launch: refuse if upstream_raw is non-empty.

    Mixed old + new NetCDFs in the same dir is the contamination state
    that bit us on job 3098028's pre-cancel state. Operator must
    explicitly choose: backup, delete, or fresh RUN_ROOT. The
    orchestrator will not silently overwrite or coexist with prior
    outputs.

    Note: gated on ``args.launch`` in the orchestrator — dry-run prints
    the plan without enforcing emptiness (Codex round-2 fix #3).
    """
    p = Path(out_dir)
    if not p.is_dir():
        # Doesn't exist yet — that's "empty" for our purposes.
        return
    nc_files = sorted(p.glob("Y*_member*_y*.nc"))
    if nc_files:
        sample = [f.name for f in nc_files[:5]]
        raise ValueError(
            f"output dir {p} is non-empty ({len(nc_files)} prior "
            f"NetCDFs). Backup, delete, or use a fresh RUN_ROOT before "
            f"submitting. First few: {sample}"
        )


def assert_output_dir_complete(
    out_dir: Path,
    plan: list,
    K: int,
    *,
    mode: str = "exact",
    expected_vars: frozenset = _EXPECTED_OUTPUT_VARS,
) -> None:
    """Post-flight: every (Y, s) in plan has a NetCDF with the right
    time dim AND the right variable set.

    Two modes (Codex round-6 fix #1 — for smoke against an existing
    populated production raw dir):
      * ``mode='exact'`` (default, backward-compat): expected_filenames
        == actual_filenames. Use for full production sweeps where
        exactly the planned set must be present, no more, no less.
      * ``mode='subset'``: expected_filenames ⊆ actual_filenames.
        Extras allowed (e.g., 95 prior production files coexisting
        with the 1 IC the smoke targets). Time-dim and variable-set
        checks still run on every expected file.

    Each plan entry must have ``Y`` and ``save_basename`` keys.
    """
    import xarray as xr

    if mode not in ("exact", "subset"):
        raise ValueError(
            f"mode must be 'exact' or 'subset', got {mode!r}"
        )

    p = Path(out_dir)
    if not p.is_dir():
        raise ValueError(f"output dir {p} does not exist")

    expected_names = {
        f"{e['save_basename']}_member000_y{e['Y']:04d}.nc" for e in plan
    }
    actual_names = {f.name for f in p.glob("Y*_member*_y*.nc")}

    missing = expected_names - actual_names
    if missing:
        raise ValueError(
            f"missing {len(missing)} expected NetCDFs in {p} "
            f"(mode={mode}): first few: {sorted(missing)[:5]}"
        )
    if mode == "exact":
        extra = actual_names - expected_names
        if extra:
            raise ValueError(
                f"unexpected {len(extra)} extra NetCDFs in {p} "
                f"(mode='exact'): first few: {sorted(extra)[:5]}"
            )
    # mode == 'subset': extras are allowed and ignored.

    for fname in sorted(expected_names):
        nc_path = p / fname
        assert_output_time_dim(nc_path, K)
        with xr.open_dataset(nc_path) as ds:
            actual_vars = frozenset(ds.data_vars)
        if actual_vars != expected_vars:
            raise ValueError(
                f"{nc_path.name}: data_vars {sorted(actual_vars)} != "
                f"expected {sorted(expected_vars)} "
                f"(missing: {sorted(expected_vars - actual_vars)}, "
                f"extra: {sorted(actual_vars - expected_vars)})"
            )


def assert_output_time_dim(nc_path: Path, K: int) -> None:
    """Raise ValueError unless the NetCDF's time dim equals K + 1 (IC + K leads)."""
    import xarray as xr

    assert_K_explicit(K)
    p = Path(nc_path)
    if not p.is_file():
        raise ValueError(f"output NetCDF not found: {p}")
    expected = K + 1
    with xr.open_dataset(p) as ds:
        if "time" not in ds.sizes:
            raise ValueError(
                f"{p} has no 'time' dim; available dims: {dict(ds.sizes)}"
            )
        actual = int(ds.sizes["time"])
    if actual != expected:
        raise ValueError(
            f"{p}: time dim == {actual}, expected K+1 = {expected} "
            f"(IC + {K} forecast leads)"
        )


__all__ = (
    "assert_K_explicit",
    "assert_final_datetime_matches",
    "assert_yaml_horizon",
    "assert_upstream_patched",
    "assert_upstream_patched_lp004",
    "assert_yamls_share_static_arch",
    "assert_output_dir_empty",
    "assert_output_dir_complete",
    "assert_output_time_dim",
)
