"""T.3 — boundary-window expansion (Phase F, plan v5 §B4).

Mirrors the loader's offset-everywhere semantics from
``data_loader_multifiles.py``:
  - Lines 954/958/960 (index==0, after canonical-year remap) and 982 (index>0):
    ``start_time = ... + dates[k]*6h + nc_bc_offset(=18h)``.
  - Lines 926-934 (`_get_boundary_data`): canonical year via leap_year/no_leap_year
    based on the already-offset ``data_dt.year``.

For an init at YEAR=121 (non-leap) rolling 1460 steps to 0122-01-01:
  - All reads target canonical year 11.
  - data_idx ∈ {0, 1, 2, 3, 4, ..., 1459} (full padded zone).

For an init at YEAR=124 (leap) rolling 1464 steps to 0125-01-01:
  - 1461 reads target canonical year 12 with data_idx ∈ {3, 4, ..., 1463}.
  - 3 reads target canonical year 11 with data_idx ∈ {0, 1, 2}.
  - Year 12 idx {0, 1, 2} are NEVER read.
  - For all 8 test-year inits (121..128), NO read targets canonical_year ∈ {121..128}.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import cftime
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _expand_boundary_reads(
    init_dt: cftime.DatetimeProlepticGregorian,
    final_dt: cftime.DatetimeProlepticGregorian,
    *,
    leap_year: int = 12,
    no_leap_year: int = 11,
    nc_bc_offset_hours: int = 18,
    data_timedelta_hours: int = 6,
) -> set[tuple[int, int]]:
    """Mirror data_loader_multifiles.py:954/958/960/982 + 926-934.

    For k in [0, year_buffer):
      data_dt = init_dt + (k * 6 + nc_bc_offset_hours) hours
      canonical = leap_year(12) if is_leap(data_dt.year) else no_leap_year(11)
      data_idx = ((data_dt - Jan1(data_dt.year)) // 6h)
    """
    pairs: set[tuple[int, int]] = set()
    n_steps = int((final_dt - init_dt).total_seconds() // 3600 // data_timedelta_hours)
    for k in range(n_steps):
        data_dt = init_dt + timedelta(hours=k * data_timedelta_hours + nc_bc_offset_hours)
        data_year = data_dt.year
        is_leap = cftime.is_leap_year(data_year, "proleptic_gregorian", has_year_zero=True)
        canonical = leap_year if is_leap else no_leap_year
        jan1 = cftime.DatetimeProlepticGregorian(data_year, 1, 1, 0, has_year_zero=True)
        data_idx = int((data_dt - jan1).total_seconds() // 3600 // data_timedelta_hours)
        pairs.add((canonical, data_idx))
    return pairs


def _init(year: int) -> cftime.DatetimeProlepticGregorian:
    return cftime.DatetimeProlepticGregorian(year, 1, 1, 0, has_year_zero=True)


# --- YEAR=121 (non-leap) ---


def test_year121_pairs_all_canonical_11() -> None:
    pairs = _expand_boundary_reads(_init(121), _init(122))
    assert all(c == 11 for (c, _) in pairs), \
        f"unexpected canonical year(s): {sorted({c for (c,_) in pairs})}"


def test_year121_data_idx_set_full_padded_zone() -> None:
    pairs = _expand_boundary_reads(_init(121), _init(122))
    idx_set = sorted(i for (_, i) in pairs)
    assert idx_set == list(range(0, 1460)), (
        f"YEAR=121 must read year 11 idx 0..1459 (full padded zone). "
        f"Got idx range [{idx_set[0]}..{idx_set[-1]}], len={len(idx_set)}"
    )


def test_year121_max_idx_within_padded_bound() -> None:
    pairs = _expand_boundary_reads(_init(121), _init(122))
    max_idx = max(i for (_, i) in pairs)
    # Year 11 padded to 1460 frames (idx 0..1459).
    assert max_idx == 1459, f"YEAR=121 max idx {max_idx} != 1459"


# --- YEAR=124 (leap) ---


def test_year124_pairs_two_canonicals() -> None:
    pairs = _expand_boundary_reads(_init(124), _init(125))
    canonicals = {c for (c, _) in pairs}
    assert canonicals == {11, 12}, f"YEAR=124 should hit {{11, 12}}, got {canonicals}"


def test_year124_year12_indices_skip_0_1_2() -> None:
    pairs = _expand_boundary_reads(_init(124), _init(125))
    year12_idx = sorted(i for (c, i) in pairs if c == 12)
    # Plan v5 §B4: year 12 idx {0, 1, 2} are NEVER read for YEAR=124.
    assert year12_idx == list(range(3, 1464)), (
        f"YEAR=124 year-12 indices should be {{3..1463}}; got [{year12_idx[0]}..{year12_idx[-1]}]"
    )


def test_year124_year11_indices_just_0_1_2() -> None:
    pairs = _expand_boundary_reads(_init(124), _init(125))
    year11_idx = sorted(i for (c, i) in pairs if c == 11)
    assert year11_idx == [0, 1, 2], f"YEAR=124 year-11 indices should be [0,1,2]; got {year11_idx}"


def test_year124_max_idx_within_padded_bound() -> None:
    pairs = _expand_boundary_reads(_init(124), _init(125))
    year12_max = max(i for (c, i) in pairs if c == 12)
    assert year12_max == 1463, f"YEAR=124 max year-12 idx {year12_max} != 1463"


# --- All 8 test years: no canonical 121..128 ---


@pytest.mark.parametrize("year", [121, 122, 123, 124, 125, 126, 127, 128])
def test_no_test_year_in_canonical_set(year: int) -> None:
    """Plan v5 §B3: years 121-128 must NEVER be read from group H5 data_dir."""
    pairs = _expand_boundary_reads(_init(year), _init(year + 1))
    forbidden = {c for (c, _) in pairs} & set(range(121, 129))
    assert not forbidden, (
        f"YEAR={year}: canonical years {sorted(forbidden)} should be excluded "
        f"(loader remaps to leap_year/no_leap_year)."
    )


@pytest.mark.parametrize("year", [121, 122, 123, 124, 125, 126, 127, 128])
def test_canonical_year_matches_leap_status(year: int) -> None:
    pairs = _expand_boundary_reads(_init(year), _init(year + 1))
    is_leap_year = cftime.is_leap_year(year, "proleptic_gregorian", has_year_zero=True)
    main_canonical = 12 if is_leap_year else 11
    # Most pairs (>= 1455) should be at main_canonical.
    main_count = sum(1 for (c, _) in pairs if c == main_canonical)
    assert main_count >= 1455, (
        f"YEAR={year} (leap={is_leap_year}): only {main_count} reads at canonical "
        f"{main_canonical}; expected >= 1455"
    )


# --- Padding bound sanity ---


def test_year11_padding_size_matches_max_read() -> None:
    """Padded year 11 = 1460 frames; max read idx = 1459. Bound is exact."""
    from sfno_training_group.tools.convert_v10_to_group_h5 import PAD_TARGETS
    pairs = _expand_boundary_reads(_init(121), _init(122))
    max_idx = max(i for (c, i) in pairs if c == 11)
    assert max_idx + 1 == PAD_TARGETS[11]["n_padded"], (
        f"max year-11 read idx + 1 = {max_idx + 1} != n_padded {PAD_TARGETS[11]['n_padded']}"
    )


def test_year12_padding_size_matches_max_read() -> None:
    from sfno_training_group.tools.convert_v10_to_group_h5 import PAD_TARGETS
    pairs = _expand_boundary_reads(_init(124), _init(125))
    max_idx = max(i for (c, i) in pairs if c == 12)
    assert max_idx + 1 == PAD_TARGETS[12]["n_padded"], (
        f"max year-12 read idx + 1 = {max_idx + 1} != n_padded {PAD_TARGETS[12]['n_padded']}"
    )
