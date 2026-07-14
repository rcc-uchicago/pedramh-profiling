"""T.7 — documentation-by-test: years 121-128 must NOT exist in the group H5 data_dir.

Per plan v5 single_ic boundary remap: long_inference reads boundary forcing
from canonical years 11/12 only (data_loader_multifiles.py:926-934). Years
121-128 are read elsewhere as v10 source for IC NetCDFs and truth — never
from the group H5 data_dir. This test fails loudly if a future change
re-introduces them.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

PRODUCTION_DATA_DIR = Path(
    os.environ.get("AIRES_GROUP_DATA_DIR_FULL",
                   f"{os.environ.get('SCRATCH', '/scratch/_undefined')}/AI-RES/data/group_sfno/sim52_full")
)


@pytest.mark.skipif(
    not PRODUCTION_DATA_DIR.is_dir(),
    reason=f"production data_dir not present: {PRODUCTION_DATA_DIR}"
)
def test_no_test_year_files_in_production_data_dir() -> None:
    """Walk data_dir; every <year>_<idx>.h5 must have year in {11} ∪ {12..111}.

    Test years 121-128 are intentionally absent. Long_inference single_ic
    reads canonical years 11/12 instead.
    """
    pattern = re.compile(r"^(\d+)_\d{4}\.h5$")
    forbidden_years = set(range(121, 129))
    allowed_years = {11} | set(range(12, 112))

    seen_years: set[int] = set()
    for path in PRODUCTION_DATA_DIR.iterdir():
        m = pattern.match(path.name)
        if not m:
            continue
        year = int(m.group(1))
        seen_years.add(year)

    forbidden_present = seen_years & forbidden_years
    assert not forbidden_present, (
        f"FORBIDDEN test-year files in data_dir: {sorted(forbidden_present)}. "
        f"Long_inference single_ic remaps boundary reads to canonical years 11/12, "
        f"so these files would never be read AND would waste disk. "
        f"See plan v5 §B3 + data_loader_multifiles.py:926-934."
    )
    unexpected = seen_years - allowed_years
    assert not unexpected, (
        f"unexpected years in data_dir: {sorted(unexpected)} "
        f"(expected subset of {sorted(allowed_years)})"
    )


def test_pad_targets_excludes_test_years() -> None:
    """Static check: PAD_TARGETS must NOT include 121-128 (they're never read)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from sfno_training_group.tools.convert_v10_to_group_h5 import PAD_TARGETS  # noqa: E402

    for year in range(121, 129):
        assert year not in PAD_TARGETS, (
            f"year {year} in PAD_TARGETS — must be removed; long_inference "
            f"single_ic remaps to leap_year/no_leap_year, so test-year files "
            f"are never read. See plan v5."
        )
    # Confirm only canonical pads.
    assert set(PAD_TARGETS.keys()) == {11, 12}
