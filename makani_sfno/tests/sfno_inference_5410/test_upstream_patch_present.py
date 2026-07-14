"""Verify the 6-hunk partial-horizon patch is present in upstream long_inference.py.

Strict counts (per docs/2026-05-08_sfno_5410_explicit_K_horizon_plan.md):
  * Exactly 4 allocator markers ``min(next_year_jan1, self.params.final_datetime)``
  * Exactly 2 continuation markers ``current_datetime < self.params.final_datetime``

Skipped on machines without the upstream tree (CI nodes, laptops).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sfno_inference_5410.preflight import assert_upstream_patched


_UPSTREAM_LONG_INFERENCE = Path(
    "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0/long_inference.py"
)


def test_upstream_patch_present():
    if not _UPSTREAM_LONG_INFERENCE.is_file():
        pytest.skip(f"upstream long_inference.py not present: {_UPSTREAM_LONG_INFERENCE}")
    # Will raise ValueError on partial / missing / duplicate apply.
    assert_upstream_patched(_UPSTREAM_LONG_INFERENCE)


def test_strict_counts():
    if not _UPSTREAM_LONG_INFERENCE.is_file():
        pytest.skip(f"upstream long_inference.py not present: {_UPSTREAM_LONG_INFERENCE}")
    text = _UPSTREAM_LONG_INFERENCE.read_text()
    alloc = text.count("min(next_year_jan1, self.params.final_datetime)")
    cont = text.count("current_datetime < self.params.final_datetime")
    assert alloc == 4, f"expected 4 allocator markers, got {alloc}"
    assert cont == 2, f"expected 2 continuation markers, got {cont}"

    # No stragglers from the pre-patch year-only continuation.
    assert "current_year < self.params.final_datetime.year" not in text, (
        "old year-only continuation marker still present — patch is incomplete"
    )


def test_helper_rejects_unpatched(tmp_path):
    """Sanity-check assert_upstream_patched flags a pristine file."""
    pristine = tmp_path / "pristine.py"
    pristine.write_text(
        "x = self.dataset.datetime_class(current_year+1, 1, 1, hour=0, has_year_zero=True)\n"
        "if current_year < self.params.final_datetime.year:\n"
        "    pass\n"
    )
    with pytest.raises(ValueError):
        assert_upstream_patched(pristine)
