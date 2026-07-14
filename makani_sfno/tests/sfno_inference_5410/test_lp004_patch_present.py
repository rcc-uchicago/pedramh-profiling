"""LP-004 partial-horizon patch presence check.

Strict count: exactly 1 occurrence of ``def reconfigure_for_ic`` in
upstream ``long_inference.py``. Any other count indicates partial
application or accidental duplication.

Skipped on machines without the upstream tree (CI, laptops).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sfno_inference_5410.preflight import (
    assert_upstream_boundary_phase,
    assert_upstream_patched_lp004,
)


_UPSTREAM_LONG_INFERENCE = Path(
    "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0/long_inference.py"
)


def test_lp004_marker_count_is_one():
    if not _UPSTREAM_LONG_INFERENCE.is_file():
        pytest.skip(f"upstream long_inference.py not present: {_UPSTREAM_LONG_INFERENCE}")
    text = _UPSTREAM_LONG_INFERENCE.read_text()
    n = text.count("def reconfigure_for_ic")
    assert n == 1, f"LP-004 marker count {n} != 1"


def test_assert_upstream_patched_lp004_passes():
    if not _UPSTREAM_LONG_INFERENCE.is_file():
        pytest.skip(f"upstream long_inference.py not present: {_UPSTREAM_LONG_INFERENCE}")
    # Will raise ValueError on any count != 1.
    assert_upstream_patched_lp004(_UPSTREAM_LONG_INFERENCE)


def test_assert_upstream_patched_lp004_rejects_unpatched(tmp_path):
    """Sanity-check that the helper flags a pristine file."""
    pristine = tmp_path / "pristine.py"
    pristine.write_text("# no reconfigure_for_ic anywhere\n")
    with pytest.raises(ValueError):
        assert_upstream_patched_lp004(pristine)


def test_assert_upstream_patched_lp004_rejects_double_apply(tmp_path):
    """Sanity-check that the helper flags duplicate application."""
    twice = tmp_path / "twice.py"
    twice.write_text(
        "def reconfigure_for_ic(self): pass\n"
        "def reconfigure_for_ic(self): pass\n"
    )
    with pytest.raises(ValueError):
        assert_upstream_patched_lp004(twice)


def test_assert_upstream_patched_lp004_missing_file(tmp_path):
    with pytest.raises(ValueError):
        assert_upstream_patched_lp004(tmp_path / "no_such_file.py")


def test_assert_upstream_boundary_phase_passes():
    if not _UPSTREAM_LONG_INFERENCE.is_file():
        pytest.skip(f"upstream long_inference.py not present: {_UPSTREAM_LONG_INFERENCE}")
    assert_upstream_boundary_phase(_UPSTREAM_LONG_INFERENCE)


def test_assert_upstream_boundary_phase_rejects_bad_offset(tmp_path):
    bad = tmp_path / "bad_offset.py"
    bad.write_text("params['nc_bc_offset'] = 18\n")
    with pytest.raises(ValueError):
        assert_upstream_boundary_phase(bad)
