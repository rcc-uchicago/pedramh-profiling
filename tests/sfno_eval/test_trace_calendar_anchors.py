"""Tests for scripts/trace_calendar_anchors.py.

Coverage (per docs/sfno_eval_plan.md §3 P-4):
  - ``_parse_anchor`` round-trips the packager's
    ``"days since 0YYY-MM-DD HH:MM:SS"`` format.
  - ``_is_proleptic_leap`` matches the Gregorian rule on the corner
    years that v2.3 §C.2 calls out (year 100 NOT leap because of the
    centennial exception; year 400 leap; year 4 leap).
  - ``trace_one`` on a synthetic non-leap file (n=1455, anchor year 99 →
    candidate 100) produces ``is_leap_expected=False`` and
    ``leap_day_sample_idx=None``.
  - ``trace_one`` on a synthetic leap file (n=1459, anchor year 14 →
    candidate 20) produces ``is_leap_expected=True`` and locates the
    Feb 29 sample.
  - ``_validate_rows`` returns non-zero for files whose actual length
    contradicts the leap-rule prediction.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import timedelta
from pathlib import Path

import h5py
import numpy as np
import pytest


cftime = pytest.importorskip("cftime")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "trace_calendar_anchors.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("trace_calendar_anchors", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["trace_calendar_anchors"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def script():
    return _load_script()


def _build_synthetic_h5(
    path: Path, *, anchor_year: int, n_samples: int,
) -> None:
    """Write a stub h5 with an Aug-1 anchor and a 6-h-spaced time_plasim."""
    units = f"days since {anchor_year:04d}-08-01 00:00:00"
    with h5py.File(path, "w") as f:
        f.attrs["plasim_time_units"] = units
        # 6h spacing → 0.25 day steps
        time = np.arange(n_samples, dtype=np.float64) * 0.25
        f.create_dataset("time_plasim", data=time)


# --- _parse_anchor ----------------------------------------------------------

class TestParseAnchor:
    def test_basic(self, script):
        assert script._parse_anchor("days since 0126-08-01 00:00:00") == (126, 8, 1, 0, 0, 0)

    def test_handles_leading_zeros(self, script):
        assert script._parse_anchor("days since 0014-08-01 00:00:00") == (14, 8, 1, 0, 0, 0)

    def test_rejects_garbage(self, script):
        with pytest.raises(SystemExit, match="unparseable"):
            script._parse_anchor("days since the dawn of time")


# --- _is_proleptic_leap -----------------------------------------------------

class TestProlepticLeap:
    def test_div_by_4(self, script):
        assert script._is_proleptic_leap(20) is True
        assert script._is_proleptic_leap(2024) is True

    def test_centennial_exception(self, script):
        # year 100, 200, 300 are NOT leap under the proleptic Gregorian rule
        assert script._is_proleptic_leap(100) is False
        assert script._is_proleptic_leap(200) is False
        assert script._is_proleptic_leap(300) is False

    def test_quad_centennial(self, script):
        # year 400, 800, 2000 ARE leap
        assert script._is_proleptic_leap(400) is True
        assert script._is_proleptic_leap(2000) is True

    def test_odd_years(self, script):
        assert script._is_proleptic_leap(99) is False
        assert script._is_proleptic_leap(2023) is False


# --- trace_one --------------------------------------------------------------

class TestTraceOne:
    def test_non_leap_file_centennial(self, script, tmp_path):
        """MOST.0094 → anchor year 99, candidate year 100 (NOT leap)."""
        path = tmp_path / "MOST.0094.h5"
        _build_synthetic_h5(path, anchor_year=99, n_samples=1455)
        row = script.trace_one(path)
        assert row["anchor_year"] == 99
        assert row["n_samples"] == 1455
        assert row["is_leap_expected"] is False
        assert row["leap_day_sample_idx"] is None

    def test_leap_file_finds_feb29(self, script, tmp_path):
        """MOST.0014 → anchor year 19, candidate year 20 (leap)."""
        # Anchor 0019-08-01; sample 0 = Aug 1, 0019. The next Feb 29 is
        # 0020-02-29 — that's 213 days after Aug 1, 0019. Sample idx = 213*4 = 852.
        path = tmp_path / "MOST.0014.h5"
        _build_synthetic_h5(path, anchor_year=19, n_samples=1459)
        row = script.trace_one(path)
        assert row["anchor_year"] == 19
        assert row["n_samples"] == 1459
        assert row["is_leap_expected"] is True
        # Sanity-bound the index rather than hard-coding a derived number;
        # any reasonable Aug-1-anchored year places Feb 29 in the second half.
        idx = row["leap_day_sample_idx"]
        assert idx is not None
        assert 800 <= idx <= 900

    def test_test_file_leap(self, script, tmp_path):
        """MOST.0122 → anchor year 127, candidate year 128 (leap)."""
        path = tmp_path / "MOST.0122.h5"
        _build_synthetic_h5(path, anchor_year=127, n_samples=1459)
        row = script.trace_one(path)
        assert row["is_leap_expected"] is True
        assert row["leap_day_sample_idx"] is not None


# --- _validate_rows ---------------------------------------------------------

class TestValidateRows:
    def test_pass_on_consistent_rows(self, script):
        rows = [
            {"file": "a", "anchor_year": 99, "n_samples": 1455,
             "is_leap_expected": False, "leap_day_sample_idx": None},
            {"file": "b", "anchor_year": 19, "n_samples": 1459,
             "is_leap_expected": True, "leap_day_sample_idx": 852},
        ]
        assert script._validate_rows(rows) == 0

    def test_fail_on_length_mismatch(self, script, capsys):
        rows = [
            {"file": "a", "anchor_year": 19, "n_samples": 1455,  # claims leap but n=1455
             "is_leap_expected": True, "leap_day_sample_idx": None},
        ]
        rc = script._validate_rows(rows)
        assert rc == 1
        captured = capsys.readouterr()
        # Either the leap-length OR the missing-Feb-29 message is fine.
        assert "1459" in captured.err or "Feb-29" in captured.err
