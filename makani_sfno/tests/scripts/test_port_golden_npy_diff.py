"""port_golden_npy_diff.py: bitwise means bitwise, and non-finite values are never hidden.

Tolerance ruling 7697688 §1.7: the old comparator used ``np.array_equal`` (NaN != NaN, and
-0.0 == 0.0), let NaN lose every argmax, and crashed when every difference was NaN.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "port_golden_npy_diff.py"


def _setup(tmp_path: Path, ref: np.ndarray, new: np.ndarray):
    run = tmp_path / "root" / "run"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"channel_names": ["PS", "T_l17"]}))
    spec = {"root": str(tmp_path / "root"), "ic": {"K": ref.shape[0]}, "checkpoints": [{"tag": "x", "run": "run"}]}
    (tmp_path / "list.json").write_text(json.dumps(spec))
    for d, arr in (("ref", ref), ("new", new)):
        (tmp_path / d).mkdir()
        np.save(tmp_path / d / ("x_K%d.npy" % ref.shape[0]), arr)
    p = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "ref"), str(tmp_path / "new"),
         "--list", str(tmp_path / "list.json"), "--label", "t"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
    )
    return p.returncode, p.stdout


def _field():
    return np.arange(2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 3, 4) + 1.0


def test_identical_nans_are_bitwise(tmp_path):
    a = _field()
    a[1, 0, 2, 3] = np.nan
    rc, out = _setup(tmp_path, a, a.copy())
    assert rc == 0, out
    assert "x bitwise=True nonfinite=1" in out
    assert "NPY_DIFF_SUMMARY t bitwise=1/1 nonfinite_mask_mismatch=0" in out


def test_signed_zero_is_not_bitwise(tmp_path):
    a = _field()
    a[0, 0, 0, 0] = 0.0
    b = a.copy()
    b[0, 0, 0, 0] = -0.0
    rc, out = _setup(tmp_path, a, b)
    assert rc == 0, out
    assert "x bitwise=False first_bad_lead=1" in out


def test_new_nan_fails_on_mask_mismatch(tmp_path):
    a = _field()
    b = a.copy()
    b[1, 1, 0, 0] = np.nan
    rc, out = _setup(tmp_path, a, b)
    assert rc == 1, out
    assert "ERROR NPY_DIFF t x nonfinite_mask_mismatch first_lead=2 nonfinite_ref=0 nonfinite_new=1" in out
    assert "nonfinite_mask_mismatch=1" in out


def test_all_differences_nan_does_not_crash(tmp_path):
    a = _field()[:1]
    b = np.full_like(a, np.nan)
    rc, out = _setup(tmp_path, a, b)
    assert rc == 1, out
    assert "no point finite in both" in out
