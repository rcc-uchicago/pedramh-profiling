"""Tests for scripts/build_test_split.py.

Coverage (per docs/sfno_eval_plan.md §A.1):
  - ``_parse_year_spec`` accepts comma, range, and mixed forms.
  - ``build_test_split`` creates relative symlinks pointing at the source
    files, is idempotent, and refuses to clobber non-symlinks.
  - The packager-attribute sanity check (``f.attrs['split'] == 'test'``)
    blocks the operation entirely if any source file fails.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "build_test_split.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_test_split", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_test_split"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def script():
    return _load_script()


def _make_h5(path: Path, *, split: str = "test", n_samples: int = 8) -> None:
    """Write a tiny stand-in h5 file with a ``split`` attribute."""
    with h5py.File(path, "w") as f:
        f.attrs["split"] = split
        f.create_dataset("fields_state", data=np.zeros((n_samples, 1, 2, 2), dtype=np.float32))


# --- _parse_year_spec -------------------------------------------------------

class TestParseYearSpec:
    def test_comma_list(self, script):
        assert script._parse_year_spec("0121,0122,0123") == [121, 122, 123]

    def test_range(self, script):
        assert script._parse_year_spec("0121-0124") == [121, 122, 123, 124]

    def test_mixed(self, script):
        assert script._parse_year_spec("0121-0124,0126,0128") == [121, 122, 123, 124, 126, 128]

    def test_strips_whitespace_and_dedupes(self, script):
        assert script._parse_year_spec(" 0121 , 0122 , 0121 ") == [121, 122]


# --- build_test_split (happy path) ------------------------------------------

class TestBuildTestSplit:
    def test_creates_relative_symlinks(self, script, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        for y in (121, 122):
            _make_h5(src / f"MOST.{y:04d}.h5", split="test")

        n = script.build_test_split(src, dst, [121, 122])

        assert n == 2
        for y in (121, 122):
            link = dst / f"MOST.{y:04d}.h5"
            assert link.is_symlink()
            # symlink target is relative
            assert not os.readlink(link).startswith("/")
            # resolves to the source file
            assert link.resolve() == (src / f"MOST.{y:04d}.h5").resolve()

    def test_idempotent(self, script, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        _make_h5(src / "MOST.0121.h5", split="test")

        script.build_test_split(src, dst, [121])
        link = dst / "MOST.0121.h5"
        first_target = os.readlink(link)
        # second run should not change the target
        script.build_test_split(src, dst, [121])
        assert os.readlink(link) == first_target

    def test_refreshes_stale_symlink(self, script, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        _make_h5(src / "MOST.0121.h5", split="test")
        # plant a symlink pointing at the wrong place
        bogus = tmp_path / "bogus.h5"
        bogus.write_text("nope")
        (dst / "MOST.0121.h5").symlink_to(bogus)

        script.build_test_split(src, dst, [121])
        assert (dst / "MOST.0121.h5").resolve() == (src / "MOST.0121.h5").resolve()


# --- guard rails ------------------------------------------------------------

class TestSanityChecks:
    def test_rejects_wrong_split_attr(self, script, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        _make_h5(src / "MOST.0121.h5", split="train")  # wrong split

        with pytest.raises(SystemExit, match="split"):
            script.build_test_split(src, dst, [121])
        # nothing should have been written
        assert not dst.exists() or not list(dst.iterdir())

    def test_missing_source_file_aborts(self, script, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        # no MOST.0121.h5
        with pytest.raises(SystemExit, match="missing source file"):
            script.build_test_split(src, dst, [121])

    def test_refuses_to_clobber_non_symlink(self, script, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        _make_h5(src / "MOST.0121.h5", split="test")
        # plant a real file at the destination
        (dst / "MOST.0121.h5").write_text("not a symlink")

        with pytest.raises(SystemExit, match="non-symlink"):
            script.build_test_split(src, dst, [121])
