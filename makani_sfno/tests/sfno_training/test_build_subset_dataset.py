"""Tests for scripts/build_subset_dataset.py.

Coverage (per docs/sfno_tiny_short_training_plan.md §Implementation deliverables):
  - Symlink farm has correct structure (train/, valid/, test/, stats/, ...).
  - Year-spec parser handles single, range, and comma forms.
  - Re-running on an existing target is idempotent.
  - Missing source year raises a clear error.
  - Refuses to overwrite a non-symlink at a target path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUILDER = _REPO_ROOT / "scripts" / "build_subset_dataset.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_subset_dataset", _BUILDER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_subset_dataset"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def builder():
    return _load_builder()


@pytest.fixture
def fake_packaged_dataset(tmp_path: Path) -> Path:
    """Packaged-dataset look-alike: train/MOST.{0003..0010}.h5, valid/MOST.{0101..0102}.h5,
    plus stats/, metadata/, config/, test/."""
    src = tmp_path / "src"
    for split, years in (("train", range(3, 11)), ("valid", (101, 102)), ("test", (121,))):
        d = src / split
        d.mkdir(parents=True)
        for y in years:
            (d / f"MOST.{y:04d}.h5").write_bytes(b"")
    for d in ("stats", "metadata", "config"):
        (src / d).mkdir()
    (src / "stats" / "global_means.npy").write_bytes(b"")
    (src / "metadata" / "data.json").write_text("{}")
    (src / "config" / "x.yaml").write_text("x: 1")
    return src


# ---------------------------------------------------------------------------
# Year-spec parser
# ---------------------------------------------------------------------------
class TestParseYearSpec:
    def test_single(self, builder):
        assert builder._parse_year_spec("3") == [3]

    def test_range(self, builder):
        assert builder._parse_year_spec("3-7") == [3, 4, 5, 6, 7]

    def test_comma(self, builder):
        assert builder._parse_year_spec("3,5,101") == [3, 5, 101]

    def test_mixed(self, builder):
        assert builder._parse_year_spec("3,5-7,101") == [3, 5, 6, 7, 101]

    def test_bad_range(self, builder):
        with pytest.raises(ValueError):
            builder._parse_year_spec("7-3")

    def test_empty(self, builder):
        with pytest.raises(ValueError):
            builder._parse_year_spec("")


# ---------------------------------------------------------------------------
# build_subset
# ---------------------------------------------------------------------------
class TestBuildSubset:
    def test_layout(self, builder, fake_packaged_dataset, tmp_path):
        dst = tmp_path / "dst"
        builder.build_subset(fake_packaged_dataset, dst, [3], [101])

        # train: only the requested year, as a symlink to the source file
        train = dst / "train"
        assert (train / "MOST.0003.h5").is_symlink()
        assert (train / "MOST.0003.h5").resolve() == (
            fake_packaged_dataset / "train" / "MOST.0003.h5"
        ).resolve()
        assert not (train / "MOST.0004.h5").exists()

        # valid: only the requested year
        assert (dst / "valid" / "MOST.0101.h5").is_symlink()
        assert not (dst / "valid" / "MOST.0102.h5").exists()

        # test/: empty dir
        assert (dst / "test").is_dir()
        assert list((dst / "test").iterdir()) == []

        # stats / metadata / config — symlinks to source dirs
        for d in ("stats", "metadata", "config"):
            assert (dst / d).is_symlink()
            assert (dst / d).resolve() == (fake_packaged_dataset / d).resolve()

    def test_short_layout(self, builder, fake_packaged_dataset, tmp_path):
        dst = tmp_path / "dst_short"
        builder.build_subset(fake_packaged_dataset, dst, [3, 4, 5, 6, 7], [101, 102])
        for y in (3, 4, 5, 6, 7):
            assert (dst / "train" / f"MOST.{y:04d}.h5").is_symlink()
        for y in (101, 102):
            assert (dst / "valid" / f"MOST.{y:04d}.h5").is_symlink()

    def test_idempotent(self, builder, fake_packaged_dataset, tmp_path):
        dst = tmp_path / "dst"
        builder.build_subset(fake_packaged_dataset, dst, [3], [101])
        # Capture inodes / targets, then re-run, then re-check.
        target_before = (dst / "train" / "MOST.0003.h5").resolve()
        builder.build_subset(fake_packaged_dataset, dst, [3], [101])
        target_after = (dst / "train" / "MOST.0003.h5").resolve()
        assert target_before == target_after
        assert (dst / "train" / "MOST.0003.h5").is_symlink()

    def test_change_subset_replaces_links(self, builder, fake_packaged_dataset, tmp_path):
        """Re-running with a different valid year must add the new link;
        we don't aggressively delete the old one (it's still a valid
        symlink to a real file), but the new link must be present."""
        dst = tmp_path / "dst"
        builder.build_subset(fake_packaged_dataset, dst, [3], [101])
        builder.build_subset(fake_packaged_dataset, dst, [3], [102])
        assert (dst / "valid" / "MOST.0102.h5").is_symlink()

    def test_missing_source_year(self, builder, fake_packaged_dataset, tmp_path):
        dst = tmp_path / "dst"
        with pytest.raises(FileNotFoundError, match="train year 999"):
            builder.build_subset(fake_packaged_dataset, dst, [999], [101])


# ---------------------------------------------------------------------------
# Cross-split year lookup (docs/sfno_full_training_plan.md §A.1)
# ---------------------------------------------------------------------------
class TestCrossSplitLookup:
    def test_train_year_from_valid_split(self, builder, fake_packaged_dataset, tmp_path):
        """Year 105 lives only in src/valid/ (per the fixture: valid={101,102}).
        Requesting year 105 in dst/train must pull it from src/valid/ via
        the cross-split lookup."""
        # Add year 105 to src/valid/ (the fixture only has 101, 102).
        (fake_packaged_dataset / "valid" / "MOST.0105.h5").write_bytes(b"")
        dst = tmp_path / "dst"
        builder.build_subset(fake_packaged_dataset, dst, [105], [101])

        link = dst / "train" / "MOST.0105.h5"
        assert link.is_symlink()
        assert link.resolve() == (
            fake_packaged_dataset / "valid" / "MOST.0105.h5"
        ).resolve(), (
            "year 105 should have been pulled from src/valid/ into dst/train/"
        )

    def test_valid_year_from_train_split(self, builder, fake_packaged_dataset, tmp_path):
        """Year 11 lives only in src/train/ (per the fixture: train={3..10}).
        Add it to src/train/ explicitly, then request it as a valid year —
        cross-split lookup must find it under src/train/."""
        (fake_packaged_dataset / "train" / "MOST.0011.h5").write_bytes(b"")
        dst = tmp_path / "dst"
        builder.build_subset(fake_packaged_dataset, dst, [3], [11])

        link = dst / "valid" / "MOST.0011.h5"
        assert link.is_symlink()
        assert link.resolve() == (
            fake_packaged_dataset / "train" / "MOST.0011.h5"
        ).resolve(), (
            "year 11 should have been pulled from src/train/ into dst/valid/"
        )

    def test_year_not_in_any_split(self, builder, fake_packaged_dataset, tmp_path):
        """Year 9999 is in no split — error must mention all three split dirs."""
        dst = tmp_path / "dst"
        with pytest.raises(FileNotFoundError, match=r"train,valid,test"):
            builder.build_subset(fake_packaged_dataset, dst, [9999], [101])

    def test_precedence_train_over_valid(self, builder, fake_packaged_dataset, tmp_path):
        """If the same year exists in both src/train/ and src/valid/, the
        train one wins (precedence: train > valid > test)."""
        # The fixture has year 3 in train; add a duplicate to valid/.
        (fake_packaged_dataset / "valid" / "MOST.0003.h5").write_bytes(b"DUP")
        dst = tmp_path / "dst"
        builder.build_subset(fake_packaged_dataset, dst, [3], [101])

        link = dst / "train" / "MOST.0003.h5"
        assert link.is_symlink()
        assert link.resolve() == (
            fake_packaged_dataset / "train" / "MOST.0003.h5"
        ).resolve(), "src/train should win over src/valid for the same year"

    def test_full_scheme_smoke(self, builder, fake_packaged_dataset, tmp_path):
        """End-to-end smoke for the planned full-run year split:
        train spans across both src/train (year 3) and src/valid (year 101);
        valid pulls from src/train (year 5)."""
        dst = tmp_path / "dst_full_smoke"
        builder.build_subset(fake_packaged_dataset, dst, [3, 101], [5])

        assert (dst / "train" / "MOST.0003.h5").resolve() == (
            fake_packaged_dataset / "train" / "MOST.0003.h5"
        ).resolve()
        assert (dst / "train" / "MOST.0101.h5").resolve() == (
            fake_packaged_dataset / "valid" / "MOST.0101.h5"
        ).resolve()
        assert (dst / "valid" / "MOST.0005.h5").resolve() == (
            fake_packaged_dataset / "train" / "MOST.0005.h5"
        ).resolve()

    def test_missing_source_subdir(self, builder, fake_packaged_dataset, tmp_path):
        # Remove stats/ to trigger the precheck.
        (fake_packaged_dataset / "stats" / "global_means.npy").unlink()
        (fake_packaged_dataset / "stats").rmdir()
        dst = tmp_path / "dst"
        with pytest.raises(FileNotFoundError, match="missing required"):
            builder.build_subset(fake_packaged_dataset, dst, [3], [101])

    def test_refuses_to_overwrite_real_dir(self, builder, fake_packaged_dataset, tmp_path):
        """If a regular dir already exists at dst/stats, refuse — won't
        silently delete the user's data."""
        dst = tmp_path / "dst"
        (dst / "stats").mkdir(parents=True)
        (dst / "stats" / "real_file.txt").write_text("hi")
        with pytest.raises(FileExistsError):
            builder.build_subset(fake_packaged_dataset, dst, [3], [101])
