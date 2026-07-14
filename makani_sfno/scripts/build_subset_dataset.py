#!/usr/bin/env python3
"""build_subset_dataset.py — Symlink-farm builder for sfno_training tiny / short / full runs.

Builds a derived dataset directory that mirrors the packager-output
layout but contains only a chosen subset of train / valid files.
``stats/``, ``metadata/``, and ``config/`` are symlinked as-is from the
source dataset, so the subset uses the **full-dataset normalization**
(this keeps tiny / short / future-full results comparable, per
docs/sfno_tiny_short_training_plan.md §A.1).

Cross-split year lookup
-----------------------
A requested year is searched across ``src/{train,valid,test}/`` in that
order — the destination split (e.g. ``dst/train/``) does NOT need to
match the source split where the H5 file lives. This supports schemes
like the group convention (train: 12-111, valid: 11) on a packager
output that put years 3-100 under ``src/train/`` and 101-120 under
``src/valid/``: the new dst/train/ pulls 12-100 from src/train and
101-111 from src/valid; dst/valid/MOST.0011.h5 pulls from src/train.
Precedence is train > valid > test (defensive against duplicate files).

Layout produced::

    {dst}/
    ├── train/MOST.{YYYY}.h5    # symlinks → {src}/{train|valid|test}/MOST.{YYYY}.h5
    ├── valid/MOST.{YYYY}.h5    # symlinks → {src}/{train|valid|test}/MOST.{YYYY}.h5
    ├── test/                    # empty
    ├── stats/                   # symlink → {src}/stats
    ├── metadata/                # symlink → {src}/metadata
    └── config/                  # symlink → {src}/config

Idempotent: re-running on an existing target replaces stale symlinks but
preserves matching ones. The ``test/`` split is always created (empty)
because PlasimTrainer's startup walks ``params.inf_data_path``; an
empty directory is sufficient (no test rollout in tiny / short / full).

Usage::

    scripts/build_subset_dataset.py \\
        --src $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128 \\
        --dst $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_tiny \\
        --train-years 3 \\
        --valid-years 101

    scripts/build_subset_dataset.py \\
        --src $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128 \\
        --dst $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_short \\
        --train-years 3-7 \\
        --valid-years 101-102

    scripts/build_subset_dataset.py \\
        --src $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128 \\
        --dst $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_full \\
        --train-years 12-111 \\
        --valid-years 11
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("build_subset_dataset")

LINKED_DIRS = ("stats", "metadata", "config")
SPLITS_AS_DIRS = ("train", "valid", "test")


def _parse_year_spec(spec: str) -> list[int]:
    """Parse a comma-separated mix of single years and inclusive ranges.

    ``"3"`` → [3]; ``"3-7"`` → [3,4,5,6,7]; ``"3,5,101-102"`` → [3,5,101,102].
    """
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo_i, hi_i = int(lo), int(hi)
            if lo_i > hi_i:
                raise ValueError(f"bad range {part!r}: lo > hi")
            out.extend(range(lo_i, hi_i + 1))
        else:
            out.append(int(part))
    if not out:
        raise ValueError(f"no years parsed from {spec!r}")
    return out


def _file_for_year(split_dir: Path, year: int) -> Path:
    return split_dir / f"MOST.{year:04d}.h5"


def _find_year_file(src: Path, year: int) -> Path:
    """Search ``src/train``, ``src/valid``, ``src/test`` (in that order) for
    ``MOST.{year:04d}.h5``. Precedence is train > valid > test (defensive,
    since our packager output has no duplicates). Raises FileNotFoundError
    if not found anywhere."""
    for split in SPLITS_AS_DIRS:
        candidate = src / split / f"MOST.{year:04d}.h5"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"year {year}: MOST.{year:04d}.h5 not found in src/{{train,valid,test}}/"
    )


def _replace_symlink(target: Path, link: Path) -> None:
    """Create / refresh a symlink at ``link`` pointing to ``target``.

    Idempotent: if a correct symlink already exists, no-op. If a stale
    or broken symlink exists, replace it. If a regular file/dir blocks
    the path, raise (we won't silently delete real data).
    """
    target = target.resolve()
    if link.is_symlink():
        try:
            current = link.resolve()
        except OSError:
            current = None
        if current == target:
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(
            f"{link} exists and is not a symlink; refusing to overwrite. "
            f"Remove it manually if you intended to."
        )
    os.symlink(target, link)
    logger.debug("linked %s → %s", link, target)


def _link_split(src: Path, dst_split: Path, years: Iterable[int], split_name: str) -> int:
    """Symlink each requested year into ``dst_split``. Source file is
    located by ``_find_year_file`` — i.e. searched across all source
    splits, so the dst split is purely a destination concept."""
    dst_split.mkdir(parents=True, exist_ok=True)
    n = 0
    for year in years:
        try:
            src_file = _find_year_file(src, year)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{split_name} year {year}: {e}") from None
        _replace_symlink(src_file, _file_for_year(dst_split, year))
        n += 1
    return n


def build_subset(src: Path, dst: Path, train_years: list[int], valid_years: list[int]) -> None:
    """Build the symlink farm. See module docstring for layout."""
    src = src.resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"--src {src} is not a directory")
    dst.mkdir(parents=True, exist_ok=True)

    # Sanity: all required source subdirs present.
    missing = [d for d in (*SPLITS_AS_DIRS, *LINKED_DIRS) if not (src / d).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"--src {src} missing required subdirectories: {missing}"
        )

    for d in LINKED_DIRS:
        _replace_symlink(src / d, dst / d)

    n_train = _link_split(src, dst / "train", train_years, "train")
    n_valid = _link_split(src, dst / "valid", valid_years, "valid")
    (dst / "test").mkdir(parents=True, exist_ok=True)

    logger.info(
        "built %s — %d train file(s), %d valid file(s), test/ empty",
        dst, n_train, n_valid,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a symlink-farm subset of a packaged dataset, sharing the "
            "stats / metadata / config of the source dataset."
        ),
    )
    p.add_argument("--src", required=True, type=Path,
                   help="Source packaged-dataset root (contains train/, valid/, stats/, ...).")
    p.add_argument("--dst", required=True, type=Path,
                   help="Destination subset root.")
    p.add_argument("--train-years", required=True, type=str,
                   help="Train years: '3', '3-7', or '3,5,7'.")
    p.add_argument("--valid-years", required=True, type=str,
                   help="Valid years: '101', '101-102', or '101,103'.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    train_years = _parse_year_spec(args.train_years)
    valid_years = _parse_year_spec(args.valid_years)
    try:
        build_subset(args.src, args.dst, train_years, valid_years)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
