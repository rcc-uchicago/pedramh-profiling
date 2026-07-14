#!/usr/bin/env python3
"""build_test_split.py — Symlink the 8 held-out test files for emulator eval.

Per docs/sfno_eval_plan.md §A.1, this stages the canonical packager test
files (``MOST.0121.h5`` .. ``MOST.0128.h5``) under a sibling directory
``test_holdout/`` so that eval inputs cannot accidentally contaminate the
``test/`` split that ``build_subset_dataset.py`` carved for retraining.

For each requested year, a relative symlink is created::

    {dst}/MOST.{YYYY}.h5  →  {src}/MOST.{YYYY}.h5

Behavior is idempotent: an existing matching symlink is left alone, a
stale or wrong-target symlink is replaced.

Sanity check baked in: every source file is opened read-only and the
``split`` h5 attribute must equal ``"test"`` (verified by the
plasim-makani packager at write time, see
``src/plasim_makani_packager/`` and the audit at
``docs/audit_snapshots/exp25_manifest.txt``). If any file fails the
check the script exits non-zero before any symlinks are written.

Usage::

    scripts/build_test_split.py \\
        --src $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128/test \\
        --dst $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_full/test_holdout \\
        --years 0121,0122,0123,0124,0125,0126,0127,0128
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import h5py


# --- year-spec parsing ------------------------------------------------------

def _parse_year_spec(spec: str) -> list[int]:
    """Parse a comma- or range-separated year spec into a sorted unique list.

    Accepts: ``"0121,0122"`` or ``"0121-0128"`` or mixed
    ``"0121-0124,0126,0128"``. Padding is preserved on output so callers
    can reconstruct ``MOST.{year:04d}.h5`` filenames directly.
    """
    out: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(chunk))
    return sorted(out)


# --- core ops ---------------------------------------------------------------

def _check_split_attr(src_file: Path, expected: str = "test") -> None:
    """Open ``src_file`` and assert ``f.attrs['split'] == expected``."""
    with h5py.File(src_file, "r") as f:
        actual = f.attrs.get("split")
        # h5py returns bytes for str attrs in some versions
        if isinstance(actual, bytes):
            actual = actual.decode("utf-8")
        if actual != expected:
            raise SystemExit(
                f"sanity check failed: {src_file} has split={actual!r}, "
                f"expected {expected!r}. Refusing to symlink."
            )


def _replace_symlink(target: Path, link: Path) -> None:
    """Create or refresh ``link`` so it points at ``target`` (relative)."""
    rel_target = os.path.relpath(target, link.parent)
    if link.is_symlink():
        if os.readlink(link) == rel_target:
            return  # already correct
        link.unlink()
    elif link.exists():
        raise SystemExit(
            f"refusing to overwrite non-symlink at {link}; "
            "remove it manually if intentional"
        )
    link.symlink_to(rel_target)


def build_test_split(src: Path, dst: Path, years: Iterable[int]) -> int:
    """Symlink each ``MOST.{YYYY}.h5`` from ``src`` into ``dst``.

    Returns the number of links created or refreshed.
    """
    src = src.resolve()
    if not src.is_dir():
        raise SystemExit(f"--src does not exist or is not a directory: {src}")

    years = list(years)
    sources: list[tuple[int, Path]] = []
    for y in years:
        s = src / f"MOST.{y:04d}.h5"
        if not s.is_file():
            raise SystemExit(f"missing source file: {s}")
        sources.append((y, s))

    # Sanity check ALL files first; only mutate the filesystem if every
    # source passes. Avoids leaving a half-built test_holdout/ on failure.
    for _, s in sources:
        _check_split_attr(s, expected="test")

    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for y, s in sources:
        link = dst / f"MOST.{y:04d}.h5"
        _replace_symlink(s, link)
        n += 1
    return n


# --- CLI --------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Symlink the 8 held-out test files for emulator eval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--src", required=True, type=Path,
        help="Source directory containing MOST.NNNN.h5 (e.g. .../sim52_astro_64x128/test/)",
    )
    p.add_argument(
        "--dst", required=True, type=Path,
        help="Destination directory for symlinks (e.g. .../sim52_full/test_holdout/)",
    )
    p.add_argument(
        "--years", required=True, type=str,
        help="Comma- or range-separated year list (e.g. '0121,0122' or '0121-0128')",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    years = _parse_year_spec(args.years)
    n = build_test_split(args.src, args.dst, years)
    print(f"linked {n} file(s) under {args.dst}", file=sys.stderr)


if __name__ == "__main__":
    main()
