#!/usr/bin/env python3
"""trace_calendar_anchors.py — Sanity-check the +5 file-index → anchor-year offset.

Per docs/sfno_eval_plan.md §3 P-4 and §C.2, each PlaSim packager file
``MOST.000Y.h5`` carries an h5 attribute
``plasim_time_units = 'days since 0(Y+5)-08-01 00:00:00'`` and a
``time_plasim`` dataset of length 1455 (non-leap) or 1459 (leap), with
values ``[0.0, 0.25, ..., n-1)*0.25]`` (days since the anchor).

This script reads a small set of representative files and prints, for
each, a CSV row with::

    file, anchor_year, n_samples, is_leap_expected, leap_day_sample_idx

so the user can eyeball that:
  - non-leap files (e.g. MOST.0094, MOST.0128) have n_samples == 1455 and
    no Feb-29 sample;
  - leap files (e.g. MOST.0014, MOST.0122) have n_samples == 1459 and
    exactly four samples whose absolute datetime falls on Feb 29.

If everything checks out, §C.2 climatology can be built without further
manual intervention.

Usage::

    scripts/trace_calendar_anchors.py --src $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_full/train \\
        --files MOST.0014.h5,MOST.0094.h5
    scripts/trace_calendar_anchors.py --src $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128/test \\
        --files MOST.0122.h5,MOST.0128.h5
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import h5py


_ANCHOR_RE = re.compile(
    r"days since (\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})"
)


def _parse_anchor(units: str) -> tuple[int, int, int, int, int, int]:
    """Parse 'days since YYYY-MM-DD HH:MM:SS' → (Y, M, D, h, m, s)."""
    m = _ANCHOR_RE.match(units)
    if m is None:
        raise SystemExit(f"unparseable plasim_time_units: {units!r}")
    return tuple(int(g) for g in m.groups())  # type: ignore[return-value]


def _is_proleptic_leap(year: int) -> bool:
    """Proleptic Gregorian leap rule, as used by cftime."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _absolute_datetime(anchor: tuple[int, int, int, int, int, int], days: float):
    """Return a cftime.DatetimeProlepticGregorian for anchor + days."""
    import cftime  # local import — keeps module import cheap
    Y, M, D, h, m, s = anchor
    base = cftime.DatetimeProlepticGregorian(Y, M, D, h, m, s)
    return base + timedelta(days=float(days))


def trace_one(path: Path) -> dict:
    """Open one h5 and produce a single audit row."""
    with h5py.File(path, "r") as f:
        units = f.attrs["plasim_time_units"]
        if isinstance(units, bytes):
            units = units.decode("utf-8")
        time_plasim = f["time_plasim"][:]
        n_samples = int(time_plasim.shape[0])

    anchor = _parse_anchor(units)
    anchor_year = anchor[0]
    candidate_year = anchor_year + 1  # the year we may pick up Feb 29 from
    is_leap_expected = _is_proleptic_leap(candidate_year)

    # Find the first sample whose absolute datetime lies on Feb 29 (or none).
    leap_day_sample_idx: int | None = None
    if is_leap_expected:
        for s in range(n_samples):
            dt = _absolute_datetime(anchor, time_plasim[s])
            if dt.month == 2 and dt.day == 29:
                leap_day_sample_idx = s
                break

    return {
        "file": path.name,
        "anchor_year": anchor_year,
        "n_samples": n_samples,
        "is_leap_expected": is_leap_expected,
        "leap_day_sample_idx": leap_day_sample_idx,
    }


def trace(src: Path, files: Iterable[str]) -> list[dict]:
    rows: list[dict] = []
    for name in files:
        path = src / name
        if not path.is_file():
            raise SystemExit(f"missing file: {path}")
        rows.append(trace_one(path))
    return rows


def _validate_rows(rows: list[dict]) -> int:
    """Return non-zero exit code if any row breaks the +5 / leap rule."""
    rc = 0
    for r in rows:
        n = r["n_samples"]
        is_leap = r["is_leap_expected"]
        if is_leap and n != 1459:
            print(f"FAIL {r['file']}: leap-year file but n_samples={n} (expected 1459)",
                  file=sys.stderr)
            rc = 1
        if (not is_leap) and n != 1455:
            print(f"FAIL {r['file']}: non-leap file but n_samples={n} (expected 1455)",
                  file=sys.stderr)
            rc = 1
        if is_leap and r["leap_day_sample_idx"] is None:
            print(f"FAIL {r['file']}: leap-year file but no Feb-29 sample found",
                  file=sys.stderr)
            rc = 1
    return rc


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sanity-check the +5 file-index → anchor-year offset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--src", required=True, type=Path,
                   help="Directory containing MOST.NNNN.h5 files")
    p.add_argument("--files", required=True, type=str,
                   help="Comma-separated h5 filenames to audit (e.g. 'MOST.0094.h5,MOST.0122.h5')")
    p.add_argument("--csv", type=Path, default=None,
                   help="Optional output CSV path; otherwise CSV is written to stdout")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    files = [s.strip() for s in args.files.split(",") if s.strip()]
    rows = trace(args.src, files)

    out = args.csv.open("w", newline="") if args.csv else sys.stdout
    writer = csv.DictWriter(
        out, fieldnames=["file", "anchor_year", "n_samples", "is_leap_expected", "leap_day_sample_idx"],
    )
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    if args.csv:
        out.close()

    sys.exit(_validate_rows(rows))


if __name__ == "__main__":
    main()
