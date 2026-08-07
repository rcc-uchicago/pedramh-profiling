#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Noise-floor verdict for ZeRO equivalence.

``compare_baselines.py`` asks "is the candidate within TOLERANCE of the
baseline?" — the right question when the baseline is reproducible. Multi-rank
DDP is **not** reproducible against itself: gradient reduction order varies run
to run, and float addition is not associative. Against that baseline, a fixed
tolerance is either so tight it fails correct code or so loose it passes
anything.

So this asks a different question:

    Is ZeRO's deviation from DDP no larger than DDP's deviation from ITSELF?

Inputs are three trajectories captured by ``equivalence_ddp.py`` at one seed:

    ddp_a   plain DDP, run 1
    ddp_b   plain DDP, run 2   -> |ddp_a - ddp_b| is the NOISE FLOOR
    zero    DDP + ZeRO-1       -> |ddp_a - zero|  is the deviation under test

Verdict: PASS when the ZeRO deviation is within ``--factor`` x the noise floor
(default 3), or below ``--abs-floor`` (default 1e-6, so an immeasurably small
floor cannot make the test impossibly strict).

Reading the result honestly:

* **deviation ~= floor** — ZeRO reproduces DDP to the precision DDP itself
  offers. That is the strongest statement available; a bitwise claim is not.
* **deviation >> floor** — real signal. Not float noise. Do not adopt.
* **floor ~= 0** — suspicious, not reassuring: two DDP runs agreeing exactly
  usually means the reduction was degenerate (e.g. the ranks saw identical
  batches), so the test never exercised what it claims to.

PASS = ``ZERO_EQUIVALENCE_OK``. Over the bound → ``ERROR ZERO_EQUIVALENCE_FAILED``.
Never widen ``--factor`` to make a failure go away (CLAUDE.md #1/#11).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FIELDS = ("loss", "surface", "upper_air", "diagnostic", "grad_norm")


def _rel(a: float, b: float) -> float:
    return abs(a - b) / (abs(b) + 1e-8)


def _traj(path: Path) -> list[dict]:
    rec = json.loads(path.read_text())
    if rec.get("world_size", 1) < 2:
        raise SystemExit(f"ERROR NOT_MULTIRANK: {path} has world_size="
                         f"{rec.get('world_size')} — ZeRO is a no-op there.")
    return rec["loss_trajectory"]


def _worst(x: list[dict], y: list[dict]) -> tuple[float, str, int]:
    """Max relative difference over every field and step, and where."""
    if len(x) != len(y):
        raise SystemExit(f"ERROR STEP_COUNT_MISMATCH: {len(x)} vs {len(y)}")
    worst, where, at = 0.0, "", -1
    for i, (a, b) in enumerate(zip(x, y)):
        for f in FIELDS:
            if f in a and f in b:
                r = _rel(a[f], b[f])
                if r > worst:
                    worst, where, at = r, f, i
    return worst, where, at


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ddp_a", type=Path)
    p.add_argument("ddp_b", type=Path, help="second plain-DDP run, same seed")
    p.add_argument("zero", type=Path)
    p.add_argument("--factor", type=float, default=3.0,
                   help="ZeRO may deviate up to this multiple of the noise floor")
    p.add_argument("--abs-floor", type=float, default=1e-6,
                   help="floor used when the measured noise floor is ~0")
    a = p.parse_args(argv)

    ddp_a, ddp_b, zero = _traj(a.ddp_a), _traj(a.ddp_b), _traj(a.zero)

    floor, floor_f, floor_i = _worst(ddp_a, ddp_b)
    dev, dev_f, dev_i = _worst(ddp_a, zero)
    bound = max(floor * a.factor, a.abs_floor)

    print("=== ZeRO equivalence — noise-floor comparison ===")
    print(f"  steps compared      : {len(ddp_a)}")
    print(f"  NOISE FLOOR (ddp vs ddp) : {floor:.3e}   worst on '{floor_f}' at step {floor_i}")
    print(f"  ZeRO deviation           : {dev:.3e}   worst on '{dev_f}' at step {dev_i}")
    print(f"  bound = max(floor x {a.factor:g}, {a.abs_floor:g}) = {bound:.3e}")
    if floor == 0.0:
        print("  ⚠ NOISE FLOOR IS EXACTLY ZERO — two DDP runs agreed bitwise.")
        print("    That usually means the reduction was degenerate (identical")
        print("    batches per rank), i.e. the test did not exercise what it claims.")
        print("    Check that the datapipe was built with distributed=True.")
    ratio = dev / floor if floor > 0 else float("inf")
    print(f"  deviation / floor        : {ratio:.2f}x")

    if dev <= bound:
        print(f"ZERO_EQUIVALENCE_OK — ZeRO reproduces DDP to within the "
              f"reduction's own run-to-run spread")
        return 0
    print(f"ERROR ZERO_EQUIVALENCE_FAILED — deviation {dev:.3e} exceeds {bound:.3e}")
    print("  This is signal, not float noise. Find the cause; do NOT raise --factor.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
