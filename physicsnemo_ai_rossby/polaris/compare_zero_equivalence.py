#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verdict for ZeRO equivalence.

⚠ REWRITTEN 2026-08-07. The original version of this file asserted that
multi-rank DDP is not reproducible against itself, because ZeRO "reduce-scatters"
and changes summation order. **Both halves were false**, and the first run built
on them was confounded.

What ZeRO-1 actually does: ``ZeroRedundancyOptimizer`` leaves DDP's gradient
all-reduce completely untouched. Each rank then runs the local optimizer over its
own parameter shard and **broadcasts** the updated parameters
(``zero_redundancy_optimizer.py``: ``step()`` -> ``_local_step()`` +
``_sync_params()``). There is no reduce-scatter.

So after the all-reduce every rank holds bit-identical gradients, and AdamW is
elementwise — which rank computes a given update cannot change its value.
**Correct ZeRO-1 is bitwise identical to plain DDP.** Use ``--require-bitwise``.

Measured, and consistent with that: two identical plain-DDP runs agreed to
0.000e+00 on every field. NCCL all-reduce is deterministic at fixed topology, so
the harness has bitwise resolution. A zero floor is the control passing, not a
degenerate test.

⚠ THE ARMS MUST USE THE SAME AdamW KERNEL. ``_wrap_zero`` drops ``fused=True``
because the wrapper rejects it, so a naive comparison runs FUSED against EAGER
AdamW. That alone produces ~1e-5 — which is exactly what the first run measured
and mistook for reduction noise. This script now prints ``optimizer_fused`` for
each arm and warns loudly when they differ.

Inputs are three trajectories from ``equivalence_ddp.py`` at one seed:

    ddp_a   plain DDP, run 1
    ddp_b   plain DDP, run 2   -> control: is the harness reproducible at all?
    zero    DDP + ZeRO-1       -> must equal ddp_a

The legacy noise-floor mode (``--factor``) is retained for a case where the
control genuinely is nonzero, but it is not the right tool here.

PASS = ``ZERO_EQUIVALENCE_OK``. Never relax the bar to make a failure go away
(CLAUDE.md #1/#11) — investigate the kernel and
``max_cross_rank_param_delta`` first.
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
                   help="floor used when the measured noise floor is ~0 "
                        "(legacy noise-floor mode only)")
    p.add_argument("--require-bitwise", action="store_true",
                   help="Demand EXACT equality. Correct in the normal case: "
                        "ZeroRedundancyOptimizer does not change gradient "
                        "reduction (DDP all-reduces, ranks broadcast updated "
                        "params), so gradients are bit-identical on every rank "
                        "and elementwise AdamW cannot depend on which rank runs "
                        "it. Requires the AdamW KERNEL held fixed across arms — "
                        "_wrap_zero drops fused=True, and fused-vs-eager alone "
                        "produces ~1e-5.")
    a = p.parse_args(argv)

    ddp_a, ddp_b, zero = _traj(a.ddp_a), _traj(a.ddp_b), _traj(a.zero)

    floor, floor_f, floor_i = _worst(ddp_a, ddp_b)
    dev, dev_f, dev_i = _worst(ddp_a, zero)
    bound = max(floor * a.factor, a.abs_floor)

    print("=== ZeRO equivalence ===")
    import json as _j
    for nm, pth in (("ddp_a", a.ddp_a), ("ddp_b", a.ddp_b), ("zero", a.zero)):
        r = _j.loads(pth.read_text())
        print(f"  {nm:6s} optimizer={r.get('optimizer_class','?'):24s} "
              f"fused={r.get('optimizer_fused','?')!s:5s} "
              f"zero={r.get('use_zero_optimizer','?')!s:5s} "
              f"lr[0]={r.get('lr_first','?')} "
              f"cross_rank_delta={r.get('max_cross_rank_param_delta','?')}")
    kernels = {_j.loads(p_.read_text()).get("optimizer_fused")
               for p_ in (a.ddp_a, a.ddp_b, a.zero)}
    if len(kernels) > 1:
        print("  ⚠ CONFOUNDED: the arms did NOT use the same AdamW kernel "
              "(fused differs). Any deviation below ~1e-5 is the kernel swap, "
              "not ZeRO. Re-run with training.optimizer.fused=false on all arms.")
    print(f"  steps compared      : {len(ddp_a)}")
    print(f"  NOISE FLOOR (ddp vs ddp) : {floor:.3e}   worst on '{floor_f}' at step {floor_i}")
    print(f"  ZeRO deviation           : {dev:.3e}   worst on '{dev_f}' at step {dev_i}")
    print(f"  bound = max(floor x {a.factor:g}, {a.abs_floor:g}) = {bound:.3e}")
    if floor == 0.0 and not a.require_bitwise:
        print("  NOTE: noise floor is exactly zero — the two DDP runs agreed")
        print("    bitwise. That is EXPECTED (NCCL all-reduce is deterministic at")
        print("    fixed topology), not degenerate. It means an exact bar is")
        print("    available: prefer --require-bitwise over this mode.")
    if not a.require_bitwise:
        ratio = dev / floor if floor > 0 else float("inf")
        print(f"  deviation / floor        : {ratio:.2f}x")

    # A run that silently fell back to plain AdamW reproduces the controls
    # EXACTLY, so under a bitwise bar it is the most certain thing to pass.
    # Refuse it here too — the capture asserts this as well, but a stale or
    # hand-made JSON must not slip past the comparator.
    zrec = _j.loads(a.zero.read_text())
    if zrec.get("use_zero_optimizer") and zrec.get("optimizer_class") not in (
            "ZeroRedundancyOptimizer", None):
        print(f"ERROR ZERO_NOT_ACTUALLY_USED — zero arm's optimizer_class is "
              f"{zrec.get('optimizer_class')!r}, not ZeroRedundancyOptimizer.")
        return 3
    for nm, pth in (("ddp_a", a.ddp_a), ("ddp_b", a.ddp_b), ("zero", a.zero)):
        d = _j.loads(pth.read_text()).get("max_cross_rank_param_delta")
        if d:
            print(f"ERROR REPLICA_DIVERGENCE in {nm}: {d:.3e} — ranks held "
                  f"different weights. Fix that before reading any deviation.")
            return 4

    if a.require_bitwise:
        # A zero floor is the EXPECTED control result here, not a red flag: it
        # says the harness is reproducible, which is what makes an exact bar
        # meaningful. (The earlier "zero floor means degenerate" warning was
        # written under the false belief that DDP is nondeterministic.)
        if floor != 0.0:
            print(f"  ⚠ control: two identical DDP runs differ by {floor:.3e}. "
                  "The harness is not reproducible, so an exact bar is not "
                  "meaningful — fix that before reading the ZeRO verdict.")
        if dev == 0.0:
            print("ZERO_EQUIVALENCE_OK — bitwise identical to plain DDP")
            return 0
        print(f"ERROR ZERO_EQUIVALENCE_FAILED — deviation {dev:.3e} is NOT zero.")
        print("  Correct ZeRO-1 must be bitwise identical: gradients are equal on")
        print("  every rank after DDP's all-reduce and AdamW is elementwise.")
        print("  FIRST check the arms used the SAME AdamW kernel (fused vs eager")
        print("  alone gives ~1e-5) — see optimizer_fused in the JSONs. Then check")
        print("  max_cross_rank_param_delta for replica divergence.")
        return 1

    if floor == 0.0:
        print("ERROR ZERO_FLOOR_IN_NOISE_MODE — the control has no spread, so "
              "this mode's bound is just --abs-floor and the ddp_b run carries "
              "no information. Use --require-bitwise.")
        return 5
    if dev <= bound:
        print(f"ZERO_EQUIVALENCE_OK — ZeRO reproduces DDP to within the "
              f"reduction's own run-to-run spread")
        return 0
    print(f"ERROR ZERO_EQUIVALENCE_FAILED — deviation {dev:.3e} exceeds {bound:.3e}")
    print("  This is signal, not float noise. Find the cause; do NOT raise --factor.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
