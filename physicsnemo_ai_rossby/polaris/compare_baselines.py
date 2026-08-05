#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply DESIGN §4.1's equivalence metric to two `equivalence.py` captures.

    max |a - b| / (|b| + 1e-8)      reporting WHERE the max occurs

`b` is the baseline (the pre-change run), so the relative error is measured
against the reference — not the candidate. Tolerance per §4.1: eager-vs-eager
fp32 ≤ 1e-5, bf16 or compiled paths ≤ 1e-2. **State the number used**, and if it
fails, find the cause — never loosen the tolerance to pass (CLAUDE.md #1/#11).

Stdlib only, so this runs anywhere including a login node.

Usage::

    python compare_baselines.py baselines/<model>/eager.json \\
                                baselines/<model>/compiled.json --tolerance 1e-2

PASS = ``EQUIVALENCE_OK`` (exit 0). Over tolerance → ``ERROR EQUIVALENCE_FAILED``
(exit 1). Captures that are not comparable at all → exit 2, because a mismatched
seed/config makes the number meaningless rather than merely large.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Differences in these invalidate the comparison outright: they mean the two runs
# were not the same experiment, so any relative error is uninterpretable.
MUST_MATCH = ("seed", "steps", "world_size", "n_params", "batch_size",
              "amp_dtype", "config_yaml_sha256")


def _rel(a: float, b: float) -> float:
    """§4.1's metric, relative to the BASELINE value."""
    return abs(a - b) / (abs(b) + 1e-8)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--tolerance", type=float, required=True,
                   help="§4.1: 1e-5 eager fp32, 1e-2 bf16/compiled. State it explicitly.")
    a = p.parse_args(argv)

    base = json.loads(a.baseline.read_text())
    cand = json.loads(a.candidate.read_text())

    mismatched = [
        (k, base.get(k), cand.get(k)) for k in MUST_MATCH if base.get(k) != cand.get(k)
    ]
    if mismatched:
        print("ERROR EQUIVALENCE_NOT_COMPARABLE — these runs are not the same "
              "experiment, so a relative error would be meaningless:")
        for k, x, y in mismatched:
            print(f"    {k}: baseline={x!r} candidate={y!r}")
        return 2

    worst = (-1.0, "")
    n_compared = 0

    # --- per-step loss trajectory ---
    bt, ct = base["loss_trajectory"], cand["loss_trajectory"]
    if len(bt) != len(ct):
        print(f"ERROR EQUIVALENCE_NOT_COMPARABLE — trajectory lengths "
              f"{len(bt)} vs {len(ct)}")
        return 2
    for i, (rb, rc) in enumerate(zip(bt, ct)):
        for key in sorted(set(rb) & set(rc)):
            r = _rel(rc[key], rb[key])
            n_compared += 1
            if r > worst[0]:
                worst = (r, f"loss_trajectory[step {i}].{key} "
                            f"(baseline {rb[key]:.6g}, candidate {rc[key]:.6g})")

    # --- forward-output summary stats ---
    bo, co = base.get("forward_output_stats", {}), cand.get("forward_output_stats", {})
    for grp in sorted(set(bo) & set(co)):
        for stat in ("mean", "std", "min", "max"):
            if stat in bo[grp] and stat in co[grp]:
                r = _rel(co[grp][stat], bo[grp][stat])
                n_compared += 1
                if r > worst[0]:
                    worst = (r, f"forward_output_stats.{grp}.{stat} "
                                f"(baseline {bo[grp][stat]:.6g}, "
                                f"candidate {co[grp][stat]:.6g})")
        if bo[grp].get("shape") != co[grp].get("shape"):
            print(f"ERROR EQUIVALENCE_SHAPE_DRIFT {grp}: "
                  f"{bo[grp].get('shape')} vs {co[grp].get('shape')}")
            return 1

    # Output hygiene (DESIGN §7): the max relative error and WHERE — never tensors.
    print(f"=== equivalence: {a.candidate.name} vs {a.baseline.name} (baseline) ===")
    print(f"  change      : {base.get('torch_compile_mode') or 'eager'} -> "
          f"{cand.get('torch_compile_mode') or 'eager'}")
    print(f"  quantities  : {n_compared}")
    print(f"  tolerance   : {a.tolerance:g}")
    print(f"  max rel err : {worst[0]:.3e}")
    print(f"  at          : {worst[1]}")
    if worst[0] > a.tolerance:
        print(f"ERROR EQUIVALENCE_FAILED max rel err {worst[0]:.3e} > {a.tolerance:g} — "
              f"the change alters what the model computes. Find the cause; do NOT "
              f"loosen the tolerance (CLAUDE.md #1/#11).")
        return 1
    print(f"EQUIVALENCE_OK {worst[0]:.3e} <= {a.tolerance:g} ({n_compared} quantities)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
