#!/usr/bin/env python3
"""Apply the pre-registered LR selection rule to the 3-arm sweep.

Exit 0 and print ``WINNER_LR=<lr> ...`` once every arm has reached epoch 3.
Exit non-zero (with a reason) while the sweep is still incomplete, so the
caller can poll the LOG FILES rather than qstat.

The rule is fixed before arms 2 and 3 exist -- see
``submit_production_when_lr_picked.sh`` for the rationale. Summary:

1. Disqualify an arm with a non-finite loss, or whose gradient norm RISES from
   epoch 2 to epoch 3. Raw loss is not comparable across LRs (a higher LR always
   descends faster early), so stability is filtered first and loss ranks second.
2. Among survivors, lowest epoch-3 validation loss wins.
3. Within 5%, prefer the LOWER LR: under warm restarts this value is the peak of
   every cycle for ~46 h, so ties break conservative.
"""
from __future__ import annotations

import math
import re
import sys
import os

ARMS = [("lr_4e4", 4.0e-4, "4.0E-4"), ("lr_1e3", 1.0e-3, "1.0E-3"), ("lr_2e3", 2.0e-3, "2.0E-3")]
NEEDED_EPOCH = 3
TIE = 0.05

_PAT = {
    "train": re.compile(r"training loss:\s+([\d.eE+-]+)"),
    "valid": re.compile(r"validation loss:\s+([\d.eE+-]+)"),
    "gnorm": re.compile(r"gradient norm:\s+([\d.eE+-]+)"),
}


def read_arm(runs: str, tag: str) -> dict[int, dict[str, float]]:
    path = os.path.join(runs, tag + ".log")
    if not os.path.exists(path):
        return {}
    out: dict[int, dict[str, float]] = {}
    cur, grab = None, 0
    with open(path, errors="replace") as fh:
        for line in fh:
            m = re.search(r"INFO:root:Epoch (\d+) summary", line)
            if m:
                cur, grab = int(m.group(1)), 24
                out[cur] = {}
                continue
            if grab and cur is not None:
                grab -= 1
                for key, pat in _PAT.items():
                    mm = pat.search(line)
                    if mm:
                        out[cur][key] = float(mm.group(1))
    return out


def main(argv: list[str]) -> int:
    runs = argv[1] if len(argv) > 1 else "."
    data = {tag: read_arm(runs, tag) for tag, _, _ in ARMS}

    missing = [t for t, _, _ in ARMS if NEEDED_EPOCH not in data[t]]
    if missing:
        print("waiting on: " + ",".join(missing))
        return 1

    survivors, rejected = [], []
    for tag, lr, lr_str in ARMS:
        d = data[tag]
        losses = [d[e].get("valid", float("nan")) for e in sorted(d)]
        losses += [d[e].get("train", float("nan")) for e in sorted(d)]
        if any(math.isnan(x) or math.isinf(x) for x in losses):
            rejected.append(f"{tag}(non-finite)")
            continue
        # Rule 1: gradient norm must not rise from epoch 2 to 3.
        g2, g3 = d[2].get("gnorm"), d[3].get("gnorm")
        if g2 is None or g3 is None:
            rejected.append(f"{tag}(no gnorm)")
            continue
        if g3 > g2:
            rejected.append(f"{tag}(gnorm rose {g2:.4f}->{g3:.4f})")
            continue
        survivors.append((d[3]["valid"], lr, lr_str, tag))

    if not survivors:
        print("NO_WINNER rejected=" + ",".join(rejected))
        return 4

    survivors.sort()  # by epoch-3 validation loss
    best_v, best_lr, best_str, best_tag = survivors[0]
    # Rule 3: within TIE, prefer the lower LR.
    close = [s for s in survivors if s[0] <= best_v * (1 + TIE)]
    if len(close) > 1:
        close.sort(key=lambda s: s[1])
        best_v, best_lr, best_str, best_tag = close[0]

    detail = " ".join(f"{t}:va3={data[t][3]['valid']:.5f},gn3={data[t][3].get('gnorm', float('nan')):.5f}"
                      for t, _, _ in ARMS)
    print(f"WINNER_LR={best_str} arm={best_tag} valid3={best_v:.5f} | {detail}"
          + (" | rejected=" + ",".join(rejected) if rejected else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
