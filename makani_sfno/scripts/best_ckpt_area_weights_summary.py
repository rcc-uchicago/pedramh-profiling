#!/usr/bin/env python3
"""best_ckpt_area_weights_summary.py — stdlib only. Ranks epochs from best_ckpt_area_weights.py on their common ICs.

Reads ``epoch_*.json`` from ``--dir`` and the run's ``out.log`` (makani's logged per-epoch "validation loss").
Compares every epoch on the ICs all epochs share (paired), under naive and band-area weights, and reports each
epoch's difference from the logged-best epoch with a paired standard error.

Sanity check: the naive ranking should reproduce the logged ranking of the best epoch. If it does not, the
measurement does not stand in for makani's validation and the band result is not interpretable.

Token: ``BEST_CKPT_CHECK n_common=… logged_best=e… naive_best=e… band_best=e… naive_matches_logged=yes|no
band_same_as_logged=yes|no``.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def logged_losses(out_log: Path) -> dict[int, float]:
    ep, v = None, {}
    for line in out_log.read_text(errors="ignore").splitlines():
        m = re.search(r"Epoch (\d+) summary", line)
        if m:
            ep = int(m.group(1))
        m = re.search(r"validation loss: ([0-9.eE+-]+)", line)
        if m and ep is not None:
            v[ep] = float(m.group(1))
    return v


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, type=Path)
    p.add_argument("--out-log", required=True, type=Path)
    args = p.parse_args()

    recs = {}
    for f in sorted(args.dir.glob("epoch_*.json")):
        d = json.loads(f.read_text())
        recs[d["epoch"]] = {ic: (n, b) for ic, n, b in zip(d["ics"], d["naive"], d["band"])}
    if len(recs) < 2:
        print(f"ERROR TOO_FEW_EPOCHS found={sorted(recs)}")
        return 2
    common = sorted(set.intersection(*(set(r) for r in recs.values())))
    if len(common) < 30:
        print(f"ERROR TOO_FEW_COMMON_ICS n_common={len(common)}")
        return 2

    logged = logged_losses(args.out_log)
    epochs = sorted(recs)
    logged_best = min(epochs, key=lambda e: logged.get(e, math.inf))
    mean = {(e, k): sum(recs[e][ic][k] for ic in common) / len(common) for e in epochs for k in (0, 1)}
    naive_best = min(epochs, key=lambda e: mean[(e, 0)])
    band_best = min(epochs, key=lambda e: mean[(e, 1)])

    def paired(e, k):
        d = [recs[e][ic][k] - recs[logged_best][ic][k] for ic in common]
        m = sum(d) / len(d)
        sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
        base = mean[(logged_best, k)]
        return 100 * m / base, 100 * 2 * sd / math.sqrt(len(d)) / base

    print(f"{'epoch':>6} {'logged':>11} {'naive':>13} {'band':>13} {'naive vs best %':>18} {'band vs best %':>18}")
    for e in sorted(epochs, key=lambda e: mean[(e, 1)]):
        dn, sn = paired(e, 0)
        db, sb = paired(e, 1)
        print(f"e{e:<5d} {logged.get(e, float('nan')):11.7f} {mean[(e, 0)]:13.7e} {mean[(e, 1)]:13.7e} "
              f"{dn:+9.3f} ±{sn:6.3f} {db:+9.3f} ±{sb:6.3f}")
    print(f"BEST_CKPT_CHECK n_common={len(common)} logged_best=e{logged_best} naive_best=e{naive_best} "
          f"band_best=e{band_best} naive_matches_logged={'yes' if naive_best == logged_best else 'no'} "
          f"band_same_as_logged={'yes' if band_best == logged_best else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
