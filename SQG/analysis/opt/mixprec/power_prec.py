"""power_prec.py -- what the M1 rule of analyze_prec.py can see (PREDICTIONS_prec_outcome.md, "Power"). numpy only.

Six ieee arms (seeds 1, 2 x 3) and four candidate arms (seeds 1, 1, 2, 2), iid normal realizations of sd 1, the
candidates shifted by delta sd. Per delta: the share of runs whose M1 verdict is WORSE (p < 0.05, Delta > 0), BETTER,
PASS_P (|Delta| <= sigma_ref) or INCONCLUSIVE. sigma_ref is estimated from the six, as in the job.

  python3 SQG/analysis/opt/mixprec/power_prec.py [--n 2000]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_prec as AP                                            # noqa: E402

SEEDS_REF, SEEDS_CAND = [1, 2, 1, 2, 1, 2], [1, 1, 2, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    sp = AP.splits(SEEDS_REF, SEEDS_CAND)
    masks = np.zeros((len(sp), 10), dtype=bool)
    for i, idx in enumerate(sp):
        masks[i, list(idx)] = True
    rng = np.random.RandomState(20261120)
    print(f"splits {len(sp)}")
    for delta in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        counts = {"WORSE": 0, "BETTER": 0, "PASS_P": 0, "INCONCLUSIVE": 0}
        for _ in range(a.n):
            v = rng.randn(10)
            v[6:] += delta
            d_all = np.abs((masks * v).sum(1) / 4 - (~masks * v).sum(1) / 6)
            obs = abs(v[6:].mean() - v[:6].mean())
            p = float(np.mean(d_all >= obs - 1e-12))
            dl, sig = v[6:].mean() - v[:6].mean(), v[:6].std(ddof=1)
            k = ("WORSE" if dl > 0 else "BETTER") if p < AP.P_LEVEL else ("PASS_P" if abs(dl) <= sig else
                                                                          "INCONCLUSIVE")
            counts[k] += 1
        print(f"PREC_POWER delta={delta:.1f}sd " + " ".join(f"{k}={v / a.n:.3f}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
