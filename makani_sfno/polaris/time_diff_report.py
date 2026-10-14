"""Publish the `time_diff_stds.npy` table BEFORE anything enables it (handoff §3 step 1).

Reads the pack's `stats/global_stds.npy` (sigma_c) and `stats/time_diff_stds.npy`
(delta_c, full 30-year train split, from `convert_e3sm_to_makani_alldata.py
--time-diff-only`) and reports, for the full contract AND the post-port-F subset:

* r_c = delta_c / sigma_c for every channel, sorted -- the spread IS the claim;
* the weight makani would realise, sigma_c / clamp(delta_c, 1e-4) (loss.py:152-154,
  the clamp in PHYSICAL units), and its max/min span;
* every channel whose delta_c sits at or under the 1e-4 clamp (PRECT is expected
  there: the clamp would weight it ~0.0008x instead of ~1x);
* Z3_l17 explicitly;
* Spearman vs job 7646192's 240-sample r_c, if its CSV is given -- the probe
  claimed the ORDERING is robust; this is the test of that claim.

Inspect only. It changes no config and writes only into --out-dir.
    python time_diff_report.py --pack <root> --out-dir <dir> [--probe-csv <csv>]
      -> TIME_DIFF_REPORT_OK ...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

CLAMP = 1e-4  # makani/utils/loss.py:154, physical units
DROPPED_F = ("SOILWATER_10CM", "TSOI_10CM")


def _rank(x):
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    return r


def spearman(a, b) -> float:
    return float(np.corrcoef(_rank(np.asarray(a)), _rank(np.asarray(b)))[0, 1])


def build_table(names, sigma, delta):
    sigma, delta = np.asarray(sigma, float), np.asarray(delta, float)
    r = delta / sigma
    w = sigma / np.maximum(delta, CLAMP)
    rows = [{"channel": n, "sigma_c": float(s), "delta_c": float(d), "r_c": float(rc),
             "weight_realised": float(wi), "weight_1_over_r": float(1.0 / rc) if rc > 0 else float("inf"),
             "clamped": bool(d <= CLAMP)}
            for n, s, d, rc, wi in zip(names, sigma, delta, r, w)]
    return sorted(rows, key=lambda x: x["r_c"])


def summarize(rows, label):
    w = np.array([x["weight_realised"] for x in rows])
    r = np.array([x["r_c"] for x in rows])
    z = next((x for x in rows if x["channel"] == "Z3_l17"), None)
    return {
        "label": label, "n_channels": len(rows),
        "r_min": float(r.min()), "r_max": float(r.max()), "r_span": float(r.max() / r.min()),
        "weight_min": float(w.min()), "weight_max": float(w.max()),
        "weight_span": float(w.max() / w.min()),
        "weight_max_channel": rows[int(np.argmax(w))]["channel"],
        "weight_share_of_top_channel": float(w.max() / w.sum()),
        "clamped_channels": [x["channel"] for x in rows if x["clamped"]],
        "slowest_10": [x["channel"] for x in rows[:10]],
        "Z3_l17": z,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pack", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--probe-csv", type=Path, default=None)
    a = p.parse_args()

    names = json.loads((a.pack / "metadata" / "data.json").read_text())["coords"]["channel"]
    sigma = np.load(a.pack / "stats" / "global_stds.npy").ravel()
    delta = np.load(a.pack / "stats" / "time_diff_stds.npy").ravel()
    if not (len(names) == sigma.size == delta.size):
        print(f"ERROR CHANNEL_ORDER_MISMATCH: names {len(names)} sigma {sigma.size} "
              f"delta {delta.size} -- makani indexes all of them by out_channels")
        return 2
    if not np.all(np.isfinite(delta)) or np.any(delta < 0):
        print("ERROR TIME_DIFF_NONFINITE")
        return 2

    full = build_table(names, sigma, delta)
    keep = [i for i, n in enumerate(names) if n not in DROPPED_F]
    sub = build_table([names[i] for i in keep], sigma[keep], delta[keep])
    out = {"full_101": summarize(full, "current contract"),
           "post_F_99": summarize(sub, "port F subset (no soil)")}

    if a.probe_csv and a.probe_csv.exists():
        with a.probe_csv.open() as fh:
            probe = {row["channel"]: float(row["r"]) for row in csv.DictReader(fh)}
        common = [x for x in full if x["channel"] in probe]
        out["probe_7646192"] = {
            "spearman_r_full_vs_probe": spearman([x["r_c"] for x in common],
                                                 [probe[x["channel"]] for x in common]),
            "n": len(common),
            "max_rel_diff_r": float(max(abs(x["r_c"] - probe[x["channel"]]) / probe[x["channel"]]
                                        for x in common)),
        }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    with (a.out_dir / "time_diff_table.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(full[0]))
        wr.writeheader()
        wr.writerows(full)
    (a.out_dir / "time_diff_summary.json").write_text(json.dumps(out, indent=2) + "\n")

    for k in ("full_101", "post_F_99"):
        s = out[k]
        print(f"{k}: r {s['r_min']:.3g}..{s['r_max']:.3g} (span {s['r_span']:.3g}); "
              f"realised weight span {s['weight_span']:.3g}, max {s['weight_max']:.4g} "
              f"on {s['weight_max_channel']} ({100 * s['weight_share_of_top_channel']:.1f}% "
              f"of the summed weight); clamped {s['clamped_channels']}")
    z = out["full_101"]["Z3_l17"]
    if z:
        print(f"Z3_l17: sigma {z['sigma_c']:.4g} delta {z['delta_c']:.4g} r {z['r_c']:.3g} "
              f"weight {z['weight_realised']:.4g}")
    if "probe_7646192" in out:
        pr = out["probe_7646192"]
        print(f"vs probe 7646192: Spearman {pr['spearman_r_full_vs_probe']:.3f} over "
              f"{pr['n']}; max rel diff in r {pr['max_rel_diff_r']:.2f}")
    print(f"TIME_DIFF_REPORT_OK out={a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
