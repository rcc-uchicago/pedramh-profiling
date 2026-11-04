"""emitall receipt for the map figures: figures/summary_index.json (stdlib, python3.6+).

Re-keys summary_table.csv by path, so a claims spec can point at /<figure>/<variable>/<model>/<lead_h or window>/
<value|ci_lo|ci_hi|sig_area_pct>. No derivation: every value is the CSV's own field, as a float (None if blank).
Specs declare the index `newer_than` the CSV, so re-run this after every plot_maps.py run.
Usage: build_summary_index.py [FIGURE_DIR]      PASS = SUMMARY_INDEX_OK
"""
import csv
import json
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
FIELDS = ("value", "ci_lo", "ci_hi", "sig_area_pct")


def num(x):
    return float(x) if x != "" else None


def build(path):
    out = {}
    for r in csv.DictReader(open(path)):
        node = out.setdefault(r["figure"], {}).setdefault(r["variable"], {}).setdefault(r["model"], {})
        node[r["lead_h"] or r["window"]] = {k: num(r[k]) for k in FIELDS}
    return out


def main():
    idx = build(os.path.join(D, "summary_table.csv"))
    json.dump(idx, open(os.path.join(D, "summary_index.json"), "w"), indent=1)
    n = sum(len(t) for f in idx.values() for v in f.values() for t in v.values())
    print("SUMMARY_INDEX_OK figures=%d rows=%d -> %s" % (len(idx), n, os.path.join(D, "summary_index.json")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
