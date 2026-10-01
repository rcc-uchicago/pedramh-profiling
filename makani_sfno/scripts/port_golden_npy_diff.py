#!/usr/bin/env python3
"""port_golden_npy_diff.py — size of a non-bitwise difference between two golden prediction sets.

For each ``<tag>_K<K>.npy`` present in both REF and NEW: bitwise equal or not, and if not, the
first differing lead, max abs and max rel error and where (lead, channel name, lat/lon index),
and max rel error at lead 1. Reports, never passes or fails a tolerance (there is none here).
Reads lead by lead through mmap so a 1.5 GB array never sits in memory twice.
"""
import argparse
import json
import os
import sys

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("ref_dir")
    p.add_argument("new_dir")
    p.add_argument("--list", required=True, help="golden_checkpoints*.json, for channel names")
    p.add_argument("--label", default="npy")
    a = p.parse_args()

    spec = json.load(open(a.list))
    K = spec["ic"]["K"]
    n_equal = n_all = 0
    for c in spec["checkpoints"]:
        f = "%s_K%d.npy" % (c["tag"], K)
        rp, np_ = os.path.join(a.ref_dir, f), os.path.join(a.new_dir, f)
        if not (os.path.exists(rp) and os.path.exists(np_)):
            print("NPY_DIFF %s %s missing ref=%s new=%s" % (a.label, c["tag"], os.path.exists(rp), os.path.exists(np_)))
            continue
        n_all += 1
        names = json.load(open(os.path.join(spec["root"], c["run"], "config.json")))["channel_names"]
        r, n = np.load(rp, mmap_mode="r"), np.load(np_, mmap_mode="r")
        if r.shape != n.shape:
            print("NPY_DIFF %s %s shape %s != %s" % (a.label, c["tag"], r.shape, n.shape))
            continue
        first, best, best_rel, rel1 = None, (0.0, None), (0.0, None), 0.0
        for k in range(r.shape[0]):
            rk, nk = np.asarray(r[k], dtype=np.float64), np.asarray(n[k], dtype=np.float64)
            if np.array_equal(rk, nk):
                continue
            if first is None:
                first = k + 1
            d = np.abs(nk - rk)
            rel = d / np.maximum(np.abs(rk), 1e-30)
            i = int(np.argmax(d))
            if d.flat[i] > best[0]:
                best = (float(d.flat[i]), (k,) + np.unravel_index(i, d.shape))
            j = int(np.argmax(rel))
            if rel.flat[j] > best_rel[0]:
                best_rel = (float(rel.flat[j]), (k,) + np.unravel_index(j, rel.shape))
            if k == 0:
                rel1 = float(rel.max())
        if first is None:
            n_equal += 1
            print("NPY_DIFF %s %s bitwise=True" % (a.label, c["tag"]))
            continue
        (ka, ca, ia, ja), (kr, cr, ir, jr) = best[1], best_rel[1]
        print("NPY_DIFF %s %s bitwise=False first_bad_lead=%d max_abs=%.3e at lead=%d channel=%s lat_idx=%d lon_idx=%d "
              "max_rel=%.3e at lead=%d channel=%s; lead1_max_rel=%.3e"
              % (a.label, c["tag"], first, best[0], ka + 1, names[ca], ia, ja, best_rel[0], kr + 1, names[cr], rel1))
    print("NPY_DIFF_SUMMARY %s bitwise=%d/%d" % (a.label, n_equal, n_all))
    return 0


if __name__ == "__main__":
    sys.exit(main())
