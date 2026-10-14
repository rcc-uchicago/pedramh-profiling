#!/usr/bin/env python3
"""port_golden_npy_diff.py — size of a non-bitwise difference between two golden prediction sets.

For each ``<tag>_K<K>.npy`` present in both REF and NEW: bitwise equal or not (raw bytes, so NaN
payloads and -0.0 vs 0.0 count), and if not, the first differing lead, the non-finite counts of
each side, and max abs error and where (lead, channel name, lat/lon index) over points finite in
both. A finite-mask mismatch is printed as ``ERROR NPY_DIFF <tag> nonfinite_mask_mismatch`` and
fails the run (rc 1). Relative error is information only: over points finite in both with a
nonzero reference, never a gate (tolerance ruling 7697688 §3.5; Tier 2 sizes are judged in z by
port_golden_z_diff.py). Reads lead by lead through mmap so a 1.5 GB array never sits in memory twice.
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
    n_equal = n_all = n_mask_bad = 0
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
        nonfin_ref = nonfin_new = 0
        mask_bad = None
        for k in range(r.shape[0]):
            rk_raw, nk_raw = np.asarray(r[k]), np.asarray(n[k])
            fin_r, fin_n = np.isfinite(rk_raw), np.isfinite(nk_raw)
            nonfin_ref += int(fin_r.size - fin_r.sum())
            nonfin_new += int(fin_n.size - fin_n.sum())
            if rk_raw.dtype == nk_raw.dtype and rk_raw.tobytes() == nk_raw.tobytes():
                continue
            if first is None:
                first = k + 1
            if mask_bad is None and not np.array_equal(fin_r, fin_n):
                mask_bad = k + 1
            both = fin_r & fin_n
            if not both.any():
                continue
            rk, nk = rk_raw.astype(np.float64), nk_raw.astype(np.float64)
            d = np.where(both, np.abs(nk - rk), 0.0)
            i = int(np.argmax(d))
            if d.flat[i] > best[0]:
                best = (float(d.flat[i]), (k,) + np.unravel_index(i, d.shape))
            relmask = both & (rk != 0.0)
            if relmask.any():
                rel = np.where(relmask, d / np.where(relmask, np.abs(rk), 1.0), 0.0)
                j = int(np.argmax(rel))
                if rel.flat[j] > best_rel[0]:
                    best_rel = (float(rel.flat[j]), (k,) + np.unravel_index(j, rel.shape))
                if k == 0:
                    rel1 = float(rel.max())
        if first is None:
            n_equal += 1
            print("NPY_DIFF %s %s bitwise=True nonfinite=%d" % (a.label, c["tag"], nonfin_ref))
            continue
        if mask_bad is not None:
            n_mask_bad += 1
            print("ERROR NPY_DIFF %s %s nonfinite_mask_mismatch first_lead=%d nonfinite_ref=%d nonfinite_new=%d"
                  % (a.label, c["tag"], mask_bad, nonfin_ref, nonfin_new))
        where = lambda b: ("lead=%d channel=%s lat_idx=%d lon_idx=%d" % (b[0] + 1, names[b[1]], b[2], b[3])
                           if b is not None else "lead=- (no point finite in both)")
        print("NPY_DIFF %s %s bitwise=False first_bad_lead=%d nonfinite_ref=%d nonfinite_new=%d max_abs=%.3e at %s "
              "info_max_rel=%.3e at %s; info_lead1_max_rel=%.3e"
              % (a.label, c["tag"], first, nonfin_ref, nonfin_new, best[0], where(best[1]),
                 best_rel[0], where(best_rel[1]), rel1))
    print("NPY_DIFF_SUMMARY %s bitwise=%d/%d nonfinite_mask_mismatch=%d" % (a.label, n_equal, n_all, n_mask_bad))
    return 1 if n_mask_bad else 0


if __name__ == "__main__":
    sys.exit(main())
