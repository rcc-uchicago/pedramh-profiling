#!/usr/bin/env python3
"""truth_annual_global_means.py — area-weighted annual global means of the truth pack.

The drift census (prereg docs/2026-10-09_pipeline_comparison_prereg.md §4.4) expresses a
channel's drift in units of truth's detrended interannual SD of the annual global mean,
from 2015-2044. That needs one number per (year, channel), not maps, so this reads the
pack's year files with h5py instead of the per-sample eval loader (which needs ~12 min a
year). Every STRIDE-th 6-h sample is used; STRIDE=5 (30 h) cycles through all four
synoptic hours, so the diurnal cycle is not aliased.

Because it reads the raw layout (``fields_state`` + ``fields_diagnostic``, channel order
from metadata/data.json) it is gated: for every year that also has a loader-built truth
map (``--check-truth-dir``/truth_<Y>.npz from build_true_climatology.py), the global mean
of that map must match this script's value within --tol x the channel's interannual SD.

Output npz: years, channel_names, annual_global_mean (Y, C), n_samples (Y,).
PASS: TRUTH_GM_OK (and TRUTH_GM_EQUIV_OK when a check is made).
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def _weights(lat):
    w = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
    return w / w.sum()


def year_mean(args):
    path, stride, w = args
    import h5py
    with h5py.File(path, "r") as f:
        st, dg = f["fields_state"], f["fields_diagnostic"]
        n = st.shape[0]
        idx = np.arange(0, n, stride)
        acc = None
        for t in idx:
            x = np.concatenate([st[t], dg[t]], axis=0).astype(np.float64)   # (C, H, W)
            gm = x.mean(axis=-1) @ w                                          # (C,)
            acc = gm if acc is None else acc + gm
    return path, acc / len(idx), len(idx)


def detrended_sd(years, series):
    """Per-channel SD of residuals from a linear fit in year (ddof=2)."""
    y = np.asarray(years, dtype=np.float64)
    A = np.vstack([y - y.mean(), np.ones_like(y)]).T
    coef, *_ = np.linalg.lstsq(A, series, rcond=None)
    res = series - A @ coef
    return np.sqrt((res ** 2).sum(axis=0) / max(len(y) - 2, 1))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pack", required=True, type=Path)
    p.add_argument("--years", required=True, type=int, nargs="+")
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--check-truth-dir", type=Path, default=None)
    p.add_argument("--sd-years", type=int, nargs=2, default=(2015, 2044))
    p.add_argument("--tol", type=float, default=0.1, help="fraction of interannual SD")
    a = p.parse_args(argv)

    meta = json.load(open(a.pack / "metadata" / "data.json"))
    names = [str(c) for c in meta["coords"]["channel"]]
    state = [str(c) for c in meta["coords"]["channel_state"]]
    diag = [str(c) for c in meta["coords"]["channel_diagnostic"]]
    if names != state + diag:
        print("ERROR CHANNEL_ORDER: coords.channel != channel_state + channel_diagnostic"); return 1
    w = _weights(meta["coords"]["lat"])

    split = {}
    for s in ("train", "valid", "test"):
        for y in meta["attrs"][f"{s}_years"]:
            split[int(y)] = s
    jobs = [(str(a.pack / split[y] / f"{y}.h5"), a.stride, w) for y in a.years]
    res = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for path, gm, n in ex.map(year_mean, jobs):
            res[int(Path(path).stem)] = (gm, n)
            print(f"  {Path(path).stem}: {n} samples")
    years = sorted(res)
    agm = np.stack([res[y][0] for y in years])
    np.savez(a.out, years=np.array(years), channel_names=np.array(names),
             annual_global_mean=agm, n_samples=np.array([res[y][1] for y in years]), stride=a.stride)
    print(f"TRUTH_GM_OK years={years[0]}-{years[-1]} n={len(years)} channels={len(names)} out={a.out}")

    if a.check_truth_dir:
        sd_sel = [i for i, y in enumerate(years) if a.sd_years[0] <= y <= a.sd_years[1]]
        if len(sd_sel) < 10:
            print("ERROR TRUTH_GM_CHECK needs the SD years in --years"); return 1
        sd = detrended_sd(np.array(years)[sd_sel], agm[sd_sel])
        worst, checked = 0.0, 0
        for i, y in enumerate(years):
            f = a.check_truth_dir / f"truth_{y}.npz"
            if not f.exists():
                continue
            t = np.load(f, allow_pickle=True)
            tn = [str(c) for c in t["channel_names"]]
            gm_map = np.asarray(t["time_mean"], dtype=np.float64).mean(axis=-1) @ _weights(t["lat"])
            for c, n in enumerate(tn):
                j = names.index(n)
                worst = max(worst, abs(gm_map[c] - agm[i, j]) / sd[j]); checked += 1
        tag = "TRUTH_GM_EQUIV_OK" if checked and worst <= a.tol else "ERROR TRUTH_GM_EQUIV"
        print(f"{tag} checked={checked} max|diff|/interannual_sd={worst:.3f} (tol {a.tol})")
        if tag != "TRUTH_GM_EQUIV_OK":
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
