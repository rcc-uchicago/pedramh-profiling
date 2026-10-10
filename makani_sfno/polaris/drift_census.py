#!/usr/bin/env python3
"""drift_census.py — the drift gate of the comparison protocol (prereg §4.4).

Per member and channel:
    drift_c = (model - truth)_annual-global-mean(Y1) - (model - truth)(Y0)
with the model's annual global mean taken from its ``monthly_mean`` (month_count
weighted, cos-lat area weights) and truth's from ``truth_annual_global_means.py``.
Expressed in units of truth's linearly detrended interannual SD (2015-2044 by default).
Per arm: the median over members of drift_c; a group (the 24 groups of
score_window_protocol.py) drifts if the median over its channels of |drift| > --thresh.
Also prints the global TREFHT drift in K (amendment 2's 0.5 K criterion).

Selection uses Y0=2045, Y1=2047. Do not run Y1 > 2047 before finalists are frozen.
PASS: DRIFT_CENSUS_OK
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_window_protocol import SOIL, arm_of, group_of  # noqa: E402
from truth_annual_global_means import detrended_sd  # noqa: E402


def member_annual_gm(path: Path, years: list[int]) -> tuple[list[str], dict]:
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        names = [str(c) for c in ds["channel"][:]]
        lat = np.asarray(ds["lat"][:], dtype=np.float64)
        w = np.cos(np.deg2rad(lat)); w /= w.sum()
        my = np.asarray(ds["month_year"][:]); cnt = np.asarray(ds["month_count"][:], dtype=np.float64)
        out = {}
        for y in years:
            sel = np.nonzero((my == y) & (cnt > 0))[0]
            if sel.size == 0:
                raise ValueError(f"NO_MONTHS {path} {y}")
            acc = 0.0
            for k in sel:
                m = np.ma.filled(ds["monthly_mean"][k], np.nan).astype(np.float64)
                acc = acc + (m.mean(axis=-1) @ w) * cnt[k]
            out[y] = acc / cnt[sel].sum()
    return names, out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--truth-gm", required=True, type=Path)
    p.add_argument("--member", required=True, type=Path, nargs="+")
    p.add_argument("--y0", type=int, default=2045)
    p.add_argument("--y1", type=int, default=2047)
    p.add_argument("--sd-years", type=int, nargs=2, default=(2015, 2044))
    p.add_argument("--thresh", type=float, default=2.0)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    if a.y1 > 2047:
        print("ERROR SEALED_WINDOW: Y1 > 2047 is the reporting window (prereg §2)"); return 1

    t = np.load(a.truth_gm, allow_pickle=True)
    tyears = [int(y) for y in t["years"]]; tnames = [str(c) for c in t["channel_names"]]
    agm = np.asarray(t["annual_global_mean"], dtype=np.float64)
    sel = [i for i, y in enumerate(tyears) if a.sd_years[0] <= y <= a.sd_years[1]]
    sd = detrended_sd(np.array(tyears)[sel], agm[sel])
    tr = {y: agm[tyears.index(y)] for y in (a.y0, a.y1)}

    per_arm: dict[str, list] = {}
    for mp in a.member:
        try:
            names, m = member_annual_gm(mp, [a.y0, a.y1])
        except ValueError as exc:
            print(f"SKIP {mp.name}: {exc}")
            continue
        if not all(np.isfinite(v).all() for v in m.values()):
            print(f"SKIP {mp.name}: non-finite annual mean"); continue
        idx = [tnames.index(n) for n in names]
        d_phys = (m[a.y1] - tr[a.y1][idx]) - (m[a.y0] - tr[a.y0][idx])
        per_arm.setdefault(arm_of(mp), []).append((names, d_phys, d_phys / sd[idx]))

    rows = []
    print(f"drift census {a.y1}-{a.y0}, unit = truth detrended interannual SD {a.sd_years[0]}-{a.sd_years[1]}")
    for arm, ms in sorted(per_arm.items()):
        names = ms[0][0]
        med_sd = np.median(np.stack([x[2] for x in ms]), axis=0)
        med_phys = np.median(np.stack([x[1] for x in ms]), axis=0)
        groups: dict[str, list] = {}
        for n, v in zip(names, med_sd):
            g = group_of(n)
            if g and n not in SOIL:
                groups.setdefault(g, []).append(abs(v))
        drifting = sorted(g for g, v in groups.items() if np.median(v) > a.thresh)
        trefht = float(med_phys[names.index("TREFHT")]) if "TREFHT" in names else float("nan")
        ps = float(med_phys[names.index("PS")]) / 100.0 if "PS" in names else float("nan")
        print(f"  {arm:<14} members={len(ms)} drifting_groups={len(drifting)} TREFHT={trefht:+.2f} K "
              f"PS={ps:+.2f} hPa {drifting}")
        rows.append(dict(arm=arm, members=len(ms), drifting_groups=len(drifting), groups=" ".join(drifting),
                         trefht_drift_K=trefht, ps_drift_hPa=ps))
    if a.out:
        with open(a.out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f"DRIFT_CENSUS_OK arms={len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
