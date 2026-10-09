#!/usr/bin/env python3
"""score_window_protocol.py — climate error over a year window, by channel group.

Implements the metric of docs/2026-10-09_pipeline_comparison_prereg.md §3
(docs/makani-knowledge d06b4916):

* window time-mean of each member = month_count-weighted mean of its ``monthly_mean``
  over months whose ``month_year`` is in [Y0, Y1] (scored leads only);
* e_c = area-weighted pattern RMSE of that map against truth's time-mean over the same
  years (``build_true_climatology.py`` npz), also as rmse_sigma = e_c / global_std_c;
* floor_c = RMS over member pairs of the pattern RMSE between two members' window maps,
  per floor arm, then RMS-pooled over the floor arms;
* r_c = e_c / floor_c; 24 groups ({T,U,V,Z3,RELHUM} x {top l00-03, mid l04-12, low l13-17}
  + 9 surface channels); A = median over members of the mean over groups of the
  group-mean r; W = max over groups of the median-member group r.

Channels are matched BY NAME against the truth npz, so 101-channel members are scored
on the shared channels; soil channels are reported in the CSV but never enter A or W.

Modes:
  --check-legacy <fidelity csv>   reproduce an existing full-window rmse_sigma table
                                  (prints WINDOW_SCORER_OK / ERROR WINDOW_SCORER_MISMATCH)
  --write-floor <npz>             compute the floor from --floor-arms and save it
  --floor <npz>                   score with a saved floor (A, W, groups)

PASS tokens: WINDOW_SCORER_OK, FLOOR_OK, WINDOW_SCORE_OK.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import re
import sys
from pathlib import Path

import numpy as np

SURFACE = ["PS", "TREFHT", "U10", "RHREFHT", "PSL", "TMQ", "FSNT", "FSNTOA", "PRECT"]
BANDS = [("top", 0, 3), ("mid", 4, 12), ("low", 13, 17)]
SOIL = {"SOILWATER_10CM", "TSOI_10CM"}


def group_of(name: str) -> str | None:
    if name in SURFACE:
        return name
    m = re.fullmatch(r"(T|U|V|Z3|RELHUM)_l(\d\d)", name)
    if not m:
        return None
    lev = int(m.group(2))
    for band, lo, hi in BANDS:
        if lo <= lev <= hi:
            return f"{m.group(1)}_{band}"
    return None


def area_weights(lat: np.ndarray) -> np.ndarray:
    w = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
    return w / w.sum()


def pattern_rmse(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Area-weighted RMS of (a - b) over (H, W) for each channel: (C,)."""
    return np.sqrt(((a - b) ** 2).mean(axis=-1) @ w)


def window_mean(path: Path, y0: int, y1: int, names_wanted: list[str]):
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        names = [str(c) for c in ds["channel"][:]]
        idx = [names.index(n) for n in names_wanted]
        years = np.asarray(ds["month_year"][:])
        count = np.asarray(ds["month_count"][:], dtype=np.float64)
        sel = np.nonzero((years >= y0) & (years <= y1) & (count > 0))[0]
        if sel.size == 0:
            raise ValueError(f"NO_MONTHS {path} in {y0}-{y1}")
        acc = None
        for k in sel:
            m = np.ma.filled(ds["monthly_mean"][k], np.nan)[idx].astype(np.float64)
            acc = m * count[k] if acc is None else acc + m * count[k]
        tm = acc / count[sel].sum()
    if not np.isfinite(tm).all():
        raise ValueError(f"NONFINITE {path}")
    return tm, int(count[sel].sum())


def arm_of(path: Path) -> str:
    stem = path.stem.removeprefix("member_")
    return stem[:-2]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--true", required=True, type=Path)
    p.add_argument("--window", required=True, type=int, nargs=2, metavar=("Y0", "Y1"))
    p.add_argument("--member", required=True, type=Path, nargs="+")
    p.add_argument("--check-legacy", type=Path, default=None)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--floor-arms", nargs="*", default=[])
    p.add_argument("--write-floor", type=Path, default=None)
    p.add_argument("--floor", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None, help="per-member per-channel CSV")
    p.add_argument("--summary", type=Path, default=None, help="per-arm summary CSV")
    a = p.parse_args(argv)

    true = np.load(a.true, allow_pickle=False)
    tnames = [str(c) for c in true["channel_names"]]
    tmean = np.asarray(true["time_mean"], dtype=np.float64)
    tstd = np.asarray(true["global_stds"], dtype=np.float64).reshape(-1)
    w = area_weights(true["lat"])

    rows, maps = [], {}
    for mp in a.member:
        import netCDF4
        with netCDF4.Dataset(mp) as ds:
            mnames = [str(c) for c in ds["channel"][:]]
        shared = [n for n in tnames if n in mnames]
        tidx = [tnames.index(n) for n in shared]
        tm, nsteps = window_mean(mp, a.window[0], a.window[1], shared)
        e = pattern_rmse(tm, tmean[tidx], w)
        maps[mp] = (shared, tm)
        for n, ec, sc in zip(shared, e, tstd[tidx]):
            rows.append(dict(arm=arm_of(mp), member=mp.name, channel=n, group=group_of(n) or "",
                             e=ec, rmse_sigma=ec / sc, steps=nsteps))
    print(f"scored {len(a.member)} members, window {a.window[0]}-{a.window[1]}")

    if a.check_legacy:
        legacy = {}
        for r in csv.DictReader(open(a.check_legacy)):
            legacy[(r["member"], r["channel"])] = float(r["rmse_sigma"])
        diffs = [abs(r["rmse_sigma"] - legacy[(r["member"], r["channel"])]) for r in rows
                 if (r["member"], r["channel"]) in legacy]
        if not diffs:
            print("ERROR WINDOW_SCORER_NO_OVERLAP with the legacy table"); return 1
        worst = max(diffs)
        by_member = {}
        for r in rows:
            by_member.setdefault(r["member"], []).append(r["rmse_sigma"])
        means = sorted(np.mean(v) for v in by_member.values())
        tag = "WINDOW_SCORER_OK" if worst <= a.tol else "ERROR WINDOW_SCORER_MISMATCH"
        print(f"{tag} compared={len(diffs)} max|d rmse_sigma|={worst:.2e} (tol {a.tol:g}) "
              f"median member mean rmse_sigma={np.median(means):.4f}")
        if worst > a.tol:
            return 1

    if a.write_floor:
        per_arm = {}
        for arm in a.floor_arms:
            mems = [mp for mp in a.member if arm_of(mp) == arm]
            if len(mems) < 2:
                print(f"ERROR FLOOR_ARM_TOO_FEW {arm} n={len(mems)}"); return 1
            names = maps[mems[0]][0]
            sq = []
            for x, y in itertools.combinations(mems, 2):
                assert maps[x][0] == maps[y][0] == names
                sq.append(pattern_rmse(maps[x][1], maps[y][1], w) ** 2)
            per_arm[arm] = dict(zip(names, np.sqrt(np.mean(sq, axis=0))))
            print(f"floor arm {arm}: members={len(mems)} pairs={len(sq)}")
        common = [n for n in tnames if all(n in d for d in per_arm.values())]
        floor = np.sqrt(np.mean([[per_arm[arm][n] ** 2 for n in common] for arm in per_arm], axis=0))
        np.savez(a.write_floor, channel_names=np.array(common), floor=floor,
                 window=np.array(a.window), arms=np.array(list(per_arm)))
        print(f"FLOOR_OK channels={len(common)} arms={list(per_arm)} out={a.write_floor} "
              f"median floor/global_std={np.median(floor / tstd[[tnames.index(n) for n in common]]):.4f}")

    if a.floor:
        fl = np.load(a.floor, allow_pickle=False)
        fmap = dict(zip([str(c) for c in fl["channel_names"]], np.asarray(fl["floor"], dtype=np.float64)))
        for r in rows:
            r["r"] = r["e"] / fmap[r["channel"]] if r["channel"] in fmap else np.nan
        summary = []
        for arm in sorted({r["arm"] for r in rows}):
            ar = [r for r in rows if r["arm"] == arm and r["group"] and r["channel"] not in SOIL]
            members = sorted({r["member"] for r in ar})
            groups = sorted({r["group"] for r in ar})
            A_m, g_med = [], {}
            per_mg = {(m, g): np.mean([r["r"] for r in ar if r["member"] == m and r["group"] == g])
                      for m in members for g in groups}
            for m in members:
                A_m.append(np.mean([per_mg[(m, g)] for g in groups]))
            for g in groups:
                g_med[g] = float(np.median([per_mg[(m, g)] for m in members]))
            wg = max(g_med, key=g_med.get)
            ch_med = {}
            for r in ar:
                ch_med.setdefault(r["channel"], []).append(r["rmse_sigma"])
            cm = np.array([np.median(v) for v in ch_med.values()])
            no_top = np.array([np.median(v) for c, v in ch_med.items()
                               if not re.fullmatch(r".*_l0[0-2]", c)])
            summary.append(dict(arm=arm, members=len(members), groups=len(groups),
                                A=float(np.median(A_m)), W=g_med[wg], W_group=wg,
                                legacy_mean=float(np.mean(cm)), median_ch=float(np.median(cm)),
                                p90_ch=float(np.percentile(cm, 90)), max_ch=float(cm.max()),
                                mean_excl_l00_02=float(no_top.mean())))
        for s in sorted(summary, key=lambda s: s["A"]):
            print(f"  {s['arm']:<12} A={s['A']:.3f} W={s['W']:.2f} ({s['W_group']}) "
                  f"legacy={s['legacy_mean']:.4f} med={s['median_ch']:.4f} p90={s['p90_ch']:.3f} "
                  f"max={s['max_ch']:.3f} n={s['members']}")
        if a.summary:
            with open(a.summary, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(summary[0]))
                wr.writeheader(); wr.writerows(summary)
        print(f"WINDOW_SCORE_OK arms={len(summary)} window={a.window[0]}-{a.window[1]}")

    if a.out:
        with open(a.out, "w", newline="") as f:
            keys = ["arm", "member", "channel", "group", "e", "rmse_sigma", "r", "steps"]
            wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            wr.writeheader()
            for r in rows:
                wr.writerow({k: r.get(k, "") for k in keys})
    return 0


if __name__ == "__main__":
    sys.exit(main())
