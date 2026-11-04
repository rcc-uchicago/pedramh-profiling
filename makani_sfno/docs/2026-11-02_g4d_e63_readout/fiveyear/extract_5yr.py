"""Five-year figure inputs from the re-rolled climate_protocol_<job> member NetCDFs, plus the matching E3SM truth.

Per arm and member:
  * gates that the re-roll IS the original five-year run:
      - its 2045-2047 window mean of the map variables (score_window_protocol.window_mean) equals the original's,
        kept in maps/clim_<ARM>.npz (float32; relative 1e-6 of the field's largest magnitude);
      - its pattern RMSE against window_truth equals the protocol score `e` in protocol_scores/<tag>_channels.csv
        (relative 1e-9), as extract_clim_maps.py checked the original;
  * monthly and annual global means of every channel, and monthly zonal means of the zonal variables, on frames
    1..1459 of each year (fiveyear_lib: identical times to the truth);
  * annual-mean maps 2045-2049 of the map variables (window_mean per year: the protocol's own convention, which
    is also how the 2045-2047 bias maps were made).
Truth: the same quantities from truth/truth_series_<year>.npz (truth_series.py).
Writes fiveyear_<ARM>.npz and fiveyear_truth.npz into --out-dir. PASS = FIVEYEAR_EXTRACT_OK
"""
import argparse
import csv
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_POL = ("/lus/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling/.claude/worktrees/"
            "eval-protocol/makani_sfno/polaris")
sys.path.insert(0, HERE)
sys.path.insert(0, EVAL_POL)
import fiveyear_lib as fl  # noqa: E402
import score_window_protocol as swp  # noqa: E402

E = "/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval"
MAPS = E + "/g4d_e63_2026-11-02/maps"
WINDOW_TRUTH = E + "/window_truth/truth_2045_2047.npz"
REF_TAG = {"G4D16": "g4d"}            # as the original extraction: --ref-tag G4D16=g4d --default-tag g4d63
DEFAULT_TAG = "g4d63"


def one_member(path):
    import netCDF4
    out = {"path": path}
    with netCDF4.Dataset(path) as ds:
        names = [str(c) for c in ds["channel"][:]]
        years = np.asarray(ds["valid_year"][:])
        frames = np.asarray(ds["valid_frame"][:])
        gm = np.ma.filled(ds["global_mean"][:], np.nan).astype(np.float64)          # (n, C), physical units
        if "zonal_mean" not in ds.variables:
            return dict(out, error="NO_ZONAL_MEAN (re-roll without --zonal-mean?)")
        zi = [names.index(k) for k in fl.ZONAL_KEYS]
        zm = np.stack([np.ma.filled(ds["zonal_mean"][:, i, :], np.nan) for i in zi], axis=1)   # (n, Z, H)
    if not (np.isfinite(gm).all() and np.isfinite(zm).all()):
        return dict(out, error="NONFINITE series")
    out["names"] = names
    out["gm_month"], out["cnt_month"] = fl.bin_by_month(gm, years, frames)
    out["gm_year"], out["n_year"] = fl.year_means(gm, years, frames)
    out["zm_month"], _ = fl.bin_by_month(zm, years, frames)
    out["ann_maps"] = np.stack([swp.window_mean(Path(path), y, y, list(fl.MAP_KEYS))[0] for y in fl.YEARS])
    out["win_maps"] = swp.window_mean(Path(path), fl.SELECTION_WINDOW[0], fl.SELECTION_WINDOW[1],
                                      list(fl.MAP_KEYS))[0]
    out["last"] = (int(years[-1]), int(frames[-1]))
    return out


def truth_arrays(tdir):
    parts = {}
    for y in (2044,) + fl.YEARS:
        p = os.path.join(tdir, "truth_series_%d.npz" % y)
        if not os.path.exists(p):
            raise SystemExit("ERROR TRUTH_MISSING %s" % p)
        parts[y] = np.load(p)
    names = [str(c) for c in parts[2045]["channel_names"]]
    for y, z in parts.items():
        if [str(c) for c in z["channel_names"]] != names:
            raise SystemExit("ERROR TRUTH_CHANNELS_DIFFER %d" % y)
        if list(z["zonal_vars"]) != list(fl.ZONAL_KEYS):
            raise SystemExit("ERROR TRUTH_ZONAL_VARS %d: %r" % (y, list(z["zonal_vars"])))
    years = np.concatenate([np.full(len(parts[y]["frames"]), y) for y in sorted(parts)])
    frames = np.concatenate([parts[y]["frames"] for y in sorted(parts)])
    gm = np.concatenate([parts[y]["gm"] for y in sorted(parts)])
    zm = np.concatenate([parts[y]["zm"] for y in sorted(parts)]).astype(np.float64)
    T = {"names": names, "lat": parts[2045]["lat"], "lon": parts[2045]["lon"]}
    T["gm_month"], T["cnt_month"] = fl.bin_by_month(gm, years, frames)
    T["gm_year"], T["n_year"] = fl.year_means(gm, years, frames)
    T["zm_month"], _ = fl.bin_by_month(zm, years, frames)
    T["ann_maps"] = np.stack([parts[y]["time_mean"][[names.index(k) for k in fl.MAP_KEYS]] for y in fl.YEARS])
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="climate_protocol_<job> of the re-roll")
    ap.add_argument("--truth-dir", default=os.path.join(HERE, "truth"))
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    import sfno_inference.climate_driver as cd                    # the calendar fiveyear_lib replicates
    if fl.month_edges() != cd.month_edges():
        print("ERROR CALENDAR_MISMATCH fiveyear_lib vs climate_driver")
        return 3
    T = truth_arrays(a.truth_dir)
    wt = np.load(WINDOW_TRUTH, allow_pickle=True)
    wnames = [str(c) for c in wt["channel_names"]]
    wmean = np.asarray(wt["time_mean"], dtype=np.float64)
    w = swp.area_weights(wt["lat"])
    ct = np.load(os.path.join(MAPS, "clim_truth.npz"))
    if not np.allclose(ct["lat"], T["lat"]):
        print("ERROR LAT_MISMATCH truth series vs clim_truth.npz")
        return 3
    np.savez(os.path.join(a.out_dir, "fiveyear_truth.npz"), channels=np.array(T["names"], dtype="U16"),
             gm_month=T["gm_month"], cnt_month=T["cnt_month"], gm_year=T["gm_year"], n_year=T["n_year"],
             zm_month=T["zm_month"].astype(np.float32), ann_maps=T["ann_maps"].astype(np.float32),
             lat=T["lat"], lon=T["lon"], lsm=ct["lsm"], months=np.array(fl.MONTHS), years=np.array(fl.YEARS),
             zonal_keys=np.array(fl.ZONAL_KEYS), map_keys=np.array(fl.MAP_KEYS))
    print("truth: %d channels; annual counts %s; months with data %d/%d"
          % (len(T["names"]), T["n_year"].tolist(), int((T["cnt_month"] > 0).sum()), fl.N_MONTHS))

    files = sorted(glob.glob(os.path.join(E, "climate_protocol_%s" % a.dir, "member_*.nc")))
    arms = {}
    for f in files:
        arms.setdefault(swp.arm_of(Path(f)), []).append(f)
    if not arms:
        print("ERROR NO_MEMBERS climate_protocol_%s" % a.dir)
        return 3
    bad = 0
    for arm, fs in sorted(arms.items()):
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            res = list(ex.map(one_member, fs))
        errs = ["%s: %s" % (os.path.basename(r["path"]), r["error"]) for r in res if "error" in r]
        if errs:
            print("ERROR MEMBER_UNUSABLE %s: %s" % (arm, "; ".join(errs)))
            bad += 1
            continue
        if any(r["names"] != T["names"] for r in res):
            print("ERROR CHANNELS_DIFFER %s vs truth" % arm)
            bad += 1
            continue
        # gate 1: the original run's window means (clim_<ARM>.npz), member by member
        orig = np.load(os.path.join(MAPS, "clim_%s.npz" % arm))
        omem = [str(m) for m in orig["members"]]
        okeys = [str(k) for k in orig["var_keys"]]
        rel1, ident, ncell = 0.0, 0, 0
        for r in res:
            b = os.path.basename(r["path"])
            if b not in omem:
                print("ERROR REROLL_MEMBER_NOT_IN_ORIGINAL %s %s" % (arm, b))
                bad += 1
                continue
            o = orig["member_maps"][omem.index(b)][[okeys.index(k) for k in fl.MAP_KEYS]]
            mine = r["win_maps"].astype(np.float32)
            scale = np.abs(o).max(axis=(1, 2)).astype(np.float64)
            rel1 = max(rel1, float((np.abs(mine.astype(np.float64) - o).max(axis=(1, 2)) / scale).max()))
            ident += int((mine == o).sum())
            ncell += o.size
        # gate 2: protocol score e, as extract_clim_maps.py
        tag = REF_TAG.get(arm, DEFAULT_TAG)
        ref = {(row["member"], row["channel"]): float(row["e"])
               for row in csv.DictReader(open(os.path.join(E, "protocol_scores", tag + "_channels.csv")))
               if row["arm"] == arm}
        rel2 = []
        for r in res:
            b = os.path.basename(r["path"])
            for j, k in enumerate(fl.MAP_KEYS):
                if (b, k) in ref:
                    e = float(swp.pattern_rmse(r["win_maps"][j][None], wmean[wnames.index(k)][None], w)[0])
                    rel2.append(abs(e - ref[(b, k)]) / ref[(b, k)])
        worst2 = max(rel2) if rel2 else float("nan")
        print("%s: %d members, ends %s; gate 1 vs clim_%s.npz: max rel %.2e, %d/%d float32 cells identical; "
              "gate 2 vs %s_channels.csv: %d values, max rel %.2e"
              % (arm, len(res), sorted(set(r["last"] for r in res)), arm, rel1, ident, ncell, tag, len(rel2),
                 worst2))
        if not (rel1 <= 1e-6 and rel2 and worst2 <= 1e-9 and len(res) == len(omem)):
            print("ERROR REROLL_GATE %s" % arm)
            bad += 1
            continue
        np.savez(os.path.join(a.out_dir, "fiveyear_%s.npz" % arm),
                 members=np.array([os.path.basename(r["path"]) for r in res], dtype="U40"),
                 channels=np.array(T["names"], dtype="U16"),
                 gm_month=np.stack([r["gm_month"] for r in res]), cnt_month=np.stack([r["cnt_month"] for r in res]),
                 gm_year=np.stack([r["gm_year"] for r in res]), n_year=np.stack([r["n_year"] for r in res]),
                 zm_month=np.stack([r["zm_month"] for r in res]).astype(np.float32),
                 ann_maps=np.stack([r["ann_maps"] for r in res]).astype(np.float32),
                 months=np.array(fl.MONTHS), years=np.array(fl.YEARS), zonal_keys=np.array(fl.ZONAL_KEYS),
                 map_keys=np.array(fl.MAP_KEYS), gate=np.array([rel1, worst2]))
        print("FIVEYEAR_ARM_OK %s -> fiveyear_%s.npz" % (arm, arm))
    if bad:
        print("ERROR FIVEYEAR_EXTRACT_FAILED arms_failed=%d" % bad)
        return 1
    print("FIVEYEAR_EXTRACT_OK arms=%d" % len(arms))
    return 0


if __name__ == "__main__":
    sys.exit(main())
