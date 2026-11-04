"""Score G4D the way Duncan et al. (2024) score ACE: one run's 5-year time-mean RMSE against E3SM, divided by
E3SM's own window-to-window RMSE (prereg.md, written before any number here existed).

    paperstyle.py --selftest                     metric code on synthetic fields   PASS = PAPERSTYLE_SELFTEST_OK
    paperstyle.py --rr-dir <climate_protocol_*> [--delete-nc]                     PASS = PAPERSTYLE_OK

Inputs: the re-roll's member NetCDFs (G4D16/36/43, 8 runs each); E3SM annual means truth_series_<year>.npz for
2035-2049 (fiveyear/truth/); for the gates, maps/clim_<ARM>.npz, protocol_scores/{g4d,g4d63}_channels.csv and
window_truth/truth_2045_2047.npz. Outputs here: results.json, results.txt, paperstyle_<ARM>.npz (each run's
2045-2049 mean maps), truth_windows.npz. Compute node, SFNO venv (netCDF4).
"""
import argparse
import csv
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

EVAL = Path("/lus/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling/.claude/worktrees/"
            "eval-protocol/makani_sfno")
sys.path.insert(0, str(EVAL / "polaris"))
import score_window_protocol as swp  # noqa: E402

E = Path("/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval")
G = E / "g4d_e63_2026-11-02"
TRUTH = G / "fiveyear" / "truth"
MAPS = G / "maps"
SCORES = E / "protocol_scores"
WINDOW_TRUTH = E / "window_truth" / "truth_2045_2047.npz"
HERE = Path(__file__).resolve().parent

ARMS = ("G4D16", "G4D36", "G4D43")
TAG = {"G4D16": "g4d", "G4D36": "g4d63", "G4D43": "g4d63"}        # where each arm's protocol score lives
MAP_KEYS = ("TREFHT", "PRECT", "U10", "U_l17", "Z3_l12")           # what maps/clim_<ARM>.npz holds
WIN, REF, REF2 = (2045, 2049), (2040, 2044), (2035, 2039)
FRAMES_PER_YEAR = 1459
REPORT = ("PRECT", "FSNTOA", "PS", "TREFHT", "U10", "Z3_l12", "U_l17", "T_l17")
UNIT = {"PRECT": (86400.0 * 1000.0, "mm/day"), "PS": (0.01, "hPa"), "FSNTOA": (1.0, "W/m2"), "TREFHT": (1.0, "K"),
        "T_l17": (1.0, "K"), "U10": (1.0, "m/s"), "U_l17": (1.0, "m/s"), "Z3_l12": (1.0, "m")}
# Duncan et al. 2024, Table S5 (papers/duncan2024_readout.md): ACE-EAMv2 RMSE, EAMv2 decade-to-decade RMSE.
PAPER = {"PRECT": ("precipitation", 0.37, 0.21), "FSNTOA": ("reflected SW at TOA (RSW)", 4.17, 1.63)}
FLOOR_TOL = 0.25


def rmse(a, b, w):
    """Area-weighted global RMS of a - b per channel, (C,): the scorer's pattern RMSE."""
    return swp.pattern_rmse(np.asarray(a, np.float64), np.asarray(b, np.float64), w)


def gmean(a, w):
    """Area-weighted global mean per channel, (C,)."""
    return np.asarray(a, np.float64).mean(axis=-1) @ w


def r10(r):
    """10-year equivalent of a 5-year ratio: noise variance ~ 1/years, run variability = E3SM's, fixed bias."""
    return float(np.sqrt(1.0 + 2.0 * max(r * r - 1.0, 0.0)))


def selftest():
    lat = np.linspace(89.5, -89.5, 180)
    w = swp.area_weights(lat)
    a = np.zeros((2, 180, 360))
    b = a + 3.0
    assert np.allclose(rmse(a, b, w), 3.0) and np.allclose(gmean(b - a, w), 3.0)
    c = a.copy()
    c[:, :90] = 1.0                                       # northern hemisphere only: RMS = sqrt(1/2)
    assert np.allclose(rmse(c, a, w), np.sqrt(0.5), atol=1e-12)
    assert r10(1.0) == 1.0 and r10(0.5) == 1.0 and abs(r10(2.0) - np.sqrt(7.0)) < 1e-12
    years = [np.full((1, 2, 2), float(k)) for k in range(5)]
    assert np.allclose(np.mean(years, axis=0), 2.0)
    print("PAPERSTYLE_SELFTEST_OK")
    return 0


def load_truth():
    names, maps, lat = None, {}, None
    for y in range(REF2[0], WIN[1] + 1):
        z = np.load(TRUTH / ("truth_series_%d.npz" % y))
        n = [str(c) for c in z["channel_names"]]
        if names is None:
            names, lat = n, np.asarray(z["lat"], np.float64)
        if n != names or int(z["n_total"]) != FRAMES_PER_YEAR:
            raise ValueError("TRUTH_YEAR_MISMATCH %d: %d channels, %d frames" % (y, len(n), int(z["n_total"])))
        maps[y] = np.asarray(z["time_mean"], np.float64)
    win = {k: np.mean([maps[y] for y in range(k[0], k[1] + 1)], axis=0) for k in (WIN, REF, REF2)}
    return names, lat, win


def one_member(job):
    """(basename, channel names, 2045-2047 means of the window-truth channels, 2045-2049 means of names)."""
    path, names, wnames = job
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        have = [str(c) for c in ds["channel"][:]]
    shared = [n for n in names if n in have]
    wshared = [n for n in wnames if n in have]
    m47, _ = swp.window_mean(Path(path), 2045, 2047, wshared)
    m49, n49 = swp.window_mean(Path(path), WIN[0], WIN[1], shared)
    return os.path.basename(path), shared, wshared, m47, m49, n49


def analyse(rr, delete_nc):
    names, lat, T = load_truth()
    w = swp.area_weights(lat)
    floor, floor2 = rmse(T[REF], T[WIN], w), rmse(T[REF2], T[REF], w)
    gdiff, gdiff2 = gmean(T[REF] - T[WIN], w), gmean(T[REF2] - T[REF], w)
    wt = np.load(WINDOW_TRUTH, allow_pickle=True)
    wnames = [str(c) for c in wt["channel_names"]]
    wmean = np.asarray(wt["time_mean"], np.float64)
    ww = swp.area_weights(wt["lat"])

    bad, res = [], {"arms": {}, "floor": {}, "trend_check": {}, "paper": {}}
    for arm in ARMS:
        files = sorted(glob.glob(os.path.join(rr, "member_%s0?.nc" % arm)))
        if len(files) != 8:
            bad.append("%s has %d member files" % (arm, len(files)))
            continue
        with ProcessPoolExecutor(max_workers=8) as ex:
            out = list(ex.map(one_member, [(f, names, wnames) for f in files]))
        orig = np.load(MAPS / ("clim_%s.npz" % arm))
        omem = [str(m) for m in orig["members"]]
        okeys = [str(k) for k in orig["var_keys"]]
        ref = {(r["member"], r["channel"]): float(r["e"])
               for r in csv.DictReader(open(SCORES / ("%s_channels.csv" % TAG[arm]))) if r["arm"] == arm}
        rel1, rel2, maps49, rows = 0.0, 0.0, [], []
        for b, shared, wshared, m47, m49, n49 in out:
            if shared != names:
                bad.append("%s %s lacks channels %s" % (arm, b, sorted(set(names) - set(shared))))
                continue
            # gate 1: the re-run's 2045-2047 means of the map variables equal the original run's
            o = orig["member_maps"][omem.index(b)][[okeys.index(k) for k in MAP_KEYS]]
            mine = m47[[wshared.index(k) for k in MAP_KEYS]].astype(np.float32)
            scale = np.abs(o).max(axis=(1, 2)).astype(np.float64)
            rel1 = max(rel1, float((np.abs(mine.astype(np.float64) - o).max(axis=(1, 2)) / scale).max()))
            # gate 2: its 2045-2047 pattern RMSE equals the protocol score e, channel by channel
            e = swp.pattern_rmse(m47, wmean[[wnames.index(k) for k in wshared]], ww)
            for j, k in enumerate(wshared):
                if (b, k) in ref:
                    rel2 = max(rel2, abs(e[j] - ref[(b, k)]) / ref[(b, k)])
            maps49.append(m49.astype(np.float32))
            rows.append({"member": b, "rmse": rmse(m49, T[WIN], w), "bias": gmean(m49 - T[WIN], w),
                         "frames": int(n49)})
        print("%s: %d members; gate 1 max rel %.2e; gate 2 max rel %.2e" % (arm, len(rows), rel1, rel2))
        if not (len(rows) == 8 and rel1 <= 1e-6 and rel2 <= 1e-9):
            bad.append("%s gates: members %d, rel1 %.2e, rel2 %.2e" % (arm, len(rows), rel1, rel2))
            continue
        np.savez(HERE / ("paperstyle_%s.npz" % arm), maps_2045_2049=np.stack(maps49), channel_names=np.array(names),
                 members=np.array([r["member"] for r in rows]), lat=lat)
        R = np.stack([r["rmse"] / floor for r in rows])                               # (8, C)
        arm_res = {"gate1_rel": rel1, "gate2_rel": rel2, "all_channel_mean_R": {
            "median": float(np.median(R.mean(axis=1))), "min": float(R.mean(axis=1).min()),
            "max": float(R.mean(axis=1).max())}, "channels": {}}
        for k in REPORT:
            c = names.index(k)
            fac = UNIT[k][0]
            rm = np.array([r["rmse"][c] for r in rows]) * fac
            bi = np.array([r["bias"][c] for r in rows]) * fac
            arm_res["channels"][k] = {
                "unit": UNIT[k][1], "rmse_median": float(np.median(rm)), "rmse_min": float(rm.min()),
                "rmse_max": float(rm.max()), "bias_median": float(np.median(bi)),
                "R_median": float(np.median(R[:, c])), "R_min": float(R[:, c].min()), "R_max": float(R[:, c].max()),
                "R10_of_median": r10(float(np.median(R[:, c])))}
        res["arms"][arm] = arm_res
    for k in REPORT:
        c = names.index(k)
        fac = UNIT[k][0]
        res["floor"][k] = {"unit": UNIT[k][1], "F": float(floor[c] * fac), "F_check": float(floor2[c] * fac),
                           "F_check_over_F": float(floor2[c] / floor[c]),
                           "floor_uncertain": bool(abs(floor2[c] / floor[c] - 1.0) > FLOOR_TOL)}
    for k in ("TREFHT", "PS"):
        c = names.index(k)
        fac = UNIT[k][0]
        res["trend_check"][k] = {"unit": UNIT[k][1], "gmean_2040_44_minus_2045_49": float(gdiff[c] * fac),
                                 "gmean_2035_39_minus_2040_44": float(gdiff2[c] * fac),
                                 "half_floor": float(floor[c] * fac / 2.0),
                                 "forced_change_flag": bool(abs(gdiff[c]) > floor[c] / 2.0)}
    for k, (label, ace, eam) in PAPER.items():
        res["paper"][k] = {"label": label, "ace_rmse": ace, "eam_floor": eam, "paper_ratio": ace / eam, "verdict": {}}
        for arm, ar in res["arms"].items():
            ch = ar["channels"][k]
            p = ace / eam
            v = ("below the paper" if ch["R_median"] < p and ch["R_max"] < p else
                 "above the paper" if ch["R_median"] > p else "mixed")
            res["paper"][k]["verdict"][arm] = v
    json.dump(res, open(HERE / "results.json", "w"), indent=1)
    np.savez(HERE / "truth_windows.npz", **{"E_%d_%d" % k: T[k].astype(np.float32) for k in (WIN, REF, REF2)},
             channel_names=np.array(names), lat=lat)
    write_txt(res)
    if bad:
        for b in bad:
            print("ERROR PAPERSTYLE_GATE %s" % b)
        print("NetCDFs kept in %s" % rr)
        return 1
    if delete_nc:
        freed = 0
        with open(HERE / "freed_nc.txt", "w") as fh:
            for f in sorted(glob.glob(os.path.join(rr, "member_*.nc"))):
                sz = os.path.getsize(f)
                fh.write("%s %d\n" % (f, sz))
                os.remove(f)
                freed += sz
        print("freed %s: %.1f GB" % (rr, freed / 1e9))
    print("PAPERSTYLE_OK arms=%d" % len(res["arms"]))
    return 0


def write_txt(res):
    L = ["G4D scored like Duncan et al. 2024 (prereg.md): one run's 2045-2049 mean vs E3SM; ratio R = RMSE / E3SM's "
         "own 2040-2044 vs 2045-2049 RMSE. Median [min, max] over 8 runs.", ""]
    L.append("E3SM floor (5-yr windows): channel, F, check F' (2035-39 vs 2040-44), F'/F, flag")
    for k, f in res["floor"].items():
        L.append("  %-7s F %.4g %s  F' %.4g  ratio %.2f%s" % (k, f["F"], f["unit"], f["F_check"], f["F_check_over_F"],
                                                             "  FLOOR-UNCERTAIN" if f["floor_uncertain"] else ""))
    L.append("Trend check (E3SM global-mean window difference vs half the floor):")
    for k, t in res["trend_check"].items():
        L.append("  %-7s %+.4g / %+.4g %s (half floor %.4g)%s" % (
            k, t["gmean_2040_44_minus_2045_49"], t["gmean_2035_39_minus_2040_44"], t["unit"], t["half_floor"],
            "  FORCED-CHANGE" if t["forced_change_flag"] else ""))
    L.append("")
    for arm, ar in res["arms"].items():
        L.append("%s  (gates: %.1e, %.1e)  all-channel mean R %.2f [%.2f, %.2f]" % (
            arm, ar["gate1_rel"], ar["gate2_rel"], ar["all_channel_mean_R"]["median"], ar["all_channel_mean_R"]["min"],
            ar["all_channel_mean_R"]["max"]))
        for k, ch in ar["channels"].items():
            L.append("  %-7s RMSE %.4g %s [%.4g, %.4g]  bias %+.3g  R %.2f [%.2f, %.2f]  R10~%.2f" % (
                k, ch["rmse_median"], ch["unit"], ch["rmse_min"], ch["rmse_max"], ch["bias_median"], ch["R_median"],
                ch["R_min"], ch["R_max"], ch["R10_of_median"]))
    L.append("")
    L.append("Against the paper (ACE-EAMv2, 10-yr):")
    for k, p in res["paper"].items():
        L.append("  %s: paper ratio %.2f (%.2f / %.2f); %s" % (
            p["label"], p["paper_ratio"], p["ace_rmse"], p["eam_floor"],
            "; ".join("%s %s" % (a, v) for a, v in p["verdict"].items())))
    open(HERE / "results.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--rr-dir")
    ap.add_argument("--delete-nc", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.rr_dir:
        print("ERROR NO_RR_DIR")
        return 2
    return analyse(a.rr_dir, a.delete_nc)


if __name__ == "__main__":
    sys.exit(main())
