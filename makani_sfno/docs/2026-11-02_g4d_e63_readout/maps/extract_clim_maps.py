"""Climatological-bias inputs from 5-yr protocol member NetCDFs (window 2045-2047, the selection window).

For every arm found in the given climate_protocol_<job> dirs: each member's 2045-2047 time mean, computed by the
protocol scorer's own `window_mean` (month_count-weighted monthly means), for the plotted variables, plus the
ensemble mean of every shared channel.
Members the scorer skips (blown up, truncated, non-finite) are skipped and listed.
Gate: per member and direct variable, the area-weighted pattern RMSE of (member - E3SM) must equal the protocol
score's `e` in <tag>_channels.csv (relative 1e-9). For a re-rolled arm (--reroll) a difference is reported as WARN.
Also writes clim_truth.npz: E3SM time means, the land-sea mask (coastlines), lat/lon.
Usage: extract_clim_maps.py --out-dir maps/ --dirs 7807790 7808xxx --ref-tag G4D16=g4d --default-tag g4d63 \
       --reroll G4D16                                                       PASS = MAPS_CLIM_ARM_OK per arm
"""
import argparse
import csv
import glob
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/lus/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling/.claude/worktrees/"
                   "eval-protocol/makani_sfno/polaris")
import maps_lib as ml  # noqa: E402
import maps_io  # noqa: E402
import score_window_protocol as swp  # noqa: E402

E = "/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval"
TRUTH = E + "/window_truth/truth_2045_2047.npz"
Y0, Y1 = 2045, 2047
DIRECT = list(ml.VAR_KEYS)


def one_member(args):
    """Window mean over the channels the member shares with E3SM, in truth order (as the scorer does: a 99-ch
    member against the 101-ch truth would otherwise fail on the soil names and be skipped)."""
    import netCDF4
    path, tnames = args
    with netCDF4.Dataset(path) as ds:
        mnames = [str(c) for c in ds["channel"][:]]
    shared = [n for n in tnames if n in mnames]
    try:
        tm, _ = swp.window_mean(Path(path), Y0, Y1, shared)
    except ValueError as exc:
        return path, None, str(exc), shared
    return path, tm, "", shared          # float64, as the scorer: the 1e-9 gate needs the same precision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dirs", nargs="+", required=True, help="climate_protocol_<job> job ids")
    ap.add_argument("--ref-tag", nargs="*", default=[], help="ARM=TAG of protocol_scores/<TAG>_channels.csv")
    ap.add_argument("--default-tag", default="g4d63")
    ap.add_argument("--reroll", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    reftag = dict(s.split("=") for s in a.ref_tag)
    true = np.load(TRUTH, allow_pickle=True)
    tnames = [str(c) for c in true["channel_names"]]
    tmean = np.asarray(true["time_mean"], dtype=np.float64)
    w = swp.area_weights(true["lat"])
    lsm, plat, plon = maps_io.land_sea_mask()
    if not np.allclose(plat, np.asarray(true["lat"], dtype=np.float64)):
        print("ERROR LAT_MISMATCH truth vs pack")
        return 3
    truth_maps = np.stack([tmean[tnames.index(k)] for k in DIRECT])
    np.savez(os.path.join(a.out_dir, "clim_truth.npz"), truth_maps=truth_maps.astype(np.float32),
             truth_all=tmean.astype(np.float32), channels=np.array(tnames, dtype="U16"), var_keys=np.array(ml.VAR_KEYS),
             lsm=lsm.astype(np.float32), lat=plat, lon=plon, window=np.array([Y0, Y1]))
    print("truth: %d channels, window %d-%d; land fraction %.3f" % (len(tnames), Y0, Y1, ml.area_mean(lsm, w)))

    members = {}
    for d in a.dirs:
        for f in sorted(glob.glob(os.path.join(E, "climate_protocol_%s" % d, "member_*.nc"))):
            members.setdefault(swp.arm_of(Path(f)), []).append(f)
    if not members:
        print("ERROR NO_MEMBERS in %s" % a.dirs)
        return 3
    bad = 0
    for arm, files in sorted(members.items()):
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            res = list(ex.map(one_member, [(f, tnames) for f in files]))
        ok = [(p, tm) for p, tm, _, _ in res if tm is not None]
        skipped = ["%s (%s)" % (os.path.basename(p), why) for p, tm, why, _ in res if tm is None]
        shared = res[0][3]
        if any(r[3] != shared for r in res):
            print("ERROR MEMBER_CHANNELS_DIFFER %s" % arm)
            bad += 1
            continue
        if not ok:
            print("ERROR NO_USABLE_MEMBERS %s: %s" % (arm, skipped))
            bad += 1
            continue
        stack = np.stack([tm for _, tm in ok]).astype(np.float64)          # (n, C_shared, H, W), truth order
        mmaps = stack[:, [shared.index(k) for k in DIRECT]]

        tag = reftag.get(arm, a.default_tag)
        ref = {}
        for r in csv.DictReader(open(os.path.join(E, "protocol_scores", tag + "_channels.csv"))):
            if r["arm"] == arm:
                ref[(r["member"], r["channel"])] = float(r["e"])
        rel = []
        for (p, _), m in zip(ok, stack):
            for k in DIRECT:
                key = (os.path.basename(p), k)
                if key in ref:
                    e = float(swp.pattern_rmse(m[shared.index(k)][None], tmean[tnames.index(k)][None], w)[0])
                    rel.append(abs(e - ref[key]) / ref[key])
        worst = max(rel) if rel else float("nan")
        print("%s: %d members used, %d skipped%s; gate vs %s_channels.csv: %d values, max rel diff %.2e"
              % (arm, len(ok), len(skipped), (" " + "; ".join(skipped)) if skipped else "", tag, len(rel), worst))
        if not rel or worst > 1e-9:
            if arm in a.reroll and rel:
                print("WARN REROLL_DIFF %s: re-roll differs from the original 5-yr scores by up to %.3e (rel)"
                      % (arm, worst))
            else:
                print("ERROR CLIM_GATE %s (max rel diff %s)" % (arm, worst))
                bad += 1
                continue
        np.savez(os.path.join(a.out_dir, "clim_%s.npz" % arm), member_maps=mmaps.astype(np.float32),
                 ens_mean_all=stack.mean(axis=0).astype(np.float32), channels=np.array(shared, dtype="U16"),
                 var_keys=np.array(ml.VAR_KEYS), members=np.array([os.path.basename(p) for p, _ in ok], dtype="U40"),
                 skipped=np.array(skipped, dtype="U200"), window=np.array([Y0, Y1]))
        print("MAPS_CLIM_ARM_OK %s -> clim_%s.npz" % (arm, arm))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
