"""Gridpoint error maps from K=56 sweep NetCDFs, for the RMSE maps / curves / significance tests.

Per arm (one sweep = 24 ICs, files <E>/<sweep>/inference/nwp/*.nc carrying `prediction` and `truth` as
(init_time=1, lead, channel, lat, lon) in physical units):
  mse_map   (n_map_leads, C, H, W)   mean over ICs of the squared error, every channel   [stored units^2]
  se_<VAR>  (n_ic, n_map_leads, H, W) per-IC squared error for the plotted variables
  rmse_ic   (n_ic, n_extract_leads, C) per-IC lat-weighted RMSE, the scorer's own function and weights
Gate: mean over ICs of rmse_ic must reproduce <ref>/scores/k56_summary.csv (metric rmse, mean_over_ic) at every
extracted lead and channel. The CSV stores 8 significant digits (score_rollout_nc.py writes %.8g), so rounding alone
reaches 5e-8 relative; the tolerance is GATE_TOL = 1e-7, while a wrong IC, lead or weight moves it by >= 1e-4.
Own sweep: above GATE_TOL, or nothing to compare -> ERROR. A re-roll checked against the original sweep's scores:
within GATE_TOL = reproduces; above it -> WARN with the number (the reproducibility check, not a failure); nothing
to compare -> ERROR (no check ran, so its NetCDFs must not be deleted).
Usage: extract_k56_maps.py --out-dir maps/ ARM:SWEEP:REFSWEEP [...]     PASS (per arm) = MAPS_K56_ARM_OK
"""
import argparse
import csv
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import maps_lib as ml  # noqa: E402

E = "/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval"
PLOT_DIRECT = list(ml.VAR_KEYS)
GATE_TOL = 1e-7
def _init():
    import torch
    torch.set_num_threads(1)


def one_ic(path):
    """All extracted leads of one IC file -> dict of arrays (float32 maps, float64 scalars)."""
    import torch
    import xarray as xr
    from sfno_eval import metrics as M
    ds = xr.open_dataset(path, decode_timedelta=False)
    try:
        chan = [str(c) for c in ds["channel"].values]
        lead_h = np.asarray(ds["lead_time"].values)
        if np.issubdtype(lead_h.dtype, np.timedelta64):
            lead_h = (lead_h / np.timedelta64(1, "h")).astype(np.int64)
        lead_h = lead_h.astype(np.int64).tolist()
        missing = [h for h in ml.EXTRACT_LEADS_H if h not in lead_h]
        if missing:
            raise SystemExit("ERROR LEADS_MISSING %s in %s" % (missing, path))
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        w = M.lat_weights(len(lat), "equiangular").to(torch.float64)
        ci = [chan.index(k) for k in PLOT_DIRECT]
        n_map, n_ext = len(ml.MAP_LEADS_H), len(ml.EXTRACT_LEADS_H)
        H, W = ds["prediction"].shape[-2:]
        out = dict(chan=chan, lat=lat, lon=np.asarray(ds["lon"].values, dtype=np.float64),
                   sq=np.zeros((n_map, len(chan), H, W), np.float32),
                   se=np.zeros((len(ml.VAR_KEYS), n_map, H, W), np.float32),
                   rmse=np.zeros((n_ext, len(chan))))
        for j, h in enumerate(ml.EXTRACT_LEADS_H):
            k = lead_h.index(h)
            p = np.asarray(ds["prediction"][0, k].values, dtype=np.float64)
            t = np.asarray(ds["truth"][0, k].values, dtype=np.float64)
            out["rmse"][j] = M.rmse_lat_weighted(torch.from_numpy(p), torch.from_numpy(t), w).numpy()
            if h in ml.MAP_LEADS_H:
                m = ml.MAP_LEADS_H.index(h)
                d2 = (p - t) ** 2
                out["sq"][m] = d2
                out["se"][:, m] = d2[ci]
        return out
    finally:
        ds.close()


def ref_means(ref_sweep):
    path = os.path.join(E, ref_sweep, "scores", "k56_summary.csv")
    ref = {}
    for r in csv.DictReader(open(path)):
        if r["metric"] == "rmse":
            ref[(r["channel"], int(r["lead_hours"]))] = (float(r["mean_over_ic"]), int(r["n_ic"]))
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit-ics", type=int, default=0, help="smoke: first N ICs only (gate then reports, n_ic differs)")
    ap.add_argument("arms", nargs="+", help="ARM:SWEEP:REFSWEEP (dirs under %s)" % E)
    a = ap.parse_args()
    bad = 0
    for spec in a.arms:
        arm, sweep, refsweep = spec.split(":")
        files = sorted(glob.glob(os.path.join(E, sweep, "inference", "nwp", "*.nc")))
        if a.limit_ics:
            files = files[:a.limit_ics]
        if not files:
            print("ERROR NO_NETCDF %s" % sweep)
            return 3
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers, initializer=_init) as ex:
            res = list(ex.map(one_ic, files))
        chan = res[0]["chan"]
        if any(r["chan"] != chan for r in res):
            print("ERROR CHANNEL_ORDER_DIFFERS %s" % arm)
            return 4
        n = len(res)
        mse = np.zeros(res[0]["sq"].shape, np.float64)
        for r in res:
            mse += r["sq"]
        mse /= n
        rmse_ic = np.stack([r["rmse"] for r in res])
        se = {k: np.stack([r["se"][i] for r in res]) for i, k in enumerate(ml.VAR_KEYS)}

        # gate against the scorer
        ref = ref_means(refsweep)
        rel = []
        for j, h in enumerate(ml.EXTRACT_LEADS_H):
            for c, name in enumerate(chan):
                if (name, h) in ref:
                    rv, rn = ref[(name, h)]
                    rel.append(abs(rmse_ic[:, j, c].mean() - rv) / rv)
        worst = max(rel) if rel else float("nan")
        own = sweep == refsweep
        print("%s: %d ICs, %d channels, %.0f s; gate vs %s/scores: %d values, max rel diff %.2e"
              % (arm, n, len(chan), time.time() - t0, refsweep, len(rel), worst))
        if a.limit_ics:
            print("  (smoke: %d of the ICs, so the gate is a report, not a check)" % n)
        elif not rel:
            print("ERROR K56_GATE_EMPTY %s: no (channel, lead) of the sweep is in %s/scores" % (arm, refsweep))
            bad += 1
            continue
        elif worst > GATE_TOL:
            if own:
                print("ERROR K56_GATE %s max rel diff %.3e vs its own scores" % (arm, worst))
                bad += 1
                continue
            print("WARN REROLL_DIFF %s: re-roll differs from the original sweep's scores by up to %.3e (rel)"
                  % (arm, worst))
        elif not own:
            print("REROLL_REPRODUCES %s: within the scores' 8-digit rounding (max rel %.1e)" % (arm, worst))
        np.savez(os.path.join(a.out_dir, "k56_%s.npz" % arm), mse_map=mse.astype(np.float32),
                 rmse_ic=rmse_ic, channels=np.array(chan, dtype="U16"),
                 lat=res[0]["lat"], lon=res[0]["lon"], map_leads_h=np.array(ml.MAP_LEADS_H),
                 extract_leads_h=np.array(ml.EXTRACT_LEADS_H), n_ic=n,
                 ic_files=np.array([os.path.basename(f) for f in files], dtype="U32"), sweep=sweep,
                 **{"se_" + k: v for k, v in se.items()})
        print("MAPS_K56_ARM_OK %s -> k56_%s.npz" % (arm, arm))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
