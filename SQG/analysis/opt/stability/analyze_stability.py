"""analyze_stability.py -- PREDICTIONS_stability.md's rule over stability_run.py's outputs (numpy only; re-runnable on a
login node).

  python3 analyze_stability.py --in <dir with run_w*.json/npz, archive.json/npz> --out <results dir>

Rule (fixed before the run):
  STABLE        every run: status COMPLETE (the full --tu), every sampled value finite, CFL_h max < 0.72 (A-SQG's abort).
  STATS window  tu in [10, 50] (the first 10 TU absorb the switch of scheme and step; >> the ~0.1 TU Lyapunov time),
                the per-TU samples (41 per run).
  S1            each run's window-mean SPE and VIE lie inside [min, max] over the 16 archived production states.
  S2            the pooled window-mean energy in each octave band of the shell spectrum E(k) (shell = round(|k|);
                bands [1,2), [2,4), ..., [512,1024), [1024,1366)) lies inside [min, max] over the 16 states.
  S3            no drift: per run, |least-squares slope of SPE vs tu over the window| x 40 TU <= the standard deviation
                of SPE over the 16 states.
  STATS_OK      S1 and S2 and S3. Verdict token: STAB_VERDICT stable=<bool> stats=<bool>.
Also reported: CFL_h median / 99th percentile / max, samples >= 0.60, ms/step (wall), each run's window means next to
the archive's mean and standard deviation.
"""
import argparse
import glob
import json
import math
import os

import numpy as np

WIN = (10.0, 50.0)
BANDS = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 512), (512, 1024),
         (1024, 1366)]
CFL_ABORT = 0.72


def band_energy(spec):
    spec = np.asarray(spec, dtype=float)
    return np.array([spec[a:b].sum() for a, b in BANDS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    arch = json.load(open(os.path.join(args.inp, "archive.json")))
    arch_spec = np.load(os.path.join(args.inp, "archive.npz"))["spectra"]
    a_spe = np.array([s["values"][0] for s in arch["states"]])
    a_vie = np.array([s["values"][1] for s in arch["states"]])
    a_band = np.array([band_energy(s) for s in arch_spec])
    out = {"archive": {"n": len(a_spe), "SPE_min_mean_max_std": [a_spe.min(), a_spe.mean(), a_spe.max(), a_spe.std()],
                       "VIE_min_mean_max_std": [a_vie.min(), a_vie.mean(), a_vie.max(), a_vie.std()],
                       "band_min": a_band.min(0).tolist(), "band_max": a_band.max(0).tolist(),
                       "cfl_DT_max": max(s["cfl_DT"] for s in arch["states"])}, "runs": {}}
    stable, s1, s3, pooled = True, True, True, []
    for p in sorted(glob.glob(os.path.join(args.inp, "run_w*.json"))):
        r = json.load(open(p))
        w = r["worker"]
        npz = np.load(p[:-5] + ".npz")
        ser = r.get("series", [])
        vals = np.array([s["values"] for s in ser]) if ser else np.zeros((0, 5))
        tu = np.array([s["tu"] for s in ser])
        finite = bool(np.all(np.isfinite(vals))) and all(math.isfinite(s["cfl"]) for s in ser)
        done = r.get("status") == "COMPLETE" and len(tu) and tu[-1] >= r["tu"] - 1e-6
        ok = bool(done and finite and r.get("cfl_max", 9) < CFL_ABORT)
        stable &= ok
        m = (tu >= WIN[0] - 1e-9) & (tu <= WIN[1] + 1e-9)
        rec = {"status": r.get("status"), "stable": ok, "cfl_max": r.get("cfl_max"), "cfl_quantiles": r.get("cfl_quantiles"),
               "n_cfl_warn": r.get("n_cfl_warn"), "n_cfl_samples": r.get("n_cfl_samples"),
               "ms_per_step_wall": r.get("ms_per_step_wall"), "state": r.get("state"), "n_window": int(m.sum())}
        if m.sum() >= 3:
            spe, vie = vals[m, 0], vals[m, 1]
            rec["SPE_mean"], rec["VIE_mean"] = float(spe.mean()), float(vie.mean())
            rec["S1"] = bool(a_spe.min() <= spe.mean() <= a_spe.max() and a_vie.min() <= vie.mean() <= a_vie.max())
            slope = float(np.polyfit(tu[m], spe, 1)[0])
            rec["SPE_slope_x40"] = slope * 40
            rec["S3"] = bool(abs(slope * 40) <= a_spe.std())
            s1 &= rec["S1"]
            s3 &= rec["S3"]
            pooled.extend(band_energy(npz["spectra"][i]) for i in np.nonzero(m)[0])
        else:
            rec["S1"] = rec["S3"] = False
            s1 = s3 = False
        out["runs"][w] = rec
        print(f"STAB_RUN w={w} status={rec['status']} stable={ok} cfl_max={rec['cfl_max']} q={rec['cfl_quantiles']} "
              f"warn={rec['n_cfl_warn']}/{rec['n_cfl_samples']} ms/step={rec['ms_per_step_wall']} "
              f"SPE={rec.get('SPE_mean')} VIE={rec.get('VIE_mean')} S1={rec['S1']} drift40={rec.get('SPE_slope_x40')} "
              f"S3={rec['S3']}")
    s2 = False
    if pooled:
        pb = np.mean(np.array(pooled), 0)
        inside = (pb >= a_band.min(0)) & (pb <= a_band.max(0))
        s2 = bool(np.all(inside))
        out["pooled_band_mean"] = pb.tolist()
        out["band_inside"] = inside.tolist()
        print("STAB_BANDS pooled/archive-mean " + " ".join(f"{x:.3f}" for x in pb / a_band.mean(0))
              + f" | inside [min,max]: {inside.tolist()}")
    out.update(stable=stable, S1=s1, S2=s2, S3=s3, stats_ok=bool(s1 and s2 and s3))
    print(f"STAB_ARCHIVE SPE {out['archive']['SPE_min_mean_max_std']} VIE {out['archive']['VIE_min_mean_max_std']} "
          f"CFL_DT max {out['archive']['cfl_DT_max']:.4f}")
    print(f"STAB_VERDICT stable={stable} stats={out['stats_ok']} (S1={s1} S2={s2} S3={s3})")
    with open(os.path.join(args.out, "stability_summary.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("STAB_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
