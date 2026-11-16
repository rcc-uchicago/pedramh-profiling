"""analyze_stability.py -- PREDICTIONS_stability.md's rule over stability_run.py's outputs (numpy only; re-runnable on a
login node). Revised after the pre-submission review (wf_8c6d4a2d-cde), before any run.

  python3 analyze_stability.py --in <dir with run_w*.json/npz, archive.json/npz> --out <results dir> [--workers 0,1,2,3]

Run status (each expected worker; a missing run_w<w>.json is MISSING):
  COMPLETE                                  the full --tu, every sample finite, CFL_h < 1.0
  UNSTABLE_NONFINITE / UNSTABLE_CFL_HARD    the run's own stops: a non-finite sample / CFL_h >= 1.0
  anything else (MISSING, STARTING, GATE_FAIL, RUNNING, CRASHED, DEADLINE) is INCONCLUSIVE: no evidence either way.
Rule (fixed before the run):
  STABILITY      STABLE if every expected run is COMPLETE; UNSTABLE if any run is UNSTABLE_*; else INCONCLUSIVE.
  PROD_POLICY    OK if every run is COMPLETE with CFL_h max < 0.72 (A-SQG_data.py's abort; o1.2 itself has none);
                 EXCEEDED if any run has a sample >= 0.72; else INCONCLUSIVE. The number of samples >= 0.60 / 0.72
                 and the longest run of consecutive samples >= 0.72 are reported.
  STATS window   tu_frame in [10, 50] on the per-TU samples (41 per complete run; asserted).
  S1             each run's window-mean SPE and VIE lie inside [min, max] over the 16 archived production states.
  S2             the pooled window-mean energy in each octave band of E(k) (shell = round(|k|); bands [1,2), ...,
                 [512,1024), [1024,1366)) lies inside [min, max] over the 16 states.
  S3             no common drift: b_w = the OLS slope of SPE vs tu on run w's fine series (every 12 frames, window
                 [10, 50]); |mean b| <= t_{0.975,3} sd(b, ddof 1) / sqrt(4), t_{0.975,3} = 3.182446. The 4 runs are
                 independent, so the test needs no model of SPE's autocorrelation (simulated on AR(1) at the run's
                 n = 16,667, tau_I 0.5-3 TU: false positives 5.0-7.7 %; power 0.31-0.40 / 0.76-0.85 / 1.00 at a common
                 drift of 1 / 2 / 4 sd(b)). A scheme-induced drift would move every run the same way.
  STATS          PASS if S1, S2, S3 all hold over 4 complete runs; FAIL if any is False over 4 complete runs; else
                 INCONCLUSIVE.
Report only (not part of any verdict): per run, |slope| <= 2 SE with SE inflated by g = 1 + 2 sum rho_k of the
detrended residuals (truncated at the first rho_k <= 0) -- anti-conservative at this length (simulated false
positives 6-20 % per run at tau_I 0.5-3 TU: the ACF estimate is biased low); S3 with the old yardstick (|slope| x 40
<= the archive's SPE std, ddof = 1);
the energy budget over the window, SPE(50) - SPE(10) vs the integral of P - eps_nu - eps_mu (trapezoid on the fine
series), as a fraction of the integral of P; per-shell ratios run / archive mean for k > 500 (min, max per 100-shell
block); an exponential tail fit of CFL_h above its 90th percentile, extrapolated to P(sample >= 0.72); ms/step (wall);
the gate at h.
Tokens: STAB_RUN, STAB_BANDS, STAB_FINITE, STAB_PROD_POLICY_OK, STAB_VERDICT, STAB_ANALYSIS_DONE.
"""
import argparse
import json
import math
import os

import numpy as np

WIN = (10.0, 50.0)
N_WIN = 41
BANDS = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 512), (512, 1024),
         (1024, 1366)]
CFL_ABORT, CFL_WARNING, CFL_HARD = 0.72, 0.60, 1.0
H_FRAME, SAMPLE_FRAMES = 2e-4, 12
UNSTABLE = ("UNSTABLE_NONFINITE", "UNSTABLE_CFL_HARD")
T975_3 = 3.182446                                                    # Student t, 0.975 quantile, 3 dof


def num(x):
    """JSON number or the driver's 'nan' / 'inf' strings -> float (anything else -> nan)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def band_energy(spec):
    spec = np.asarray(spec, dtype=float)
    return np.array([spec[a:b].sum() for a, b in BANDS])


def tau_factor(x):
    """g = 1 + 2 sum_{k>=1} rho_k of x (mean removed), summed up to the first rho_k <= 0 (initial positive sequence)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    if n < 4 or not np.any(x):
        return 1.0
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    rho = acf / acf[0]
    g = 1.0
    for k in range(1, n):
        if rho[k] <= 0:
            break
        g += 2.0 * rho[k]
    return float(g)


def slope_test(t, y):
    """OLS slope of y vs t, its autocorrelation-inflated SE, g, N_eff; passes if |slope| <= 2 SE."""
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    tc = t - t.mean()
    sxx = float(np.sum(tc ** 2))
    slope = float(np.sum(tc * (y - y.mean())) / sxx)
    resid = y - y.mean() - slope * tc
    g = tau_factor(resid)
    sigma2 = float(np.sum(resid ** 2) / max(len(y) - 2, 1))
    se = math.sqrt(sigma2 * g / sxx)
    return {"slope": slope, "se": se, "g": g, "n": len(y), "n_eff": len(y) / g, "pass": bool(abs(slope) <= 2 * se)}


def common_drift_test(slopes):
    """|mean b| <= t_{0.975,3} sd(b) / sqrt(4) over 4 independent runs' slopes."""
    b = np.asarray(slopes, dtype=float)
    if len(b) != 4:
        raise ValueError(f"common_drift_test is set up for 4 runs, got {len(b)}")
    mean, sd = float(b.mean()), float(b.std(ddof=1))
    half = T975_3 * sd / math.sqrt(len(b))
    return {"slopes": b.tolist(), "mean": mean, "sd": sd, "halfwidth": half, "pass": bool(abs(mean) <= half)}


def trapezoid(y, t):
    y, t = np.asarray(y, dtype=float), np.asarray(t, dtype=float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(t)))


def tail_fit(cfl, level=CFL_ABORT):
    """Exponential tail above the 90th percentile u: P(X >= level) ~ 0.1 exp(-(level - u) / mean(X - u | X > u))."""
    c = np.asarray(cfl, dtype=float)
    c = c[np.isfinite(c)]
    if len(c) < 100:
        return None
    u = float(np.quantile(c, 0.9))
    ex = c[c > u] - u
    if not len(ex) or ex.mean() <= 0:
        return None
    p = 0.1 * math.exp(-(level - u) / float(ex.mean()))
    return {"u90": u, "mean_excess": float(ex.mean()), "p_sample_ge": p, "level": level}


def load_run(inp, w):
    """(json dict or None, npz dict or None) for worker w."""
    pj, pn = os.path.join(inp, f"run_w{w}.json"), os.path.join(inp, f"run_w{w}.npz")
    r = None
    if os.path.exists(pj):
        try:
            r = json.load(open(pj))
        except (OSError, ValueError):
            r = {"status": "UNREADABLE"}
    z = None
    if os.path.exists(pn):
        try:
            with np.load(pn) as f:
                z = {k: f[k] for k in f.files}
        except (OSError, ValueError):
            z = None
    return r, z


def classify(r):
    """'COMPLETE' / 'UNSTABLE' / 'INCONCLUSIVE' and the raw status."""
    st = "MISSING" if r is None else str(r.get("status", "MISSING"))
    if st in UNSTABLE:
        return "UNSTABLE", st
    if st != "COMPLETE":
        return "INCONCLUSIVE", st
    ser = r.get("series", [])
    vals = [num(v) for s in ser for v in s.get("values", [])] + [num(s.get("cfl")) for s in ser]
    if not ser or not all(math.isfinite(v) for v in vals):
        return "UNSTABLE", "UNSTABLE_NONFINITE"
    if num(r.get("cfl_max")) >= CFL_HARD:
        return "UNSTABLE", "UNSTABLE_CFL_HARD"
    if round(num(ser[-1].get("tu_frame")), 6) < num(r.get("tu")) - 1e-6:
        return "INCONCLUSIVE", "SHORT"
    return "COMPLETE", st


def analyze(inp, workers):
    arch = json.load(open(os.path.join(inp, "archive.json")))
    with np.load(os.path.join(inp, "archive.npz")) as f:
        arch_spec = np.asarray(f["spectra"], dtype=float)
    a_val = np.array([[num(v) for v in s["values"]] for s in arch["states"]])
    a_spe, a_vie = a_val[:, 0], a_val[:, 1]
    a_band = np.array([band_energy(s) for s in arch_spec])
    a_shell = arch_spec.mean(0)
    out = {"archive": {"n": len(a_spe), "SPE_min_mean_max_std_ddof1": [a_spe.min(), a_spe.mean(), a_spe.max(),
                                                                        a_spe.std(ddof=1)],
                       "VIE_min_mean_max_std_ddof1": [a_vie.min(), a_vie.mean(), a_vie.max(), a_vie.std(ddof=1)],
                       "band_min": a_band.min(0).tolist(), "band_max": a_band.max(0).tolist(),
                       "cfl_DT_max": max(num(s["cfl_DT"]) for s in arch["states"])}, "runs": {}}
    classes, s1s, s3s, pooled, shells, n_win_ok, exceeded = {}, [], [], [], [], 0, False
    for w in workers:
        r, z = load_run(inp, w)
        cls, st = classify(r)
        classes[w] = (cls, st)
        r = r or {}
        rec = {"class": cls, "status": st, "cfl_max": num(r.get("cfl_max")), "cfl_quantiles": r.get("cfl_quantiles"),
               "n_cfl_samples": r.get("n_cfl_samples"), "n_cfl_ge_060": r.get("n_cfl_ge_060"),
               "n_cfl_ge_072": r.get("n_cfl_ge_072"), "longest_run_ge_072": r.get("longest_run_ge_072"),
               "ms_per_step_wall": r.get("ms_per_step_wall"), "state": r.get("state"), "gate_h": r.get("gate_h"),
               "full_qh_path": r.get("full_qh_path"), "npz": z is not None}
        if (r.get("n_cfl_ge_072") or 0) > 0 or rec["cfl_max"] >= CFL_ABORT:
            exceeded = True
        ser = r.get("series", [])
        if cls == "COMPLETE" and z is not None:
            tu = np.array([round(num(s["tu_frame"]), 6) for s in ser])
            vals = np.array([[num(v) for v in s["values"]] for s in ser])
            m = (tu >= WIN[0]) & (tu <= WIN[1])
            rec["n_window"] = int(m.sum())
            spec = np.asarray(z.get("spectra", np.zeros((0, 1))), dtype=float)
            if rec["n_window"] == N_WIN and len(spec) == len(ser):
                n_win_ok += 1
                spe, vie = vals[m, 0], vals[m, 1]
                rec["SPE_mean"], rec["VIE_mean"] = float(spe.mean()), float(vie.mean())
                rec["S1"] = bool(a_spe.min() <= spe.mean() <= a_spe.max() and a_vie.min() <= vie.mean() <= a_vie.max())
                s1s.append(rec["S1"])
                pooled.extend(band_energy(spec[i]) for i in np.nonzero(m)[0])
                shells.append(spec[m].mean(0))
                old = float(np.polyfit(tu[m], spe, 1)[0])
                rec["S3_report_archive_std"] = {"slope_x40": old * 40, "pass": bool(abs(old * 40) <= a_spe.std(ddof=1))}
                fine = np.asarray(z.get("fine", np.zeros((0, 7))), dtype=float).reshape(-1, 7)
                fr = np.rint(fine[:, 0] / H_FRAME).astype(np.int64)
                fm = (fr % SAMPLE_FRAMES == 0) & (fine[:, 0] >= WIN[0] - 1e-9) & (fine[:, 0] <= WIN[1] + 1e-9)
                if fm.sum() >= 100 and np.all(np.isfinite(fine[fm, 1])):
                    rec["S3_run_2se_report"] = slope_test(fine[fm, 0], fine[fm, 1])
                    s3s.append(rec["S3_run_2se_report"]["slope"])
                    tf, sp = fine[fm, 0], fine[fm, 1]
                    inj = trapezoid(fine[fm, 5], tf)
                    net = trapezoid(fine[fm, 5] - fine[fm, 3] - fine[fm, 4], tf)
                    rec["energy_budget"] = {"dSPE": float(sp[-1] - sp[0]), "int_P_minus_eps": float(net),
                                            "int_P": float(inj), "residual_over_int_P": float((sp[-1] - sp[0] - net) / inj)}
                    kmax = r.get("kmax")
                    h = num((r.get("state") or {}).get("h"))
                    if kmax:
                        rec["cfl_tail"] = tail_fit(fine[fine[:, 0] >= WIN[0], 6] * float(kmax) * h)
                else:
                    rec["S3_run_2se_report"] = None
        out["runs"][str(w)] = rec
        s3 = rec.get("S3_run_2se_report") or {}
        print(f"STAB_RUN w={w} {cls}/{st} cfl_max={rec['cfl_max']:.4f} q={rec['cfl_quantiles']} "
              f">=0.60:{rec['n_cfl_ge_060']} >=0.72:{rec['n_cfl_ge_072']}/{rec['n_cfl_samples']} "
              f"ms/step={rec['ms_per_step_wall']} SPE={rec.get('SPE_mean')} VIE={rec.get('VIE_mean')} "
              f"S1={rec.get('S1')} slope={s3.get('slope')} (2SE report: {s3.get('pass')}, se {s3.get('se')}, "
              f"n_eff {s3.get('n_eff')}) budget={(rec.get('energy_budget') or {}).get('residual_over_int_P')}")
    n = len(workers)
    cl = [c for c, _ in classes.values()]
    stability = "STABLE" if cl.count("COMPLETE") == n else ("UNSTABLE" if "UNSTABLE" in cl else "INCONCLUSIVE")
    if exceeded:
        prod = "EXCEEDED"
    elif cl.count("COMPLETE") == n:
        prod = "OK"
    else:
        prod = "INCONCLUSIVE"
    s2 = None
    if pooled:
        pb = np.mean(np.array(pooled), 0)
        inside = (pb >= a_band.min(0)) & (pb <= a_band.max(0))
        s2 = bool(np.all(inside))
        out["pooled_band_mean"] = pb.tolist()
        out["band_inside"] = inside.tolist()
        sh = np.mean(np.array(shells), 0)
        ratio = sh / np.where(a_shell > 0, a_shell, np.nan)
        out["high_k_shell_ratio"] = {f"{a}-{a + 99}": [float(np.nanmin(ratio[a:a + 100])),
                                                        float(np.nanmax(ratio[a:a + 100]))]
                                     for a in range(500, min(len(ratio), 1366), 100)}
        print("STAB_BANDS pooled/archive-mean " + " ".join(f"{x:.3f}" for x in pb / a_band.mean(0))
              + f" | inside [min,max]: {inside.tolist()}")
    s1 = all(s1s) if s1s else None
    complete_stats = n_win_ok == n == 4 and len(s1s) == n and len(s3s) == n and s2 is not None
    s3a = None
    if len(s3s) == 4:
        d = out["S3_common_drift"] = common_drift_test(s3s)
        s3a = d["pass"]
        print(f"STAB_DRIFT mean_slope={d['mean']:.4g} sd={d['sd']:.4g} halfwidth={d['halfwidth']:.4g} "
              f"x40TU: mean={d['mean'] * 40:.4g} vs archive SPE std {a_spe.std(ddof=1):.4g} pass={d['pass']}")
    if complete_stats:
        stats = "PASS" if (s1 and s2 and s3a) else "FAIL"
    else:
        stats = "INCONCLUSIVE"
    out.update(workers=list(workers), stability=stability, prod_policy=prod, S1=s1, S2=s2, S3=s3a, stats=stats,
               run_status={str(w): st for w, (_, st) in classes.items()})
    print(f"STAB_ARCHIVE SPE {out['archive']['SPE_min_mean_max_std_ddof1']} "
          f"VIE {out['archive']['VIE_min_mean_max_std_ddof1']} CFL_DT max {out['archive']['cfl_DT_max']:.4f}")
    print(f"STAB_FINITE pass={stability == 'STABLE'} ({stability})")
    print(f"STAB_PROD_POLICY_OK pass={prod == 'OK'} ({prod})")
    runs = ",".join(f"{w}:{st}" for w, (_, st) in classes.items())
    print(f"STAB_VERDICT stability={stability} prod_policy={prod} stats={stats} (S1={s1} S2={s2} S3={s3a}) "
          f"runs={runs}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", default="0,1,2,3")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    out = analyze(args.inp, [int(x) for x in args.workers.split(",") if x != ""])
    with open(os.path.join(args.out, "stability_summary.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("STAB_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
