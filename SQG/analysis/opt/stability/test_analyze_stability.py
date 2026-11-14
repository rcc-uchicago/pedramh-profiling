"""test_analyze_stability.py -- analyze_stability.py on synthetic run / archive files (numpy). Prints
STAB_ANALYSIS_TEST_OK.

Case 1: 4 clean runs (SPE = archive mean + AR(1) noise with its window trend removed + slopes +-1e-4, +-5e-5, so the
mean slope is 0): STABLE, prod OK, stats PASS, 41-sample windows, energy budget closes.
Case 2: a common drift (2e-3 / TU, spread 1e-4) in all 4 runs: stats FAIL on S3 only (S1 holds: centred at tu 30).
Case 3: one run UNSTABLE_NONFINITE with 'nan' strings and no npz: UNSTABLE, no crash, stats INCONCLUSIVE.
Case 4: worker 3 missing: INCONCLUSIVE.
Case 5: a finite complete run with 3 samples >= 0.72: STABLE but prod EXCEEDED.
Case 6: slope_test (report only) on well-resolved AR(1) (phi 0.9, n 4000): g ~ (1 + phi) / (1 - phi), false
positives near 5 %; common_drift_test on fixed slopes.
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_stability as AS                                       # noqa: E402

NB = 2049
H = 8e-6
KMAX = 1365.0
SPE0, VIE0 = 1.0, 2.0


def ar1(rng, n, phi, sd):
    x = np.empty(n)
    x[0] = rng.normal(0, sd)
    e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def spectrum(scale=1.0):
    k = np.arange(NB, dtype=float)
    s = np.zeros(NB)
    s[1:] = scale * k[1:] ** -3.0
    return s


def write_archive(d, rng):
    states, spectra = [], []
    for i in range(16):
        f = 1 + 0.05 * rng.normal()
        states.append({"member": 1 + i // 4, "t": 100.0 * (1 + i % 4), "values": [SPE0 * f, VIE0 * f, 0.1, 0.1, 0.2],
                       "umax": 1.0, "cfl_DT": 0.3 + 0.01 * i})
        spectra.append(spectrum(f))
    json.dump({"states": states}, open(os.path.join(d, "archive.json"), "w"))
    np.savez(os.path.join(d, "archive.npz"), spectra=np.asarray(spectra))


def write_run(d, w, rng, drift=0.0, cfl_peak=0.65, n072=0, status="COMPLETE", nan=False, npz=True):
    frames = 250000
    fr = np.arange(12, frames + 1, 12)
    fr = np.union1d(fr, np.arange(5000, frames + 1, 5000))
    tu = fr * AS.H_FRAME
    noise = ar1(rng, len(tu), 0.95, 0.01)
    m = (fr % 12 == 0) & (tu >= 10.0) & (tu <= 50.0)                 # the analyzer's S3 samples: zero their trend
    noise -= np.polyfit(tu[m], noise[m], 1)[0] * (tu - tu[m].mean())
    spe = SPE0 + noise + drift * (tu - 30.0)
    vie = 2 * spe
    umax = (cfl_peak - 0.05) / (KMAX * H) + 0.0 * tu
    umax[len(tu) // 2] = cfl_peak / (KMAX * H)
    fine = np.column_stack([tu, spe, vie, 0.1 + 0 * tu, 0.1 + 0 * tu, 0.2 + 0 * tu, umax])
    per = np.nonzero(fr % 5000 == 0)[0]
    series = [{"tu_frame": float(tu[i]), "tu": float(tu[i]) + 2 * H, "values": [float(spe[i]), float(vie[i]), 0.1,
                                                                                  0.1, 0.2],
               "cfl": float(umax[i] * KMAX * H)} for i in per]
    if nan:
        series[-1]["values"] = ["nan"] * 5
        series[-1]["cfl"] = "nan"
    r = {"worker": w, "tu": 50.0, "status": status, "series": series, "cfl_max": cfl_peak, "n_cfl_samples": len(tu),
         "n_cfl_ge_060": int(cfl_peak >= 0.6), "n_cfl_ge_072": n072, "longest_run_ge_072": n072,
         "cfl_quantiles": {"0.5": cfl_peak - 0.05}, "ms_per_step_wall": 1.6, "kmax": KMAX, "state": {"h": H}}
    json.dump(r, open(os.path.join(d, f"run_w{w}.json"), "w"))
    if npz:
        np.savez(os.path.join(d, f"run_w{w}.npz"), spectra=np.asarray([spectrum() for _ in series]),
                 subs=np.zeros((len(series), 2, 2), np.float32), fine=fine)


def run(d):
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        out = AS.analyze(d, [0, 1, 2, 3])
    return out, buf.getvalue()


def case(fn):
    d = tempfile.mkdtemp(prefix="stab_test_")
    try:
        rng = np.random.default_rng(7)
        write_archive(d, rng)
        fn(d, rng)
        return run(d)
    finally:
        shutil.rmtree(d)


def main():
    slopes = (1e-4, -1e-4, 5e-5, -5e-5)

    def clean(d, rng, skip=()):
        for w in range(4):
            if w not in skip:
                write_run(d, w, rng, drift=slopes[w])

    out, log = case(clean)
    assert out["stability"] == "STABLE" and out["prod_policy"] == "OK", out
    assert out["stats"] == "PASS", (out["S1"], out["S2"], out["S3"], [r.get("S3") for r in out["runs"].values()])
    assert all(r["n_window"] == AS.N_WIN for r in out["runs"].values())
    assert "STAB_VERDICT stability=STABLE prod_policy=OK stats=PASS" in log, log
    b = out["runs"]["0"]["energy_budget"]
    assert abs(b["int_P_minus_eps"]) < 1e-9, b                     # P - eps_nu - eps_mu = 0 here

    def drifted(d, rng):
        for w in range(4):
            write_run(d, w, rng, drift=2e-3 + slopes[w])
    out, log = case(drifted)
    assert out["stats"] == "FAIL" and out["S3"] is False and out["S1"] is True and out["S2"] is True, out
    assert abs(out["S3_common_drift"]["mean"] - 2e-3) < 1e-9 and "STAB_DRIFT" in log
    assert out["runs"]["2"]["S3_run_2se_report"]["pass"] is False

    def nonfinite(d, rng):
        clean(d, rng, skip=(1,))
        write_run(d, 1, rng, status="UNSTABLE_NONFINITE", nan=True, npz=False)
    out, log = case(nonfinite)
    assert out["stability"] == "UNSTABLE" and out["stats"] == "INCONCLUSIVE", out
    assert "1:UNSTABLE_NONFINITE" in log

    def missing(d, rng):
        clean(d, rng, skip=(3,))
    out, log = case(missing)
    assert out["stability"] == "INCONCLUSIVE" and out["prod_policy"] == "INCONCLUSIVE", out
    assert out["run_status"]["3"] == "MISSING" and out["stats"] == "INCONCLUSIVE"

    def exceeded(d, rng):
        clean(d, rng, skip=(0,))
        write_run(d, 0, rng, cfl_peak=0.75, n072=3)
    out, log = case(exceeded)
    assert out["stability"] == "STABLE" and out["prod_policy"] == "EXCEEDED", out
    assert "STAB_PROD_POLICY_OK pass=False" in log

    rng = np.random.default_rng(11)
    phi, n = 0.9, 4000
    t = np.arange(n, dtype=float)
    gs, fp = [], 0
    for _ in range(200):
        r = AS.slope_test(t, ar1(rng, n, phi, 1.0))
        gs.append(r["g"])
        fp += not r["pass"]
    g_true = (1 + phi) / (1 - phi)
    assert 0.7 * g_true < np.median(gs) < 1.3 * g_true, (np.median(gs), g_true)
    assert 0.01 <= fp / 200 <= 0.12, fp
    assert AS.common_drift_test([1.0, -1.0, 0.5, -0.5])["pass"] is True
    assert AS.common_drift_test([1.0, 1.1, 0.9, 1.05])["pass"] is False
    print(f"STAB_ANALYSIS_TEST_OK (AR1 g median {np.median(gs):.1f} vs {g_true:.1f}; S3 false-positive {fp}/200)")


if __name__ == "__main__":
    main()
