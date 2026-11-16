"""calibrate_s3_2se.py -- false-positive rate of analyze_stability.slope_test (|slope| <= 2 SE, SE inflated by the
estimated g) on AR(1) noise with no drift, at the real run's sample count (40 TU / 0.0024 TU = 16,667) and integrated
autocorrelation times tau_I = 0.5, 1, 2, 3 TU. Run from the repo root: python3 <this> [reps=300].

Result (2026-11-12, 300 reps, seed 3), why the per-run test is report-only:
  tau_I 0.5: g_med/g_true 0.84, false positives 6.3 %;  1: 0.75, 8.0 %;  2: 0.57, 20.3 %;  3: 0.48, 18.3 %.
"""
import sys

import numpy as np

sys.path.insert(0, "SQG/analysis/opt/stability")
import analyze_stability as AS                                       # noqa: E402

rng = np.random.default_rng(3)
dt = 0.0024
n = int(round(40 / dt))
t = 10 + dt * np.arange(n)
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def ar1(phi, n):
    e = rng.normal(0, np.sqrt(1 - phi ** 2), n)
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


for tau in (0.5, 1.0, 2.0, 3.0):
    g_true = 2 * tau / dt                       # g = 1 + 2 sum rho = 2 tau_I / dt
    phi = (g_true - 1) / (g_true + 1)
    fp, gs = 0, []
    for _ in range(REPS):
        r = AS.slope_test(t, ar1(phi, n))
        fp += not r["pass"]
        gs.append(r["g"])
    print(f"tau_I={tau} TU  N_eff_true={n / g_true:.1f}  g_med/g_true={np.median(gs) / g_true:.2f}  "
          f"false-positive={fp}/{REPS}={fp / REPS:.3f}  P(all 4 pass)={(1 - fp / REPS) ** 4:.2f}", flush=True)
