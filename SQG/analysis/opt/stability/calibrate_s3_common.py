"""calibrate_s3_common.py -- false-positive rate of analyze_stability's S3 (the between-run common-drift test
|mean b| <= t_{0.975,3} sd(b) / 2) on 4 independent AR(1) runs with no drift, at the real sample count (16,667 per
run), tau_I = 0.5, 1, 2, 3 TU; and its power for a common drift of 1 / 2 / 4 sd(one slope).
Run from the repo root: python3 <this> [reps=300].

Result (2026-11-12, 300 reps, seed 5):
  tau_I 0.5: false positives 5.7 %, power 0.33 / 0.76 / 1.00;  1: 6.0 %, 0.34 / 0.80 / 1.00;
  tau_I 2:   5.0 %, 0.31 / 0.79 / 1.00;                        3: 7.7 %, 0.40 / 0.85 / 1.00   (MC error ~ +-1.3 %).
"""
import sys
from math import sqrt

import numpy as np

rng = np.random.default_rng(5)
dt = 0.0024
n = int(round(40 / dt))
t = 10 + dt * np.arange(n)
tc = t - t.mean()
sxx = float(np.sum(tc ** 2))
T975_3 = 3.182446
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def ar1(phi):
    e = rng.normal(0, np.sqrt(1 - phi ** 2), n)
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


for tau in (0.5, 1.0, 2.0, 3.0):
    g = 2 * tau / dt
    phi = (g - 1) / (g + 1)
    slopes = np.array([[float(np.sum(tc * ar1(phi)) / sxx) for _ in range(4)] for _ in range(REPS)])
    sd_b = slopes.std(ddof=1, axis=1)
    se_true = sqrt(g / sxx)                                       # sd of one slope
    fp = int(np.sum(np.abs(slopes.mean(1)) > T975_3 * sd_b / 2))
    pw = []
    for k in (1.0, 2.0, 4.0):                                     # common drift = k * sd(one slope)
        s2 = slopes + k * se_true
        pw.append(float(np.mean(np.abs(s2.mean(1)) > T975_3 * s2.std(ddof=1, axis=1) / 2)))
    print(f"tau_I={tau}  false-positive={fp}/{REPS}={fp / REPS:.3f}  power at drift 1/2/4 x sd(b): "
          f"{pw[0]:.2f}/{pw[1]:.2f}/{pw[2]:.2f}  sd(b)*40={se_true * 40:.3g} (units of the noise sd)", flush=True)
