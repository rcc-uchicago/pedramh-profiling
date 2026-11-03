"""Unit tests of maps_lib on synthetic fields with known answers. PASS = TEST_MAPS_LIB_OK.

Runs with the login node's system python3 + numpy (no venv), and again inside the extraction job.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maps_lib as ml  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else ": " + detail))
    if not ok:
        fails.append(name)


# 1. Isothermal hydrostatic column: z = H ln(PS / p) is exactly linear in ln p, so the interpolation is exact.
H = 7000.0
hyam = np.array([0.005, 0.03, 0.10, 0.20, 0.25, 0.20, 0.10, 0.02, 0.0])
hybm = np.array([0.0, 0.0, 0.05, 0.15, 0.30, 0.50, 0.75, 0.93, 0.998])
p0 = 1.0e5
ps = np.array([[101325.0, 85000.0], [60000.0, 45000.0]])        # the last column is above 500 hPa
p = ml.level_pressure(ps, hyam, hybm, p0)
check("pressure increases downward", bool(np.all(np.diff(p, axis=0) > 0)))
z = H * np.log(ps[None] / p)
z5 = ml.height_at_pressure(z, p)
want = H * np.log(ps / 50000.0)
okc = np.isfinite(z5[:, :][[0, 0, 1], [0, 1, 0]])
check("Z500 exact where 500 hPa is inside the column",
      bool(okc.all()) and np.allclose(z5[[0, 0, 1], [0, 1, 0]], want[[0, 0, 1], [0, 1, 0]], rtol=0, atol=1e-9),
      "%s vs %s" % (z5, want))
check("Z500 NaN where PS < 500 hPa (no extrapolation)", bool(np.isnan(z5[1, 1])), str(z5[1, 1]))

# 2. A target exactly on a level returns that level's height.
p_on = ml.level_pressure(np.array([[100000.0]]), hyam, hybm, p0)
z_on = np.arange(len(hyam), 0, -1, dtype=float)[:, None, None] * 1000.0
k = 4
check("target on a level", np.isclose(ml.height_at_pressure(z_on, p_on, target=p_on[k, 0, 0])[0, 0], z_on[k, 0, 0]))

# 3. z500_from_channels picks levels by name regardless of order.
L = len(ml.Z3_LEVELS)
hy18 = np.linspace(0.004, 0.0, L)
hb18 = np.linspace(0.0, 0.998, L)
ps2 = np.full((2, 3), 98000.0)
p18 = ml.level_pressure(ps2, hy18, hb18, p0)
z18 = H * np.log(ps2[None] / p18)
names = ["PS"] + list(ml.Z3_LEVELS)
stack = np.concatenate([ps2[None], z18])
perm = np.random.RandomState(0).permutation(len(names))
z5c = ml.z500_from_channels(stack[perm], [names[i] for i in perm], hy18, hb18, p0)
check("z500_from_channels, shuffled channels", np.allclose(z5c, H * np.log(98000.0 / 50000.0), atol=1e-9))

# 4. Area weights and area mean: a field equal to cos-weighted constant.
lat = np.linspace(89.5, -89.5, 180)
w = ml.lat_weights(lat)
check("lat weights sum to 1", np.isclose(w.sum(), 1.0))
f = np.ones((180, 360)) * 3.0
f[0, 0] = np.nan
check("area mean ignores NaN, renormalises", np.isclose(ml.area_mean(f, w), 3.0))
g = np.where(lat[:, None] > 0, 1.0, 0.0) * np.ones((1, 360))
check("area mean of a hemisphere indicator = 0.5", np.isclose(ml.area_mean(g, w), 0.5))

# 5. Benjamini-Hochberg against a hand-worked example (alpha 0.10, m 10).
pv = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.061, 0.074, 0.205, 0.212, 0.216])
# thresholds 0.01k. Rank 3 fails on its own (0.039 > 0.03) but the step-up keeps it, because rank 5 passes
# (0.042 <= 0.05) and rank 6 does not (0.061 > 0.06): the first five are rejected.
check("BH hand example (step-up)", ml.fdr_mask(pv, 0.10).tolist() == [True] * 5 + [False] * 5,
      str(ml.fdr_mask(pv, 0.10)))
check("BH rejects at equality p == alpha k / m", bool(ml.fdr_mask(np.array([0.05, 0.9]), 0.10)[0]))
check("BH NaN never rejected", not ml.fdr_mask(np.array([np.nan, 0.0]), 0.1)[0])
check("BH all null", not ml.fdr_mask(np.full(100, 0.9), 0.1).any())

# 6. Longitude re-centring.
lon = np.arange(0.5, 360.0, 1.0)
fld = np.tile(lon, (2, 1))
r, lc = ml.roll_lon(fld, lon)
check("roll_lon order and values", lc[0] == -179.5 and lc[-1] == 179.5 and r[0, 0] == 180.5 and r[0, 180] == 0.5)

# 7. Units table: PRECT m/s -> mm/day.
fac = dict((k, f) for k, _, _, f in ml.VARS)
check("PRECT factor", np.isclose(2.89413e-08 * fac["PRECT"], 2.5005, atol=1e-3))
check("map leads in extract leads", set(ml.MAP_LEADS_H) <= set(ml.EXTRACT_LEADS_H))

print("TEST_MAPS_LIB_OK" if not fails else "ERROR TEST_MAPS_LIB %s" % fails)
sys.exit(1 if fails else 0)
