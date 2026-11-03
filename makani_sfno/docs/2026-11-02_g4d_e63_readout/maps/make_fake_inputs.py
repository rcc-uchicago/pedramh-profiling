"""Synthetic npz inputs with the real files' keys and plausible magnitudes, to exercise plot_maps.py before any
real extraction exists (smoke only; the figures it yields are meaningless). Writes into --out-dir."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maps_lib as ml  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", required=True)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
rng = np.random.default_rng(0)
lat = np.linspace(89.5, -89.5, 180)
lon = np.arange(0.5, 360.0, 1.0)
LA, LO = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")


def smooth(scale, k=6):
    f = np.zeros_like(LA)
    for _ in range(k):
        m, n = rng.integers(1, 6, 2)
        f += rng.normal() * np.cos(m * LA + rng.uniform(0, 6)) * np.cos(n * LO + rng.uniform(0, 6))
    return scale * f / k


lsm = (smooth(1.0, 4) + 0.15 * np.cos(LA) > 0.05).astype(np.float32)
size = {"TREFHT": 1.5, "PRECT": 4e-8, "U10": 1.5, "U_l17": 2.0, "Z3_l12": 30.0}
chans = ["PS", "TREFHT", "U10", "PRECT", "U_l17", "Z3_l12"]
ics = np.array(["2048_ic%03d.nc" % i for i in range(12)] + ["2049_ic%03d.nc" % i for i in range(12)])
for arm, skill in (("A243", 1.0), ("G4D16", 1.15), ("G4D36", 1.08), ("G4D43", 1.06), ("G4D56", 1.03), ("G4D63", 1.0)):
    n_ic, nm, ne = 24, len(ml.MAP_LEADS_H), len(ml.EXTRACT_LEADS_H)
    grow = np.array([1 + h / 60.0 for h in ml.MAP_LEADS_H])
    se = {}
    for v in ml.VAR_KEYS:
        base = (np.abs(smooth(size[v])) + 0.3 * size[v]) * skill
        e = base[None, None] * grow[None, :, None, None] * rng.normal(1, 0.3, (n_ic, nm, 1, 1))
        e = e * (1 + 0.2 * rng.standard_normal((n_ic, nm, 180, 360)))
        se[v] = (e ** 2).astype(np.float32)
    mse = np.zeros((nm, len(chans), 180, 360), np.float32)
    for c, v in enumerate(chans):
        if v in se:
            mse[:, c] = se[v].mean(axis=0)
    gr = np.array([1 + h / 80.0 for h in ml.EXTRACT_LEADS_H])
    rmse_ic = np.stack([[[size.get(v, 100.0) * skill * g * rng.normal(1, 0.05) for v in chans] for g in gr]
                        for _ in range(n_ic)])
    np.savez(os.path.join(a.out_dir, "k56_%s.npz" % arm), mse_map=mse, rmse_ic=rmse_ic,
             channels=np.array(chans, dtype="U16"), lat=lat, lon=lon, map_leads_h=np.array(ml.MAP_LEADS_H),
             extract_leads_h=np.array(ml.EXTRACT_LEADS_H), n_ic=n_ic, ic_files=ics, sweep="fake",
             **{"se_" + k: v for k, v in se.items()})
truth = np.stack([smooth(size[v] * 10) for v in ml.VAR_KEYS]).astype(np.float32)
np.savez(os.path.join(a.out_dir, "clim_truth.npz"), truth_maps=truth, var_keys=np.array(ml.VAR_KEYS), lsm=lsm,
         lat=lat, lon=lon, window=np.array([2045, 2047]))
for arm in ("G4D16", "G4D36", "G4D43", "G4D56", "G4D63"):
    bias = np.stack([smooth(size[v]) for v in ml.VAR_KEYS])
    mm = truth[None] + bias[None] + np.stack([[smooth(size[v] * 0.4) for v in ml.VAR_KEYS] for _ in range(8)])
    np.savez(os.path.join(a.out_dir, "clim_%s.npz" % arm), member_maps=mm.astype(np.float32),
             var_keys=np.array(ml.VAR_KEYS), members=np.array(["member_%s%02d.nc" % (arm, i) for i in range(8)]),
             window=np.array([2045, 2047]))
print("FAKE_INPUTS_OK %s" % a.out_dir)
