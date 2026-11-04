"""Fake fiveyear_<ARM>.npz / fiveyear_truth.npz with the real shapes, grid and land-sea mask, for a layout smoke of
plot_5yr.py before the real extraction exists. Values: a seasonal cycle, a per-checkpoint drift and run-to-run noise.
Usage: make_fake_5yr.py OUT_DIR      PASS = FAKE_5YR_OK
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fiveyear_lib as fl  # noqa: E402

MAPS = os.path.join(os.path.dirname(HERE), "maps")
BASE = {"TREFHT": 288.0, "PRECT": 3.4e-8, "U10": 7.0, "U_l17": 0.5, "Z3_l12": 5600.0, "PS": 98560.0, "TMQ": 25.0}
AMP = {"TREFHT": 2.0, "PRECT": 2e-9, "U10": 0.3, "U_l17": 0.4, "Z3_l12": 40.0, "PS": 30.0, "TMQ": 2.0}


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(1)
    ct = np.load(os.path.join(MAPS, "clim_truth.npz"))
    lat, lon, lsm = ct["lat"], ct["lon"], ct["lsm"]
    H, W = len(lat), len(lon)
    chans = list(fl.SERIES_KEYS) + ["X%02d" % i for i in range(5)]
    C = len(chans)
    mon = np.array([m for _, m in fl.MONTHS])
    season = np.cos(2 * np.pi * (mon - 1) / 12.0)
    tgm = np.stack([BASE.get(c, 1.0) + AMP.get(c, 0.1) * season for c in chans], axis=1)       # (63, C)
    tyear = np.stack([np.full(5, BASE.get(c, 1.0)) for c in chans], axis=1)                    # (5, C)
    zonal = np.cos(np.deg2rad(lat))
    tzm = np.stack([np.stack([BASE[k] * (0.9 + 0.1 * zonal) + AMP[k] * s * np.sin(np.deg2rad(lat)) for k in
                              fl.ZONAL_KEYS]) for s in season])                                # (63, Z, H)
    tmaps = np.stack([np.stack([BASE[k] * (0.9 + 0.1 * zonal[:, None] * np.cos(np.deg2rad(lon))[None])
                                for k in fl.MAP_KEYS]) for _ in fl.YEARS])                     # (5, M, H, W)
    np.savez(os.path.join(out, "fiveyear_truth.npz"), channels=np.array(chans), gm_month=tgm,
             cnt_month=np.full(63, 120), gm_year=tyear, n_year=np.full(5, 1459), zm_month=tzm.astype(np.float32),
             ann_maps=tmaps.astype(np.float32), lat=lat, lon=lon, lsm=lsm, months=np.array(fl.MONTHS),
             years=np.array(fl.YEARS), zonal_keys=np.array(fl.ZONAL_KEYS), map_keys=np.array(fl.MAP_KEYS))
    t_month = np.arange(63) / 12.0
    for k, arm in enumerate(("G4D16", "G4D36", "G4D43", "G4D56", "G4D63")):
        drift = 0.02 * (k + 1)
        n = 8
        scale = np.array([AMP.get(c, 0.1) for c in chans])
        gm = tgm[None] + scale[None, None] * (drift * t_month[None, :, None] + 0.15 * rng.normal(size=(n, 63, C)))
        gy = tyear[None] + scale[None, None] * (drift * (np.arange(5) + 1)[None, :, None]
                                               + 0.1 * rng.normal(size=(n, 5, C)))
        zs = np.array([AMP[z] for z in fl.ZONAL_KEYS])[None, None, :, None]
        zm = tzm[None] + zs * (drift * np.sin(np.deg2rad(3 * lat))[None, None, None] * (1 + t_month)[None, :, None, None]
                               + 0.2 * rng.normal(size=(n, 63, len(fl.ZONAL_KEYS), H)))
        ms = np.array([AMP[m] for m in fl.MAP_KEYS])[None, None, :, None, None]
        am = tmaps[None] + ms * (drift * np.sin(np.deg2rad(2 * lat))[:, None][None, None, None] * 5
                                 + 0.3 * rng.normal(size=(n, 5, len(fl.MAP_KEYS), H, W)))
        np.savez(os.path.join(out, "fiveyear_%s.npz" % arm), members=np.array(["member_%s%02d.nc" % (arm, i)
                                                                             for i in range(n)]),
                 channels=np.array(chans), gm_month=gm, cnt_month=np.full((n, 63), 120), gm_year=gy,
                 n_year=np.full((n, 5), 1459), zm_month=zm.astype(np.float32), ann_maps=am.astype(np.float32),
                 months=np.array(fl.MONTHS), years=np.array(fl.YEARS), zonal_keys=np.array(fl.ZONAL_KEYS),
                 map_keys=np.array(fl.MAP_KEYS), gate=np.array([0.0, 0.0]))
    print("FAKE_5YR_OK %s" % out)


if __name__ == "__main__":
    main()
