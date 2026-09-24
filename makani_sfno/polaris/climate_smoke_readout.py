#!/usr/bin/env python3
"""Read out a climate-driver member NetCDF (G4 numbers; torch-free, netCDF4).

Per member: status, earliest channels past 3σ (anomaly RMS vs time_means, in
global_stds), median crossings, and the global-mean drift of selected channels
in physical units (value at lead L minus value at lead 1).

    python polaris/climate_smoke_readout.py member_A.nc member_B.nc
"""
from __future__ import annotations

import argparse
import json

import netCDF4
import numpy as np

DRIFT_CHANNELS = ("Z3_l17", "Z3_l10", "T_l17", "PS", "TREFHT")
LEADS = (120, 240, 368, 480, 594, 600)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("nc", nargs="+")
    p.add_argument("--top", type=int, default=8)
    args = p.parse_args()
    for path in args.nc:
        with netCDF4.Dataset(path) as f:
            names = [str(x) for x in f["channel"][:]]
            metrics = [str(x) for x in f["metric"][:]]
            fb = np.array(f["first_bad_step"][:])          # (metric, thr, C)
            mc = np.array(f["median_cross_step"][:])
            gm = np.array(f["global_mean"][:])             # (T, C) physical
            n = gm.shape[0]
            a = {k: f.getncattr(k) for k in f.ncattrs()}
        print(f"== {a['member_id']} ckpt_epoch={a['ckpt_epoch']} status={a['status']} "
              f"steps_run={n} truncated_at={a['truncated_at_step']} ({a['truncated_channel']})")
        m = metrics.index("anom_rms_sigma")
        order = [c for c in np.argsort(np.where(fb[m, 0] > 0, fb[m, 0], 10**9)) if fb[m, 0, c] > 0]
        print("  first past 3σ (anom_rms): " + ", ".join(
            f"{names[c]}@{fb[m, 0, c]}" for c in order[: args.top]) + (" (none)" if not order else ""))
        print("  median_cross " + json.dumps({metrics[i]: {"3x": int(mc[i, 0]), "10x": int(mc[i, 1])}
                                              for i in range(len(metrics))}))
        for ch in DRIFT_CHANNELS:
            if ch not in names:
                continue
            c = names.index(ch)
            vals = {L: gm[L - 1, c] - gm[0, c] for L in LEADS if L <= n}
            print(f"  drift {ch:7s} (phys, lead L − lead 1): "
                  + " ".join(f"L{L}={v:+.4g}" for L, v in vals.items())
                  + f"  | lead1={gm[0, c]:.6g}")
    print("CLIMATE_READOUT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
