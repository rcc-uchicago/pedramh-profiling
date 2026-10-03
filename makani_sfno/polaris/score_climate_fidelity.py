#!/usr/bin/env python3
"""score_climate_fidelity.py — does a rollout's 5-year mean climate match reality?

Compares one or more ``member_*.nc`` files (from ``climate_rollout.py`` /
``polaris_climate_run.pbs``, each carrying its own ``time_mean`` field -- the
temporal mean over every scored lead) against the true time-mean built by
``build_true_climatology.py`` over the same window. This is the climate-fidelity
answer to "RMSE and accuracy of the 5-year rollout": per-channel, area-weighted
**mean-state bias** and **pattern RMSE** of the climatological field, not an
instantaneous forecast-skill score (which is not meaningful at a 5-year lead --
see ``build_true_climatology.py``'s docstring).

Per channel: bias = area-weighted mean(member - true); pattern_rmse =
area-weighted RMS(member - true). Both also reported in units of the channel's
global_std ("sigma"), since raw units aren't comparable across channels (PS in
Pa vs a mixing ratio near 0) -- the same normalisation convention the rest of
this driver uses for stability metrics.

PASS token: CLIMATE_FIDELITY_OK n_members=<n> out=<csv path>

Usage:
    python score_climate_fidelity.py --true climatology.npz \\
        --member member_B2200.nc member_B2201.nc ... --std-path <global_stds.npy> \\
        --out fidelity.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--true", required=True, type=Path, help="build_true_climatology.py output")
    p.add_argument("--member", required=True, type=Path, nargs="+", help="member_*.nc file(s)")
    p.add_argument("--std-path", required=True, type=Path,
                   help="global_stds.npy for sigma-normalised bias/RMSE (same file the run used)")
    p.add_argument("--out", required=True, type=Path, help="CSV output path")
    p.add_argument("--top", type=int, default=10, help="Worst channels to print (default 10)")
    return p.parse_args(argv)


def area_weighted_bias_rmse(member_mean, true_mean, weights):
    """Per-channel area-weighted bias and pattern RMSE, both ``(C,)``.

    ``member_mean``/``true_mean``: ``(C, H, W)`` physical units, same channel
    order. ``weights``: ``(H,)`` equiangular cos-lat weights summing to 1.
    """
    import numpy as np

    diff = member_mean - true_mean                      # (C, H, W)
    zonal = diff.mean(axis=-1)                           # (C, H) -- plain mean over lon
    bias = zonal @ weights                                # (C,)
    zonal_sq = (diff ** 2).mean(axis=-1)
    rmse = np.sqrt(zonal_sq @ weights)
    return bias, rmse


def main(argv=None) -> int:
    args = _parse_args(argv)
    import numpy as np
    import netCDF4

    true = np.load(args.true, allow_pickle=True)
    true_mean = true["time_mean"]                         # (C, H, W)
    true_names = [str(c) for c in true["channel_names"]]
    std = np.load(args.std_path).astype(np.float64).reshape(-1)
    if std.size != len(true_names):
        raise ValueError(f"STD_CHANNEL_MISMATCH: {std.size} stds for {len(true_names)} "
                         f"true-climatology channels")

    weights = None
    rows = []
    for mpath in args.member:
        with netCDF4.Dataset(mpath, "r") as ds:
            member_names = [str(c) for c in ds["channel"][:]]
            if member_names != true_names:
                raise ValueError(f"CHANNEL_MISMATCH: {mpath} channels != true climatology's "
                                 f"(first mismatch at "
                                 f"{next((i for i, (a, b) in enumerate(zip(member_names, true_names)) if a != b), '?')})")
            member_mean = np.asarray(ds["time_mean"][:], dtype=np.float64)   # (C, H, W)
            lat = np.asarray(ds["lat"][:], dtype=np.float64)
        if weights is None:
            w = np.cos(np.deg2rad(lat))
            weights = w / w.sum()
        if member_mean.shape != true_mean.shape:
            raise ValueError(f"SHAPE_MISMATCH: {mpath} time_mean {member_mean.shape} != "
                             f"true climatology {true_mean.shape}")
        bias, rmse = area_weighted_bias_rmse(member_mean, true_mean, weights)
        for c, name in enumerate(true_names):
            rows.append(dict(member=mpath.name, channel=name,
                             bias=float(bias[c]), rmse=float(rmse[c]),
                             bias_sigma=float(bias[c] / std[c]), rmse_sigma=float(rmse[c] / std[c])))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["member", "channel", "bias", "rmse", "bias_sigma", "rmse_sigma"])
        w.writeheader()
        w.writerows(rows)

    # --- concise read-out: worst channels by pattern RMSE, in sigma units, per member ---
    by_member: dict[str, list[dict]] = {}
    for r in rows:
        by_member.setdefault(r["member"], []).append(r)
    for member, mrows in by_member.items():
        mrows.sort(key=lambda r: -r["rmse_sigma"])
        print(f"{member}: worst {args.top} channels by pattern RMSE (sigma units)")
        for r in mrows[: args.top]:
            print(f"  {r['channel']:<16} bias={r['bias']:+.4g} ({r['bias_sigma']:+.3f}sigma) "
                  f"rmse={r['rmse']:.4g} ({r['rmse_sigma']:.3f}sigma)")

    print(f"CLIMATE_FIDELITY_OK n_members={len(args.member)} out={args.out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- one greppable line, then the traceback
        print(f"ERROR CLIMATE_FIDELITY_FAILED {type(exc).__name__}: {str(exc).splitlines()[0][:300]}")
        raise
