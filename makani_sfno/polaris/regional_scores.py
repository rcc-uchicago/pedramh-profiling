"""Land / ocean NRMSE + ACC panels for rollout NetCDFs, and the port F acceptance rule.

Architect review §6 (2026-09-23): dropping SOILWATER_10CM / TSOI_10CM removes the
model's land-surface memory. A global median over 99 channels cannot see whether
that hurt the land, so this scores the four channels a soil-moisture feedback would
act through, over land and over ocean from the SAME pass.

PRE-REGISTERED 2026-09-23, before any post-F number exists (peer-reviewed):
  * mask   = pack forcing `lsm` time mean > 0.5 is land, the rest ocean (a
             fractional threshold is a choice; this is it). Weights = the global
             scorer's lat weights restricted to the mask -- with an all-ones mask
             these ARE `sfno_eval.metrics.rmse_lat_weighted` / `acc` (tested).
  * panel  = PRECT, TREFHT, RHREFHT, TMQ at 126 h and 336 h; NRMSE = RMSE / A_truth.
  * base   = the 101-channel prod1n_b32_sgdr rollouts RE-SCORED by this same code
             (not the old h5): the four channels exist in both contracts.
  * rule   = `land_rule`: ACCEPT iff no panel channel's LAND NRMSE worsens by more
             than 1.92 % (the project bar) at either lead. A land regression while
             the global 99-channel median NRMSE336 improves is SOIL_FEEDBACK_MATTERED:
             route to the frozen-climatology variant, NOT acceptance. Otherwise REJECT.
             ACC is reported alongside and never overrides.

    python regional_scores.py --nc-dir <inference/nwp> --pack <root> --out <dir>
      -> REGIONAL_SCORES_OK ... ; writes regional_scores.csv
    python regional_scores.py --rule --base <base.csv> --new <new.csv> \
        --base-median <k56_readout_common99.json> --new-median <new k56_readout.json>
      -> LAND_RULE <VERDICT> ...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PANEL = ("PRECT", "TREFHT", "RHREFHT", "TMQ")
LEADS_H = (126, 336)
LSM_THRESHOLD = 0.5
BAR = 0.0192


def region_weights(lat_w: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(H, W) cell weights: lat weight / n_lon, zero off-mask, renormalized to sum 1."""
    w = np.asarray(lat_w, float)[:, None] / mask.shape[1] * mask.astype(float)
    s = w.sum()
    if s <= 0:
        raise ValueError("EMPTY_REGION: mask selects no cells")
    return w / s


def masked_metrics(p, t, clim, W):
    """rmse, acc, a_truth over (..., H, W) with cell weights W (sum 1)."""
    rmse = np.sqrt(((p - t) ** 2 * W).sum(axis=(-2, -1)))
    pa, ta = p - clim, t - clim
    num = (pa * ta * W).sum(axis=(-2, -1))
    den = np.sqrt((pa ** 2 * W).sum(axis=(-2, -1)) * (ta ** 2 * W).sum(axis=(-2, -1)))
    acc = num / (den + 1e-12)
    a_truth = np.sqrt((ta ** 2 * W).sum(axis=(-2, -1)))
    return rmse, acc, a_truth


def land_rule(base: dict, new: dict, base_median336: float, new_median336: float) -> dict:
    """base/new: {(region, channel, lead_h): {"nrmse": x, "acc": y}}. Returns verdict dict."""
    worse = []
    for ch in PANEL:
        for h in LEADS_H:
            b, n = base[("land", ch, h)]["nrmse"], new[("land", ch, h)]["nrmse"]
            rel = (n - b) / b
            if not np.isfinite(rel) or rel > BAR:
                worse.append({"channel": ch, "lead_h": h, "base": b, "new": n, "rel": rel})
    if not worse:
        verdict = "ACCEPT"
    elif new_median336 < base_median336:
        verdict = "SOIL_FEEDBACK_MATTERED"
    else:
        verdict = "REJECT"
    return {"verdict": verdict, "worse_over_land": worse,
            "median336": {"base": base_median336, "new": new_median336}}


def _read_csv(path: Path) -> dict:
    out = {}
    with path.open() as fh:
        for r in csv.DictReader(fh):
            out[(r["region"], r["channel"], int(r["lead_hours"]))] = {
                "nrmse": float(r["nrmse"]), "acc": float(r["acc"])}
    return out


def _score(args) -> int:
    import xarray as xr
    import torch
    from sfno_eval import metrics as M
    import score_rollout_nc as S
    import convert_e3sm_to_makani_alldata as C

    files = sorted(args.nc_dir.glob("*.nc"))[: args.limit_files or None]
    if not files:
        print(f"ERROR NO_ROLLOUT_NC: {args.nc_dir}")
        return 2
    with xr.open_dataset(files[0], decode_timedelta=False) as ds0:
        n_lat, n_lon = int(ds0.sizes["lat"]), int(ds0.sizes["lon"])
        chan0 = [str(c) for c in ds0["channel"].values]
        lead_h = np.asarray(ds0["lead_time"].values).astype(np.int64)
    missing = [c for c in PANEL if c not in chan0]
    if missing or any(h not in lead_h for h in LEADS_H):
        print(f"ERROR PANEL_UNAVAILABLE: channels missing {missing}, leads {list(lead_h[:3])}...")
        return 2
    tm = args.pack / "stats" / "time_means.npy"
    clim_all = S._select_clim_channels(S._load_climatology(tm, n_lat, n_lon), chan0, tm).numpy()
    ci = [chan0.index(c) for c in PANEL]
    clim = clim_all[ci]
    lsm = np.load(args.pack / "stats" / "forcing_time_means.npy")[0, C.FORCING_CHANNELS.index("lsm")]
    land = lsm > LSM_THRESHOLD
    lat_w = M.lat_weights(n_lat, "equiangular").to(torch.float64).numpy()
    Ws = {"land": region_weights(lat_w, land), "ocean": region_weights(lat_w, ~land),
          "global": region_weights(lat_w, np.ones_like(land))}
    ki = [int(np.where(lead_h == h)[0][0]) for h in LEADS_H]

    acc = {}
    for path in files:
        with xr.open_dataset(path, decode_timedelta=False) as ds:
            for h, k in zip(LEADS_H, ki):
                p = np.asarray(ds["prediction"][0, k].isel(channel=ci).values, np.float64)
                t = np.asarray(ds["truth"][0, k].isel(channel=ci).values, np.float64)
                for reg, W in Ws.items():
                    r, a, at = masked_metrics(p, t, clim, W)
                    for j, ch in enumerate(PANEL):
                        acc.setdefault((reg, ch, h), []).append((r[j] / at[j], a[j], r[j]))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "regional_scores.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["region", "channel", "lead_hours", "nrmse", "acc", "rmse", "n_ic"])
        for (reg, ch, h), v in sorted(acc.items()):
            v = np.asarray(v)
            wr.writerow([reg, ch, h, f"{v[:, 0].mean():.8g}", f"{v[:, 1].mean():.8g}",
                         f"{v[:, 2].mean():.8g}", len(v)])
    print(f"land cells {int(land.sum())} of {land.size} (lsm > {LSM_THRESHOLD}); {len(files)} ICs")
    print(f"REGIONAL_SCORES_OK out={args.out}/regional_scores.csv")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nc-dir", type=Path)
    p.add_argument("--pack", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--limit-files", type=int, default=None)
    p.add_argument("--rule", action="store_true")
    p.add_argument("--base", type=Path)
    p.add_argument("--new", type=Path)
    p.add_argument("--base-median", type=Path)
    p.add_argument("--new-median", type=Path)
    a = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if not a.rule:
        return _score(a)
    bm = json.loads(a.base_median.read_text())["readout"]["nrmse336"]
    nm = json.loads(a.new_median.read_text())["readout"]["nrmse336"]
    v = land_rule(_read_csv(a.base), _read_csv(a.new), bm, nm)
    for w in v["worse_over_land"]:
        print(f"  land {w['channel']} {w['lead_h']}h NRMSE {w['base']:.4f} -> {w['new']:.4f} "
              f"({100 * w['rel']:+.2f} %)")
    print(f"LAND_RULE {v['verdict']} median336 {bm:.4f} -> {nm:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
