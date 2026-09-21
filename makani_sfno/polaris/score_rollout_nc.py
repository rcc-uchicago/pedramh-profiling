#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
"""Score E3SM rollout NetCDFs offline: the per-lead curve, and the K=56 read-out.

Consumes what `scripts/eval_inference.py --mode nwp` writes (one NetCDF per IC,
physical units, `prediction`/`truth`/`init_state`) and produces the
`(leads x channels)` RMSE / ACC / anomaly-amplitude curves plus the single
pre-registered verdict of `docs/2026-09-20_k56_readout_prereg.md` §3.

WHY THIS EXISTS AND NOT `scripts/score_nwp.py`.  That scorer serves the PLaSim
track: it weights latitudes with `legendre_gauss_lat_weights`, which change G
measured over-weighting the polar row by 1.50x on our equiangular 180x360 pack;
it requires a calendar-binned climatology NetCDF that E3SM has no builder for;
and its sanity gate is PLaSim-specific (`tas`, `zg500`, `MOST.`-stripped IC
names, a 52-state-channel persistence assumption).  `polaris_eval_inference.pbs`
ported stage 1 of the SLURM chain and deliberately not stages 2-4 for exactly
this reason.  This is the E3SM sibling of stage 2 (CLAUDE.md #7), and it reuses
`sfno_eval.metrics` for the formulas so our numbers stay identical in convention
to the PLaSim track's -- only the quadrature and the climatology differ.

TWO OUTPUTS, TWO CONSUMERS.  The read-out JSON answers "CRPS or n_future=8?".
The `(56 x 101)` RMSE array is `sigma(k)` for the lagged ensemble's weighted
combination (`w_k ~ 1/sigma(k)^2`, end-to-end plan task 13), which today has no
weights beyond lead 20.

PASS = `K56_SCORE_OK` on stdout.  Per CLAUDE.md #14 rc=0 is not the signal: a
run that scored zero files, or that could not reach the pre-registered leads,
exits 0 from the OS' point of view and must not read as a result.

Run under the SFNO venv on a compute node (CLAUDE.md #3):

    python polaris/score_rollout_nc.py \\
        --nc-dir  $MEMBER_ROOT/runs/makani_eval/prod1n_b32_sgdr_K56/inference/nwp \\
        --time-means $MEMBER_ROOT/data/e3sm_makani_alldata_production/stats/time_means.npy \\
        --out-dir $MEMBER_ROOT/runs/makani_eval/prod1n_b32_sgdr_K56/scores
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# The metric formulas live in the shared eval tree; import them rather than
# re-deriving, so a PLaSim number and an E3SM number differ only in quadrature
# and climatology (prereg §2).
_MAKANI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_MAKANI_ROOT / "src"))

logger = logging.getLogger("score_rollout_nc")

# Pre-registered anchors (prereg §3). Leads in hours; the slope ratio is
# late-window over early-window on the median NRMSE curve.
EARLY_LEADS_H = (30, 126)
LATE_LEADS_H = (240, 336)

# Pre-registered thresholds. Named constants because the rule must be greppable
# and diffable -- a changed number here is a changed decision, and the test
# file pins every branch they select.
VR_COLLAPSED = 0.60
VR_PRESERVED = 0.85
NRMSE_AT_MEAN = 1.15
NRMSE_PAST_MEAN = 1.30
NRMSE_DRIFT = 1.60
NRMSE_CHANNEL_BLOWUP = 3.0
SLOPE_SATURATING = 0.25
SLOPE_CLIMBING = 0.60
ACC_DECAYED = 0.30
ACC_HIGH = 0.50

# A channel whose truth anomaly amplitude is this small is constant for
# scoring purposes; NRMSE/VR would divide by noise. Reported, never silently
# dropped.
AMPLITUDE_FLOOR = 1e-12


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score E3SM rollout NetCDFs and apply the K=56 read-out rule.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--nc-dir", required=True, type=Path,
                   help="Directory of per-IC rollout NetCDFs (inference/nwp)")
    p.add_argument("--time-means", required=True, type=Path,
                   help="Pack climatology, stats/time_means.npy, (1, C, H, W) physical units")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Where k56_metrics.h5 / k56_summary.csv / k56_readout.json go")
    p.add_argument("--grid-type", default="equiangular",
                   help="Quadrature for the lat weights (default: equiangular, the E3SM pack)")
    p.add_argument("--limit-files", type=int, default=None,
                   help="Score only the first N NetCDFs (smoke)")
    p.add_argument("--limit-leads", type=int, default=None,
                   help="Score only the first N leads of each file (smoke)")
    p.add_argument("--provenance", default="",
                   help="Free-form string recorded in the outputs (job id, git sha)")
    return p.parse_args()


def _load_climatology(path: Path, grid_lat: int, grid_lon: int):
    """Return the pack's time-mean climatology as a ``(C, H, W)`` float64 tensor.

    `convert_e3sm_to_makani_alldata.py`'s stats pass writes `(1, C, H, W)` float32
    in **raw physical units**, averaged over the training split -- the same array
    makani's validation ACC uses, which is what makes this curve comparable with
    the existing 126 h one (prereg §2). The rollout NetCDFs are de-normalized to
    physical units too, so no scaling is applied to either side.
    """
    import torch

    arr = np.load(path)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise SystemExit(
            f"CLIM_SHAPE_UNEXPECTED: {path} has shape {arr.shape}; expected (1, C, H, W) or (C, H, W)"
        )
    if arr.shape[-2:] != (grid_lat, grid_lon):
        raise SystemExit(
            f"CLIM_GRID_MISMATCH: climatology grid {arr.shape[-2:]} vs NetCDF grid "
            f"{(grid_lat, grid_lon)}. Refusing to score against a different grid."
        )
    return torch.from_numpy(np.asarray(arr, dtype=np.float64))


def _lat_weighted_mean(field, w):
    """Latitude-weighted area mean over ``(..., lat, lon)``, matching the RMSE weighting."""
    return (field.mean(dim=-1) * w.to(field.dtype)).sum(dim=-1)


def _score_one_file(path: Path, *, clim, w, limit_leads: int | None):
    """Return per-lead, per-channel metrics for one IC as a dict of ``(K, C)`` arrays.

    Shapes: the NetCDF carries `prediction`/`truth` as `(init_time=1, lead, channel,
        lat, lon)`; each lead is read on its own so a 2 GB file never has to be
        resident whole.
    Precision: float64 throughout. The metrics are sums over 6.5M cells and the
        source is float32; accumulating in float32 would cost digits for free.

    `a_pred` / `a_truth` are anomaly amplitudes about the climatology, computed as
    `rmse_lat_weighted(field, clim)` -- algebraically the same quantity, so the
    blurring diagnostic `VR = a_pred / a_truth` inherits the RMSE weighting exactly.
    """
    import torch
    import xarray as xr
    from sfno_eval import metrics as M

    # decode_timedelta=False pins the lead coord to the integers on disk. Current
    # xarray decodes a `units = "hours"` attribute into timedelta64 and warns that
    # the default is about to change again -- the same drift that broke
    # test_nc_writer's round-trip assertion. The normalisation below stays as a
    # guard for anyone who opens the file without this flag.
    ds = xr.open_dataset(path, decode_timedelta=False)
    try:
        chan = [str(c) for c in ds["channel"].values]
        lead_h = np.asarray(ds["lead_time"].values)
        if np.issubdtype(lead_h.dtype, np.timedelta64):
            lead_h = (lead_h / np.timedelta64(1, "h")).astype(np.int64)
        lead_h = lead_h.astype(np.int64)
        n_lead = len(lead_h) if limit_leads is None else min(limit_leads, len(lead_h))
        n_chan = len(chan)

        out = {k: np.full((n_lead, n_chan), np.nan, dtype=np.float64)
               for k in ("rmse", "acc", "a_pred", "a_truth", "bias")}

        for k in range(n_lead):
            # Index the DataArray, then `.values` -- NOT `.values[0, k]`, which
            # would materialise all 1.47 GB of the variable once per lead.
            p = torch.from_numpy(
                np.asarray(ds["prediction"][0, k].values, dtype=np.float64))
            t = torch.from_numpy(
                np.asarray(ds["truth"][0, k].values, dtype=np.float64))
            out["rmse"][k] = M.rmse_lat_weighted(p, t, w).numpy()
            out["acc"][k] = M.acc(p, t, clim, w).numpy()
            out["a_pred"][k] = M.rmse_lat_weighted(p, clim, w).numpy()
            out["a_truth"][k] = M.rmse_lat_weighted(t, clim, w).numpy()
            out["bias"][k] = _lat_weighted_mean(p - t, w).numpy()
    finally:
        ds.close()

    return out, chan, lead_h[:n_lead]


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Elementwise ratio with a floor on the denominator, NaN where the floor bites."""
    out = np.full_like(num, np.nan)
    ok = den > AMPLITUDE_FLOOR
    out[ok] = num[ok] / den[ok]
    return out


def readout_from_curves(lead_h: np.ndarray, nrmse: np.ndarray, vr: np.ndarray,
                        acc: np.ndarray) -> dict:
    """Reduce the `(K, C)` curves to the handful of numbers the rule in prereg §3 reads.

    Aggregation is the **median over channels** (prereg §2): the 101 channels sit on
    different scales and a mean would report whatever the worst few do. Slopes are
    per-hour finite differences between the pre-registered lead anchors, never fitted
    -- a fit invites a choice of window after the fact.

    Returns a dict with `available: False` when the sweep is too short to reach the
    336 h anchor, which is an honest "cannot decide", not a degraded verdict.
    """
    lead_h = np.asarray(lead_h)
    idx = {int(h): int(np.where(lead_h == h)[0][0])
           for h in set(EARLY_LEADS_H + LATE_LEADS_H) if h in lead_h}
    needed = set(EARLY_LEADS_H + LATE_LEADS_H)
    if not needed.issubset(idx):
        return {"available": False,
                "missing_leads_h": sorted(needed - set(idx)),
                "max_lead_h": int(lead_h.max()) if lead_h.size else None}

    med_nrmse = np.nanmedian(nrmse, axis=1)
    med_vr = np.nanmedian(vr, axis=1)
    med_acc = np.nanmedian(acc, axis=1)

    e0, e1 = EARLY_LEADS_H
    l0, l1 = LATE_LEADS_H
    slope_early = (med_nrmse[idx[e1]] - med_nrmse[idx[e0]]) / (e1 - e0)
    slope_late = (med_nrmse[idx[l1]] - med_nrmse[idx[l0]]) / (l1 - l0)
    r_slope = float(slope_late / slope_early) if abs(slope_early) > 0 else float("nan")

    per_channel_max = np.nanmax(nrmse[idx[l1]]) if nrmse.shape[1] else np.nan

    return {
        "available": True,
        "nrmse336": float(med_nrmse[idx[l1]]),
        "nrmse126": float(med_nrmse[idx[e1]]),
        "vr336": float(med_vr[idx[l1]]),
        "vr126": float(med_vr[idx[e1]]),
        "acc336": float(med_acc[idx[l1]]),
        "acc126": float(med_acc[idx[e1]]),
        "slope_early_per_h": float(slope_early),
        "slope_late_per_h": float(slope_late),
        "r_slope": r_slope,
        "max_channel_nrmse336": float(per_channel_max),
    }


def classify_regime(r: dict) -> dict:
    """Apply the pre-registered branch rule of `docs/2026-09-20_k56_readout_prereg.md` §3.

    Fixed on 2026-09-20 **before** any number from job 7633207 was computed, and
    pinned branch-by-branch by `test_score_rollout_nc.py` so it cannot quietly move
    once the numbers are in. The order of the tests is part of the rule: drift is
    checked first because a rollout leaving the attractor makes both arms the wrong
    spend, and ACC gates nothing -- it only has to be *consistent* with the branch.

    Returns `{branch, reason, acc_consistent}` where branch is one of
    `INSUFFICIENT_HORIZON`, `DRIFT_FIRST`, `CRPS`, `NFUTURE8`, `AMBIGUOUS`.
    """
    if not r.get("available"):
        return {"branch": "INSUFFICIENT_HORIZON",
                "reason": f"sweep does not reach the pre-registered anchors: "
                          f"missing {r.get('missing_leads_h')}",
                "acc_consistent": None}

    nrmse, vr, acc, slope = r["nrmse336"], r["vr336"], r["acc336"], r["r_slope"]

    # Name the clause that fired. The two are very different findings -- a median
    # over threshold is a whole-model drift, a max over threshold can be one
    # channel -- and a compound message invites reading the first as the cause.
    if nrmse > NRMSE_DRIFT:
        return {"branch": "DRIFT_FIRST",
                "reason": f"median NRMSE336={nrmse:.3f} > {NRMSE_DRIFT}: the rollout as a "
                          "whole is leaving the attractor, not merely losing skill",
                "acc_consistent": None}
    if r["max_channel_nrmse336"] > NRMSE_CHANNEL_BLOWUP:
        return {"branch": "DRIFT_FIRST",
                "reason": f"worst channel NRMSE336={r['max_channel_nrmse336']:.3f} > "
                          f"{NRMSE_CHANNEL_BLOWUP} while the median is {nrmse:.3f}: at least "
                          "one channel is diverging -- identify it before spending on either arm",
                "acc_consistent": None}

    if vr <= VR_COLLAPSED and nrmse <= NRMSE_AT_MEAN:
        branch = {"branch": "CRPS",
                  "reason": f"VR336={vr:.3f} <= {VR_COLLAPSED} and NRMSE336={nrmse:.3f} "
                            f"<= {NRMSE_AT_MEAN}: the forecast has collapsed onto the "
                            "climatological mean -- mode-averaging, which no rollout "
                            "depth repairs"}
        # ACC must not contradict a blurring verdict; if it does, say so rather
        # than letting the stronger-sounding branch stand (prereg §3).
        branch["acc_consistent"] = bool(acc <= ACC_HIGH)
        if not branch["acc_consistent"]:
            branch["branch"] = "AMBIGUOUS"
            branch["reason"] += (f" -- BUT ACC336={acc:.3f} >= {ACC_HIGH} contradicts it; "
                                 "downgraded to AMBIGUOUS pending explanation")
        return branch

    if vr >= VR_PRESERVED and (slope >= SLOPE_CLIMBING or nrmse >= NRMSE_PAST_MEAN):
        branch = {"branch": "NFUTURE8",
                  "reason": f"VR336={vr:.3f} >= {VR_PRESERVED} with R_slope={slope:.3f} "
                            f"and NRMSE336={nrmse:.3f}: error still growing at realistic "
                            "amplitude -- exposure bias, so buy depth"}
        branch["acc_consistent"] = bool(acc >= ACC_DECAYED)
        return branch

    return {"branch": "AMBIGUOUS",
            "reason": f"VR336={vr:.3f}, NRMSE336={nrmse:.3f}, R_slope={slope:.3f} fall "
                      "between the pre-registered bands; report both curves rather than "
                      "spending node-hours on a coin flip",
            "acc_consistent": None}


def _write_outputs(out_dir: Path, *, curves: dict, chan: list[str], lead_h: np.ndarray,
                   n_ic: int, readout: dict, verdict: dict, provenance: str,
                   grid_type: str, nc_dir: Path, clim_path: Path) -> None:
    """Write the h5 curve bundle, the long-form CSV, and the read-out JSON."""
    import h5py

    out_dir.mkdir(parents=True, exist_ok=True)

    h5_path = out_dir / "k56_metrics.h5"
    with h5py.File(h5_path, "w") as f:
        for name, arr in curves.items():
            f.create_dataset(name, data=arr, compression="gzip", compression_opts=4)
        f.create_dataset("lead_hours", data=np.asarray(lead_h, dtype=np.int64))
        f.create_dataset("channel", data=np.array(chan, dtype=h5py.string_dtype()))
        f.attrs.update(n_ic=n_ic, grid_type=grid_type, nc_dir=str(nc_dir),
                       climatology=str(clim_path), provenance=provenance,
                       prereg="docs/2026-09-20_k56_readout_prereg.md")

    csv_path = out_dir / "k56_summary.csv"
    with csv_path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["channel", "lead_hours", "metric", "mean_over_ic", "std_over_ic", "n_ic"])
        for m in ("rmse", "acc", "a_pred", "a_truth", "nrmse", "vr", "bias"):
            mean, std = curves[f"{m}_mean"], curves[f"{m}_std"]
            for k, h in enumerate(lead_h):
                for c, name in enumerate(chan):
                    wr.writerow([name, int(h), m, f"{mean[k, c]:.8g}", f"{std[k, c]:.8g}", n_ic])

    (out_dir / "k56_readout.json").write_text(json.dumps(
        {"readout": readout, "verdict": verdict, "n_ic": n_ic,
         "grid_type": grid_type, "nc_dir": str(nc_dir), "climatology": str(clim_path),
         "provenance": provenance,
         "prereg": "docs/2026-09-20_k56_readout_prereg.md"},
        indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    import torch
    from sfno_eval import metrics as M

    nc_files = sorted(args.nc_dir.glob("*.nc"))
    if args.limit_files:
        nc_files = nc_files[: args.limit_files]
    if not nc_files:
        print(f"ERROR NO_ROLLOUT_NC: no *.nc under {args.nc_dir}", file=sys.stderr)
        return 2

    # Grid and channel contract come from the first file; every later file must
    # agree exactly. A silent channel-order difference between ICs would average
    # unrelated variables together (the v10.0/v10.1 contamination pattern).
    import xarray as xr
    with xr.open_dataset(nc_files[0], decode_timedelta=False) as ds0:
        n_lat, n_lon = int(ds0.sizes["lat"]), int(ds0.sizes["lon"])
        chan0 = [str(c) for c in ds0["channel"].values]

    clim = _load_climatology(args.time_means, n_lat, n_lon)
    if clim.shape[0] != len(chan0):
        raise SystemExit(
            f"CHANNEL_COUNT_MISMATCH: climatology has {clim.shape[0]} channels, "
            f"NetCDF has {len(chan0)} ({chan0[:3]} … {chan0[-1]}). Refusing to score: "
            "the two are indexed positionally and a mismatch misaligns every number."
        )
    w = M.lat_weights(n_lat, args.grid_type).to(torch.float64)

    acc_sum = acc_sq = None
    lead_h = None
    t0 = time.time()
    for i, path in enumerate(nc_files):
        per, chan, lh = _score_one_file(path, clim=clim, w=w, limit_leads=args.limit_leads)
        if chan != chan0:
            raise SystemExit(f"CHANNEL_ORDER_DRIFT: {path.name} disagrees with {nc_files[0].name}")
        if lead_h is None:
            lead_h = lh
            acc_sum = {k: np.zeros_like(v) for k, v in per.items()}
            acc_sq = {k: np.zeros_like(v) for k, v in per.items()}
        elif not np.array_equal(lead_h, lh):
            raise SystemExit(f"LEAD_GRID_DRIFT: {path.name} has different lead_time")

        per["nrmse"] = _ratio(per["rmse"], per["a_truth"])
        per["vr"] = _ratio(per["a_pred"], per["a_truth"])
        for k, v in per.items():
            acc_sum.setdefault(k, np.zeros_like(v))
            acc_sq.setdefault(k, np.zeros_like(v))
            acc_sum[k] += v
            acc_sq[k] += v * v
        logger.info("scored %s (%d/%d, %.1fs elapsed)", path.name, i + 1, len(nc_files),
                    time.time() - t0)

    n = len(nc_files)
    curves = {}
    for k, s in acc_sum.items():
        mean = s / n
        var = np.maximum(acc_sq[k] / n - mean * mean, 0.0)
        curves[f"{k}_mean"] = mean
        curves[f"{k}_std"] = np.sqrt(var)

    readout = readout_from_curves(lead_h, curves["nrmse_mean"], curves["vr_mean"],
                                  curves["acc_mean"])
    verdict = classify_regime(readout)

    _write_outputs(args.out_dir, curves=curves, chan=chan0, lead_h=lead_h, n_ic=n,
                   readout=readout, verdict=verdict, provenance=args.provenance,
                   grid_type=args.grid_type, nc_dir=args.nc_dir, clim_path=args.time_means)

    n_bad = int(np.sum(~np.isfinite(curves["rmse_mean"])))
    print(f"scored {n} ICs x {len(lead_h)} leads x {len(chan0)} channels "
          f"({args.grid_type} quadrature) in {(time.time() - t0) / 60:.1f} min")
    print(f"non-finite RMSE cells: {n_bad}")
    if readout.get("available"):
        print(f"median NRMSE  126h={readout['nrmse126']:.3f}  336h={readout['nrmse336']:.3f} "
              f"(1.0 = climatological mean, 1.414 = decorrelated)")
        print(f"median VR     126h={readout['vr126']:.3f}  336h={readout['vr336']:.3f} "
              f"(amplitude kept = 1)")
        print(f"median ACC    126h={readout['acc126']:.3f}  336h={readout['acc336']:.3f} "
              f"(secondary -- annual-mean climatology biases this HIGH)")
        print(f"R_slope (240-336h)/(30-126h) = {readout['r_slope']:.3f}")
    print(f"BRANCH {verdict['branch']}: {verdict['reason']}")
    print(f"wrote {args.out_dir}/k56_metrics.h5, k56_summary.csv, k56_readout.json")

    # rc=0 is not the pass signal (CLAUDE.md #14). A sweep too short to reach the
    # pre-registered anchors produced curves but not a decision -- do not let it
    # read as one.
    if verdict["branch"] == "INSUFFICIENT_HORIZON":
        print("ERROR K56_SCORE_NO_VERDICT: curves written, but the horizon cannot "
              "reach the pre-registered 336 h anchor", file=sys.stderr)
        return 3
    print(f"K56_SCORE_OK n_ic={n} branch={verdict['branch']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
