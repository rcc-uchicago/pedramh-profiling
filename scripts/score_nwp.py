#!/usr/bin/env python3
"""score_nwp.py — compute the Phase 1 NWP scorecard from per-IC NetCDFs.

Implements docs/sfno_eval_plan.md §D. Reads
``{out_root}/inference/nwp/*.nc`` plus the climatology at
``{out_root}/baselines/climatology_proleptic.nc`` and produces:

  - ``scores/nwp_scorecard.csv`` — tidy long format, columns
    ``model, channel, lead_hours, ic_year, ic_sample_idx, metric, value``.
  - ``scores/nwp_scorecard_summary.csv`` — IC-averaged.
  - ``scores/bias_maps_<channel>_<lead>.npy`` — for the 5 key channels
    at the 6 scored leads (§D.4).

The sanity gate (§D.6) is enforced at the end; non-zero exit code if any
of the three conditions fail::

  - Emulator RMSE on `tas` at 6 h <  persistence RMSE on `tas` at 6 h
  - Emulator ACC  on `zg5` at 24 h >  0.6
  - Emulator RMSE finite (no NaN/Inf) for all (channel, lead_time) pairs

Persistence is computed only for the 52 state channels; pr_6h
persistence is reported as NaN per §C.1.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))


# Lead times to score (§D.4); h = lead_time in hours, k = step index
_SCORED_LEADS_H = (6, 24, 72, 120, 240, 336)
# Bias-map channels (§D.3)
_BIAS_CHANNELS = ("tas", "pr_6h", "zg5", "ua5", "ta5")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute the Phase 1 NWP scorecard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--out-root", required=True, type=Path,
                   help="Eval output root (output of eval_inference.py)")
    p.add_argument("--clim-nc", required=True, type=Path,
                   help="Climatology NetCDF (output of compute_climatology.py)")
    p.add_argument("--scorecard-out", type=Path, default=None,
                   help="Override scores/nwp_scorecard.csv path")
    return p.parse_args()


def _load_clim_for_lookup(clim_nc: Path):
    """Return ``(mean_arr, n_contrib_arr, doys, hour_quarters, channels)``."""
    import xarray as xr
    ds = xr.open_dataset(clim_nc)
    return (
        ds["mean"].values,           # (366, 4, n_chan, H, W)
        ds["n_contributors"].values, # (366, 4)
        ds["doy"].values,
        ds["hour_quarter"].values,
        list(ds["channel"].values),
    )


def _date_for_lead(file_anchor: str, time_plasim_at_ic: float, lead_hours: int):
    """Return (month, day, hour) of the lead-h target sample."""
    import re
    from datetime import timedelta
    import cftime
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})", file_anchor)
    if m is None:
        raise ValueError(f"unparseable file_anchor: {file_anchor!r}")
    Y, M, D, h, mi, s = (int(g) for g in m.groups())
    base = cftime.DatetimeProlepticGregorian(Y, M, D, h, mi, s)
    dt = base + timedelta(days=time_plasim_at_ic) + timedelta(hours=lead_hours)
    return dt.month, dt.day, dt.hour


def _compute_metrics_for_one_ic(
    nc_path: Path,
    *,
    clim_mean: np.ndarray,
    clim_n: np.ndarray,
    channels: list[str],
    lat_weights: np.ndarray,
    rows: list[dict],
    bias_accumulators: dict,
):
    """Append RMSE/ACC rows and update bias accumulators for one NetCDF."""
    import xarray as xr
    import torch
    from sfno_eval import metrics as M
    from sfno_eval.climatology import calendar_bin

    ds = xr.open_dataset(nc_path)
    try:
        pred = torch.from_numpy(ds["prediction"].values[0]).float()  # (K, n_chan, H, W)
        truth = torch.from_numpy(ds["truth"].values[0]).float()
        init_state = torch.from_numpy(ds["init_state"].values[0]).float()  # (52, H, W)
        lead_time = ds["lead_time"].values  # (K,) hours
        chan_names = list(ds["channel"].values)
        ic_file = ds.attrs["ic_file"]
        ic_sample_idx = int(ds.attrs["ic_sample_idx"])
        file_anchor = ds.attrs["file_anchor"]
        time_plasim_at_ic = float(ds.attrs["time_plasim_at_ic"])
    finally:
        ds.close()

    lat_w = torch.from_numpy(lat_weights).float()
    K, n_chan, H, W = pred.shape
    ic_year = ic_file.replace("MOST.", "").replace(".h5", "")

    # Derive channel indices we care about for the bias accumulators.
    bias_chan_idx = {c: chan_names.index(c) for c in _BIAS_CHANNELS if c in chan_names}

    for h in _SCORED_LEADS_H:
        if h not in lead_time:
            continue  # NWP K=56 covers up to 336 h; if shorter, skip
        k = int(np.where(lead_time == h)[0][0])
        # === EMULATOR RMSE / ACC per channel ===
        for c, name in enumerate(chan_names):
            rmse_em = float(M.rmse_lat_weighted(pred[k, c], truth[k, c], lat_w))
            rows.append(dict(
                model="emulator", channel=name, lead_hours=int(h),
                ic_year=ic_year, ic_sample_idx=ic_sample_idx,
                metric="rmse", value=rmse_em,
            ))

            # ACC needs climatology at the lead-h target's calendar bin.
            month, day, hour = _date_for_lead(file_anchor, time_plasim_at_ic, h)
            doy_idx, hq_idx = calendar_bin(month, day, hour)
            if clim_n[doy_idx, hq_idx] > 0:
                cm = torch.from_numpy(clim_mean[doy_idx, hq_idx, c]).float()
                acc_em = float(M.acc(pred[k, c], truth[k, c], cm, lat_w))
                rows.append(dict(
                    model="emulator", channel=name, lead_hours=int(h),
                    ic_year=ic_year, ic_sample_idx=ic_sample_idx,
                    metric="acc", value=acc_em,
                ))

        # === PERSISTENCE RMSE — STATE CHANNELS ONLY ===
        # init_state has 52 channels (no diagnostic). pr_6h gets NaN.
        n_state = init_state.shape[0]
        for c, name in enumerate(chan_names):
            if c < n_state:
                rmse_p = float(M.rmse_lat_weighted(init_state[c], truth[k, c], lat_w))
                rows.append(dict(
                    model="persistence", channel=name, lead_hours=int(h),
                    ic_year=ic_year, ic_sample_idx=ic_sample_idx,
                    metric="rmse", value=rmse_p,
                ))
            else:
                # diagnostic-only channel — persistence undefined (§C.1)
                rows.append(dict(
                    model="persistence", channel=name, lead_hours=int(h),
                    ic_year=ic_year, ic_sample_idx=ic_sample_idx,
                    metric="rmse", value=float("nan"),
                ))

        # === BIAS MAP accumulators (5 key channels) ===
        for name, c in bias_chan_idx.items():
            key = (name, int(h))
            err = (pred[k, c] - truth[k, c]).numpy()
            bias_accumulators.setdefault(key, []).append(err)


def _summarize(rows: list[dict], summary_path: Path):
    """Write IC-averaged scorecard."""
    import collections
    grouped: dict[tuple, list[float]] = collections.defaultdict(list)
    for r in rows:
        if r["value"] != r["value"]:  # NaN
            continue
        key = (r["model"], r["channel"], r["lead_hours"], r["metric"])
        grouped[key].append(r["value"])

    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "channel", "lead_hours", "metric", "mean", "std", "n_ics"])
        for (model, ch, lead, metric), values in sorted(grouped.items()):
            arr = np.asarray(values, dtype=np.float64)
            w.writerow([model, ch, lead, metric, float(arr.mean()), float(arr.std()), len(arr)])


def _enforce_sanity_gate(rows: list[dict]) -> int:
    """Apply §D.6 gate. Return 0 on pass, 1 on fail."""
    import collections
    rc = 0

    # Build a quick (model, channel, lead, metric) → list-of-values lookup.
    bucket: dict[tuple, list[float]] = collections.defaultdict(list)
    for r in rows:
        bucket[(r["model"], r["channel"], r["lead_hours"], r["metric"])].append(r["value"])

    def _mean(key):
        vals = [v for v in bucket.get(key, []) if v == v]  # drop NaN
        return float(np.mean(vals)) if vals else float("nan")

    em_tas_6h = _mean(("emulator", "tas", 6, "rmse"))
    pers_tas_6h = _mean(("persistence", "tas", 6, "rmse"))
    em_zg5_24h_acc = _mean(("emulator", "zg5", 24, "acc"))

    print(f"[gate] emulator RMSE tas 6h = {em_tas_6h:.4f}")
    print(f"[gate] persistence RMSE tas 6h = {pers_tas_6h:.4f}")
    print(f"[gate] emulator ACC zg5 24h = {em_zg5_24h_acc:.4f}")

    if not (em_tas_6h < pers_tas_6h):
        print(f"[gate] FAIL: emulator RMSE tas 6h ({em_tas_6h:.4f}) "
              f">= persistence ({pers_tas_6h:.4f})", file=sys.stderr)
        rc = 1
    if not (em_zg5_24h_acc > 0.6):
        print(f"[gate] FAIL: emulator ACC zg5 24h ({em_zg5_24h_acc:.4f}) <= 0.6",
              file=sys.stderr)
        rc = 1

    # Finite-everywhere check.
    nans = [r for r in rows if r["model"] == "emulator" and r["value"] != r["value"]]
    if nans:
        print(f"[gate] FAIL: {len(nans)} NaN/Inf rows in emulator metrics", file=sys.stderr)
        rc = 1

    if rc == 0:
        print("[gate] PASS")
    return rc


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("score_nwp")

    from sfno_eval import metrics as M

    nc_dir = args.out_root / "inference" / "nwp"
    nc_files = sorted(nc_dir.glob("*.nc"))
    if not nc_files:
        raise SystemExit(f"no NWP NetCDFs found under {nc_dir}")
    logger.info("scoring %d NWP NetCDFs", len(nc_files))

    clim_mean, clim_n, _, _, _ = _load_clim_for_lookup(args.clim_nc)

    # Lat weights — read once.
    H = clim_mean.shape[-2]
    lat_weights = M.legendre_gauss_lat_weights(H).numpy()

    rows: list[dict] = []
    bias_accumulators: dict = {}

    for nc in nc_files:
        _compute_metrics_for_one_ic(
            nc,
            clim_mean=clim_mean, clim_n=clim_n,
            channels=[],  # not used downstream
            lat_weights=lat_weights,
            rows=rows,
            bias_accumulators=bias_accumulators,
        )

    scores_dir = args.out_root / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    sc_path = args.scorecard_out or (scores_dir / "nwp_scorecard.csv")
    with sc_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "channel", "lead_hours", "ic_year", "ic_sample_idx", "metric", "value",
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info("wrote %s (%d rows)", sc_path, len(rows))

    summary_path = scores_dir / "nwp_scorecard_summary.csv"
    _summarize(rows, summary_path)
    logger.info("wrote %s", summary_path)

    # Bias maps (one .npy per (channel, lead_hours)).
    for (name, lead), errs in bias_accumulators.items():
        arr = np.stack(errs).mean(axis=0).astype(np.float32)
        out_path = scores_dir / f"bias_maps_{name}_{lead}h.npy"
        np.save(out_path, arr)
    logger.info("wrote %d bias maps", len(bias_accumulators))

    return _enforce_sanity_gate(rows)


if __name__ == "__main__":
    sys.exit(main())
