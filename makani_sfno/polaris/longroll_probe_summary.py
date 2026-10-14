#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
"""Long-rollout probe read-out: does the spin-up survive, and does a training-year IC help?

Companion to `polaris/polaris_longroll_probe.pbs`, which rolls the production checkpoint
from the same frame of a TRAINING year and a TEST year (Oct 1 00:00 = frame 1092 on the
noleap calendar) for as many steps as one A100 holds (~200 = 50 days), scores each arm
with `score_rollout_nc.py`, and calls this to read the two curve bundles side by side.

Three questions, all raised by jesswan's protocol -- initialise Oct 2044, discard Oct-Dec,
score 2045-2049 against climatology:

1. SPIN-UP SURVIVAL.  `docs/2026-09-10_longroll_blowup_analysis.md` §2 extrapolates the
   median channel crossing 1.0x climatological std near step 95 and 1.41x near step 154,
   and blow-up is observed near step 500.  Her 92-day discard is 368 steps.  `summarize`
   prints where the crossings actually land, whether the curve is still decelerating, and
   which channels are already past the 3x blow-up clause at the last lead -- i.e. the
   state the run is in when scoring would begin.
2. TRAINING-YEAR CONTAMINATION.  If the model memorised 2044 step pairs, the 2044 arm
   should beat the 2048 arm at the earliest leads and the gap should close.  n=1 per arm,
   same calendar date, different weather: a gap is *indicative*, not a measurement, and is
   flagged only when it is both large and broad across channels.
3. FORCING REPETITION.  `forcing-check` compares the 7 forcing channels between year files
   at the same frame.  If they are identical, the frozen-annual-cycle finding in CHANGELOG
   holds for the packed data, and 2045-2049 differ from training only in the atmospheric
   state -- which bears on what "held-out years" means for a forced climate run.

PASS = `LONGROLL_PROBE_OK` on stdout (CLAUDE.md #14).  Torch-free: numpy, h5py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

# Prereg §3 anchors, reused so the numbers here mean the same as the K=56 read-out.
EARLY_WINDOW_H = (30, 126)
LATE_WINDOW_H = 96           # slope over the final 96 h of whatever horizon was reached
BLOWUP_NRMSE = 3.0           # score_rollout_nc.NRMSE_CHANNEL_BLOWUP
CONTAM_LEADS_H = (6, 24, 72, 168, 336)
CONTAM_RATIO_MAX = 0.90      # train/test median-NRMSE ratio below this ...
CONTAM_FRAC_MIN = 0.75       # ... on more than this fraction of channels => flag


def crossing_hours(lead_h: np.ndarray, med: np.ndarray, level: float):
    """First lead (hours) at which the median curve reaches `level`, or None if it never does."""
    idx = np.flatnonzero(np.nan_to_num(np.asarray(med), nan=-np.inf) >= level)
    return int(lead_h[idx[0]]) if idx.size else None


def _slope(lead_h: np.ndarray, med: np.ndarray, h0: int, h1: int) -> float:
    """Secant slope of `med` per hour between the leads nearest h0 and h1."""
    i0 = int(np.argmin(np.abs(lead_h - h0)))
    i1 = int(np.argmin(np.abs(lead_h - h1)))
    if i1 == i0:
        return float("nan")
    return float((med[i1] - med[i0]) / (lead_h[i1] - lead_h[i0]))


def slope_ratio(lead_h: np.ndarray, med: np.ndarray) -> float:
    """Late-window slope over the prereg early-window slope: <1 decelerating, ~1 linear, >1 accelerating.

    The early window is the prereg's 30-126 h so the ratio is comparable with the K=56
    read-out's `R_slope`; the late window is the last `LATE_WINDOW_H` of the horizon
    actually reached, since the probe's K is set by GPU memory, not by design.
    """
    lead_h = np.asarray(lead_h)
    early = _slope(lead_h, med, *EARLY_WINDOW_H)
    late = _slope(lead_h, med, int(lead_h[-1]) - LATE_WINDOW_H, int(lead_h[-1]))
    return float("nan") if not np.isfinite(early) or early == 0 else late / early


def blowup_channels(nrmse_last: np.ndarray, chan: Sequence[str], thresh: float = BLOWUP_NRMSE):
    """Channels past the blow-up clause at the final lead, worst first, as (nrmse, name)."""
    vals = np.nan_to_num(np.asarray(nrmse_last, dtype=float), nan=0.0)
    return sorted(((float(vals[i]), chan[i]) for i in np.flatnonzero(vals > thresh)), reverse=True)


def worst_channels(nrmse_last: np.ndarray, chan: Sequence[str], n: int = 5):
    """The `n` channels with the largest final-lead NRMSE, worst first."""
    vals = np.nan_to_num(np.asarray(nrmse_last, dtype=float), nan=0.0)
    order = np.argsort(vals)[::-1][:n]
    return [(float(vals[i]), chan[i]) for i in order]


def contamination_table(lead_h: np.ndarray, nrmse_train: np.ndarray, nrmse_test: np.ndarray,
                        leads: Sequence[int] = CONTAM_LEADS_H):
    """Per requested lead: (lead_h, median train/test NRMSE ratio, fraction of channels train < test).

    NRMSE rather than RMSE because the two arms verify against different years, whose
    anomaly amplitudes differ; NRMSE divides that out.  Leads not present are skipped.
    """
    rows = []
    lead_h = np.asarray(lead_h)
    for h in leads:
        k = np.flatnonzero(lead_h == h)
        if not k.size:
            continue
        a, b = np.asarray(nrmse_train[k[0]], float), np.asarray(nrmse_test[k[0]], float)
        ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
        if not ok.any():
            continue
        rows.append((int(h), float(np.median(a[ok] / b[ok])), float(np.mean(a[ok] < b[ok]))))
    return rows


def contamination_flag(rows, ratio_max: float = CONTAM_RATIO_MAX, frac_min: float = CONTAM_FRAC_MIN) -> bool:
    """True only when the training-year arm is substantially AND broadly better at the first lead."""
    if not rows:
        return False
    _, ratio, frac = rows[0]
    return ratio < ratio_max and frac > frac_min


def load_curves(path: Path):
    """Read the `(K, C)` mean curves `score_rollout_nc.py` writes to `k56_metrics.h5`."""
    import h5py

    with h5py.File(path, "r") as f:
        lead_h = np.asarray(f["lead_hours"][()], dtype=np.int64)
        chan = [c.decode() if isinstance(c, bytes) else str(c) for c in f["channel"][()]]
        cur = {k: np.asarray(f[k][()], dtype=float)
               for k in ("nrmse_mean", "vr_mean", "acc_mean", "bias_mean", "a_truth_mean", "rmse_mean")}
    return lead_h, chan, cur


def report_arm(name: str, lead_h: np.ndarray, chan: Sequence[str], cur: dict) -> dict:
    """Print one arm's survival read-out and return the numbers for the caller."""
    med = np.nanmedian(cur["nrmse_mean"], axis=1)
    medvr = np.nanmedian(cur["vr_mean"], axis=1)
    medacc = np.nanmedian(cur["acc_mean"], axis=1)
    c10 = crossing_hours(lead_h, med, 1.0)
    c141 = crossing_hours(lead_h, med, 1.414)
    r = slope_ratio(lead_h, med)
    bl = blowup_channels(cur["nrmse_mean"][-1], chan)
    print(f"LONGROLL_PROBE arm={name} final_lead_h={int(lead_h[-1])} ({lead_h[-1] / 24:.0f} d) "
          f"nrmse_final={med[-1]:.3f} vr_final={medvr[-1]:.3f} acc_final={medacc[-1]:.3f} "
          f"cross_1.0_h={c10} cross_1.41_h={c141} slope_ratio={r:.3f} n_blowup_channels={len(bl)}")
    # One line per 5 days so the shape of the curve is in the log, not just its ends.
    for k in range(0, len(lead_h), 20):
        print(f"  {name} lead={int(lead_h[k]):5d}h ({lead_h[k] / 24:4.0f} d)  "
              f"median nrmse={med[k]:.3f}  vr={medvr[k]:.3f}  acc={medacc[k]:.3f}")
    for v, c in worst_channels(cur["nrmse_mean"][-1], chan):
        k = chan.index(c)
        print(f"  {name} worst channel {c}: nrmse={v:.2f}  vr={cur['vr_mean'][-1, k]:.2f}  "
              f"bias/a_truth={cur['bias_mean'][-1, k] / max(cur['a_truth_mean'][-1, k], 1e-12):+.2f}")
    return {"cross_1.0_h": c10, "cross_1.41_h": c141, "slope_ratio": r,
            "n_blowup": len(bl), "nrmse_final": float(med[-1]), "vr_final": float(medvr[-1])}


def cmd_summarize(args) -> int:
    lt, ct, cur_t = load_curves(args.train_scores)
    ls, cs, cur_s = load_curves(args.test_scores)
    if ct != cs:
        print("ERROR LONGROLL_CHANNEL_MISMATCH: the two arms disagree on channel order", file=sys.stderr)
        return 2
    print(f"probe horizon: train arm {len(lt)} leads, test arm {len(ls)} leads "
          f"(jesswan's discard is 368 steps = 2208 h; anything shorter is a trend, not a verdict)")
    report_arm("train_year", lt, ct, cur_t)
    report_arm("test_year", ls, cs, cur_s)
    n = min(len(lt), len(ls))
    rows = contamination_table(lt[:n], cur_t["nrmse_mean"][:n], cur_s["nrmse_mean"][:n])
    for h, ratio, frac in rows:
        print(f"CONTAMINATION lead={h}h train/test_median_nrmse={ratio:.3f} frac_channels_train_better={frac:.2f}")
    print(f"CONTAMINATION_FLAG={contamination_flag(rows)} "
          f"(flag needs ratio<{CONTAM_RATIO_MAX} on >{CONTAM_FRAC_MIN:.0%} of channels at the first lead; "
          f"n=1 per arm -- indicative only)")
    if n < 56:
        print("ERROR LONGROLL_PROBE_TOO_SHORT: fewer than 56 leads scored", file=sys.stderr)
        return 3
    print("LONGROLL_PROBE_OK")
    return 0


def cmd_forcing_check(args) -> int:
    import h5py

    paths = [Path(p) for p in args.files]
    handles = [h5py.File(p, "r") for p in paths]
    try:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in handles[0]["channel_forcing"][()]]
        identical = True
        for fr in args.frames:
            base = np.asarray(handles[0]["forcing"][fr], dtype=np.float64)
            state0 = np.asarray(handles[0]["fields_state"][fr, :3], dtype=np.float64)
            for h, p in zip(handles[1:], paths[1:]):
                d = np.abs(np.asarray(h["forcing"][fr], dtype=np.float64) - base).max(axis=(1, 2))
                for nm, v in zip(names, d):
                    print(f"FORCING_CHECK frame={fr} {paths[0].name}_vs_{p.name} channel={nm} max_abs_diff={v:.3e}")
                identical &= bool((d <= args.atol).all())
                ds = np.abs(np.asarray(h["fields_state"][fr, :3], dtype=np.float64) - state0).max()
                print(f"STATE_CHECK   frame={fr} {paths[0].name}_vs_{p.name} first3_state_channels_max_abs_diff={ds:.3e}")
        print(f"FORCING_REPEATS_ANNUAL_CYCLE={identical} (all {len(names)} forcing channels identical across "
              f"{len(paths)} files at frames {list(args.frames)}, atol={args.atol:g})")
    finally:
        for h in handles:
            h.close()
    return 0


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize", help="read two k56_metrics.h5 bundles (train-year arm, test-year arm)")
    s.add_argument("--train-scores", required=True, type=Path)
    s.add_argument("--test-scores", required=True, type=Path)
    s.set_defaults(fn=cmd_summarize)
    f = sub.add_parser("forcing-check", help="compare forcing channels across year files at the same frames")
    f.add_argument("files", nargs="+", help="two or more packed year h5 files; the first is the reference")
    f.add_argument("--frames", type=int, nargs="+", default=[0, 1092, 1459])
    f.add_argument("--atol", type=float, default=1e-6)
    f.set_defaults(fn=cmd_forcing_check)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
