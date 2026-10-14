#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
"""Score the n_future ladder arms against the base checkpoint, per lead time.

WHY THIS EXISTS. The tables in `2026-09-11_nfuture_ladder_result.md` were
computed ad hoc and never committed, so no later arm could be added to them
without re-deriving the protocol from prose. Two arms finished after that doc
was written (the n_f=4 replication 7607361, and D1 7606726), and re-deriving
"median over channels of the per-channel ratio" by hand is exactly the kind of
step where a ladder silently stops being comparable.

PROTOCOL, fixed here so it cannot drift:
  - RMSE / L1 : median over the 101 channels of (arm/base - 1), in percent.
                NEGATIVE IS BETTER.
  - ACC       : median over channels of (arm - base), an absolute difference,
                because ACC is already a correlation in [-1, 1] and a ratio
                near zero explodes. POSITIVE IS BETTER.
  - "better on both" counts channels improving RMSE *and* ACC at that lead.
    This is the blurring check: a hedged forecast wins RMSE and loses ACC, so
    neither metric alone can separate skill from variance damping.

Scorecard runs write one `scores/metrics_ep0000.h5` holding (21 leads x 101
channels) for each of RMSE / ACC / L1; a TRAINING run writes one such file per
epoch. The two are NOT interchangeable -- a training run validates on its own
sample count, the scorecard fixes 512 samples at valid_autoreg_steps=20 -- so
mixing them in one table is a category error and `--source` makes the choice
explicit rather than implicit in a path.
"""

import argparse
import os
import sys

import h5py
import numpy as np

EXPROOT = os.environ.get(
    "EXPROOT",
    "/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_mn_scaling/e3sm_mn_scaling",
)

# The ladder as run. Base first -- every other row is a ratio against it.
BASE = "score_prod1n_b32_sgdr_va20_pl"
ARMS = [
    ("C1  n_f=1 24ep b16", "score_c1_rollout_full_b16_va20_pl"),
    ("n_f=1  1ep b8", "score_nf1_proxy_b8_r1_va20_nf"),
    ("n_f=1  1ep b8 (diag r1)", "score_nf1_diag_b8_r1_va20_nf"),
    ("n_f=3  1ep b8", "score_nf3_proxy_b8_r1_va20_nf"),
    ("n_f=4  1ep b8  r1", "score_nf4_proxy_b8_r1_va20_nf"),
    ("n_f=4  1ep b8  r2", "score_nf4_proxy_b8_r2_va20_nf"),
]
# Snapshot ensemble members: same recipe, different epoch. Their spread against
# base is the null distribution -- the bar any arm has to clear.
SNAPSHOTS = [
    ("snap_e203", "score_snap_e203_va20_nf"),
    ("snap_e223", "score_snap_e223_va20_nf"),
    ("snap_e243", "score_snap_e243_va20_nf"),
]


def load(run_dir, epoch=None):
    """Return {metric: (leads, channels, data[lead, channel])} for one run."""
    sdir = os.path.join(EXPROOT, run_dir, "scores")
    if not os.path.isdir(sdir):
        return None
    files = sorted(f for f in os.listdir(sdir) if f.endswith(".h5"))
    if not files:
        return None
    fname = files[-1] if epoch is None else "metrics_ep%04d.h5" % epoch
    out = {}
    with h5py.File(os.path.join(sdir, fname), "r") as f:
        for m in ("RMSE", "ACC", "L1"):
            if m not in f:
                continue
            chans = [c.decode() if isinstance(c, bytes) else str(c) for c in f[m]["channel"][:]]
            out[m] = (f[m]["lead_time"][:], chans, f[m]["metric_data"][:])
    out["_file"] = os.path.join(sdir, fname)
    return out


def summarize(arm, base):
    """Per-lead RMSE %, ACC delta, and channel win counts against base."""
    leads, _, r_a = arm["RMSE"]
    _, _, r_b = base["RMSE"]
    _, _, a_a = arm["ACC"]
    _, _, a_b = base["ACC"]
    rmse_pct = np.median(r_a / r_b - 1.0, axis=1) * 100.0
    acc_delta = np.median(a_a - a_b, axis=1)
    rmse_win = (r_a < r_b).sum(axis=1)
    acc_win = (a_a > a_b).sum(axis=1)
    both_win = ((r_a < r_b) & (a_a > a_b)).sum(axis=1)
    return leads, rmse_pct, acc_delta, rmse_win, acc_win, both_win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["scorecard", "training"], default="scorecard")
    ap.add_argument("--extra", action="append", default=[],
                    help="label=run_dir to append to the ladder")
    ap.add_argument("--curve", action="append", default=[],
                    help="label=run_dir to print the full per-lead curve for")
    args = ap.parse_args()

    base = load(BASE)
    if base is None:
        print("ERROR NO_BASE %s" % os.path.join(EXPROOT, BASE))
        return 2
    leads = base["RMSE"][0]

    arms = list(ARMS)
    for spec in args.extra:
        # rpartition, not partition: arm labels carry '=' ("n_f=4 1ep b8").
        label, _, run = spec.rpartition("=")
        arms.append((label, run))

    # sigma_0: how far apart do same-recipe snapshots land? Anything inside this
    # is indistinguishable from re-running the same experiment.
    snap_last = []
    for label, run in SNAPSHOTS:
        d = load(run)
        if d is None:
            continue
        _, rp, _, _, _, _ = summarize(d, base)
        snap_last.append(rp[-1])
    bar = None
    if len(snap_last) > 1:
        sigma0 = float(np.std(snap_last, ddof=1))
        bar = 2.0 * sigma0
        print("sigma_0 = %.2f %% (n=%d snapshots, RMSE %% at lead %d h); "
              "bar = 2*sigma_0 = %.2f %%" % (sigma0, len(snap_last), leads[-1], bar))
    print()

    hdr = "%-26s %10s %10s %6s %12s %10s" % (
        "arm", "lead-%d" % leads[0], "best", "@h", "lead-%d" % leads[-1], "ACC@%dh" % leads[-1])
    print(hdr)
    print("-" * len(hdr))
    for label, run in arms:
        d = load(run)
        if d is None:
            print("%-26s  MISSING (%s)" % (label, run))
            continue
        lt, rp, ad, rw, aw, bw = summarize(d, base)
        i = int(np.argmin(rp))
        flag = ""
        if bar is not None:
            flag = " *" if rp[-1] < -bar else ("  " if rp[-1] < 0 else " x")
        print("%-26s %9.2f%% %9.2f%% %6d %11.2f%% %+10.5f  both %d/101%s"
              % (label, rp[0], rp[i], lt[i], rp[-1], ad[-1], bw[-1], flag))

    for spec in args.curve:
        # rpartition, not partition: arm labels carry '=' ("n_f=4 1ep b8").
        label, _, run = spec.rpartition("=")
        d = load(run)
        if d is None:
            print("\n%s: MISSING (%s)" % (label, run))
            continue
        lt, rp, ad, rw, aw, bw = summarize(d, base)
        print("\n=== per-lead curve: %s ===" % label)
        print("%6s %10s %12s %8s %8s %8s" % ("lead_h", "RMSE%", "dACC", "rmse_w", "acc_w", "both"))
        for k in range(len(lt)):
            print("%6d %9.2f%% %+11.5f %8d %8d %8d"
                  % (lt[k], rp[k], ad[k], rw[k], aw[k], bw[k]))

    return 0


if __name__ == "__main__":
    sys.exit(main())
