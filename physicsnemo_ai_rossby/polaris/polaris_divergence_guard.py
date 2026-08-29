#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
"""Kill a training job when its epoch loss regresses past the running minimum.

WHY THIS EXISTS
---------------
On 2026-08-29 the 48-node production run diverged at epoch 11 — loss went
0.0397 -> 1.067 -> 2.442 the moment warmup ended at peak LR 1.46e-3 — and then
kept training for 32 more epochs, recovering only to 0.2483 (6x worse than its
own epoch-10 minimum). It cost ~650 node-hours before a human looked.

Nothing flagged it. That is the point:

  * loss was DESCENDING MONOTONICALLY for the last 20 epochs, so a
    previous-epoch comparison looks healthy;
  * ``gpu_busy_frac`` stayed at 0.99;
  * no exception, no NaN, no NCCL error, rc=0 throughout.

Only the loss-vs-epoch SHAPE reveals it, and only against the running MINIMUM
— never against the previous epoch. That is the comparison this guard makes.

It is deliberately NOT a scheduler change. ``ReduceLROnPlateau`` would be the
reactive fix, but this trainer only offers OneCycleLR / CosineAnnealingLR /
LinearWarmupCosineAnnealingLR (train_loop.py:234-274), and cosine gives almost
no protection where it matters: at the epoch-11 failure the LR was **1.0% below
peak**, because cosine is flat near its maximum by construction.

USAGE (from the PBS launcher, backgrounded before mpiexec):
    python polaris_divergence_guard.py --log "${LOG}" --pid $$ \
        --factor 3.0 --patience 1 &

PASS/FAIL is left to the launcher: this only prints a greppable line and
signals the process group. A guard that silently changes a run is worse than
no guard.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import sys
import time

# Matches the trainer's own per-epoch line, e.g.
#   [train][INFO] - Epoch 14 Metrics: lr =  1.433e-03, ..., loss =  2.442e+00, ...
# Anchored on BOTH "Epoch N Metrics" and "loss =" so a reworded log line makes
# the guard go silent-but-visible (see --require-progress) rather than
# silently never firing.
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s+Metrics:\s*lr\s*=\s*([0-9.eE+-]+).*?\bloss\s*=\s*([0-9.eE+-]+)", re.S
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def scan(path: str) -> list[tuple[int, float]]:
    """Every (epoch, loss) pair currently in the log, in file order."""
    try:
        with open(path, errors="replace") as fh:
            text = ANSI_RE.sub("", fh.read())
    except FileNotFoundError:
        return [], {}
    out, lrs = [], {}
    for m in EPOCH_RE.finditer(text):
        try:
            ep = int(m.group(1))
            out.append((ep, float(m.group(3))))
            lrs[ep] = float(m.group(2))
        except ValueError:
            continue
    return out, lrs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", required=True, help="the tee'd training log")
    p.add_argument("--pid", type=int, default=0,
                   help="process group to signal on divergence (0 = report only)")
    p.add_argument("--qdel-jobid", default="",
                   help="PBS job id to qdel on divergence. PREFERRED over --pid on a "
                        "scheduler: killing our own process group would also kill the "
                        "job script before it can run the parser and print its verdict, "
                        "whereas qdel is the scheduler's own mechanism and needs no "
                        "PID plumbing through the mpiexec|tee pipeline.")
    p.add_argument("--factor", type=float, default=3.0,
                   help="fire when loss > factor x running minimum (default 3.0)")
    p.add_argument("--patience", type=int, default=1,
                   help="consecutive bad epochs required before firing")
    p.add_argument("--min-epochs", type=int, default=3,
                   help="ignore the first N epochs; early loss is legitimately noisy")
    p.add_argument("--expect-lr", type=float, default=0.0,
                   help="if >0, FIRE when the first observed epoch LR differs from this "
                        "by more than --lr-tol. Catches the 2026-08-29 failure where a "
                        "checkpoint resume silently restored the OLD lr and the requested "
                        "one never applied — the launcher echoed the requested value, which "
                        "was true and meaningless.")
    p.add_argument("--lr-tol", type=float, default=0.2,
                   help="fractional tolerance for --expect-lr (default 20%%)")
    p.add_argument("--poll", type=float, default=60.0)
    p.add_argument("--require-progress-s", type=float, default=0.0,
                   help="if >0, warn when no new epoch is seen for this long "
                        "(catches a reworded log line silently disabling the guard)")
    args = p.parse_args(argv)

    checked_lr = [False]
    lrs: dict[int, float] = {}
    best = float("inf")
    best_ep = None
    bad = 0
    seen = 0
    last_new = time.time()
    print("DIVERGENCE_GUARD armed: factor=%.2f patience=%d min_epochs=%d log=%s"
          % (args.factor, args.patience, args.min_epochs, args.log), flush=True)

    while True:
        rows, lrs = scan(args.log)
        if len(rows) > seen:
            last_new = time.time()
            for ep, loss in rows[seen:]:
                if args.expect_lr > 0 and not checked_lr[0]:
                    checked_lr[0] = True
                    got = lrs.get(ep)
                    if got is not None and abs(got - args.expect_lr) > args.lr_tol * args.expect_lr:
                        print("ERROR LR_MISMATCH epoch=%d requested=%.4e observed=%.4e — the "
                              "requested LR did NOT take effect (a checkpoint resume restores "
                              "the optimizer's lr). Use warm_start instead of resume."
                              % (ep, args.expect_lr, got), flush=True)
                        return fire(args, ep, loss, best, best_ep)
                    print("LR_CHECK_OK epoch=%d observed=%.4e (requested %.4e)"
                          % (ep, got if got else float("nan"), args.expect_lr), flush=True)
                if loss != loss:  # NaN
                    print("DIVERGENCE_GUARD_FIRED epoch=%d loss=NaN" % ep, flush=True)
                    return fire(args, ep, loss, best, best_ep)
                if ep <= args.min_epochs:
                    best, best_ep = (loss, ep) if loss < best else (best, best_ep)
                    continue
                if loss < best:
                    best, best_ep, bad = loss, ep, 0
                    continue
                # Compared against the running MINIMUM, not the previous epoch —
                # the 2026-08-29 divergence descended monotonically for 20 epochs
                # while sitting 6x above its own minimum.
                if loss > args.factor * best:
                    bad += 1
                    print("DIVERGENCE_GUARD_WARN epoch=%d loss=%.5f > %.2f x min "
                          "%.5f (epoch %s), strike %d/%d"
                          % (ep, loss, args.factor, best, best_ep, bad, args.patience),
                          flush=True)
                    if bad >= args.patience:
                        return fire(args, ep, loss, best, best_ep)
                else:
                    bad = 0
            seen = len(rows)
        elif args.require_progress_s > 0 and time.time() - last_new > args.require_progress_s:
            print("DIVERGENCE_GUARD_STALE: no epoch line matched in %.0f s. Either the run "
                  "is slower than expected, or the trainer's epoch log line was reworded and "
                  "this guard is now inert — check EPOCH_RE." % args.require_progress_s,
                  flush=True)
            last_new = time.time()
        time.sleep(args.poll)


def fire(args, ep, loss, best, best_ep) -> int:
    print("ERROR DIVERGENCE_GUARD_FIRED epoch=%d loss=%.5f best=%.5f (epoch %s) "
          "ratio=%.1fx" % (ep, loss, best, best_ep,
                           (loss / best) if best else float("nan")), flush=True)
    print("  The run is training but is above its own best by more than the "
          "allowed factor. Restart from the epoch-%s checkpoint at a lower LR; do "
          "NOT resume the damaged state." % best_ep, flush=True)
    if args.qdel_jobid:
        rc = os.system("qdel %s" % args.qdel_jobid)
        print("  qdel %s -> rc=%d" % (args.qdel_jobid, rc), flush=True)
    elif args.pid:
        try:
            os.killpg(os.getpgid(args.pid), signal.SIGTERM)
            print("  signalled process group of pid %d" % args.pid, flush=True)
        except Exception as exc:  # noqa: BLE001
            print("  WARN could not signal pid %d: %r" % (args.pid, exc), flush=True)
    else:
        print("  REPORT-ONLY (no --qdel-jobid/--pid): the run continues. "
              "Node-hours keep burning until someone looks.", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
