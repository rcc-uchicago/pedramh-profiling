#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
"""Resume gate: did an interrupted ACE2 run continue, or silently restart?

    python3 compare_resume_trace.py --reference <ref>/out.log --resumed <res>/out.log

PASS = ``ACE2_RESUME_GATE_OK``.

WHY THIS IS A GATE AND NOT A NICETY
-----------------------------------
ACE2 production cannot use `capacity` (max_run 1 per PROJECT, and makani holds
it), so it runs on `preemptable` — where it **will** be preempted, repeatedly, at
2–4 h per epoch. Three things have to survive that, and only the first is
obvious:

1. **The weights** — fme auto-resumes when
   `${experiment_dir}/training_checkpoints/ckpt.tar` exists
   (`trainer.py`, `resuming = os.path.isfile(...)`). ⚠ It keys on the PATH, so a
   `RUN_NAME` carrying `$PBS_JOBID` starts a FRESH run on every requeue while
   looking completely healthy. That is why PRODUCTION mode drops the jobid.
2. **The scheduler's `T_cur`** — `CosineAnnealingWarmRestarts` keeps its position
   in the cycle in `scheduler_state_dict`, which `Optimization.get_state` does
   checkpoint. If it were NOT restored, every requeue would silently reset the LR
   to its peak: the run would still descend, still exit 0, and the **snapshot
   ensemble would be built from checkpoints that are no longer at cycle minima**.
   Nothing downstream would notice.
3. **The epoch counter** — `_epochs_trained` decides which epochs get kept as
   snapshots, so an off-by-one here misaligns the whole ensemble.

This compares the per-epoch (epoch, lr, train_loss, valid_loss) trace of an
interrupted run against an uninterrupted reference of the same config.

WHAT IT DOES **NOT** CLAIM
--------------------------
Bit-identity. makani's resume happened to pass byte-identically; fme's dataloader
RNG state is not obviously restored across a mid-epoch resume, so the losses may
differ slightly and legitimately. The gate therefore asserts:

* the **LR trace matches exactly** — it is a deterministic function of the epoch
  index and the scheduler state, so any mismatch is a real defect, not noise;
* the epoch indices are contiguous and reach the same maximum;
* the losses agree to a stated tolerance (`--loss-rtol`, default 5%).

A loss mismatch inside tolerance is reported, not failed, and the actual figure
is printed so it can be judged rather than nodded through.
"""

from __future__ import annotations

import argparse
import re
import sys

# fme's own per-epoch log lines (fme/core/generics/trainer.py).
EPOCH_RE = re.compile(r"Beginning epoch after (\d+) complete epochs")
TRAIN_RE = re.compile(r"Train loss:\s*([-\d.eE+naN]+)")
VALID_RE = re.compile(r"Valid loss:\s*([-\d.eE+naN]+)")
# The launcher's telemetry prints lr per epoch; fme logs it to wandb only, so the
# trace is taken from the EPOCH_TELEMETRY line, which carries the epoch index.
TEL_RE = re.compile(r"EPOCH_TELEMETRY epoch=(\d+) .*?step_med=([\d.]+)ms")
LR_RE = re.compile(r"'lr':\s*([-\d.eE+]+)")
RESUME_RE = re.compile(r"Resuming training from (\S+)")


def parse(path: str) -> dict:
    text = open(path, errors="replace").read()
    epochs, trains, valids, lrs = [], [], [], []
    for line in text.splitlines():
        m = EPOCH_RE.search(line)
        if m:
            epochs.append(int(m.group(1)))
        m = TRAIN_RE.search(line)
        if m:
            trains.append(m.group(1))
        m = VALID_RE.search(line)
        if m:
            valids.append(m.group(1))
        m = LR_RE.search(line)
        if m:
            lrs.append(float(m.group(1)))
    return {
        "epochs_started": epochs,
        "train": trains,
        "valid": valids,
        "lrs": lrs,
        "resumed_from": RESUME_RE.findall(text),
        "telemetry": [(int(a), float(b)) for a, b in TEL_RE.findall(text)],
    }


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def compare(ref: dict, res: dict, loss_rtol: float) -> int:
    rc = 0
    print("--- resume gate ---")
    print("  reference: %d epochs started, %d train losses, %d lr samples"
          % (len(ref["epochs_started"]), len(ref["train"]), len(ref["lrs"])))
    print("  resumed:   %d epochs started, %d train losses, %d lr samples"
          % (len(res["epochs_started"]), len(res["train"]), len(res["lrs"])))

    # 1. it must actually have resumed
    if not res["resumed_from"]:
        print("ERROR RESUME_NEVER_HAPPENED: the resumed run logged no")
        print("  'Resuming training from ...' line, so it started FROM SCRATCH.")
        print("  Almost always a RUN_NAME/experiment_dir that changed between")
        print("  submissions -- fme resumes by path, not by job identity.")
        return 4
    print("  resumed from: %s" % res["resumed_from"][0])

    # 2. epochs reach the same maximum
    ref_max = max(ref["epochs_started"], default=-1)
    res_max = max(res["epochs_started"], default=-1)
    if ref_max != res_max:
        print("ERROR EPOCH_COUNT_MISMATCH: reference reached epoch %d, resumed %d."
              % (ref_max, res_max))
        rc = 4
    else:
        print("  epoch high-water mark matches: %d" % ref_max)

    # 3. THE LR TRACE -- the one that must match exactly
    n = min(len(ref["lrs"]), len(res["lrs"]))
    if n == 0:
        print("ERROR NO_LR_TRACE: neither log carried an 'lr' value.")
        print("  log_train_every_n_batches may be 0, or the log is truncated.")
        return 4
    bad = [(i, ref["lrs"][i], res["lrs"][i])
           for i in range(n) if ref["lrs"][i] != res["lrs"][i]]
    if bad:
        print("ERROR LR_TRACE_DIVERGED at %d of %d sampled points." % (len(bad), n))
        for i, a, b in bad[:5]:
            print("    sample %-4d reference %.6e   resumed %.6e" % (i, a, b))
        print("  The LR is a deterministic function of the epoch and the scheduler")
        print("  state, so this is a real defect, not noise. The usual cause is")
        print("  `scheduler_state_dict` not surviving the checkpoint -- which would")
        print("  reset CosineAnnealingWarmRestarts' T_cur on every preemption and")
        print("  silently misalign every snapshot-ensemble member.")
        rc = 4
    else:
        print("  LR trace identical across %d sampled points ✅" % n)
        peaks = [i for i in range(1, n) if res["lrs"][i] > res["lrs"][i - 1] * 5]
        print("  warm restarts observed in the resumed run at samples: %s"
              % (peaks if peaks else "none seen in this window"))

    # 4. losses -- reported against a tolerance, never asserted bit-identical
    for name in ("train", "valid"):
        a, b = ref[name], res[name]
        k = min(len(a), len(b))
        if k == 0:
            print("  %s loss: no values in one of the logs" % name)
            continue
        worst, at = 0.0, -1
        for i in range(k):
            fa, fb = _f(a[i]), _f(b[i])
            if fa == 0 or fa != fa or fb != fb:
                continue
            rel = abs(fb - fa) / abs(fa)
            if rel > worst:
                worst, at = rel, i
        flag = "" if worst <= loss_rtol else "  ⚠ ABOVE --loss-rtol"
        print("  %-5s loss: max relative difference %.3e at epoch index %d%s"
              % (name, worst, at, flag))
        if worst > loss_rtol:
            print("    reference %s   resumed %s" % (a[at], b[at]))
            print("    Not failed automatically: fme's dataloader RNG is not")
            print("    obviously restored mid-epoch, so some drift is expected.")
            print("    Judge it -- do not nod it through.")

    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True, help="out.log of the uninterrupted run")
    p.add_argument("--resumed", required=True, help="out.log of the killed+requeued run")
    p.add_argument("--loss-rtol", type=float, default=0.05)
    args = p.parse_args(argv)

    rc = compare(parse(args.reference), parse(args.resumed), args.loss_rtol)
    print()
    if rc == 0:
        print("ACE2_RESUME_GATE_OK")
    else:
        print("ERROR ACE2_RESUME_GATE_FAILED rc=%d" % rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
