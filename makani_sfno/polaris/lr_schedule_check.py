#!/usr/bin/env python3
"""Gate S1a (fine-tune stability handoff §3a.1): how makani composes warmup + cosine.

Three independent readings of the per-epoch learning rate; all must agree with
the closed form before SCHED_TMAX is chosen:

  simulate  makani's own ``Driver.get_scheduler`` (installed venv, its torch), a
            dummy optimizer, stepped once per epoch as ``deterministic_trainer.py:380``
            does. Prints the LR each epoch trains at.
  ckpt      the ``optimizer_state_dict`` LR saved in real checkpoints. A checkpoint
            is written AFTER ``scheduler.step()`` (``:380`` then ``:399``), so the LR
            stored at the end of epoch e is the LR epoch e+1 trains at.
  formula   epoch 1 = lr * lr_start (warmup); epoch e >= 2 runs cosine step t = e - 2:
            eta_min + (lr - eta_min) * (1 + cos(pi * t / T_max)) / 2.

    python polaris/lr_schedule_check.py simulate --epochs 24 --tmax 22 23
    python polaris/lr_schedule_check.py ckpt --tmax 100 RUN/training_checkpoints/ckpt_mp0_v*.tar

Output: one ``LR`` line per epoch and ``LR_SCHEDULE_OK`` or ``ERROR LR_SCHEDULE_MISMATCH``.
Run on a compute node (imports torch + makani).
"""
from __future__ import annotations

import argparse
import math
import re
import sys

LR, LR_START, ETA_MIN, WARMUP = 4.0e-4, 0.01, 1.0e-6, 1
RTOL = 1e-6


def formula(epoch: int, tmax: int, lr=LR, lr_start=LR_START, eta_min=ETA_MIN) -> float:
    """LR that epoch ``epoch`` (1-based) trains at, for a 1-epoch warmup."""
    if epoch <= WARMUP:
        return lr * lr_start
    t = epoch - 1 - WARMUP
    return eta_min + (lr - eta_min) * (1 + math.cos(math.pi * t / tmax)) / 2


class _Params(dict):
    __getattr__ = dict.__getitem__


def simulate(epochs: int, tmax: int) -> list[float]:
    import torch
    from makani.utils.driver import Driver

    p = _Params(scheduler="CosineAnnealingLR", scheduler_T_max=tmax, scheduler_min_lr=ETA_MIN,
                lr_warmup_steps=WARMUP, lr_start=LR_START, lr=LR)
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=LR)
    sched = Driver.get_scheduler(None, opt, p)          # does not touch self
    out = []
    for _ in range(epochs):
        out.append(opt.param_groups[0]["lr"])            # the LR this epoch trains at
        opt.step()
        sched.step()                                     # once per epoch, as :380
    return out


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= RTOL * max(abs(b), 1e-12)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--epochs", type=int, default=24)
    s.add_argument("--tmax", type=int, nargs="+", required=True)
    c = sub.add_parser("ckpt")
    c.add_argument("--tmax", type=int, required=True)
    c.add_argument("paths", nargs="+")
    a = ap.parse_args(argv)
    bad = 0

    if a.cmd == "simulate":
        import torch
        print(f"torch={torch.__version__}")
        for tmax in a.tmax:
            lrs = simulate(a.epochs, tmax)
            at_min = [e for e, v in enumerate(lrs, 1) if _close(v, ETA_MIN)]
            for e, v in enumerate(lrs, 1):
                ok = _close(v, formula(e, tmax))
                bad += not ok
                print(f"LR sim tmax={tmax} epoch={e:2d} lr={v:.6e} formula={formula(e, tmax):.6e}"
                      f"{'' if ok else '  MISMATCH'}")
            print(f"SUMMARY sim tmax={tmax} epochs_at_min={at_min} last_epoch_lr={lrs[-1]:.6e}")
    else:
        import torch
        rows = []
        for path in a.paths:
            try:
                ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            except Exception:  # noqa: BLE001 -- legacy (non-zip) format cannot mmap
                ck = torch.load(path, map_location="cpu", weights_only=False)
            ep = int(ck["epoch"])
            lr = float(ck["optimizer_state_dict"]["param_groups"][0]["lr"])
            rows.append((ep, lr, path))
            del ck
        for ep, lr, path in sorted(rows):
            nxt = ep + 1                                  # saved after step => next epoch's LR
            f = formula(nxt, a.tmax)
            ok = _close(lr, f)
            alt = formula(nxt + 1, a.tmax)                # the t = e - 1 hypothesis
            bad += not ok
            print(f"LR ckpt epoch={ep:2d} saved_lr(=epoch {nxt})={lr:.9e} formula(t=e-2)={f:.9e} "
                  f"alt(t=e-1)={alt:.9e} {'ok' if ok else 'MISMATCH'} {re.sub(r'.*/', '', path)}")
    if bad:
        print(f"ERROR LR_SCHEDULE_MISMATCH n={bad}")
        return 1
    print("LR_SCHEDULE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
