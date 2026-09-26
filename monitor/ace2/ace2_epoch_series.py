#!/usr/bin/env python3
"""Recompute the per-epoch series of an fme (ACE2) run from its own files. Torch-free, numpy-free.

    python3 monitor/ace2/ace2_epoch_series.py [<exp_dir>]

Prints one row per completed epoch (end time, wall s, train loss, lr used in that epoch, best
validation loss after it), the expected lr from the warm-restart cosine, the mean epoch wall, and
the projection of how many epochs fit before a deadline. Reads only ``out.log`` and the key set of
``metrics/metrics.jsonl``; safe on the process-capped login node (one process, no numpy, Python 3.6).
"""
import argparse
import datetime as dt
import json
import math
import re
import sys

DEFAULT_EXP = "/eagle/projects/lighthouse-uchicago/members/mehta5/runs/ace2_polaris/ace2_prod_1n_b8"


def cosine_lr(epoch: int, peak: float, eta_min: float, t0: int) -> float:
    """LR fme prints at the end of ``epoch`` (per-epoch stepping, T_cur = (epoch-1) mod t0)."""
    t_cur = (epoch - 1) % t0
    return eta_min + (peak - eta_min) * (1 + math.cos(math.pi * t_cur / t0)) / 2


def parse_out_log(lines):
    rows, cur = {}, None
    for l in lines:
        m = re.search(r"Time taken for epoch (\d+) is ([0-9.]+) sec", l)
        if m:
            cur = int(m.group(1))
            rows.setdefault(cur, {})["wall_s"] = float(m.group(2))
            rows[cur]["t"] = l[:19]
        m = re.search(r"Train loss: ([0-9.eE+-]+|nan)", l)
        if m and cur is not None and "train" not in rows[cur]:
            rows[cur]["train"] = float(m.group(1))
        m = re.search(r"\blr: ([0-9.eE+-]+)", l)
        if m and cur is not None and "lr" not in rows[cur]:
            rows[cur]["lr"] = float(m.group(1))
        m = re.search(r"(\d+) complete epochs and 0 additional batches.*best_validation_loss ([0-9.]+)", l)
        if m:
            rows.setdefault(int(m.group(1)), {})["best_val"] = float(m.group(2))
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("exp_dir", nargs="?", default=DEFAULT_EXP)
    p.add_argument("--peak", type=float, default=3e-4)
    p.add_argument("--eta-min", type=float, default=1e-6)
    p.add_argument("--t0", type=int, default=9)
    p.add_argument("--epochs", type=int, default=27)
    p.add_argument("--start", default="2026-09-25T03:24:08", help="job stime (UTC)")
    p.add_argument("--walltime-h", type=float, default=72.0)
    args = p.parse_args(argv)

    lines = open(args.exp_dir + "/out.log", errors="replace").read().splitlines()
    rows = parse_out_log(lines)
    print("epoch  end_time(UTC)        wall_s   train     lr        expected_lr  best_val")
    walls = []
    for e in sorted(rows):
        r = rows[e]
        if "wall_s" in r:
            walls.append(r["wall_s"])
        lr = r.get("lr", float("nan"))
        exp = cosine_lr(e, args.peak, args.eta_min, args.t0)
        flag = "" if (math.isnan(lr) or abs(lr - exp) <= 1e-9 * max(1.0, abs(exp)) + 1e-12) else "  <-- OFF THE COSINE"
        print(f"{e:5d}  {r.get('t','?'):19s}  {r.get('wall_s', float('nan')):7.0f}  {r.get('train', float('nan')):.4f}  "
              f"{lr:.3e}  {exp:.3e}  {r.get('best_val', float('nan')):.6f}{flag}")
    if walls:
        mean = sum(walls) / len(walls)
        start = dt.datetime.strptime(args.start, "%Y-%m-%dT%H:%M:%S")   # py3.6: no fromisoformat
        deadline = start + dt.timedelta(hours=args.walltime_h)
        fit = args.walltime_h * 3600 / mean
        done = max(rows) if rows else 0
        print(f"\nmean epoch wall = {mean:.0f} s = {mean/3600:.3f} h; {args.walltime_h:.0f} h fits {fit:.2f} epochs; "
              f"deadline {deadline.isoformat()}")
        for e in range(done + 1, args.epochs + 1):
            eta = start + dt.timedelta(seconds=mean * e)
            mark = "  (needs a RESUME)" if eta > deadline else ""
            if e % args.t0 == 0 or e == done + 1 or e == args.epochs or eta > deadline:
                print(f"  epoch {e:2d} ends ~{eta.isoformat(timespec='minutes')}{mark}")
    try:
        keys = {}
        with open(args.exp_dir + "/metrics/metrics.jsonl", errors="replace") as fh:
            for line in fh:
                try:
                    for k in json.loads(line):
                        keys[k] = keys.get(k, 0) + 1
                except Exception:
                    continue
        print(f"\nmetrics.jsonl: {len(keys)} distinct keys; val/mean/loss present: {'val/mean/loss' in keys}")
    except FileNotFoundError:
        print("\nmetrics.jsonl: not found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
