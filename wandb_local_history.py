#!/usr/bin/env python
"""Dump per-step wandb history straight from a run's local datastore file --
no wandb.ai login, no network, works entirely offline on a compute/login node.

Why this exists: both PanguWeather and ai-rossby log ~101 per-variable/
per-level metrics (`train_{var}_lwrmse`, `train_{var}_level{level:.4f}_lwrmse`,
see `ai_rossby_finegrained_wandb_handoff.md`) EVERY iteration, which is more
detail than the stdout/.err logs ever print. That detail otherwise only lives
in the wandb backend -- but wandb's local ".wandb" datastore file (leveldb-log
format under `<run dir>/run-<id>.wandb`) is the source of truth it syncs FROM,
and it is readable directly with the `wandb` package's own `DataStore` reader.
Every run directory under this project's `$MEMBER_ROOT/wandb/` (Pangu) or
`$MEMBER_ROOT/runs/*/wandb/` (ai-rossby) is group-readable by `lighthouse-uchicago`
(confirmed via `namei -l`), so anyone in that group can run this without any
wandb credentials at all.

Usage
-----
Find the run directory (one of):
    ls /eagle/projects/lighthouse-uchicago/members/mehta5/wandb/wandb/           # Pangu runs
    ls /eagle/projects/lighthouse-uchicago/members/mehta5/runs/*/wandb/wandb/    # ai-rossby runs
(each is named `run-<timestamp>-<8charid>`; the newest one is the live/latest run)

Then, from a compute node (or the login node -- this script never imports
torch/physicsnemo, only `wandb`'s local reader, so it's safe there too):

    module use /soft/modulefiles && module load conda && conda activate base
    python3 wandb_local_history.py <run_dir_or_.wandb_file> --summary
    python3 wandb_local_history.py <run_dir_or_.wandb_file> --keys-like '_lwrmse$' --csv out.csv

`--summary` prints, per matched key, the mean over the first and last
`--window` fraction of logged steps and the percent change between them --
the same "is this variable still improving or has it plateaued" check used
throughout this project's convergence analysis. `--csv` instead writes every
matched key's full per-step series (one row per logged step) for loading into
pandas/Excel/whatever.
"""

from __future__ import annotations

import argparse
import csv as csv_module
import json
import re
import sys
from pathlib import Path


def _find_wandb_file(path: Path) -> Path:
    """Accept either a run directory or the .wandb file itself."""
    if path.is_file():
        return path
    matches = sorted(path.glob("run-*.wandb")) + sorted(path.glob("*.wandb"))
    if not matches:
        raise FileNotFoundError(f"No .wandb file found under {path}")
    return matches[0]


def _scan_history(wandb_file: Path, key_filter):
    """Yield (step, {key: float_value}) for every history record matching
    `key_filter`, in file order. Uses wandb's own leveldb-log reader --
    `scan_data()` (not `scan_record()`) because it handles the 32 KB block
    padding/continuation records; calling `scan_record()` directly desyncs
    across block boundaries and raises spurious checksum errors."""
    import wandb  # noqa: F401  (import needed before DataStore per its own assert)
    from wandb.sdk.internal import datastore
    from wandb.proto import wandb_internal_pb2 as pb

    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_file))
    while True:
        data = ds.scan_data()
        if data is None:
            return
        rec = pb.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        if not rec.HasField("history"):
            continue
        step = None
        row = {}
        for item in rec.history.item:
            k = "/".join(item.nested_key) if item.nested_key else item.key
            if k == "_step":
                try:
                    step = json.loads(item.value_json)
                except Exception:
                    pass
                continue
            if key_filter(k):
                try:
                    row[k] = float(json.loads(item.value_json))
                except (ValueError, TypeError):
                    pass
        if row:
            yield step, row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_path", help="Run directory or a .wandb file path")
    ap.add_argument(
        "--keys-like", default=r"_lwrmse$",
        help="Regex a metric key must match to be included (default: %(default)r)",
    )
    ap.add_argument("--csv", default=None, help="Write full per-step series to this CSV")
    ap.add_argument("--summary", action="store_true", help="Print first-vs-last-window trend per key")
    ap.add_argument(
        "--window", type=float, default=0.2,
        help="Fraction of steps to average at each end for --summary (default: 0.2)",
    )
    args = ap.parse_args()

    wandb_file = _find_wandb_file(Path(args.run_path))
    key_re = re.compile(args.keys_like)
    print(f"Reading {wandb_file} (keys matching /{args.keys_like}/) ...", file=sys.stderr)

    steps = []
    series: dict[str, list] = {}
    for step, row in _scan_history(wandb_file, key_re.search):
        steps.append(step)
        for k, v in row.items():
            series.setdefault(k, [None] * (len(steps) - 1)).append(v)
        for k in series:
            if len(series[k]) < len(steps):
                series[k].append(None)

    if not steps:
        print("No matching history records found.", file=sys.stderr)
        return 1
    print(f"{len(steps)} logged steps, {len(series)} matching keys, "
          f"step range [{steps[0]}, {steps[-1]}]", file=sys.stderr)

    if args.csv:
        keys = sorted(series)
        with open(args.csv, "w", newline="") as fh:
            w = csv_module.writer(fh)
            w.writerow(["step"] + keys)
            for i, step in enumerate(steps):
                w.writerow([step] + [series[k][i] for k in keys])
        print(f"Wrote {args.csv}", file=sys.stderr)

    if args.summary or not args.csv:
        n = len(steps)
        w = max(1, int(n * args.window))
        rows = []
        for k in sorted(series):
            first = [x for x in series[k][:w] if x is not None]
            last = [x for x in series[k][-w:] if x is not None]
            if not first or not last:
                continue
            m1, m2 = sum(first) / len(first), sum(last) / len(last)
            pct = 100 * (m2 - m1) / m1 if m1 else float("nan")
            rows.append((k, m1, m2, pct))
        rows.sort(key=lambda r: r[3])
        print(f"\n{'key':45s} {'first_' + str(args.window):>14s} {'last_' + str(args.window):>14s} {'pct_change':>10s}")
        for k, m1, m2, pct in rows:
            print(f"{k:45s} {m1:14.5g} {m2:14.5g} {pct:9.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
