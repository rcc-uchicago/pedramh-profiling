#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
"""Turn one ai-rossby multi-node run into one row of the scaling CSV.

Called by ``polaris_ai_rossby_multinode_scaling.pbs``.  It lives in its own file
rather than inline in the PBS heredoc so that it can be tested without an
allocation -- see ``test_parse_ai_rossby_scaling.py``.  That is not tidiness:
the guards below are the difference between a scaling number and a
plausible-looking number, and every one of them is cheap to get subtly wrong.

Adapted from ``makani_sfno/polaris/parse_makani_scaling.py``.  The GUARDS are
kept; the LOG GRAMMAR is this harness's, because the two trainers print nothing
in common:

    examples/weather/ai_rossby/train.py:887   (every rank)
        "steps_per_epoch={S}, stages={G}, total_epochs={E}, world_size={W},
         device=cuda:{L}"
    examples/weather/ai_rossby/epoch_telemetry.py:271   (rank 0)
        "EPOCH_TELEMETRY epoch={E} n={N} step_med={X}ms gpu_busy={Y}% ..."

The timings are read from the epoch-telemetry **CSV**, not from that printed
line: the print is rounded to 0.1 ms and drops p90/mean/std/wall, while the CSV
is an existing cross-project contract (``epoch_telemetry.py`` COLUMNS, asserted
against the PanguWeather copy by ``epoch_telemetry_test.py``).  Inventing a
second timing path when a contracted one exists is how two numbers that should
agree start disagreeing.

Four checks matter more than the numbers:

* ``transport`` -- which network NCCL actually selected.  A silent fallback off
  the aws-ofi-nccl plugin looks exactly like "Slingshot is slow", and a scaling
  table that cannot name its transport is not evidence about an interconnect.
* ``world_sizes_seen`` -- if the rank shim does not apply, physicsnemo's
  DistributedManager warns "Assuming this is a single process job" and you get N
  independent world_size=1 trainers whose step time is a perfectly plausible
  number and means nothing about scaling.  Read ONLY off the trainer's own
  banner line, never off anything the launcher echoed -- a launcher that printed
  "world_size=16" in its header would satisfy this guard by construction.
* ``ranks_reporting`` -- the same question asked a second, independent way:
  how many distinct PALS rank labels actually emitted that banner.  A rank that
  died before the banner leaves ``world_sizes_seen`` correct and this wrong.
* ``n_steps`` vs the requested ``--steps`` -- an arm that ran a different number
  of steps than the arm it is compared against is not comparable, and a short
  arm is exactly what a walltime kill or an early ``max_iterations`` break
  produces.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

# Column order is a cross-run contract (CLAUDE.md #10): a renamed or reordered
# column silently invalidates every prior comparison in the file.  Config --
# plugin, progress model, NCCL_PROTO, store -- deliberately does NOT get columns
# (handoff rule #4): it lives in the log header and the launcher pins, and rows
# from different configs go in different files rather than different columns.
FIELDS = [
    "jobid",
    "nodes",
    "ranks",
    "local_batch",
    "global_batch",
    "data",
    "rep",
    "gpu_order",
    "steps",            # requested
    "n_steps",          # actually timed (must match, see check())
    "step_med_ms",
    "step_p90_ms",
    "step_mean_ms",
    "step_std_ms",
    # Derived from step_med, i.e. from GPU step time -- the same definition the
    # makani scaling CSV uses, so the two tables' per-rank numbers mean the same
    # thing. samples_s_wall is the trainer's own wall-clock figure and includes
    # loader idle; gpu_busy_frac is the ratio between the two views.
    "samples_s_rank",
    "samples_s_total",
    "samples_s_wall",
    "gpu_busy_frac",
    "epoch_wall_s",
    "peak_mem_gb",
    "transport",
    "world_sizes_seen",
    "ranks_reporting",
    "torch",
    "env_source",
    "omp_threads",
    "log",
]

# The trainer's per-rank startup banner (train.py:887). Anchored on
# steps_per_epoch= so that nothing the LAUNCHER prints can be mistaken for it.
BANNER_RE = re.compile(
    r"steps_per_epoch\s*=\s*\d+.*?\bworld_size\s*=\s*(\d+)", re.S
)
# PALS `--label` prefixes every line with "<rank>: ".
LABEL_RE = re.compile(r"^\s*(\d+):")


def parse_log(text: str) -> dict:
    """Extract transport and the two independent world-size views."""
    # To end-of-line, not \S+: the name NCCL prints for the plugin we care about
    # is "AWS Libfabric" -- two words. A \S+ capture silently records it as
    # "AWS", which reads as a different transport than the one that ran.
    nets = sorted({m.strip() for m in re.findall(r"Using network (.+)$", text, re.M)})

    world_sizes: set[int] = set()
    labels: set[str] = set()
    banner_lines = 0
    for line in text.splitlines():
        m = BANNER_RE.search(line)
        if not m:
            continue
        banner_lines += 1
        world_sizes.add(int(m.group(1)))
        lab = LABEL_RE.match(line)
        if lab:
            labels.add(lab.group(1))
    # Labels when PALS --label is on (the launcher always passes it); otherwise
    # fall back to counting banners, which is right as long as every rank logs
    # one -- it does, the banner is not rank-0-gated.
    ranks_reporting = len(labels) if labels else banner_lines

    return {
        "transport": "|".join(nets) if nets else "UNKNOWN",
        "world_sizes_seen": "|".join(map(str, sorted(world_sizes))),
        "ranks_reporting": ranks_reporting,
        "_world_sizes": sorted(world_sizes),
        "_banner_lines": banner_lines,
    }


def read_telemetry(csv_path: str, run_name: str) -> dict:
    """Last epoch-telemetry row for ``run_name``, or {} if there is none.

    Last, not first: a multi-epoch functional check leaves several, and the
    later ones are past dataloader warmup.  Filtering by run_name rather than
    taking the file's final row matters because the CSV is append-only and a
    previous arm's row would otherwise be reported as this arm's result.
    """
    if not csv_path or not os.path.exists(csv_path):
        return {}
    hit: dict = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("run_name") == run_name:
                hit = row
    return hit


def _f(row: dict, key: str):
    """Float from a telemetry cell, or "" -- never a crash on a blank."""
    val = row.get(key, "")
    if val in ("", None):
        return ""
    try:
        return float(val)
    except (TypeError, ValueError):
        return ""


def build_row(args, parsed: dict, tel: dict, torch_version: str = "") -> dict:
    step_med = _f(tel, "step_med_ms")
    # Weak scaling holds local_batch fixed, so per-rank throughput is the
    # quantity that should stay flat as nodes grow; the total is what a user
    # feels. Report both -- quoting only the total hides a per-rank regression
    # behind the added hardware.
    sps_rank = (args.local_batch / (step_med / 1000.0)) if step_med else ""
    row = {
        "jobid": args.jobid,
        "nodes": args.nodes,
        "ranks": args.ranks,
        "local_batch": args.local_batch,
        "global_batch": args.global_batch,
        "data": args.data,
        "rep": args.rep,
        "gpu_order": args.gpu_order,
        "steps": args.steps,
        "n_steps": tel.get("n_steps", ""),
        "step_med_ms": step_med,
        "step_p90_ms": _f(tel, "step_p90_ms"),
        "step_mean_ms": _f(tel, "step_mean_ms"),
        "step_std_ms": _f(tel, "step_std_ms"),
        "samples_s_rank": round(sps_rank, 4) if sps_rank != "" else "",
        "samples_s_total": round(sps_rank * args.ranks, 4) if sps_rank != "" else "",
        "samples_s_wall": _f(tel, "samples_per_s"),
        "gpu_busy_frac": _f(tel, "gpu_busy_frac"),
        "epoch_wall_s": _f(tel, "epoch_wall_s"),
        "peak_mem_gb": _f(tel, "peak_mem_gb"),
        "transport": parsed["transport"],
        "world_sizes_seen": parsed["world_sizes_seen"],
        "ranks_reporting": parsed["ranks_reporting"],
        "torch": torch_version,
        "env_source": args.env_source,
        "omp_threads": args.omp_threads,
        "log": os.path.basename(args.log),
    }
    assert list(row) == FIELDS, "row keys drifted from FIELDS"
    return row


def append_row(csv_path: str, row: dict) -> None:
    """Append, refusing to write into a file whose header has drifted."""
    newfile = not os.path.exists(csv_path)
    if not newfile:
        with open(csv_path, newline="") as fh:
            existing = next(csv.reader(fh), [])
        if existing and existing != FIELDS:
            raise SystemExit(
                "ERROR SCALING_CSV_SCHEMA_DRIFT\n"
                "  on disk : %s\n"
                "  this run: %s\n"
                "  Appending would corrupt every prior comparison. Rename the old file."
                % (",".join(existing), ",".join(FIELDS))
            )
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if newfile:
            writer.writeheader()
        writer.writerow(row)


def check(row: dict, parsed: dict, tel: dict, args) -> int:
    """Report on the row's trustworthiness. Returns a process exit code."""
    rc = 0
    ranks = args.ranks

    if not tel:
        print("ERROR NO_TELEMETRY_ROW: no epoch-telemetry row for run_name=%r."
              % args.run_name)
        print("  Either AI_ROSSBY_EPOCH_TELEMETRY was not 1, or the epoch never")
        print("  ended (a walltime kill mid-epoch writes nothing). The row is not")
        print("  a measurement.")
        return 4

    if row["step_med_ms"] == "":
        print("ERROR NO_STEP_TIMING: the telemetry row carries no step_med_ms.")
        rc = 4

    if row["transport"] == "UNKNOWN":
        print("WARN NO_TRANSPORT_LINE: no 'Using network' line in the log.")
        print("  NCCL_DEBUG is probably below INFO. The timing is recorded but the")
        print("  transport that produced it is not -- say so if you table this row.")

    if parsed["_banner_lines"] == 0:
        print("ERROR NO_TRAINER_BANNER: no 'steps_per_epoch=... world_size=...' line.")
        print("  That line is where BOTH world-size guards read from, so its absence")
        print("  means they did not run -- not that they passed. If train.py's banner")
        print("  was reworded, update BANNER_RE and this file's docstring together.")
        rc = 4

    ws = parsed["_world_sizes"]
    if ranks > 1 and ws and max(ws) != ranks:
        print("ERROR WORLD_SIZE_MISMATCH: log reports world_size %s for a %d-rank launch."
              % (ws, ranks))
        print("  The classic cause is the rank shim not being applied, giving N")
        print("  independent world_size=1 trainers. This row is NOT a scaling point.")
        rc = 4

    reporting = parsed["ranks_reporting"]
    if reporting and reporting != ranks:
        print("ERROR RANKS_REPORTING_MISMATCH: %d ranks logged the startup banner, "
              "%d were launched." % (reporting, ranks))
        print("  world_size can be right while ranks are missing -- a rank that died")
        print("  before the banner leaves the others training a short global batch.")
        rc = 4

    # Independent of the log: the trainer wrote its own idea of the world size.
    tel_gpus = tel.get("n_gpus", "")
    if tel_gpus not in ("", None) and int(tel_gpus) != ranks:
        print("ERROR TELEMETRY_NGPUS_MISMATCH: telemetry says n_gpus=%s, launched %d."
              % (tel_gpus, ranks))
        rc = 4

    n_steps = row["n_steps"]
    if n_steps not in ("", None) and args.steps and int(n_steps) != args.steps:
        print("ERROR STEP_COUNT_MISMATCH: timed %s steps, requested %d."
              % (n_steps, args.steps))
        print("  Arms of a ladder must run the same number of steps or their step")
        print("  averages cover different amounts of warmup. Do not table this row.")
        rc = 4

    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--telemetry-csv", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--jobid", default="")
    p.add_argument("--nodes", type=int, required=True)
    p.add_argument("--ranks", type=int, required=True)
    p.add_argument("--local-batch", type=int, required=True)
    p.add_argument("--global-batch", type=int, required=True)
    p.add_argument("--data", default="real")
    p.add_argument("--rep", default="1")
    p.add_argument("--gpu-order", default="forward")
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--env-source", default="")
    p.add_argument("--omp-threads", default="")
    args = p.parse_args(argv)

    with open(args.log, errors="replace") as fh:
        parsed = parse_log(fh.read())
    tel = read_telemetry(args.telemetry_csv, args.run_name)

    torch_version = ""
    try:  # informational only: never fail a parse because torch is absent
        import torch

        torch_version = torch.__version__
    except Exception:  # noqa: BLE001
        pass

    row = build_row(args, parsed, tel, torch_version)
    append_row(args.csv, row)

    print("--- scaling row ---")
    for key in (
        "nodes",
        "ranks",
        "global_batch",
        "n_steps",
        "step_med_ms",
        "samples_s_rank",
        "samples_s_total",
        "gpu_busy_frac",
        "peak_mem_gb",
        "transport",
        "world_sizes_seen",
        "ranks_reporting",
    ):
        print("  %-17s %s" % (key, row[key]))
    print("csv -> %s" % args.csv)

    return check(row, parsed, tel, args)


if __name__ == "__main__":
    sys.exit(main())
