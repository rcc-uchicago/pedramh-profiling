#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
"""Turn one Makani multi-node run's stdout into one row of the scaling CSV.

Called by ``polaris_makani_multinode_scaling.pbs``.  It lives in its own file
rather than inline in the PBS heredoc so that it can be tested without an
allocation -- see ``test_parse_makani_scaling.py``, which is the only reason a
bug like "the CSV column was empty because the shell variable was never
exported" gets caught before 16 A100s are spent producing it.

It parses only lines that exist in the installed source, quoted here so a
future makani bump that renames one is traceable:

    makani/utils/training/deterministic_trainer.py:527
        "Average step time after step {N}: {X} ms"
    makani/utils/training/deterministic_trainer.py:529
        "Average effective io rate after step {N}: {Y} GB/s"
    makani/utils/training/deterministic_trainer.py:441
        "Total training time is {Z} sec"
    sfno_training/train_plasim.py
        "Communicators wireup time: {W}s"

Two of the checks matter more than the numbers:

* ``transport`` -- which network NCCL actually selected.  A silent fallback off
  the aws-ofi-nccl plugin looks exactly like "Slingshot is slow", and a scaling
  table that cannot name its transport is not evidence about an interconnect.
* ``world_sizes_seen`` -- guards the failure this whole launcher path is most
  exposed to.  If the rank shim does not apply, physicsnemo's DistributedManager
  warns "Assuming this is a single process job" and you get N independent
  world_size=1 trainers whose step time is a perfectly plausible number and
  means nothing about scaling.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

# The plugin that binds Slingshot's cxi provider on Polaris. Named here so the
# guard's remedy line and the launcher default cannot drift apart.
OFI_PLUGIN_CXI = "/soft/libraries/aws-ofi-nccl/v1.6.0-libfabric-1.22.0/lib"

# Column order is a cross-run contract (CLAUDE.md #10): a renamed or reordered
# column silently invalidates every prior comparison in the file.
FIELDS = [
    "jobid",
    "nodes",
    "ranks",
    "local_batch",
    "global_batch",
    "h_par",
    "w_par",
    "data",
    "rep",
    "gpu_order",
    # An nsys row is TRUNCATED BY DESIGN: makani's CUDAProfiler calls
    # sys.exit(0) at capture_range_stop (exit_on_stop defaults True), so the run
    # ends mid-epoch, total_train_s is empty, and the step average covers fewer
    # steps -- under profiler overhead besides. Its own column so an nsys row can
    # never be quietly averaged in with a clean one.
    "nsys",
    "steps",
    "step_ms",
    "samples_s_rank",
    "samples_s_total",
    "io_gbs",
    "wireup_s",
    "total_train_s",
    "transport",
    # Added 2026-09-17. This changes the header, so the writer will refuse an
    # existing CSV -- deliberately: every row written before this column existed
    # may have been a tcp row, and mixing those with cxi rows in one file is the
    # error this column was added to prevent.
    "provider",
    "world_sizes_seen",
    "torch",
    "env_source",
    "omp_threads",
    "log",
]


def _last_float(text: str, pattern: str) -> float | str:
    """Last match of a single-group float pattern, or "" if it never appeared.

    Last, not first: makani prints the running average every
    print_timings_frequency steps, so the final print is the one over the most
    steps and the least warmup.
    """
    found = re.findall(pattern, text)
    return float(found[-1]) if found else ""


def parse_log(text: str) -> dict:
    """Extract the measured quantities from a run's stdout."""
    # To end-of-line, not \S+: the name NCCL prints for the plugin we care about
    # is "AWS Libfabric" -- two words. A \S+ capture silently records it as
    # "AWS", which reads as a different transport than the one that ran.
    nets = sorted({m.strip() for m in re.findall(r"Using network (.+)$", text, re.M)})
    # The PROVIDER, not the plugin. "Using network AWS Libfabric" above names the
    # dlopen'd plugin and is byte-identical whether aws-ofi-nccl bound Slingshot's
    # cxi provider or silently fell back to tcp -- so `transport` alone could never
    # fail, and for three weeks it did not (7566145 ran on tcp; prereg #2 scored a
    # HIT against a check with no failure mode -- CLAUDE.md #10).
    # Both spellings matter: v1.6.0 prints "Selected Provider is cxi (found 2 nics)",
    # v1.21.1 prints "Selected provider is tcp, fabric is ...". Matching one spelling
    # records UNKNOWN on the other, which is the same blindness in a new place.
    provs = sorted({m.lower() for m in re.findall(r"Selected [Pp]rovider is (\w+)", text)})
    # `\s*` on BOTH sides of the separator. train_plasim's "DDP launch summary"
    # column-pads its keys -- "world_size                = 16" -- so a pattern
    # anchored as `world_size[=:]` matches nothing and this guard, which exists
    # to catch N independent world_size=1 trainers, would silently never fire.
    # The looser form also still catches physicsnemo-style "world_size=16".
    world_sizes = sorted(
        {int(x) for x in re.findall(r"\bworld_size\s*[=:]\s*(\d+)", text)}
    )
    return {
        "step_ms": _last_float(text, r"Average step time after step \d+:\s*([0-9.]+)\s*ms"),
        "io_gbs": _last_float(
            text, r"Average effective io rate after step \d+:\s*([0-9.]+)\s*GB/s"
        ),
        "total_train_s": _last_float(text, r"Total training time is\s*([0-9.]+)\s*sec"),
        "wireup_s": _last_float(text, r"Communicators wireup time:\s*([0-9.]+)\s*s"),
        "transport": "|".join(nets) if nets else "UNKNOWN",
        "provider": "|".join(provs) if provs else "UNKNOWN",
        "world_sizes_seen": "|".join(map(str, world_sizes)),
        "_world_sizes": world_sizes,
    }


def build_row(args, parsed: dict, torch_version: str = "") -> dict:
    step_ms = parsed["step_ms"]
    # Weak scaling holds the per-GPU work fixed, so per-rank throughput is the
    # quantity that should stay flat as nodes grow; the total is what a user
    # feels. Report both -- quoting only the total hides a per-rank regression
    # behind the added hardware.
    #
    # DERIVE BOTH FROM global_batch, NOT from local_batch x ranks.
    # makani's --batch_size is split across the DATA group, not the world
    # (train_plasim._resolve_batch_sizes), so under model parallelism
    # `local_batch` is per DATA GROUP and there are only ranks/(h_par*w_par)
    # of them. The old `local_batch * ranks` therefore overstated throughput by
    # exactly h_par*w_par on every spatial row -- h2w2 at global batch 32 read
    # 222 samples/s where the truth is 55.5 (found 2026-09-01, jobs 7580122 /
    # 7580162). It is an identity on pure-DDP rows (h_par=w_par=1 =>
    # local_batch*ranks == global_batch), so every existing h1w1 row is
    # unchanged and stays comparable.
    sps_total = (args.global_batch / (step_ms / 1000.0)) if step_ms else ""
    sps_rank = (sps_total / args.ranks) if sps_total != "" else ""
    row = {
        "jobid": args.jobid,
        "nodes": args.nodes,
        "ranks": args.ranks,
        "local_batch": args.local_batch,
        "global_batch": args.global_batch,
        "h_par": args.h_par,
        "w_par": args.w_par,
        "data": args.data,
        "rep": args.rep,
        "gpu_order": args.gpu_order,
        "nsys": args.nsys,
        "steps": args.steps,
        "step_ms": step_ms,
        "samples_s_rank": round(sps_rank, 4) if sps_rank != "" else "",
        "samples_s_total": round(sps_total, 4) if sps_total != "" else "",
        "io_gbs": parsed["io_gbs"],
        "wireup_s": parsed["wireup_s"],
        "total_train_s": parsed["total_train_s"],
        "transport": parsed["transport"],
        "provider": parsed["provider"],
        "world_sizes_seen": parsed["world_sizes_seen"],
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


def check(row: dict, parsed: dict, ranks: int) -> int:
    """Report on the row's trustworthiness. Returns a process exit code."""
    rc = 0
    if row["step_ms"] == "":
        print("ERROR NO_STEP_TIMING: the log has no 'Average step time' line.")
        print("  Either print_timings_frequency did not fire, or training never")
        print("  reached the first multiple of it. The row is not a measurement.")
        rc = 4
    if row["transport"] == "UNKNOWN":
        print("WARN NO_TRANSPORT_LINE: no 'Using network' line in the log.")
        print("  NCCL_DEBUG is probably below INFO. The timing is recorded but the")
        print("  transport that produced it is not -- say so if you table this row.")
    # Which fabric actually carried the traffic. Gated on NODES, not ranks: a
    # 1-node run never initialises the net plugin, so there is no provider line
    # to find and demanding one would fail every single-node arm.
    nodes = int(row["nodes"]) if str(row["nodes"]).isdigit() else 0
    if nodes > 1:
        prov = row["provider"]
        if prov == "UNKNOWN":
            print("ERROR FABRIC_UNVERIFIED: no 'Selected Provider' line in the log.")
            print("  Set NCCL_DEBUG=INFO. Without it this row cannot say whether it")
            print("  crossed Slingshot or TCP, and those differ by 5.2x (7629082).")
            rc = 4
        elif prov != "cxi":
            # ALLOW_TCP=1 is an acknowledgement, not a bypass: the row is still
            # labelled tcp and still must not be tabled against a cxi row. It
            # exists because v1.6.0 (the only plugin that binds cxi here) wedges
            # by message size -- 7565896 on a 41,088-element broadcast, 7629096
            # on a 512 KB all_gather -- so "use the fabric" is sometimes not
            # available, and pretending otherwise would just move the lie.
            if os.environ.get("ALLOW_TCP") == "1":
                print("WARN FABRIC_TCP_ACKNOWLEDGED: provider=%s, ALLOW_TCP=1." % prov)
                print("  Recorded as a tcp row. It is ~5x slower than cxi (7629082)")
                print("  and is NOT comparable with any cxi row in this file.")
            else:
                print("ERROR FABRIC_NOT_SLINGSHOT: provider=%s, expected cxi." % prov)
                print("  aws-ofi-nccl fell back off the fabric. This is a TCP row, not")
                print("  a Slingshot row; the 128-node production run 7566145 was one")
                print("  and nobody noticed for three weeks.")
                print("  Fix:      -v OFI_PLUGIN=%s" % OFI_PLUGIN_CXI)
                print("  Or admit: -v ALLOW_TCP=1  (if the cxi plugin wedges your shape)")
                rc = 4
    ws = parsed["_world_sizes"]
    if ranks > 1 and ws and max(ws) != ranks:
        print(
            "ERROR WORLD_SIZE_MISMATCH: log reports world_size %s for a %d-rank launch."
            % (ws, ranks)
        )
        print("  The classic cause is the rank shim not being applied, giving N")
        print("  independent world_size=1 trainers. This row is NOT a scaling point.")
        rc = 4
    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--jobid", default="")
    p.add_argument("--nodes", type=int, required=True)
    p.add_argument("--ranks", type=int, required=True)
    p.add_argument("--local-batch", type=int, required=True)
    p.add_argument("--global-batch", type=int, required=True)
    p.add_argument("--h-par", type=int, default=1)
    p.add_argument("--w-par", type=int, default=1)
    p.add_argument("--data", default="real")
    p.add_argument("--rep", default="1")
    p.add_argument("--gpu-order", default="default")
    p.add_argument("--nsys", default="0", help="1 if this run was an nsys capture (truncated)")
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--env-source", default="")
    p.add_argument("--omp-threads", default="")
    args = p.parse_args(argv)

    with open(args.log, errors="replace") as fh:
        parsed = parse_log(fh.read())

    torch_version = ""
    try:  # informational only: never fail a parse because torch is absent
        import torch

        torch_version = torch.__version__
    except Exception:  # noqa: BLE001
        pass

    row = build_row(args, parsed, torch_version)
    append_row(args.csv, row)

    print("--- scaling row ---")
    for key in (
        "nodes",
        "ranks",
        "global_batch",
        "step_ms",
        "samples_s_rank",
        "samples_s_total",
        "wireup_s",
        "io_gbs",
        "transport",
        "provider",
        "world_sizes_seen",
    ):
        print("  %-17s %s" % (key, row[key]))
    print("csv -> %s" % args.csv)

    return check(row, parsed, args.ranks)


if __name__ == "__main__":
    sys.exit(main())
