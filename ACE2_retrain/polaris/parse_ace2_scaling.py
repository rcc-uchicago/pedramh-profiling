#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
"""Turn one ACE2 Polaris run into one row of the scaling CSV.

Called by ``polaris_ace2_1node.pbs`` and ``polaris_ace2_multinode.pbs``.  It
lives in its own file rather than inline in a PBS heredoc so it can be tested
without an allocation -- see ``test_parse_ace2_scaling.py``.  That is not
tidiness: the guards below are the difference between a scaling number and a
plausible-looking number.

Adapted from ``physicsnemo_ai_rossby/polaris/parse_ai_rossby_scaling.py``.

**FIELDS is byte-identical to that file's**, deliberately: the two tables must
concatenate (CLAUDE.md #10).  Two columns carry a harness-specific *value* while
keeping their contracted *meaning*:

* ``data`` -- the store this arm read.  ``nc`` for the single 2.4 TB NetCDF,
  ``zarr`` after a conversion.  Same role as ai-rossby's ``real``/``synthetic``:
  it names what was on the other end of the loader, which for ACE2 is the axis
  most likely to decide the result (§2a of the multi-node handoff).
* ``local_batch`` -- samples per GPU.  ⚠ fme's ``batch_size`` is **GLOBAL**
  (``fme/core/distributed/torch_distributed.py:110`` divides by ``total_ranks``),
  so the launcher computes ``global = local * ranks`` and passes both.  Getting
  this backwards turns a weak-scaling ladder into a strong-scaling one while the
  column names stay put.

The LOG GRAMMAR is ACE2's.  fme prints no world-size line anywhere -- ``grep``
over ``ace_exp/fme`` finds only ``"DONE ---- rank N"`` -- so ``ace2_telemetry.py``
prints the banner both world-size guards read::

    ACE2_BANNER steps_per_epoch=240 world_size=4 rank=0 local_batch=1 ...

It is printed by the TRAINER PROCESS after ``init_process_group``, from a
``print`` rather than ``logging`` (fme routes logging to rank 0 only, which would
make ``ranks_reporting`` read 1 on every arm).

Five guards, each a real failure mode observed on this cluster:

* ``transport`` -- which network NCCL actually selected.  A silent fallback off
  the aws-ofi-nccl plugin looks exactly like "Slingshot is slow".
* ``world_sizes_seen`` -- read ONLY off the trainer's own banner, never off
  anything the launcher echoed.  If the PALS rank shim does not apply, fme's
  ``TorchDistributed.is_available()`` returns False (no ``RANK`` in env), it
  silently constructs ``NonDistributed``, and you get N independent
  ``world_size=1`` trainers whose step time is perfectly plausible and means
  nothing about scaling.
* ``ranks_reporting`` -- the same question asked a second, independent way: how
  many distinct PALS rank labels emitted that banner.  A rank that died before
  the banner leaves ``world_sizes_seen`` correct and this wrong.
* ``n_steps`` vs the requested ``--steps`` -- arms of a ladder must time the same
  number of steps or their averages cover different amounts of warmup.  A short
  arm is what a walltime kill produces.
* telemetry ``n_gpus`` -- the trainer's own idea of the world size, written by a
  different code path than the banner.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

# Column order is a cross-run AND cross-project contract (CLAUDE.md #10): a
# renamed or reordered column silently invalidates every prior comparison, and
# this list is shared verbatim with parse_ai_rossby_scaling.py so the two tables
# concatenate. Config -- plugin, progress model, NCCL_ALGO, store -- deliberately
# does NOT get columns: it lives in the log header, and rows from different
# configs go in different FILES rather than different columns.
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
    # makani and ai-rossby scaling CSVs use, so the per-rank numbers mean the
    # same thing. samples_s_wall is the telemetry's own wall-clock figure and
    # includes loader idle; gpu_busy_frac is the ratio between the two views.
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

# The trainer's per-rank startup banner (ace2_telemetry.py::_banner). Anchored on
# the ACE2_BANNER token AND on steps_per_epoch= so that nothing the LAUNCHER
# prints can be mistaken for it.
BANNER_RE = re.compile(
    r"ACE2_BANNER\s+steps_per_epoch\s*=\s*-?\d+.*?\bworld_size\s*=\s*(\d+)"
)
# PALS `--label` prefix.
# ⚠ MEASURED on job 7586496, and it is NOT what the ai-rossby parser this was
# adapted from assumes. PALS emits
#     x3005c0s13b0n0.hsn.cm.polaris.alcf.anl.gov 3: ACE2_BANNER ...
# i.e. HOSTNAME then rank, not a bare "3: ". Against `^\s*(\d+):` no line matches,
# `labels` stays empty, and ranks_reporting silently falls back to counting
# banner lines -- so the two world-size guards, which exist to be INDEPENDENT of
# each other, both end up reading the same quantity. It still catches a dead rank
# (31 banners for 32 ranks), so nothing looked broken; it just quietly stopped
# being a second opinion.
# The optional host token is non-capturing; `\d{1,5}:` followed by whitespace
# keeps a bare "20:33:12" timestamp from reading as rank 20.
LABEL_RE = re.compile(r"^\s*(?:\S+\s+)?(\d{1,5}):\s")
# Same banner, for the wrap guard.
SPE_RE = re.compile(r"ACE2_BANNER\s+steps_per_epoch\s*=\s*(-?\d+)")


def parse_log(text: str) -> dict:
    """Extract transport and the two independent world-size views."""
    # To end-of-line, not \S+: the name NCCL prints for the plugin we care about
    # is "AWS Libfabric" -- two words. A \S+ capture silently records it as
    # "AWS", which reads as a different transport than the one that ran.
    nets = sorted({m.strip() for m in re.findall(r"Using network (.+)$", text, re.M)})

    world_sizes: set[int] = set()
    labels: set[str] = set()
    banner_lines = 0
    steps_per_epoch: set[int] = set()
    for line in text.splitlines():
        m = BANNER_RE.search(line)
        if not m:
            continue
        banner_lines += 1
        world_sizes.add(int(m.group(1)))
        spe = SPE_RE.search(line)
        if spe:
            steps_per_epoch.add(int(spe.group(1)))
        lab = LABEL_RE.match(line)
        if lab:
            labels.add(lab.group(1))
    # Labels when PALS --label is on (both launchers pass it); otherwise fall
    # back to counting banners, which is right as long as every rank logs one --
    # it does, the banner is not rank-0-gated.
    ranks_reporting = len(labels) if labels else banner_lines

    return {
        "transport": "|".join(nets) if nets else "UNKNOWN",
        "world_sizes_seen": "|".join(map(str, sorted(world_sizes))),
        "ranks_reporting": ranks_reporting,
        "_world_sizes": sorted(world_sizes),
        "_banner_lines": banner_lines,
        "_steps_per_epoch": sorted(steps_per_epoch),
    }


def read_telemetry(csv_path: str, run_name: str) -> dict:
    """Last epoch-telemetry row for ``run_name``, or {} if there is none.

    Last, not first: a multi-epoch functional check leaves several, and the later
    ones are past dataloader warmup. Filtering by run_name rather than taking the
    file's final row matters because the CSV is append-only and a previous arm's
    row would otherwise be reported as this arm's result.
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
        print("  Either ACE2_EPOCH_TELEMETRY was not 1, the launcher did not go")
        print("  through ace2_telemetry.py, or the epoch never ended (a walltime")
        print("  kill mid-epoch writes nothing). The row is not a measurement.")
        return 4

    if row["step_med_ms"] == "":
        print("ERROR NO_STEP_TIMING: the telemetry row carries no step_med_ms.")
        rc = 4

    if row["transport"] == "UNKNOWN":
        print("WARN NO_TRANSPORT_LINE: no 'Using network' line in the log.")
        print("  NCCL_DEBUG is probably below INFO. The timing is recorded but the")
        print("  transport that produced it is not -- say so if you table this row.")

    if parsed["_banner_lines"] == 0:
        print("ERROR NO_TRAINER_BANNER: no 'ACE2_BANNER ... world_size=...' line.")
        print("  That line is where BOTH world-size guards read from, so its absence")
        print("  means they did not run -- not that they passed. Most likely the")
        print("  launcher invoked `python -m fme.ace.train` directly instead of")
        print("  ace2_telemetry.py, which is what installs the banner.")
        rc = 4

    ws = parsed["_world_sizes"]
    if ws and max(ws) != ranks:
        print("ERROR WORLD_SIZE_MISMATCH: log reports world_size %s for a %d-rank launch."
              % (ws, ranks))
        print("  The classic cause is the rank shim not being applied: with no RANK in")
        print("  the environment fme falls back to NonDistributed, giving N independent")
        print("  world_size=1 trainers. This row is NOT a scaling point.")
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

    # The loader's OWN epoch length, from the banner, versus what was asked for.
    # This is the wrap guard in its exact form: the launcher sizes the epoch with
    # `sample_with_replacement = STEPS * global_batch` so that every rank serves
    # exactly `STEPS` steps at every node count. If that override is ever dropped
    # or mis-derived the loader serves the whole dataset instead, the arm times a
    # different experiment, and -- because the epoch length then moves with the
    # global batch -- the bias sits along the very axis being swept.
    spe = parsed["_steps_per_epoch"]
    if spe and args.steps and set(spe) != {args.steps}:
        print("ERROR EPOCH_LENGTH_MISMATCH: loader serves %s steps/rank/epoch, "
              "this arm asked for %d." % (spe, args.steps))
        print("  Check the train_loader.sample_with_replacement override.")
        rc = 4

    n_steps = row["n_steps"]
    if n_steps not in ("", None) and args.steps and int(n_steps) != args.steps:
        print("ERROR STEP_COUNT_MISMATCH: timed %s steps, requested %d."
              % (n_steps, args.steps))
        print("  Arms of a ladder must run the same number of steps or their step")
        print("  averages cover different amounts of warmup. Do not table this row.")
        rc = 4

    # ACE2-specific, and the one this harness exists to surface first. ai-rossby
    # held >=0.976 on every arm, which is what let its penalty be attributed to
    # the collective. ACE2 reads one 2.4 TB NetCDF on ONE Lustre OST from every
    # rank, so a low value here means the loader, and the whole ring/tree/comms
    # analysis is then the wrong lens. WARN, not ERROR: it is a valid measurement
    # of a different thing.
    busy = row["gpu_busy_frac"]
    if busy != "" and busy < 0.85:
        print("WARN LOW_GPU_BUSY_FRAC: %.3f < 0.85." % busy)
        print("  %.1f%% of the epoch is loader/other idle, not compute and not NCCL."
              % (100 * (1 - busy)))
        print("  Check `lfs getstripe` on the .nc before attributing anything to the")
        print("  fabric -- see the multi-node handoff §2/§2a.")

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
    p.add_argument("--data", default="nc")
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
        "local_batch",
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
