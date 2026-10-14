#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
"""Report per-collective SIZES from NCCL flight-recorder dumps.

    python3 ACE2_retrain/polaris/read_nccl_trace.py <run_dir>/nccl_trace/fr_*.pickle

This answers one question, and it is the question the whole Polaris multi-node
handoff turns on:

    **How big is ACE2's LARGEST SINGLE all-reduce?**

Not its total gradient volume -- that is the reasoning error the handoff keeps as
a worked example. The Polaris fabric defect triggers on the size of an
*individual collective*: a TREE all-reduce somewhere between 25 MiB and 1000 MiB
silently returns partially reduced data (head of the buffer reduced, tail
untouched, ranks disagreeing) and then hangs. DDP buckets gradients into many
all-reduces per step, so total volume says nothing about exposure.

What is already established, and what this script decides:

* fme wraps with stock DDP and never sets ``bucket_cap_mb``
  (``fme/core/distributed/torch_distributed.py:182-193``); ``bucket_cap_mb``
  appears zero times in all of ``ACE2_retrain/``.
* ``PROFILING_PLAN.md:171`` measured **11.4 buckets/step, ~165 MB each** on
  Midway, and DDP never splits one parameter across buckets, so the 212.34 MB
  dhconv weight should get its own bucket ⇒ **~212 MB expected maximum**.
* ⚠ But the obvious counter-argument -- "bucketing keeps every message small" --
  is NOT established either. ai-rossby's flight recorder showed a single
  all_reduce of its **entire 1.18 B-parameter model** (``numel=1182108160``) in
  BOTH the default-25 MB-bucket run and the forced-one-bucket run: byte-identical
  stuck collectives under a 200x difference in ``bucket_cap_mb``. Why DDP
  coalesced there is still open in the CHANGELOG. So ACE2's exposure is
  **UNKNOWN**, and this dump is what settles it.

⚠ THE RUN MUST HAVE ASKED FOR THE DUMP. ``TORCH_NCCL_DUMP_ON_TIMEOUT=1`` writes
the buffer only when the watchdog fires; a SUCCESSFUL run leaves nothing behind.
``-v FR_DUMP=1`` on ``polaris_ace2_train.pbs`` makes every rank dump explicitly
at epoch end.

Reads pickle, and falls back to JSON, because torch has moved between
``_dump_nccl_trace`` (pickled bytes) and ``_dump_nccl_trace_json`` across the
versions this project runs.

PASS = ``NCCL_TRACE_READ_OK``.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import sys

# Bytes per element. Only what the flight recorder actually reports for this
# workload; anything unlisted is counted as elements and SAID SO, never silently
# assumed to be 4 bytes -- that assumption is what turned ACE2's 2.67 GB of
# gradients into a "1.82 GB" figure in an earlier draft of the handoff, because
# the dhconv weights are complex64 at 8 B/element.
ITEMSIZE = {
    "torch.float32": 4, "float32": 4, "torch.float": 4, "Float": 4,
    "torch.float64": 8, "float64": 8, "Double": 8,
    "torch.float16": 2, "float16": 2, "Half": 2,
    "torch.bfloat16": 2, "bfloat16": 2, "BFloat16": 2,
    "torch.complex64": 8, "complex64": 8, "ComplexFloat": 8,
    "torch.complex128": 16, "complex128": 16, "ComplexDouble": 16,
    "torch.int64": 8, "int64": 8, "Long": 8,
    "torch.int32": 4, "int32": 4, "Int": 4,
    "torch.uint8": 1, "uint8": 1, "Byte": 1,
}


def load(path: str) -> dict:
    with open(path, "rb") as fh:
        blob = fh.read()
    try:
        return pickle.loads(blob)
    except Exception:  # noqa: BLE001 — the JSON spelling of the same dump
        return json.loads(blob.decode())


def entries(dump) -> list:
    """The per-collective records, whichever shape this torch used."""
    if isinstance(dump, dict):
        for key in ("entries", "records"):
            if key in dump:
                return list(dump[key])
        return []
    return list(dump)


def _numel(sizes) -> int:
    """Elements in the first tensor of a collective's input list."""
    if not sizes:
        return 0
    first = sizes[0]
    if isinstance(first, int):          # already flattened
        return int(first)
    n = 1
    for d in first:
        n *= int(d)
    return n


def describe(entry: dict) -> tuple:
    """(op, numel, bytes_or_None, dtype) for one flight-recorder record."""
    op = entry.get("profiling_name") or entry.get("op_name") or entry.get("name") or "?"
    sizes = entry.get("input_sizes") or entry.get("output_sizes") or []
    numel = _numel(sizes)
    dtype = str(entry.get("input_dtypes") or entry.get("dtype") or "?")
    if isinstance(entry.get("input_dtypes"), (list, tuple)) and entry["input_dtypes"]:
        dtype = str(entry["input_dtypes"][0])
    item = ITEMSIZE.get(dtype)
    return op, numel, (numel * item if item else None), dtype


def report(paths, threshold_mib: float = 25.0) -> int:
    if not paths:
        print("ERROR NCCL_TRACE_NO_FILES: nothing matched.")
        print("  A run that SUCCEEDED writes no dump unless -v FR_DUMP=1 was passed.")
        return 2

    per_op: dict = collections.defaultdict(int)
    biggest = None
    unknown_dtypes: set = set()
    n_entries = 0

    for path in paths:
        try:
            dump = load(path)
        except Exception as exc:  # noqa: BLE001
            print("ERROR NCCL_TRACE_UNREADABLE %s: %s: %s"
                  % (path, type(exc).__name__, exc))
            return 2
        recs = entries(dump)
        print("  %-48s %d records" % (os.path.basename(path), len(recs)))
        for e in recs:
            if not isinstance(e, dict):
                continue
            n_entries += 1
            op, numel, nbytes, dtype = describe(e)
            per_op[op] += 1
            if nbytes is None:
                unknown_dtypes.add(dtype)
            key = nbytes if nbytes is not None else numel
            if biggest is None or key > biggest[0]:
                biggest = (key, op, numel, nbytes, dtype, os.path.basename(path))

    if not n_entries:
        print("ERROR NCCL_TRACE_EMPTY: files parsed but hold no collective records.")
        print("  TORCH_NCCL_TRACE_BUFFER_SIZE / TORCH_FR_BUFFER_SIZE may be 0.")
        return 2

    print("\n--- collectives by op ---")
    for op, n in sorted(per_op.items(), key=lambda kv: -kv[1]):
        print("  %-28s %d" % (op, n))

    _, op, numel, nbytes, dtype, where = biggest
    print("\n--- LARGEST SINGLE COLLECTIVE ---")
    print("  op      %s" % op)
    print("  numel   %d" % numel)
    print("  dtype   %s" % dtype)
    if nbytes is None:
        print("  bytes   UNKNOWN -- dtype not in ITEMSIZE. ⚠ Do NOT assume 4 B/element;")
        print("          the dhconv weights are complex64 (8 B). Add the dtype and re-run.")
    else:
        print("  bytes   %d  (%.2f MB decimal, %.2f MiB)"
              % (nbytes, nbytes / 1e6, nbytes / 1024**2))
    print("  from    %s" % where)

    if unknown_dtypes:
        print("\nWARN UNKNOWN_DTYPES: %s -- those records were ranked by element count,"
              % ", ".join(sorted(unknown_dtypes)))
        print("  not by bytes, so the maximum above may be understated.")

    # The verdict, stated in the terms the handoff asks for.
    print("\n--- exposure to the Polaris TREE all-reduce defect ---")
    print("  Measured window: >25 MiB passes, >=1000 MiB fails (Tree). The gap is untested.")
    if nbytes is None:
        print("  VERDICT UNDETERMINED -- see the dtype warning above.")
    elif nbytes / 1024**2 < threshold_mib:
        print("  Largest collective is under %.0f MiB, i.e. under the largest size that has"
              % threshold_mib)
        print("  been measured to PASS on Tree. ACE2 is very likely unexposed.")
    else:
        print("  Largest collective (%.2f MiB) sits ABOVE the largest passing probe and"
              % (nbytes / 1024**2))
        print("  in the untested gap. Keep NCCL_ALGO=Ring, and this is a new data point")
        print("  for the ALCF ticket: probe this exact size app-free with")
        print("  physicsnemo_ai_rossby/polaris/polaris_ai_rossby_nccl_mn_probe.pbs")
        # ceil, not round: the probe must be at least as large as the collective
        # it stands in for, or a pass would not cover the real traffic.
        print("  -v BUCKET_MB=%d." % -(-nbytes // 1024**2))

    print("\nNCCL_TRACE_READ_OK entries=%d largest_bytes=%s"
          % (n_entries, nbytes if nbytes is not None else "unknown"))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="*", help="flight-recorder dump files")
    p.add_argument("--pass-threshold-mib", type=float, default=25.0,
                   help="largest all-reduce size measured to PASS on Tree (default 25)")
    args = p.parse_args(argv)
    return report(args.paths, args.pass_threshold_mib)


if __name__ == "__main__":
    sys.exit(main())
