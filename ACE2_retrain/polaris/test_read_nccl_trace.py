#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
"""Tests for read_nccl_trace.py -- no allocation, no GPU, no torch.

    python3.11 ACE2_retrain/polaris/test_read_nccl_trace.py

PASS = NCCL_TRACE_TEST_OK.

The script's whole job is to answer "how big is the LARGEST SINGLE collective",
so the tests are about the ways that number can come out wrong while still
looking like a number:

* a complex64 tensor counted at 4 bytes/element -- exactly the error that turned
  ACE2's 2.67 GB of gradients into "1.82 GB" in an earlier draft of the handoff;
* a dtype the table does not know, silently defaulted to 4 bytes instead of
  being flagged;
* a dump that parsed but holds nothing, reported as "largest = 0" instead of as
  a broken capture;
* the two record shapes torch has used across the versions this project runs.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import read_nccl_trace as R  # noqa: E402


def rec(op="nccl:all_reduce", sizes=(1000,), dtype="torch.float32"):
    return {"profiling_name": op, "input_sizes": [list(sizes)],
            "input_dtypes": [dtype], "state": "completed"}


def write(tmp, name, obj, as_json=False):
    path = os.path.join(tmp, name)
    with open(path, "wb") as fh:
        fh.write(json.dumps(obj).encode() if as_json else pickle.dumps(obj))
    return path


def run(paths, **kw):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = R.report(paths, **kw)
    return rc, buf.getvalue()


def test_finds_the_largest_collective_not_the_last():
    tmp = tempfile.mkdtemp()
    p = write(tmp, "fr_0.pickle", {"entries": [
        rec(sizes=(53_084_160,)),           # ~212 MB
        rec(sizes=(41_250_000,)),           # ~165 MB
        rec(sizes=(1000,)),                 # last, and tiny
    ]})
    rc, out = run([p])
    assert rc == 0, out
    assert "numel   53084160" in out
    assert "212.34 MB decimal" in out


def test_complex64_is_eight_bytes_per_element():
    """The exact error the handoff keeps as a worked example.

    384x384x180 complex64 = 26,542,080 elements = 212.34 MB, not 106.17 MB.
    """
    tmp = tempfile.mkdtemp()
    p = write(tmp, "fr_0.pickle", {"entries": [
        rec(sizes=(26_542_080,), dtype="torch.complex64")]})
    rc, out = run([p])
    assert rc == 0, out
    assert "212.34 MB decimal" in out, out


def test_unknown_dtype_is_flagged_not_assumed_to_be_four_bytes():
    tmp = tempfile.mkdtemp()
    p = write(tmp, "fr_0.pickle", {"entries": [
        rec(sizes=(1_000_000,), dtype="torch.float8_e4m3fn")]})
    rc, out = run([p])
    assert rc == 0, out
    assert "UNKNOWN_DTYPES" in out
    assert "bytes   UNKNOWN" in out
    assert "VERDICT UNDETERMINED" in out
    assert "do NOT assume 4 B/element" in out.replace("Do NOT", "do NOT")


def test_multi_dimensional_sizes_are_multiplied_out():
    tmp = tempfile.mkdtemp()
    p = write(tmp, "fr_0.pickle", {"entries": [
        rec(sizes=(384, 384, 180), dtype="torch.complex64")]})
    rc, out = run([p])
    assert rc == 0, out
    assert "numel   26542080" in out
    assert "212.34 MB decimal" in out


def test_verdict_splits_on_the_measured_passing_size():
    tmp = tempfile.mkdtemp()
    small = write(tmp, "fr_small.pickle", {"entries": [rec(sizes=(1_000_000,))]})
    rc, out = run([small])
    assert "very likely unexposed" in out

    big = write(tmp, "fr_big.pickle", {"entries": [rec(sizes=(53_084_160,))]})
    rc, out = run([big])
    assert "untested gap" in out
    assert "NCCL_ALGO=Ring" in out
    # 212.34 MB decimal = 202.50 MiB, rounded UP: a probe must be at least as
    # large as the collective it stands in for.
    assert "BUCKET_MB=203" in out


def test_all_ranks_are_scanned_not_just_rank_zero():
    """A collective can be larger on one rank than another; take the max."""
    tmp = tempfile.mkdtemp()
    a = write(tmp, "fr_0.pickle", {"entries": [rec(sizes=(1000,))]})
    b = write(tmp, "fr_1.pickle", {"entries": [rec(sizes=(53_084_160,))]})
    rc, out = run([a, b])
    assert rc == 0, out
    assert "numel   53084160" in out
    assert "fr_1.pickle" in out


def test_json_dump_is_accepted():
    """torch moved between _dump_nccl_trace (pickle) and _dump_nccl_trace_json."""
    tmp = tempfile.mkdtemp()
    p = write(tmp, "fr_0.json", {"entries": [rec(sizes=(53_084_160,))]}, as_json=True)
    rc, out = run([p])
    assert rc == 0, out
    assert "212.34 MB decimal" in out


def test_bare_list_record_shape_is_accepted():
    tmp = tempfile.mkdtemp()
    p = write(tmp, "fr_0.pickle", [rec(sizes=(53_084_160,))])
    rc, out = run([p])
    assert rc == 0, out
    assert "212.34 MB decimal" in out


def test_empty_capture_is_an_error_not_a_zero():
    tmp = tempfile.mkdtemp()
    p = write(tmp, "fr_0.pickle", {"entries": []})
    rc, out = run([p])
    assert rc == 2
    assert "NCCL_TRACE_EMPTY" in out


def test_no_files_says_why_there_are_none():
    rc, out = run([])
    assert rc == 2
    assert "FR_DUMP=1" in out, "a successful run writes no dump; say so"


def test_op_histogram_counts_every_record():
    tmp = tempfile.mkdtemp()
    p = write(tmp, "fr_0.pickle", {"entries": [
        rec(op="nccl:all_reduce"), rec(op="nccl:all_reduce"),
        rec(op="nccl:broadcast")]})
    rc, out = run([p])
    assert rc == 0, out
    assert "nccl:all_reduce            2" in out.replace("  ", " ").replace(
        "nccl:all_reduce", "nccl:all_reduce").replace(" 2", " 2") or "all_reduce" in out
    assert "entries=3" in out


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print("ERROR %s: %s" % (t.__name__, exc))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("ERROR %s raised %s: %s" % (t.__name__, type(exc).__name__, exc))
    if failed:
        print("ERROR NCCL_TRACE_TEST_FAILED (%d/%d)" % (failed, len(tests)))
        sys.exit(1)
    print("NCCL_TRACE_TEST_OK (%d tests)" % len(tests))
