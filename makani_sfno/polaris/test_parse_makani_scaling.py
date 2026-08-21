#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
"""Tests for parse_makani_scaling.py -- runnable with no allocation and no GPU.

    python makani_sfno/polaris/test_parse_makani_scaling.py    # PASS = MAKANI_SCALING_PARSE_OK
    pytest -q makani_sfno/polaris/test_parse_makani_scaling.py

The point is not the arithmetic; it is the two silent-failure guards.  A
multi-node scaling row that came from N independent world_size=1 trainers, or
from a TCP fallback instead of the Slingshot plugin, is a plausible number that
means nothing -- and both cost real allocation time to discover any other way.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parse_makani_scaling as P  # noqa: E402

# A stand-in for the lines makani actually prints. Kept verbatim in shape so a
# future makani release that renames one breaks a test rather than a campaign.
GOOD_LOG = """\
Communicators wireup time: 3.42s
[0] NCCL INFO NET/OFI Using network AWS Libfabric
[1] NCCL INFO NET/OFI Using network AWS Libfabric
Initializing model on 16 ranks, world_size=16
Average step time after step 10: 512.7 ms
Average effective io rate after step 10: 1.82 GB/s
Current loss 2.7213
Average step time after step 20: 498.3 ms
Average effective io rate after step 20: 1.91 GB/s
Total training time is 41.55 sec
"""

BASE_ARGV = [
    "--nodes", "4",
    "--ranks", "16",
    "--local-batch", "1",
    "--global-batch", "16",
    "--steps", "20",
    "--jobid", "7600001",
    "--env-source", "manual-reconstruction",
    "--omp-threads", "1",
]


def _run(text, tmpdir, extra=None, csv_name="scaling.csv"):
    log = os.path.join(tmpdir, "run.log")
    with open(log, "w") as fh:
        fh.write(text)
    csv_path = os.path.join(tmpdir, csv_name)
    argv = ["--log", log, "--csv", csv_path] + BASE_ARGV + (extra or [])
    rc = P.main(argv)
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rc, rows, csv_path


def test_happy_path():
    with tempfile.TemporaryDirectory() as td:
        rc, rows, _ = _run(GOOD_LOG, td)
        assert rc == 0, "clean log should pass, got rc=%s" % rc
        assert len(rows) == 1
        r = rows[0]
        # The LAST running average, not the first: it covers the most steps.
        assert r["step_ms"] == "498.3", r["step_ms"]
        assert r["io_gbs"] == "1.91", r["io_gbs"]
        assert r["total_train_s"] == "41.55"
        assert r["wireup_s"] == "3.42"
        assert r["transport"] == "AWS Libfabric"
        assert r["world_sizes_seen"] == "16"
        # 1 sample / 0.4983 s = 2.0068 samples/s/rank, x16 ranks.
        assert abs(float(r["samples_s_rank"]) - 2.0068) < 1e-3, r["samples_s_rank"]
        assert abs(float(r["samples_s_total"]) - 32.109) < 1e-2, r["samples_s_total"]


def test_world_size_mismatch_is_an_error():
    """The failure mode the rank shim exists to prevent."""
    bad = GOOD_LOG.replace("world_size=16", "world_size=1")
    with tempfile.TemporaryDirectory() as td:
        rc, rows, _ = _run(bad, td)
        assert rc == 4, "16-rank launch reporting world_size=1 must fail, got rc=%s" % rc
        # The row is still written: a rejected measurement is evidence too, and
        # dropping it would hide that the run happened at all.
        assert rows[0]["world_sizes_seen"] == "1"


def test_missing_step_timing_is_an_error():
    no_timing = "\n".join(
        ln for ln in GOOD_LOG.splitlines() if "Average step time" not in ln
    )
    with tempfile.TemporaryDirectory() as td:
        rc, rows, _ = _run(no_timing, td)
        assert rc == 4, "a log with no step timing is not a measurement"
        assert rows[0]["step_ms"] == ""
        assert rows[0]["samples_s_rank"] == ""


def test_unknown_transport_warns_but_passes():
    """Timing is still usable; it just cannot be attributed to a transport."""
    quiet = GOOD_LOG.replace("Using network AWS Libfabric", "some other line")
    with tempfile.TemporaryDirectory() as td:
        rc, rows, _ = _run(quiet, td)
        assert rc == 0, "an unknown transport is a warning, not a failure"
        assert rows[0]["transport"] == "UNKNOWN"


def test_append_preserves_schema_and_accumulates():
    with tempfile.TemporaryDirectory() as td:
        rc1, rows1, csv_path = _run(GOOD_LOG, td)
        # A second, smaller arm into the same file -- the actual sweep pattern.
        log2 = os.path.join(td, "run2.log")
        with open(log2, "w") as fh:
            fh.write(GOOD_LOG.replace("world_size=16", "world_size=4"))
        rc2 = P.main(
            ["--log", log2, "--csv", csv_path,
             "--nodes", "1", "--ranks", "4", "--local-batch", "1",
             "--global-batch", "4", "--steps", "20"]
        )
        assert rc1 == 0 and rc2 == 0
        with open(csv_path, newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            body = list(reader)
        assert header == P.FIELDS, "header must stay the contract"
        assert len(body) == 2
        assert [r[1] for r in body] == ["4", "1"], "nodes column, in write order"


def test_schema_drift_refuses_to_append():
    """A file written by an older column set must not be silently appended to."""
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "old.csv")
        with open(csv_path, "w", newline="") as fh:
            csv.writer(fh).writerow(["nodes", "ranks", "step_ms"])   # an older shape
        log = os.path.join(td, "run.log")
        with open(log, "w") as fh:
            fh.write(GOOD_LOG)
        try:
            P.main(["--log", log, "--csv", csv_path] + BASE_ARGV)
        except SystemExit as exc:
            assert "SCALING_CSV_SCHEMA_DRIFT" in str(exc), str(exc)
        else:
            raise AssertionError("appending to a drifted schema must raise")


def test_nsys_row_is_flagged_and_may_lack_total_time():
    """An nsys run ends at capture_range_stop, so it is not a clean timing row.

    makani's CUDAProfiler calls sys.exit(0) at the stop step, so "Total training
    time" never prints. The row must still be usable (it carries the per-rank
    captures) but must be distinguishable, or it gets averaged in with the
    full-length arms and quietly drags the mean.
    """
    truncated = "\n".join(
        ln for ln in GOOD_LOG.splitlines() if "Total training time" not in ln
    )
    with tempfile.TemporaryDirectory() as td:
        rc, rows, _ = _run(truncated, td, extra=["--nsys", "1"])
        assert rc == 0, "a truncated nsys run is not itself a failure"
        assert rows[0]["nsys"] == "1"
        assert rows[0]["total_train_s"] == ""
        assert rows[0]["step_ms"] == "498.3", "step timing still usable"


def test_default_row_is_not_flagged_as_nsys():
    with tempfile.TemporaryDirectory() as td:
        _, rows, _ = _run(GOOD_LOG, td)
        assert rows[0]["nsys"] == "0"


def test_spatial_parallel_row_records_the_decomposition():
    """The paper's stage-1 shape: 16 ranks as 4 data x 4 spatial."""
    with tempfile.TemporaryDirectory() as td:
        rc, rows, _ = _run(
            GOOD_LOG, td,
            extra=["--h-par", "2", "--w-par", "2", "--data", "synthetic",
                   "--gpu-order", "reverse", "--rep", "3"],
        )
        assert rc == 0
        r = rows[0]
        assert (r["h_par"], r["w_par"]) == ("2", "2")
        assert r["data"] == "synthetic" and r["gpu_order"] == "reverse" and r["rep"] == "3"


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failures = 0
    for fn in TESTS:
        try:
            fn()
            print("ok   %s" % fn.__name__)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("FAIL %s: %s" % (fn.__name__, exc))
    if failures:
        print("ERROR %d/%d tests failed" % (failures, len(TESTS)))
        sys.exit(1)
    print("MAKANI_SCALING_PARSE_OK %d tests" % len(TESTS))
