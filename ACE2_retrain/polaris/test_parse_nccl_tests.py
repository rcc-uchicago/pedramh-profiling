#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
"""Tests for parse_nccl_tests.py -- no allocation, no GPU, no torch.

    python3 ACE2_retrain/polaris/test_parse_nccl_tests.py
    pytest -q ACE2_retrain/polaris/test_parse_nccl_tests.py

PASS = ACE2_NCCL_TESTS_PARSE_OK.

The rows are copied from a REAL 4-node log (job 7629096, recorded verbatim in
polaris_nccl_metrics.md §3) rather than invented, because every guard here keys
on nccl-tests' exact output shape: the two `#wrong` columns, the `Out of bounds
values` terminator, and the PALS label that sits in front of all of it.

Each test is a way this probe could report a pass it has not earned -- a
truncated sweep, checking switched off, a tcp arm -- which are the three ways the
TCP-era Tree measurements were readable as fabric evidence when they were not.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parse_nccl_tests as P  # noqa: E402

PALS = "x3005c0s13b0n0.hsn.cm.polaris.alcf.anl.gov 0: "
HDR = (PALS + "#       size         count      type   redop    root     time   "
       "algbw   busbw  #wrong     time   algbw   busbw  #wrong")
PROV = PALS + "x3005c0s13b0n0:12345:12400 [0] NCCL INFO NET/OFI Selected Provider is cxi (found 2 nics)"
PROV_TCP = (PALS + "x3005c0s13b0n0:12345:12400 [0] NCCL INFO NET/OFI Selected "
            "provider is tcp, fabric is 10.201.0.0/16 (found 2 nics)")

# Verbatim from 7629096 (metrics §3), label prefix added as PALS emits it.
ROWS = [
    "   134217728      33554432     float     sum      -1  8776.81   15.29   28.67       0  8860.42   15.15   28.40       0",
    "   268435456      67108864     float     sum      -1  17560.8   15.29   28.66       0  17598.6   15.25   28.60       0",
    "  1073741824     268435456     float     sum      -1  70570.3   15.22   28.53       0  70527.4   15.22   28.55       0",
    "  2147483648     536870912     float     sum      -1   141223   15.21   28.51       0   141004   15.23   28.56       0",
]
TERM = [PALS + "# Out of bounds values : 0 OK",
        PALS + "# Avg bus bandwidth    : 10.4127 "]


def arm(name="tree", algo="Tree", rows=None, provider=PROV, terminated=True,
        nodes=None):
    rows = ROWS if rows is None else rows
    marker = "=== ARM %s algo=%s proto=Simple%s ===" % (
        name, algo, "" if nodes is None else " nodes=%d" % nodes)
    out = [marker, provider or "", HDR]
    out += [PALS + r for r in rows]
    if terminated:
        out += TERM
    return "\n".join(x for x in out if x) + "\n"


def run(text, nodes=2, ranks=8):
    """check() on a parsed log. Returns (rc, stdout)."""
    with tempfile.TemporaryDirectory() as td:
        log = os.path.join(td, "probe.log")
        with open(log, "w") as fh:
            fh.write(text)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = P.main(["--log", log, "--nodes", str(nodes), "--ranks", str(ranks)])
    return rc, buf.getvalue()


# --- the happy path -------------------------------------------------------


def test_two_clean_arms_pass():
    rc, out = run(arm("tree_2g", "Tree") + arm("ring_2g", "Ring"))
    assert rc == 0, out
    assert "ACE2_NCCL_TESTS_OK arms=2/2" in out


def test_rows_and_metadata_are_read_through_the_pals_label():
    arms = P.parse_log(arm())
    assert len(arms) == 1
    assert arms[0]["algo"] == "Tree"
    assert len(arms[0]["rows"]) == 4
    assert arms[0]["rows"][-1]["size"] == 2147483648
    assert arms[0]["providers"] == {"cxi"}
    assert arms[0]["avg_busbw"] == 10.4127


def test_header_before_the_first_arm_marker_is_not_a_row():
    """The launcher's own header prints digits-first lines (node lists, sizes)."""
    noise = "1073741824 268435456 float sum -1 1.0 1.0 1.0 0 1.0 1.0 1.0 0\n"
    arms = P.parse_log(noise + arm())
    assert len(arms) == 1
    assert len(arms[0]["rows"]) == 4


# --- the three ways a probe lies ------------------------------------------


def test_truncated_sweep_is_not_a_pass():
    """THE case: the Tree defect printed rows and then went silent.

    Under `timeout` the arm exits nonzero, but a shorter sweep is also what a
    smaller `-e` produces -- so the terminator, not the row count, is the gate.
    """
    rc, out = run(arm("tree_2g", "Tree", terminated=False))
    assert rc == 4
    assert "ARM_DID_NOT_FINISH" in out
    assert "2147483648" in out          # names the size it stopped at


def test_wrong_values_are_reported_per_size():
    """Silent corruption: the collective returned, and the answer is wrong."""
    bad = list(ROWS)
    bad[2] = bad[2].replace("28.53       0", "28.53   33554432")
    rc, out = run(arm("tree_1g", "Tree", rows=bad))
    assert rc == 4
    assert "ARM_WRONG_VALUES" in out
    assert "1073741824" in out


def test_split_row_still_has_its_wrong_column_checked():
    """MEASURED, job 7631550: 8 ranks' NCCL_DEBUG split table rows mid-line.

    The first version of this parser dropped such rows silently, so their `#wrong`
    column -- the only thing this probe exists to read -- went unchecked. A split
    row must still be counted, still be checked, and be NAMED as split.
    """
    # the real shape: eight numeric tail fields, prefix gone
    split_ok = "      72481.1   25.16   44.02       0  72469.2   25.16   44.03       0"
    rc, out = run(arm("ace2_startup_Tree", "Tree", rows=[split_ok]))
    assert rc == 0, out
    assert "ARM_SPLIT_ROWS" in out
    assert "ARM_NO_TABLE" not in out          # the arm plainly ran

    split_bad = split_ok.replace("44.03       0", "44.03   33554432")
    rc, out = run(arm("ace2_startup_Tree", "Tree", rows=[split_bad]))
    assert rc == 4, out
    assert "ARM_WRONG_VALUES" in out
    assert "split row" in out                 # named, not silently sized


def test_numbers_outside_a_table_are_not_read_as_a_split_row():
    """The tail pattern is 8 bare numbers -- it must be anchored to a table."""
    noise = ("=== ARM x algo=Tree proto=Simple nodes=1 ===\n"
             "  72481.1   25.16   44.02       0  72469.2   25.16   44.03       0\n")
    arms = P.parse_log(noise)
    assert arms[0]["rows"] == []


def test_check_disabled_is_an_error_not_a_blank():
    """`-c 0` prints N/A -- a corruption probe that cannot see corruption."""
    na = [r.replace("28.67       0", "28.67     N/A").replace("28.40       0", "28.40     N/A")
          for r in ROWS]
    rc, out = run(arm("tree_2g", "Tree", rows=na))
    assert rc == 4
    assert "ARM_CHECK_DISABLED" in out


def test_tcp_arm_is_rejected():
    rc, out = run(arm("tree_2g", "Tree", provider=PROV_TCP))
    assert rc == 4
    assert "ARM_FABRIC_NOT_SLINGSHOT" in out


def test_multinode_arm_without_a_provider_line_is_rejected():
    rc, out = run(arm("tree_2g", "Tree", provider=None))
    assert rc == 4
    assert "ARM_FABRIC_UNVERIFIED" in out


def test_single_node_arm_needs_no_provider_line():
    """1 node never initialises the net plugin; demanding one fails the control."""
    rc, out = run(arm("intra_node", "Tree", provider=None), nodes=1, ranks=4)
    assert rc == 0, out
    assert "FABRIC" not in out


def test_per_arm_nodes_override_exempts_the_intra_node_control():
    """The control runs inside ONE node of a 2-node allocation.

    Without the marker's `nodes=1` the fabric guard would fail the only arm that
    deliberately takes the fabric out of the comparison.
    """
    text = arm("intra_control", "Tree", provider=None, nodes=1) + arm("tree_2g", "Tree")
    rc, out = run(text, nodes=2)
    assert rc == 0, out
    assert "ACE2_NCCL_TESTS_OK arms=2/2" in out


def test_allow_tcp_downgrades_to_a_warning():
    os.environ["ALLOW_TCP"] = "1"
    try:
        rc, out = run(arm("tree_2g", "Tree", provider=PROV_TCP))
    finally:
        del os.environ["ALLOW_TCP"]
    assert rc == 0, out
    assert "ARM_FABRIC_TCP_ACKNOWLEDGED" in out


def test_an_arm_that_never_printed_a_table_is_an_error():
    rc, out = run("=== ARM tree_2g algo=Tree proto=Simple ===\n"
                  "mpiexec: rank 3 exited with code 1\n")
    assert rc == 4
    assert "ARM_NO_TABLE" in out


def test_no_arms_at_all_is_an_error_not_a_pass():
    """An empty log must not read as 'nothing went wrong'."""
    rc, out = run("qsub: job killed: walltime exceeded\n")
    assert rc == 4
    assert "NCCL_TESTS_NO_ARMS" in out


def test_nonzero_out_of_bounds_terminator_is_caught():
    text = arm().replace("Out of bounds values : 0 OK",
                         "Out of bounds values : 12 FAILED")
    rc, out = run(text)
    assert rc == 4
    assert "ARM_OUT_OF_BOUNDS" in out


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
        print("ERROR ACE2_NCCL_TESTS_PARSE_FAILED (%d/%d)" % (failed, len(tests)))
        sys.exit(1)
    print("ACE2_NCCL_TESTS_PARSE_OK (%d tests)" % len(tests))
