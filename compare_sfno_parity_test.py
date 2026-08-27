#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the §4.9 dhconv-XIO carve-out in ``compare_sfno_parity.py``.

    python compare_sfno_parity_test.py          # PASS = SFNO_PARITY_XIO_TEST_OK
    pytest -q compare_sfno_parity_test.py

Standard library only, no torch: safe on a login node (CLAUDE.md #3).

WHY THIS EXISTS.  The carve-out widens a correctness gate, and a widened gate
that is wrong in the permissive direction is worse than no gate.  The claim it
rests on is narrow: **while ``PANGU_DHCONV_XIO`` is unset the two trees run the
same arithmetic, and while it is set they are different models** (the dhconv
weight is stored ``[modes_lat, in, out]`` rather than ``[in, out, modes_lat]``,
so 95.8% of the parameters change shape).  These tests pin both halves --
especially the second, which is the one nobody would notice was broken.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "compare_sfno_parity", Path(__file__).resolve().parent / "compare_sfno_parity.py"
)
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

FAILURES = []


def check(cond, name):
    print("  %s %s" % ("ok  " if cond else "FAIL", name))
    if not cond:
        FAILURES.append(name)


@contextlib.contextmanager
def knob(value):
    """Set/unset PANGU_DHCONV_XIO for the duration, restoring the old value."""
    old = os.environ.get(P.XIO_ENV)
    if value is None:
        os.environ.pop(P.XIO_ENV, None)
    else:
        os.environ[P.XIO_ENV] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(P.XIO_ENV, None)
        else:
            os.environ[P.XIO_ENV] = old


def _source_check():
    """check_source()'s exit code and its stdout."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = P.check_source()
    return rc, buf.getvalue()


def test_knob_off_passes():
    with knob(None):
        rc, out = _source_check()
    check(rc == 0, "knob unset: source parity passes")
    check("SFNO_SOURCE_DIVERGED" not in out, "knob unset: no divergence error")


def test_knob_explicit_zero_passes():
    with knob("0"):
        rc, _ = _source_check()
    check(rc == 0, "knob=0 behaves like unset")


def test_knob_on_fails_loudly():
    with knob("1"):
        rc, out = _source_check()
    check(rc == 1, "knob=1: source parity FAILS")
    check("SFNO_XIO_KNOB_SET" in out, "knob=1: its own greppable error token")
    # The failure must NOT be reported as a generic file-level divergence: with
    # the knob on the two sides are different MODELS, and the message has to say
    # so or the next reader will 'fix' it by extending the allowlist.
    check("SFNO_SOURCE_DIVERGED" not in out,
          "knob=1: reported as a model difference, not a line diff")


def test_is_allowed_is_conditional():
    line = 'if self.operator_type == "dhconv" and dhconv_weight_is_xio():'
    check(line in P.XIO_KNOB_DIFF_LINES, "the sample line is in the carve-out set")
    with knob(None):
        off = P._is_allowed(line)
    with knob("1"):
        on = P._is_allowed(line)
    check(off is True, "_is_allowed: XIO line allowed while the knob is off")
    check(on is False, "_is_allowed: same line NOT allowed while the knob is on")


def test_unrelated_line_never_allowed():
    line = "self.weight = nn.Parameter(torch.zeros(3, 4))"   # not in either set
    with knob(None):
        off = P._is_allowed(line)
    with knob("1"):
        on = P._is_allowed(line)
    check(off is False and on is False,
          "an unrelated changed line is rejected in both states")


def test_carve_out_disjoint_from_unconditional_allowlist():
    # _is_allowed consults ALLOWED_DIFF_LINES FIRST, so a line present in both
    # sets would be allowed even with the knob on -- i.e. the conditional half of
    # this carve-out would be silently dead for that line.
    overlap = P.XIO_KNOB_DIFF_LINES & P.ALLOWED_DIFF_LINES
    check(not overlap, "carve-out disjoint from the unconditional allowlist")


def test_comments_and_blanks_still_free():
    with knob("1"):
        check(P._is_allowed("   ") and P._is_allowed("# a comment"),
              "blanks and comments stay inert even with the knob on")


def main() -> int:
    for fn in sorted(
        (v for k, v in globals().items() if k.startswith("test_")),
        key=lambda f: f.__code__.co_firstlineno,
    ):
        fn()
    if FAILURES:
        print("ERROR SFNO_PARITY_XIO_TEST_FAILED: %s" % ", ".join(FAILURES))
        return 1
    print("SFNO_PARITY_XIO_TEST_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
