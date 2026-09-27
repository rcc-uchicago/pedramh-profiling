#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
"""Score an `nccl-tests` run: did every arm finish, and was every answer right?

    python3 parse_nccl_tests.py --log <probe.log> [--ranks 8] [--nodes 2]

PASS = ``ACE2_NCCL_TESTS_OK``.

Called by ``polaris_ace2_tree_probe.pbs``.  It lives in its own file so the pass
gate can be tested without an allocation (``test_parse_nccl_tests.py``) -- the
same reason ``parse_ace2_scaling.py`` is not an inline heredoc.

THE QUESTION THIS INSTRUMENT ANSWERS
    A Polaris **Tree** all-reduce between 25 MiB and 1000 MiB was measured to
    SILENTLY RETURN PARTIALLY REDUCED DATA and then hang (jobs 7569805/7569817/
    7571147).  ACE2 pins ``NCCL_ALGO=Ring`` because of it, and pays Tree's
    small-message latency for the privilege.  But every one of those probes ran
    on aws-ofi-nccl 1.21.1, i.e. **over Ethernet** -- so the defect is real and
    the attribution is not (slingshot handoff §2a).  Re-running the same table on
    the cxi plugin separates "a tcp-path bug" from "an NCCL Tree bug".

THREE WAYS THIS CAN LIE, AND WHAT STOPS EACH
    1. **A hang reads as a pass.**  `mpiexec` killed at walltime, or under
       `timeout`, exits nonzero -- but so does a hundred benign things, and a
       truncated log's last row looks like any other row.  The discriminator is
       nccl-tests' own terminator: ``# Out of bounds values : 0 OK``.  An arm
       without it DID NOT FINISH, however many rows it printed (CLAUDE.md #14),
       and that is exactly what the Tree corruption looked like: rows, then
       silence.
    2. **Corruption reads as a pass.**  nccl-tests only compares against the
       expected reduction when the check is enabled (`-c 1`); with `-c 0` the
       `#wrong` column prints ``N/A``.  A silent-corruption probe run with
       checking off is a label with no failure value, so ``N/A`` is an ERROR
       here, not a blank.
    3. **TCP reads as Slingshot.**  ``Using network AWS Libfabric`` is
       byte-identical on cxi and on the tcp fallback; the discriminating line is
       ``Selected Provider is cxi``.  This is the check whose absence cost three
       weeks (handoff §8), so a multi-node arm with no provider line fails.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# PALS `--label` prefixes every line with "<hostname> <rank>: " on Polaris (NOT
# the bare "<rank>: " the nccl-tests docs show), and NCCL_DEBUG output from 8
# ranks is interleaved with the table. Strip an optional prefix of either shape
# before matching anything.
LABEL_RE = re.compile(r"^\s*(?:\S+\s+)?\d{1,5}:\s")
# One `=== ARM <name> algo=<a> proto=<p> [nodes=<n>] ===` line per mpiexec the
# launcher runs. `nodes=` is optional and overrides --nodes for that arm alone:
# the intra-node control runs inside one node of a 2-node allocation, never
# touches the net plugin, and so has no provider line to find. Without the
# override the fabric guard would fail the one arm whose job is to show what the
# same collective does with the fabric taken out of the picture.
ARM_RE = re.compile(
    r"===\s*ARM\s+(\S+)\s+algo=(\S+)\s+proto=(\S+)(?:\s+nodes=(\d+))?\s*==="
)
# size count type redop root | time algbw busbw #wrong | time algbw busbw #wrong
# The two #wrong columns are captured as STRINGS: "N/A" is a legal value there
# and means the correctness check was off, which is a failure, not a zero.
ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+\S+\s+\S+\s+\S+"
    r"\s+\S+\s+\S+\s+\S+\s+(\S+)"
    r"\s+\S+\s+\S+\s+\S+\s+(\S+)\s*$"
)
# ⚠ A ROW CAN ARRIVE WITH ITS HEAD BITTEN OFF, and the first version of this
# parser silently dropped those -- i.e. their `#wrong` column, the whole point of
# the instrument, went unchecked. MEASURED on job 7631550: PALS `--label
# --line-buffer` interleaves NCCL_DEBUG output from 8 ranks with rank 0's table,
# so a row can be split mid-line and reach the log as just its eight numeric
# tail fields:
#     72481.1   25.16   44.02       0  72469.2   25.16   44.03       0
# That cost the `ace2_startup_*` arms their only row each (rows=0 while the arm
# had plainly completed). Matched separately, and only INSIDE a table, so the
# size is recorded as None rather than guessed -- an unknown size still carries a
# checkable answer.
ROW_TAIL_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+|N/A)"
    r"\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+|N/A)\s*$"
)
# Where a table begins. Anchors ROW_TAIL_RE so a bare run of numbers in the
# launcher's header can never be read as a measurement.
TABLE_START_RE = re.compile(r"#\s*size\s+count\s+type\s+redop")
OOB_RE = re.compile(r"#\s*Out of bounds values\s*:\s*(\d+)\s*(\w+)")
BUSBW_RE = re.compile(r"#\s*Avg bus bandwidth\s*:\s*([\d.]+)")
PROVIDER_RE = re.compile(r"Selected [Pp]rovider is (\w+)")


def parse_log(text: str) -> list:
    """Split the log into arms and read each one's table.

    Everything before the first ``=== ARM`` marker is dropped: it is the
    launcher's header, and a header that happened to contain a digits-first line
    must not be read as a measurement row.
    """
    arms: list = []
    cur = None
    for raw in text.splitlines():
        line = LABEL_RE.sub("", raw)
        m = ARM_RE.search(line)
        if m:
            cur = {
                "name": m.group(1), "algo": m.group(2), "proto": m.group(3),
                "nodes": int(m.group(4)) if m.group(4) else None,
                "rows": [], "terminated": False, "oob": None,
                "avg_busbw": None, "providers": set(), "in_table": False,
            }
            arms.append(cur)
            continue
        if cur is None:
            continue
        if TABLE_START_RE.search(line):
            cur["in_table"] = True
            continue
        prov = PROVIDER_RE.search(line)
        if prov:
            cur["providers"].add(prov.group(1).lower())
        oob = OOB_RE.search(line)
        if oob:
            # The terminator. nccl-tests prints it once, last, and only after the
            # whole sweep completes -- which is precisely why its ABSENCE is the
            # hang signature this probe exists to record.
            cur["terminated"] = True
            cur["oob"] = (int(oob.group(1)), oob.group(2).upper())
            continue
        bw = BUSBW_RE.search(line)
        if bw:
            cur["avg_busbw"] = float(bw.group(1))
            continue
        row = ROW_RE.match(line)
        if row:
            cur["in_table"] = True
            cur["rows"].append({
                "size": int(row.group(1)),
                "count": int(row.group(2)),
                "wrong_in_place": row.group(3),
                "wrong_out_of_place": row.group(4),
                "split": False,
            })
            continue
        tail = ROW_TAIL_RE.match(line) if cur["in_table"] else None
        if tail:
            cur["rows"].append({
                "size": None,            # the size field was lost with the prefix
                "count": None,
                "wrong_in_place": tail.group(4),
                "wrong_out_of_place": tail.group(8),
                "split": True,
            })
    return arms


def _wrong_sizes(arm: dict) -> list:
    """Sizes whose reduction did not match the expected answer."""
    bad = []
    for row in arm["rows"]:
        for key in ("wrong_in_place", "wrong_out_of_place"):
            val = row[key]
            if val in ("N/A", "n/a"):
                continue          # reported separately -- a different defect
            # A split row reports `size=None`; it is still checked, and named as
            # "<unknown, split row>" rather than dropped.
            where = row["size"] if row["size"] is not None else "<split row>"
            try:
                if int(val) != 0:
                    bad.append((where, key, val))
            except ValueError:
                bad.append((where, key, val))
    return bad


def _unchecked(arm: dict) -> bool:
    return any(row["wrong_in_place"] in ("N/A", "n/a") for row in arm["rows"])


def check(arms: list, nodes: int, ranks: int) -> int:
    """Print a verdict per arm. Returns a process exit code."""
    if not arms:
        print("ERROR NCCL_TESTS_NO_ARMS: no '=== ARM ... ===' marker in the log.")
        print("  The launcher prints one per mpiexec, so none means nothing ran --")
        print("  not that everything passed.")
        return 4

    rc = 0
    n_ok = 0
    for arm in arms:
        provider = "|".join(sorted(arm["providers"])) if arm["providers"] else "UNKNOWN"
        ok = True

        if not arm["rows"]:
            print("ERROR ARM_NO_TABLE %s: the arm printed no measurement rows."
                  % arm["name"])
            print("  It died before the sweep began -- read its stderr above; this")
            print("  arm says nothing about Tree, Ring or the fabric.")
            ok = False
        elif not arm["terminated"]:
            # THE signature of the defect under investigation.
            last = arm["rows"][-1]["size"] or -1
            print("ERROR ARM_DID_NOT_FINISH %s (algo=%s): %d rows, last size %d B, "
                  "no 'Out of bounds values' line."
                  % (arm["name"], arm["algo"], len(arm["rows"]), last))
            print("  The sweep stopped mid-way. That is what the tcp-era Tree")
            print("  corruption looked like (7569805/7569817/7571147): rows, then")
            print("  silence. Record the last size -- it IS the measurement.")
            ok = False

        if _unchecked(arm):
            print("ERROR ARM_CHECK_DISABLED %s: #wrong reads N/A." % arm["name"])
            print("  nccl-tests was run without `-c 1`, so it never compared the")
            print("  result against the expected reduction. A silent-corruption")
            print("  probe that cannot detect corruption is a label, not a guard.")
            ok = False

        bad = _wrong_sizes(arm)
        if bad:
            print("ERROR ARM_WRONG_VALUES %s (algo=%s): %d size(s) reduced wrongly."
                  % (arm["name"], arm["algo"], len(bad)))
            for size, which, val in bad[:6]:
                # %s, not %d: a split row's size is the string "<split row>".
                print("    size=%s B %s=%s" % (size, which, val))
            print("  This is SILENT CORRUPTION, not a crash: the collective")
            print("  returned and the data is wrong. No tolerance applies")
            print("  (CLAUDE.md #11).")
            ok = False

        if arm["oob"] and arm["oob"][1] != "OK":
            print("ERROR ARM_OUT_OF_BOUNDS %s: terminator says '%d %s'."
                  % (arm["name"], arm["oob"][0], arm["oob"][1]))
            ok = False

        # The fabric, asked per arm: each arm is its own mpiexec and so its own
        # NCCL init, and a probe comparing Tree against Ring over two different
        # transports would be comparing nothing.
        arm_nodes = arm["nodes"] if arm["nodes"] is not None else nodes
        if arm_nodes > 1:
            if provider == "UNKNOWN":
                print("ERROR ARM_FABRIC_UNVERIFIED %s: no 'Selected Provider' line."
                      % arm["name"])
                print("  Set NCCL_DEBUG=INFO. Without it the arm cannot say which")
                print("  wire it used, and that is the whole question here.")
                ok = False
            elif provider != "cxi":
                if os.environ.get("ALLOW_TCP") == "1":
                    print("WARN ARM_FABRIC_TCP_ACKNOWLEDGED %s: provider=%s, ALLOW_TCP=1."
                          % (arm["name"], provider))
                    print("  This arm re-measures the TCP path deliberately. It cannot")
                    print("  settle whether the Tree defect survives on Slingshot.")
                else:
                    print("ERROR ARM_FABRIC_NOT_SLINGSHOT %s: provider=%s, expected cxi."
                          % (arm["name"], provider))
                    print("  The point of this probe is the fabric. A tcp arm just")
                    print("  re-measures what handoff §2a already recorded.")
                    ok = False

        n_split = sum(1 for r in arm["rows"] if r.get("split"))
        if n_split:
            # A WARN, not an error: the answer is intact, only the size label is
            # lost. Named because a silent recovery would hide that the table and
            # the debug stream are fighting over one stdout.
            print("WARN ARM_SPLIT_ROWS %s: %d of %d rows lost their size prefix to "
                  "interleaved NCCL_DEBUG output." % (arm["name"], n_split, len(arm["rows"])))
            print("  Their #wrong columns ARE still checked. Set NCCL_DEBUG_FILE so the")
            print("  table and the debug stream stop sharing stdout (the launcher does).")
        print("  %-20s algo=%-6s proto=%-6s provider=%-8s rows=%-3d (split %d) "
              "finished=%-5s avg_busbw=%s"
              % (arm["name"], arm["algo"], arm["proto"], provider, len(arm["rows"]),
                 n_split, arm["terminated"],
                 "%.2f" % arm["avg_busbw"] if arm["avg_busbw"] is not None else "-"))
        if ok:
            n_ok += 1
        else:
            rc = 4

    print("arms OK: %d/%d  (nodes=%d ranks=%d)" % (n_ok, len(arms), nodes, ranks))
    if rc == 0:
        print("ACE2_NCCL_TESTS_OK arms=%d/%d" % (n_ok, len(arms)))
    else:
        print("ERROR ACE2_NCCL_TESTS_FAILED arms=%d/%d" % (n_ok, len(arms)))
    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", required=True)
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--ranks", type=int, default=0)
    args = p.parse_args(argv)

    with open(args.log, errors="replace") as fh:
        arms = parse_log(fh.read())
    return check(arms, args.nodes, args.ranks)


if __name__ == "__main__":
    sys.exit(main())
