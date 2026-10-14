#!/usr/bin/env python3
"""port_part0_compare.py — the m5 prereg Part 0 verdicts from port_golden_train.py traces (stdlib only).

  n2   A1 A2 B1 B2 --null-out n2_null.json --clipped-ref-out clipped_ref.json
       P0.1 N2-PRE. Per arm: ``N2_ARM arm=<a|b> clipped=<k>/20 repeat_bitwise=<yes|no>``; then
       ``N2_PRETEST clipped=<min k>/20 bitwise=<yes|no> max_rel=<x> at=<step>:<field>`` (arm a run 1
       vs arm b run 1). Writes the per-step a-vs-b rel as the G3 null and arm a run 1 as the
       clipped reference; ``GOLDEN_MATCH clipped_in_job`` when arm a's two runs are bitwise.
       rc 1 unless both arms are clipped=20/20 and repeat_bitwise=yes (ruling 7697688 §2b).
  n1   TRACE REF       N1_TRAIN_PRETEST: ``bitwise=<yes|no> first=<step>:<field>``; rc 1 if not bitwise.
  crps R1 R2 --ref-out crps_ref.json
       P0.3: ``GOLDEN_MATCH crps_in_job`` when loss, grad norm and params sha are bitwise; else
       ``CRPS_IN_JOB_NOT_BITWISE`` (report-only per P0.3) — rc 0 either way, the ref is written
       only when bitwise.

Values in the traces are ``float.hex`` strings; equality is string equality of the hex (bitwise).
"""
from __future__ import annotations

import argparse
import json
import sys

STEPS = 20


def _load(path):
    with open(path) as f:
        return json.load(f)


def _f(h):
    return float.fromhex(h)


def _rel(new, ref):
    a, b = _f(new), _f(ref)
    if a == b:
        return 0.0
    return abs(a - b) / abs(b) if b != 0.0 else float("inf")


def _first_diff(t, r):
    """First (step, field) where two traces differ bitwise, else None."""
    if len(t["steps"]) != len(r["steps"]):
        return ("-", "n_steps")
    for s, q in zip(t["steps"], r["steps"]):
        for field in ("loss", "grad_norm", "clipped"):
            if s.get(field) != q.get(field):
                return (s["step"], field)
    for field in ("train_loss_epoch", "valid_loss"):
        if t.get(field) != r.get(field):
            return ("epoch", field)
    return None


def cmd_n2(a):
    arms = {"a": (_load(a.a1), _load(a.a2)), "b": (_load(a.b1), _load(a.b2))}
    ok, clipped_min = True, STEPS
    for arm, (r1, r2) in arms.items():
        k = sum(1 for s in r1["steps"] if s.get("clipped"))
        n = len(r1["steps"])
        rep = _first_diff(r1, r2) is None
        clipped_min = min(clipped_min, k if n == STEPS else 0)
        print("N2_ARM arm=%s clipped=%d/%d repeat_bitwise=%s" % (arm, k, n, "yes" if rep else "no"))
        ok = ok and n == STEPS and k == STEPS and rep
    ra, rb = arms["a"][0], arms["b"][0]
    null, worst = [], (0.0, "-:-")
    for sa, sb in zip(ra["steps"], rb["steps"]):
        row = {"step": sa["step"]}
        for field in ("loss", "grad_norm"):
            r = _rel(sb[field], sa[field])
            row[field + "_rel"] = r
            if r > worst[0]:
                worst = (r, "%s:%s" % (sa["step"], field))
        null.append(row)
    bitwise = _first_diff(ra, rb) is None
    print("N2_PRETEST clipped=%d/%d bitwise=%s max_rel=%.3e at=%s"
          % (clipped_min, STEPS, "yes" if bitwise else "no", worst[0], worst[1]))
    with open(a.null_out, "w") as f:
        json.dump({"definition": "per-step rel |b-a|/|a|, arm a = pin clip_grads, arm b = main arithmetic, "
                                 "run 1 of each (m5 prereg P0.1, ruling 7697688 row G3)",
                   "steps": null}, f, indent=1, sort_keys=True)
    if _first_diff(*arms["a"]) is None:
        with open(a.clipped_ref_out, "w") as f:
            json.dump(ra, f, indent=1, sort_keys=True)
        print("GOLDEN_MATCH clipped_in_job")
    if not ok:
        print("ERROR N2_PRETEST_RED (both arms must be clipped=20/20 and repeat_bitwise=yes)")
    return 0 if ok else 1


def cmd_n1(a):
    d = _first_diff(_load(a.trace), _load(a.ref))
    if d is None:
        print("N1_TRAIN_PRETEST bitwise=yes first=-")
        return 0
    print("N1_TRAIN_PRETEST bitwise=no first=%s:%s (STOP before m5 prereg Parts 1+)" % d)
    return 1


def cmd_crps(a):
    r1, r2 = _load(a.r1), _load(a.r2)
    d = _first_diff(r1, r2)
    if d is None and r1.get("params_sha256_after") is None:
        d = ("epoch", "params_sha256_after missing")
    elif d is None and r1.get("params_sha256_after") != r2.get("params_sha256_after"):
        d = ("epoch", "params_sha256_after")
    if d is None:
        with open(a.ref_out, "w") as f:
            json.dump(r1, f, indent=1, sort_keys=True)
        print("GOLDEN_MATCH crps_in_job steps=%d" % len(r1["steps"]))
    else:
        print("CRPS_IN_JOB_NOT_BITWISE first=%s:%s (M5 CRPS probe becomes report-only, P0.3)" % d)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    n2 = sub.add_parser("n2")
    for k in ("a1", "a2", "b1", "b2"):
        n2.add_argument(k)
    n2.add_argument("--null-out", required=True)
    n2.add_argument("--clipped-ref-out", required=True)
    n1 = sub.add_parser("n1")
    n1.add_argument("trace")
    n1.add_argument("ref")
    cr = sub.add_parser("crps")
    cr.add_argument("r1")
    cr.add_argument("r2")
    cr.add_argument("--ref-out", required=True)
    a = p.parse_args(argv)
    if a.cmd is None:
        p.error("one of n2, n1, crps")
    return {"n2": cmd_n2, "n1": cmd_n1, "crps": cmd_crps}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
