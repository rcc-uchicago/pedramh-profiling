"""doses_from_trab.py -- gate D's doses read from a gates A/B results dir, so run_tr_gateD_polaris.pbs can be submitted
before A/B has run (TRAB_RES; PREDICTIONS_gates_inrow.md). The rule is PREDICTIONS_gates_tilerows.md's, unchanged:
alpha_lo = max(1, 1000 / R) to 2 significant figures, alpha_hi = 10 alpha_lo, R = gates_ab_tr.py's size_ratio.
Standard library only (python >= 3.6).

  python3 doses_from_trab.py --res <polaris_trab_<job>> --candidate <arm> [--log <the A/B job's PBS output>]
      -> prints "TRD_DOSES <alpha_lo> <alpha_hi> <pattern json> <R>"
  python3 doses_from_trab.py --self-test

Refuses (an ERROR line, exit 2) unless the A/B run is valid, judged this candidate, wrote its pattern and a finite
R > 0, and (with --log) ended TRAB_JOB_DONE failures=0. Gate D runs whatever gate B's verdict is.
"""
import argparse
import json
import math
import os
import sys
import tempfile

OLD_CANDIDATE = "tilerows_if25"        # gates_ab_tr.json before 2026-11-16 has no "candidate" key: it was this arm


class Refuse(Exception):
    pass


def fmt(x):
    """Dose as the arm directories name it: a860_sA, a1.5_sA."""
    return str(int(x)) if x == int(x) else repr(x)


def doses(r):
    lo = float(f"{max(1.0, 1000.0 / r):.2g}")
    return fmt(lo), fmt(round(10 * lo, 10))


def read(res, cand, log=None):
    p = os.path.join(res, "gates_ab_tr.json")
    if not os.path.isfile(p):
        raise Refuse(f"TRD_TRAB_MISSING {p}")
    with open(p) as fh:
        j = json.load(fh)
    if j.get("candidate", OLD_CANDIDATE) != cand:
        raise Refuse(f"TRD_TRAB_CANDIDATE {j.get('candidate', OLD_CANDIDATE)} != {cand}")
    if (j.get("verdict") or {}).get("valid") is not True:
        raise Refuse(f"TRD_TRAB_INVALID {j.get('verdict')}")
    r = (j.get("gate_D_input") or {}).get("size_ratio")
    if not isinstance(r, (int, float)) or isinstance(r, bool) or not math.isfinite(r) or r <= 0:
        raise Refuse(f"TRD_TRAB_RATIO {r!r}")
    pat = os.path.join(res, f"bias_pattern_{cand}.json")
    if not os.path.isfile(pat) or os.path.getsize(pat) == 0:
        raise Refuse(f"TRD_TRAB_PATTERN {pat}")
    if log is not None:
        text = open(log).read() if os.path.isfile(log) else ""
        if "TRAB_JOB_DONE failures=0" not in text:
            raise Refuse(f"TRD_TRAB_NOT_CLEAN {log}")
    lo, hi = doses(float(r))
    return lo, hi, os.path.abspath(pat), float(r)


def refused(code, *args):
    try:
        read(*args)
    except Refuse as e:
        return str(e).startswith(code)
    return False


def self_test():
    assert doses(1.1642652492635268) == ("860", "8600")        # job 7841299's R -> job 7843532's doses
    assert doses(3000.0) == ("1", "10") and doses(667.0) == ("1.5", "15") and doses(0.0123) == ("81000", "810000")
    assert doses(1000 / 1.4) == ("1.4", "14")
    with tempfile.TemporaryDirectory() as t:
        def write(**kw):
            j = {"candidate": "x", "verdict": {"valid": True}, "gate_D_input": {"size_ratio": 1.1642652492635268}}
            j.update(kw)
            json.dump(j, open(os.path.join(t, "gates_ab_tr.json"), "w"))
        log = os.path.join(t, "ab.OU")
        write()
        open(os.path.join(t, "bias_pattern_x.json"), "w").write("{}")
        open(log, "w").write("...\nTRAB_JOB_DONE failures=0\n")
        lo, hi, pat, _ = read(t, "x", log)
        assert (lo, hi) == ("860", "8600") and pat.endswith("bias_pattern_x.json")
        assert refused("TRD_TRAB_CANDIDATE", t, "y")
        open(log, "w").write("TRAB_JOB_DONE failures=1\n")
        assert refused("TRD_TRAB_NOT_CLEAN", t, "x", log)
        assert refused("TRD_TRAB_NOT_CLEAN", t, "x", os.path.join(t, "no_such.OU"))
        for k, v, code in (("verdict", {"valid": False}, "TRD_TRAB_INVALID"),
                           ("gate_D_input", {"size_ratio": "nan"}, "TRD_TRAB_RATIO"),
                           ("gate_D_input", {"size_ratio": 0.0}, "TRD_TRAB_RATIO"),
                           ("gate_D_input", None, "TRD_TRAB_RATIO")):
            write(**{k: v})
            assert refused(code, t, "x"), (k, v)
        write()
        os.remove(os.path.join(t, "bias_pattern_x.json"))
        assert refused("TRD_TRAB_PATTERN", t, "x")
        os.remove(os.path.join(t, "gates_ab_tr.json"))
        assert refused("TRD_TRAB_MISSING", t, "x")
    print("TRD_DOSES_TEST_OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res")
    ap.add_argument("--candidate")
    ap.add_argument("--log")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    if not (a.res and a.candidate):
        print("ERROR TRD_ARGS --res and --candidate are required")
        return 2
    try:
        lo, hi, pat, r = read(a.res, a.candidate, a.log)
    except Refuse as e:
        print(f"ERROR {e}")
        return 2
    print(f"TRD_DOSES {lo} {hi} {pat} {r!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
