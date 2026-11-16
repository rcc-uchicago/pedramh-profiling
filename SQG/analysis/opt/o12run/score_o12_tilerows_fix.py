"""score_o12_tilerows_fix.py -- PREDICTIONS_o12_tilerows.md's follow-up (F1-F7, the expand fix) scored mechanically
from a results dir (arm logs + o12_extrapolation.json). Standard library only (python >= 3.6).

  python3 score_o12_tilerows_fix.py --res SQG/analysis/results/polaris_o12tr_<job>     # prints O12TRFIX_SCORE ...
  python3 score_o12_tilerows_fix.py --self-test                                         # prints O12TRFIX_TEST_OK
A prediction without its data is UNSCORED (never counted as confirmed).
"""
import argparse
import json
import os
import re
import sys
import tempfile

TR = ("tilerows_if25", "tilerows_if25_scatter_t", "tilerows_if25_concat_t")
BASE_IF32 = 1.2404                                                   # job 7840805


def lines(res, arm):
    p = os.path.join(res, f"{arm}.log")
    return [l.split("\r")[-1].strip() for l in open(p, errors="replace")] if os.path.exists(p) else []


def last_timing(res, arm):
    t = [l for l in lines(res, arm) if l.startswith("O12_TR_TIMING")]
    if not t:
        return None
    return {k: float(v) for k, v in re.findall(r"(\w+_ms)=([\d.naN]+)", t[-1])}


def score(res):
    exp = os.path.join(res, "o12_extrapolation.json")
    ex = json.load(open(exp)) if os.path.exists(exp) else {}
    s = {a: (ex.get("solvers", {}).get(a) or {}).get("pieces", {}).get("s_step") for a in ("if32_25",) + TR}
    checks = [l.endswith("concat_equals_scatter=True") for a in TR for l in lines(res, a)
              if l.startswith("O12_TR_EXPAND_CHECK")]
    gates = [any(l.startswith("O12_TR_GATE_VERDICT") and l.endswith("pass=True") for l in lines(res, a))
             if lines(res, a) else None for a in TR]
    ts, tc = last_timing(res, "tilerows_if25_scatter_t"), last_timing(res, "tilerows_if25_concat_t")
    P = {
        "F1": (all(checks) if checks else None),
        "F2": None if not ts else ts["handoff_expand_ms"] >= 15.0,
        "F3": None if not tc else tc["handoff_expand_ms"] <= 3.0,
        "F4": None if s["tilerows_if25"] is None else s["tilerows_if25"] <= 1.00,
        "F5": None if None in (s["tilerows_if25"], s["if32_25"]) else s["tilerows_if25"] <= 0.80 * s["if32_25"],
        "F6": None if s["if32_25"] is None else abs(s["if32_25"] / BASE_IF32 - 1) <= 0.01,
        "F7": None if any(g is None for g in gates) else all(gates),
    }
    verdict = {k: ("UNSCORED" if v is None else ("CONFIRMED" if v else "REFUTED")) for k, v in P.items()}
    return {"predictions": verdict, "s_step": s, "timing_scatter": ts, "timing_concat": tc,
            "expand_checks": checks, "gates": dict(zip(TR, gates))}


def self_test():
    with tempfile.TemporaryDirectory() as d:
        def w(arm, body):
            open(os.path.join(d, f"{arm}.log"), "w").write(body)
        for a in TR:
            w(a, f"O12_TR_EXPAND_CHECK {a} concat_equals_scatter=True\nO12_TR_GATE_VERDICT {a} pass=True\n")
        w("tilerows_if25_scatter_t", "O12_TR_GATE_VERDICT x pass=True\nO12_TR_TIMING x expand=scatter frames=600 "
          "entry_sync_ms=1.00 start_ms=0.00 run_ms=39.50 handoff_expand_ms=24.10 outside_ms=1.20\n")
        w("tilerows_if25_concat_t", "O12_TR_GATE_VERDICT x pass=True\nO12_TR_TIMING x expand=concat frames=600 "
          "entry_sync_ms=1.00 start_ms=0.00 run_ms=39.50 handoff_expand_ms=0.60 outside_ms=1.20\n")
        json.dump({"solvers": {"if32_25": {"pieces": {"s_step": 1.241}},
                               "tilerows_if25": {"pieces": {"s_step": 0.905}}}}, open(os.path.join(d, "o12_extrapolation.json"), "w"))
        r = score(d)
        assert all(v == "CONFIRMED" for v in r["predictions"].values()), r["predictions"]
        json.dump({"solvers": {"if32_25": {"pieces": {"s_step": 1.30}},
                               "tilerows_if25": {"pieces": {"s_step": 1.20}}}}, open(os.path.join(d, "o12_extrapolation.json"), "w"))
        pv = score(d)["predictions"]
        assert pv["F4"] == "REFUTED" and pv["F5"] == "REFUTED" and pv["F6"] == "REFUTED", pv
        os.remove(os.path.join(d, "tilerows_if25_concat_t.log"))
        pv = score(d)["predictions"]
        assert pv["F3"] == "UNSCORED" and pv["F7"] == "UNSCORED", pv
    print("O12TRFIX_TEST_OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    r = score(a.res)
    for k, v in r["predictions"].items():
        print(f"O12TRFIX_PRED {k} {v}")
    print(f"O12TRFIX_FACTS s_step={r['s_step']} timing_scatter={r['timing_scatter']} timing_concat={r['timing_concat']} "
          f"gates={r['gates']} expand_checks={r['expand_checks']}")
    c = list(r["predictions"].values())
    print(f"O12TRFIX_SCORE confirmed={c.count('CONFIRMED')} refuted={c.count('REFUTED')} unscored={c.count('UNSCORED')}")
    with open(os.path.join(a.res, "score_o12_tilerows_fix.json"), "w") as fh:
        json.dump(r, fh, indent=1)


if __name__ == "__main__":
    sys.exit(main())
