"""score_o12_inrow.py -- PREDICTIONS_o12_inrow.md scored mechanically from a results dir (arm logs + extrapolation
json; score_o12_tilerows.arm_facts reads them). Standard library only (python >= 3.6).

  python3 score_o12_inrow.py --res SQG/analysis/results/polaris_o12tr_<job>      # prints O12IR_SCORE ...
  python3 score_o12_inrow.py --self-test
A prediction without its data is UNSCORED.
"""
import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_o12_tilerows as ST                                       # noqa: E402

BASE, NEW = "tilerows_if25", "tilerows_inrow_if25"
OPTION_C_S_STEP = 0.9261                                             # job 7840909


def score(res):
    exp = os.path.join(res, "o12_extrapolation.json")
    ex = json.load(open(exp)) if os.path.exists(exp) else None
    F = {a: ST.arm_facts(res, a, ex) for a in (BASE, NEW)}
    b, n = F[BASE]["s_step"], F[NEW]["s_step"]
    P = {
        "Q1": all(F[a]["run_done"] and F[a]["finite"] for a in F),
        "Q2": F[NEW]["gate"],
        "Q3": None if b is None else abs(b / OPTION_C_S_STEP - 1) <= 0.02,
        "Q4": None if None in (b, n) else 0.950 <= n / b <= 0.975,
        "Q5": None if F[NEW]["hours14"] is None else 48.0 <= F[NEW]["hours14"] <= 49.8,
        "Q6": None if not F[NEW]["paths"] else all(v == "dlpack" for v in F[NEW]["paths"].values())
        and len(F[NEW]["paths"]) == 2,
    }
    counts = bool(F[NEW]["gate"] and F[NEW]["run_done"] and F[NEW]["finite"] and P["Q6"])
    return P, F, counts


def self_test():
    with tempfile.TemporaryDirectory() as d:
        for a, gate in ((BASE, True), (NEW, True)):
            with open(os.path.join(d, f"{a}.log"), "w") as fh:
                fh.write("O12_TR_PATH x jax_to_torch=dlpack\nO12_TR_PATH x torch_to_jax=dlpack\n"
                         f"O12_TR_GATE_VERDICT x pass={gate}\nepoch 0: train 0.08 val 0.06\nO12_RUN_DONE solver=x\n")
        json.dump({"solvers": {a: {"pieces": {"s_step": s}, "parsed": {"peak": 21.0},
                                   "projections": {ST.CFG14: {"hours": h}}}
                               for a, s, h in ((BASE, 0.926, 50.9), (NEW, 0.889, 48.9))}},
                  open(os.path.join(d, "o12_extrapolation.json"), "w"))
        P, _, counts = score(d)
        assert counts and all(v is True for v in P.values()), P
        P, _, counts = score(tempfile.mkdtemp())
        assert not counts and P["Q2"] is None and P["Q4"] is None, P
    print("O12IR_SCORE_TEST_OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    P, F, counts = score(a.res)
    for k, v in P.items():
        print(f"O12IR_PRED {k} {'UNSCORED' if v is None else ('CONFIRMED' if v else 'REFUTED')}")
    for arm, f in F.items():
        print(f"O12IR_ARM {arm} s_step={f['s_step']} hours14={f['hours14']} gate={f['gate']} run_done={f['run_done']} "
              f"paths={f['paths']}")
    vals = list(P.values())
    print(f"O12IR_SCORE confirmed={vals.count(True)} refuted={vals.count(False)} unscored={vals.count(None)} "
          f"| measurement counts: {counts}")


if __name__ == "__main__":
    main()
