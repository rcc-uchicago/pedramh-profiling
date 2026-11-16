"""score_o12_tilerows.py -- PREDICTIONS_o12_tilerows.md scored mechanically from a results dir (the arm logs + the
extrapolation json). Standard library only (python >= 3.6).

  python3 score_o12_tilerows.py --res SQG/analysis/results/polaris_o12tr_<job>      # prints O12TR_SCORE ...
  python3 score_o12_tilerows.py --self-test                                          # prints O12TR_SCORE_TEST_OK
A prediction without its data is UNSCORED (never counted as confirmed).
"""
import argparse
import json
import math
import os
import re
import sys
import tempfile

ARMS = ("lean32ek_fast", "if32_25", "tilerows40", "tilerows_if25", "tilerows_if25_cufft114")
NEW = ARMS[1:]
TR_ARMS = ("tilerows40", "tilerows_if25", "tilerows_if25_cufft114")
CFG14 = "4 nodes, 14 GPUs, batch 16 (o1.2's global batch 224)"
EPOCH = re.compile(r"^epoch (\d+): train ([\d.naife+-]+)\s+val ([\d.naife+-]+)")


def arm_facts(res, arm, ex):
    """What the scorer needs from one arm: gate verdict, DLPack paths, run done, finite losses, s_step, peak, val."""
    f = {"gate": None, "paths": {}, "run_done": False, "losses": [], "s_step": None, "peak": None, "hours14": None}
    p = os.path.join(res, f"{arm}.log")
    if os.path.exists(p):
        for line in open(p, errors="replace"):
            line = line.split("\r")[-1].strip()
            if line.startswith("O12_TR_GATE_VERDICT"):
                f["gate"] = line.endswith("pass=True")
            elif line.startswith("O12_TR_PATH"):
                for k, v in re.findall(r"(jax_to_torch|torch_to_jax)=(\S+)", line):
                    f["paths"][k] = v
            elif line.startswith("O12_RUN_DONE"):
                f["run_done"] = True
            m = EPOCH.match(line)
            if m:
                f["losses"].append((float(m.group(2)), float(m.group(3))))
    s = (ex or {}).get("solvers", {}).get(arm)
    if s:
        f["s_step"] = s["pieces"].get("s_step")
        f["peak"] = s["parsed"].get("peak")
        pr = s["projections"].get(CFG14)
        f["hours14"] = pr["hours"] if pr else None
    f["finite"] = bool(f["losses"]) and all(math.isfinite(a) and math.isfinite(b) for a, b in f["losses"])
    f["val_last"] = f["losses"][-1][1] if f["losses"] else None
    return f


def score(res):
    exp = os.path.join(res, "o12_extrapolation.json")
    ex = json.load(open(exp)) if os.path.exists(exp) else None
    F = {a: arm_facts(res, a, ex) for a in ARMS}
    s = {a: F[a]["s_step"] for a in ARMS}

    def within(x, lo, hi):
        return None if x is None else lo <= x <= hi

    def all_known(vals):
        return None if any(v is None for v in vals) else all(vals)

    P = {}
    P[1] = all_known([F["tilerows40"]["gate"], F["tilerows_if25"]["gate"]])
    P[2] = all(F[a]["run_done"] and F[a]["finite"] for a in ARMS)
    P[3] = within(s["lean32ek_fast"], 1.68, 1.75)
    P[4] = within(s["if32_25"], 1.17, 1.31)
    P[5] = within(s["tilerows40"], 1.10, 1.25)
    P[6] = within(s["tilerows_if25"], 0.84, 0.97)
    P[7] = None if None in (s["tilerows_if25"], s["if32_25"]) else s["tilerows_if25"] <= 0.80 * s["if32_25"]
    P[8] = F["tilerows_if25"]["run_done"]
    P[9] = (None if None in (s["tilerows_if25"], s["tilerows_if25_cufft114"])
            else abs(s["tilerows_if25"] / s["tilerows_if25_cufft114"] - 1) <= 0.03)
    ran = [a for a in TR_ARMS if F[a]["paths"]]
    P[10] = (None if not ran else all(F[a]["paths"].get(k) == "dlpack" for a in ran
                                      for k in ("jax_to_torch", "torch_to_jax")))
    P[11] = within(F["tilerows_if25"]["hours14"], 45.0, 54.0)
    P[12] = all_known([None if F[a]["peak"] is None else F[a]["peak"] <= 21.5 for a in NEW])
    v0 = F["lean32ek_fast"]["val_last"]
    P[13] = (None if v0 is None or any(F[a]["val_last"] is None for a in NEW)
             else all(abs(F[a]["val_last"] / v0 - 1) <= 0.01 for a in NEW))
    verdict = {k: ("UNSCORED" if v is None else ("CONFIRMED" if v else "REFUTED")) for k, v in P.items()}
    return {"predictions": verdict, "facts": F,
            "gates": {"G_TR": {a: F[a]["gate"] for a in TR_ARMS},
                      "G_RUN": {a: bool(F[a]["run_done"] and F[a]["finite"]) for a in ARMS},
                      "WORTH": P[7]}}


def write_log(d, arm, gate=True, s_step=1.0, val=0.058, done=True, path="dlpack"):
    with open(os.path.join(d, f"{arm}.log"), "w") as fh:
        if arm in TR_ARMS:
            fh.write(f"O12_TR_PATH {arm} jax_to_torch={path}\nO12_TR_PATH {arm} torch_to_jax=dlpack\n")
            fh.write(f"O12_TR_GATE_VERDICT {arm} pass={gate}\n")
        for e in range(4):
            fh.write(f"epoch {e}: train 0.0600  val {val:.4f} (persistence 0.9000)  lr 1e-4  {s_step:.3f} s/step  (60 s)\n")
        if done:
            fh.write(f"O12_RUN_DONE solver={arm}\n")


def self_test():
    with tempfile.TemporaryDirectory() as d:
        steps = {"lean32ek_fast": 1.71, "if32_25": 1.24, "tilerows40": 1.18, "tilerows_if25": 0.89,
                 "tilerows_if25_cufft114": 0.90}
        ex = {"solvers": {a: {"pieces": {"s_step": v}, "parsed": {"peak": 21.0},
                              "projections": {CFG14: {"hours": 49.0 if a == "tilerows_if25" else 90.0}}}
                          for a, v in steps.items()}}
        for a, v in steps.items():
            write_log(d, a, s_step=v)
        json.dump(ex, open(os.path.join(d, "o12_extrapolation.json"), "w"))
        r = score(d)
        assert all(v == "CONFIRMED" for v in r["predictions"].values()), r["predictions"]
        write_log(d, "tilerows_if25", gate=False, path="host")
        os.remove(os.path.join(d, "tilerows_if25_cufft114.log"))
        r = score(d)
        pv = r["predictions"]
        assert pv[1] == "REFUTED" and pv[10] == "REFUTED" and pv[2] == "REFUTED", pv
        os.remove(os.path.join(d, "o12_extrapolation.json"))
        r = score(d)
        assert r["predictions"][6] == "UNSCORED" and r["predictions"][11] == "UNSCORED", r["predictions"]
    print("O12TR_SCORE_TEST_OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    r = score(a.res)
    for k, v in sorted(r["predictions"].items()):
        print(f"O12TR_PRED {k} {v}")
    for a_, f in r["facts"].items():
        print(f"O12TR_ARM {a_} s_step={f['s_step']} gate={f['gate']} run_done={f['run_done']} finite={f['finite']} "
              f"peak={f['peak']} val_last={f['val_last']} hours14={f['hours14']} paths={f['paths']}")
    c = list(r["predictions"].values())
    print(f"O12TR_SCORE confirmed={c.count('CONFIRMED')} refuted={c.count('REFUTED')} unscored={c.count('UNSCORED')} "
          f"G_TR={r['gates']['G_TR']} WORTH={r['gates']['WORTH']}")
    with open(os.path.join(a.res, "score_o12_tilerows.json"), "w") as fh:
        json.dump(r, fh, indent=1)


if __name__ == "__main__":
    sys.exit(main())
