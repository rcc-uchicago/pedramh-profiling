"""table_o12_solvers.py -- one markdown table of every o1.2 solver arm: the solver step on one A100, o1.2's measured
seconds per optimizer step (o12_polaris.py runs), and o12_extrapolate.py's full-schedule A100 wall-clock. The
tilerows_inrow_if25 row is a PROJECTION (inrow/project_o12_inrow.py), not a measurement. Stdlib only.

  python3 table_o12_solvers.py
"""
import json
import os

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
DDP, STEPS_IF, FRAMES_PER_OPT = 0.005, 25, 12
# (arm, o1.2 job with its measured pieces, solver steps per frame, the probe's ms/solver step key)
ROWS = [("ref", "polaris_o12_7835278", 40, ("seed1", "ref")),
        ("lean_u1", "polaris_o12_7835278", 40, ("inrow", "lean_u1")),
        ("lean32ek_fast", "polaris_o12_7835278", 40, ("inrow", "lean32ek_fast")),
        ("if32_25", "polaris_o12tr_7840909", 25, ("inrow", "if32")),
        ("tilerows_if25", "polaris_o12tr_7840909", 25, ("inrow", "tilerows32_if_eager"))]


def load(*p):
    return json.load(open(os.path.join(RES, *p)))


def proj(pr, key):
    return [v for k, v in pr.items() if key in k][0]


def main():
    seed1 = load("polaris_opt_round1_7814300", "seed1.json")["bench"]["ms_per_step"]
    bench = load("polaris_inrow_7853317", "inrow_probe_prod.json")["bench_ms_per_step"]
    ms = {"seed1": seed1, "inrow": bench}
    out = []
    for arm, job, spf, (src, key) in ROWS:
        v = load(job, "o12_extrapolation.json")["solvers"][arm]
        p14, p16, p4 = proj(v["projections"], "14 GPUs"), proj(v["projections"], "16 GPUs"), proj(v["projections"],
                                                                                                 "1 node")
        out.append({"arm": arm, "spf": spf, "ms": ms[src][key], "s_step": v["pieces"]["s_step"], "h14": p14["hours"],
                    "nh14": p14["node_hours"], "h16": p16["hours"], "h4": p4["hours"], "src": job.split("_")[-1]})
    # tilerows_inrow_if25: tilerows_if25's pieces with the IF arms' measured saving (project_o12_inrow.py)
    X = load("polaris_o12tr_7840909", "o12_extrapolation.json")["solvers"]["tilerows_if25"]
    d = (bench["tilerows32_if_eager"] - bench["inrow32_if_eager"]) * 1e-3
    s2, f2 = X["pieces"]["s_step"] - FRAMES_PER_OPT * STEPS_IF * d, X["pieces"]["s_frame"] - STEPS_IF * d
    hp = {}
    for key, nodes in (("14 GPUs", 4), ("16 GPUs", 4), ("1 node", 1)):
        pr = proj(X["projections"], key)
        t = pr["terms_s"]
        new = dict(t, fill=t["fill"] / X["pieces"]["s_frame"] * f2, steps=pr["steps"] * (s2 + (DDP if nodes > 1 else 0)))
        hp[key] = sum(new.values()) / 3600
    out.append({"arm": "tilerows_inrow_if25 (projected)", "spf": 25, "ms": bench["inrow32_if_eager"], "s_step": s2,
                "h14": hp["14 GPUs"], "nh14": hp["14 GPUs"] * 4, "h16": hp["16 GPUs"], "h4": hp["1 node"],
                "src": "7853317 + 7840909"})
    ref = out[0]
    print("| solver | steps / frame | ms / solver step (1 A100) | o1.2 s / opt step | × ref | 14 A100s (h) | node-h | "
          "16 A100s (h) | 4 A100s, 1 node (h) | source jobs |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in out:
        print(f"| `{r['arm']}` | {r['spf']} | {r['ms']:.3f} | {r['s_step']:.3f} | {ref['s_step'] / r['s_step']:.2f}× | "
              f"{r['h14']:.1f} | {r['nh14']:.0f} | {r['h16']:.1f} | {r['h4']:.1f} | {r['src']} |")
    print("TABLE_O12_SOLVERS_OK")


if __name__ == "__main__":
    main()
