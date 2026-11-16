"""test_analyze_timestep.py -- analyze_timestep.py on synthetic worker JSONs (standard library). Prints
TS_ANALYSIS_TEST_OK.

Case 1: prod first order, IF third order at 2.5% of prod40, CFL 0.42: n = 20 fails CFL (0.84), so n_adm = 25.
Case 2: one band of if28 worse than prod40 on one worker: n = 28 not ACC_OK (n_adm still 25: the rule takes the
smallest admissible n, each n checked on its own).
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_timestep as AT                                        # noqa: E402

NB = 8


def ladder(w, bump28=False):
    p40 = [1e-5 * (b + 1) for b in range(NB)]
    errs = {}
    for n in (80, 40, 32, 28, 25, 20):
        errs[f"prod{n}"] = {k: [v * 40 / n for v in p40] for k in ("sub", "af")}
        errs[f"if{n}"] = {k: [0.025 * v * (40 / n) ** 3 for v in p40] for k in ("sub", "af")}
    if bump28:
        errs["if28"]["af"][3] = 2 * p40[3]
    errs["lean_u1"] = {k: [1e-13] * NB for k in ("sub", "af")}
    errs["if40_vs_prod40"] = {k: list(p40) for k in ("sub", "af")}
    runs = {name: {"seconds": 1.0, "finite": True} for name in list(errs) + ["ref"] if name != "if40_vs_prod40"}
    return {"worker": w, "mode": "ladder", "state": {"cfl": 0.40}, "control": {"prod40_vs_lean_u1_rel": 1e-14},
            "cfl_ref": {"max": 0.42}, "runs": runs, "errors": errs}


def aux():
    return {"worker": 3, "mode": "aux", "cfl_states": [{"member": 1, "t": 100.0, "cfl": 0.37}],
            "ms_per_step": {"prod_rt": 6.4, "if_rt": 6.41, "lean32ek_fast": 2.70, "if32": 2.71},
            "f32_if_accuracy": {"if32_vs_if64": {k: [1e-5] * NB for k in ("sub", "af")},
                                "fast_vs_u1": {k: [1e-5] * NB for k in ("sub", "af")}}}


def run(workers):
    d = tempfile.mkdtemp()
    try:
        for x in workers:
            with open(os.path.join(d, f"worker_{x['worker']}.json"), "w") as f:
                json.dump(x, f)
        old = sys.argv
        sys.argv = ["analyze_timestep.py", "--in", d, "--out", d]
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                AT.main()
        finally:
            sys.argv = old
        with open(os.path.join(d, "timestep_summary.json")) as f:
            return json.load(f), buf.getvalue()
    finally:
        shutil.rmtree(d)


def check(name, ok, detail=""):
    if not ok:
        raise SystemExit(f"ERROR TS_ANALYSIS_TEST {name} {detail}")
    print(f"ok {name}")


def main():
    s, text = run([ladder(0), ladder(1), ladder(2), aux()])
    check("valid", s["valid"] is True, s["valid_why"])
    check("cfl_max", abs(s["cfl_max_at_DT"] - 0.42) < 1e-12)
    check("n20_cfl_fails", s["admissibility"]["20"]["cfl_ok"] is False and s["admissibility"]["20"]["acc_ok"] is True)
    check("n_adm_25", s["n_adm"] == 25, s["n_adm"])
    check("tokens", "TS_VERDICT n_adm=25 valid=True" in text and "TS_ANALYSIS_DONE" in text)
    check("projection", abs(s["projection"]["lean32ek_fast"]["25"]["speedup"] - 1.695 / 1.2094) < 2e-3)
    s2, _ = run([ladder(0), ladder(1, bump28=True), ladder(2), aux()])
    check("bump28_flagged", s2["admissibility"]["28"]["acc_ok"] is False and "w1 af bands [3]" in
          s2["admissibility"]["28"]["notes"][0], s2["admissibility"]["28"])
    bad = ladder(0)
    bad["control"]["prod40_vs_lean_u1_rel"] = 1e-9
    s3, _ = run([bad, ladder(1), ladder(2), aux()])
    check("invalid_control", s3["valid"] is False)
    print("TS_ANALYSIS_TEST_OK")


if __name__ == "__main__":
    main()
