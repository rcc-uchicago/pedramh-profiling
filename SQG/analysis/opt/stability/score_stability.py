"""score_stability.py -- PREDICTIONS_stability.md (revised table) scored mechanically from a results dir
(stability_summary.json + run_w*.json). Standard library only (python >= 3.6).

  python3 score_stability.py --res SQG/analysis/results/polaris_stability_<job>      # prints STAB_SCORE ...
  python3 score_stability.py --self-test                                             # prints STAB_SCORE_TEST_OK
A prediction without its data is UNSCORED (never counted as confirmed).
"""
import argparse
import json
import math
import os
import sys
import tempfile

WORKERS = ("0", "1", "2", "3")


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def score(res):
    sp = os.path.join(res, "stability_summary.json")
    S = json.load(open(sp)) if os.path.exists(sp) else {}
    runs = S.get("runs", {})
    R = {}
    for w in WORKERS:
        p = os.path.join(res, f"run_w{w}.json")
        R[w] = json.load(open(p)) if os.path.exists(p) else None

    def flag(v):
        return None if v is None else bool(v)

    P = {}
    gates = [((R[w] or {}).get("gate_h") or {}).get("pass") for w in WORKERS]
    P["0"] = None if any(g is None for g in gates) else all(gates)
    P["1"] = None if "stability" not in S else S["stability"] == "STABLE"
    P["1b"] = None if "prod_policy" not in S else S["prod_policy"] == "OK"
    cmax = [num((runs.get(w) or {}).get("cfl_max")) for w in WORKERS]
    P["2"] = None if any(c is None for c in cmax) else 0.55 <= max(cmax) <= 0.70
    n60 = [(runs.get(w) or {}).get("n_cfl_ge_060") for w in WORKERS]
    P["3"] = (True if any((n or 0) > 0 for n in n60) else (None if any(n is None for n in n60) else False))
    for k, key in (("4", "S1"), ("5", "S2"), ("6", "S3")):
        P[k] = flag(S.get(key))
    ms = [num((runs.get(w) or {}).get("ms_per_step_wall")) for w in WORKERS]
    complete = [(runs.get(w) or {}).get("class") == "COMPLETE" for w in WORKERS]
    P["7"] = None if not all(complete) or any(m is None for m in ms) else all(m <= 1.75 for m in ms)
    wall = [num((R[w] or {}).get("wall_run_s")) for w in WORKERS]
    P["8"] = None if not all(complete) or any(x is None for x in wall) else all(x <= 3.5 * 3600 for x in wall)
    verdict = {k: ("UNSCORED" if v is None else ("CONFIRMED" if v else "REFUTED")) for k, v in P.items()}
    facts = {"stability": S.get("stability"), "prod_policy": S.get("prod_policy"), "stats": S.get("stats"),
             "cfl_max": cmax, "n_cfl_ge_060": n60, "ms_per_step_wall": ms, "wall_run_s": wall, "gate_h": gates,
             "run_status": S.get("run_status")}
    return {"predictions": verdict, "facts": facts}


def self_test():
    with tempfile.TemporaryDirectory() as d:
        runs = {w: {"class": "COMPLETE", "cfl_max": 0.65, "n_cfl_ge_060": 3 if w == "2" else 0,
                    "ms_per_step_wall": 1.62} for w in WORKERS}
        S = {"stability": "STABLE", "prod_policy": "OK", "S1": True, "S2": True, "S3": True, "stats": "PASS",
             "runs": runs}
        json.dump(S, open(os.path.join(d, "stability_summary.json"), "w"))
        for w in WORKERS:
            json.dump({"gate_h": {"pass": True}, "wall_run_s": 10000.0}, open(os.path.join(d, f"run_w{w}.json"), "w"))
        r = score(d)
        assert all(v == "CONFIRMED" for v in r["predictions"].values()), r["predictions"]
        S["runs"]["1"] = {"class": "UNSTABLE", "cfl_max": 1.02, "n_cfl_ge_060": 40, "ms_per_step_wall": 1.6}
        S["stability"], S["prod_policy"], S["S3"], S["stats"] = "UNSTABLE", "EXCEEDED", None, "INCONCLUSIVE"
        json.dump(S, open(os.path.join(d, "stability_summary.json"), "w"))
        os.remove(os.path.join(d, "run_w3.json"))
        pv = score(d)["predictions"]
        assert pv["1"] == "REFUTED" and pv["1b"] == "REFUTED" and pv["2"] == "REFUTED", pv
        assert pv["0"] == "UNSCORED" and pv["6"] == "UNSCORED" and pv["7"] == "UNSCORED" and pv["3"] == "CONFIRMED", pv
    print("STAB_SCORE_TEST_OK")


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
        print(f"STAB_PRED {k} {v}")
    print(f"STAB_FACTS {json.dumps(r['facts'])}")
    c = list(r["predictions"].values())
    print(f"STAB_SCORE confirmed={c.count('CONFIRMED')} refuted={c.count('REFUTED')} unscored={c.count('UNSCORED')}")
    with open(os.path.join(a.res, "score_stability.json"), "w") as fh:
        json.dump(r, fh, indent=1)


if __name__ == "__main__":
    sys.exit(main())
