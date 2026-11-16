"""score_gates_tilerows.py -- PREDICTIONS_gates_tilerows.md scored mechanically (standard library; python >= 3.6).

  python3 score_gates_tilerows.py [--ab <polaris_trab_<job>>] [--e <polaris_tre_<job>>] [--d <gate D summary json>]
  python3 score_gates_tilerows.py --self-test
A prediction without its data is UNSCORED (never counted as confirmed). Prints GATES_TR_SCORE ... last.
"""
import argparse
import json
import math
import os
import sys
import tempfile

CAND, TWIN = "tilerows_if25", "if32_25"


def jload(p):
    return json.load(open(p)) if p and os.path.exists(p) else None


def score(ab_dir=None, e_dir=None, d_path=None):
    P = {}
    ab = jload(os.path.join(ab_dir, "gates_ab_tr.json")) if ab_dir else None
    if ab:
        v = ab.get("validity", {})
        P["A1"] = bool(v.get("V_frames_lean_u1_bitwise") and v.get("n_bitwise_checks") == 64
                       and v.get("V_sealed_reproduced") and v.get("V_floor_protocol_match"))
        ga = ab.get("gate_A", {}).get(CAND)
        P["A2"] = None if not ga else bool(ga["pass"] and max(ga["var_over_floor"]) <= 1e-5)
        gb = ab.get("gate_B", {}).get(CAND)
        if gb:
            bands = [b for b in gb["bands"].values() if b.get("verdict") != "NO_DATA"]
            P["A3"] = all(b["B1"] and b["B2"] for b in bands)
            P["A4"] = gb["overall"] == "NOT_PASS" and any(b["B3"] is False for b in bands)
        else:
            P["A3"] = P["A4"] = None
        acc = ab.get("accuracy_vs_ref160", {})
        if CAND in acc and "lean_u1" in acc:
            P["A5"] = sum(acc[CAND][k] < acc["lean_u1"][k] for k in acc[CAND]) >= 6
        else:
            P["A5"] = None
        ga2, gb2 = ab.get("gate_A", {}).get(TWIN), ab.get("gate_B", {}).get(TWIN)
        P["A6"] = None if not (ga and ga2 and gb and gb2) else bool(
            ga["pass"] == ga2["pass"] and gb["overall"] == gb2["overall"])
        r = (ab.get("gate_D_input") or {}).get("size_ratio")
        P["A7"] = None if r is None or not math.isfinite(r) else bool(0.5 <= r <= 20)
    else:
        for k in ("A1", "A2", "A3", "A4", "A5", "A6", "A7"):
            P[k] = None
    es = jload(os.path.join(e_dir, f"candidate_{CAND}", "energy_summary.json")) if e_dir else None
    if es:
        v = es.get("validity", {})
        P["E1"] = all(v.get(k) is True for k in ("V1_bitwise", "V2_diag", "V5_run", "V3_null", "V4_calibration"))
        P["E2"] = es.get("verdict") == "PASS_E"
        c = es.get("candidate", {})
        P["E3"] = None if "S" not in c else c["S"]["max_abs_s"] <= 3e-5
        P["E4"] = None if not c else any(c[i]["detected"] for i in c)
        ms = es.get("timing", {}).get(CAND)
        P["E5"] = None if ms is None else ms <= 2.0
    else:
        for k in ("E1", "E2", "E3", "E4", "E5"):
            P[k] = None
    d = jload(d_path) if d_path else None
    if d:                                                            # analyze_dose.py's dose_summary.json (best)
        vd = d.get("verdict", {})
        P["D1"] = vd.get("verdict") == "PASS_D"
        of = d.get("overfit", {})
        P["D2"] = None if not of else all(not x.get("overfit", True) for k, x in of.items() if k.startswith("a0_"))
        P["D3"] = vd.get("power_drift")
    else:
        P["D1"] = P["D2"] = P["D3"] = None
    verdict = {k: ("UNSCORED" if v is None else ("CONFIRMED" if v else "REFUTED")) for k, v in P.items()}
    return verdict


def self_test():
    with tempfile.TemporaryDirectory() as t:
        ab, e = os.path.join(t, "ab"), os.path.join(t, "e", f"candidate_{CAND}")
        os.makedirs(ab)
        os.makedirs(e)
        band = {"verdict": "FAIL_FLOOR", "B1": True, "B2": True, "B3": False}
        json.dump({"validity": {"V_frames_lean_u1_bitwise": True, "n_bitwise_checks": 64, "V_sealed_reproduced": True,
                                "V_floor_protocol_match": True},
                   "gate_A": {CAND: {"pass": True, "var_over_floor": [1e-7] * 8},
                              TWIN: {"pass": True, "var_over_floor": [1e-7] * 8}},
                   "gate_B": {CAND: {"overall": "NOT_PASS", "bands": {"1-2": band}},
                              TWIN: {"overall": "NOT_PASS", "bands": {"1-2": band}}},
                   "accuracy_vs_ref160": {"lean_u1": {str(i): 1.0 for i in range(8)},
                                          CAND: {str(i): 0.5 for i in range(8)}},
                   "gate_D_input": {"size_ratio": 3.0}}, open(os.path.join(ab, "gates_ab_tr.json"), "w"))
        json.dump({"validity": {"V1_bitwise": True, "V2_diag": True, "V5_run": True, "V3_null": True,
                                "V4_calibration": True}, "verdict": "PASS_E",
                   "candidate": {"S": {"max_abs_s": 1.2e-5, "detected": True}, "V": {"max_abs_s": 1e-6,
                                                                                      "detected": False}},
                   "timing": {CAND: 1.8}}, open(os.path.join(e, "energy_summary.json"), "w"))
        v = score(ab, os.path.join(t, "e"))
        assert all(v[k] == "CONFIRMED" for k in v if k[0] in "AE"), v
        assert all(v[k] == "UNSCORED" for k in ("D1", "D2", "D3")), v
        assert all(x == "UNSCORED" for x in score().values())
    print("GATES_TR_SCORE_TEST_OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab")
    ap.add_argument("--e")
    ap.add_argument("--d")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    v = score(a.ab, a.e, a.d)
    for k, x in v.items():
        print(f"GATES_TR_PRED {k} {x}")
    c = list(v.values())
    print(f"GATES_TR_SCORE confirmed={c.count('CONFIRMED')} refuted={c.count('REFUTED')} "
          f"unscored={c.count('UNSCORED')}")


if __name__ == "__main__":
    sys.exit(main())
