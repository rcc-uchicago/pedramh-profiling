"""score_gates_inrow.py -- PREDICTIONS_gates_inrow.md scored mechanically (standard library; python >= 3.6).

  python3 score_gates_inrow.py [--ab <polaris_trab_<job>>] [--e <polaris_tre_<job>>] [--d <gate D dose_summary.json>]
  python3 score_gates_inrow.py --self-test
A6 / A7 / A9 compare with job 7841299 (--ab-old). A prediction without its data is UNSCORED (never counted as
confirmed). Prints GATES_IR_SCORE ... last.
"""
import argparse
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doses_from_trab as DT                                         # noqa: E402  (the dose rule)

CAND, TWIN = "tilerows_inrow_if25", "tilerows_if25"
AB_OLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "polaris_trab_7841299")


def jload(p):
    return json.load(open(p)) if p and os.path.exists(p) else None


def detection_only(bands):
    """The non-PASS bands, if each fails B3 alone (B1, B2 and power hold); else None."""
    out = []
    for name, b in bands.items():
        if b.get("verdict") in ("PASS", "NO_DATA"):
            continue
        if not (b["B1"] and b["B2"] and b["power"] and b["B3"] is False):
            return None
        out.append(name)
    return sorted(out)


def verdicts(gb):
    return {k: b.get("verdict") for k, b in gb["bands"].items()}


def score(ab_dir=None, e_dir=None, d_path=None, ab_old=AB_OLD):
    P = {}
    ab = jload(os.path.join(ab_dir, "gates_ab_tr.json")) if ab_dir else None
    old = jload(os.path.join(ab_old, "gates_ab_tr.json")) if ab_old else None
    if ab:
        v = ab.get("validity", {})
        P["A1"] = bool(v.get("V_frames_lean_u1_bitwise") and v.get("n_bitwise_checks") == 64
                       and v.get("V_sealed_reproduced") and v.get("V_floor_protocol_match"))
        ga, gb = ab.get("gate_A", {}).get(CAND), ab.get("gate_B", {}).get(CAND)
        ga2, gb2 = ab.get("gate_A", {}).get(TWIN), ab.get("gate_B", {}).get(TWIN)
        P["A2"] = None if not ga else bool(ga["pass"] and max(ga["var_over_floor"]) <= 1e-5)
        if gb:
            bands = [b for b in gb["bands"].values() if b.get("verdict") != "NO_DATA"]
            P["A3"] = all(b["B1"] and b["B2"] for b in bands)
            det = detection_only(gb["bands"])
            P["A4"] = None if not gb2 else bool(gb["overall"] == "NOT_PASS" and det
                                                and det == detection_only(gb2["bands"]))
        else:
            P["A3"] = P["A4"] = None
        P["A5"] = None if not (ga and ga2 and gb and gb2) else bool(
            ga["pass"] == ga2["pass"] and verdicts(gb) == verdicts(gb2))
        r = (ab.get("gate_D_input") or {}).get("size_ratio")
        r_ok = isinstance(r, (int, float)) and math.isfinite(r) and r > 0
        r_old = ((old or {}).get("gate_D_input") or {}).get("size_ratio")
        P["A6"] = None if not (r_ok and r_old) else abs(r / r_old - 1) <= 0.02
        P["A7"] = None if not r_ok else DT.doses(r) == ("860", "8600")
        acc = ab.get("accuracy_vs_ref160", {})
        P["A8"] = None if not (CAND in acc and TWIN in acc) else all(
            abs(acc[CAND][k] / acc[TWIN][k] - 1) <= 0.10 for k in acc[CAND])
        oa, ob = (old or {}).get("gate_A", {}).get(TWIN), (old or {}).get("gate_B", {}).get(TWIN)
        P["A9"] = None if not (ga2 and gb2 and oa and ob) else bool(
            ga2["pass"] == oa["pass"] and verdicts(gb2) == verdicts(ob))
    else:
        for k in ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"):
            P[k] = None
    es = jload(os.path.join(e_dir, f"candidate_{CAND}", "energy_summary.json")) if e_dir else None
    et = jload(os.path.join(e_dir, f"candidate_{TWIN}", "energy_summary.json")) if e_dir else None
    if es:
        v = es.get("validity", {})
        P["E1"] = all(v.get(k) is True for k in ("V1_bitwise", "V2_diag", "V5_run", "V3_null", "V4_calibration"))
        P["E2"] = es.get("verdict") == "PASS_E"
        c, t = es.get("candidate", {}), (et or {}).get("candidate", {})
        if "S" in c and "V" in c and "S" in t and "V" in t:
            rs = c["S"]["max_abs_s"] / t["S"]["max_abs_s"]
            P["E3"] = bool(0.8 <= rs <= 1.25 and c["V"]["max_abs_s"] <= 2 * t["V"]["max_abs_s"])
        else:
            P["E3"] = None
        P["E4"] = None if not c else any(c[i]["detected"] for i in c)
        ms, ms2 = es.get("timing", {}).get(CAND), (et or {}).get("timing", {}).get(TWIN)
        P["E5"] = None if ms is None or ms2 is None else ms < ms2
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
    return {k: ("UNSCORED" if v is None else ("CONFIRMED" if v else "REFUTED")) for k, v in P.items()}


def self_test():
    with tempfile.TemporaryDirectory() as t:
        ab, ab0, e = os.path.join(t, "ab"), os.path.join(t, "ab0"), os.path.join(t, "e")
        for d in (ab, ab0, os.path.join(e, f"candidate_{CAND}"), os.path.join(e, f"candidate_{TWIN}")):
            os.makedirs(d)
        ok = {"verdict": "PASS", "B1": True, "B2": True, "B3": True, "power": True}
        det = {"verdict": "FAIL_FLOOR", "B1": True, "B2": True, "B3": False, "power": True}
        bands = {"1-2": ok, "2-4": det}
        ga = {"pass": True, "var_over_floor": [1e-7] * 8}
        acc = {k: 1e-5 for k in ("1-2", "2-4")}
        new = {"validity": {"V_frames_lean_u1_bitwise": True, "n_bitwise_checks": 64, "V_sealed_reproduced": True,
                            "V_floor_protocol_match": True},
               "gate_A": {CAND: ga, TWIN: ga},
               "gate_B": {CAND: {"overall": "NOT_PASS", "bands": bands}, TWIN: {"overall": "NOT_PASS", "bands": bands}},
               "accuracy_vs_ref160": {"lean_u1": acc, CAND: acc, TWIN: dict(acc, **{"2-4": 1.05e-5})},
               "gate_D_input": {"size_ratio": 1.165}}                  # 1000 / R = 858 -> 860; A7 needs R in (1.156, 1.170]
        json.dump(new, open(os.path.join(ab, "gates_ab_tr.json"), "w"))
        json.dump({"gate_A": {TWIN: ga}, "gate_B": {TWIN: {"overall": "NOT_PASS", "bands": bands}},
                   "gate_D_input": {"size_ratio": 1.1643}}, open(os.path.join(ab0, "gates_ab_tr.json"), "w"))
        val = {"V1_bitwise": True, "V2_diag": True, "V5_run": True, "V3_null": True, "V4_calibration": True}
        for arm, s, vv, ms in ((CAND, 2.0e-5, 2.5e-6, 1.66), (TWIN, 1.96e-5, 2.4e-6, 1.78)):
            json.dump({"validity": val, "verdict": "PASS_E",
                       "candidate": {"S": {"max_abs_s": s, "detected": True}, "V": {"max_abs_s": vv,
                                                                                  "detected": True}},
                       "timing": {arm: ms}}, open(os.path.join(e, f"candidate_{arm}", "energy_summary.json"), "w"))
        dp = os.path.join(t, "dose_summary.json")
        json.dump({"verdict": {"verdict": "PASS_D", "power_drift": True},
                   "overfit": {"a0_sA": {"overfit": False}, "a0_sB": {"overfit": False}}}, open(dp, "w"))
        v = score(ab, e, dp, ab_old=ab0)
        assert all(x == "CONFIRMED" for x in v.values()), v
        # a B1 failure is not detection only (same band verdict, so A5 holds); R + 12% moves the doses (770)
        new["gate_B"][CAND]["bands"] = {"1-2": ok, "2-4": dict(det, B1=False)}
        new["gate_D_input"]["size_ratio"] = 1.30
        json.dump(new, open(os.path.join(ab, "gates_ab_tr.json"), "w"))
        v = score(ab, e, dp, ab_old=ab0)
        assert v["A3"] == v["A4"] == v["A6"] == v["A7"] == "REFUTED", v
        assert v["A5"] == v["A9"] == "CONFIRMED", v
        assert all(x == "UNSCORED" for x in score(ab_old=None).values())
    print("GATES_IR_SCORE_TEST_OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab")
    ap.add_argument("--e")
    ap.add_argument("--d")
    ap.add_argument("--ab-old", default=AB_OLD)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    v = score(a.ab, a.e, a.d, a.ab_old)
    for k, x in v.items():
        print(f"GATES_IR_PRED {k} {x}")
    c = list(v.values())
    print(f"GATES_IR_SCORE confirmed={c.count('CONFIRMED')} refuted={c.count('REFUTED')} "
          f"unscored={c.count('UNSCORED')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
