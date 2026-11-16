"""score_mixprec.py -- PREDICTIONS_mixprec.md scored mechanically (standard library; python >= 3.6).

  python3 score_mixprec.py [--p1 <polaris_mpab_<job>>] [--p2 <polaris_o12tr_<job>>]
  python3 score_mixprec.py --self-test
A prediction without its data is UNSCORED. Prints MIXPREC_SCORE ... last.
"""
import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "o12run"))
import score_o12_tilerows as ST                                      # noqa: E402  (arm_facts: o1.2 log facts)

B = "tilerows_inrow_if25"
REF = f"{B}@none"
STEP_MS = 1.470
P2_ARMS = (B, f"{B}_tf32", f"{B}_bf16")


def jload(p):
    return json.load(open(p)) if p and os.path.exists(p) else None


def text(p):
    return open(p).read() if p and os.path.exists(p) else ""


def score_p1(d):
    P = dict.fromkeys(("V1", "V2", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"))
    s = jload(os.path.join(d, "mixprec_summary.json")) if d else None
    if not s:
        return P
    ref = jload(os.path.join(d, REF, "gates_ab_tr.json")) or {}
    v = ref.get("validity", {})
    P["V1"] = bool("MIXPREC_TEST_OK torch=ok" in text(os.path.join(d, "test_mixprec.log"))
                   and v.get("V_frames_lean_u1_bitwise") and v.get("n_bitwise_checks") == 64
                   and v.get("V_sealed_reproduced") and v.get("V_floor_protocol_match"))
    P["V2"] = bool(s["V_wrap"]["ok"])
    cls = {a.partition("@")[2]: r.get("class") for a, r in s["arms"].items()}

    def c(pol, want, neg=False):
        if cls.get(pol) in (None, "MISSING", "INVALID"):
            return None
        return (cls[pol] != want) if neg else (cls[pol] == want)
    P["P1"] = c("bf16_all", "FAIL_AB")
    P["P2"] = c("fp16_all", "PASS", neg=True)
    P["P3"] = c("fp16_kx128", "PASS", neg=True)
    P["P4"] = c("ad1", "PASS")
    P["P5"] = c("ad32", "PASS")
    P["P6"] = c("ad1024", "PASS", neg=True)
    ad1 = s["arms"].get(f"{B}@ad1") or {}
    P["P7"] = None if "f16_mean" not in ad1 else ad1["f16_mean"] <= 0.25
    P["P8"] = s.get("best_pass_prize_ms", 0.0) <= 0.05 * STEP_MS
    return P


def score_p2(d):
    P = dict.fromkeys(("M1", "M2", "M3", "M4", "M5", "M6"))
    if not d:
        return P
    ex = jload(os.path.join(d, "o12_extrapolation.json"))
    F = {a: ST.arm_facts(d, a, ex) for a in P2_ARMS}
    have = [a for a in P2_ARMS if os.path.exists(os.path.join(d, f"{a}.log"))]
    if not have:
        return P
    P["M1"] = all(F[a]["run_done"] and F[a]["finite"] and F[a]["gate"] for a in P2_ARMS)
    t, b = text(os.path.join(d, f"{B}_tf32.log")), text(os.path.join(d, f"{B}_bf16.log"))
    P["M2"] = None if not (t and b) else bool("O12_PREC_CHECK tf32 strict_fp32_check=failed" in t
                                               and "O12_PREC_FORWARD bf16 output_dtype=torch.bfloat16" in b)
    s = {a: F[a]["s_step"] for a in P2_ARMS}
    P["M3"] = None if s[B] is None else abs(s[B] / 0.8885 - 1) <= 0.02
    P["M4"] = None if None in (s[B], s[f"{B}_tf32"]) else s[f"{B}_tf32"] <= 0.95 * s[B]
    P["M5"] = None if None in (s[B], s[f"{B}_bf16"]) else s[f"{B}_bf16"] <= 0.90 * s[B]
    P["M6"] = None if None in (s[f"{B}_tf32"], s[f"{B}_bf16"]) else s[f"{B}_bf16"] < s[f"{B}_tf32"]
    return P


def verdicts(P):
    return {k: ("UNSCORED" if v is None else ("CONFIRMED" if v else "REFUTED")) for k, v in P.items()}


def self_test():
    with tempfile.TemporaryDirectory() as t:
        os.makedirs(os.path.join(t, REF))
        json.dump({"validity": {"V_frames_lean_u1_bitwise": True, "n_bitwise_checks": 64, "V_sealed_reproduced": True,
                                "V_floor_protocol_match": True}}, open(os.path.join(t, REF, "gates_ab_tr.json"), "w"))
        open(os.path.join(t, "test_mixprec.log"), "w").write("MIXPREC_TEST_OK torch=ok (2.3.1)\n")
        arms = {"none": "PASS", "bf16_all": "FAIL_AB", "fp16_all": "BIAS_UP", "fp16_kx128": "FAIL_AB", "ad1": "PASS",
                "ad32": "PASS", "ad1024": "BIAS_UP"}
        summ = {"V_wrap": {"ok": True}, "best_pass_prize_ms": 0.05,
                "arms": {f"{B}@{p}": {"class": c, "f16_mean": 0.1} for p, c in arms.items()}}
        json.dump(summ, open(os.path.join(t, "mixprec_summary.json"), "w"))
        v = verdicts(score_p1(t))
        assert all(x == "CONFIRMED" for x in v.values()), v
        summ["arms"][f"{B}@ad1"]["class"] = "BIAS_UP"
        summ["best_pass_prize_ms"] = 0.2
        json.dump(summ, open(os.path.join(t, "mixprec_summary.json"), "w"))
        v = verdicts(score_p1(t))
        assert v["P4"] == "REFUTED" and v["P8"] == "REFUTED" and v["P1"] == "CONFIRMED", v
        for a, s_step, extra in ((B, 0.889, ""), (f"{B}_tf32", 0.80, "O12_PREC_CHECK tf32 strict_fp32_check=failed (x)\n"),
                                 (f"{B}_bf16", 0.70, "O12_PREC_FORWARD bf16 output_dtype=torch.bfloat16\n")):
            open(os.path.join(t, f"{a}.log"), "w").write(
                f"O12_TR_PATH x jax_to_torch=dlpack\nO12_TR_PATH x torch_to_jax=dlpack\nO12_TR_GATE_VERDICT x pass=True\n"
                f"{extra}epoch 0: train 0.08 val 0.06\nO12_RUN_DONE solver={B}\n")
        json.dump({"solvers": {a: {"pieces": {"s_step": s}, "parsed": {"peak": 21.0},
                                   "projections": {ST.CFG14: {"hours": 48.0}}}
                               for a, s in ((B, 0.889), (f"{B}_tf32", 0.80), (f"{B}_bf16", 0.70))}},
                  open(os.path.join(t, "o12_extrapolation.json"), "w"))
        v = verdicts(score_p2(t))
        assert all(x == "CONFIRMED" for x in v.values()), v
    assert all(x == "UNSCORED" for x in verdicts({**score_p1(None), **score_p2(None)}).values())
    print("MIXPREC_SCORE_TEST_OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1")
    ap.add_argument("--p2")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    v = verdicts({**score_p1(a.p1), **score_p2(a.p2)})
    for k, x in v.items():
        print(f"MIXPREC_PRED {k} {x}")
    c = list(v.values())
    print(f"MIXPREC_SCORE confirmed={c.count('CONFIRMED')} refuted={c.count('REFUTED')} unscored={c.count('UNSCORED')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
