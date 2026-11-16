"""test_rank_round1.py -- stdlib self-test of rank_round1.py on synthetic seed JSONs. Prints RANK_TEST_OK.

Checks: (1) the fastest variant overall FAILS its gate and must not win; (2) the winner is the fastest PASSING variant
even though it is not bitwise (no bitwise penalty); (3) a non-bitwise passing variant ranks above a slower bitwise one;
(4) the float64-only best is reported; (5) a missing seed2 file still yields a ranking from seed1 alone;
(6) no input at all is an error. Mirrors the JSON shapes written by run_seed1.py and seed2_opt.py.
"""
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rank_round1 as R  # noqa: E402


def seed1_json():
    def gate(pass_, err, bits):
        return {"pass": pass_, "reason": None if pass_ else f"1step_rel_l2={err:.3e}>1e-12_in_T0_synthetic",
                "1step": {"random": {"max_rel_l2": err / 10}, "synthetic": {"max_rel_l2": err}},
                "40step": {"random": {"max_rel_l2": err}, "synthetic": {"max_rel_l2": err}},
                "480step": {"random": {"bits_equal": bits}, "synthetic": {"bits_equal": bits}}}
    return {"seed": 1, "thresholds": {"G1_1step_rel_l2_per_field": 1e-12},
            "variants": {"lean": {}, "lean_u1": {"unroll1": True}, "lean_split": {"split": True}},
            "gate": {"lean": gate(True, 0.0, True), "lean_u1": gate(True, 3e-16, False), "lean_split": gate(False, 2e-9, False)},
            "bench": {"ms_per_step": {"ref": 9.30, "lean": 7.40, "lean_u1": 6.00, "lean_split": 5.00},
                      "speedup": {"lean": 1.2568, "lean_u1": 1.55, "lean_split": 1.86}},
            "optstep": {"variant": "lean_u1", "ref": {"s_median": 4.47}, "lean_u1": {"s_median": 2.90}, "speedup": 1.54},
            "long": {"variants": {"lean": {"coarse_rel_l2": 0.0, "energy_rel_diff": 0.0},
                                  "lean_u1": {"coarse_rel_l2": 2e-7, "energy_rel_diff": 1e-9},
                                  "lean_split": {"skipped": "budget"}}}}


def seed2_json():
    def var(pass_, thr, err, bits, ms, opt=None, reasons=()):
        lg = {ic: {"480_steps": {"fields_bitwise": bits}, "bitwise_480": bits, "coarse_rel_l2_end": 3e-4,
                   "spe_rel_diff_end": -2e-6} for ic in ("random", "synthetic")}
        d = {"gate_threshold": thr, "gate_pass": pass_, "gate_reasons": list(reasons),
             "gate": {"random": {"rel_l2_max": err / 5}, "synthetic": {"rel_l2_max": err}}, "long": lg,
             "ms_per_step": ms, "speedup": round(9.29 / ms, 3)}
        if opt is not None:
            d["optstep_s"] = opt
        return d
    return {"ref_ms_per_step": 9.29, "ref_optstep_s": 4.46, "fastest_passing": "mixed",
            "variants": {"lean64": var(True, 1e-12, 5e-16, False, 5.4, 2.6),
                         "mixed": var(True, 1e-5, 4e-7, False, 3.2, 1.6),
                         "lean32": var(False, 1e-5, 3e-5, False, 2.6, None, ["rel1_synthetic=3.000e-05>1e-05"])}}


def run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = R.main(argv)
    return rc, buf.getvalue()


def main():
    with tempfile.TemporaryDirectory() as d:
        p1, p2 = os.path.join(d, "seed1.json"), os.path.join(d, "seed2.json")
        with open(p1, "w") as fh:
            json.dump(seed1_json(), fh)
        with open(p2, "w") as fh:
            json.dump(seed2_json(), fh)
        out_txt = os.path.join(d, "ranking.txt")
        rc, out = run(["--seed1", p1, "--seed2", p2, "--out", out_txt])
        assert rc == 0 and "OPT_RANK_OK" in out, out
        assert "OPT_RANK_WINNER mixed (seed 2" in out, out                 # fastest PASSING, non-bitwise: wins
        assert "OPT_RANK_WINNER lean32" not in out and "OPT_RANK_WINNER lean_split" not in out, out   # FAIL never wins
        assert "OPT_RANK_BEST_F64 lean64" in out, out                     # fastest float64-only passing variant
        rows = R.rank(R.rows_seed1(seed1_json()) + R.rows_seed2(seed2_json()))
        order = [r["variant"] for r in rows]
        assert order[:4] == ["mixed", "lean64", "lean_u1", "lean"], order   # passing by speed; failed ones after
        assert order.index("lean_u1") < order.index("lean"), order         # non-bitwise but faster ranks above bitwise
        assert all(not r["gate_pass"] for r in rows[4:]), order            # the failed ones trail, whatever their speed
        assert os.path.isfile(out_txt) and "OPT_RANK_OK" in open(out_txt).read()
        # the measured 1-step error and bitwise flag appear as information in the table
        assert "3.00e-16(1e-12)" in out and "no" in out and "yes" in out, out
        # seed1 alone (seed2 missing): still a ranking, with a WARN
        rc, out = run(["--seed1", p1, "--seed2", os.path.join(d, "absent.json")])
        assert rc == 0 and "OPT_RANK_WINNER lean_u1 (seed 1" in out and "WARN OPT_RANK_MISSING_INPUT" in out, out
        # nothing to read: an error token, non-zero exit
        rc, out = run(["--seed1", os.path.join(d, "none.json")])
        assert rc == 1 and "ERROR OPT_RANK_NO_INPUT" in out, out
        # every variant failing: table printed, explicit WARN, no winner line
        j2 = seed2_json()
        for v in j2["variants"].values():
            v["gate_pass"] = False
        with open(p2, "w") as fh:
            json.dump(j2, fh)
        rc, out = run(["--seed2", p2])
        assert rc == 0 and "WARN OPT_RANK_NO_PASSING_VARIANT" in out and "OPT_RANK_WINNER" not in out, out
    print("RANK_TEST_OK rank_round1: failed gates never win, non-bitwise is not penalised, f64-only best reported")


if __name__ == "__main__":
    main()
