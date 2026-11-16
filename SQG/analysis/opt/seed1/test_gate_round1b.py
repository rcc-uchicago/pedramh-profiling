"""test_gate_round1b.py -- run_seed1.gate_reason under round 1b's t rule, with hand-built comparison dicts (no JAX).

Round 1 (job 7814300) failed lean / lean_u1 / lean_split only because t (float32 time counter, unread by any field)
differed in its last bits after 40 steps. Round 1b: tc exact, level 2, |t - t_ref| <= 1e-6 |t_ref| at 1/40/480 steps.
    python3 SQG/analysis/opt/seed1/test_gate_round1b.py        -> SEED1_GATE_R1B_TEST_OK
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))     # SQG/analysis: profile_solver (stdlib at import)
sys.path.insert(0, HERE)
import run_seed1 as R  # noqa: E402


def cmp_(rel=0.0, t=1e-3, tref=1e-3, tc=True, lvl=2, fin=True):
    d = {"finite": fin, "max_rel_l2": rel, "worst_field": "qh", "t_equal": t == tref, "tc_equal": tc, "ablevel": lvl,
         "t_rel_diff": abs(t - tref) / abs(tref), "ref_finite": True}
    d["meta_ok"] = bool(tc and lvl == 2 and d["t_rel_diff"] <= R.T_REL_TOL)   # as run_seed1.compare
    return d


def gate(**over):
    g = {n: {"random": cmp_(), "synthetic": cmp_()} for n in ("1step", "40step", "480step")}
    for k, v in over.items():
        n, ic = k.split("__")
        g[n][ic] = v
    return g


def main():
    checks = [
        (gate(), None),
        # round 1's failure mode: t off in its last bits -> passes now (exact t stays reported as t_equal False)
        (gate(**{"40step__random": cmp_(t=2.1e-4 * (1 + 1e-7), tref=2.1e-4)}), None),
        (gate(**{"40step__random": cmp_(t=2.1e-4 * (1 + 1e-5), tref=2.1e-4)}), "40step_tc_ablevel_or_t"),
        (gate(**{"480step__synthetic": cmp_(tc=False)}), "480step_tc_ablevel_or_t"),
        (gate(**{"1step__random": cmp_(lvl=1)}), "1step_tc_ablevel_or_t"),
        (gate(**{"1step__random": cmp_(rel=2e-12)}), "1step_rel_l2"),             # field gates unchanged
        (gate(**{"40step__synthetic": cmp_(rel=2e-9)}), "40step_rel_l2"),
        (gate(**{"480step__random": cmp_(fin=False)}), "nonfinite_after_480"),
    ]
    for i, (g, want) in enumerate(checks):
        got = R.gate_reason(g)
        if (want is None and got is not None) or (want is not None and not (got or "").startswith(want)):
            raise SystemExit(f"ERROR SEED1_GATE_R1B_TEST case {i}: got {got!r}, want {want!r}")
    print(f"SEED1_GATE_R1B_TEST_OK {len(checks)} cases (t within {R.T_REL_TOL:g} passes; tc/level/fields unchanged)")


if __name__ == "__main__":
    main()
