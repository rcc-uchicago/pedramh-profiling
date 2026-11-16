"""test_analyze_energy.py -- analyze_energy.py on synthetic budgets with known answers. numpy only, Python >= 3.6.

Each synthetic arm run has smooth P(t), eps(t) and X(t) = X0 + int (P - eps) + S int P (analytic), so its closure
residual is r = S exactly. Q, A and E columns are set to -2 r_IF P / dt, -sA P and -sE P, so r_IF, sA and sE are known
too. Checks Simpson's rule, recovery of r / s / predictions, every verdict branch, the deadline truncation path and
the npz + json round trip. Prints ENERGY_ANALYSIS_TEST_OK.
"""
import copy
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_energy as AE                                          # noqa: E402

FIELDS = ("SPE", "VIE", "P_S", "nu_S", "mu_S", "P_V", "nu_V", "mu_V",
          "Q_S", "Q_V", "B_S", "B_V", "A_S", "A_V", "E_S", "E_V")
H, DT, NC = 2e-4, 5e-6, 2500
R_REF = {"S": -1.5e-4, "V": -3e-6}
SA = {"S": 2e-3, "V": 5e-3}


def series(S, rIF, sA, sE, phase, n=NC):
    """vals (n + 1, 16) with closure residual S[inv], integrating-factor term rIF[inv], sA[inv], sE[inv]."""
    t = H * np.arange(n + 1)
    v = np.zeros((n + 1, len(FIELDS)))
    for inv, p0, x0, w1, w2 in (("S", 50.0, 40.0, 30.0, 17.0), ("V", 4.0, 9.0, 11.0, 7.0)):
        c = {k: FIELDS.index(x) for k, x in AE.COL[inv].items()}
        P = p0 * (1.0 + 0.3 * np.sin(w1 * t + phase))
        eps = p0 * (1.0 + 0.2 * np.cos(w2 * t))
        iP = p0 * (t + 0.3 * (np.cos(phase) - np.cos(w1 * t + phase)) / w1)
        ieps = p0 * (t + 0.2 * np.sin(w2 * t) / w2)
        v[:, c["X"]] = x0 + iP - ieps + S[inv] * iP
        v[:, c["P"]] = P
        v[:, c["nu"]] = 0.9 * eps
        v[:, c["mu"]] = 0.1 * eps
        v[:, c["Q"]] = -2.0 * rIF[inv] * P / DT
        v[:, c["B"]] = 1.1 * v[:, c["Q"]]
        v[:, c["A"]] = -sA[inv] * P
        v[:, c["E"]] = -sE[inv] * P
    return v


def make_runs(extra=None, bitwise=True, calls=None, crash=(), sa=None):
    """4 workers x 5 arms. extra[arm] = {inv: added s}; sA columns for dquant / lean32 = sa (default SA)."""
    extra = extra or {}
    calls = calls or {}
    sa = sa or SA
    add = {"lean_u1": {"S": 0.0, "V": 0.0}, "lean32ek_fast": {"S": 5e-7, "V": 5e-7},
           "f64ic32": {"S": 1e-9, "V": -1e-9}, "dquant": dict(sa), "lean32": {"S": 3e-3, "V": 6e-3}}
    add.update(extra)
    runs = {}
    for w in range(4):
        arms, info = {}, {}
        for arm in AE.REQUIRED + (AE.POS,):
            S = {inv: R_REF[inv] + add[arm][inv] for inv in AE.INV}
            sA = sa if arm in ("dquant", "lean32") else {"S": 0.0, "V": 0.0}
            n = calls.get((w, arm), NC)
            arms[arm] = series(S, R_REF, sA, {"S": 1e-8, "V": 1e-8}, 0.3 * w, n=n)
            info[arm] = {"calls": n, "ms_per_step": 6.4 if arm in ("lean_u1", "f64ic32", "dquant") else 2.7,
                         "finite": True, "cfl_max": 0.3, "diag_max_rel": 1e-15,
                         "error": "RuntimeError: x" if arm in crash else None, "stopped_by_deadline": n < NC}
            if arm in crash:
                del arms[arm]
        wj = {"worker": w, "member": w + 1, "t": 100.0 * (w + 1), "fields": list(FIELDS), "h": H, "dt": DT,
              "arms": info, "bitwise_vs_gateD": {"0": True, "1": True, "2500": bool(bitwise or w)},
              "static_at_ic": {"diag_rel": 1e-15}}
        runs[w] = (wj, arms)
    return runs


def close(a, b, tol):
    return abs(a - b) <= tol * max(abs(b), 1e-300)


def main():
    # Simpson is exact for cubics; trapezoid is not.
    t = np.linspace(0.0, 1.0, 11)
    assert close(AE.simpson(t ** 3, 0.1), 0.25, 1e-13), AE.simpson(t ** 3, 0.1)
    assert not close(AE.trapezoid(t ** 3, 0.1), 0.25, 1e-3)
    try:
        AE.simpson(t[:-1], 0.1)
        raise AssertionError("odd interval count accepted")
    except ValueError:
        pass

    # Recovery on the PASS scenario.
    out = AE.analyze(make_runs())
    k = "0.5"
    assert out["tau_eval"] == 0.5, out["tau_eval"]
    for inv in AE.INV:
        ref = out["per"]["lean_u1"][inv][k]
        assert all(close(r, R_REF[inv], 1e-6) for r in ref["r"]), (inv, ref["r"])
        assert all(close(r, R_REF[inv], 1e-9) for r in ref["r_IF"]), (inv, ref["r_IF"])
        assert all(q < 1e-6 for q in ref["quad"]), ref["quad"]
        cal = out["per"]["dquant"][inv][k]
        assert all(close(s, SA[inv], 1e-4) for s in cal["s"]), cal["s"]
        assert all(close(s, SA[inv], 1e-9) for s in cal["sA"]), cal["sA"]
        cand = out["per"]["lean32ek_fast"][inv][k]
        assert all(abs(s - 5e-7) < 1e-9 for s in cand["s"]), cand["s"]
    assert out["verdict"] == "PASS_E", (out["verdict"], out["why"])
    assert out["candidate"]["S"]["equiv_double"] and not out["candidate"]["V"]["equiv_double"], out["candidate"]
    assert out["candidate"]["S"]["detected"] and out["candidate"]["V"]["detected"]
    assert close(out["lean32"]["V"]["ratio_s_over_sA"], 6e-3 / 5e-3, 1e-3), out["lean32"]

    # Verdict branches.
    cases = [
        ("FAIL_E", dict(extra={"lean32ek_fast": {"S": 3e-4, "V": 1e-7}})),
        ("FAIL_E", dict(extra={"lean32ek_fast": {"S": 1e-7, "V": -2e-4}})),
        ("INCONCLUSIVE_CALIBRATION", dict(extra={"dquant": {"S": 2 * SA["S"], "V": 2 * SA["V"]}})),
        ("INCONCLUSIVE_NOISE", dict(extra={"f64ic32": {"S": 5e-5, "V": 0.0}})),
        ("INCONCLUSIVE_PROD_MISMATCH", dict(bitwise=False)),
        ("INCONCLUSIVE_RUN", dict(calls={(2, "dquant"): 500})),
        ("PASS_E", dict(calls={(1, "dquant"): 1500})),
        ("PASS_E", dict(crash=("lean32",))),
        ("INCONCLUSIVE_RUN", dict(crash=("f64ic32",))),
    ]
    for want, kw in cases:
        got = AE.analyze(make_runs(**kw))
        assert got["verdict"] == want, (want, kw, got["verdict"], got["why"])
    trunc = AE.analyze(make_runs(calls={(1, "dquant"): 1500}))
    assert trunc["tau_eval"] == 0.3, trunc["tau_eval"]
    assert AE.analyze(make_runs(crash=("lean32",))).get("lean32") is None

    # One calibration invariant below the SNR floor (5e-5 < 10 x 9e-6; miscalibrated 3x but not qualifying): the other
    # must carry V4.
    weak = copy.deepcopy(make_runs(extra={"f64ic32": {"S": 1e-9, "V": 9e-6}, "dquant": {"S": 2e-3, "V": 1.5e-4}},
                                   sa={"S": 2e-3, "V": 5e-5}))
    got = AE.analyze(weak)
    assert got["validity"]["calibration"]["V"]["qualifies"] is False, got["validity"]["calibration"]
    assert got["verdict"] == "PASS_E", (got["verdict"], got["why"])

    # npz + json round trip through load() and main()'s writers.
    d = tempfile.mkdtemp(prefix="energy_test_")
    try:
        runs = make_runs()
        for w, (wj, arms) in runs.items():
            with open(os.path.join(d, "worker_%d.json" % w), "w") as fh:
                json.dump(wj, fh)
            for arm, vals in arms.items():
                np.savez(os.path.join(d, "budget_w%d_%s.npz" % (w, arm)), vals=vals, fields=np.array(FIELDS))
        back = AE.analyze(AE.load(d))
        assert back["verdict"] == "PASS_E" and back["tau_eval"] == 0.5, back["verdict"]
        AE.write_table(back, os.path.join(d, "t.csv"))
        with open(os.path.join(d, "s.json"), "w") as fh:
            json.dump(AE.jsonable(back), fh)
        lines = AE.report(back)
        assert lines[-1].startswith("ENERGY_VERDICT PASS_E") and len(lines) <= 12, lines
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("ENERGY_ANALYSIS_TEST_OK verdict branches %d, recovery r/s/r_IF/sA, truncation, round trip (numpy %s)"
          % (len(cases) + 2, np.__version__))


if __name__ == "__main__":
    main()
