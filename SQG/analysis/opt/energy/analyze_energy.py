"""analyze_energy.py -- the decision rule of PREDICTIONS_energy.md, applied to energy_budget.py's output.

  python analyze_energy.py --in <dir with budget_w*_*.npz + worker_*.json> --out <results dir>

numpy + standard library only, Python >= 3.6 (so it runs on a login node's system python3 too). For each worker (IC),
arm, invariant X in (S = SPE, V = VIE) and tau in TAUS, with Simpson's rule on the 40-step samples:
  R = X(tau) - X(0) - int (P_X - eps_X),  r = R / int P_X              closure residual, fraction of the injection
  s = r(arm) - r(lean_u1)                                              paired spurious source
  r_IF = -(dt/2) int Q_X / int P_X,  r_bal = -(dt/2) int B_X / int P_X   the integrating-factor term (two forms)
  sA = -int A_X / int P_X,  sE = -int E_X / int P_X                      mechanism-A / lean32ek-table predictions
  quad = |Simpson - trapezoid| of int (P_X - eps_X) / int P_X           the quadrature bound
  time means of X, P_X, eps_X over [0, tau]
Writes energy_summary.json and energy_table.csv; prints ENERGY_VERDICT <verdict> and ENERGY_ANALYSIS_DONE.
"""
import argparse
import csv
import glob
import json
import math
import os
import re

import numpy as np

TAUS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
REF, CAND, NULL, CAL, POS = "lean_u1", "lean32ek_fast", "f64ic32", "dquant", "lean32"
REQUIRED = (REF, CAND, NULL, CAL)
INV = ("S", "V")
COL = {"S": {"X": "SPE", "P": "P_S", "nu": "nu_S", "mu": "mu_S", "Q": "Q_S", "B": "B_S", "A": "A_S", "E": "E_S"},
       "V": {"X": "VIE", "P": "P_V", "nu": "nu_V", "mu": "mu_V", "Q": "Q_V", "B": "B_V", "A": "A_V", "E": "E_V"}}
# The decision rule (PREDICTIONS_energy.md, fixed before the first run).
T_E = 1e-4
NULL_MAX = 0.1 * T_E
CAL_TOL = 0.10
CAL_MIN_SNR = 10.0
TAU_MIN = 0.2
CFL_MAX = 0.72
DIAG_TOL = 1e-12
EQUIV_FRAC = 0.1
DETECT_SNR = 3.0


def simpson(y, h):
    """Composite Simpson's rule over an even number of intervals."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y) - 1
    if n < 2 or n % 2:
        raise ValueError("simpson needs an even number >= 2 of intervals, got %d" % n)
    return h / 3.0 * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2]))


def trapezoid(y, h):
    y = np.asarray(y, dtype=np.float64)
    return h * (0.5 * y[0] + 0.5 * y[-1] + np.sum(y[1:-1]))


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else repr(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def terms(vals, fields, h, dt, n, inv):
    """The budget quantities of one arm run, over its first n intervals, for invariant inv."""
    c = {k: fields.index(v) for k, v in COL[inv].items()}
    y = vals[:n + 1]
    X, P = y[:, c["X"]], y[:, c["P"]]
    eps = y[:, c["nu"]] + y[:, c["mu"]]
    tau = n * h
    IP = simpson(P, h)
    net = simpson(P - eps, h)
    R = X[-1] - X[0] - net
    return {"r": R / IP, "IP": IP, "quad": abs(net - trapezoid(P - eps, h)) / abs(IP),
            "r_IF": -0.5 * dt * simpson(y[:, c["Q"]], h) / IP, "r_bal": -0.5 * dt * simpson(y[:, c["B"]], h) / IP,
            "sA": -simpson(y[:, c["A"]], h) / IP, "sE": -simpson(y[:, c["E"]], h) / IP,
            "Xbar": simpson(X, h) / tau, "Pbar": IP / tau, "epsbar": simpson(eps, h) / tau}


def load(indir):
    """{worker: (worker json, {arm: vals})}."""
    runs = {}
    for fn in sorted(glob.glob(os.path.join(indir, "worker_*.json"))):
        with open(fn) as fh:
            wj = json.load(fh)
        w = int(wj["worker"])
        arms = {}
        for npz in glob.glob(os.path.join(indir, "budget_w%d_*.npz" % w)):
            arm = re.match(r"budget_w\d+_(.+)\.npz$", os.path.basename(npz)).group(1)
            with np.load(npz) as z:
                arms[arm] = np.asarray(z["vals"], dtype=np.float64)
        runs[w] = (wj, arms)
    return runs


def analyze(runs):
    workers = sorted(runs)
    if not workers:
        return {"verdict": "INCONCLUSIVE_RUN", "why": ["no worker output"]}
    wj0 = runs[workers[0]][0]
    fields = list(wj0["fields"])
    h, dt = float(wj0["h"]), float(wj0["dt"])

    def calls(w, arm):
        info = runs[w][0]["arms"].get(arm, {})
        if info.get("error") is not None or arm not in runs[w][1]:
            return 0
        return min(int(info.get("calls", 0)), len(runs[w][1][arm]) - 1)

    def reached(arm_list, w_list):
        ok = [t for t in TAUS if all(calls(w, a) >= int(round(t / h)) for w in w_list for a in arm_list)]
        return max(ok) if ok else None

    tau_eval = reached(REQUIRED, workers)
    tau_pos = reached((REF, POS), workers)
    out = {"rules": {"T_E": T_E, "NULL_MAX": NULL_MAX, "CAL_TOL": CAL_TOL, "CAL_MIN_SNR": CAL_MIN_SNR,
                     "TAU_MIN": TAU_MIN, "CFL_MAX": CFL_MAX, "DIAG_TOL": DIAG_TOL, "EQUIV_FRAC": EQUIV_FRAC,
                     "DETECT_SNR": DETECT_SNR},
           "ics": [{"worker": w, "member": runs[w][0]["member"], "t": runs[w][0]["t"],
                    "static_at_ic": runs[w][0].get("static_at_ic", {})} for w in workers],
           "tau_eval": tau_eval, "tau_lean32": tau_pos, "h": h, "dt": dt, "per": {}, "timing": {}, "runs": {}}

    # ---- per arm x invariant x tau: lists over workers ------------------------------------------------------------
    per = out["per"]
    for arm in (REF, CAND, NULL, CAL, POS):
        per[arm] = {}
        for inv in INV:
            per[arm][inv] = {}
            for tau in TAUS:
                n = int(round(tau / h))
                if not all(calls(w, arm) >= n and calls(w, REF) >= n for w in workers):
                    continue
                rows = [terms(runs[w][1][arm], fields, h, dt, n, inv) for w in workers]
                refs = [terms(runs[w][1][REF], fields, h, dt, n, inv) for w in workers]
                d = {k: [r[k] for r in rows] for k in rows[0]}
                d["s"] = [r["r"] - q["r"] for r, q in zip(rows, refs)]
                d["dXbar_rel"] = [(r["Xbar"] - q["Xbar"]) / q["Xbar"] for r, q in zip(rows, refs)]
                d["dPbar_rel"] = [(r["Pbar"] - q["Pbar"]) / q["Pbar"] for r, q in zip(rows, refs)]
                d["depsbar_rel"] = [(r["epsbar"] - q["epsbar"]) / q["epsbar"] for r, q in zip(rows, refs)]
                per[arm][inv]["%g" % tau] = d
    for w in workers:
        out["runs"][str(w)] = {a: {k: runs[w][0]["arms"][a].get(k) for k in
                                   ("calls", "ms_per_step", "finite", "cfl_max", "diag_max_rel", "error",
                                    "stopped_by_deadline")} for a in runs[w][0]["arms"]}
    for arm in (REF, CAND, NULL, CAL, POS):
        ms = [runs[w][0]["arms"].get(arm, {}).get("ms_per_step") for w in workers]
        ms = [m for m in ms if m is not None]
        out["timing"][arm] = float(np.median(ms)) if ms else None

    # ---- validity -------------------------------------------------------------------------------------------------
    why = []
    bits = [v for w in workers for v in runs[w][0].get("bitwise_vs_gateD", {}).values()]
    v1 = bool(bits) and all(bits) and all(len(runs[w][0].get("bitwise_vs_gateD", {})) > 0 for w in workers)
    if not v1:
        why.append("V1 lean_u1 frames not bitwise equal to gate D's (or not checked)")
    diag = [runs[w][0]["arms"].get(a, {}).get("diag_max_rel") for w in workers for a in REQUIRED]
    diag += [runs[w][0].get("static_at_ic", {}).get("diag_rel") for w in workers]
    v2 = all(d is not None and float(d) <= DIAG_TOL for d in diag)
    if not v2:
        why.append("V2 budget() vs GD.diagnostics above %g (or missing)" % DIAG_TOL)
    finite = all(runs[w][0]["arms"].get(a, {}).get("finite") is True for w in workers for a in REQUIRED)
    cfl = [runs[w][0]["arms"].get(a, {}).get("cfl_max") for w in workers for a in REQUIRED]
    cfl_ok = all(c is not None and float(c) < CFL_MAX for c in cfl)
    v5 = finite and cfl_ok and tau_eval is not None and tau_eval >= TAU_MIN - 1e-12
    if not v5:
        why.append("V5 run: finite=%s cfl_ok=%s tau_eval=%s" % (finite, cfl_ok, tau_eval))
    validity = {"V1_bitwise": v1, "V2_diag": v2, "V5_run": v5}
    verdict = None
    if v5:
        k = "%g" % tau_eval
        null = {inv: max(abs(x) for x in per[NULL][inv][k]["s"]) for inv in INV}
        v3 = all(null[inv] <= NULL_MAX for inv in INV)
        if not v3:
            why.append("V3 null max|s| %s > %g" % ({i: "%.2e" % null[i] for i in INV}, NULL_MAX))
        cal = {}
        for inv in INV:
            pred = float(np.mean(per[CAL][inv][k]["sA"]))
            meas = float(np.mean(per[CAL][inv][k]["s"]))
            qual = abs(pred) >= CAL_MIN_SNR * null[inv] and pred != 0.0
            ratio = meas / pred if pred != 0.0 else None
            cal[inv] = {"pred_sA_mean": pred, "meas_s_mean": meas, "ratio": ratio, "qualifies": bool(qual),
                        "ok": bool(qual and ratio is not None and abs(ratio - 1.0) <= CAL_TOL)}
        v4 = any(c["qualifies"] for c in cal.values()) and all(c["ok"] for c in cal.values() if c["qualifies"])
        if not v4:
            why.append("V4 calibration %s" % {i: (c["qualifies"], c["ratio"]) for i, c in cal.items()})
        validity.update({"V3_null": v3, "null_max_abs_s": null, "V4_calibration": v4, "calibration": cal})
        cand = {inv: {"max_abs_s": max(abs(x) for x in per[CAND][inv][k]["s"]),
                      "mean_s": float(np.mean(per[CAND][inv][k]["s"])),
                      "mean_r_ref": float(np.mean(per[REF][inv][k]["r"]))} for inv in INV}
        for inv in INV:
            c = cand[inv]
            c["pass"] = c["max_abs_s"] <= T_E
            c["equiv_double"] = abs(c["mean_s"]) <= EQUIV_FRAC * abs(c["mean_r_ref"])
            c["detected"] = abs(c["mean_s"]) > DETECT_SNR * null[inv]
            c["over_T_E"] = c["max_abs_s"] / T_E
        out["candidate"] = cand
        if not v1:
            verdict = "INCONCLUSIVE_PROD_MISMATCH"
        elif not v2:
            verdict = "INCONCLUSIVE_DIAG"
        elif not v3:
            verdict = "INCONCLUSIVE_NOISE"
        elif not v4:
            verdict = "INCONCLUSIVE_CALIBRATION"
        else:
            verdict = "PASS_E" if all(cand[inv]["pass"] for inv in INV) else "FAIL_E"
        if tau_pos is not None:
            kp = "%g" % tau_pos
            out["lean32"] = {inv: {"tau": tau_pos, "mean_s": float(np.mean(per[POS][inv][kp]["s"])),
                                   "mean_sA": float(np.mean(per[POS][inv][kp]["sA"])),
                                   "ratio_s_over_sA": (float(np.mean(per[POS][inv][kp]["s"]))
                                                       / float(np.mean(per[POS][inv][kp]["sA"]))
                                                       if float(np.mean(per[POS][inv][kp]["sA"])) else None),
                                   "over_T_E": abs(float(np.mean(per[POS][inv][kp]["s"]))) / T_E} for inv in INV}
    else:
        verdict = "INCONCLUSIVE_RUN"
    out["validity"] = validity
    out["verdict"] = verdict
    out["why"] = why
    return out


def write_table(out, path):
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["arm", "inv", "tau", "ic", "r", "s", "r_IF", "r_bal", "sA", "sE", "quad", "dXbar_rel",
                     "dPbar_rel", "depsbar_rel"])
        for arm, a in out.get("per", {}).items():
            for inv, b in a.items():
                for tau, d in b.items():
                    for i in range(len(d["r"])):
                        wr.writerow([arm, inv, tau, i] + ["%.6e" % d[k][i] for k in
                                                          ("r", "s", "r_IF", "r_bal", "sA", "sE", "quad",
                                                           "dXbar_rel", "dPbar_rel", "depsbar_rel")])


def report(out):
    """At most ~12 lines."""
    lines = []
    k = "%g" % out["tau_eval"] if out.get("tau_eval") else None
    if k:
        for inv in INV:
            ref = out["per"][REF][inv][k]
            seg = ["%s tau=%s r_ref=%.3e (r_IF %.3e, quad %.1e)" % (
                "SPE" if inv == "S" else "VIE", k, np.mean(ref["r"]), np.mean(ref["r_IF"]), max(ref["quad"]))]
            for arm in (CAND, NULL, CAL, POS):
                d = out["per"].get(arm, {}).get(inv, {}).get(k)
                if d:
                    seg.append("%s s=%.2e (max|s| %.2e%s)" % (arm, np.mean(d["s"]), max(abs(x) for x in d["s"]),
                                                             ", sA %.2e" % np.mean(d["sA"]) if arm in (CAL, POS)
                                                             else ""))
            lines.append("ENERGY_TABLE " + "; ".join(seg))
    if "candidate" in out:
        for inv, c in out["candidate"].items():
            lines.append("ENERGY_CANDIDATE %s max|s|=%.2e (%.2e x T_E) equiv_double=%s detected=%s" % (
                inv, c["max_abs_s"], c["over_T_E"], c["equiv_double"], c["detected"]))
    lines.append("ENERGY_TIMING ms/step " + " ".join("%s=%s" % (a, out["timing"].get(a)) for a in out.get("timing", {})))
    lines.append("ENERGY_VERDICT %s tau_eval=%s%s" % (out["verdict"], out.get("tau_eval"),
                                                      (" why: " + " | ".join(out["why"])) if out.get("why") else ""))
    return lines


def main():
    global CAND, REQUIRED
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--candidate", default=CAND,
                    help="the arm judged (default lean32ek_fast; PREDICTIONS_gates_tilerows.md: tilerows_if25)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    CAND = a.candidate                                                # the same rule, another candidate
    REQUIRED = (REF, CAND, NULL, CAL)
    out = analyze(load(a.indir))
    out["candidate_arm"] = CAND
    with open(os.path.join(a.out, "energy_summary.json"), "w") as fh:
        json.dump(jsonable(out), fh, indent=1)
    write_table(out, os.path.join(a.out, "energy_table.csv"))
    for line in report(out):
        print(line, flush=True)
    print("ENERGY_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
