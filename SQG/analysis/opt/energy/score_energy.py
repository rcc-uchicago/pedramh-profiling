"""score_energy.py <results dir> -- score PREDICTIONS_energy.md's 17 predictions from energy_summary.json + worker_*.json.
numpy-free (stdlib only)."""
import glob
import json
import os
import sys

d = sys.argv[1]
S = json.load(open(os.path.join(d, "energy_summary.json")))
W = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "worker_*.json")))]
k = "%g" % S["tau_eval"] if S.get("tau_eval") else None
T_E = S["rules"]["T_E"]
per = S["per"]


def mean(x):
    return sum(x) / len(x)


def g(arm, inv, key, tau=None):
    return per[arm][inv][tau or k][key]


res = []


def score(i, ok, txt):
    res.append((i, ok, txt))


bits = [v for w in W for v in w.get("bitwise_vs_gateD", {}).values()]
score(1, bool(bits) and all(bits) and all(w.get("bitwise_vs_gateD") for w in W),
      "bitwise %d/%d checks; per IC %s" % (sum(bits), len(bits), [sorted(w.get("bitwise_vs_gateD", {})) for w in W]))
dr = [a.get("diag_max_rel") for w in W for a in w["arms"].values() if a.get("diag_max_rel") is not None]
dr += [w["static_at_ic"]["diag_rel"] for w in W]
score(2, max(dr) <= 1e-12, "max diag rel %.2e" % max(dr))
rS = g("lean_u1", "S", "r")
score(3, -5e-4 <= mean(rS) <= -5e-5, "r_SPE(ref) mean %.3e per IC %s" % (mean(rS), ["%.3e" % x for x in rS]))
rIF = g("lean_u1", "S", "r_IF")
ratio = [a / b for a, b in zip(rS, rIF)]
score(4, 0.8 <= mean(rS) / mean(rIF) <= 1.2, "r/r_IF mean-ratio %.3f per IC %s; r_bal %s" % (
    mean(rS) / mean(rIF), ["%.3f" % x for x in ratio], ["%.3e" % x for x in g("lean_u1", "S", "r_bal")]))
rV = g("lean_u1", "V", "r")
score(5, max(abs(x) for x in rV) <= 1e-4, "r_VIE(ref) %s (r_IF %s, quad %s)" % (
    ["%.3e" % x for x in rV], ["%.3e" % x for x in g("lean_u1", "V", "r_IF")],
    ["%.1e" % x for x in g("lean_u1", "V", "quad")]))
nul = {i: max(abs(x) for x in g("f64ic32", i, "s")) for i in ("S", "V")}
score(6, max(nul.values()) <= 1e-6, "null max|s| S %.2e V %.2e" % (nul["S"], nul["V"]))
cal = S["validity"].get("calibration", {})
score(7, S["validity"].get("V4_calibration") is True, "calibration %s" % {
    i: (c["qualifies"], None if c["ratio"] is None else round(c["ratio"], 4)) for i, c in cal.items()})
sAm = {i: mean(g("dquant", i, "sA")) for i in ("S", "V")}
score(8, max(abs(v) for v in sAm.values()) >= T_E, "dquant mean sA S %.3e V %.3e" % (sAm["S"], sAm["V"]))
l32 = S.get("lean32")
if l32:
    score(9, max(abs(l32[i]["mean_s"]) for i in ("S", "V")) >= T_E, "lean32 tau %s mean s S %.3e V %.3e" % (
        l32["S"]["tau"], l32["S"]["mean_s"], l32["V"]["mean_s"]))
    r10 = l32["V"]["ratio_s_over_sA"]
    score(10, r10 is not None and 0.5 <= r10 <= 2.0, "lean32 s/sA V %s S %s" % (r10, l32["S"]["ratio_s_over_sA"]))
else:
    score(9, None, "lean32 missing")
    score(10, None, "lean32 missing")
c = S.get("candidate", {})
score(11, max(c[i]["max_abs_s"] for i in ("S", "V")) <= 1e-6, "ek max|s| S %.2e V %.2e; mean s S %.2e V %.2e" % (
    c["S"]["max_abs_s"], c["V"]["max_abs_s"], c["S"]["mean_s"], c["V"]["mean_s"]))
score(12, S["verdict"] == "PASS_E", "verdict %s" % S["verdict"])
score(13, all(c[i]["equiv_double"] for i in ("S", "V")), "equiv_double %s" % {i: c[i]["equiv_double"] for i in "SV"})
score(14, any(c[i]["detected"] for i in ("S", "V")), "detected %s" % {i: c[i]["detected"] for i in "SV"})
ok15, txt15 = True, []
for i in ("S", "V"):
    ref = mean(g("dquant", i, "s"))
    for tau in ("0.1", "0.2", "0.3", "0.4", "0.5"):
        if tau in per["dquant"][i]:
            v = mean(per["dquant"][i][tau]["s"])
            ok15 = ok15 and abs(v / ref - 1) <= 0.2
            txt15.append("%s%s:%.3f" % (i, tau, v / ref))
score(15, ok15, "dquant s(tau)/s(tau_eval) " + " ".join(txt15))
tm = S["timing"]
tgt = {"lean_u1": 6.40, "lean32ek_fast": 2.70, "lean32": 3.20}
score(16, all(tm.get(a) is not None and abs(tm[a] / v - 1) <= 0.05 for a, v in tgt.items()),
      "ms/step " + " ".join("%s=%s" % (a, tm.get(a)) for a in tm))
dx = [abs(x) for i in ("S", "V") for x in per["lean32ek_fast"][i].get("0.5", {}).get("dXbar_rel", [float("inf")])]
score(17, max(dx) <= 1e-3, "ek paired time-mean rel diff max %.2e" % max(dx))
n_ok = sum(1 for r in res if r[1] is True)
for i, ok, txt in res:
    print("P%-2d %-5s %s" % (i, {True: "CONF", False: "REFUT", None: "N/A"}[ok], txt))
print("SCORE %d/%d confirmed; verdict %s tau_eval %s" % (n_ok, len(res), S["verdict"], S.get("tau_eval")))
