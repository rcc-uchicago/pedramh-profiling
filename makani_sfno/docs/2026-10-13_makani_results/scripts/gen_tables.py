"""Emit the markdown tables for the results write-up from the recomputed JSON (no hand transcription)."""
import json
import math
import os
import re
import statistics as st

T = os.environ.get("MAKANI_RESULTS_WORK", os.path.dirname(os.path.abspath(__file__))) + "/"
R = json.load(open(T + "epochs/all.json"))
ARMS = json.load(open(T + "arms.json"))
SURV = {}
for _d in json.load(open(T + "protocol_survival.json")):
    if _d["tag"] not in SURV or _d["n"] > SURV[_d["tag"]]["n"]:
        SURV[_d["tag"]] = _d  # tag "A" also has a 1-member probe (7649650); keep the 8-member protocol run
RUNS = "/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_mn_scaling/"
out = []
P = out.append


def ser(k):
    return {int(e): v for e, v in R[k].items()}


def betacf(a, b, x):
    c, d = 1.0, 1 - (a + b) * x / (a + 1)
    d = 1 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a - 1 + m2) * (a + m2))
        d = 1 / (1 + aa * d)
        c = 1 + aa / c
        h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1 + m2))
        d = 1 / (1 + aa * d)
        c = 1 + aa / c
        h *= d * c
    return h


def ibeta(a, b, x):
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    return bt * betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * betacf(b, a, 1 - x) / b


def p_t(t, df):
    return ibeta(df / 2, 0.5, df / (df + t * t))


def paired(x, y):
    ax, ay = ARMS[x]["A_m"], ARMS[y]["A_m"]
    c = sorted(set(ax) & set(ay))
    d = [ax[m] - ay[m] for m in c]
    mean = sum(d) / len(d)
    se = st.stdev(d) / math.sqrt(len(d))
    return mean, se, p_t(mean / se, len(d) - 1)


def cp(k, n=8):
    lo = 0.0 if k == 0 else None
    hi = 1.0 if k == n else None
    def cdf(k_, p):
        return sum(math.factorial(n) // (math.factorial(i) * math.factorial(n - i)) * p ** i * (1 - p) ** (n - i) for i in range(k_ + 1))
    def bis(f):
        a, b = 0.0, 1.0
        for _ in range(80):
            m = (a + b) / 2
            if f(m):
                b = m
            else:
                a = m
        return (a + b) / 2
    if lo is None:
        lo = bis(lambda p: 1 - cdf(k - 1, p) >= 0.025)
    if hi is None:
        hi = bis(lambda p: cdf(k, p) <= 0.025)
    return lo, hi


def yaml_last(tag, key):
    v = None
    for l in open(RUNS + "e3sm_mn_scaling.%s.yaml" % tag):
        m = re.match(r"\s*%s:\s*(.*)$" % key, l)
        if m:
            v = m.group(1)
    return v


def fA(a):
    return ("%.2f" % a) if a < 400 else "> 400"


def ptxt(p):
    return ("%.1e" % p) if p < 1e-3 else ("%.3f" % p)


# ---------- Table 2: training ----------
info = [("128-node run", "P128", "prod128_alldata_v2", 128), ("A", "A", "prod1n_b32_sgdr", 1), ("B", "B", "nf4_prod_b16_r1", 2),
        ("F-warm", "Fwarm", "f_nosoil_2n_b32_e43_warm", 2), ("F-scratch", "Fscratch", "f_nosoil_2n_b32_e23_scratch", 2),
        ("F-depth4", "Fdepth4", "fsF_anneal_nf4_b16_rtmax100_scratch", 2), ("G1", "G1", "f_nosoil_2n_b32_e23_scratch_fp32", 2),
        ("G1-dry", "G1D", "f_nosoil_2n_b32_e63_scratch_fp32_dryair", 2)]
P("| arm | nodes | steps per epoch | global batch (samples per update) | epochs | updates | compute (node-hours) | median epoch time (s) | final validation loss | best validation loss (epoch) | best EMA validation loss (epoch) |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
for name, k, tag, nodes in info:
    d = ser(k)
    eps = sorted(d)
    steps = d[eps[-1]]["steps"]
    spe = yaml_last(tag, "n_train_samples_per_epoch")
    gb = ("%.0f" % (float(spe) / steps)) if spe else "?"
    upd = sum(d[e]["steps"] for e in eps)
    nh = sum(d[e]["t"] for e in eps) * nodes / 3600
    va = [(d[e]["va"], e) for e in eps]
    em = [(d[e]["ema"], e) for e in eps if d[e]["ema"] is not None]
    P("| %s | %d | %d | %s | %d | {:,} | %.1f | %.0f | %.6f | %.6f (e%d) | %s |".format(int(upd)) % (
        name, nodes, steps, gb, len(eps), nh, st.median(d[e]["t"] for e in eps), d[eps[-1]]["va"], min(va)[0], min(va)[1],
        ("%.6f (e%d)" % min(em)) if em else "no EMA kept"))
P("")

# ---------- Table 3: 5-yr survival ----------
rows3 = [("A e243 (raw)", "A"), ("B e01 (raw; its lowest-validation-loss checkpoint)", "B"), ("B e22 (raw)", "B22"), ("B e24 (raw)", "B24"),
         ("F-scratch e20 (raw)", "FS20"), ("F-scratch e21 (raw)", "FS21"), ("F-scratch e22 (raw)", "FS22"), ("F-scratch e23 (raw)", "FS23"),
         ("F-scratch e41 (raw)", "FS41"), ("F-scratch e42 (raw)", "FS42"), ("F-scratch e43 (raw)", "FS43"),
         ("F-scratch e23 (EMA) = FSEMA", "FSEMA"), ("F-depth4 e14 (raw)", "FD14"), ("F-depth4 e22 (raw)", "FD22"), ("F-depth4 e24 (raw)", "FD24")]
P("| checkpoint | rollouts finite to 2049-12-31 | 95% interval for the finite fraction | first non-finite step, range (years) |")
P("|---|---|---|---|")
for name, tag in rows3:
    s = SURV[tag]
    lo, hi = cp(s["complete"])
    tr = ("%d to %d (%.2f to %.2f)" % (s["trunc"][0], s["trunc"][-1], s["trunc"][0] / 1460, s["trunc"][-1] / 1460)) if s["trunc"] else "none"
    P("| %s | %d/8 | %.2f to %.2f | %s |" % (name, s["complete"], lo, hi, tr))
P("")

# ---------- Table 4: accuracy + gates ----------
rows4 = ["B22", "B24", "B22dryair", "B24dryair", "B22posclamp", "DRYAIR24", "FSEMA", "FSE26", "FSE30", "FD22", "FD24", "FD24dryair",
         "FDE28", "FDE32", "G1E15", "G1E15fix", "G1E18", "G1D20", "G1D39", "G1D59"]
P("| candidate | finite (of 8) | score A | worst group W | A minus B22 (paired mean ± SE; p) | gate 2: water physical | gate 3: \\|dPS\\| ≤ 1 hPa (range, hPa) | drifting groups (gate 4: ≤ 3) |")
P("|---|---|---|---|---|---|---|---|")
for k in rows4:
    a = ARMS[k]
    s = SURV[k]
    if k == "B22":
        diff = "reference"
    else:
        m, se, p = paired(k, "B22")
        diff = "%+.3f ± %.3f; %s" % (m, se, ptxt(p))
    g2 = ("pass" if a["gate2"] else "fail (%d of 8 members)" % a["gate2_bad_members"]) if "gate2" in a else "not read"
    lo, hi = a.get("dps_range", (float("nan"), float("nan")))
    dps = ("%+.2f to %+.2f" % (lo, hi)) if abs(lo) < 1e4 else "%.1e to %+.2f" % (lo, hi)
    g3 = ("pass (%s)" if a.get("gate3") else "fail (%s)") % dps
    P("| %s | %d | %.2f | %.2f | %s | %s | %s | %s |" % (k, s["complete"], a["A"], a["W"], diff, g2, g3, a.get("drifting", "—")))
P("")

# ---------- Table 4b: inference-only dry-air fix pairs ----------
P("| checkpoint | A without fix | A with fix | with minus without (paired ± SE; p) | \\|dPS\\| without → with (max over members, hPa) |")
P("|---|---|---|---|---|")
for base, fx in (("B22", "B22dryair"), ("B24", "B24dryair"), ("FD22", "FD22dryair"), ("FD24", "FD24dryair"), ("G1E15", "G1E15fix"), ("G1E16", "G1E16fix"), ("G1E19", "G1E19fix")):
    m, se, p = paired(fx, base)
    d0 = max(abs(v["dps"]) for v in ARMS[base]["budget"].values())
    d1 = max(abs(v["dps"]) for v in ARMS[fx]["budget"].values())
    P("| %s | %.2f | %.2f | %+.3f ± %.3f; %s | %.2f → %.2f |" % (base, ARMS[base]["A"], ARMS[fx]["A"], m, se, ptxt(p), d0, d1))
P("")

# ---------- Table 5: sweeps, every level ----------
def cyc(e):
    return (e - 4) % 20


for pre, run, title, single in (("FSE", "Fscratch", "F-scratch, EMA weights, bf16 rollouts", True),
                                ("G1E", "G1", "G1, EMA weights, fp32 rollouts", True),
                                ("G1D", "G1D", "G1-dry, EMA weights, fp32 rollouts with the dry-air constraint", True),
                                ("FDE", "Fdepth4", "F-depth4, EMA weights, bf16 rollouts (no warm restarts)", False)):
    d = ser(run)
    tags = sorted((int(t[len(pre):]), t) for t in SURV if t.startswith(pre) and t[len(pre):].isdigit())
    P("**%s**" % title)
    P("")
    P("| epoch | %sEMA validation loss | finite (of 8) | score A | max \\|dPS\\| (hPa) | water gate |" % ("cycle position (0–19) | " if single else ""))
    P("|---|%s---|---|---|---|---|" % ("---|" if single else ""))
    for e, t in tags:
        a = ARMS.get(t)
        s = SURV[t]
        em = d.get(e, {}).get("ema")
        if a and "budget" in a:
            dp = max(abs(v["dps"]) for v in a["budget"].values())
            dps = ("%.2f" % dp) if dp < 1e4 else "not finite"
            wg = "pass" if a.get("gate2") else "fail"
        else:
            dps, wg = "not read", "not read"
        P("| %d | %s%s | %d | %s | %s | %s |" % (e, ("%d | " % cyc(e)) if single else "", ("%.6f" % em) if em else "—", s["complete"],
                                               fA(a["A"]) if a else "not scored", dps, wg))
    P("")
open(T + "tables.md", "w").write("\n".join(out) + "\n")
print(open(T + "tables.md").read())
