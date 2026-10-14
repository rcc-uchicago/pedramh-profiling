"""2 x 2 precision cross read-out (docs/2026-10-13_precision_cross_prereg.md, docs/makani-knowledge).

Training precision T (F-scratch = bf16, G1 = fp32) x rollout precision R (bf16, fp32), EMA epochs 15-19.
Per-member score A = mean over the 24 groups of the group-mean r_c, read from the scorer's
<tag>_channels.csv (the value score_window_protocol.py takes the median of). Effects are paired by start
index and epoch, pooled per member over epochs, then mean +- SE over members with a paired t-test.

Stdlib only (runs on a login node for its test, and in-job after PROTOCOL_SCORE_OK).
PASS = PRECISION_CROSS_RESULT outcome=<n> ...
"""
import argparse
import csv
import math
import os
import re
import statistics as st

EPOCHS = (15, 16, 17, 18, 19)          # rolled out and reported per epoch
PRIMARY = (15, 16, 18, 19)             # pooled estimate: the epochs where the bf16/bf16 cell is 8/8 healthy
                                       # (FSE17 is 4/8, a collapse checkpoint; fixed before the new cells ran)
# cell -> arm prefix; arm = prefix + 2-digit epoch (exact names: "FSE15" is not "FSEF15").
CELLS = {("bf16", "bf16"): "FSE", ("fp32", "fp32"): "G1E", ("bf16", "fp32"): "FSEF", ("fp32", "bf16"): "G1EB"}
SIG = 3.3          # house threshold: |effect| > 3.3 SE
BUDGET = re.compile(r"^(\S+?)(\d\d)\s+n=\s*(\d+)\s+dPS=\s*([-+0-9.eE]+).*?\(min\s+([-+0-9.eE]+)\).*?"
                    r"lastmonth neg=\s*([-+0-9.eE]+)%")
N_STEPS0, STRIDE = 7667, 16      # member i runs 7667 - 16 i steps (polaris_climate_run.pbs)
BLOWN_DPS_HPA = 100.0            # |dPS| above this = a blown member that the read-out did not truncate


def _betacf(a, b, x):
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a - 1 + m2) * (a + m2))
        d = 1.0 / (1.0 + aa * d)
        c = 1.0 + aa / c
        h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1 + m2))
        d = 1.0 / (1.0 + aa * d)
        c = 1.0 + aa / c
        h *= d * c
    return h


def t_pvalue(t, df):
    """Two-sided Student-t p-value via the regularized incomplete beta function."""
    if df < 1 or not math.isfinite(t):
        return float("nan") if not math.isfinite(t) else 0.0
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    return bt * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * _betacf(b, a, 1 - x) / b


def member_scores(path):
    """{arm: {member_index: A_m}} from a scorer channels CSV."""
    groups = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            m = re.search(r"(\d\d)\.nc$", r["member"]).group(1)
            groups.setdefault(r["arm"], {}).setdefault(m, {}).setdefault(r["group"], []).append(float(r["r"]))
    return {arm: {m: sum(sum(v) / len(v) for v in g.values()) / len(g) for m, g in mem.items()}
            for arm, mem in groups.items()}


def budgets(path):
    out = {}
    if path and os.path.exists(path):
        for line in open(path, errors="replace"):
            b = BUDGET.match(line.strip())
            if b:
                arm, mem, n, dps, tmin, neg = b.groups()
                out.setdefault(arm, {})[mem] = (float(dps), float(tmin), float(neg), int(n))
    return out


def healthy(bud, arm, m):
    """Prereg inclusion rule: ran to 2049-12-31, global TMQ > 0 throughout, |dPS| <= 100 hPa."""
    b = bud.get(arm, {}).get(m)
    if b is None:
        return False
    dps, tmin, _neg, n = b
    return n == N_STEPS0 - STRIDE * int(m) and tmin > 0 and math.isfinite(dps) and abs(dps) <= BLOWN_DPS_HPA


def effect(A, bud, minuend, subtrahend):
    """Pooled paired effect A[minuend cell] - A[subtrahend cell] over EPOCHS; returns dict."""
    per_member, per_epoch = {}, {}
    for e in EPOCHS:
        ax, ay = "%s%02d" % (CELLS[minuend], e), "%s%02d" % (CELLS[subtrahend], e)
        x, y = A.get(ax, {}), A.get(ay, {})
        d = {m: x[m] - y[m] for m in set(x) & set(y)
             if healthy(bud, ax, m) and healthy(bud, ay, m)}
        if d:
            per_epoch[e] = (sum(d.values()) / len(d), len(d))
        if e in PRIMARY:
            for m, v in d.items():
                per_member.setdefault(m, []).append(v)
    pooled = [sum(v) / len(v) for v in per_member.values()]
    n = len(pooled)
    if n < 2:
        return dict(mean=float("nan"), se=float("nan"), t=float("nan"), p=float("nan"), n=n, per_epoch=per_epoch)
    mean = sum(pooled) / n
    se = st.stdev(pooled) / math.sqrt(n)
    t = mean / se if se > 0 else (math.copysign(float("inf"), mean) if mean else 0.0)
    return dict(mean=mean, se=se, t=t, p=t_pvalue(t, n - 1), n=n, per_epoch=per_epoch)


def classify(eff, gate_ok):
    if not gate_ok:
        return 4, "unresolved: AMP_BF16_EQUIV_OK missing, so the fp32-trained/bf16-rollout cell is void"
    sig_neg = {k: (v["n"] >= 2 and v["mean"] < 0 and abs(v["t"]) > SIG) for k, v in eff.items()}
    sig_any = {k: (v["n"] >= 2 and abs(v["t"]) > SIG) for k, v in eff.items()}
    R = ("R|T=bf16", "R|T=fp32")
    T = ("T|R=bf16", "T|R=fp32")
    if all(sig_neg[k] for k in R) and not any(sig_any[k] for k in T):
        return 1, "rollout precision explains the gap"
    if all(sig_neg[k] for k in T) and not any(sig_any[k] for k in R):
        return 2, "training precision explains the gap"
    if any(sig_neg[k] for k in R) and any(sig_neg[k] for k in T):
        return 3, "both contribute"
    return 4, "unresolved"


def run(existing, new, gate_path, budget_paths, out_path):
    A = member_scores(existing)
    A.update({k: v for k, v in member_scores(new).items() if k not in A})
    gate_txt = open(gate_path).read() if gate_path and os.path.exists(gate_path) else ""
    gate_ok = any(l.startswith("AMP_BF16_EQUIV_OK") for l in gate_txt.splitlines())
    if not gate_ok:  # the fp32-trained / bf16-rollout cell is void (prereg, Gate)
        A = {k: v for k, v in A.items() if not k.startswith(CELLS[("fp32", "bf16")])}
    bud = {}
    for bp in budget_paths:
        for arm, mem in budgets(bp).items():
            bud.setdefault(arm, mem)
    eff = {
        "R|T=bf16": effect(A, bud, ("bf16", "fp32"), ("bf16", "bf16")),
        "R|T=fp32": effect(A, bud, ("fp32", "fp32"), ("fp32", "bf16")),
        "T|R=bf16": effect(A, bud, ("fp32", "bf16"), ("bf16", "bf16")),
        "T|R=fp32": effect(A, bud, ("fp32", "fp32"), ("bf16", "fp32")),
        "Total": effect(A, bud, ("fp32", "fp32"), ("bf16", "bf16")),
    }
    # interaction = R|T=fp32 - R|T=bf16, paired by member over epochs
    inter = {}
    for e in PRIMARY:
        cells = {c: A.get("%s%02d" % (p, e), {}) for c, p in CELLS.items()}
        common = set.intersection(*(set(v) for v in cells.values())) if all(cells.values()) else set()
        for m in common:
            if all(healthy(bud, "%s%02d" % (CELLS[c], e), m) for c in CELLS):
                i = (cells[("fp32", "fp32")][m] - cells[("fp32", "bf16")][m]) - \
                    (cells[("bf16", "fp32")][m] - cells[("bf16", "bf16")][m])
                inter.setdefault(m, []).append(i)
    pooled = [sum(v) / len(v) for v in inter.values()]
    if len(pooled) >= 2:
        im, ise = sum(pooled) / len(pooled), st.stdev(pooled) / math.sqrt(len(pooled))
        it = im / ise if ise > 0 else 0.0
        eff["Interaction"] = dict(mean=im, se=ise, t=it, p=t_pvalue(it, len(pooled) - 1), n=len(pooled), per_epoch={})
    outcome, label = classify({k: eff[k] for k in ("R|T=bf16", "R|T=fp32", "T|R=bf16", "T|R=fp32")}, gate_ok)

    lines = ["precision cross (prereg 2026-10-13): paired per-member A differences, pooled over EMA epochs "
             + ", ".join(str(e) for e in PRIMARY) + " (per-epoch values include 17)",
             "gate: " + (next(l for l in gate_txt.splitlines() if l.startswith("AMP_BF16_EQUIV_OK"))
                         if gate_ok else "MISSING (AMP_BF16_EQUIV_OK not found) -> fp32/bf16 cell void")]
    lines.append("cell medians of A by epoch over healthy members (healthy/8):")
    for c, p in CELLS.items():
        row = []
        for e in EPOCHS:
            arm = "%s%02d" % (p, e)
            v = [x for m, x in A.get(arm, {}).items() if healthy(bud, arm, m)]
            row.append("e%d %s (%d)" % (e, ("%.2f" % st.median(v)) if v else "-", len(v)))
        lines.append("  T=%s R=%s %-5s %s" % (c[0], c[1], p, "  ".join(row)))
    for k, v in eff.items():
        pe = " ".join("e%d %+.3f(n%d)" % (e, m, n) for e, (m, n) in sorted(v["per_epoch"].items()))
        lines.append("%-12s %+.3f +- %.3f  t=%+.1f  p=%.2g  members=%d  %s" % (k, v["mean"], v["se"], v["t"], v["p"], v["n"], pe))
    tot = eff["Total"]["mean"]
    if math.isfinite(tot) and tot:
        lines.append("shares of Total: R|T=bf16 %.0f%%, T|R=fp32 %.0f%%" % (
            100 * eff["R|T=bf16"]["mean"] / tot, 100 * eff["T|R=fp32"]["mean"] / tot))
    for c, p in CELLS.items():
        for e in EPOCHS:
            arm = "%s%02d" % (p, e)
            if arm in bud:
                b = bud[arm]
                lines.append("budget %-7s members=%d max|dPS|=%.3g hPa water_gate=%s" % (
                    arm, len(b), max(abs(x[0]) for x in b.values()),
                    "pass" if all(x[1] > 0 and x[2] <= 1.0 for x in b.values()) else "fail"))
    lines.append("PRECISION_CROSS_RESULT outcome=%d (%s)" % (outcome, label))
    text = "\n".join(lines) + "\n"
    if out_path:
        open(out_path, "w").write(text)
    print(text, end="")
    return outcome, eff


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--existing", required=True, help="channels CSV holding FSE15-19 and G1E15-19 (g1fs)")
    p.add_argument("--new", required=True, help="channels CSV holding FSEF15-19 and G1EB15-19 (pxc)")
    p.add_argument("--gate", required=True, help="gate.txt written by polaris_precision_cross.pbs")
    p.add_argument("--budget", nargs="+", required=True,
                   help="<tag>_budget.txt files; they define which members are healthy (prereg inclusion rule)")
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    run(a.existing, a.new, a.gate, a.budget, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
