"""Figure 3 + Table 6: the checkpoint choice. Score A, PS drift and negative-water area per candidate,
with the gates; the pick (lowest A among gate 1-3 passers) highlighted. Stdlib only.

  python3 make_fig3.py <outdir>      -> <outdir>/fig3_candidates.svg and ./table6.md
"""
import json
import math
import os
import statistics as st
import sys

HERE = os.environ.get("MAKANI_RESULTS_WORK", os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..")
ARMS = json.load(open(os.path.join(HERE, "arms.json")))
SURV = {}
for d in json.load(open(os.path.join(HERE, "protocol_survival.json"))):
    if d["tag"] not in SURV or d["n"] > SURV[d["tag"]]["n"]:
        SURV[d["tag"]] = d

# (arm, label, training precision, depth, dry-air constraint, EMA)
ROWS = [("G1D39", "G1-dry EMA e39", "fp32", 1, "trained in", "yes"),
        ("G1D59", "G1-dry EMA e59", "fp32", 1, "trained in", "yes"),
        ("G1E15", "G1 EMA e15", "fp32", 1, "none", "yes"),
        ("G1E15fix", "G1 EMA e15 + fix", "fp32", 1, "at prediction", "yes"),
        ("G1E18", "G1 EMA e18", "fp32", 1, "none", "yes"),
        ("G1E19fix", "G1 EMA e19 + fix", "fp32", 1, "at prediction", "yes"),
        ("FDE32", "F-depth4 EMA e32", "bf16", 5, "none", "yes"),
        ("FSE30", "F-scratch EMA e30", "bf16", 1, "none", "yes"),
        ("FSEMA", "F-scratch EMA e23", "bf16", 1, "none", "yes"),
        ("B22", "B e22", "bf16", 5, "none", "no"),
        ("B22dryair", "B e22 + fix", "bf16", 5, "at prediction", "no"),
        ("B24", "B e24", "bf16", 5, "none", "no"),
        ("DRYAIR24", "DRYAIR e24", "bf16", 5, "trained in", "no")]
PICK = "G1D39"


def stats(arm):
    a = ARMS[arm]
    b = a["budget"].values()
    dps = max(abs(x["dps"]) for x in b)
    neg = max(x["neg"] for x in b)
    g1 = SURV[arm]["complete"] == 8
    g2, g3 = a["gate2"], a["gate3"]
    return dict(A=a["A"], W=a["W"], Am=sorted(a["A_m"].values()), dps=dps, neg=neg, fin=SURV[arm]["complete"],
                g1=g1, g2=g2, g3=g3, g123=g1 and g2 and g3, drift=a.get("drifting"))


def betacf(a, b, x):
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a - 1 + m2) * (a + m2))
        d = 1.0 / (1.0 + aa * d); c = 1.0 + aa / c; h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1 + m2))
        d = 1.0 / (1.0 + aa * d); c = 1.0 + aa / c; h *= d * c
    return h


def p_t(t, df):
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    return bt * betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * betacf(b, a, 1 - x) / b


def paired(x, y):
    ax, ay = ARMS[x]["A_m"], ARMS[y]["A_m"]
    d = [ax[m] - ay[m] for m in sorted(set(ax) & set(ay))]
    mean = sum(d) / len(d)
    se = st.stdev(d) / math.sqrt(len(d))
    return mean, se, p_t(mean / se, len(d) - 1)


S = {arm: stats(arm) for arm, *_ in ROWS}

# ------------------------------------------------------------------ Table 6
t = ["| candidate | training precision | depth | dry-air constraint | EMA | finite (of 8) | score A | A minus pick (paired ± SE; p) | worst group W | max \\|dPS\\| (hPa) | max negative-water area (%) | gates 1–3 | drifting groups (≤ 3) |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for arm, label, prec, depth, dry, ema in ROWS:
    s = S[arm]
    if arm == PICK:
        diff = "**pick**"
    else:
        m, se, p = paired(arm, PICK)
        diff = "%+.3f ± %.3f; %s" % (m, se, ("%.1e" % p) if p < 1e-3 else ("%.3f" % p))
    fails = [g for g, ok in (("1", s["g1"]), ("2", s["g2"]), ("3", s["g3"])) if not ok]
    gates = "pass" if not fails else "fail (" + ", ".join(fails) + ")"
    dps = ("%.2f" % s["dps"]) if s["dps"] < 1e4 else "not finite"
    name = ("**%s**" % label) if arm == PICK else label
    t.append("| %s | %s | %d | %s | %s | %d | %.2f | %s | %.2f | %s | %.1f | %s | %s |" % (
        name, prec, depth, dry, ema, s["fin"], s["A"], diff, s["W"], dps, s["neg"], gates, s["drift"]))
open(os.path.join(HERE, "table6.md"), "w").write("\n".join(t) + "\n")

# ------------------------------------------------------------------ Figure 3
STYLE = """<style>
.viz{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--s1:#2a78d6;
--band:#2a78d6;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
@media (prefers-color-scheme:dark){.viz{--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;
--s1:#3987e5;--band:#3987e5}}
.bg{fill:var(--surface)} .t{fill:var(--ink);font-size:14px;font-weight:600} .st{fill:var(--ink2);font-size:12px}
.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums} .lab{fill:var(--ink2);font-size:12px}
.rowlab{fill:var(--ink2);font-size:12px} .pick{fill:var(--ink);font-size:12px;font-weight:600}
.grid{stroke:var(--grid);stroke-width:1} .axis{stroke:var(--axis);stroke-width:1}
.gate{stroke:var(--ink2);stroke-width:1} .band{fill:var(--band);fill-opacity:0.08}
.wh{stroke:var(--muted);stroke-width:2;stroke-linecap:round} .whp{stroke:var(--s1);stroke-width:2;stroke-linecap:round}
.dot{fill:var(--muted);stroke:var(--surface);stroke-width:2} .dotp{fill:var(--s1);stroke:var(--surface);stroke-width:2}
.hol{fill:var(--surface);stroke:var(--muted);stroke-width:2} .holp{fill:var(--surface);stroke:var(--s1);stroke-width:2}
</style>"""
W, ROW, TOP = 900, 27, 118
H = TOP + ROW * len(ROWS) + 70
LX = 175                                    # end of row labels
PANELS = [  # key, x0, width, (lo, hi), log, gate, title, ticks, fmt
    ("A", 190, 215, (1.0, 5.0), False, None, "(a) score A (lower is better)", [1, 2, 3, 4, 5], lambda v: "%g" % v),
    ("dps", 445, 190, (0.1, 1000.0), True, 1.0, "(b) max |dPS| (hPa, log)", [0.1, 1, 10, 100, 1000], lambda v: "%g" % v),
    ("neg", 675, 190, (0.0, 6.0), False, 1.0, "(c) max negative-water area (%)", [0, 1, 2, 4, 6], lambda v: "%g" % v)]


def X(p, v):
    _, x0, w, (lo, hi), log, *_ = p
    v = min(max(v, lo), hi)
    n = (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) if log else (v - lo) / (hi - lo)
    return x0 + w * n


body = ['<text class="t" x="24" y="30">Figure 3. The checkpoint choice: score A against the gates, 2045-2047 window</text>',
        '<text class="st" x="24" y="50">Filled marker: passes that panel\'s gate (b: |dPS| within 1 hPa; c: at most 1% negative area). '
        'In (a), filled = passes gates 1-3.</text>',
        '<text class="st" x="24" y="66">Whiskers in (a): range of the 8 rollouts\' A. Highlighted row: the pick, lowest A among gate 1-3 passers.</text>']
ybot = TOP + ROW * len(ROWS)
for i, (arm, *_r) in enumerate(ROWS):
    if arm == PICK:
        y = TOP + ROW * i
        body.append('<rect class="band" x="8" y="%.1f" width="%d" height="%d"/>' % (y, W - 16, ROW))
for p in PANELS:
    key, x0, w, (lo, hi), log, gate, title, ticks, fmt = p
    body.append('<text class="lab" x="%d" y="%d">%s</text>' % (x0, TOP - 14, title))
    for v in ticks:
        x = X(p, v)
        body.append('<line class="grid" x1="%.1f" x2="%.1f" y1="%d" y2="%d"/>' % (x, x, TOP, ybot))
        body.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">%s</text>' % (x, ybot + 16, fmt(v)))
    body.append('<line class="axis" x1="%d" x2="%d" y1="%d" y2="%d"/>' % (x0, x0 + w, ybot, ybot))
    if gate is not None:
        x = X(p, gate)
        body.append('<line class="gate" x1="%.1f" x2="%.1f" y1="%d" y2="%d"/>' % (x, x, TOP - 4, ybot))
        body.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">gate</text>' % (x, ybot + 30))
for i, (arm, label, *_r) in enumerate(ROWS):
    s = S[arm]
    y = TOP + ROW * i + ROW / 2
    pk = arm == PICK
    body.append('<text class="%s" x="%d" y="%.1f" text-anchor="end">%s</text>' % ("pick" if pk else "rowlab", LX, y + 4, label))
    pa, pb, pc = PANELS
    lo, hi = s["Am"][0], s["Am"][-1]
    body.append('<line class="%s" x1="%.1f" x2="%.1f" y1="%.1f" y2="%.1f"/>' % ("whp" if pk else "wh", X(pa, lo), X(pa, hi), y, y))
    if hi > pa[3][1]:   # a rollout's A beyond the axis: say so rather than clip silently
        body.append('<text class="tick" x="%.1f" y="%.1f">&gt;</text>' % (X(pa, hi) + 8, y + 4))
    for p, val, ok, txt in ((pa, s["A"], s["g123"], "A %.2f (rollouts %.2f to %s), gates 1-3 %s" % (
                                s["A"], lo, ("%.2f" % hi) if hi < 1e3 else "not finite", "pass" if s["g123"] else "fail")),
                            (pb, s["dps"], s["g3"], "max |dPS| %s hPa" % (("%.2f" % s["dps"]) if s["dps"] < 1e4 else "not finite")),
                            (pc, s["neg"], s["g2"], "max negative-water area %.1f%%, water gate %s" % (s["neg"], "pass" if s["g2"] else "fail"))):
        cls = ("dotp" if pk else "dot") if ok else ("holp" if pk else "hol")
        body.append('<circle class="%s" cx="%.1f" cy="%.1f" r="5"><title>%s: %s</title></circle>' % (cls, X(p, val), y, label, txt))
        if val > p[3][1]:
            body.append('<text class="tick" x="%.1f" y="%.1f">&gt;</text>' % (X(p, val) + 8, y + 4))
body.append('<text class="st" x="24" y="%d">Gate 4 (at most 3 drifting groups) is failed by every candidate; see Table 6.</text>' % (H - 14))
svg = ('<svg xmlns="http://www.w3.org/2000/svg" class="viz" viewBox="0 0 %d %d" width="%d" height="%d" role="img" '
       'aria-label="Figure 3: checkpoint choice">%s<rect class="bg" width="%d" height="%d"/>%s</svg>\n'
       % (W, H, W, H, STYLE, W, H, "".join(body)))
open(os.path.join(OUT, "fig3_candidates.svg"), "w").write(svg)
print("FIG3_OK rows=%d pick=%s" % (len(ROWS), PICK))
