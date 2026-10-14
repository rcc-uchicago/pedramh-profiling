"""Figure 4 + Table 7: the 2 x 2 training x rollout precision cross (Result 15). Stdlib only.

Reads the scorer outputs (g1fs = existing cells, pxc = new cells) once and caches what it uses in
cross_cells.json, so the figure and table regenerate offline from this folder.
  python3 make_fig4.py <outdir>    -> <outdir>/fig4_precision_cross.svg, ./table7.md, ./cross_cells.json
"""
import json
import math
import os
import statistics as st
import sys

HERE = os.environ.get("MAKANI_RESULTS_WORK", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import precision_cross_analysis as pca  # noqa: E402  (copy of eval/pipeline-protocol 6645c247)

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..")
SCORES = "/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval/protocol_scores/"
CACHE = os.path.join(HERE, "cross_cells.json")
CELLS = pca.CELLS                     # (T, R) -> arm prefix

if not os.path.exists(CACHE):
    A = pca.member_scores(SCORES + "g1fs_channels.csv")
    A.update({k: v for k, v in pca.member_scores(SCORES + "pxc_channels.csv").items() if k not in A})
    bud = {}
    for f in ("g1fs_budget.txt", "pxc_budget.txt"):
        for arm, mem in pca.budgets(SCORES + f).items():
            bud.setdefault(arm, mem)
    keep = {"%s%02d" % (p, e) for p in CELLS.values() for e in pca.EPOCHS}
    json.dump({"A": {k: v for k, v in A.items() if k in keep}, "bud": {k: v for k, v in bud.items() if k in keep}},
              open(CACHE, "w"), indent=0)
C = json.load(open(CACHE))
A, BUD = C["A"], {k: {m: tuple(x) for m, x in v.items()} for k, v in C["bud"].items()}

# ------------------------------------------------------------------ Table 7 (effects)
eff = {
    "rollout fp32 − bf16, bf16-trained (R|T=bf16)": pca.effect(A, BUD, ("bf16", "fp32"), ("bf16", "bf16")),
    "rollout fp32 − bf16, fp32-trained (R|T=fp32)": pca.effect(A, BUD, ("fp32", "fp32"), ("fp32", "bf16")),
    "training fp32 − bf16, bf16 rollouts (T|R=bf16)": pca.effect(A, BUD, ("fp32", "bf16"), ("bf16", "bf16")),
    "training fp32 − bf16, fp32 rollouts (T|R=fp32)": pca.effect(A, BUD, ("fp32", "fp32"), ("bf16", "fp32")),
    "total: fp32/fp32 − bf16/bf16": pca.effect(A, BUD, ("fp32", "fp32"), ("bf16", "bf16")),
}
t = ["| effect (negative = fp32 better) | pooled, epochs 15/16/18/19 (mean ± SE) | p | rollouts | per epoch: 15 / 16 / 17 / 18 / 19 |",
     "|---|---|---|---|---|"]
for k, v in eff.items():
    pe = " / ".join(("%+.2f" % v["per_epoch"][e][0]) + ("" if v["per_epoch"][e][1] == 8 else " (n%d)" % v["per_epoch"][e][1])
                    if e in v["per_epoch"] else "—" for e in pca.EPOCHS)
    t.append("| %s | %+.3f ± %.3f | %.1e | %d | %s |" % (k.replace("|", "\\|"), v["mean"], v["se"], v["p"], v["n"], pe))
open(os.path.join(HERE, "table7.md"), "w").write("\n".join(t) + "\n")

# ------------------------------------------------------------------ Figure 4
def cell(T, R, e):
    arm = "%s%02d" % (CELLS[(T, R)], e)
    v = [x for m, x in A.get(arm, {}).items() if pca.healthy(BUD, arm, m)]
    return (st.median(v) if v else None), len(v)


STYLE = """<style>
.viz{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--s1:#2a78d6;--s2:#eb6834;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
@media (prefers-color-scheme:dark){.viz{--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;
--s1:#3987e5;--s2:#d95926}}
.bg{fill:var(--surface)} .t{fill:var(--ink);font-size:14px;font-weight:600} .st{fill:var(--ink2);font-size:12px}
.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums} .lab{fill:var(--ink2);font-size:12px}
.grid{stroke:var(--grid);stroke-width:1} .axis{stroke:var(--axis);stroke-width:1}
.ln{fill:none;stroke-width:2;stroke-linecap:round} .l1{stroke:var(--s1)} .l2{stroke:var(--s2)}
.f1{fill:var(--s1);stroke:var(--surface);stroke-width:2} .f2{fill:var(--s2);stroke:var(--surface);stroke-width:2}
.h1{fill:var(--surface);stroke:var(--s1);stroke-width:2} .h2{fill:var(--surface);stroke:var(--s2);stroke-width:2}
.blown{fill:var(--muted);font-size:11px}
</style>"""
W, H = 900, 440
X0, PW, GAP, TOP, PH = 78, 146, 18, 112, 230
LO, HI = 1.0, 30.0
SERIES = (("bf16", "1", "F-scratch (bf16-trained)"), ("fp32", "2", "G1 (fp32-trained)"))


def Y(v):
    v = min(max(v, LO), HI)
    return TOP + PH * (1 - (math.log10(v) - math.log10(LO)) / (math.log10(HI) - math.log10(LO)))


body = ['<text class="t" x="24" y="30">Figure 4. Training precision x rollout precision: both matter, and they interact</text>',
        '<text class="st" x="24" y="50">Score A (median over healthy rollouts, log scale) for bf16 and fp32 rollouts, one panel per EMA epoch.</text>',
        '<text class="st" x="24" y="66">Filled marker: 8 of 8 rollouts healthy; hollow: fewer. "none healthy" = no rollout reached 2049 sane.</text>']
for i, e in enumerate(pca.EPOCHS):
    x0 = X0 + i * (PW + GAP)
    xs = {"bf16": x0 + PW * 0.25, "fp32": x0 + PW * 0.75}
    for v in (1, 2, 3, 5, 10, 20, 30):
        y = Y(v)
        body.append('<line class="grid" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>' % (x0, x0 + PW, y, y))
        if i == 0:
            body.append('<text class="tick" x="%d" y="%.1f" text-anchor="end">%g</text>' % (x0 - 6, y + 4, v))
    body.append('<line class="axis" x1="%d" x2="%d" y1="%d" y2="%d"/>' % (x0, x0 + PW, TOP + PH, TOP + PH))
    for R, x in xs.items():
        body.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">%s</text>' % (x, TOP + PH + 16, R))
    body.append('<text class="lab" x="%.1f" y="%d" text-anchor="middle">epoch %d</text>' % (x0 + PW / 2, TOP - 12, e))
    for T, k, name in SERIES:
        pts = []
        for R in ("bf16", "fp32"):
            a, n = cell(T, R, e)
            if a is None:
                body.append('<text class="blown" x="%.1f" y="%d" text-anchor="middle">none healthy</text>' % (
                    xs[R], TOP + 14 if T == "fp32" else TOP + 28))
                continue
            pts.append((xs[R], Y(a)))
            cls = ("f%s" % k) if n == 8 else ("h%s" % k)
            body.append('<circle class="%s" cx="%.1f" cy="%.1f" r="5"><title>%s, epoch %d, %s rollout: A %.2f (%d of 8 healthy)</title></circle>'
                        % (cls, xs[R], Y(a), name, e, R, a, n))
        if len(pts) == 2:
            body.insert(len(body) - 2, '<path class="ln l%s" d="M%.1f,%.1f L%.1f,%.1f"/>' % (k, pts[0][0], pts[0][1], pts[1][0], pts[1][1]))
body.append('<text class="lab" x="%d" y="%d" text-anchor="middle">rollout precision</text>' % (X0 + (5 * PW + 4 * GAP) / 2, TOP + PH + 36))
ly = H - 24
for j, (T, k, name) in enumerate(SERIES):
    x = X0 + j * 260
    body.append('<line class="ln l%s" x1="%d" x2="%d" y1="%d" y2="%d"/>' % (k, x, x + 22, ly, ly))
    body.append('<circle class="f%s" cx="%d" cy="%d" r="4.5"/>' % (k, x + 11, ly))
    body.append('<text class="lab" x="%d" y="%d">%s</text>' % (x + 30, ly + 4, name))
svg = ('<svg xmlns="http://www.w3.org/2000/svg" class="viz" viewBox="0 0 %d %d" width="%d" height="%d" role="img" '
       'aria-label="Figure 4: precision cross">%s<rect class="bg" width="%d" height="%d"/>%s</svg>\n'
       % (W, H, W, H, STYLE, W, H, "".join(body)))
open(os.path.join(OUT, "fig4_precision_cross.svg"), "w").write(svg)
print("FIG4_OK")
