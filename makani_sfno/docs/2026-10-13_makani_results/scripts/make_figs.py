"""Draw the two result figures as static SVG (stdlib only; no matplotlib on the login node).

fig1: validation loss of A and P128 against optimizer updates and against node-hours.
fig2: learning-rate schedule, 5-yr protocol score A, and EMA validation loss by epoch for the
      three single-step arms (F-scratch EMA, G1 EMA, G1-dry EMA).
Colors: the dataviz skill's validated reference palette, first three categorical slots.
"""
import json
import math
import os
import sys

T = os.environ.get("MAKANI_RESULTS_WORK", os.path.dirname(os.path.abspath(__file__))) + "/"
OUT = sys.argv[1]
R = json.load(open(T + "epochs/all.json"))
ARMS = json.load(open(T + "arms.json"))
SURV = {d["tag"]: d for d in json.load(open(T + "protocol_survival.json"))}

STYLE = """<style>
.viz{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--band:#0b0b0b;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
@media (prefers-color-scheme:dark){.viz{--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--band:#ffffff}}
.bg{fill:var(--surface)} .t{fill:var(--ink);font-size:14px;font-weight:600} .st{fill:var(--ink2);font-size:12px}
.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums} .lab{fill:var(--ink2);font-size:12px}
.grid{stroke:var(--grid);stroke-width:1} .axis{stroke:var(--axis);stroke-width:1}
.band{fill:var(--band);fill-opacity:0.05}
.l1{stroke:var(--s1)} .l2{stroke:var(--s2)} .l3{stroke:var(--s3)} .lr{stroke:var(--ink2)}
.ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.f1{fill:var(--s1)} .f2{fill:var(--s2)} .f3{fill:var(--s3)} .ring{stroke:var(--surface);stroke-width:2}
.hollow{fill:var(--surface);stroke-width:2}
</style>"""


def fmt(v):
    if v >= 1000:
        return "{:,.0f}".format(v)
    if v >= 1:
        return ("%.0f" % v) if abs(v - round(v)) < 1e-9 else ("%g" % v)
    return "%g" % v


class Panel:
    def __init__(self, x0, y0, w, h, xr, yr, xlog=False, ylog=False):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.xr, self.yr, self.xlog, self.ylog = xr, yr, xlog, ylog

    def _n(self, v, r, log):
        if log:
            return (math.log10(v) - math.log10(r[0])) / (math.log10(r[1]) - math.log10(r[0]))
        return (v - r[0]) / (r[1] - r[0])

    def X(self, v):
        return self.x0 + self.w * self._n(v, self.xr, self.xlog)

    def Y(self, v):
        return self.y0 + self.h * (1 - self._n(v, self.yr, self.ylog))

    def frame(self, xticks, yticks, xlabel, ylabel, yfmt=fmt, xfmt=fmt):
        s = []
        for v in yticks:
            y = self.Y(v)
            s.append('<line class="grid" x1="%.1f" x2="%.1f" y1="%.1f" y2="%.1f"/>' % (self.x0, self.x0 + self.w, y, y))
            s.append('<text class="tick" x="%.1f" y="%.1f" text-anchor="end">%s</text>' % (self.x0 - 6, y + 4, yfmt(v)))
        s.append('<line class="axis" x1="%.1f" x2="%.1f" y1="%.1f" y2="%.1f"/>' % (self.x0, self.x0 + self.w, self.y0 + self.h, self.y0 + self.h))
        for v in xticks:
            x = self.X(v)
            s.append('<line class="axis" x1="%.1f" x2="%.1f" y1="%.1f" y2="%.1f"/>' % (x, x, self.y0 + self.h, self.y0 + self.h + 4))
            s.append('<text class="tick" x="%.1f" y="%.1f" text-anchor="middle">%s</text>' % (x, self.y0 + self.h + 17, xfmt(v)))
        if xlabel:
            s.append('<text class="lab" x="%.1f" y="%.1f" text-anchor="middle">%s</text>' % (self.x0 + self.w / 2, self.y0 + self.h + 36, xlabel))
        s.append('<text class="lab" x="%.1f" y="%.1f">%s</text>' % (self.x0, self.y0 - 8, ylabel))
        return s

    def line(self, pts, cls):
        if len(pts) < 2:
            return ""
        d = "M" + " L".join("%.1f,%.1f" % (self.X(x), self.Y(y)) for x, y in pts)
        return '<path class="ln %s" d="%s"/>' % (cls, d)


def svg(w, h, body, title):
    return ('<svg xmlns="http://www.w3.org/2000/svg" class="viz" viewBox="0 0 %d %d" width="%d" height="%d" role="img" '
            'aria-label="%s">%s<rect class="bg" width="%d" height="%d"/>%s</svg>\n' % (w, h, w, h, title, STYLE, w, h, "".join(body)))


def ser(k):
    return {int(e): v for e, v in R[k].items()}


# ---------------- fig 1 ----------------
A, P = ser("A"), ser("P128")


def cum(d, steps_key, nodes):
    up, nh, out = 0, 0.0, []
    for e in sorted(d):
        up += d[e]["steps"]
        nh += d[e]["t"] * nodes / 3600
        out.append((e, up, nh, d[e]["va"]))
    return out


ca, cp = cum(A, "steps", 1), cum(P, "steps", 128)
W, H = 900, 445
body = ['<text class="t" x="24" y="30">Figure 1. The 128-node run and A make the same progress per optimizer update</text>',
        '<text class="st" x="24" y="50">Validation loss on the same 512-sample validation set (101 channels).</text>',
        '<text class="st" x="24" y="66">Left: against optimizer updates. Right: against compute in node-hours.</text>']
yr = (0.0125, 0.12)
yt = [0.0125, 0.015, 0.02, 0.03, 0.05, 0.1]
pl = Panel(80, 105, 340, 250, (80, 4e5), yr, True, True)
pr = Panel(520, 105, 340, 250, (0.1, 400), yr, True, True)
body += pl.frame([100, 1000, 1e4, 1e5], yt, "optimizer updates (count, log scale)", "validation loss (log scale)",
                 yfmt=lambda v: "%g" % v)
body += pr.frame([0.1, 1, 10, 100], yt, "compute (node-hours, log scale)", "validation loss (log scale)",
                 yfmt=lambda v: "%g" % v)
for p, xi in ((pl, 1), (pr, 2)):
    body.append(p.line([(r[xi], r[3]) for r in ca], "l1"))
    body.append(p.line([(r[xi], r[3]) for r in cp], "l2"))
    for r, cls, name in ((ca[-1], "f1", "A, epoch 243"), (cp[-1], "f2", "128-node run, epoch 100")):
        body.append('<circle class="%s ring" cx="%.1f" cy="%.1f" r="5"><title>%s: %s %s, validation loss %.6f</title></circle>' % (
            cls, p.X(r[xi]), p.Y(r[3]), name, fmt(r[xi]) if xi == 1 else "%.1f" % r[xi], "updates" if xi == 1 else "node-hours", r[3]))
# equal-updates marker (A epoch 7 = 9,576 updates, the first epoch past the 128-node run's 8,500)
a7 = [r for r in ca if r[1] >= 8500][0]
body.append('<circle class="f1 ring" cx="%.1f" cy="%.1f" r="5"><title>A, epoch %d: %s updates, validation loss %.6f</title></circle>' % (
    pl.X(a7[1]), pl.Y(a7[3]), a7[0], fmt(a7[1]), a7[3]))
body.append('<text class="st" x="%.1f" y="%.1f">128-node e100 and A e7</text>' % (pl.X(a7[1]) + 12, pl.Y(a7[3]) - 30))
body.append('<text class="st" x="%.1f" y="%.1f">8,500 vs %s updates</text>' % (pl.X(a7[1]) + 12, pl.Y(a7[3]) - 15, fmt(a7[1])))
body.append('<text class="st" x="%.1f" y="%.1f" text-anchor="end">A e243</text>' % (pl.X(ca[-1][1]) - 4, pl.Y(ca[-1][3]) - 26))
a8 = [r for r in ca if r[3] <= cp[-1][3]][0]
body.append('<circle class="f1 ring" cx="%.1f" cy="%.1f" r="5"><title>A, epoch %d: %.2f node-hours, validation loss %.6f</title></circle>' % (
    pr.X(a8[2]), pr.Y(a8[3]), a8[0], a8[2], a8[3]))
body.append('<text class="st" x="%.1f" y="%.1f" text-anchor="end">A e8: %.1f node-h</text>' % (pr.X(a8[2]) - 9, pr.Y(a8[3]) + 16, a8[2]))
body.append('<text class="st" x="%.1f" y="%.1f" text-anchor="end">%.0f node-h</text>' % (pr.X(cp[-1][2]) - 6, pr.Y(cp[-1][3]) + 20, cp[-1][2]))
# legend (2 series)
lx, ly = 80, 412
for i, (cls, name) in enumerate((("l1", "A: 1 node, global batch 32, 243 epochs"), ("l2", "128-node run: global batch 512, 100 epochs"))):
    x = lx + i * 380
    body.append('<line class="ln %s" x1="%d" x2="%d" y1="%d" y2="%d"/>' % (cls, x, x + 22, ly, ly))
    body.append('<text class="lab" x="%d" y="%d">%s</text>' % (x + 30, ly + 4, name))
open(os.path.join(OUT, "fig1_batch_and_cost.svg"), "w").write(svg(W, H, body, "Figure 1: validation loss vs updates and node-hours"))

# ---------------- fig 2 ----------------
def lr_at(e, peak=2e-3, lo=1e-6, warm=3, t0=20, start=2e-5):
    if e <= warm:
        return start + (peak - start) * (e - 1) / warm
    t = (e - warm - 1) % t0
    return lo + (peak - lo) * (1 + math.cos(math.pi * t / t0)) / 2


W, H = 900, 830
E0, E1 = 13.5, 63.5
body = ['<text class="t" x="24" y="30">Figure 2. Single-step checkpoints late in each learning-rate cycle fail the 5-year rollout</text>',
        '<text class="st" x="24" y="50">Shaded: the last 4 epochs of each warm-restart cycle (epochs 20-23, 40-43, 60-63).</text>',
        '<text class="st" x="24" y="66">Filled marker: all 8 five-year rollouts stayed finite. Hollow marker: fewer than 8.</text>']
pa = Panel(90, 110, 760, 130, (E0, E1), (0, 2.2e-3))
pb = Panel(90, 300, 760, 270, (E0, E1), (1.2, 400), False, True)
pc = Panel(90, 625, 760, 130, (E0, E1), (0.0130, 0.0160))
xt = [15, 20, 23, 30, 40, 43, 50, 60, 63]
for p in (pa, pb, pc):
    for s, e in ((19.5, 23.5), (39.5, 43.5), (59.5, 63.5)):
        body.append('<rect class="band" x="%.1f" y="%.1f" width="%.1f" height="%.1f"/>' % (p.X(s), p.y0, p.X(e) - p.X(s), p.h))
body += pa.frame(xt, [0, 1e-3, 2e-3], None, "(a) learning rate from the schedule: warm-restart cosine, peak 2e-3, 20-epoch period",
                 yfmt=lambda v: "0" if v == 0 else "%.0e" % v)
body.append(pa.line([(e, lr_at(e)) for e in range(14, 64)], "lr"))
body += pb.frame(xt, [1.5, 2, 3, 5, 10, 30, 100], None,
                 "(b) 5-year protocol score A, log scale (lower is better; values above 400 drawn at the top line)")
body.append('<text class="tick" x="%.1f" y="%.1f" text-anchor="end">&gt;400</text>' % (pb.x0 - 6, pb.y0 + 4))
series = (("FSE", "Fscratch", "1", "F-scratch EMA (bf16)"), ("G1E", "G1", "2", "G1 EMA (fp32 F-scratch rerun)"),
          ("G1D", "G1D", "3", "G1-dry EMA (fp32 + trained dry-air)"))
for pre, run, k, name in series:
    pts = []
    for tag, a in ARMS.items():
        if tag.startswith(pre) and tag[len(pre):].isdigit():
            e = int(tag[len(pre):])
            if pre == "FSE" and e > 43:
                continue
            pts.append((e, a["A"], SURV[tag]["complete"]))
    pts.sort()
    runs, cur = [], []
    for e, a, c in pts:
        if cur and e != cur[-1][0] + 1:
            runs.append(cur)
            cur = []
        cur.append((e, min(a, 400.0)))
    runs.append(cur)
    for r in runs:
        body.append(pb.line(r, "l" + k))
    for e, a, c in pts:
        y = min(a, 400.0)
        cls = ("f%s ring" % k) if c == 8 else ("hollow l%s" % k)
        body.append('<circle class="%s" cx="%.1f" cy="%.1f" r="4.5"><title>%s epoch %d: A %s, %d of 8 rollouts finite</title></circle>' % (
            cls, pb.X(e), pb.Y(y), name, e, ("%.2f" % a) if a < 1e4 else "%.1e" % a, c))
    d = ser(run)
    body.append(pc.line([(e, d[e]["ema"]) for e in sorted(d) if e >= 14 and d[e]["ema"] is not None and (pre != "FSE" or e <= 43)], "l" + k))
body += pc.frame(xt, [0.013, 0.014, 0.015, 0.016], "epoch",
                 "(c) validation loss of the EMA weights (99 channels)",
                 yfmt=lambda v: "%.3f" % v)
ly = 805
for i, (pre, run, k, name) in enumerate(series):
    x = (90, 300, 560)[i]
    body.append('<circle class="f%s ring" cx="%d" cy="%d" r="4.5"/>' % (k, x + 6, ly))
    body.append('<text class="lab" x="%d" y="%d">%s</text>' % (x + 18, ly + 4, name))
open(os.path.join(OUT, "fig2_cycle_position.svg"), "w").write(svg(W, H, body, "Figure 2: learning-rate cycle position vs 5-year rollout outcome"))
print("wrote", os.listdir(OUT))
