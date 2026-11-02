"""plot_decorrelation.py -- figure for decorrelation.py results (results/decorrelation_<tag>.csv + _summary.json).

  python plot_decorrelation.py [--tag full]   -> results/decorrelation_<tag>.png

Panels (one y-axis each): (a) C(tau) of theta_m16 and of the pair increment d = x(t+0.02) - x(t);
(b) 1 - C(tau) log-log with the small-lag power-law fit extended to the online sampling interval 2e-4;
(c) e-folding time per |k| band on the 256^2 grid. Vertical guides at 0.02/100, 0.02/10, 0.02.
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = {"theta": ("#2a78d6", "o", "θ (input / state)"), "incr": ("#eb6834", "s", "increment θ(t+0.02) − θ(t) (target)")}
GUIDES = [(2e-4, "Δt/100\n(online)"), (2e-3, "Δt/10"), (0.02, "Δt")]
PAIR_T = 0.02


def load(tag):
    rows = list(csv.DictReader(open(os.path.join(HERE, "results", f"decorrelation_{tag}.csv"))))
    summ = json.load(open(os.path.join(HERE, "results", f"decorrelation_{tag}_summary.json")))
    return rows, summ


def curve(rows, field, band="all"):
    """Merged curve: dense below 0.02, archive from 0.02 on (as decorrelation.py)."""
    pts = {}
    for r in rows:
        if r["field"] != field or r["band"] != band or int(r["lag_frames"]) == 0:
            continue
        t = float(r["lag_time"])
        if (r["source"] == "dense" and t < PAIR_T - 1e-12) or r["source"] == "archive":
            pts[t] = (float(r["mean"]), float(r["std"]))
    t = np.array(sorted(pts))
    return t, np.array([pts[x][0] for x in t]), np.array([pts[x][1] for x in t])


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def guides(ax, label_y):
    for x, lab in GUIDES:
        ax.axvline(x, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=0)
        if label_y is not None:
            ax.text(x * 1.08, label_y, lab, color=INK2, fontsize=7, va="top")


def plot_bands(tag, summ):
    """Multigrid summary (decorrelation.py --factors 1 ... 16): per |k| band, full 4096^2 grid (filled) vs the
    256^2 training view (hollow): (a) variance fraction, (b) C(0.02), (c) e-folding time."""
    full, coarse = summ["m1"], summ["m16"]
    bands = [b for b in full["theta"]["bandvar"] if (full["theta"]["bandvar"][b] or 0) > 1e-7
             or (full["incr"]["bandvar"][b] or 0) > 1e-7]
    xs = np.arange(len(bands))
    lim = next((i for i, b in enumerate(bands) if b.startswith("k128-")), None)   # first band beyond 256^2 Nyquist
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), facecolor=SURF)
    panels = [("bandvar", "share of variance in band", "(a) where the variance sits", True),
              ("bands_C_at_0.02", "correlation after one step C(0.02)", "(b) correlation after one step Δt = 0.02", False),
              ("bands_efold_time", "e-folding time (model time)", "(c) e-folding time by band", True)]
    for ax, (key, ylab, title, logy) in zip(axs, panels):
        style(ax)
        for f, (col, mk, lab) in SERIES.items():
            for grid, filled in ((full, True), (coarse, False)):
                d = grid[f][key]
                y = np.array([d.get(b) if d.get(b) is not None else np.nan for b in bands], dtype=float)
                kw = dict(color=col, marker=mk, ms=7, mew=1.5)
                kw |= dict(mfc=col, mec=SURF, lw=2) if filled else dict(mfc=SURF, mec=col, lw=0)
                name = f"{lab.split(' (')[0]} — {'4096² full grid' if filled else '256² training view'}"
                ax.plot(xs + (0 if filled else 0.12), y, label=name, **kw)
        if lim is not None:
            ax.axvline(lim - 0.5, color=INK2, lw=1)
            ax.text(lim - 0.45, 0.98, " beyond 256²\n grid limit", transform=ax.get_xaxis_transform(), color=INK2,
                    fontsize=7, va="top")
        if key == "bands_C_at_0.02":
            ax.axhline(0, color=AXIS, lw=0.8)
            ax.axhline(-0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
            ax.text(0, -0.53, "−0.5 = differencing uncorrelated frames", color=INK2, fontsize=7, va="top")
            ax.set_ylim(-0.65, 1.08)
        if key == "bands_efold_time":
            for x, lab in GUIDES:
                ax.axhline(x, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=0)
                ax.text(len(bands) - 0.6, x * 1.12, lab.replace("\n", " "), color=INK2, fontsize=7, ha="right")
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(xs, [b.replace("k", "") for b in bands], rotation=35, fontsize=8)
        ax.set_xlabel("|k| band (octaves)", color=INK2, fontsize=9)
        ax.set_ylabel(ylab, color=INK2, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    axs[0].legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="lower left")
    fig.suptitle("SQG θ: per-band decorrelation, full 4096² grid (filled) vs 256² point-subsampled training view "
                 "(hollow); the 1024² view matches the full grid up to its own limit", color=INK, fontsize=11, x=0.01,
                 ha="left")
    fig.tight_layout()
    out = os.path.join(HERE, "results", f"decorrelation_{tag}_bands.png")
    fig.savefig(out, dpi=130, facecolor=SURF)
    print(f"PLOT_OK {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    rows, summ = load(args.tag)
    if "m1" in summ:
        return plot_bands(args.tag, summ)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURF)
    # (a) C(tau)
    ax = axs[0]
    style(ax)
    for f, (col, mk, lab) in SERIES.items():
        t, c, s = curve(rows, f)
        ax.plot(t, c, color=col, lw=2, marker=mk, ms=4.5, mec=SURF, mew=1, label=lab)
        ax.text(t[-1], c[-1], "  θ" if f == "theta" else "  increment", color=INK2, fontsize=8, va="center")
    ax.set_xscale("log")
    ax.axhline(0, color=AXIS, lw=0.8)
    guides(ax, 1.08)
    ax.set_ylim(-0.25, 1.1)
    ax.set_xlim(1.2e-4, 30)
    ax.set_xlabel("lag τ (model time)", color=INK2, fontsize=9)
    ax.set_ylabel("spatial correlation C(τ), 256² field", color=INK2, fontsize=9)
    ax.set_title("(a) correlation vs lag", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")

    # (b) 1 - C, log-log, with fit extrapolation
    ax = axs[1]
    style(ax)
    for f, (col, mk, lab) in SERIES.items():
        t, c, _ = curve(rows, f)
        sel = (t <= 0.5) & (c < 1)
        ax.plot(t[sel], 1 - c[sel], color=col, lw=2, marker=mk, ms=4.5, mec=SURF, mew=1, label=lab)
        fit = summ[f]["small_lag_fit"]
        if fit:
            tt = np.logspace(np.log10(2e-4), np.log10(0.0025), 20)
            ax.plot(tt, fit["a"] * tt ** fit["p"], color=col, lw=1.5, ls=(0, (2, 2)))
            v = fit["one_minus_C_at_2e-4"]
            ax.plot([2e-4], [v], marker=mk, ms=8, mfc=SURF, mec=col, mew=1.5)
            ax.text(2.4e-4, v, f" 1−C ≈ {v:.1e}\n (extrapolated, p={fit['p']:.2f})", color=INK2, fontsize=7, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    guides(ax, None)
    ax.set_xlim(1.2e-4, 0.7)
    ax.set_xlabel("lag τ (model time)", color=INK2, fontsize=9)
    ax.set_ylabel("1 − C(τ)  (new, uncorrelated part)", color=INK2, fontsize=9)
    ax.set_title("(b) small lags: measured ≥ 0.0025, dashed = power-law fit", color=INK, fontsize=10, loc="left")

    # (c) e-folding time per band
    ax = axs[2]
    style(ax)
    bands = [b for b in summ["theta"]["bands_efold_time"] if b != "all"]
    xs = np.arange(len(bands))
    for k, (f, (col, mk, lab)) in enumerate(SERIES.items()):
        e = summ[f]["bands_efold_time"]
        y = np.array([e[b] if e[b] is not None else np.nan for b in bands], dtype=float)
        ax.plot(xs + (k - 0.5) * 0.12, y, color=col, lw=0, marker=mk, ms=8, mec=SURF, mew=1.5, label=lab)
    for x, lab in GUIDES:
        ax.axhline(x, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=0)
        ax.text(len(bands) - 0.55, x * 1.1, lab.replace("\n", " "), color=INK2, fontsize=7, ha="right")
    ax.set_yscale("log")
    ax.set_xticks(xs, [b.replace("k", "|k| ") for b in bands], rotation=30, fontsize=8)
    ax.set_ylabel("e-folding time (model time)", color=INK2, fontsize=9)
    ax.set_title("(c) e-folding time by wavenumber band (256² grid)", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")

    m = summ["members"]
    fig.suptitle(f"SQG θ[::16, ::16] temporal decorrelation — dense members {m['dense']}, archive members {m['archive']}",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    out = os.path.join(HERE, "results", f"decorrelation_{args.tag}.png")
    fig.savefig(out, dpi=130, facecolor=SURF)
    print(f"PLOT_OK {out}")


if __name__ == "__main__":
    main()
