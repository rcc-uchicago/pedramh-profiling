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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    rows, summ = load(args.tag)

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
