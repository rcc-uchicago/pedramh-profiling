"""Five-year figures for G4D16/36/43/56/63: global-mean time series, bias by year, zonal-mean Hovmoller, annual
bias maps. Reads fiveyear_<ARM>.npz + fiveyear_truth.npz (extract_5yr.py); writes PNGs + summary_5yr.csv.

Uncertainty, by figure (all treat E3SM as one realisation and the eight runs' spread as internal variability,
the test of the 2045-2047 bias maps):
  * time series: the eight runs drawn individually (their spread is the uncertainty);
  * bias by year: 95% interval bias +- t_{0.975,7} * s * sqrt(1/8 + 1); filled marker = interval excludes 0;
  * Hovmoller and annual maps: dots where the bias exceeds internal variability (same t, Benjamini-Hochberg FDR
    10% over the panel's cells); annual maps also give a bootstrap CI over the runs of the global RMS bias,
    re-centred on the estimate (see fig_annual_maps: the raw percentile interval is shifted up by resampling).
The selection window 2045-2047 (used to rank checkpoints) is shaded; 2048-2049 were never used for selection.
PASS = FIVEYEAR_PLOT_OK
"""
import argparse
import csv
import os
import sys
import textwrap

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAPS = os.path.join(os.path.dirname(HERE), "maps")
sys.path.insert(0, HERE)
sys.path.insert(0, MAPS)
import fiveyear_lib as fl  # noqa: E402
import maps_lib as ml  # noqa: E402
import plot_maps as pm  # noqa: E402  (styles, map drawing, s3; importing draws nothing)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from scipy import stats  # noqa: E402

ARMS = ("G4D16", "G4D36", "G4D43", "G4D56", "G4D63")
# G4D16 keeps its orange from the other G4D figures (the current pick); the longer-trained checkpoints are one
# blue ramp, light -> dark with training length.
COLOR = {"G4D16": "#eb6834", "G4D36": "#6da7ec", "G4D43": "#3987e5", "G4D56": "#256abf", "G4D63": "#0d366b"}
VINFO = {k: (name, unit, fac) for k, name, unit, fac in fl.SERIES_VARS}
T975 = stats.t.ppf(0.975, 7)
RNG = np.random.default_rng(20261103)
ROWS = []
FIRST_PLOTTED = fl.MONTHS.index((2044, 11))      # Oct 2044 is partial for every run (starts Oct 1-29)


def month_x():
    return np.array([y + (m - 0.5) / 12.0 for y, m in fl.MONTHS])


def shade_window(ax, label=True, centred=False):
    """Selection window 2045-2047: on a decimal-year axis it spans 2045.0-2048.0; on an axis of integer year
    markers (centred=True) it spans the three markers, 2044.5-2047.5."""
    lo, hi = fl.SELECTION_WINDOW[0], fl.SELECTION_WINDOW[1] + 1
    if centred:
        lo, hi = lo - 0.5, hi - 0.5
    ax.axvspan(lo, hi, color="#eeede9", lw=0, zorder=0)
    if label:
        ax.text(lo + 0.05, 0.98, "selection window", transform=ax.get_xaxis_transform(),
                fontsize=6.5, color=pm.INK2, va="top")


def style(ax):
    ax.grid(True, color=pm.GRID, lw=0.5)
    ax.tick_params(labelsize=7, colors=pm.INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fine_norm(values):
    """Diverging levels for bias fields whose extremes sit on a few polar rows: 12 steps up to the 95th percentile
    of |bias|, beyond which the colourbar's arrows saturate. (The 2045-2047 bias maps' 98th-percentile, 6-step scale
    put most of the globe in two pale bins here.)"""
    vmax = float(np.nanpercentile(np.abs(values), 95))
    lev = MaxNLocator(nbins=12, symmetric=True).tick_values(-vmax, vmax)
    return mcolors.BoundaryNorm(lev, pm.DIV.N, extend="both"), lev


def unit_plain(u):
    return u.replace("$^{-1}$", "^-1").replace("$^{-2}$", "^-2")


def series(z, key, what):
    i = list(z["channels"]).index(key)
    return z[what][..., i].astype(np.float64) * VINFO[key][2]


def row(**kw):
    base = dict(figure="", variable="", model="", year="", window="", metric="", value="", ci_lo="", ci_hi="",
                unit="", n="", sig_area_pct="", significant="")
    base.update(kw)
    ROWS.append(base)


# ------------------------------------------------------------------------------------------------ time series
def fig_ts_arm(Z, T, arm, out):
    x = month_x()[FIRST_PLOTTED:]
    fig, axs = plt.subplots(4, 2, figsize=(13, 12.5))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.05, hspace=0.55, wspace=0.18)
    for ax, key in zip(axs.flat, fl.SERIES_KEYS):
        name, unit, fac = VINFO[key]
        mem = series(Z[arm], key, "gm_month")[:, FIRST_PLOTTED:]
        tru = series(T, key, "gm_month")[FIRST_PLOTTED:]
        shade_window(ax)
        for k in range(mem.shape[0]):
            ax.plot(x, mem[k], color=COLOR[arm], lw=0.6, alpha=0.35)
        ax.plot(x, mem.mean(axis=0), color=COLOR[arm], lw=1.7, label="%s, mean of %d runs" % (arm, mem.shape[0]))
        ax.plot(x, tru, color=pm.INK, lw=1.2, label="E3SM")
        ax.set_title(name, fontsize=9, color=pm.INK, loc="left")
        ax.set_ylabel("Global mean (%s)" % unit, fontsize=8, color=pm.INK2)
        ax.set_xlabel("Year", fontsize=8, color=pm.INK2)
        ax.set_xlim(x[0] - 0.05, x[-1] + 0.05)
        style(ax)
    leg = axs.flat[-1]
    leg.axis("off")
    h = [plt.Line2D([], [], color=COLOR[arm], lw=0.6, alpha=0.5), plt.Line2D([], [], color=COLOR[arm], lw=1.7),
         plt.Line2D([], [], color=pm.INK, lw=1.2)]
    leg.legend(h, ["each of the 8 runs", "mean of the 8 runs", "E3SM (the target)"], loc="center left",
               fontsize=9, frameon=False)
    fig.suptitle("%s: monthly global (area-weighted) means of the eight five-year runs and E3SM, Nov 2044 - Dec 2049"
                 "\nRuns start Oct 1, 5, ..., 29 2044 and are free-running; shaded = selection window 2045-2047."
                 % pm.LABEL[arm], fontsize=10.5, color=pm.INK, x=0.07, ha="left")
    pm.finish(fig, os.path.join(out, "ts_global_%s.png" % arm))


def fig_ts_bias_all(Z, T, out):
    x = month_x()[FIRST_PLOTTED:]
    arms = [a for a in ARMS if a in Z]
    fig, axs = plt.subplots(4, 2, figsize=(13, 12.5))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.05, hspace=0.55, wspace=0.18)
    for ax, key in zip(axs.flat, fl.SERIES_KEYS):
        name, unit, fac = VINFO[key]
        tru = series(T, key, "gm_month")[FIRST_PLOTTED:]
        shade_window(ax)
        ax.axhline(0, color=pm.INK2, lw=0.7)
        for arm in arms:
            b = series(Z[arm], key, "gm_month")[:, FIRST_PLOTTED:].mean(axis=0) - tru
            ax.plot(x, b, color=COLOR[arm], lw=1.2, label=pm.LABEL[arm])
        ax.set_title(name, fontsize=9, color=pm.INK, loc="left")
        ax.set_ylabel("Bias, model - E3SM (%s)" % unit, fontsize=8, color=pm.INK2)
        ax.set_xlabel("Year", fontsize=8, color=pm.INK2)
        ax.set_xlim(x[0] - 0.05, x[-1] + 0.05)
        style(ax)
    leg = axs.flat[-1]
    leg.axis("off")
    leg.legend([plt.Line2D([], [], color=COLOR[a], lw=1.6) for a in arms], [pm.LABEL[a] for a in arms],
               loc="center left", fontsize=9, frameon=False)
    fig.suptitle("Monthly global-mean bias of each checkpoint (mean of its eight five-year runs minus E3SM), "
                 "Nov 2044 - Dec 2049\nShaded = selection window 2045-2047. Year-by-year intervals: "
                 "bias_by_year.png.", fontsize=10.5, color=pm.INK, x=0.07, ha="left")
    pm.finish(fig, os.path.join(out, "ts_bias_all.png"))


# ------------------------------------------------------------------------------------------------ bias by year
def fig_bias_by_year(Z, T, out):
    arms = [a for a in ARMS if a in Z]
    years = np.array(fl.YEARS)
    offs = np.linspace(-0.26, 0.26, len(arms))
    fig, axs = plt.subplots(4, 2, figsize=(13, 12.5))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.05, hspace=0.55, wspace=0.18)
    for ax, key in zip(axs.flat, fl.SERIES_KEYS):
        name, unit, fac = VINFO[key]
        tru = series(T, key, "gm_year")                                   # (5,)
        shade_window(ax, centred=True)
        ax.axhline(0, color=pm.INK2, lw=0.7)
        for arm, o in zip(arms, offs):
            mem = series(Z[arm], key, "gm_year")                          # (n, 5)
            bias, t, sds, dof = fl.bias_stats(mem, tru)
            half = stats.t.ppf(0.975, dof) * sds
            sig = np.abs(bias) > half
            ax.errorbar(years + o, bias, yerr=half, fmt="none", ecolor=COLOR[arm], elinewidth=1.0, capsize=2)
            ax.scatter(years + o, bias, s=22, zorder=3, edgecolors=COLOR[arm], linewidths=1.1,
                       facecolors=[COLOR[arm] if s_ else pm.SURFACE for s_ in sig])
            for y, b, h, s_ in zip(years, bias, half, sig):
                row(figure="bias_by_year", variable=key, model=arm, year=int(y),
                    metric="annual global-mean bias, mean of runs minus E3SM", value=b, ci_lo=b - h, ci_hi=b + h,
                    unit=unit_plain(unit), n=mem.shape[0], significant=int(s_))
            # drift: per-run OLS slope of the annual bias over 2045-2049; mean and 95% interval over runs, inflated
            # by sqrt(1/n + 1) like every other test here -- E3SM's own year-to-year noise puts a slope into the
            # single truth series that is common to all runs and invisible to their spread
            slopes = np.array([np.polyfit(years, m_ - tru, 1)[0] for m_ in mem])
            ns = len(slopes)
            sm, sh = slopes.mean(), stats.t.ppf(0.975, ns - 1) * slopes.std(ddof=1) * np.sqrt(1.0 / ns + 1.0)
            row(figure="bias_by_year", variable=key, model=arm, year="2045-2049",
                metric="linear trend of the annual global-mean bias (per year); 95% interval inflated sqrt(1/n+1)", value=sm, ci_lo=sm - sh,
                ci_hi=sm + sh, unit=unit_plain(unit) + "/yr", n=len(slopes))
        ax.set_xticks(years)
        ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
        ax.set_title(name, fontsize=9, color=pm.INK, loc="left")
        ax.set_ylabel("Annual-mean bias, model - E3SM (%s)" % unit, fontsize=8, color=pm.INK2)
        ax.set_xlabel("Year", fontsize=8, color=pm.INK2)
        style(ax)
    leg = axs.flat[-1]
    leg.axis("off")
    h = [plt.Line2D([], [], color=COLOR[a], marker="o", lw=0, ms=5) for a in arms]
    h += [plt.Line2D([], [], color=pm.INK2, marker="o", lw=0, ms=5, mfc=pm.INK2),
          plt.Line2D([], [], color=pm.INK2, marker="o", lw=0, ms=5, mfc=pm.SURFACE)]
    leg.legend(h, [pm.LABEL[a] for a in arms] + ["filled: interval excludes 0", "open: within internal variability"],
               loc="center left", fontsize=9, frameon=False)
    fig.suptitle("Annual-mean global bias by year, 2045-2049: mean of eight five-year runs minus E3SM\n"
                 "Bars: 95% interval, bias +- t(0.975, 7) x s x sqrt(1/8 + 1), with the run spread s as internal "
                 "variability and E3SM counted\nas one more realisation. Shaded = selection window 2045-2047.",
                 fontsize=10.5, color=pm.INK, x=0.07, ha="left")
    pm.finish(fig, os.path.join(out, "bias_by_year.png"))


# ------------------------------------------------------------------------------------------------ Hovmoller
def fig_hovmoller(Z, T, key, out):
    arms = [a for a in ARMS if a in Z]
    name, unit, fac = VINFO[key]
    zi = list(T["zonal_keys"]).index(key)
    lat = T["lat"]
    w = ml.lat_weights(lat)
    x = month_x()[FIRST_PLOTTED:]
    tru = T["zm_month"][FIRST_PLOTTED:, zi].astype(np.float64).T * fac            # (H, M)
    cells = []
    for arm in arms:
        mem = Z[arm]["zm_month"][:, FIRST_PLOTTED:, zi].astype(np.float64).transpose(0, 2, 1) * fac   # (n, H, M)
        bias, t, _, dof = fl.bias_stats(mem, tru)
        sig = ml.fdr_mask(2 * stats.t.sf(np.abs(t), dof), pm.ALPHA_FDR)
        rms = float(np.sqrt((w[:, None] * bias ** 2).sum(axis=0).mean()))
        cells.append((arm, bias, sig, rms, mem.shape[0]))
    norm, lev = fine_norm(np.stack([c[1] for c in cells]))
    fig, axs = plt.subplots(len(arms), 1, figsize=(12, 2.3 * len(arms) + 1.4), sharex=True, squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.90, top=0.89, bottom=0.06, hspace=0.42)
    for ax, (arm, bias, sig, rms, n) in zip(axs[:, 0], cells):
        order = np.argsort(lat)
        im = ax.pcolormesh(np.append(x - 1 / 24, x[-1] + 1 / 24), np.append(np.sort(lat) - 0.5, np.sort(lat)[-1] + 0.5),
                           bias[order], cmap=pm.DIV, norm=norm, shading="flat")
        jj, ii = np.nonzero(sig[order][::3, :])
        ax.plot(x[ii], np.sort(lat)[::3][jj], ".", ms=1.3, color=pm.INK, alpha=0.7, mew=0)
        shade_edges = [fl.SELECTION_WINDOW[0], fl.SELECTION_WINDOW[1] + 1]
        for e_ in shade_edges:
            ax.axvline(e_, color=pm.INK, lw=0.8, ls="--")
        area = 100.0 * float(sig.mean())
        ax.set_title("%s: zonal-mean %s bias; RMS %s %s (area-weighted, over all months); %.0f%% of cells "
                     "significant; %d runs" % (pm.LABEL[arm], key, pm.s3(rms), unit, area, n),
                     fontsize=7.5, color=pm.INK, loc="left")
        ax.set_yticks([-60, -30, 0, 30, 60])
        ax.set_yticklabels(["60°S", "30°S", "0°", "30°N", "60°N"])
        ax.set_ylabel("Latitude", fontsize=8, color=pm.INK2)
        ax.tick_params(labelsize=7, colors=pm.INK2)
        row(figure="hovmoller_%s" % key, variable=key, model=arm, window="2044-11 to 2049-12",
            metric="area-weighted RMS of zonal-mean bias over latitudes and months", value=rms,
            unit=unit_plain(unit), n=n, sig_area_pct=area)
    axs[-1, 0].set_xlabel("Year (dashed lines: selection window 2045-2047)", fontsize=8, color=pm.INK2)
    cax = fig.add_axes([0.915, 0.06, 0.012, 0.84])
    cb = fig.colorbar(im, cax=cax, ticks=lev, extend="both")
    cb.set_label("%s zonal-mean bias, model - E3SM (%s)" % (key, unit), color=pm.INK, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=pm.INK2)
    fig.suptitle("%s: zonal-mean bias by month, mean of eight five-year runs minus E3SM, Nov 2044 - Dec 2049\n"
                 "Dots: bias larger than internal variability at a 10%% false discovery rate (t-test against the run "
                 "spread,\ninflated for E3SM's single realisation); dots on every third latitude. Colours saturate at "
                 "the 95th percentile of |bias|." % name,
                 fontsize=10, color=pm.INK, x=0.07, ha="left")
    pm.finish(fig, os.path.join(out, "hovmoller_%s.png" % key))


# ------------------------------------------------------------------------------------------------ annual maps
def fig_annual_maps(Z, T, key, out):
    arms = [a for a in ARMS if a in Z]
    name, unit, fac = VINFO[key]
    mi = list(T["map_keys"]).index(key)
    lat, lon, lsm = T["lat"], T["lon"], T["lsm"].astype(np.float64)
    w = ml.lat_weights(lat)
    cells = {}
    for yi, year in enumerate(fl.YEARS):
        truth = T["ann_maps"][yi, mi].astype(np.float64)
        for arm in arms:
            mm = Z[arm]["ann_maps"][:, yi, mi].astype(np.float64)                 # (n, H, W)
            n = mm.shape[0]
            bias, t, _, dof = fl.bias_stats(mm, truth)
            sig = ml.fdr_mask(2 * stats.t.sf(np.abs(t), dof), pm.ALPHA_FDR)
            rms = float(np.sqrt(ml.area_mean((bias * fac) ** 2, w)))
            # Bootstrap over runs, re-centred on the estimate. A resample with repeated runs has a noisier mean,
            # which pushes every resampled RMS up: where run-to-run noise dominates (single-year PRECT) the raw
            # percentile interval excluded its own estimate (e.g. 0.545 [0.546, 0.591]). Subtracting the bootstrap
            # bias, mean(vals) - rms, gives the sampling spread of this 8-run RMS around its own value.
            vals = np.array([np.sqrt(ml.area_mean(((mm[ix].mean(axis=0) - truth) * fac) ** 2, w))
                             for ix in (RNG.integers(0, n, n) for _ in range(pm.NBOOT))])
            lo, hi = np.percentile(vals - (vals.mean() - rms), [2.5, 97.5])
            cells[(year, arm)] = (bias * fac, sig, rms, lo, hi, n)
    norm, lev = fine_norm(np.stack([c[0] for c in cells.values()]))
    ny, na = len(fl.YEARS), len(arms)
    fig, axs = plt.subplots(ny, na, figsize=(3.9 * na + 1.2, 2.55 * ny + 1.6), squeeze=False)
    fig.subplots_adjust(left=0.05, right=0.90, top=0.89, bottom=0.07, wspace=0.10, hspace=0.62)
    for i, year in enumerate(fl.YEARS):
        for j, arm in enumerate(arms):
            bias, sig, rms, lo, hi, n = cells[(year, arm)]
            ax = axs[i, j]
            im = pm.draw_map(ax, bias, lon, lat, lsm, pm.DIV, norm, sig=sig)
            pm.geo_axes(ax, lon, lat, left=(j == 0), bottom=(i == ny - 1))
            area = 100 * ml.area_mean(sig.astype(float), w)
            win = " (selection window)" if fl.SELECTION_WINDOW[0] <= year <= fl.SELECTION_WINDOW[1] else ""
            ax.set_title("%s, %d%s\nRMS bias %s %s [95%% CI %s, %s]\n%.0f%% of area significant; %d runs"
                         % (arm, year, win, pm.s3(rms), unit, pm.s3(lo), pm.s3(hi), area, n), fontsize=6.8,
                         color=pm.INK, loc="left")
            row(figure="annual_bias_%s" % key, variable=key, model=arm, year=year,
                metric="global RMS of annual-mean ensemble-mean bias (model - E3SM); CI = bootstrap over runs, re-centred", value=rms, ci_lo=lo, ci_hi=hi,
                unit=unit_plain(unit), n=n, sig_area_pct=area)
    cax = fig.add_axes([0.915, 0.25, 0.009, 0.50])
    cb = fig.colorbar(im, cax=cax, ticks=lev, extend="both")
    cb.set_label("%s bias,\nmodel - E3SM (%s)" % (key, unit), color=pm.INK, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=pm.INK2)
    fig.suptitle("%s: annual-mean bias by year, mean of eight five-year runs minus E3SM (rows: year; columns: "
                 "checkpoint)\nDots: bias larger than internal variability at a 10%% false discovery rate (t-test "
                 "against the run spread, inflated for E3SM's single realisation). Colours saturate at the 95th "
                 "percentile of |bias| (arrows)." % textwrap.shorten(name, 80), fontsize=10, color=pm.INK, x=0.05, ha="left")
    pm.finish(fig, os.path.join(out, "annual_bias_%s.png" % key))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=HERE)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "figures"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": pm.SURFACE, "figure.facecolor": pm.SURFACE})
    tp = os.path.join(a.in_dir, "fiveyear_truth.npz")
    if not os.path.exists(tp):
        print("ERROR NO_TRUTH %s" % tp)
        return 3
    T = np.load(tp)
    Z = {arm: np.load(os.path.join(a.in_dir, "fiveyear_%s.npz" % arm)) for arm in ARMS
         if os.path.exists(os.path.join(a.in_dir, "fiveyear_%s.npz" % arm))}
    missing = [arm for arm in ARMS if arm not in Z]
    for arm in Z:
        fig_ts_arm(Z, T, arm, a.out_dir)
    fig_ts_bias_all(Z, T, a.out_dir)
    fig_bias_by_year(Z, T, a.out_dir)
    for key in fl.HOVMOLLER_KEYS:
        fig_hovmoller(Z, T, key, a.out_dir)
    for key in fl.MAP_KEYS:
        fig_annual_maps(Z, T, key, a.out_dir)
    with open(os.path.join(a.out_dir, "summary_5yr.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(ROWS[0].keys()))
        wr.writeheader()
        wr.writerows(ROWS)
    print("wrote summary_5yr.csv (%d rows)" % len(ROWS))
    if missing:
        print("WARN MISSING_ARMS %s" % missing)
        return 2
    print("FIVEYEAR_PLOT_OK figures=%d" % len([f for f in os.listdir(a.out_dir) if f.endswith(".png")]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
