"""Figures for the G4D-after-63-epochs read-out: RMSE maps, RMSE-difference maps, RMSE-vs-lead curves, and
climatological-bias maps, each with a statement of uncertainty. Reads the npz written by extract_k56_maps.py /
extract_clim_maps.py; writes PNGs + summary_table.csv + FIGURES.md into --out-dir.

Uncertainty, by figure:
  * RMSE maps: 95% bootstrap CI (2000 resamples of the 24 start dates) of the global RMSE, in each title.
  * RMSE-difference maps: dots where the paired difference in squared error is significant
    (t-test over the start dates, Benjamini-Hochberg FDR 10% across grid cells); bootstrap CI of the global
    difference in the title.
  * RMSE-vs-lead curves: 95% t-interval of the mean over start dates of the per-start global RMSE (bands).
  * Bias maps: dots where the ensemble-mean bias exceeds internal variability (t-test with the members' spread,
    variance inflated by (1 + 1/n) for E3SM's single realisation; FDR 10%); bootstrap CI over members of the
    global RMS bias in the title, re-centred on the estimate (see fig_bias: the raw percentile interval is
    shifted up by resampling).
Colour: sequential = one blue ramp light->dark; diverging = blue <-> red around a neutral grey; categorical lines
use the first three slots of the dataviz reference palette in fixed order (the all-pairs-safe cap for small
multiples). PASS = MAPS_PLOT_OK
"""
import argparse
import csv
import math
import os
import sys
import textwrap

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import maps_lib as ml  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from scipy import stats  # noqa: E402

INK, INK2, GRID, SURFACE, NAN_GREY = "#1f1f1d", "#5c5c57", "#e4e3df", "#fcfcfb", "#d9d8d4"
SEQ = mcolors.LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = mcolors.LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f4a9a3", "#e34948", "#8c1f1e"])
for cm in (SEQ, DIV):
    cm.set_bad(NAN_GREY)
SLOT = {"A243": "#2a78d6", "G4D16": "#eb6834", "G4D56": "#1baf7a"}     # categorical slots 1-3, fixed per model

LABEL = {"A243": "A243 (single-step, epoch 243)", "G4D16": "G4D16 (current pick, epoch 16)",
         "G4D36": "G4D36 (epoch 36)", "G4D43": "G4D43 (epoch 43)", "G4D56": "G4D56 (epoch 56)",
         "G4D63": "G4D63 (epoch 63)"}
MAP_MODELS = ("A243", "G4D16", "G4D56", "G4D63")
DIFFS_A243 = tuple((m, "A243") for m in ("G4D16", "G4D36", "G4D43", "G4D56", "G4D63"))   # operator: companions vs A243
DIFFS_LENGTH = (("G4D36", "G4D16"), ("G4D56", "G4D16"), ("G4D63", "G4D16"))          # same SGDR position / cycle end
RATIO_MODELS = ("G4D16", "G4D56")
CURVE_MODELS = ("A243", "G4D16", "G4D56")
BIAS_MODELS = ("G4D16", "G4D36", "G4D43", "G4D56", "G4D63")
VINFO = {k: (name, unit, fac) for k, name, unit, fac in ml.VARS}
NBOOT, ALPHA_FDR, STIPPLE_EVERY = 2000, 0.10, 4
RNG = np.random.default_rng(20261102)
ROWS = []          # summary table


def s3(x, sign=False):
    """3 significant figures, trailing zeros kept (1.00, 0.716, 175): '%.3g' prints 1 and 1.1, which read unevenly."""
    if not math.isfinite(x):
        return "nan"
    if x == 0:
        return "0.00"
    p = max(0, 2 - int(math.floor(math.log10(abs(x)))))
    return ("%+.*f" if sign else "%.*f") % (p, x)


def plain_unit(u):
    return u.replace("$^{-1}$", "^-1")


def boot_ci(stat, n, centre=None):
    """Percentile 95% CI of stat(indices) over NBOOT resamples of range(n) with replacement. With centre (the
    full-sample estimate), the resamples are first shifted by the bootstrap bias, mean(vals) - centre."""
    vals = np.array([stat(RNG.integers(0, n, n)) for _ in range(NBOOT)])
    if centre is not None:
        vals = vals - (vals.mean() - centre)
    return np.percentile(vals, [2.5, 97.5])


def geo_axes(ax, lon, lat, left, bottom):
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-60, 61, 30))
    lon_lab = lambda x: "0°" if x == 0 else ("%d°%s" % (abs(x), "E" if x > 0 else "W") if abs(x) != 180 else "180°")
    lat_lab = lambda y: "0°" if y == 0 else "%d°%s" % (abs(y), "N" if y > 0 else "S")
    ax.set_xticklabels([lon_lab(int(x)) for x in np.arange(-180, 181, 60)] if bottom else [])
    ax.set_yticklabels([lat_lab(int(y)) for y in np.arange(-60, 61, 30)] if left else [])
    if bottom:
        ax.set_xlabel("Longitude", color=INK2, fontsize=8)
    if left:
        ax.set_ylabel("Latitude", color=INK2, fontsize=8)
    ax.tick_params(colors=INK2, labelsize=7, length=2, width=0.4)
    for s in ax.spines.values():
        s.set_linewidth(0.4)
        s.set_color(INK2)


def draw_map(ax, field, lon, lat, lsm, cmap, norm, sig=None):
    f, lc = ml.roll_lon(field, lon)
    im = ax.imshow(f, cmap=cmap, norm=norm, extent=(-180, 180, -90, 90), origin="upper" if lat[0] > lat[-1] else
                   "lower", interpolation="nearest", aspect="equal")
    m, _ = ml.roll_lon(lsm, lon)
    ax.contour(lc, lat, m, levels=[0.5], colors="#3a3a36", linewidths=0.35)
    if sig is not None:
        s, _ = ml.roll_lon(sig, lon)
        jj, ii = np.nonzero(s[::STIPPLE_EVERY, ::STIPPLE_EVERY])
        ax.plot(lc[::STIPPLE_EVERY][ii], lat[::STIPPLE_EVERY][jj], ".", ms=0.9, color=INK, alpha=0.75, mew=0)
    return im


def nice_norm(vmin, vmax, diverging=False):
    if diverging:
        top = MaxNLocator(nbins=6, symmetric=True).tick_values(-vmax, vmax)
        return mcolors.BoundaryNorm(top, DIV.N), top
    lev = MaxNLocator(nbins=10, min_n_ticks=6).tick_values(vmin, vmax)
    return mcolors.BoundaryNorm(lev, SEQ.N), lev


def finish(fig, path):
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote %s" % os.path.basename(path))


# ------------------------------------------------------------------------------------------------ data access
def load(d, name):
    p = os.path.join(d, name)
    return np.load(p) if os.path.exists(p) else None


def k56_var(z, var):
    """(mse_map[n_map, H, W], se_ic[n_ic, n_map, H, W]) in stored units^2."""
    if var == "Z500":
        mse = np.nanmean(z["se_Z500"], axis=0)
    else:
        mse = z["mse_map"][:, list(z["channels"]).index(var)].astype(np.float64)
    return mse, z["se_" + var].astype(np.float64)


def fig_rmse_maps(K, lsm, out):
    for var in ml.VAR_KEYS:
        name, unit, fac = VINFO[var]
        models = [m for m in MAP_MODELS if m in K]
        nl = len(ml.MAP_LEADS_H)
        fig, axs = plt.subplots(len(models), nl, figsize=(4.1 * nl, 2.35 * len(models) + 1.4), squeeze=False)
        fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.12, wspace=0.08, hspace=0.42)
        for j, h in enumerate(ml.MAP_LEADS_H):
            fields = {m: np.sqrt(k56_var(K[m], var)[0][j]) * fac for m in models}
            vmax = np.nanpercentile(np.stack(list(fields.values())), 99)
            norm, lev = nice_norm(0.0, vmax)
            for i, m in enumerate(models):
                z = K[m]
                lat, lon = z["lat"], z["lon"]
                w = ml.lat_weights(lat)
                se_ic = k56_var(z, var)[1][:, j]
                gmse = ml.area_mean(se_ic, w)                     # per-IC global MSE, (n_ic,)
                g = np.sqrt(gmse.mean()) * fac
                lo, hi = boot_ci(lambda ix: np.sqrt(gmse[ix].mean()) * fac, len(gmse))
                ax = axs[i, j]
                im = draw_map(ax, fields[m], lon, lat, lsm, SEQ, norm)
                geo_axes(ax, lon, lat, left=(j == 0), bottom=(i == len(models) - 1))
                ax.set_title("%s, lead %d h\nglobal RMSE %s %s  [95%% CI %s, %s]"
                             % (LABEL[m], h, s3(g), unit, s3(lo), s3(hi)), fontsize=7.5, color=INK, loc="left")
                ROWS.append(dict(figure="rmse_maps_%s" % var, variable=var, model=m, lead_h=h, window="",
                                 metric="global RMSE (sqrt of area-mean MSE over start dates)", value=g, ci_lo=lo,
                                 ci_hi=hi, unit=plain_unit(unit), n=len(gmse), sig_area_pct=""))
            cax = fig.add_axes([axs[-1, j].get_position().x0, 0.055, axs[-1, j].get_position().width, 0.015])
            cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=lev[::2])
            cb.set_label("RMSE at %d h (%s)" % (h, unit), color=INK, fontsize=8)
            cb.ax.tick_params(labelsize=7, colors=INK2)
        fig.suptitle("%s: forecast RMSE by grid cell at lead times 6, 30, 60 and 270 h\n"
                     "24 forecasts started monthly through the test years 2048-2049; CI = bootstrap over start dates"
                     % name, fontsize=10, color=INK, x=0.05, ha="left")
        finish(fig, os.path.join(out, "rmse_maps_%s.png" % var))


def fig_rmse_diffs(K, lsm, out, pair_list, tag, what):
    for var in ml.VAR_KEYS:
        name, unit, fac = VINFO[var]
        pairs = [(a, b) for a, b in pair_list if a in K and b in K]
        if not pairs:
            continue
        nl = len(ml.MAP_LEADS_H)
        H = 2.35 * len(pairs) + 2.6                     # inches; the bottom 1.45 in hold x labels + colorbar + caption
        fig, axs = plt.subplots(len(pairs), nl, figsize=(4.1 * nl, H), squeeze=False)
        fig.subplots_adjust(left=0.05, right=0.99, top=1 - 1.15 / H, bottom=1.45 / H, wspace=0.08, hspace=0.62)
        for j, h in enumerate(ml.MAP_LEADS_H):
            cells = []
            for a, b in pairs:
                if list(K[a]["ic_files"]) != list(K[b]["ic_files"]):
                    raise SystemExit("ERROR IC_MISMATCH %s vs %s" % (a, b))
                mse_a, se_a = k56_var(K[a], var)
                mse_b, se_b = k56_var(K[b], var)
                diff = (np.sqrt(mse_a[j]) - np.sqrt(mse_b[j])) * fac
                d = se_a[:, j] - se_b[:, j]                         # paired, per start date
                n = d.shape[0]
                with np.errstate(invalid="ignore", divide="ignore"):
                    t = d.mean(axis=0) / (d.std(axis=0, ddof=1) / np.sqrt(n))
                p = 2 * stats.t.sf(np.abs(t), n - 1)
                sig = ml.fdr_mask(p, ALPHA_FDR)
                cells.append((a, b, diff, sig, se_a[:, j], se_b[:, j]))
            vmax = np.nanpercentile(np.abs(np.stack([c[2] for c in cells])), 99)
            norm, lev = nice_norm(-vmax, vmax, diverging=True)
            for i, (a, b, diff, sig, sa, sb) in enumerate(cells):
                z = K[a]
                lat, lon = z["lat"], z["lon"]
                w = ml.lat_weights(lat)
                ga, gb = ml.area_mean(sa, w), ml.area_mean(sb, w)
                g = (np.sqrt(ga.mean()) - np.sqrt(gb.mean())) * fac
                lo, hi = boot_ci(lambda ix: (np.sqrt(ga[ix].mean()) - np.sqrt(gb[ix].mean())) * fac, len(ga))
                area = 100 * ml.area_mean(sig.astype(float), w)
                ax = axs[i, j]
                im = draw_map(ax, diff, lon, lat, lsm, DIV, norm, sig=sig)
                geo_axes(ax, lon, lat, left=(j == 0), bottom=(i == len(pairs) - 1))
                ax.set_title("%s minus %s, lead %d h\nglobal difference %s %s [95%% CI %s, %s]\n"
                             "%.0f%% of area significant" % (a, b, h, s3(g, True), unit, s3(lo, True), s3(hi, True),
                                                             area), fontsize=7.2, color=INK, loc="left")
                ROWS.append(dict(figure="rmse_%s_%s" % (tag, var), variable=var, model="%s-%s" % (a, b), lead_h=h,
                                 window="", metric="global RMSE difference", value=g, ci_lo=lo, ci_hi=hi,
                                 unit=plain_unit(unit), n=len(ga), sig_area_pct=area))
            cax = fig.add_axes([axs[-1, j].get_position().x0, 0.62 / H, axs[-1, j].get_position().width, 0.1 / H])
            cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=lev)
            cb.set_label("RMSE difference at %d h (%s)\nblue = first model more accurate" % (h, unit),
                         color=INK, fontsize=8)
            cb.ax.tick_params(labelsize=7, colors=INK2)
        fig.suptitle("%s: difference in forecast RMSE by grid cell, %s (first model minus second)\n"
                     "Dots: difference significant at a 10%% false discovery rate (paired t-test over the 24 start "
                     "dates, Benjamini-Hochberg across grid cells; dots on a %d° lattice)" % (name, what, STIPPLE_EVERY),
                     fontsize=10, color=INK, x=0.05, ha="left")
        finish(fig, os.path.join(out, "rmse_%s_%s.png" % (tag, var)))


def fig_curves(K, out):
    models = [m for m in CURVE_MODELS if m in K]
    leads = np.array(ml.EXTRACT_LEADS_H)
    nv = len(ml.VAR_KEYS)
    fig, axs = plt.subplots(1, nv, figsize=(4.0 * nv, 4.0))
    fig.subplots_adjust(left=0.05, right=0.96, top=0.72, bottom=0.16, wspace=0.55)
    for ax, var in zip(axs, ml.VAR_KEYS):
        name, unit, fac = VINFO[var]
        ends = []
        for m in models:
            z = K[m]
            r = (z["z500_rmse_ic"] if var == "Z500" else z["rmse_ic"][:, :, list(z["channels"]).index(var)]) * fac
            n = r.shape[0]
            mu = r.mean(axis=0)
            half = stats.t.ppf(0.975, n - 1) * r.std(axis=0, ddof=1) / np.sqrt(n)
            ax.fill_between(leads, mu - half, mu + half, color=SLOT[m], alpha=0.18, lw=0)
            ax.plot(leads, mu, "-o", color=SLOT[m], lw=1.4, ms=3.2, mec=SURFACE, mew=0.6, label=LABEL[m])
            ends.append([mu[-1], m])
            for h, v, lo, hi in zip(leads, mu, mu - half, mu + half):
                ROWS.append(dict(figure="rmse_curves", variable=var, model=m, lead_h=int(h), window="",
                                 metric="mean over start dates of per-start global RMSE", value=v, ci_lo=lo,
                                 ci_hi=hi, unit=plain_unit(unit), n=n, sig_area_pct=""))
        # direct labels at the right end, nudged apart so they never overlap
        ends.sort()
        span = ax.get_ylim()[1] - ax.get_ylim()[0]
        for k in range(1, len(ends)):
            ends[k][0] = max(ends[k][0], ends[k - 1][0] + 0.07 * span)
        for y, m in ends:
            ax.annotate(m, (leads[-1], y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7,
                        color=INK2, annotation_clip=False)
        ax.set_title(textwrap.fill(name, 38), fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("Lead time (h)", fontsize=8, color=INK2)
        ax.set_ylabel("Global RMSE (%s)" % unit, fontsize=8, color=INK2)
        ax.set_xticks([0, 48, 96, 144, 192, 240, 288, 336])
        ax.set_xlim(0, 336)
        ax.set_ylim(bottom=0)
        ax.grid(True, color=GRID, lw=0.5)
        ax.tick_params(labelsize=7, colors=INK2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Global (area-weighted) forecast RMSE vs lead time, 24 start dates in 2048-2049\n"
                 "Line = mean over start dates; band = 95% confidence interval (t-interval over start dates)",
                 fontsize=10, color=INK, x=0.05, ha="left")
    finish(fig, os.path.join(out, "rmse_curves.png"))


def fig_ratio_curves(K, out):
    """Companion to fig_curves: each G4D model's global RMSE divided by A243's, start date by start date."""
    if "A243" not in K:
        return
    models = [m for m in RATIO_MODELS if m in K]
    leads = np.array(ml.EXTRACT_LEADS_H)
    nv = len(ml.VAR_KEYS)
    fig, axs = plt.subplots(1, nv, figsize=(4.0 * nv, 4.0))
    fig.subplots_adjust(left=0.05, right=0.96, top=0.72, bottom=0.16, wspace=0.55)
    za = K["A243"]
    for ax, var in zip(axs, ml.VAR_KEYS):
        name, unit, fac = VINFO[var]
        ra = za["z500_rmse_ic"] if var == "Z500" else za["rmse_ic"][:, :, list(za["channels"]).index(var)]
        ax.axhline(1.0, color=SLOT["A243"], lw=1.0)
        ax.annotate("A243", (leads[-1], 1.0), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7,
                    color=INK2, annotation_clip=False)
        for m in models:
            z = K[m]
            if list(z["ic_files"]) != list(za["ic_files"]):
                raise SystemExit("ERROR IC_MISMATCH %s vs A243" % m)
            r = z["z500_rmse_ic"] if var == "Z500" else z["rmse_ic"][:, :, list(z["channels"]).index(var)]
            ratio = r.mean(axis=0) / ra.mean(axis=0)
            n = r.shape[0]
            boots = np.array([r[ix].mean(axis=0) / ra[ix].mean(axis=0)
                              for ix in (RNG.integers(0, n, n) for _ in range(NBOOT))])
            lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
            ax.fill_between(leads, lo, hi, color=SLOT[m], alpha=0.18, lw=0)
            ax.plot(leads, ratio, "-o", color=SLOT[m], lw=1.4, ms=3.2, mec=SURFACE, mew=0.6, label=LABEL[m])
            ax.annotate(m, (leads[-1], ratio[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=7, color=INK2, annotation_clip=False)
            for h, v, a_, b_ in zip(leads, ratio, lo, hi):
                ROWS.append(dict(figure="rmse_ratio_to_A243", variable=var, model=m, lead_h=int(h), window="",
                                 metric="global RMSE / A243 global RMSE (means over start dates)", value=v,
                                 ci_lo=a_, ci_hi=b_, unit="ratio", n=n, sig_area_pct=""))
        ax.set_title(textwrap.fill(name, 38), fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("Lead time (h)", fontsize=8, color=INK2)
        ax.set_ylabel("RMSE relative to A243 (ratio)", fontsize=8, color=INK2)
        ax.set_xticks([0, 48, 96, 144, 192, 240, 288, 336])
        ax.set_xlim(0, 336)
        ax.grid(True, color=GRID, lw=0.5)
        ax.tick_params(labelsize=7, colors=INK2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Global forecast RMSE relative to A243 (below 1 = more accurate than A243), 24 start dates in "
                 "2048-2049\nBand = 95% bootstrap CI, start dates resampled jointly for both models (paired)",
                 fontsize=10, color=INK, x=0.05, ha="left")
    finish(fig, os.path.join(out, "rmse_ratio_to_A243.png"))


def fig_bias(C, T, lsm, out):
    models = [m for m in BIAS_MODELS if m in C]
    lat, lon = T["lat"], T["lon"]
    w = ml.lat_weights(lat)
    nv = len(ml.VAR_KEYS)
    fig, axs = plt.subplots(nv, len(models), figsize=(3.9 * len(models) + 1.2, 2.55 * nv + 1.6), squeeze=False)
    fig.subplots_adjust(left=0.05, right=0.90, top=0.89, bottom=0.07, wspace=0.10, hspace=0.62)
    for i, var in enumerate(ml.VAR_KEYS):
        name, unit, fac = VINFO[var]
        vi = list(T["var_keys"]).index(var)
        truth = T["truth_maps"][vi].astype(np.float64)
        cells = []
        for m in models:
            mm = C[m]["member_maps"][:, list(C[m]["var_keys"]).index(var)].astype(np.float64)   # (n, H, W)
            n = mm.shape[0]
            bias = (mm.mean(axis=0) - truth) * fac
            with np.errstate(invalid="ignore", divide="ignore"):
                t = (mm.mean(axis=0) - truth) / (mm.std(axis=0, ddof=1) * np.sqrt(1.0 / n + 1.0))
            p = 2 * stats.t.sf(np.abs(t), n - 1)
            sig = ml.fdr_mask(p, ALPHA_FDR)
            rms = np.sqrt(ml.area_mean(bias ** 2, w))
            # Re-centred on rms, as the five-year annual maps are (fiveyear/plot_5yr.py): a resample with repeated
            # members has a noisier mean, which pushes every resampled RMS up, so the raw percentile interval sits
            # above the estimate. Same draws as before; only the interval's position changes.
            lo, hi = boot_ci(lambda ix: np.sqrt(ml.area_mean(((mm[ix].mean(axis=0) - truth) * fac) ** 2, w)), n,
                             centre=rms)
            cells.append((m, bias, sig, rms, lo, hi, n))
        vmax = np.nanpercentile(np.abs(np.stack([c[1] for c in cells])), 98)
        norm, lev = nice_norm(-vmax, vmax, diverging=True)
        for j, (m, bias, sig, rms, lo, hi, n) in enumerate(cells):
            ax = axs[i, j]
            im = draw_map(ax, bias, lon, lat, lsm, DIV, norm, sig=sig)
            geo_axes(ax, lon, lat, left=(j == 0), bottom=(i == nv - 1))
            area = 100 * ml.area_mean(sig.astype(float), w)
            ax.set_title("%s, %s\nRMS bias %s %s [95%% CI %s, %s]\n%.0f%% of area significant; %d runs"
                         % (LABEL[m], var, s3(rms), unit, s3(lo), s3(hi), area, n), fontsize=6.8, color=INK,
                         loc="left")
            ROWS.append(dict(figure="bias_maps", variable=var, model=m, lead_h="", window="2045-2047",
                             metric="global RMS of ensemble-mean bias (model - E3SM); CI = bootstrap over runs, "
                             "re-centred", value=rms, ci_lo=lo, ci_hi=hi,
                             unit=plain_unit(unit), n=n, sig_area_pct=area))
        pos = axs[i, -1].get_position()
        cax = fig.add_axes([0.915, pos.y0, 0.009, pos.height])
        cb = fig.colorbar(im, cax=cax, ticks=lev)
        cb.set_label("%s bias,\nmodel - E3SM (%s)" % (var, unit), color=INK, fontsize=7.5)
        cb.ax.tick_params(labelsize=6.5, colors=INK2)
    fig.suptitle("Climatological bias averaged over 2045-2047 only: mean of the 8 five-year runs minus E3SM. The runs "
                 "start in Oct 2044 and end in Dec 2049;\n2045-2047 (their years 1-3) is the selection window the "
                 "checkpoints were ranked on, and 2048-2049 is not included. CI = bootstrap over the runs, "
                 "re-centred on the estimate.\nDots: bias larger than internal variability at a 10% false discovery "
                 "rate (t-test against the member spread, inflated for E3SM's single realisation). A243 is not "
                 "shown: all its five-year runs blow up within 0.6 years.", fontsize=9.5, color=INK, x=0.05,
                 ha="left")
    finish(fig, os.path.join(out, "bias_maps.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=HERE)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "figures"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": SURFACE, "figure.facecolor": SURFACE})
    K = {m: z for m in ("A243", "G4D16", "G4D36", "G4D43", "G4D56", "G4D63")
         for z in [load(a.in_dir, "k56_%s.npz" % m)] if z is not None}
    C = {m: z for m in BIAS_MODELS for z in [load(a.in_dir, "clim_%s.npz" % m)] if z is not None}
    T = load(a.in_dir, "clim_truth.npz")
    missing = [m for m in MAP_MODELS if m not in K] + [m for m in BIAS_MODELS if m not in C] + ([] if T else ["truth"])
    if T is None:
        print("ERROR NO_TRUTH clim_truth.npz (land-sea mask and E3SM means)")
        return 3
    lsm = T["lsm"].astype(np.float64)
    fig_rmse_maps(K, lsm, a.out_dir)
    fig_rmse_diffs(K, lsm, a.out_dir, DIFFS_A243, "vs_A243", "each G4D checkpoint against A243")
    fig_rmse_diffs(K, lsm, a.out_dir, DIFFS_LENGTH, "vs_G4D16", "longer G4D training against G4D16")
    fig_curves(K, a.out_dir)
    fig_ratio_curves(K, a.out_dir)
    fig_bias(C, T, lsm, a.out_dir)
    with open(os.path.join(a.out_dir, "summary_table.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(ROWS[0].keys()))
        wr.writeheader()
        wr.writerows(ROWS)
    print("wrote summary_table.csv (%d rows)" % len(ROWS))
    if missing:
        print("WARN MISSING_INPUTS %s: their panels are absent" % missing)
        return 2
    print("MAPS_PLOT_OK figures=%d" % len([f for f in os.listdir(a.out_dir) if f.endswith(".png")]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
