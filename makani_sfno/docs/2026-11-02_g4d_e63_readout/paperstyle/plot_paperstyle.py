"""Figures for the paper-style comparison (prereg.md, results.json): G4D's five-year error against E3SM's own natural
variability, the way Duncan et al. (2024) score ACE.

  paperstyle_ratios.png    R = one run's 2045-2049 RMSE / E3SM's 2040-2044 vs 2045-2049 RMSE, per variable, for
                           G4D16/36/43 (dot = median of 8 runs, bar = range), with ACE-EAMv2's ratios from the paper
  paperstyle_maps.png      one representative run's 2045-2049 error map per checkpoint, next to E3SM's own
                           window-to-window difference, on the same colour scale per variable
  paperstyle_channels.png  median R for all 99 channels, grouped by variable and level
  paperstyle_figures.csv   every plotted number

Reads paperstyle_<ARM>.npz and truth_windows.npz (paperstyle.py) and maps/clim_truth.npz (longitude, land mask).
Gate: R recomputed here for the reported channels must equal results.json to float32-storage precision (GATE_REL).
Compute node, SFNO venv.   PASS = PAPERSTYLE_PLOT_OK
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAPS = "/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval/g4d_e63_2026-11-02/maps"
sys.path.insert(0, MAPS)
sys.path.insert(0, "/lus/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling/.claude/worktrees/"
                   "eval-protocol/makani_sfno/polaris")
import maps_lib as ml  # noqa: E402
import plot_maps as pm  # noqa: E402  (styles, map drawing, s3)
import score_window_protocol as swp  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ARMS = ("G4D16", "G4D36", "G4D43")
COLOR = {"G4D16": "#eb6834", "G4D36": "#6da7ec", "G4D43": "#3987e5"}   # as the five-year figures
FAC = {"PRECT": (86400.0 * 1000.0, "mm day$^{-1}$"), "PS": (0.01, "hPa"), "PSL": (0.01, "hPa")}
LABEL = {"PRECT": "precipitation (PRECT)", "FSNTOA": "net sunlight at TOA (FSNTOA)\n= reflected SW, as in the paper",
         "PS": "surface pressure (PS)", "TREFHT": "2 m temperature (TREFHT)", "U10": "10 m wind speed (U10)",
         "Z3_l12": "height, level 12 (Z3_l12)", "U_l17": "lowest-level zonal wind (U_l17)",
         "T_l17": "lowest-level temperature (T_l17)"}
RATIO_ORDER = ("PRECT", "FSNTOA", "U10", "U_l17", "TREFHT", "T_l17", "Z3_l12", "PS")
MAP_ROWS = ("PRECT", "FSNTOA", "PS", "Z3_l12")
PAPER = {"PRECT": 0.37 / 0.21, "FSNTOA": 4.17 / 1.63}    # Duncan et al. 2024, Table S5: ACE RMSE / EAMv2 floor
FAMILIES = ("T", "U", "V", "Z3", "RELHUM")
ROWS = []
SUMMARY = {}


def fac(k):
    return FAC.get(k, (1.0, {"FSNTOA": "W m$^{-2}$", "FSNT": "W m$^{-2}$", "TREFHT": "K", "U10": "m s$^{-1}$",
                             "TMQ": "kg m$^{-2}$", "RHREFHT": "%", "Z3_l12": "m"}.get(k, "")))


TRUTH = "/eagle/projects/lighthouse-uchicago/members/mehta5/runs/makani_eval/g4d_e63_2026-11-02/fiveyear/truth"
# The runs' maps are stored as float32 (paperstyle_<ARM>.npz), so R recomputed here can match results.json only to
# float32 precision: relative ~1e-7 per field, more where an error is a small difference of large values. The first
# version of this gate used 1e-9, which float32 storage cannot meet (7814245: 5.4e-7); 1e-5 still catches any
# different computation. The ratio figure plots results.json's own numbers; only the 99-channel strip and the maps use
# the recomputed fields. E3SM's windows are rebuilt in float64 from the annual means, as paperstyle.py did.
GATE_REL = 1e-5


def window(years):
    return np.mean([np.load(os.path.join(TRUTH, "truth_series_%d.npz" % y))["time_mean"].astype(np.float64)
                    for y in range(years[0], years[1] + 1)], axis=0)


def load():
    tw = np.load(os.path.join(HERE, "truth_windows.npz"))
    names = [str(c) for c in tw["channel_names"]]
    lat = np.asarray(tw["lat"], np.float64)
    w = swp.area_weights(lat)
    e45, e40 = window((2045, 2049)), window((2040, 2044))
    floor = swp.pattern_rmse(e40, e45, w)
    runs = {}
    for a in ARMS:
        z = np.load(os.path.join(HERE, "paperstyle_%s.npz" % a))
        if [str(c) for c in z["channel_names"]] != names:
            raise ValueError("CHANNELS_DIFFER %s" % a)
        maps = z["maps_2045_2049"].astype(np.float64)                                     # (8, C, H, W)
        rmse = np.stack([swp.pattern_rmse(m, e45, w) for m in maps])                     # (8, C)
        runs[a] = {"maps": maps, "members": [str(m) for m in z["members"]], "R": rmse / floor, "rmse": rmse}
    ct = np.load(os.path.join(MAPS, "clim_truth.npz"))
    if not np.allclose(ct["lat"], lat):
        raise ValueError("LAT_MISMATCH clim_truth vs truth_windows")
    return names, lat, np.asarray(ct["lon"], np.float64), ct["lsm"].astype(np.float64), w, e45, e40, floor, runs


def gate(names, runs):
    res = json.load(open(os.path.join(HERE, "results.json")))
    worst = 0.0
    for a in ARMS:
        for k, ch in res["arms"][a]["channels"].items():
            mine = float(np.median(runs[a]["R"][:, names.index(k)]))
            worst = max(worst, abs(mine - ch["R_median"]) / ch["R_median"])
    print("gate: R recomputed from the stored maps vs results.json, max rel %.2e (limit %.0e)" % (worst, GATE_REL))
    if worst > GATE_REL:
        raise ValueError("R_MISMATCH %.2e" % worst)
    return res


def fig_ratios(names, runs, res, out):
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.80, bottom=0.25)
    xs = np.arange(len(RATIO_ORDER))
    off = {"G4D16": -0.22, "G4D36": 0.0, "G4D43": 0.22}
    for a in ARMS:
        for i, k in enumerate(RATIO_ORDER):
            ch = res["arms"][a]["channels"][k]                       # results.json's own numbers, not recomputed
            x = xs[i] + off[a]
            ax.plot([x, x], [ch["R_min"], ch["R_max"]], color=COLOR[a], lw=2, solid_capstyle="round", zorder=2)
            ax.plot(x, ch["R_median"], "o", ms=8, color=COLOR[a], mec=pm.SURFACE, mew=1.5, zorder=3,
                    label=a if i == 0 else None)
            ROWS.append(dict(figure="paperstyle_ratios", variable=k, model=a, value=ch["R_median"],
                             lo=ch["R_min"], hi=ch["R_max"], unit="ratio"))
            if k in PAPER:
                r10 = res["arms"][a]["channels"][k]["R10_of_median"]
                ax.plot(x, r10, "o", ms=8, mfc="none", mec=COLOR[a], mew=1.5, zorder=3,
                        label="≈10-yr equivalent (model-based)" if (a == "G4D43" and k == "PRECT") else None)
                ROWS.append(dict(figure="paperstyle_ratios", variable=k + "_R10", model=a, value=r10, lo="", hi="",
                                 unit="ratio"))
    for i, k in enumerate(RATIO_ORDER):
        if k in PAPER:
            ax.plot([xs[i] - 0.38, xs[i] + 0.38], [PAPER[k]] * 2, color=pm.INK, lw=2, zorder=4,
                    label="ACE-EAMv2, Duncan et al. 2024 (10-yr)" if k == "PRECT" else None)
            ax.annotate("paper %s" % pm.s3(PAPER[k]), (xs[i] + 0.4, PAPER[k]), xytext=(10, 0),
                        textcoords="offset points", va="center", fontsize=8, color=pm.INK)
            ROWS.append(dict(figure="paperstyle_ratios", variable=k, model="ACE-EAMv2 (paper)", value=PAPER[k],
                             lo="", hi="", unit="ratio"))
    ax.axhline(1.0, color=pm.INK2, lw=0.8, zorder=1)
    ax.annotate("1 = indistinguishable from E3SM's own 5-yr variability", (xs[-1] + 0.45, 1.0), xytext=(0, 4),
                textcoords="offset points", ha="right", va="bottom", fontsize=8, color=pm.INK2)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 40)
    ax.set_xlim(-0.6, len(RATIO_ORDER) - 0.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[k] for k in RATIO_ORDER], rotation=30, ha="right", fontsize=8.5, color=pm.INK)
    ax.set_ylabel("error ÷ E3SM's natural variability (log)", color=pm.INK, fontsize=9)
    ax.grid(axis="y", color=pm.GRID, lw=0.6)
    for s in ax.spines.values():
        s.set_linewidth(0.4)
        s.set_color(pm.INK2)
    ax.tick_params(colors=pm.INK2, labelsize=8)
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=3)
    fig.suptitle("G4D's five-year error measured the way Duncan et al. (2024) measure ACE\n"
                 "Error = one run's 2045-2049 mean vs E3SM's (global RMS of the difference map). Natural variability = "
                 "E3SM's 2040-2044 mean vs its 2045-2049 mean.\nDot = median of 8 runs, bar = range; the paper reports "
                 "only precipitation, reflected sunlight and OLR (no OLR channel here).",
                 fontsize=9.5, color=pm.INK, x=0.08, ha="left")
    pm.finish(fig, os.path.join(out, "paperstyle_ratios.png"))


def representative(runs, a):
    """The run whose mean R over all channels is the median of the 8 (the lower middle one)."""
    m = runs[a]["R"].mean(axis=1)
    return int(np.argsort(m)[(len(m) - 1) // 2])


def fig_maps(names, lat, lon, lsm, w, e45, e40, runs, out):
    nr, nc = len(MAP_ROWS), 1 + len(ARMS)
    fig, axs = plt.subplots(nr, nc, figsize=(3.9 * nc + 1.2, 2.55 * nr + 1.6), squeeze=False)
    fig.subplots_adjust(left=0.05, right=0.90, top=0.86, bottom=0.07, wspace=0.10, hspace=0.62)
    pick = {a: representative(runs, a) for a in ARMS}
    for i, k in enumerate(MAP_ROWS):
        c = names.index(k)
        f, unit = fac(k)
        cells = [("E3SM: 2040-2044 minus 2045-2049\n(natural variability)", (e40[c] - e45[c]) * f, None)]
        for a in ARMS:
            j = pick[a]
            cells.append(("%s, run %s, minus E3SM 2045-2049" % (a, runs[a]["members"][j][-6:-3]),
                          (runs[a]["maps"][j, c] - e45[c]) * f, a))
        vmax = np.nanpercentile(np.abs(np.stack([x[1] for x in cells])), 98)
        norm, lev = pm.nice_norm(-vmax, vmax, diverging=True)
        for j, (title, field, a) in enumerate(cells):
            ax = axs[i, j]
            im = pm.draw_map(ax, field, lon, lat, lsm, pm.DIV, norm)
            pm.geo_axes(ax, lon, lat, left=(j == 0), bottom=(i == nr - 1))
            rms = float(np.sqrt(ml.area_mean(field ** 2, ml.lat_weights(lat))))
            ax.set_title("%s\n%s: RMS %s %s" % (title, k, pm.s3(rms), unit), fontsize=6.8, color=pm.INK, loc="left")
            ROWS.append(dict(figure="paperstyle_maps", variable=k, model=a or "E3SM window difference", value=rms,
                             lo="", hi="", unit=unit.replace("$^{-1}$", "^-1").replace("$^{-2}$", "^-2")))
        pos = axs[i, -1].get_position()
        cax = fig.add_axes([0.915, pos.y0, 0.009, pos.height])
        cb = fig.colorbar(im, cax=cax, ticks=lev)
        cb.set_label("%s difference (%s)" % (k, unit), color=pm.INK, fontsize=7.5)
        cb.ax.tick_params(labelsize=6.5, colors=pm.INK2)
    fig.suptitle("Where the five-year error sits, against E3SM's own natural variability (same colour scale per row)\n"
                 "Left: E3SM's 2040-2044 mean minus its 2045-2049 mean. Others: one run's 2045-2049 mean minus E3SM's; the "
                 "run shown per checkpoint is the one with the median all-channel error.", fontsize=9.5, color=pm.INK,
                 x=0.05, ha="left")
    pm.finish(fig, os.path.join(out, "paperstyle_maps.png"))


def channel_order(names):
    surf = [n for n in names if "_l" not in n]
    fam = [[n for n in names if n.startswith(f + "_l")] for f in FAMILIES]
    return surf, fam


def fig_channels(names, runs, out):
    surf, fam = channel_order(names)
    order = surf + [n for g in fam for n in g]
    fig, ax = plt.subplots(figsize=(15, 5.6))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.74, bottom=0.19)
    x = np.arange(len(order))
    for a in ARMS:
        med = np.median(runs[a]["R"], axis=0)
        y = np.array([med[names.index(n)] for n in order])
        SUMMARY[a] = {"channel_median_of_R": float(np.median(med)), "channel_mean_of_R": float(np.mean(med)),
                      "channel_mean_excl_Z3_l15_l17": float(np.mean([med[names.index(n)] for n in names
                                                                    if n not in ("Z3_l15", "Z3_l16", "Z3_l17")])),
                      "n_channels_above_3": int((med > 3).sum()), "n_channels": len(med),
                      "R": {n: float(med[names.index(n)]) for n in names}}
        start = 0
        for g in [surf] + fam:
            seg = slice(start, start + len(g))
            ax.plot(x[seg], y[seg], "-o", ms=3.5, lw=1.2, color=COLOR[a],
                    label=("%s (median over channels %s, mean %s)" % (
                        a, pm.s3(SUMMARY[a]["channel_median_of_R"]), pm.s3(SUMMARY[a]["channel_mean_of_R"])))
                    if start == 0 else None)
            start += len(g)
        for n, v in zip(order, y):
            ROWS.append(dict(figure="paperstyle_channels", variable=n, model=a, value=float(v), lo="", hi="",
                             unit="ratio"))
    ax.axhline(1.0, color=pm.INK2, lw=0.8)
    ax.set_yscale("log")
    zi = order.index("Z3_l17")
    ax.annotate("Z3_l17: the lowest level follows the terrain,\nso E3SM barely varies there and the ratio explodes",
                (zi, SUMMARY["G4D43"]["R"]["Z3_l17"]), xytext=(-12, -6), textcoords="offset points", ha="right",
                va="top", fontsize=7.5, color=pm.INK2)
    start = 0
    ticks, labels = [], []
    for name, g in [("surface", surf)] + list(zip(FAMILIES, fam)):
        if start:
            ax.axvline(start - 0.5, color=pm.GRID, lw=0.8)
        ax.annotate(name if name == "surface" else "%s, levels 0 (top) - 17" % name, (start + (len(g) - 1) / 2, 1.0),
                    xycoords=("data", "axes fraction"), xytext=(0, 3), textcoords="offset points", ha="center",
                    va="bottom", fontsize=8, color=pm.INK)
        if name == "surface":
            ticks += list(range(start, start + len(g)))
            labels += g
        else:
            ticks += [start, start + len(g) - 1]
            labels += [g[0].split("_l")[1], g[-1].split("_l")[1]]
        start += len(g)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=7, color=pm.INK2)
    ax.set_xlim(-0.8, len(order) - 0.2)
    ax.set_ylabel("error ÷ E3SM's natural variability (log)", color=pm.INK, fontsize=9)
    ax.grid(axis="y", color=pm.GRID, lw=0.6)
    for s in ax.spines.values():
        s.set_linewidth(0.4)
        s.set_color(pm.INK2)
    ax.tick_params(colors=pm.INK2, labelsize=8)
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.055, 0.885), fontsize=8, frameon=False, ncol=3)
    fig.suptitle("The same ratio for all 99 channels (median of 8 runs; 1 = indistinguishable from E3SM's 5-yr "
                 "variability)\nMost channels sit near the median. The outliers are surface pressure and geopotential "
                 "height, rising toward the surface; sea-level pressure is not among them.", fontsize=9.5,
                 color=pm.INK, x=0.06, ha="left")
    pm.finish(fig, os.path.join(out, "paperstyle_channels.png"))


def main():
    out = os.path.join(HERE, "figures")
    os.makedirs(out, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": pm.SURFACE, "figure.facecolor": pm.SURFACE})
    names, lat, lon, lsm, w, e45, e40, floor, runs = load()
    res = gate(names, runs)
    fig_ratios(names, runs, res, out)
    fig_maps(names, lat, lon, lsm, w, e45, e40, runs, out)
    fig_channels(names, runs, out)
    with open(os.path.join(out, "paperstyle_figures.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(ROWS[0].keys()))
        wr.writeheader()
        wr.writerows(ROWS)
    print("wrote paperstyle_figures.csv (%d rows)" % len(ROWS))
    json.dump(SUMMARY, open(os.path.join(out, "paperstyle_figures_summary.json"), "w"), indent=1)
    print("PAPERSTYLE_PLOT_OK figures=3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
