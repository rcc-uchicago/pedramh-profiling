"""Climatological bias maps for the reporting window 2048-2049, plus a like-for-like window table.

The 2048-2049 map is maps/figures/bias_maps.png's method, with the same tests, re-centred CI and colour limits per
variable, applied to the last two years of the same five-year runs. Descriptive: it selects nothing.

Inputs:
  * fiveyear_<ARM>.npz / fiveyear_truth.npz: per-year window means (the scorer's window_mean, one year at a time);
  * maps/clim_<ARM>.npz / clim_truth.npz: the 2045-2047 window means that bias_maps.png was drawn from.
A two-year window mean is the equal-weight mean of its two annual means, because every full year has the same
number of 6-hourly frames (noleap, 1460). Gates, before anything is drawn:
  1. the mean of the 2045, 2046 and 2047 annual means equals the 2045-2047 window mean in clim_<ARM>.npz, member by
     member, and the same for E3SM (relative 1e-6: the annual means are stored as float32);
  2. every member, and E3SM, has the same frame count in every year (n_year).
Outputs (figures/):
  * bias_maps_2048_2049.png;
  * summary_window_bias.csv and .json: RMS bias, re-centred CI and significant area per variable x checkpoint for
    - 2045-2047, copied from maps/figures/summary_table.csv (the published numbers; the value is re-checked);
    - 2046-2047, two years like the reporting window, inside the selection window, after the first-15-month
      growth of the TREFHT bias;
    - 2048-2049, the reporting window.
PASS = WINDOW_BIAS_OK
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAPS = os.path.join(os.path.dirname(HERE), "maps")
sys.path.insert(0, HERE)
sys.path.insert(0, MAPS)
import fiveyear_lib as fl  # noqa: E402
import maps_lib as ml  # noqa: E402
import plot_maps as pm  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ARMS = pm.BIAS_MODELS
REPORT, LIKE = (2048, 2049), (2046, 2047)
HEADLINE_REPORT = (
    "Climatological bias averaged over 2048-2049 only: mean of the 8 five-year runs minus E3SM. The runs start in Oct "
    "2044 and end in Dec 2049;\n2048-2049 (their years 4-5) is the reporting window, never used to rank checkpoints. "
    "Colours use the 2045-2047 map's scale. CI = bootstrap over the runs, re-centred on the estimate.")  # emitall-lint: allow-literal protocol windows and member count in a title, not results
GATE_REL = 1e-6


def window_maps(ann, years):
    """Equal-weight mean over the given years of ann (..., n_years, K, H, W) -> (..., K, H, W), float64."""
    yi = [fl.YEARS.index(y) for y in range(years[0], years[1] + 1)]
    return np.asarray(ann, dtype=np.float64)[..., yi, :, :, :].mean(axis=-4)


def rel_err(a, b):
    """max |a - b| / max |b| per channel (axis -3), worst channel."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float((np.abs(a - b).max(axis=(-2, -1)) / np.abs(b).max(axis=(-2, -1))).max())


def main():
    out = os.path.join(HERE, "figures")
    Z = {a: np.load(os.path.join(HERE, "fiveyear_%s.npz" % a)) for a in ARMS}
    T5 = np.load(os.path.join(HERE, "fiveyear_truth.npz"))
    C45 = {a: np.load(os.path.join(MAPS, "clim_%s.npz" % a)) for a in ARMS}
    T45 = np.load(os.path.join(MAPS, "clim_truth.npz"))
    keys = [str(k) for k in T5["map_keys"]]

    # gate 1: annual means reproduce the 2045-2047 window means
    bad = []
    tk = [list(T45["var_keys"]).index(k) for k in keys]
    e_t = rel_err(window_maps(T5["ann_maps"], fl.SELECTION_WINDOW), T45["truth_maps"][tk])
    print("gate 1 E3SM: mean of 2045-2047 annual means vs clim_truth.npz, max rel %.2e" % e_t)
    if not e_t <= GATE_REL:
        bad.append("E3SM gate 1 %.2e" % e_t)
    for a in ARMS:
        cm = [str(m) for m in C45[a]["members"]]
        zm = [str(m) for m in Z[a]["members"]]
        if sorted(cm) != sorted(zm):
            bad.append("%s members differ" % a)
            continue
        ck = [list(C45[a]["var_keys"]).index(k) for k in keys]
        mine = window_maps(Z[a]["ann_maps"], fl.SELECTION_WINDOW)                   # (n, K, H, W), fiveyear order
        ref = C45[a]["member_maps"][[cm.index(m) for m in zm]][:, ck]
        e = rel_err(mine, ref)
        print("gate 1 %s: %d members, max rel %.2e" % (a, len(zm), e))
        if not e <= GATE_REL:
            bad.append("%s gate 1 %.2e" % (a, e))
    # gate 2: equal frame counts in every year, so equal weights are the window mean
    counts = {"E3SM": np.asarray(T5["n_year"]).reshape(1, -1)}
    counts.update({a: np.asarray(Z[a]["n_year"]) for a in ARMS})
    for who, c in counts.items():
        if not (c == c.flat[0]).all():
            bad.append("%s n_year unequal %s" % (who, np.unique(c).tolist()))
    print("gate 2: frames per year %s" % sorted(set(int(x) for c in counts.values() for x in c.ravel())))
    if bad:
        for b in bad:
            print("ERROR WINDOW_GATE %s" % b)
        return 1

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": pm.SURFACE, "figure.facecolor": pm.SURFACE})
    lsm = T45["lsm"].astype(np.float64)
    scale = pm.bias_scale(C45, T45)                                    # bias_maps.png's colour limits

    def as_clim(years):
        C = {a: {"member_maps": window_maps(Z[a]["ann_maps"], years), "var_keys": keys} for a in ARMS}
        T = {"truth_maps": window_maps(T5["ann_maps"], years), "var_keys": keys, "lat": T5["lat"], "lon": T5["lon"]}
        return C, T

    C48, T48 = as_clim(REPORT)
    pm.fig_bias(C48, T48, lsm, out, window=REPORT, fname="bias_maps_2048_2049.png", headline=HEADLINE_REPORT,
                scale=scale)

    # the like-for-like window: same panels' numbers, no figure
    C46, T46 = as_clim(LIKE)
    w = ml.lat_weights(T5["lat"])
    for var in ml.VAR_KEYS:
        fac, unit = pm.VINFO[var][2], pm.VINFO[var][1]
        truth = T46["truth_maps"][keys.index(var)]
        for a in ARMS:
            bias, sig, rms, lo, hi, n = pm.bias_panel(C46[a]["member_maps"][:, keys.index(var)], truth, fac, w)
            pm.ROWS.append(dict(figure="window_table", variable=var, model=a, lead_h="", window="%d-%d" % LIKE,
                                metric=pm.ROWS[0]["metric"], value=rms, ci_lo=lo, ci_hi=hi, unit=pm.plain_unit(unit),
                                n=n, sig_area_pct=100 * ml.area_mean(sig.astype(float), w)))

    # 2045-2047: the published rows, value re-checked against the clim npz
    pub = [r for r in csv.DictReader(open(os.path.join(MAPS, "figures", "summary_table.csv")))
           if r["figure"] == "bias_maps"]
    w45 = ml.lat_weights(T45["lat"])
    worst = 0.0
    for r in pub:
        var, a = r["variable"], r["model"]
        fac = pm.VINFO[var][2]
        truth = T45["truth_maps"][list(T45["var_keys"]).index(var)].astype(np.float64)
        mm = C45[a]["member_maps"][:, list(C45[a]["var_keys"]).index(var)].astype(np.float64)
        rms = float(np.sqrt(ml.area_mean(((mm.mean(axis=0) - truth) * fac) ** 2, w45)))
        worst = max(worst, abs(rms - float(r["value"])) / float(r["value"]))
    print("published 2045-2047 RMS re-computed: max rel %.2e" % worst)
    if worst > 1e-12:
        print("ERROR PUBLISHED_MISMATCH %.2e" % worst)
        return 1

    rows = [dict(variable=r["variable"], model=r["model"], window=r["window"], value=float(r["value"]),
                 ci_lo=float(r["ci_lo"]), ci_hi=float(r["ci_hi"]), sig_area_pct=float(r["sig_area_pct"]),
                 unit=r["unit"], n=int(r["n"]), source="maps/figures/summary_table.csv") for r in pub]
    rows += [dict(variable=r["variable"], model=r["model"], window=r["window"], value=float(r["value"]),
                  ci_lo=float(r["ci_lo"]), ci_hi=float(r["ci_hi"]), sig_area_pct=float(r["sig_area_pct"]),
                  unit=r["unit"], n=int(r["n"]),
                  source="bias_maps_2048_2049.png" if r["figure"] != "window_table" else "this table only")
             for r in pm.ROWS]
    with open(os.path.join(out, "summary_window_bias.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    idx = {}
    for r in rows:
        idx.setdefault(r["variable"], {}).setdefault(r["model"], {})[r["window"]] = {
            k: r[k] for k in ("value", "ci_lo", "ci_hi", "sig_area_pct")}
    for var in idx:                                      # derived here, not in a claims spec
        for a in idx[var]:
            d = idx[var][a]
            d["ratio_report_vs_like"] = d["2048-2049"]["value"] / d["2046-2047"]["value"]
            d["ratio_report_vs_selection"] = d["2048-2049"]["value"] / d["2045-2047"]["value"]
    json.dump(idx, open(os.path.join(out, "summary_window_bias.json"), "w"), indent=1)

    print("RMS bias by window (2045-2047 | 2046-2047 | 2048-2049; report/like):")
    for var in ml.VAR_KEYS:
        for a in ARMS:
            d = idx[var][a]
            print("  %-6s %-5s %s | %s | %s  %.3f" % (var, a, pm.s3(d["2045-2047"]["value"]),
                                                     pm.s3(d["2046-2047"]["value"]), pm.s3(d["2048-2049"]["value"]),
                                                     d["ratio_report_vs_like"]))
    print("wrote summary_window_bias.csv (%d rows) and .json" % len(rows))
    print("WINDOW_BIAS_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
