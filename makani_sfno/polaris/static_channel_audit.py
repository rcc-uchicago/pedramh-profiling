#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
"""Cold-critic audit of the `Z3_l17`-is-topography conclusion, across all 101 channels.

Job 7646192 showed `Z3_l17` is the slowest channel in the pack and carries the largest
systematic share of its 336 h error, and a follow-up correlation put its time-mean field
at r = +0.985 against the `topo` forcing.  From that I concluded: the channel is mostly a
static map the model is already given as an input, so the fix is to stop predicting it
rather than to weight it 9518x.

**Every load-bearing step of that has a way to be wrong, and this file tries each one.**
Six pre-registered attacks, thresholds fixed here before any number was computed.

    A1  OUTLIER?      I treated Z3_l17 as a special case.  If many channels have the same
                      static-to-forecastable ratio, "delete it" does not scale and the
                      right answer is a normalization change, not a channel change.
                      FALSIFIED (as an outlier) if >= 5 channels have S_c > 300.

    A2  WHICH OTHERS? Generalise "surface topography" beyond Z3: regress every channel's
                      time-mean field on ALL FOUR static forcings (lsm, topo, glacier,
                      natveg).  R^2 > 0.90 means the channel's spatial structure is
                      reconstructible from inputs the model already has.  Reported as a
                      list, not a verdict -- this is the question, not a hypothesis.

    A3  IS H4 A Z3 ARTIFACT?  Job 7646192's headline was Spearman(r_c, bias^2/MSE) =
                      -0.647 over 101 channels.  But 18 of those are Z3 levels of one
                      variable, and they sit in both tails.  Recompute with the Z3 family
                      REMOVED.
                      FALSIFIED if |rho| < 0.30 on the remaining channels -- then the
                      "loss blindness drives drift" claim was one variable wearing a
                      trenchcoat.

    A4  IS THE BIAS ACTUALLY LINEAR IN LEAD?  The "static error integrating" story
                      predicts bias(k) ~ m*k.  Fit it per channel.
                      FALSIFIED for Z3_l17 if R^2 < 0.90.

    A5  IS THE BIAS TERRAIN-STRUCTURED?  My scale_factor=3 spectral-reconstruction story
                      is pure speculation.  If it is right, the 336 h bias MAP for Z3_l17
                      is spatially organised by terrain.
                      FALSIFIED if |corr(bias_map, topo)| < 0.30 AND
                      |corr(bias_map, |grad topo|)| < 0.30.

    A6  IS IT REDUNDANT?  I claimed the time-varying part is a thickness ~ near-surface
                      temperature, hence already carried by TREFHT/T_l17.  Layer thickness
                      ~33 m at sea level predicts dZ/dT ~ 33/288 ~ 0.115 m/K.
                      FALSIFIED if the median per-cell temporal correlation with TREFHT is
                      < 0.50.  The fitted slope is reported against the 0.115 prediction.

⚠ PASS = `STATIC_AUDIT_OK` and it means ONLY that the audit ran.  The verdicts are the
A<n> lines, and falsifying my own conclusion is the successful outcome here, not the
failure one (CLAUDE.md #14).

Compute node, SFNO venv (xarray/h5py).  CPU-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

# Pre-registered. Moving these after seeing numbers forfeits the point.
A1_S_THRESHOLD = 300.0
A1_MAX_PEERS = 5
A2_R2_STATIC = 0.90
A3_MIN_ABS_RHO = 0.30
A4_MIN_R2 = 0.90
A5_MIN_ABS_CORR = 0.30
A6_MIN_CORR = 0.50
A6_PREDICTED_SLOPE = 0.115          # m/K, from a ~33 m layer at ~288 K


def _args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pack-root", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path,
                   help="e3sm_alldata_full.yaml -- channel + forcing names, pinned there")
    p.add_argument("--k56-h5", required=True, type=Path)
    p.add_argument("--nc-dir", required=True, type=Path, help="K=56 rollout NetCDFs")
    p.add_argument("--prev-csv", type=Path, default=None,
                   help="tendency_norm_probe.csv from job 7646192, for r_c")
    p.add_argument("--target", default="Z3_l17")
    p.add_argument("--partner", default="TREFHT")
    p.add_argument("--mask", default="auto", choices=("auto", "none", "land", "ocean"),
                   help="Restrict the target's spatial statistics. `auto` picks land when "
                        "the channel is exactly constant in time over ocean, i.e. a "
                        "land-only field carrying a fill value there. Unmasked numbers are "
                        "reported alongside, because the CONTRAST is the finding.")
    p.add_argument("--test-year", type=int, default=2048)
    p.add_argument("--n-ic", type=int, default=6, help="ICs for the A5 bias map")
    p.add_argument("--a6-samples", type=int, default=120)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--provenance", default="")
    return p.parse_args()


def _spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def _pearson(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _names(cfg: Path):
    txt = cfg.read_text()
    chan = json.loads(re.search(r"channel_names:\s*(\[.*?\])\s*\n", txt, re.S).group(1))
    forc = json.loads(re.search(r"forcing_channel_names:\s*(\[.*?\])\s*\n",
                                txt, re.S).group(1))
    return chan, forc


def _family(name: str) -> str:
    m = re.match(r"^(.*)_l\d\d$", name)
    return m.group(1) if m else name


def main() -> int:
    a = _args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    import h5py
    import xarray as xr

    chan, forc = _names(a.config)
    C = len(chan)
    S = a.pack_root / "stats"
    tm = np.load(S / "time_means.npy")[0].astype(np.float64)        # (C, H, W)
    ftm = np.load(S / "forcing_time_means.npy")[0].astype(np.float64)
    stds = np.load(S / "global_stds.npy").ravel().astype(np.float64)
    H, W = tm.shape[1], tm.shape[2]
    topo = ftm[forc.index("topo")]
    print(f"channels={C} grid={H}x{W} forcings={forc}")

    with h5py.File(a.k56_h5, "r") as f:
        bias = np.asarray(f["bias_mean"][:], float)          # (K, C)
        rmse = np.asarray(f["rmse_mean"][:], float)
        a_truth = np.asarray(f["a_truth_mean"][:], float)
        lead_h = np.asarray(f["lead_hours"][:], np.int64)
        kchan = [c.decode() if isinstance(c, bytes) else str(c) for c in f["channel"][:]]
    if kchan != chan:
        print("ERROR CHANNEL_ORDER_MISMATCH: k56 vs config")
        return 2
    with np.errstate(divide="ignore", invalid="ignore"):
        bias_frac = bias[-1] ** 2 / np.maximum(rmse[-1] ** 2, 1e-300)

    r_c = np.full(C, np.nan)
    if a.prev_csv and a.prev_csv.exists():
        by = {r["channel"]: r for r in csv.DictReader(a.prev_csv.open())}
        r_c = np.array([float(by[c]["r"]) if c in by else np.nan for c in chan])

    # ---------------- A1: static-to-forecastable ratio, every channel ----------------
    # sigma_static = spatial std of the per-cell TIME MEAN (the part that never moves)
    # a_truth      = truth's temporal anomaly amplitude (the part that is forecastable)
    sig_static = tm.std(axis=(1, 2))
    a_tr = np.nanmedian(a_truth, axis=0)              # lead-robust
    with np.errstate(divide="ignore", invalid="ignore"):
        Sc = sig_static / np.maximum(a_tr, 1e-300)
    order = np.argsort(-Sc)
    print("\n" + "=" * 78)
    print("A1 -- static-to-forecastable ratio  S_c = std_space(time_mean) / a_truth")
    print("=" * 78)
    print(f"{'channel':18s}{'S_c':>12s}{'sig_static':>12s}{'a_truth':>11s}"
          f"{'r_c':>11s}{'bias_frac':>10s}")
    for i in order[:15]:
        print(f"{chan[i]:18s}{Sc[i]:12.4g}{sig_static[i]:12.5g}{a_tr[i]:11.4g}"
              f"{r_c[i]:11.4g}{bias_frac[i]:10.4f}")
    peers = [chan[i] for i in order if Sc[i] > A1_S_THRESHOLD]
    tgt = chan.index(a.target)
    a1 = "OUTLIER CONFIRMED" if len(peers) < A1_MAX_PEERS else "FALSIFIED -- NOT AN OUTLIER"
    print(f"\nchannels with S_c > {A1_S_THRESHOLD}: {len(peers)} -> {peers}")
    print(f"{a.target}: S_c = {Sc[tgt]:.4g}  (rank {list(order).index(tgt) + 1} of {C})")
    print(f"A1 {a1}")

    # ---------------- A2: which channels are reconstructible from static inputs -------
    static_names = [n for n in ("lsm", "topo", "glacier", "natveg") if n in forc]
    X = np.stack([ftm[forc.index(n)].ravel() for n in static_names]
                 + [np.ones(H * W)], axis=1)
    XtXi = np.linalg.pinv(X.T @ X)
    r2 = np.zeros(C)
    for c in range(C):
        y = tm[c].ravel()
        beta = XtXi @ (X.T @ y)
        ss_res = float(((y - X @ beta) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2[c] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    static_like = [i for i in np.argsort(-r2) if r2[i] > A2_R2_STATIC]
    print("\n" + "=" * 78)
    print(f"A2 -- spatial structure reconstructible from {static_names} (R^2 > {A2_R2_STATIC})")
    print("=" * 78)
    print(f"{len(static_like)} of {C} channels clear it:")
    print(f"{'channel':18s}{'R2_static':>11s}{'S_c':>12s}{'r_c':>11s}{'bias_frac':>10s}")
    for i in static_like[:25]:
        print(f"{chan[i]:18s}{r2[i]:11.5f}{Sc[i]:12.4g}{r_c[i]:11.4g}{bias_frac[i]:10.4f}")
    print(f"corr(time_mean, topo) alone for {a.target}: "
          f"{_pearson(tm[tgt], topo):+.6f}")

    # ---------------- A3: does the H4 correlation survive without the Z3 family? ------
    fam = np.array([_family(c) for c in chan])
    keep = fam != _family(a.target)
    rho_all = _spearman(r_c, bias_frac)
    rho_noz3 = _spearman(r_c[keep], bias_frac[keep])
    # and a one-per-family subsample, since 18 levels of one variable are not 18 draws
    firsts = [np.where(fam == f)[0][0] for f in dict.fromkeys(fam)]
    rho_fam = _spearman(r_c[firsts], bias_frac[firsts])
    a3 = ("SURVIVES" if np.isfinite(rho_noz3) and abs(rho_noz3) >= A3_MIN_ABS_RHO
          else "FALSIFIED -- H4 WAS A Z3 ARTIFACT")
    print("\n" + "=" * 78)
    print("A3 -- is job 7646192's Spearman(-0.647) just the Z3 family?")
    print("=" * 78)
    print(f"all {C} channels              : {rho_all:+.4f}")
    print(f"excluding the {int((~keep).sum())} {_family(a.target)} levels "
          f"({int(keep.sum())} left): {rho_noz3:+.4f}   (need |rho| >= {A3_MIN_ABS_RHO})")
    print(f"one channel per family ({len(firsts)})  : {rho_fam:+.4f}")
    print(f"A3 {a3}")

    # ---------------- A4: is bias linear in lead? -------------------------------------
    k = lead_h.astype(float) / 6.0
    slope = np.zeros(C); r2_lin = np.zeros(C)
    for c in range(C):
        y = bias[:, c]
        ok = np.isfinite(y)
        if ok.sum() < 5:
            slope[c] = r2_lin[c] = np.nan
            continue
        m = float((k[ok] * y[ok]).sum() / (k[ok] ** 2).sum())        # through the origin
        ss_res = float(((y[ok] - m * k[ok]) ** 2).sum())
        ss_tot = float((y[ok] ** 2).sum())
        slope[c], r2_lin[c] = m, 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    a4 = ("SUPPORTED" if r2_lin[tgt] >= A4_MIN_R2
          else "FALSIFIED -- bias is not linear in lead")
    print("\n" + "=" * 78)
    print("A4 -- bias(k) = m*k, i.e. a constant per-step error integrating")
    print("=" * 78)
    print(f"{a.target}: slope {slope[tgt]:+.5g} per step, R^2 {r2_lin[tgt]:.4f} "
          f"(need >= {A4_MIN_R2})")
    top = np.argsort(-bias_frac)[:6]
    for i in top:
        print(f"  {chan[i]:18s} bias_frac {bias_frac[i]:.3f}  slope {slope[i]:+.4g}  "
              f"R2_linear {r2_lin[i]:.4f}")
    print(f"A4 {a4}")

    # ---------------- the mask, and whether this channel is land-only -----------------
    # Soil/land fields carry a FILL over ocean (converter: "0.0 for all land-only
    # fields"), so a cell there is exactly constant in time. Global spatial statistics on
    # such a channel are ~71 % constant by construction, which would flatter every
    # spread number the way `global_stds` flattered Z3_l17. Detect it, do not assume it.
    lsm = ftm[forc.index("lsm")] if "lsm" in forc else None
    tp = a.pack_root / "test" / f"{a.test_year}.h5"
    with h5py.File(tp, "r") as f:
        st, dg = f["fields_state"], f["fields_diagnostic"]
        n_state = st.shape[1]
        n = min(a.a6_samples, st.shape[0])

        def _read(ci: int) -> np.ndarray:
            """The pack splits the 101 channels across two datasets.

            `channel_names` is state-then-diagnostic, so index 100 (`PRECT`) is
            `fields_diagnostic[:, 0]`, not `fields_state[:, 100]` -- which is an
            IndexError, and was one until 2026-09-23 (job 7646367).
            """
            if ci < n_state:
                return st[:n, ci].astype(np.float64)
            return dg[:n, ci - n_state].astype(np.float64)

        zz = _read(tgt)
        tt = _read(chan.index(a.partner))
    const_in_time = zz.std(axis=0) == 0.0                 # exact, not a tolerance
    frac_const = float(const_in_time.mean())
    land = (lsm > 0.5) if lsm is not None else (topo > 1.0)
    mode = a.mask
    if mode == "auto":
        mode = "land" if frac_const > 0.25 else "none"
    sel = {"none": np.ones((H, W), bool), "land": land, "ocean": ~land}[mode]
    print(f"\nmask: requested={a.mask} -> using '{mode}'  "
          f"({int(sel.sum())}/{H * W} cells); the channel is EXACTLY constant in time on "
          f"{frac_const * 100:.1f}% of cells")
    if frac_const > 0.25:
        agree = float((const_in_time == (~land)).mean())
        print(f"  constant-in-time cells match the ocean mask on {agree * 100:.1f}% of the grid"
              f"; fill value there = {np.unique(zz[:, const_in_time])[:3]}")
    sig_static_masked = float(tm[tgt][sel].std())
    print(f"  {a.target}: sigma_static unmasked {sig_static[tgt]:.5g} -> masked "
          f"{sig_static_masked:.5g};  S_c unmasked {Sc[tgt]:.4g} -> masked "
          f"{sig_static_masked / max(a_tr[tgt], 1e-300):.4g}")

    # ---------------- A5: is the 336 h bias map terrain-structured? -------------------
    nc = sorted(a.nc_dir.glob("*.nc"))[: a.n_ic]
    gy, gx = np.gradient(topo)
    gradmag = np.sqrt(gy ** 2 + gx ** 2)
    acc_bias = np.zeros((H, W)); acc_pred = np.zeros((H, W)); acc_truth = np.zeros((H, W))
    acc_se = np.zeros((H, W))
    for path in nc:
        with xr.open_dataset(path, decode_timedelta=False) as ds:
            ci = [str(x) for x in ds["channel"].values].index(a.target)
            p = np.asarray(ds["prediction"][0, -1, ci].values, float)
            t = np.asarray(ds["truth"][0, -1, ci].values, float)
        acc_bias += (p - t); acc_pred += p; acc_truth += t; acc_se += (p - t) ** 2
    nn = max(len(nc), 1)
    bmap, pmap, tmap, semap = acc_bias / nn, acc_pred / nn, acc_truth / nn, acc_se / nn
    c_topo = _pearson(bmap[sel], topo[sel])
    c_grad = _pearson(bmap[sel], gradmag[sel])
    a5 = ("SUPPORTED" if max(abs(c_topo), abs(c_grad)) >= A5_MIN_ABS_CORR
          else "FALSIFIED -- the bias is not terrain-organised")
    print("\n" + "=" * 78)
    print(f"A5 -- is the {a.target} 336 h bias map organised by terrain? "
          f"({len(nc)} ICs, mask={mode})")
    print("=" * 78)
    print(f"corr(bias_map, topo)        = {c_topo:+.4f}")
    print(f"corr(bias_map, |grad topo|) = {c_grad:+.4f}   (need max |.| >= {A5_MIN_ABS_CORR})")
    print(f"bias mean/std: all {bmap.mean():+.5g} / {bmap.std():.5g}   "
          f"land {bmap[land].mean():+.5g} / {bmap[land].std():.5g}   "
          f"ocean {bmap[~land].mean():+.5g} / {bmap[~land].std():.5g}")
    print(f"A5 {a5}")

    # ---------------- A7: does the model respect the fill? ----------------------------
    # Only meaningful for a channel that IS filled somewhere. A model that manufactures
    # values where the truth is a constant fill is producing physically meaningless output
    # over that region -- and since these are fed-back state channels, it then eats it.
    a7 = None
    if frac_const > 0.25:
        w_lat = np.cos(np.deg2rad(np.linspace(-89.5, 89.5, H)))[:, None] * np.ones((1, W))
        off = const_in_time
        mse_tot = float((semap * w_lat).sum())
        mse_off = float((semap[off] * w_lat[off]).sum())
        print("\n" + "=" * 78)
        print(f"A7 -- does the model respect the fill outside the valid region?")
        print("=" * 78)
        print(f"truth over filled cells: mean {tmap[off].mean():+.6g}  "
              f"std {tmap[off].std():.6g}  (should be the constant fill)")
        print(f"pred  over filled cells: mean {pmap[off].mean():+.6g}  "
              f"std {pmap[off].std():.6g}  max|pred| {np.abs(pmap[off]).max():.6g}")
        print(f"valid-region truth std for scale: {tmap[~off].std():.6g}")
        print(f"share of the lat-weighted 336 h MSE coming from FILLED cells: "
              f"{100 * mse_off / max(mse_tot, 1e-300):.2f}%")
        a7 = {"frac_filled": frac_const,
              "pred_mean_on_fill": float(pmap[off].mean()),
              "pred_std_on_fill": float(pmap[off].std()),
              "valid_truth_std": float(tmap[~off].std()),
              "mse_share_from_fill_pct": 100 * mse_off / max(mse_tot, 1e-300)}

    # ---------------- A6: is the forecastable part redundant with the partner? --------
    za = zz - zz.mean(axis=0, keepdims=True)
    ta = tt - tt.mean(axis=0, keepdims=True)
    num = (za * ta).sum(axis=0)
    den = np.sqrt((za ** 2).sum(axis=0) * (ta ** 2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        cell_corr = np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)
        cell_slope = np.where((ta ** 2).sum(axis=0) > 0,
                              num / np.maximum((ta ** 2).sum(axis=0), 1e-300), np.nan)
    cell_corr = np.where(sel, cell_corr, np.nan)          # score only the valid region
    cell_slope = np.where(sel, cell_slope, np.nan)
    med_corr = float(np.nanmedian(cell_corr))
    med_slope = float(np.nanmedian(cell_slope))
    a6 = ("SUPPORTED" if med_corr >= A6_MIN_CORR
          else "FALSIFIED -- not explained by near-surface temperature")
    print("\n" + "=" * 78)
    print(f"A6 -- is {a.target}'s time variation just {a.partner}? ({n} samples, per cell)")
    print("=" * 78)
    print(f"median per-cell temporal corr = {med_corr:+.4f}   (need >= {A6_MIN_CORR})")
    print(f"median per-cell slope         = {med_slope:+.5f} m/K   "
          f"(thickness physics predicts ~{A6_PREDICTED_SLOPE:.3f})")
    print(f"corr quartiles: {np.nanpercentile(cell_corr, [25, 50, 75])}")
    print(f"A6 {a6}")

    # ---------------- artefacts + verdicts --------------------------------------------
    with (a.out_dir / "static_channel_audit.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["channel", "family", "sigma_static", "a_truth", "S_c", "R2_static",
                    "r_c", "bias_frac_336h", "bias_slope_per_step", "bias_R2_linear"])
        for i, nm in enumerate(chan):
            w.writerow([nm, fam[i], f"{sig_static[i]:.8g}", f"{a_tr[i]:.8g}",
                        f"{Sc[i]:.8g}", f"{r2[i]:.6f}", f"{r_c[i]:.8g}",
                        f"{bias_frac[i]:.8g}", f"{slope[i]:.8g}", f"{r2_lin[i]:.6f}"])
    res = {"provenance": a.provenance, "target": a.target,
           "A1": {"verdict": a1, "peers": peers, "S_target": float(Sc[tgt])},
           "A2": {"n_static_like": len(static_like),
                  "channels": [chan[i] for i in static_like]},
           "A3": {"verdict": a3, "rho_all": rho_all, "rho_excl_family": rho_noz3,
                  "rho_one_per_family": rho_fam},
           "A4": {"verdict": a4, "slope": float(slope[tgt]), "r2": float(r2_lin[tgt])},
           "A5": {"verdict": a5, "corr_topo": c_topo, "corr_gradtopo": c_grad,
                  "n_ic": len(nc)},
           "A6": {"verdict": a6, "median_corr": med_corr, "median_slope": med_slope,
                  "predicted_slope": A6_PREDICTED_SLOPE},
           "A7": a7, "mask": mode, "frac_constant_in_time": frac_const,
           "sigma_static_masked": sig_static_masked}
    (a.out_dir / "static_channel_audit.json").write_text(
        json.dumps(res, indent=2, sort_keys=True, default=str) + "\n")

    print("\n" + "=" * 78)
    print("VERDICTS (the audit running is not one of them)")
    print("=" * 78)
    print(f"  A1 {a.target} is an outlier ....... {a1}")
    print(f"  A2 static-like channels ......... {len(static_like)} of {C}")
    print(f"  A3 H4 survives without {_family(a.target):<8s} . {a3}")
    print(f"  A4 bias linear in lead .......... {a4}")
    print(f"  A5 bias is terrain-organised .... {a5}")
    print(f"  A6 redundant with {a.partner:<10s} .... {a6}")
    if a7 is not None:
        print(f"  A7 fill respected? .............. pred std on fill "
              f"{a7['pred_std_on_fill']:.4g} vs valid-region {a7['valid_truth_std']:.4g}; "
              f"{a7['mse_share_from_fill_pct']:.1f}% of MSE is outside the valid region")
    print(f"wrote {a.out_dir}/static_channel_audit.{{csv,json}}")
    print(f"STATIC_AUDIT_OK channels={C} ics={len(nc)} samples={n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
