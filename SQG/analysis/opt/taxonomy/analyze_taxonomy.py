"""analyze_taxonomy.py -- the taxonomy job's analysis (item 3): noise vs bias of each solver variant's error in o1.2's
0.02 delta, per band, with two positive controls. numpy only. Thresholds are pre-registered in PREDICTIONS_taxonomy.md
and fixed in the constants below; do not change them after a run.

  python analyze_taxonomy.py --in <worker dir> --out <summary dir> --sealed-dir <dir outside the repo> [--unseal]

Inputs: taxonomy.py's pair_w*_s*_p*.npz, tables_w*.npz and worker_*.json. Notation, per pair (one start state S_j),
on a 256^2 rfft spectrum (`sub`: rfft2 of o1.2's float32 frame; `af`: the alias-free coarse spectrum):
  x = F_ref(0)  (the learner's input),  d = F_ref(L) - F_ref(0)  (the reference delta, lean_u1 = production),
  e = F_v(L) - F_ref(L)  (variant v's paired delta error), L = 4,000 steps (o1.2's pair).
1. e_delta(lag) = |e| / |d| overall (physical L2, as delta_fidelity.py) and per coarse |k| band; e_frame.
2. The regression e = alpha x + beta d (real coefficients; Re and Im parts as rows; one row pair per unique half-plane
   mode), with cluster-robust covariance (clusters = archived states, CR1 factor S/(S-1)):
   - level 1, pooled per band: a smooth bias (e.g. a time-step or splitting error, the planted shift). In units of
     the band's delta: B_x = alpha rms(x)/rms(d), B_d = beta; MDE = 2.8 SE (80% power, 5% two-sided) in the same units.
   - level 2, per shell (all modes with one k_x^2 + k_y^2): a mode-specific bias such as float32's quantized damping
     (the same rate error for every mode of a shell, random sign between shells). Wald W per shell, p from
     F(2, S-1) (exact for 2 numerator df: p = (1 + W/(S-1))^(-(S-1)/2)); per band the fraction of shells with
     p < P_SHELL (normal |z| > 3) and the debiased learnable bias sqrt(sum_n (th' A th - tr(V A)) / sum |d|^2).
3. Control a (mechanism A, dquant = float64 with only its decay rounded to float32): per shell its alpha should be the
   static prediction -drate_n x 0.02 (drate_n from the tables actually used). kappa_b = energy-weighted slope of the
   fitted vs predicted alpha over the band's shells; PASS if kappa_b in CTRL_A_RANGE for CTRL_A_BANDS (k < 8, where
   0.02 is short against the flow's time scales); higher bands are reported.
4. Control b (recovery and power): `inj` = lean32ek + a planted rate shift (per band, per worker drate_wb), `inj64` =
   the same shift in float64 (noise-free truth, first pairs). Level-1 regressions on regressors scaled by
   drate_wb / drate_0b, sub frames, every band:
   - recovery: alpha(e_inj - e_lean32ek) / alpha(e_inj64) on the same pairs in CTRL_B_RANGE (lean32ek's own coefficient
     cancels in the difference);
   - power: |alpha(e_inj64)| >= CTRL_B_Z x lean32ek's SE (all pairs): the planted bias would be seen at z >= 3.
   This paired form replaced "z(inj) >= 3 and alpha_inj / alpha_inj64 in range" after the harness smoke 7823653: that
   ratio contains lean32ek's own coefficient, so it would void the run exactly when lean32ek is biased
   (PREDICTIONS_taxonomy.md addendum; changed before the archive run).
5. Negative controls (f64ic32, lean32sr): their shell flag rates and band |z| > 3 counts, against nominal 0.27%.
Candidates (lean32ek = lean32ek_fast if bitwise, dt30, dt32): their bias statistics are SEALED into
<sealed-dir>/sealed_taxonomy.json (sha256 printed and kept in the summary) until jesswan / Weidong sign item 4's
gates; --unseal prints them. The dt order ratio is sealed too (with the unsealed D80 it reveals |D30|).
Prints TAXO_* lines and TAXO_ANALYSIS_DONE last.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys

import numpy as np

NC = 256
HALF = NC // 2
LAGS = (1, 10, 25, 50, 100)
STRIDE = 40
LAG_T = 0.02
DT = 5e-6
BANDS = ((1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 182))
BNAME = tuple(f"{a}-{b}" for a, b in BANDS)
# ---- pre-registered (PREDICTIONS_taxonomy.md) --------------------------------------------------------------------
Z_MDE = 2.8
P_SHELL = 0.0027
SHELL_FRAC_MAX = 0.01
CTRL_A_BANDS = (0, 1, 2)
CTRL_A_RANGE = (0.8, 1.2)
CTRL_B_Z = 3.0
CTRL_B_RANGE = (0.7, 1.3)
FAST_MS_MAX = 2.9
ORDER_RANGE = (-1.6, -0.55)
MDE_PRED = (8.0e-6, 8.8e-7, 2.6e-7, 1.2e-7, 1.2e-7, 1.2e-7, 1.2e-7, 1.2e-7)


# ---- geometry ---------------------------------------------------------------------------------------------------------
def band_of(kap):
    out = np.full(np.shape(kap), -1, dtype=np.int64)
    for i, (a, b) in enumerate(BANDS):
        out[(kap >= a) & (kap < b)] = i
    return out


def geometry():
    kx = np.arange(HALF + 1, dtype=np.float64)
    ky = np.fft.fftfreq(NC, 1.0 / NC)
    KX, KY = np.meshgrid(kx, ky)                                     # (256, 129): rows k_y, columns k_x (rfft2 layout)
    kap = np.sqrt(KX ** 2 + KY ** 2)
    shell = (KX ** 2 + KY ** 2).astype(np.int64)
    unique = ((KX > 0) & (KX < HALF)) | (((KX == 0) | (KX == HALF)) & (KY > 0))
    af_valid = (np.abs(KX) <= HALF - 1) & (np.abs(KY) <= HALF - 1)
    return {"KX": KX, "KY": KY, "kap": kap, "shell": shell, "band": band_of(kap), "unique": unique,
            "af_valid": af_valid}


# ---- regressions ------------------------------------------------------------------------------------------------------
def _bc(idx, w, n):
    return np.bincount(idx, weights=w, minlength=n)[:n]


def regress2(gid, ngroups, cl, ncl, x, d, e):
    """Per group g: e = alpha x + beta d, real coefficients, complex rows. gid (P, Q) group of each (pair, mode) or
    -1; cl (P,) cluster of each pair. Returns a dict of per-group arrays (coef (G, 2), cov (G, 2, 2), A (G, 2, 2),
    S (clusters with data), ok)."""
    m = gid >= 0
    g = gid[m]
    c = np.broadcast_to(np.asarray(cl).reshape((-1,) + (1,) * (gid.ndim - 1)), gid.shape)[m]
    xx, dd, ee = x[m], d[m], e[m]
    a11, a12, a22 = _bc(g, np.abs(xx) ** 2, ngroups), _bc(g, np.real(np.conj(xx) * dd), ngroups), \
        _bc(g, np.abs(dd) ** 2, ngroups)
    b1, b2 = _bc(g, np.real(np.conj(xx) * ee), ngroups), _bc(g, np.real(np.conj(dd) * ee), ngroups)
    det = a11 * a22 - a12 ** 2
    ok = det > 1e-12 * np.maximum(a11 * a22, 1e-300)
    sdet = np.where(ok, det, 1.0)
    alpha = np.where(ok, (a22 * b1 - a12 * b2) / sdet, 0.0)
    beta = np.where(ok, (a11 * b2 - a12 * b1) / sdet, 0.0)
    u = ee - alpha[g] * xx - beta[g] * dd
    key = c * ngroups + g
    s1 = _bc(key, np.real(np.conj(xx) * u), ncl * ngroups).reshape(ncl, ngroups)
    s2 = _bc(key, np.real(np.conj(dd) * u), ncl * ngroups).reshape(ncl, ngroups)
    present = _bc(key, np.ones_like(g, dtype=np.float64), ncl * ngroups).reshape(ncl, ngroups) > 0
    S = present.sum(axis=0)
    m11, m12, m22 = (s1 * s1).sum(0), (s1 * s2).sum(0), (s2 * s2).sum(0)
    i11, i12, i22 = a22 / sdet, -a12 / sdet, a11 / sdet
    k = np.where(S > 1, S / np.maximum(S - 1, 1), np.nan)
    v11 = k * (i11 * i11 * m11 + 2 * i11 * i12 * m12 + i12 * i12 * m22)
    v12 = k * (i11 * i12 * m11 + (i11 * i22 + i12 * i12) * m12 + i12 * i22 * m22)
    v22 = k * (i12 * i12 * m11 + 2 * i12 * i22 * m12 + i22 * i22 * m22)
    coef = np.stack([alpha, beta], 1)
    cov = np.stack([np.stack([v11, v12], 1), np.stack([v12, v22], 1)], 1)
    A = np.stack([np.stack([a11, a12], 1), np.stack([a12, a22], 1)], 1)
    return {"coef": coef, "cov": cov, "A": A, "S": S, "ok": ok & (S > 2)}


def wald_p(fit):
    """Per group: W = th' V^-1 th and p = (1 + W/(S-1))^(-(S-1)/2), the F(2, S-1) tail (W/2 ~ F(2, S-1))."""
    th, V, S = fit["coef"], fit["cov"], fit["S"].astype(np.float64)
    det = V[:, 0, 0] * V[:, 1, 1] - V[:, 0, 1] ** 2
    good = fit["ok"] & (det > 0)
    sdet = np.where(good, det, 1.0)
    W = (V[:, 1, 1] * th[:, 0] ** 2 - 2 * V[:, 0, 1] * th[:, 0] * th[:, 1] + V[:, 0, 0] * th[:, 1] ** 2) / sdet
    df = np.maximum(S - 1, 1)
    p = np.where(good, (1 + W / df) ** (-df / 2), np.nan)
    return np.where(good, W, np.nan), p


def band_rms(v, mask_b):
    return float(np.sqrt(np.mean(np.abs(v[:, mask_b]) ** 2))) if mask_b.any() else float("nan")


def level1(G, cl, ncl, x, d, e, valid):
    """Per band pooled regression -> per band dict with B_x, B_d, z, MDE (delta units)."""
    gid = np.where(valid & G["unique"] & (G["band"] >= 0), G["band"], -1)
    gid = np.broadcast_to(gid, x.shape)
    fit = regress2(gid, len(BANDS), cl, ncl, x, d, e)
    out = {}
    for b in range(len(BANDS)):
        mb = (gid[0] == b)
        rx, rd = band_rms(x, mb), band_rms(d, mb)
        if not fit["ok"][b] or not (rd > 0):
            out[BNAME[b]] = None
            continue
        sa, sb = math.sqrt(fit["cov"][b, 0, 0]), math.sqrt(fit["cov"][b, 1, 1])
        a, bb = fit["coef"][b]
        sc = rx / rd
        out[BNAME[b]] = {"alpha": a, "beta": bb, "se_alpha": sa, "se_beta": sb, "B_x": a * sc, "B_d": bb,
                         "z_x": a / sa if sa > 0 else float("nan"), "z_d": bb / sb if sb > 0 else float("nan"),
                         "MDE_x": Z_MDE * sa * sc, "MDE_d": Z_MDE * sb, "S": int(fit["S"][b]), "rx_over_rd": sc}
    return out


def shell_ids(G, valid):
    m = valid & G["unique"] & (G["band"] >= 0)
    sh = G["shell"][m]
    uniq, inv = np.unique(sh, return_inverse=True)
    gid = np.full(G["shell"].shape, -1, dtype=np.int64)
    gid[m] = inv
    band_of_shell = band_of(np.sqrt(uniq.astype(np.float64)))
    return gid, uniq, band_of_shell


def level2(G, cl, ncl, x, d, e, valid):
    """Per shell regression -> per band: n_shells, frac(p < P_SHELL), debiased learnable bias and its noise level
    (delta units). Also returns the per-shell fit (for control a)."""
    gid2, uniq, bos = shell_ids(G, valid)
    gid = np.broadcast_to(gid2, x.shape)
    fit = regress2(gid, len(uniq), cl, ncl, x, d, e)
    W, p = wald_p(fit)
    th, V, A = fit["coef"], fit["cov"], fit["A"]
    sig = np.einsum("gi,gij,gj->g", th, A, th)
    noise = np.einsum("gij,gji->g", V, A)
    out = {}
    for b in range(len(BANDS)):
        sel = (bos == b) & fit["ok"] & np.isfinite(p)
        mb = (gid2 >= 0) & (G["band"] == b)
        dd = float(np.sum(np.abs(d[:, mb]) ** 2))
        if not sel.any() or dd <= 0:
            out[BNAME[b]] = None
            continue
        lb = float(np.sum(sig[sel] - noise[sel]))
        out[BNAME[b]] = {"n_shells": int(sel.sum()), "frac_p": float(np.mean(p[sel] < P_SHELL)),
                         "learnable_bias": math.sqrt(max(lb, 0.0) / dd), "learnable_signed_sq": lb / dd,
                         "noise_level": math.sqrt(float(np.sum(noise[sel])) / dd)}
    return out, {"fit": fit, "p": p, "shells": uniq, "band_of_shell": bos, "gid": gid2}


def edelta_bands(G, x_unused, d, e, valid):
    """Per band |e| / |d| over all (not only unique) modes, as delta_fidelity.py, pooled over pairs (median of
    per-pair ratios)."""
    out = {}
    for b in range(len(BANDS)):
        mb = valid & (G["band"] == b)
        if not mb.any():
            out[BNAME[b]] = None
            continue
        num = np.sum(np.abs(e[:, mb]) ** 2, axis=1)
        den = np.sum(np.abs(d[:, mb]) ** 2, axis=1)
        r = np.sqrt(num[den > 0] / den[den > 0])
        out[BNAME[b]] = float(np.median(r)) if r.size else None
    return out


# ---- loading ----------------------------------------------------------------------------------------------------------
def load(indir):
    workers = {}
    for f in sorted(glob.glob(os.path.join(indir, "worker_*.json"))):
        with open(f) as fh:
            w = json.load(fh)
        workers[int(w["worker"])] = w
    tables = {}
    for f in sorted(glob.glob(os.path.join(indir, "tables_w*.npz"))):
        w = int(re.search(r"tables_w(\d+)\.npz", f).group(1))
        with np.load(f) as z:
            tables[w] = {k: z[k] for k in z.files}
    pairs = []
    for f in sorted(glob.glob(os.path.join(indir, "pair_w*_s*_p*.npz"))):
        w, s, j = (int(v) for v in re.search(r"pair_w(\d+)_s(\d+)_p(\d+)\.npz", f).groups())
        with np.load(f) as z:
            pairs.append({"w": w, "s": s, "j": j, "data": {k: z[k] for k in z.files}})
    return workers, tables, pairs


def arm_arrays(pairs, arm, kind, lag, clusters):
    """Stacked (x, d, e, cl, workers) of `arm` vs lean_u1 at `lag` (frames index) over the pairs that have it."""
    xs, ds, es, cls, ws = [], [], [], [], []
    for p in pairs:
        D = p["data"]
        key = f"{arm}__{kind}__{lag}"
        if key not in D:
            continue
        if kind == "sub":
            r0 = np.fft.rfft2(D[f"lean_u1__sub__0"].astype(np.float64))
            rL = np.fft.rfft2(D[f"lean_u1__sub__{lag}"].astype(np.float64))
            vL = np.fft.rfft2(D[key].astype(np.float64))
        else:
            r0, rL, vL = D["lean_u1__af__0"], D[f"lean_u1__af__{lag}"], D[key]
        xs.append(r0)
        ds.append(rL - r0)
        es.append(vL - rL)
        cls.append(clusters[(p["w"], p["s"])])
        ws.append(p["w"])
    if not xs:
        return None
    return np.stack(xs), np.stack(ds), np.stack(es), np.asarray(cls), np.asarray(ws)


def phys_edelta(pairs, arm, lag):
    """Overall e_delta and e_frame (physical L2 on o1.2's float32 frames), per pair."""
    ed, ef = [], []
    for p in pairs:
        D = p["data"]
        key = f"{arm}__sub__{lag}"
        if key not in D:
            continue
        r0 = D["lean_u1__sub__0"].astype(np.float64)
        rL = D[f"lean_u1__sub__{lag}"].astype(np.float64)
        vL = D[key].astype(np.float64)
        ed.append(np.linalg.norm(vL - rL) / np.linalg.norm(rL - r0))
        ef.append(np.linalg.norm(vL - rL) / np.linalg.norm(rL))
    return (float(np.median(ed)) if ed else None), (float(np.median(ef)) if ef else None), len(ed)


# ---- main -------------------------------------------------------------------------------------------------------------
def fmt(v, f=".2e"):
    return "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sealed-dir", required=True)
    ap.add_argument("--unseal", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.sealed_dir, exist_ok=True)
    workers, tables, pairs = load(args.indir)
    if not pairs:
        raise SystemExit("ERROR TAXO_NO_PAIRS")
    G = geometry()
    clusters = {}
    for p in pairs:
        clusters.setdefault((p["w"], p["s"]), len(clusters))
    ncl = len(clusters)
    L = LAGS[-1]
    allv = np.ones_like(G["unique"])
    summ = {"n_pairs": len(pairs), "n_states": ncl, "workers": sorted(workers), "thresholds": {
        "Z_MDE": Z_MDE, "P_SHELL": P_SHELL, "SHELL_FRAC_MAX": SHELL_FRAC_MAX, "CTRL_A_BANDS": CTRL_A_BANDS,
        "CTRL_A_RANGE": CTRL_A_RANGE, "CTRL_B_Z": CTRL_B_Z, "CTRL_B_RANGE": CTRL_B_RANGE, "FAST_MS_MAX": FAST_MS_MAX,
        "ORDER_RANGE": ORDER_RANGE}}
    sealed = {}
    print(f"TAXO_DATA pairs={len(pairs)} states={ncl} workers={sorted(workers)}", flush=True)

    # ---- facts: states, bitwise, timing ------------------------------------------------------------------------
    states = [{"worker": w, **{k: s[k] for k in ("member", "t", "spe", "cfl")}} for w, W in workers.items()
              for s in W["states"]]
    summ["states"] = states
    bw = {w: W.get("bitwise", {}) for w, W in workers.items()}
    prod_ok = all(all(v.get("prod_vs_lean_u1_frames", {"-": False}).values()) for v in bw.values())
    fast_bit = all(all(v.get("fast_vs_lean32ek_state", {"-": False}).values()) for v in bw.values())
    dt40_ok = all(all(r["frame"] and r["state_hash"] for r in v.get("dt40_vs_lean_u1", [])) and
                  v.get("dt40_vs_lean_u1") for v in bw.values())
    ms = {}
    for k in ("lean_u1", "lean32ek", "lean32ek_fast", "lean32sr", "lean32"):
        vals = [W["ms_per_step"][k] for W in workers.values() if k in W.get("ms_per_step", {})]
        ms[k] = float(np.median(vals)) if vals else None
    summ.update({"bitwise": {"prod_eq_lean_u1": prod_ok, "fast_eq_lean32ek": fast_bit, "dt40_eq_lean_u1": dt40_ok},
                 "ms_per_step": ms})
    print(f"TAXO_CONTROL bitwise prod==lean_u1 {prod_ok}; lean32ek_fast==lean32ek {fast_bit}; dt40==lean_u1 "
          f"{dt40_ok}", flush=True)
    print("TAXO_BENCH ms/step (median over workers): " + " ".join(f"{k}={fmt(v, '.3f')}" for k, v in ms.items()),
          flush=True)
    fast_pass = bool(fast_bit and ms["lean32ek_fast"] is not None and ms["lean32ek_fast"] <= FAST_MS_MAX)
    summ["fast_form_pass"] = fast_pass
    print(f"TAXO_FAST {'PASS' if fast_pass else 'FAIL'} bitwise={fast_bit} ms={fmt(ms['lean32ek_fast'], '.3f')} "
          f"(gate <= {FAST_MS_MAX})", flush=True)

    # ---- e_delta tables (sizes; candidates' sizes are not bias statistics, except the dt arms) ----------------
    arms_size = ("lean32ek", "lean32", "lean32sr", "f64ic32", "dquant", "inj", "inj64")
    ed = {}
    for arm in arms_size:
        row = {"overall": {}, "e_frame": {}, "sub": {}, "af": {}, "rem": {}}
        for lag in LAGS:
            o, f, n = phys_edelta(pairs, arm, lag)
            row["overall"][lag * STRIDE], row["e_frame"][lag * STRIDE] = o, f
        for kind in ("sub", "af"):
            arr = arm_arrays(pairs, arm, kind, L, clusters)
            if arr is not None:
                row[kind] = edelta_bands(G, arr[0], arr[1], arr[2], G["af_valid"] if kind == "af" else allv)
                row["n_pairs"] = len(arr[3])
        a_s, a_a = arm_arrays(pairs, arm, "sub", L, clusters), arm_arrays(pairs, arm, "af", L, clusters)
        if a_s is not None and a_a is not None:
            row["rem"] = edelta_bands(G, None, a_s[1] - a_a[1], a_s[2] - a_a[2], allv)
        ed[arm] = row
    summ["e_delta"] = ed
    drms = []
    for p in pairs:
        D = p["data"]
        dlt = D[f"lean_u1__sub__{L}"].astype(np.float64) - D["lean_u1__sub__0"].astype(np.float64)
        drms.append(float(np.sqrt(np.mean(dlt ** 2))))
    summ["delta_rms"] = {"median": float(np.median(drms)), "min": float(np.min(drms)), "max": float(np.max(drms))}
    print(f"TAXO_DELTA rms(Delta) at 4,000 steps (o1.2 units): median {np.median(drms):.4f} "
          f"[{np.min(drms):.4f}, {np.max(drms):.4f}] over {len(drms)} pairs (synthetic job 7819310: 0.825)", flush=True)
    for arm in arms_size:
        r = ed[arm]
        print(f"TAXO_EDELTA {arm}: overall " + " ".join(f"{k}:{fmt(v)}" for k, v in r["overall"].items())
              + " | sub@4000 " + " ".join(f"k{n}={fmt(r['sub'].get(n))}" for n in BNAME)
              + " | af k1-2 " + fmt(r["af"].get("1-2")) + " rem k1-2 " + fmt(r["rem"].get("1-2")), flush=True)
    ek = ed["lean32ek"]
    alias_ratio = (ek["af"].get("1-2") / ek["sub"].get("1-2")) if ek["sub"].get("1-2") else None
    summ["alias_split_k12"] = alias_ratio
    print(f"TAXO_ALIAS lean32ek k1-2 alias-free / subsampled e_delta = {fmt(alias_ratio, '.3f')} "
          f"(prediction <= 0.3)", flush=True)

    # ---- per-shell error spectrum (sub, lag 4000): white vs red ----------------------------------------------
    spec = {}
    for arm in ("lean32ek", "lean32", "lean32sr", "f64ic32"):
        arr = arm_arrays(pairs, arm, "sub", L, clusters)
        if arr is None:
            continue
        e2 = np.mean(np.abs(arr[2]) ** 2, axis=0)
        per_band = [float(np.mean(e2[(G["band"] == b) & G["unique"]])) for b in range(len(BANDS))]
        spec[arm] = {"mean_abs_e2_per_band": dict(zip(BNAME, per_band)),
                     "spread_max_over_min": max(per_band) / min(per_band) if min(per_band) > 0 else None}
    summ["error_spectrum"] = spec
    print("TAXO_SPECTRUM per-mode |e|^2 spread over bands (max/min): "
          + " ".join(f"{k}={fmt(v['spread_max_over_min'], '.1f')}" for k, v in spec.items())
          + " (synthetic: lean32ek 5, lean32 260)", flush=True)

    # ---- MDE from lean32ek's own scatter (reported BEFORE any z-score) -----------------------------------------
    reg = {}
    for kind in ("sub", "af"):
        arr = arm_arrays(pairs, "lean32ek", kind, L, clusters)
        valid = G["af_valid"] if kind == "af" else allv
        reg[("lean32ek", kind)] = (level1(G, arr[3], ncl, arr[0], arr[1], arr[2], valid),
                                   level2(G, arr[3], ncl, arr[0], arr[1], arr[2], valid)[0])
    l1s = reg[("lean32ek", "sub")][0]
    mde = {n: (None if l1s[n] is None else {"MDE_x": l1s[n]["MDE_x"], "MDE_d": l1s[n]["MDE_d"],
                                            "pred_MDE_x": MDE_PRED[i]}) for i, n in enumerate(BNAME)}
    l2s = reg[("lean32ek", "sub")][1]
    for n in BNAME:
        if mde[n] is not None and l2s[n] is not None:
            mde[n]["shell_noise_level"] = l2s[n]["noise_level"]
    summ["mde_lean32ek_sub"] = mde
    print("TAXO_MDE lean32ek (sub, delta units, 2.8 SE) x-coef: " + " ".join(
        f"k{n}={fmt(mde[n]['MDE_x'] if mde[n] else None)}(pred {MDE_PRED[i]:.1e})" for i, n in enumerate(BNAME)),
        flush=True)

    # ---- control a: mechanism A pattern (dquant) ---------------------------------------------------------------
    ctrl_a = {}
    arr = arm_arrays(pairs, "dquant", "af", L, clusters)
    tab = tables[sorted(tables)[0]]
    with np.errstate(divide="ignore", invalid="ignore"):
        drate = -np.log1p((tab["decay32"] - tab["decay64"]) / tab["decay64"]) / DT
    alpha_pred_map = -drate * LAG_T
    if arr is not None:
        l2, det = level2(G, arr[3], ncl, arr[0], arr[1], arr[2], G["af_valid"])
        fit, sh, bos, gid2 = det["fit"], det["shells"], det["band_of_shell"], det["gid"]
        apred = np.zeros(len(sh))
        for i in range(len(sh)):
            vals = alpha_pred_map[gid2 == i]
            apred[i] = float(np.mean(vals)) if vals.size else np.nan
        w = fit["A"][:, 0, 0]
        for b in range(len(BANDS)):
            sel = (bos == b) & fit["ok"] & np.isfinite(apred)
            if not sel.any():
                ctrl_a[BNAME[b]] = None
                continue
            kap_b = float(np.sum(w[sel] * fit["coef"][sel, 0] * apred[sel]) / np.sum(w[sel] * apred[sel] ** 2))
            ctrl_a[BNAME[b]] = {"kappa": kap_b, "n_shells": int(sel.sum()),
                                "learnable_bias": l2[BNAME[b]]["learnable_bias"] if l2[BNAME[b]] else None}
        summ["control_a_pairs"] = int(len(arr[3]))
    a_pass = bool(ctrl_a) and all(ctrl_a.get(BNAME[b]) is not None and
                                  CTRL_A_RANGE[0] <= ctrl_a[BNAME[b]]["kappa"] <= CTRL_A_RANGE[1]
                                  for b in CTRL_A_BANDS)
    summ["control_a"] = {"bands": ctrl_a, "pass": a_pass}
    print(f"TAXO_CONTROL a (dquant, mechanism A; alpha_n vs -drate_n x 0.02) {'PASS' if a_pass else 'FAIL'}: kappa "
          + " ".join(f"k{n}={fmt(ctrl_a[n]['kappa'] if ctrl_a.get(n) else None, '.3f')}" for n in BNAME)
          + f" (gate {CTRL_A_RANGE} on k<8); learnable bias of mechanism A: "
          + " ".join(f"k{n}={fmt(ctrl_a[n]['learnable_bias'] if ctrl_a.get(n) else None)}" for n in BNAME[:4]),
          flush=True)

    # ---- control b: planted shift, recovery and power (sub), paired so lean32ek's own coefficient cancels -------
    drate_wb = {w: np.asarray(W["injection"]["drate_b"], dtype=np.float64) for w, W in workers.items()
                if W.get("injection")}
    w0 = sorted(drate_wb)[0]
    bidx = np.maximum(G["band"], 0)

    def scaled(x, d, ws):
        scale = np.stack([drate_wb[w] / np.where(drate_wb[w0] > 0, drate_wb[w0], 1.0) for w in ws])   # (P, bands)
        return x * scale[:, bidx], d * scale[:, bidx]

    ctrl_b = {}
    first = [p for p in pairs if f"inj64__sub__{L}" in p["data"]]
    a_in, a_ek, a_64 = (arm_arrays(first, a, "sub", L, clusters) for a in ("inj", "lean32ek", "inj64"))
    a_ek_all = arm_arrays(pairs, "lean32ek", "sub", L, clusters)
    b_pass = all(a is not None for a in (a_in, a_ek, a_64, a_ek_all))
    if b_pass:
        sx, sd = scaled(a_64[0], a_64[1], a_64[4])
        rec = level1(G, a_64[3], ncl, sx, sd, a_in[2] - a_ek[2], allv)    # the shift as recovered in lean32ek
        tru = level1(G, a_64[3], ncl, sx, sd, a_64[2], allv)              # the shift in float64 (truth)
        sxa, sda = scaled(a_ek_all[0], a_ek_all[1], a_ek_all[4])
        noise = level1(G, a_ek_all[3], ncl, sxa, sda, a_ek_all[2], allv)  # only its SE is used (never alpha)
        for n in BNAME:
            r, t, s = rec.get(n), tru.get(n), noise.get(n)
            if r is None or t is None or s is None or t["alpha"] == 0 or not s["se_alpha"] > 0:
                ctrl_b[n] = None
                b_pass = False
                continue
            ratio = r["alpha"] / t["alpha"]
            z_pow = abs(t["alpha"]) / s["se_alpha"]
            ok = z_pow >= CTRL_B_Z and CTRL_B_RANGE[0] <= ratio <= CTRL_B_RANGE[1]
            b_pass = b_pass and ok
            ctrl_b[n] = {"recovery_ratio": ratio, "z_power": z_pow, "B_x_truth": t["B_x"], "pass": bool(ok)}
    summ["control_b"] = {"bands": ctrl_b, "pass": b_pass, "drate_b": {str(w): v.tolist() for w, v in drate_wb.items()},
                         "form": "paired: alpha(inj - lean32ek) / alpha(inj64) on the same pairs; power = "
                                 "|alpha(inj64)| / SE(lean32ek, all pairs) (addendum, after smoke 7823653)"}
    print(f"TAXO_CONTROL b (planted shift: recovered in lean32ek vs float64 truth; power vs lean32ek's noise) "
          f"{'PASS' if b_pass else 'FAIL'}: "
          + " ".join(f"k{n}:{'ok' if (ctrl_b[n] and ctrl_b[n]['pass']) else 'FAIL'}"
                     f"(ratio={fmt(ctrl_b[n]['recovery_ratio'] if ctrl_b[n] else None, '.3f')},"
                     f"z={fmt(ctrl_b[n]['z_power'] if ctrl_b[n] else None, '.1f')})" for n in BNAME), flush=True)
    valid_run = bool(prod_ok and a_pass and b_pass)
    summ["valid"] = valid_run

    # ---- non-candidates: lean32, SR, mechanism B (unsealed) --------------------------------------------------
    nonc = {}
    for arm in ("lean32", "lean32sr", "f64ic32"):
        arr = arm_arrays(pairs, arm, "sub", L, clusters)
        if arr is None:
            continue
        nonc[arm] = {"level1": level1(G, arr[3], ncl, arr[0], arr[1], arr[2], allv),
                     "level2": level2(G, arr[3], ncl, arr[0], arr[1], arr[2], allv)[0]}
    a32, adq = arm_arrays(pairs, "lean32", "af", L, clusters), arm_arrays(pairs, "dquant", "af", L, clusters)
    if a32 is not None and adq is not None:
        ediff = a32[2] - adq[2]
        nonc["lean32_minus_dquant_af"] = {"level1": level1(G, a32[3], ncl, a32[0], a32[1], ediff, G["af_valid"]),
                                          "level2": level2(G, a32[3], ncl, a32[0], a32[1], ediff, G["af_valid"])[0]}
    summ["non_candidates"] = nonc
    # negative controls: f64ic32 (one float32 rounding of the start state) and lean32sr (unbiased rounding) carry no
    # learnable bias by construction, so their flag rates calibrate the tests (nominal 0.27% of shells, |z| > 3 in
    # 0.27% of band cells)
    calib = {}
    for arm in ("f64ic32", "lean32sr"):
        if arm not in nonc:
            continue
        l1, l2 = nonc[arm]["level1"], nonc[arm]["level2"]
        n_sh = sum(l2[n]["n_shells"] for n in BNAME[:6] if l2.get(n))
        n_fl = sum(l2[n]["n_shells"] * l2[n]["frac_p"] for n in BNAME[:6] if l2.get(n))
        zs = [abs(l1[n][k]) for n in BNAME if l1.get(n) for k in ("z_x", "z_d")]
        calib[arm] = {"shells": n_sh, "shell_flag_rate_k_lt_64": n_fl / n_sh if n_sh else None,
                      "band_cells": len(zs), "band_cells_abs_z_gt_3": int(sum(z > 3 for z in zs))}
    summ["negative_control_calibration"] = calib
    print("TAXO_CALIBRATION negative controls (nominal 0.0027): " + " ".join(
        f"{a}: shells {fmt(c['shell_flag_rate_k_lt_64'], '.4f')} of {c['shells']}, band |z|>3 "
        f"{c['band_cells_abs_z_gt_3']}/{c['band_cells']}" for a, c in calib.items()), flush=True)
    for arm, r in nonc.items():
        l1, l2 = r["level1"], r["level2"]
        print(f"TAXO_BIAS {arm}: z_x " + " ".join(f"k{n}={fmt(l1[n]['z_x'] if l1.get(n) else None, '.1f')}"
                                                  for n in BNAME)
              + " | shells p<0.0027 " + " ".join(f"k{n}={fmt(l2[n]['frac_p'] if l2.get(n) else None, '.3f')}"
                                                for n in BNAME[:4])
              + " | learnable bias k1-2 " + fmt(l2["1-2"]["learnable_bias"] if l2.get("1-2") else None), flush=True)

    # ---- candidates: SEALED --------------------------------------------------------------------------------------
    cand = {"lean32ek": {"level1_sub": reg[("lean32ek", "sub")][0], "level2_sub": reg[("lean32ek", "sub")][1],
                         "level1_af": reg[("lean32ek", "af")][0], "level2_af": reg[("lean32ek", "af")][1],
                         "lean32ek_fast_bitwise": fast_bit}}
    dt = {}
    for n in (30, 32, 80):
        for kind in ("sub", "af"):
            arr = arm_arrays(pairs, f"dt{n}", kind, L, clusters)
            if arr is None:
                continue
            valid = G["af_valid"] if kind == "af" else allv
            dt[(n, kind)] = arr
            if n == 80:
                continue
            cand.setdefault(f"dt{n}", {})[f"level1_{kind}"] = level1(G, arr[3], ncl, arr[0], arr[1], arr[2], valid)
            cand[f"dt{n}"][f"level2_{kind}"] = level2(G, arr[3], ncl, arr[0], arr[1], arr[2], valid)[0]
            cand[f"dt{n}"][f"e_delta_{kind}"] = edelta_bands(G, arr[0], arr[1], arr[2], valid)
    order = {}
    for kind in ("sub", "af"):
        if (30, kind) in dt and (80, kind) in dt:
            d30, d80 = dt[(30, kind)][2], dt[(80, kind)][2]
            row = {}
            for b in range(len(BANDS)):
                mb = (G["band"] == b) & (G["af_valid"] if kind == "af" else True)
                den = float(np.sum(np.abs(d30[:, mb]) ** 2))
                row[BNAME[b]] = float(np.sum(np.real(np.conj(d30[:, mb]) * d80[:, mb])) / den) if den > 0 else None
            order[kind] = row
    if (80, "sub") in dt:
        x, d, e80, cl, _ = dt[(80, "sub")]
        pte = {"e_delta_D80_sub": edelta_bands(G, x, d, e80, allv),
               "note": "D(2.5e-6) = Delta(dt/2) - Delta(dt); production's own error is -2 D80 (first order) to "
                       "-8/7 D80 (third order)"}
        if (80, "af") in dt:
            xa, da, ea, _, _ = dt[(80, "af")]
            pte["e_delta_D80_af"] = edelta_bands(G, xa, da, ea, G["af_valid"])
        summ["production_time_error_dt80"] = pte
        for kind in ("sub", "af"):
            if f"e_delta_D80_{kind}" in pte:
                print(f"TAXO_DT80 production time error |D(2.5e-6)|/|Delta| per band ({kind}): " + " ".join(
                    f"k{n}={fmt(v)}" for n, v in pte[f"e_delta_D80_{kind}"].items()), flush=True)
    cand["dt_order_ratio"] = order
    sealed["candidates"] = cand
    summ["dt_arms_valid"] = dt40_ok

    # ---- seal ----------------------------------------------------------------------------------------------------
    blob = json.dumps(jsonable(sealed), indent=1, sort_keys=True).encode()
    sha = hashlib.sha256(blob).hexdigest()
    spath = os.path.join(args.sealed_dir, "sealed_taxonomy.json")
    with open(spath, "wb") as fh:
        fh.write(blob)
    summ["sealed"] = {"path": spath, "sha256": sha, "contents": sorted(sealed["candidates"])}
    print(f"TAXO_SEALED sha256={sha} path={spath} (lean32ek, dt30, dt32 bias statistics; dt order ratio) -- open "
          f"after item 4's gates are signed", flush=True)
    if args.unseal:
        summ["unsealed"] = sealed
        for k in ("lean32ek", "dt30", "dt32"):
            c = cand.get(k)
            if not c:
                continue
            l1, l2 = c["level1_sub"], c["level2_sub"]
            print(f"TAXO_UNSEALED {k}: B_x " + " ".join(f"k{n}={fmt(l1[n]['B_x'] if l1.get(n) else None)}"
                                                       for n in BNAME)
                  + " | z_x " + " ".join(f"k{n}={fmt(l1[n]['z_x'] if l1.get(n) else None, '.1f')}" for n in BNAME)
                  + " | shells p<0.0027 " + " ".join(f"k{n}={fmt(l2[n]['frac_p'] if l2.get(n) else None, '.3f')}"
                                                    for n in BNAME), flush=True)
        print("TAXO_UNSEALED dt order ratio D80/D30: " + json.dumps(jsonable(order)), flush=True)

    with open(os.path.join(args.out, "taxonomy_summary.json"), "w") as fh:
        json.dump(jsonable(summ), fh, indent=1)
    print(f"TAXO_ANALYSIS_DONE valid={valid_run} control_a={a_pass} control_b={b_pass} prod_bitwise={prod_ok} "
          f"dt_valid={dt40_ok} fast={fast_pass}", flush=True)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else repr(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    return x


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:
        first = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ""
        print(f"ERROR TAXO_ANALYSIS_CRASHED {type(exc).__name__}: {first}", flush=True)
        raise
