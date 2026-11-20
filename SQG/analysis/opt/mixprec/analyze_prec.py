"""analyze_prec.py -- the training-outcome comparison of a network precision against ieee (PREDICTIONS_prec_outcome.md;
rules and thresholds fixed there before the job). numpy only (login node OK).

  python3 analyze_prec.py --ref <ieee arm dir> ... --cand <candidate arm dir> ... --prec bf16 --frames <frames dir>
                          --out <results dir> --which best|last [--steps 195250]

Every arm is one gate-D training run at dose 0 (gateD/train_dose.py), rolled out by gateD/rollout_dose.py in strict
fp32. Only the training precision differs. The ieee arms are gate D's a0_sA / a0_sB (jobs 7835786, 7843532, 7854209);
the candidate arms are this job's.

1. **Validity** (INVALID otherwise):
   - every arm finished --steps at dose 0;
   - the ref arms are ieee and the candidate arms --prec, with bf16 seen in the forward (forward_census);
   - every arm used the same training and val segments, val pairs and rollout initial conditions;
   - every seed has at least one arm of each kind.
2. **Validation loss.** M1 = the minimum val_clean, the val of best.pt (decides); M2 = the final val_clean (reported).
   - Delta = mean(cand) - mean(ref); sigma_ref = the SD of the ref arms (ddof 1): ieee's own retraining spread.
   - p = exact permutation p-value of |Delta|. Arm labels are shuffled within each seed (every split, counting the
     observed one).
3. **Rollouts** (gate D's quantities and windows, analyze_dose.py): R(w) for E_low and each of the 8 bands, per
   250-step window, up to the first step where any arm has fewer than half its initial conditions alive.
   - Cell statistic t_c = |mean_cand - mean_ref| / s_c, with s_c the SD over all arms (it does not change under
     relabelling).
   - Family statistic T = max over cells. p_FW is its exact stratified permutation p-value.
4. **Stability.** Blown-up initial conditions per arm.
5. **Verdict** (evaluated for --which; best.pt decides, last.pt is reported), first match:
   - INVALID;
   - UNSTABLE if a candidate arm blows up more initial conditions than the worst ref arm;
   - WORSE if p(M1) < 0.05 and Delta > 0;
   - ROLLOUT_DIFFERS if p_FW < 0.05;
   - BETTER if p(M1) < 0.05 and Delta < 0;
   - PASS_P if |Delta(M1)| <= sigma_ref;
   - otherwise INCONCLUSIVE (an effect above ieee's spread that the test cannot confirm).
6. **Information.**
   - s_step, peak memory (the candidate arms log it), overfit ratio;
   - RMS differences of the one-step predictions: within ref, cand vs ref, within cand.

Writes prec_summary.json. Prints PREC_* lines and PREC_ANALYSIS_DONE verdict=... last.
"""
import argparse
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GATED = os.path.join(os.path.dirname(HERE), "gateD")
sys.path.insert(0, GATED)
import dose_common as DC                                             # noqa: E402
import analyze_dose as AD                                            # noqa: E402  (climatology, overfit, window_means)

P_LEVEL, STEPS = 0.05, 195250
CONSISTENT = ("n_train_pairs", "train_segments", "val_segments", "n_val", "persistence_val", "batch", "steps")


def load_arm(d, which):
    """One arm: config, history, rollout arrays (E, blow, ic_tags, val_pick, preds) or a reason it is missing."""
    hf, cf, rf = (os.path.join(d, f) for f in ("history.json", "config.json", f"rollout_{which}.npz"))
    missing = [os.path.basename(f) for f in (hf, cf, rf) if not os.path.exists(f)]
    if missing:
        return {"dir": d, "missing": missing}
    with open(hf) as fh:
        h = json.load(fh)
    with open(cf) as fh:
        cfg = json.load(fh)
    with np.load(rf) as z:
        ro = {k: z[k] for k in ("E", "blow", "ic_tags", "val_pick", "preds")}
    return {"dir": d, "name": os.path.basename(d.rstrip("/")), "cfg": cfg, "hist": h["hist"], "ro": ro,
            "seed": int(cfg["seed"]), "prec": cfg.get("prec", "ieee")}


def splits(seeds_ref, seeds_cand):
    """Every relabelling that keeps each seed's number of candidate arms: lists of candidate index sets over the
    pooled arms (ref first, then cand), the observed split included."""
    pooled = list(seeds_ref) + list(seeds_cand)
    strata = sorted(set(pooled))
    per = []
    for s in strata:
        idx = [i for i, v in enumerate(pooled) if v == s]
        k = sum(1 for v in seeds_cand if v == s)
        per.append(list(itertools.combinations(idx, k)))
    return [tuple(sorted(i for part in combo for i in part)) for combo in itertools.product(*per)]


def perm_p(values, n_ref, all_splits, stat):
    """Exact permutation p-value: the share of splits with stat >= the observed one (observed = the last n_cand)."""
    v = np.asarray(values, dtype=np.float64)
    obs_idx = tuple(range(n_ref, len(v)))

    def s_of(idx):
        m = np.zeros(len(v), dtype=bool)
        m[list(idx)] = True
        return stat(v[~m], v[m])

    obs = s_of(obs_idx)
    null = np.array([s_of(idx) for idx in all_splits])
    return float(np.mean(null >= obs - 1e-12 * max(1.0, abs(obs)))), float(obs), len(all_splits)


def scalar_test(name, ref_vals, cand_vals, all_splits):
    ref_vals, cand_vals = np.asarray(ref_vals, float), np.asarray(cand_vals, float)
    delta = float(cand_vals.mean() - ref_vals.mean())
    sig = float(ref_vals.std(ddof=1)) if len(ref_vals) > 1 else float("nan")
    p, _, n = perm_p(np.concatenate([ref_vals, cand_vals]), len(ref_vals), all_splits,
                     lambda r, c: abs(c.mean() - r.mean()))
    return {"metric": name, "ref": ref_vals.tolist(), "cand": cand_vals.tolist(), "ref_mean": float(ref_vals.mean()),
            "cand_mean": float(cand_vals.mean()), "delta": delta, "delta_rel": delta / float(ref_vals.mean()),
            "sigma_ref": sig, "sigma_cand": float(cand_vals.std(ddof=1)) if len(cand_vals) > 1 else float("nan"),
            "z": delta / sig if sig else float("nan"), "p": p, "n_splits": n,
            "cand_in_ref_range": int(np.sum((cand_vals >= ref_vals.min()) & (cand_vals <= ref_vals.max())))}


def validity(ref, cand, prec, steps):
    bad = []
    for arm in ref + cand:
        if "missing" in arm:
            bad.append(f"{arm['dir']}: missing {arm['missing']}")
    if bad:
        return bad
    for arm in ref + cand:
        c = arm["cfg"]
        if int(arm["hist"][-1]["step"]) != steps or int(c["steps"]) != steps:
            bad.append(f"{arm['name']}: {arm['hist'][-1]['step']} of {steps} steps")
        if float(c["alpha"]) != 0.0:
            bad.append(f"{arm['name']}: alpha {c['alpha']}")
    for arm in ref:
        if arm["prec"] != "ieee":
            bad.append(f"{arm['name']}: ref arm has prec {arm['prec']}")
    for arm in cand:
        if arm["prec"] != prec:
            bad.append(f"{arm['name']}: candidate arm has prec {arm['prec']}, not {prec}")
        fc = arm["cfg"].get("forward_census") or {}
        if prec == "bf16" and not fc.get("bf16"):
            bad.append(f"{arm['name']}: no bf16 Conv/Linear output in the training forward ({fc})")
        if prec == "tf32" and "fp32_errors_after" not in (arm["cfg"].get("prec_record") or {}):
            bad.append(f"{arm['name']}: no tf32 proof in prec_record")
    first = ref[0] if ref else cand[0]
    for arm in ref + cand:
        for k in CONSISTENT:
            if arm["cfg"].get(k) != first["cfg"].get(k):
                bad.append(f"{arm['name']}: {k} differs from {first['name']}")
        for k in ("val_pick", "ic_tags"):
            if not np.array_equal(arm["ro"][k], first["ro"][k]):
                bad.append(f"{arm['name']}: rollout {k} differs from {first['name']}")
        if arm["ro"]["E"].shape[1:] != first["ro"]["E"].shape[1:]:
            bad.append(f"{arm['name']}: rollout E shape {arm['ro']['E'].shape}")
    seeds_r, seeds_c = {a["seed"] for a in ref}, {a["seed"] for a in cand}
    for s in seeds_r | seeds_c:
        if s not in seeds_r or s not in seeds_c:
            bad.append(f"seed {s} lacks a ref or a candidate arm")
    return bad


def rollout_cells(arms, clim, clim_low):
    """{quantity: [(window, [R per arm])]} on the common usable windows, plus the usable step count."""
    T = min(a["ro"]["E"].shape[0] for a in arms) - 1
    alive = [np.isfinite(a["ro"]["E"][:T + 1, :, 0]) for a in arms]
    bad = [int(np.argmax(al.mean(1) < AD.SURVIVE_MIN)) for al in alive if (al.mean(1) < AD.SURVIVE_MIN).any()]
    T_use = min([T] + [b - 1 for b in bad])
    n_win = max(0, T_use // AD.WINDOW)
    quantities = {"E_low": (lambda E: E[..., list(DC.LOW_BANDS)].sum(-1), clim_low)}
    for b, nm in enumerate(DC.BNAME):
        quantities[f"k{nm}"] = ((lambda E, b=b: E[..., b]), clim[b])
    cells = {}
    for q, (fn, cl) in quantities.items():
        per_arm = [AD.window_means(fn(a["ro"]["E"][:T_use + 1]) / cl - 1.0, al[:T_use + 1]) for a, al in
                   zip(arms, alive)]
        cells[q] = [[per_arm[i][w][0] for i in range(len(arms))] for w in range(n_win)]
    return cells, int(T_use), int(n_win)


def pred_rms(arms):
    """Median RMS of one-step prediction differences: within ref, cand vs ref, within cand."""
    out = {}
    P = {a["name"] + ("" if a["kind"] == "ref" else "*"): (a["kind"], a["ro"]["preds"]) for a in arms}
    rows = {"ref_ref": [], "cand_ref": [], "cand_cand": []}
    keys = list(P)
    for i, j in itertools.combinations(range(len(keys)), 2):
        ki, kj = P[keys[i]][0], P[keys[j]][0]
        d = float(np.sqrt(np.mean((P[keys[i]][1].astype(np.float64) - P[keys[j]][1].astype(np.float64)) ** 2)))
        tag = "ref_ref" if ki == kj == "ref" else ("cand_cand" if ki == kj == "cand" else "cand_ref")
        rows[tag].append(d)
    for k, v in rows.items():
        out[k] = {"median": float(np.median(v)) if v else None, "n": len(v)}
    return out


def decide(st, m1, fam):
    """The verdict of PREDICTIONS_prec_outcome.md, first match."""
    if st["max_cand_blown"] > st["max_ref_blown"]:
        return "UNSTABLE"
    if m1["p"] < P_LEVEL and m1["delta"] > 0:
        return "WORSE"
    if fam["p_fw"] is not None and fam["p_fw"] < P_LEVEL:
        return "ROLLOUT_DIFFERS"
    if m1["p"] < P_LEVEL and m1["delta"] < 0:
        return "BETTER"
    return "PASS_P" if abs(m1["delta"]) <= m1["sigma_ref"] else "INCONCLUSIVE"


def analyze(ref, cand, prec, clim, clim_low, which, steps=STEPS):
    """ref, cand: lists of load_arm() dicts. Returns the summary dict (verdict included)."""
    res = {"which": which, "prec": prec, "ref_arms": [a["dir"] for a in ref], "cand_arms": [a["dir"] for a in cand]}
    bad = validity(ref, cand, prec, steps)
    res["validity"] = bad
    if bad:
        res["verdict"] = "INVALID"
        return res
    for a in ref:
        a["kind"] = "ref"
    for a in cand:
        a["kind"] = "cand"
    arms = ref + cand
    all_splits = splits([a["seed"] for a in ref], [a["seed"] for a in cand])
    of = {a["dir"]: AD.overfit(a["hist"]) for a in arms}
    res["overfit"] = of
    m1 = scalar_test("val_best", [of[a["dir"]]["val_min"] for a in ref], [of[a["dir"]]["val_min"] for a in cand],
                     all_splits)
    m2 = scalar_test("val_final", [of[a["dir"]]["val_final"] for a in ref],
                     [of[a["dir"]]["val_final"] for a in cand], all_splits)
    tr = scalar_test("train_final", [of[a["dir"]]["train_final"] for a in ref],
                     [of[a["dir"]]["train_final"] for a in cand], all_splits)
    res["val"] = {"M1_val_best": m1, "M2_val_final": m2, "train_final": tr}

    cells, T_use, n_win = rollout_cells(arms, clim, clim_low)
    n_ref = len(ref)
    names, mat = [], []
    for q, rows in cells.items():
        for w, vals in enumerate(rows):
            names.append((q, w))
            mat.append(vals)
    fam = {"usable_steps": T_use, "windows": n_win, "cells": len(names), "p_fw": None, "T": None, "per_cell": []}
    if mat:
        M = np.asarray(mat, dtype=np.float64)                                    # (cells, arms)
        s = M.std(axis=1, ddof=1)
        s[s == 0] = np.inf

        def T_of(cand_idx):
            m = np.zeros(M.shape[1], dtype=bool)
            m[list(cand_idx)] = True
            return float(np.max(np.abs(M[:, m].mean(1) - M[:, ~m].mean(1)) / s))

        obs = T_of(tuple(range(n_ref, M.shape[1])))
        null = np.array([T_of(idx) for idx in all_splits])
        fam["T"], fam["p_fw"] = obs, float(np.mean(null >= obs - 1e-12 * max(1.0, obs)))
        sig_ref = M[:, :n_ref].std(axis=1, ddof=1)
        for (q, w), row, sr, sa in zip(names, M, sig_ref, s):
            d = float(row[n_ref:].mean() - row[:n_ref].mean())
            fam["per_cell"].append({"q": q, "window_TU": [w * AD.WINDOW * 0.02, (w + 1) * AD.WINDOW * 0.02],
                                    "ref_mean": float(row[:n_ref].mean()), "cand_mean": float(row[n_ref:].mean()),
                                    "delta": d, "sigma_ref": float(sr), "z_ref": d / float(sr) if sr else None,
                                    "t_all": abs(d) / float(sa)})
    res["rollout"] = fam
    st = {a["dir"]: {"blown": int((a["ro"]["blow"] >= 0).sum()), "n_ic": int(len(a["ro"]["blow"]))} for a in arms}
    res["stability"] = {"per_arm": st, "max_ref_blown": max(st[a["dir"]]["blown"] for a in ref),
                        "max_cand_blown": max(st[a["dir"]]["blown"] for a in cand)}
    info = {}
    for a in arms:
        h = a["hist"]
        ss = sorted(r["s_step"] for r in h[1:]) or [h[0]["s_step"]]
        info[a["dir"]] = {"kind": a["kind"], "seed": a["seed"], "prec": a["prec"], "s_step_median": ss[len(ss) // 2],
                          "peak_GiB": max((r.get("peak_GiB") or 0.0) for r in h) or None}
    res["speed_memory"] = info
    med = lambda k: float(np.median([v["s_step_median"] for v in info.values() if v["kind"] == k]))   # noqa: E731
    res["s_step_ratio_cand_over_ref"] = med("cand") / med("ref")
    res["one_step_pred_rms"] = pred_rms(arms)
    res["verdict"] = decide(res["stability"], m1, fam)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--prec", default="bf16")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--which", default="best", choices=("best", "last"))
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    G = DC.geometry()
    clim, _, n_clim = AD.climatology(a.frames, G)
    clim_low = float(clim[list(DC.LOW_BANDS)].sum())
    ref = [load_arm(d, a.which) for d in a.ref]
    cand = [load_arm(d, a.which) for d in a.cand]
    res = analyze(ref, cand, a.prec, clim, clim_low, a.which, a.steps)
    res["climatology_frames"] = n_clim
    with open(os.path.join(a.out, "prec_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    for b in res["validity"]:
        print(f"ERROR PREC_INVALID {b}", flush=True)
    if res["verdict"] != "INVALID":
        for k, m in res["val"].items():
            print(f"PREC_VAL {k} [{a.which}]: ieee {m['ref_mean']:.6f} (sd {m['sigma_ref']:.2e}, n {len(m['ref'])}) "
                  f"{a.prec} {m['cand_mean']:.6f} (n {len(m['cand'])}) delta {m['delta']:+.2e} "
                  f"({m['delta_rel']:+.2%}, {m['z']:+.2f} sd) p {m['p']:.3f}/{m['n_splits']} in-range "
                  f"{m['cand_in_ref_range']}/{len(m['cand'])}", flush=True)
        f = res["rollout"]
        worst = max(f["per_cell"], key=lambda c: c["t_all"]) if f["per_cell"] else None
        print(f"PREC_ROLLOUT [{a.which}] usable {f['usable_steps']} steps, {f['windows']} windows, {f['cells']} cells: "
              f"T {f['T']} p_FW {f['p_fw']}" + (f"; largest {worst['q']} {worst['window_TU'][0]:g}-"
                                                f"{worst['window_TU'][1]:g} TU delta {worst['delta']:+.3f} "
                                                f"({worst['z_ref']:+.2f} sd_ref)" if worst else ""), flush=True)
        s = res["stability"]
        print(f"PREC_STABILITY blown: max ieee {s['max_ref_blown']}, max {a.prec} {s['max_cand_blown']}", flush=True)
        pk = [v["peak_GiB"] for v in res["speed_memory"].values() if v["kind"] == "cand" and v["peak_GiB"]]
        print(f"PREC_SPEED s_step {a.prec}/ieee {res['s_step_ratio_cand_over_ref']:.3f}; {a.prec} peak "
              f"{max(pk) if pk else float('nan'):.2f} GiB; one-step pred rms "
              + " ".join(f"{k} {v['median']:.2e}" for k, v in res["one_step_pred_rms"].items() if v["median"]),
              flush=True)
    print(f"PREC_ANALYSIS_DONE verdict={res['verdict']} which={a.which}", flush=True)


if __name__ == "__main__":
    main()
