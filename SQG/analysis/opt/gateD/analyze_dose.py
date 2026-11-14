"""analyze_dose.py -- gate D's analysis (item 4). Rules and thresholds are fixed in PREDICTIONS_gateD.md before the
run. numpy only (login node OK).

  python3 analyze_dose.py --run <run dir with the arm dirs> --frames <frames dir> --out <results dir> [--which best]

Arms (directory -> dose, seed): a0_sA, a0_sB (dose 0, two seeds), a1000_sA, a10000_sA (paired with a0_sA).

1. **Overfitting**, per arm, from history.json:
   - the minimum of val_clean and its step;
   - final / minimum (OVERFIT if ≥ 1.10);
   - final val_clean / final train.
2. **Absorption** of the injected bias, one step on the fixed held-out pairs:
     kappa_a = Σ (f_a − f_0A) · B_hat(x, y) / (dose_a · Σ B_hat · B_hat)
   Noise reference: the same projection of f_0B − f_0A, divided by each dose.
3. **Drift.**
   - r_a(t) = E_low(rollout) / E_low(climatology) − 1, the same for each band. Climatology: all stored frames, every
     0.02.
   - Usable time is up to the first step where any arm has fewer than half its initial conditions alive.
   - Windows: 250 steps = 5 time units.
   - R_a(w) is the mean of r over the alive initial conditions and the window's steps.
   - Noise: S(w) = |R_0B − R_0A| (seed spread); N(w) = max(S, 2 · SE_IC of R_0A).
   - Dose effect: Delta_a(w) = R_a − R_0A. Real-bias effect: Delta_1000(w) / 1000.
4. **Verdict** (PREDICTIONS_gateD.md):
   - PASS_D if |Delta_1000(w)| / 1000 < 0.1 · N(w) in every usable window, for E_low and every band.
   - INCONCLUSIVE_NO_POWER if the dose-10000 positive control neither absorbs (kappa < 0.3) nor drifts
     (|Delta_10000| ≤ N in every window).
   - INCONCLUSIVE_UNSTABLE if no full window is usable.
   - Otherwise FAIL_D.

Writes dose_summary.json. Prints DOSE_* lines and DOSE_ANALYSIS_DONE verdict=... last.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dose_common as DC                                             # noqa: E402

# doses: env DOSE_LO / DOSE_HI (the arm directories are a<DOSE_LO>_sA / a<DOSE_HI>_sA, as run_tr_gateD_polaris.pbs names
# them; PREDICTIONS_gates_tilerows.md, 2026-11-13); unset = job 7835786's 1000 / 10000.
_LO, _HI = os.environ.get("DOSE_LO", "1000"), os.environ.get("DOSE_HI", "10000")
ARMS = {"a0_sA": 0.0, "a0_sB": 0.0, f"a{_LO}_sA": float(_LO), f"a{_HI}_sA": float(_HI)}
REF, SEED_B, LOW_D, HIGH_D = "a0_sA", "a0_sB", f"a{_LO}_sA", f"a{_HI}_sA"
WINDOW, NOISE_FACTOR, SURVIVE_MIN, KAPPA_POWER, OVERFIT_RATIO = 250, 0.1, 0.5, 0.3, 1.10


def climatology(frames_dir, G, members=(1, 2, 3, 4)):
    _, arrs = DC.open_segments(frames_dir, list(members))
    es = [DC.band_energy_np(np.asarray(a[::DC.LAG], dtype=np.float64), G) for a in arrs]
    e = np.concatenate(es)
    return e.mean(0), e.std(0), len(e)


def overfit(hist):
    vc = np.array([h["val_clean"] for h in hist])
    i = int(np.argmin(vc))
    return {"val_min": float(vc[i]), "val_min_step": int(hist[i]["step"]), "val_final": float(vc[-1]),
            "final_over_min": float(vc[-1] / vc[i]), "train_final": float(hist[-1]["train"]),
            "val_over_train_final": float(vc[-1] / hist[-1]["train"]), "overfit": bool(vc[-1] / vc[i] >= OVERFIT_RATIO),
            "steps": int(hist[-1]["step"])}


def window_means(rel, alive):
    """rel (T+1, n) relative deviation; alive (T+1, n) -> per-window (mean over ICs, SE over ICs)."""
    T = rel.shape[0] - 1
    out = []
    for w in range(T // WINDOW):
        sl = slice(1 + w * WINDOW, 1 + (w + 1) * WINDOW)
        per_ic = np.array([np.nanmean(np.where(alive[sl, i], rel[sl, i], np.nan)) for i in range(rel.shape[1])])
        ok = np.isfinite(per_ic)
        m = float(np.mean(per_ic[ok])) if ok.any() else float("nan")
        se = float(np.std(per_ic[ok], ddof=1) / math.sqrt(ok.sum())) if ok.sum() > 1 else float("nan")
        out.append((m, se))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--which", default="best")
    ap.add_argument("--val-members", default="4")
    ap.add_argument("--n-val", type=int, default=512)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    G = DC.geometry()
    cx, cd = DC.bias_maps(G)
    clim, clim_sd, n_clim = climatology(a.frames, G)
    clim_low = clim[list(DC.LOW_BANDS)].sum()
    res = {"which": a.which, "climatology": {"band_mean": clim.tolist(), "band_sd": clim_sd.tolist(), "n_frames": n_clim,
                                            "low_mean": float(clim_low)}, "arms": {},
           "doses": {"low": ARMS[LOW_D], "high": ARMS[HIGH_D], "pattern": DC.PATTERN}}
    print(f"DOSE_CLIM frames={n_clim} E_low={clim_low:.4e} bands=" + " ".join(f"{v:.3e}" for v in clim), flush=True)

    ro, of = {}, {}
    for arm, dose in ARMS.items():
        d = os.path.join(a.run, arm)
        hf, rf = os.path.join(d, "history.json"), os.path.join(d, f"rollout_{a.which}.npz")
        if not (os.path.exists(hf) and os.path.exists(rf)):
            print(f"ERROR DOSE_ARM_MISSING {arm} history={os.path.exists(hf)} rollout={os.path.exists(rf)}", flush=True)
            continue
        with open(hf) as fh:
            of[arm] = overfit(json.load(fh)["hist"])
        with np.load(rf) as z:
            ro[arm] = {k: z[k] for k in z.files}
        o = of[arm]
        print(f"DOSE_OVERFIT {arm} dose={dose:g}: val_clean min {o['val_min']:.5f} at step {o['val_min_step']}, final "
              f"{o['val_final']:.5f} (x{o['final_over_min']:.3f}), train final {o['train_final']:.5f}, val/train "
              f"{o['val_over_train_final']:.2f} -> {'OVERFIT' if o['overfit'] else 'no'}", flush=True)
    res["overfit"] = of
    if REF not in ro:
        print("DOSE_ANALYSIS_DONE verdict=ERROR_NO_REFERENCE_ARM", flush=True)
        return

    # ---- absorption (one step) -----------------------------------------------------------------------------------
    # the val pairs are rebuilt from the indices the rollouts recorded (numpy Generator streams may differ across
    # versions: the job's sqg-env numpy 1.26 vs the login node's 1.17), and every arm must have used the same ones
    _, va_arr = DC.open_segments(a.frames, [int(m) for m in a.val_members.split(",")])
    seg, kk = DC.pair_index(va_arr)
    pick = ro[REF]["val_pick"]
    if any(not np.array_equal(r["val_pick"], pick) for r in ro.values()):
        print("ERROR DOSE_VAL_PICK_DIFFERS between arms", flush=True)
    xv = np.stack([va_arr[seg[i]][kk[i]] for i in pick]).astype(np.float64)
    yv = np.stack([va_arr[seg[i]][kk[i] + DC.LAG] for i in pick]).astype(np.float64)
    B = DC.bias_hat_np(xv, yv, cx, cd)
    bb = float(np.sum(B * B))
    f0 = ro[REF]["preds"].astype(np.float64)
    kap = {}
    for arm, dose in ARMS.items():
        if arm == REF or arm not in ro:
            continue
        proj = float(np.sum((ro[arm]["preds"].astype(np.float64) - f0) * B))
        if dose:
            kap[arm] = proj / (dose * bb)
        else:
            kap[arm] = {f"per_dose_{int(dd)}": proj / (dd * bb) for dd in (ARMS[LOW_D], ARMS[HIGH_D])}
    res["absorption"] = kap
    print("DOSE_ABSORB kappa: " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else
                                           f"{k}(noise ref)={json.dumps({kk: round(vv, 4) for kk, vv in v.items()})}"
                                           for k, v in kap.items()), flush=True)

    # ---- drift ---------------------------------------------------------------------------------------------------
    T = min(r["E"].shape[0] for r in ro.values()) - 1
    alive = {arm: np.isfinite(r["E"][:T + 1, :, 0]) for arm, r in ro.items()}
    surv = {arm: al.mean(1) for arm, al in alive.items()}
    bad = [int(np.argmax(s < SURVIVE_MIN)) for s in surv.values() if (s < SURVIVE_MIN).any()]
    T_use = min([T] + [b - 1 for b in bad])
    n_win = max(0, T_use // WINDOW)
    res["stability"] = {arm: {"blown": int((r["blow"] >= 0).sum()), "n_ic": int(len(r["blow"])),
                              "first_blow": int(r["blow"][r["blow"] >= 0].min()) if (r["blow"] >= 0).any() else -1}
                        for arm, r in ro.items()}
    res["usable_steps"], res["windows"] = int(T_use), int(n_win)
    print(f"DOSE_STABILITY usable_steps={T_use} windows={n_win} (of {T // WINDOW}) " + " ".join(
        f"{arm}: blown {s['blown']}/{s['n_ic']} first {s['first_blow']}" for arm, s in res["stability"].items()),
        flush=True)
    quantities = {"E_low": (lambda E: E[..., list(DC.LOW_BANDS)].sum(-1), clim_low)}
    for b, nm in enumerate(DC.BNAME):
        quantities[f"k{nm}"] = ((lambda E, b=b: E[..., b]), clim[b])
    drift, verdict_ok, power_drift = {}, True, False
    for q, (fn, cl) in quantities.items():
        R = {arm: window_means(fn(r["E"][:T_use + 1]) / cl - 1.0, alive[arm][:T_use + 1]) for arm, r in ro.items()}
        rows = []
        for w in range(n_win):
            r0, se0 = R[REF][w]
            S = abs(R[SEED_B][w][0] - r0) if SEED_B in R else float("nan")
            Nw = max(S if math.isfinite(S) else 0.0, 2 * se0 if math.isfinite(se0) else 0.0)
            d_lo = R[LOW_D][w][0] - r0 if LOW_D in R else float("nan")
            d_hi = R[HIGH_D][w][0] - r0 if HIGH_D in R else float("nan")
            real = abs(d_lo) / ARMS[LOW_D] if math.isfinite(d_lo) else float("nan")
            ok = math.isfinite(real) and real < NOISE_FACTOR * Nw
            verdict_ok &= ok
            power_drift |= math.isfinite(d_hi) and abs(d_hi) > Nw
            rows.append({"window_TU": [w * WINDOW * 0.02, (w + 1) * WINDOW * 0.02], "R_ref": r0, "SE_ref": se0,
                         "seed_spread": S, "noise": Nw, "delta_1000": d_lo, "delta_10000": d_hi,
                         "real_bias_effect": real, "pass": bool(ok),
                         "linearity_10000_over_1000": (d_hi / d_lo) if (math.isfinite(d_lo) and d_lo) else None})
        drift[q] = rows
    res["drift"] = drift
    for q in ("E_low",) + tuple(f"k{nm}" for nm in DC.BNAME):
        rows = drift[q]
        if not rows:
            continue
        print(f"DOSE_DRIFT {q}: " + " | ".join(
            f"{r['window_TU'][0]:g}-{r['window_TU'][1]:g}TU ref {r['R_ref']:+.3f} noise {r['noise']:.3f} "
            f"d1e3 {r['delta_1000']:+.4f} d1e4 {r['delta_10000']:+.4f} real {r['real_bias_effect']:.1e}"
            f"{'' if r['pass'] else ' FAIL'}" for r in rows), flush=True)
    kap_hi = kap.get(HIGH_D)
    power = (isinstance(kap_hi, float) and kap_hi >= KAPPA_POWER) or power_drift
    if n_win == 0:
        verdict = "INCONCLUSIVE_UNSTABLE"
    elif not power:
        verdict = "INCONCLUSIVE_NO_POWER"
    elif verdict_ok:
        verdict = "PASS_D"
    else:
        verdict = "FAIL_D"
    res["verdict"] = {"verdict": verdict, "power_absorb": bool(isinstance(kap_hi, float) and kap_hi >= KAPPA_POWER),
                      "power_drift": bool(power_drift), "all_windows_pass": bool(verdict_ok)}
    with open(os.path.join(a.out, "dose_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"DOSE_ANALYSIS_DONE verdict={verdict} power_absorb={res['verdict']['power_absorb']} "
          f"power_drift={power_drift} windows={n_win}", flush=True)


if __name__ == "__main__":
    main()
