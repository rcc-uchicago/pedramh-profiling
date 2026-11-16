"""gates_ab.py -- item 4's gates A and B (ACCEPTANCE_gates_AB.md, rules fixed before unsealing) on job 7824121, and
the scoring of PREDICTIONS_taxonomy.md P2 / P5. numpy only (login node OK, OPENBLAS_NUM_THREADS=1).

  python3 gates_ab.py --raw <taxonomy raw dir> --out <results dir>       # prints GATES_AB_DONE

1. The alias floor from PRODUCTION frames (lean_u1, all pairs): D_b, A_b, eps_floor_b = sqrt(A_b / D_b,valid),
   T_b = g_b eps_floor_b with g_b = 0.1 / sqrt(n_b). Checked against the protocol's table.
2. Gate A per arm: Var_b(e) / A_b < 0.01 in every band (same Parseval code).
3. The sealed file: its sha256 must equal SEALED_SHA, else ERROR and nothing is evaluated. Gate B per arm and band:
   B1 |B_x|, |B_d| < T_b; B2 level-2 learnable bias < T_b; B3 shell fraction <= p0 + 2 sqrt(p0 (1 - p0) / n) in
   bands with >= 1% of Delta's energy; power MDE_x, MDE_d < T_b (else INCONCLUSIVE). PASS / FAIL_FLOOR /
   INCONCLUSIVE.
4. P2 and P5 as written.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_taxonomy as AT                                         # noqa: E402  (geometry, bands; numpy only)

SEALED_SHA = "6f1d0e5153918a3b14faf6e29795d91263a032479121bf87a4417fb173b78be1"
N_B = (53, 56, 17, 4.9, 1.8, 1, 1, 1)                                 # band memory in 0.02 steps (TEAM_REPORT R4)
P0 = 11 / 2392                                                        # pooled negative-control shell rate (7824121)
GATE_A_FRAC = 0.01
ENERGY_MIN = 0.01                                                     # B3 applies in bands with >= 1% of Delta's energy
LAG = 100
# the protocol's table (ACCEPTANCE_gates_AB.md), to the digits shown there
PROTOCOL = {"eps_floor": (0.087, 0.033, 0.029, 0.042, 0.064, 0.132, 0.412, 0.730),
            "T": (1.2e-3, 4.4e-4, 7.1e-4, 1.9e-3, 4.7e-3, 1.3e-2, 4.1e-2, 7.3e-2)}
ARMS = ("lean32ek", "dt30", "dt32")


def sig(x, n):
    return float(f"{x:.{n - 1}e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    G = AT.geometry()
    nb = len(AT.BANDS)
    w = np.where((G["KX"] > 0) & (G["KX"] < AT.HALF), 2.0, 1.0)
    norm = 1.0 / AT.NC ** 4
    valid = G["af_valid"]
    masks = [G["band"] == b for b in range(nb)]

    def ms(spec, m):
        return norm * float(np.sum(w[m] * np.abs(spec[m]) ** 2))

    # ---- 1-2. floor + gate A (production frames; arm errors) -------------------------------------------------------
    D, Dv, A = np.zeros(nb), np.zeros(nb), np.zeros(nb)
    var = {arm: np.zeros(nb) for arm in ARMS}
    npairs = {arm: 0 for arm in ARMS}
    files = sorted(glob.glob(os.path.join(a.raw, "pair_w*_s*_p*.npz")))
    for fn in files:
        with np.load(fn) as z:
            s0 = np.fft.rfft2(z["lean_u1__sub__0"].astype(np.float64))
            sL = np.fft.rfft2(z[f"lean_u1__sub__{LAG}"].astype(np.float64))
            dsub = sL - s0
            dalias = np.where(valid, dsub - (z[f"lean_u1__af__{LAG}"] - z["lean_u1__af__0"]), 0.0)
            arm_e = {arm: np.fft.rfft2(z[f"{arm}__sub__{LAG}"].astype(np.float64)) - sL
                     for arm in ARMS if f"{arm}__sub__{LAG}" in z.files}
        for b in range(nb):
            D[b] += ms(dsub, masks[b])
            Dv[b] += ms(dsub, masks[b] & valid)
            A[b] += ms(dalias, masks[b] & valid)
        for arm, e in arm_e.items():
            npairs[arm] += 1
            for b in range(nb):
                var[arm][b] += ms(e, masks[b])
    n = len(files)
    D, Dv, A = D / n, Dv / n, A / n
    eps = np.sqrt(A / Dv)
    g = 0.1 / np.sqrt(np.asarray(N_B))
    T = g * eps
    share = D / D.sum()
    match = all(round(float(eps[b]), 3) == PROTOCOL["eps_floor"][b] for b in range(nb)) and \
        all(sig(float(T[b]), 2) == PROTOCOL["T"][b] for b in range(nb))
    print(f"GATES_FLOOR pairs={n} total MS(Delta)={D.sum():.4e} alias={A.sum():.4e}; protocol table "
          f"{'MATCHES' if match else 'DIFFERS'}", flush=True)
    for b, name in enumerate(AT.BNAME):
        print(f"  k{name:>7}: D_b {D[b]:.3e} A_b {A[b]:.3e} eps_floor {eps[b]:.3f} g_b {g[b]:.4f} T_b {T[b]:.2e} "
              f"energy {share[b]:.4f}", flush=True)
    res = {"floor": {"D_b": D.tolist(), "A_b": A.tolist(), "eps_floor": eps.tolist(), "g_b": g.tolist(),
                     "T_b": T.tolist(), "energy_share": share.tolist(), "protocol_match": match, "pairs": n},
           "gate_A": {}, "gate_B": {}, "bands": list(AT.BNAME)}
    for arm in ARMS:
        if not npairs[arm]:
            continue
        v = var[arm] / npairs[arm]
        ratio = v / A
        ok = [bool(r < GATE_A_FRAC) for r in ratio]
        res["gate_A"][arm] = {"pairs": npairs[arm], "var_b": v.tolist(), "var_over_floor": ratio.tolist(),
                              "pass_b": ok, "pass": all(ok)}
        print(f"GATE_A {arm} ({npairs[arm]} pairs) {'PASS' if all(ok) else 'FAIL'}: Var(e)/A_b " + " ".join(
            f"k{nm}={r:.2e}" for nm, r in zip(AT.BNAME, ratio)), flush=True)

    # ---- 3. the sealed file -----------------------------------------------------------------------------------------
    spath = os.path.join(a.raw, "sealed", "sealed_taxonomy.json")
    with open(spath, "rb") as fh:
        blob = fh.read()
    sha = hashlib.sha256(blob).hexdigest()
    if sha != SEALED_SHA:
        print(f"ERROR GATES_SEALED_SHA_MISMATCH {sha} != {SEALED_SHA}", flush=True)
        sys.exit(3)
    print(f"GATES_SEALED_OK sha256={sha} (opened 2026-11-10, user's decision)", flush=True)
    cand = json.loads(blob)["candidates"]
    res["sealed_sha256"] = sha
    for arm in ARMS:
        c = cand.get(arm)
        if not c:
            continue
        l1, l2 = c["level1_sub"], c["level2_sub"]
        per = {}
        for b, name in enumerate(AT.BNAME):
            r1, r2 = l1.get(name), l2.get(name)
            if r1 is None or r2 is None:
                per[name] = {"verdict": "NO_DATA"}
                continue
            Tb = float(T[b])
            b1 = abs(r1["B_x"]) < Tb and abs(r1["B_d"]) < Tb
            b2 = r2["learnable_bias"] < Tb
            applies = share[b] >= ENERGY_MIN
            lim = P0 + 2 * math.sqrt(P0 * (1 - P0) / r2["n_shells"])
            b3 = (r2["frac_p"] <= lim) if applies else None
            power = r1["MDE_x"] < Tb and r1["MDE_d"] < Tb
            if not power:                                         # the test could not see a bias of size T_b
                verdict = "INCONCLUSIVE"
            elif b1 and b2 and b3 is not False:
                verdict = "PASS"
            else:
                verdict = "FAIL_FLOOR"
            per[name] = {"verdict": verdict, "T_b": Tb, "B_x": r1["B_x"], "B_d": r1["B_d"], "z_x": r1["z_x"],
                         "z_d": r1["z_d"], "MDE_x": r1["MDE_x"], "MDE_d": r1["MDE_d"],
                         "learnable_bias": r2["learnable_bias"], "noise_level": r2["noise_level"],
                         "frac_p": r2["frac_p"], "n_shells": r2["n_shells"], "b3_applies": bool(applies),
                         "b3_limit": lim, "B1": bool(b1), "B2": bool(b2), "B3": b3, "power": bool(power)}
        overall = "PASS" if all(p["verdict"] == "PASS" for p in per.values()) else "NOT_PASS"
        res["gate_B"][arm] = {"bands": per, "overall": overall}
        print(f"GATE_B {arm} {overall}: " + " ".join(f"k{nm}={p['verdict']}" for nm, p in per.items()), flush=True)
        for nm, p in per.items():
            if p["verdict"] == "NO_DATA":
                continue
            print(f"    k{nm:>7}: T_b {p['T_b']:.2e} | B_x {p['B_x']:+.2e} B_d {p['B_d']:+.2e} (z {p['z_x']:+.1f} / "
                  f"{p['z_d']:+.1f}; MDE {p['MDE_x']:.1e} / {p['MDE_d']:.1e}) | learnable {p['learnable_bias']:.2e} "
                  f"(noise {p['noise_level']:.1e}) | shells {p['frac_p']:.3f} of {p['n_shells']}"
                  + (f" (limit {p['b3_limit']:.3f})" if p["b3_applies"] else " (B3 n/a)"), flush=True)

    # ---- 4. predictions P2, P5 ----------------------------------------------------------------------------------------
    l1, l2 = cand["lean32ek"]["level1_sub"], cand["lean32ek"]["level2_sub"]
    zs = all(abs(l1[nm]["z_x"]) < 3 and abs(l1[nm]["z_d"]) < 3 for nm in AT.BNAME if l1.get(nm))
    shells = all(l2[nm]["frac_p"] <= 0.01 for nm in AT.BNAME[:6] if l2.get(nm))
    lb = all(l2[nm]["learnable_bias"] <= l1[nm]["MDE_x"] for nm in AT.BNAME if l1.get(nm) and l2.get(nm))
    res["P2"] = {"z_all_lt_3": zs, "shells_le_1pct_to_k64": shells, "learnable_le_MDE": lb}
    print(f"SCORE P2 (lean32ek no detectable learnable bias): |z|<3 all bands {zs}; shells <=1% each band to k64 "
          f"{shells}; learnable <= MDE {lb}", flush=True)
    order = cand.get("dt_order_ratio", {}).get("sub", {})
    inr = {nm: (v is not None and -1.6 <= v <= -0.55) for nm, v in order.items()}
    res["P5"] = {"ratio_sub": order, "in_range": inr}
    print("SCORE P5 (dt order ratio D80/D30 in [-1.6, -0.55], sub): " + " ".join(
        f"k{nm}={(v if v is None else f'{v:+.3f}')}{'' if inr[nm] else '(out)'}" for nm, v in order.items())
        + f" -> {sum(inr.values())}/{len(inr)} in range", flush=True)
    with open(os.path.join(a.out, "gates_ab.json"), "w") as fh:
        json.dump(AT.jsonable(res), fh, indent=1)
    print(f"GATES_AB_DONE -> {os.path.join(a.out, 'gates_ab.json')}", flush=True)


if __name__ == "__main__":
    main()
