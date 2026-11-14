"""gates_ab_tr.py -- gates A and B (ACCEPTANCE_gates_AB.md: rules and thresholds unchanged) for tilerows_if25 (the
candidate) and if32_25 (reported), on tr_pairs.py's frames merged with job 7824121's stored production pairs; plus,
for information, each arm's and production's 0.02-delta error against the converged ref160; and tilerows_if25's
band-level bias pattern for gate D. numpy only (OPENBLAS_NUM_THREADS=1). PREDICTIONS_gates_tilerows.md first.

  python3 gates_ab_tr.py --taxo <7824121 raw dir> --new <tr_pairs out dir> --out <results dir>    # GATES_AB_TR_DONE

Validity (V_AB): the pipeline is the one that judged lean32ek. lean32ek's level-1 / level-2 sub statistics are
recomputed here with analyze_taxonomy's functions from the stored pairs and must equal the opened sealed file's
(sha256 checked) to 1e-9 relative; the alias floor must match the protocol table to its digits; every new pair's
regenerated lean_u1 frames must have been bitwise equal to the stored ones (tr_pairs.py's worker json).
Gate A per arm: Var_b(e) / A_b < 0.01 in every band, e = F_arm(100 frames) - F_lean_u1(4,000 steps) (sub).
Gate B per arm and band: B1 |B_x|, |B_d| < T_b; B2 learnable bias < T_b; B3 shell fraction <= p0 + 2 sqrt(p0 (1 - p0)
/ n) in bands with >= 1% of Delta's energy; power MDE_x, MDE_d < T_b (else INCONCLUSIVE). PASS / FAIL_FLOOR.
Information: e_b(X vs ref160) = rms_b(X_100 - ref_100) / rms_b(ref_100 - X_0) on the 16 first pairs, for X =
lean_u1, tilerows_if25, if32_25 (sub spectra), as the time-step probe measured.
Gate D input: bias_pattern_tilerows_if25.json (level-1 alpha / beta, the format of bias_pattern_lean32ek.json) and
size_ratio = rms(B_hat_tr) / rms(B_hat_lean32ek) over the 64 pairs (sub, lag 100).
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

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(OPT, "taxonomy"))
sys.path.insert(0, os.path.join(OPT, "gateD"))
import analyze_taxonomy as AT                                         # noqa: E402
import gates_ab as GA                                                 # noqa: E402  (constants: N_B, P0, PROTOCOL, ...)

LAG = 100
ARMS = ("tilerows_if25", "if32_25")
CANDIDATE = "tilerows_if25"
ORIG_KEYS = ("lean_u1__sub__0", f"lean_u1__sub__{LAG}", "lean_u1__af__0", f"lean_u1__af__{LAG}",
             f"lean32ek__sub__{LAG}")
PATTERN_EK = os.path.join(OPT, "gateD", "bias_pattern_lean32ek.json")


def load_merged(taxo, new):
    """Pairs as analyze_taxonomy.load builds them (sorted by file name), the stored production keys + the new arms."""
    pairs = []
    for f in sorted(glob.glob(os.path.join(taxo, "pair_w*_s*_p*.npz"))):
        name = os.path.basename(f)
        w, s, j = (int(v) for v in re.search(r"pair_w(\d+)_s(\d+)_p(\d+)\.npz", name).groups())
        with np.load(f) as z:
            data = {k: z[k] for k in ORIG_KEYS if k in z.files}
        nf = os.path.join(new, name)
        if os.path.exists(nf):
            with np.load(nf) as z:
                data.update({k: z[k] for k in z.files})
        pairs.append({"w": w, "s": s, "j": j, "name": name, "data": data, "new": os.path.exists(nf)})
    return pairs


def rules_b(l1, l2, T, share):
    """gates_ab.py's gate-B rule, per band (copied logic, same constants)."""
    per = {}
    for b, name in enumerate(AT.BNAME):
        r1, r2 = l1.get(name), l2.get(name)
        if r1 is None or r2 is None:
            per[name] = {"verdict": "NO_DATA"}
            continue
        Tb = float(T[b])
        b1 = abs(r1["B_x"]) < Tb and abs(r1["B_d"]) < Tb
        b2 = r2["learnable_bias"] < Tb
        applies = share[b] >= GA.ENERGY_MIN
        lim = GA.P0 + 2 * math.sqrt(GA.P0 * (1 - GA.P0) / r2["n_shells"])
        b3 = (r2["frac_p"] <= lim) if applies else None
        power = r1["MDE_x"] < Tb and r1["MDE_d"] < Tb
        if not power:
            verdict = "INCONCLUSIVE"
        elif b1 and b2 and b3 is not False:
            verdict = "PASS"
        else:
            verdict = "FAIL_FLOOR"
        per[name] = {"verdict": verdict, "T_b": Tb, "B_x": r1["B_x"], "B_d": r1["B_d"], "z_x": r1["z_x"],
                     "z_d": r1["z_d"], "MDE_x": r1["MDE_x"], "MDE_d": r1["MDE_d"], "alpha": r1["alpha"],
                     "beta": r1["beta"], "learnable_bias": r2["learnable_bias"], "noise_level": r2["noise_level"],
                     "frac_p": r2["frac_p"], "n_shells": r2["n_shells"], "b3_applies": bool(applies), "b3_limit": lim,
                     "B1": bool(b1), "B2": bool(b2), "B3": b3, "power": bool(power)}
    overall = "PASS" if all(p["verdict"] == "PASS" for p in per.values()) else "NOT_PASS"
    return per, overall


def rel_close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxo", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    G = AT.geometry()
    nb = len(AT.BANDS)
    w = np.where((G["KX"] > 0) & (G["KX"] < AT.HALF), 2.0, 1.0)
    norm = 1.0 / AT.NC ** 4
    valid = G["af_valid"]
    masks = [G["band"] == b for b in range(nb)]
    allv = np.ones_like(G["unique"])

    def ms(spec, m):
        return norm * float(np.sum(w[m] * np.abs(spec[m]) ** 2))

    pairs = load_merged(a.taxo, a.new)
    clusters = {}
    for p in pairs:
        clusters.setdefault((p["w"], p["s"]), len(clusters))
    ncl = len(clusters)
    res = {"pairs": len(pairs), "new_pairs": sum(p["new"] for p in pairs), "states": ncl, "validity": {}}
    print(f"TRAB_DATA pairs={len(pairs)} with new arms={res['new_pairs']} states={ncl}", flush=True)

    # ---- validity: regenerated production frames, sealed reproduction ---------------------------------------------
    bits = []
    for f in sorted(glob.glob(os.path.join(a.new, "worker_*.json"))):
        with open(f) as fh:
            bits += [b["equal"] for b in json.load(fh).get("bitwise_vs_7824121", [])]
    v_frames = bool(bits) and all(bits) and len(bits) == len(pairs)
    spath = os.path.join(a.taxo, "sealed", "sealed_taxonomy.json")
    with open(spath, "rb") as fh:
        blob = fh.read()
    sha_ok = hashlib.sha256(blob).hexdigest() == GA.SEALED_SHA
    sealed = json.loads(blob)["candidates"]["lean32ek"]
    arr = AT.arm_arrays(pairs, "lean32ek", "sub", LAG, clusters)
    l1e = AT.level1(G, arr[3], ncl, arr[0], arr[1], arr[2], allv)
    l2e = AT.level2(G, arr[3], ncl, arr[0], arr[1], arr[2], allv)[0]
    rep = []
    for nm in AT.BNAME:
        s1, s2 = sealed["level1_sub"].get(nm), sealed["level2_sub"].get(nm)
        if s1 is None or l1e.get(nm) is None:
            continue
        rep += [rel_close(l1e[nm][k], s1[k]) for k in ("B_x", "B_d", "z_x", "z_d", "MDE_x", "MDE_d")]
        rep += [rel_close(l2e[nm][k], s2[k]) for k in ("learnable_bias", "frac_p", "noise_level")]
    v_sealed = sha_ok and bool(rep) and all(rep)
    res["validity"].update({"V_frames_lean_u1_bitwise": v_frames, "n_bitwise_checks": len(bits),
                            "V_sealed_sha": sha_ok, "V_sealed_reproduced": v_sealed, "n_reproduced": len(rep)})
    print(f"TRAB_VALID lean_u1 regenerated == stored: {v_frames} ({len(bits)} pairs); sealed sha {sha_ok}; lean32ek "
          f"stats reproduced {sum(rep)}/{len(rep)} -> {v_sealed}", flush=True)

    # ---- floor + gate A ----------------------------------------------------------------------------------------------
    D, Dv, A = np.zeros(nb), np.zeros(nb), np.zeros(nb)
    var = {arm: np.zeros(nb) for arm in ARMS}
    npairs = {arm: 0 for arm in ARMS}
    for p in pairs:
        z = p["data"]
        s0 = np.fft.rfft2(z["lean_u1__sub__0"].astype(np.float64))
        sL = np.fft.rfft2(z[f"lean_u1__sub__{LAG}"].astype(np.float64))
        dsub = sL - s0
        dalias = np.where(valid, dsub - (z[f"lean_u1__af__{LAG}"] - z["lean_u1__af__0"]), 0.0)
        for b in range(nb):
            D[b] += ms(dsub, masks[b])
            Dv[b] += ms(dsub, masks[b] & valid)
            A[b] += ms(dalias, masks[b] & valid)
        for arm in ARMS:
            k = f"{arm}__sub__{LAG}"
            if k in z:
                e = np.fft.rfft2(z[k].astype(np.float64)) - sL
                npairs[arm] += 1
                for b in range(nb):
                    var[arm][b] += ms(e, masks[b])
    n = len(pairs)
    D, Dv, A = D / n, Dv / n, A / n
    eps = np.sqrt(A / Dv)
    g = 0.1 / np.sqrt(np.asarray(GA.N_B))
    T = g * eps
    share = D / D.sum()
    match = all(round(float(eps[b]), 3) == GA.PROTOCOL["eps_floor"][b] for b in range(nb)) and \
        all(GA.sig(float(T[b]), 2) == GA.PROTOCOL["T"][b] for b in range(nb))
    res["validity"]["V_floor_protocol_match"] = bool(match)
    res["floor"] = {"A_b": A.tolist(), "eps_floor": eps.tolist(), "T_b": T.tolist(), "energy_share": share.tolist()}
    print(f"TRAB_FLOOR protocol table {'MATCHES' if match else 'DIFFERS'}", flush=True)
    res["gate_A"], res["gate_B"] = {}, {}
    for arm in ARMS:
        if not npairs[arm]:
            continue
        v = var[arm] / npairs[arm]
        ratio = v / A
        ok = [bool(r < GA.GATE_A_FRAC) for r in ratio]
        res["gate_A"][arm] = {"pairs": npairs[arm], "var_over_floor": ratio.tolist(), "pass_b": ok, "pass": all(ok)}
        print(f"GATE_A {arm} ({npairs[arm]} pairs) {'PASS' if all(ok) else 'FAIL'}: Var(e)/A_b " + " ".join(
            f"k{nm}={r:.2e}" for nm, r in zip(AT.BNAME, ratio)), flush=True)

    # ---- gate B ------------------------------------------------------------------------------------------------------
    l1s = {}
    for arm in ARMS:
        arr = AT.arm_arrays(pairs, arm, "sub", LAG, clusters)
        if arr is None:
            continue
        l1 = AT.level1(G, arr[3], ncl, arr[0], arr[1], arr[2], allv)
        l2 = AT.level2(G, arr[3], ncl, arr[0], arr[1], arr[2], allv)[0]
        l1s[arm] = l1
        per, overall = rules_b(l1, l2, T, share)
        res["gate_B"][arm] = {"bands": per, "overall": overall}
        print(f"GATE_B {arm} {overall}: " + " ".join(f"k{nm}={p['verdict']}" for nm, p in per.items()), flush=True)
        for nm, p in per.items():
            if p["verdict"] == "NO_DATA":
                continue
            print(f"    k{nm:>7}: T_b {p['T_b']:.2e} | B_x {p['B_x']:+.2e} B_d {p['B_d']:+.2e} (z {p['z_x']:+.1f} / "
                  f"{p['z_d']:+.1f}) | learnable {p['learnable_bias']:.2e} | shells {p['frac_p']:.3f} of "
                  f"{p['n_shells']}" + (f" (limit {p['b3_limit']:.3f})" if p["b3_applies"] else " (B3 n/a)"),
                  flush=True)

    # ---- information: accuracy against the converged reference ----------------------------------------------------
    acc = {}
    firsts = [p for p in pairs if f"ref160__sub__{LAG}" in p["data"]]
    for X in ("lean_u1",) + ARMS:
        rows = []
        for p in firsts:
            z = p["data"]
            key = f"lean_u1__sub__{LAG}" if X == "lean_u1" else f"{X}__sub__{LAG}"
            if key not in z:
                continue
            r0 = np.fft.rfft2(z["lean_u1__sub__0"].astype(np.float64))
            ref = np.fft.rfft2(z[f"ref160__sub__{LAG}"].astype(np.float64))
            xL = np.fft.rfft2(z[key].astype(np.float64))
            rows.append([math.sqrt(ms(xL - ref, masks[b]) / ms(ref - r0, masks[b])) for b in range(nb)])
        if rows:
            acc[X] = {nm: float(np.median([r[b] for r in rows])) for b, nm in enumerate(AT.BNAME)}
    res["accuracy_vs_ref160"] = acc
    for X, r in acc.items():
        print(f"TRAB_ACC {X:>14} vs ref160 (median of {len(firsts)} first pairs): " + " ".join(
            f"k{nm}={v:.2e}" for nm, v in r.items()), flush=True)

    # ---- gate D input -----------------------------------------------------------------------------------------------
    if CANDIDATE in l1s:
        l1 = l1s[CANDIDATE]
        pat = {"source": f"gates_ab_tr.py on tr_pairs (job dir {os.path.abspath(a.new)}): {CANDIDATE} level1_sub",
               "meaning": f"per coarse band b, {CANDIDATE}'s paired error vs lean_u1 at the 0.02 lag ~= alpha_b * "
                          "x_hat + beta_b * d_hat on o1.2 sub-frame rfft2 spectra (theta/8 units)",
               "bands": [list(b) for b in AT.BANDS],
               "alpha": [l1[nm]["alpha"] if l1.get(nm) else 0.0 for nm in AT.BNAME],
               "beta": [l1[nm]["beta"] if l1.get(nm) else 0.0 for nm in AT.BNAME],
               "z_x": [l1[nm]["z_x"] if l1.get(nm) else 0.0 for nm in AT.BNAME],
               "z_d": [l1[nm]["z_d"] if l1.get(nm) else 0.0 for nm in AT.BNAME]}
        pth = os.path.join(a.out, f"bias_pattern_{CANDIDATE}.json")
        with open(pth, "w") as fh:
            json.dump(AT.jsonable(pat), fh, indent=1)
        import dose_common as DC
        GD_ = DC.geometry()
        maps = {k: DC.bias_maps(GD_, pattern=p_) for k, p_ in (("tr", pth), ("ek", PATTERN_EK))}
        msq = {"tr": 0.0, "ek": 0.0}
        for p in pairs:
            z = p["data"]
            x, y = z["lean_u1__sub__0"].astype(np.float64), z[f"lean_u1__sub__{LAG}"].astype(np.float64)
            for k, (cx, cd) in maps.items():
                msq[k] += float(np.mean(DC.bias_hat_np(x, y, cx, cd) ** 2))
        ratio = math.sqrt(msq["tr"] / msq["ek"]) if msq["ek"] > 0 else float("nan")
        res["gate_D_input"] = {"pattern": pth, "rms_bhat_tr": math.sqrt(msq["tr"] / len(pairs)),
                               "rms_bhat_ek": math.sqrt(msq["ek"] / len(pairs)), "size_ratio": ratio}
        print(f"TRAB_PATTERN {pth} rms(B_hat) tr={res['gate_D_input']['rms_bhat_tr']:.3e} "
              f"lean32ek={res['gate_D_input']['rms_bhat_ek']:.3e} size_ratio={ratio:.4g}", flush=True)

    v = res["validity"]
    valid_all = v["V_frames_lean_u1_bitwise"] and v["V_sealed_reproduced"] and v["V_floor_protocol_match"]
    ga = res["gate_A"].get(CANDIDATE, {}).get("pass")
    gb = res["gate_B"].get(CANDIDATE, {}).get("overall")
    res["verdict"] = {"valid": bool(valid_all), "gate_A": ga, "gate_B": gb}
    print(f"TRAB_VERDICT valid={valid_all} {CANDIDATE}: gate_A={'PASS' if ga else 'FAIL'} gate_B={gb}", flush=True)
    with open(os.path.join(a.out, "gates_ab_tr.json"), "w") as fh:
        json.dump(AT.jsonable(res), fh, indent=1)
    print(f"GATES_AB_TR_DONE -> {os.path.join(a.out, 'gates_ab_tr.json')}", flush=True)


if __name__ == "__main__":
    main()
