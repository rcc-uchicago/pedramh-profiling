"""analyze_mixprec.py -- probe 1's verdict per arm (PREDICTIONS_mixprec.md, written first), from
  <res>/<arm>/gates_ab_tr.json   gates_ab_tr.py run with that arm as the candidate (the unchanged A/B rules)
  <new>/worker_*.json            tr_pairs.py's workers: the mixprec column fractions
  <new>/pair_*.npz, <wrap>/pair_*.npz   V_wrap: the @none arm against job 7854208's tilerows_inrow_if25 frames
numpy only.

  python3 analyze_mixprec.py --res <dir> --new <tr_pairs out> --wrap <trab_7854208 raw dir> --arms a+b+... [--ref ARM]

Class per arm (reference = the @none arm, whose own A/B verdict is tilerows_inrow_if25's):
  PASS     valid; gate A PASS; every gate-B band PASS or FAIL_FLOOR on B3 alone (B1, B2, power hold); and the bias
           pattern's size (gates_ab_tr.py's size_ratio) <= 2 x the reference's
  BIAS_UP  as PASS but the size ratio > 2 x the reference's (would need gate D)
  FAIL_AB  gate A FAIL, or any band failing B1 / B2 / power
  INVALID  the shared validity (V_frames, V_sealed, V_floor) fails
Prize (upper bound): 0.5 x mean over the 6 arrays of the 16-bit column fraction x the step (1.470 ms, job 7853317).
Prints MIXPREC_ARM lines and MIXPREC_DONE last.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mixprec as MP                                                 # noqa: E402

LAGS = (0, 1, 10, 25, 50, 100)
STEP_MS = 1.470                                                      # tilerows_inrow_if25's IF step, job 7853317
WRAP_KEY = "tilerows_inrow_if25"


def detection_only(gb):
    for b in gb["bands"].values():
        v = b.get("verdict")
        if v in ("PASS", "NO_DATA"):
            continue
        if not (v == "FAIL_FLOOR" and b["B1"] and b["B2"] and b["power"] and b["B3"] is False):
            return False
    return True


def classify(j, arm, ref_size):
    if not j["verdict"]["valid"]:
        return "INVALID", None
    size = (j.get("gate_D_input") or {}).get("size_ratio")
    ratio = size / ref_size if (size is not None and ref_size) else None
    if not (j["gate_A"][arm]["pass"] and detection_only(j["gate_B"][arm])):
        return "FAIL_AB", ratio
    return ("PASS" if ratio is not None and ratio <= 2.0 else "BIAS_UP"), ratio


def v_wrap(new, wrap, ref):
    eq, n = 0, 0
    for f in sorted(glob.glob(os.path.join(new, "pair_w*_s*_p*.npz"))):
        g = os.path.join(wrap, os.path.basename(f))
        if not os.path.exists(g):
            continue
        n += 1
        with np.load(f) as a, np.load(g) as b:
            ok = True
            for k in LAGS[1:]:
                for kind, dt in (("sub", np.uint32), ("af", np.uint64)):
                    x, y = a[f"{ref}__{kind}__{k}"], b[f"{WRAP_KEY}__{kind}__{k}"]
                    ok = ok and x.shape == y.shape and np.array_equal(x.view(dt), y.view(dt))
        eq += bool(ok)
    return eq, n


def fractions(new):
    acc = {}
    for f in sorted(glob.glob(os.path.join(new, "worker_*.json"))):
        for pol, arrays in (json.load(open(f)).get("mixprec") or {}).items():
            for a, d in arrays.items():
                acc.setdefault(pol, {}).setdefault(a, []).append(d)
    out = {}
    for pol, arrays in acc.items():
        out[pol] = {a: {k: float(np.mean([d[k] for d in ds])) for k in ("bf16", "fp16", "f16")}
                    for a, ds in arrays.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--wrap", required=True)
    ap.add_argument("--arms", required=True, help="'+'- or ','-separated")
    ap.add_argument("--ref", default="tilerows_inrow_if25@none")
    a = ap.parse_args()
    arms = [x for x in a.arms.replace("+", ",").split(",") if x]
    J = {}
    for arm in arms:
        p = os.path.join(a.res, arm, "gates_ab_tr.json")
        if os.path.exists(p):
            J[arm] = json.load(open(p))
    ref_size = ((J.get(a.ref) or {}).get("gate_D_input") or {}).get("size_ratio")
    eq, n = v_wrap(a.new, a.wrap, a.ref)
    fr = fractions(a.new)
    out = {"ref": a.ref, "step_ms": STEP_MS, "V_wrap": {"equal": eq, "pairs": n, "ok": n == 64 and eq == n},
           "arms": {}}
    print(f"MIXPREC_VWRAP {a.ref} == job 7854208's {WRAP_KEY}: {eq}/{n} pairs bitwise", flush=True)
    for arm in arms:
        if arm not in J:
            out["arms"][arm] = {"class": "MISSING"}
            print(f"MIXPREC_ARM {arm} MISSING", flush=True)
            continue
        j = J[arm]
        cls, ratio = classify(j, arm, ref_size)
        if cls == "INVALID":                                         # no gate-B dict for the arm (job 7859359)
            out["arms"][arm] = {"class": cls}
            print(f"MIXPREC_ARM {arm} INVALID", flush=True)
            continue
        pol = arm.partition("@")[2]
        f = fr.get(pol, {})
        f16 = float(np.mean([f[x]["f16"] for x in MP.POINTS])) if all(x in f for x in MP.POINTS) else 0.0
        prize = MP.projected_ms(f, STEP_MS) if f else 0.0
        acc = j.get("accuracy_vs_ref160", {})
        acc_ratio = (max(acc[arm][k] / acc[a.ref][k] for k in acc[arm]) if arm in acc and a.ref in acc else None)
        gb = j["gate_B"][arm]
        row = {"class": cls, "size_ratio": (j.get("gate_D_input") or {}).get("size_ratio"), "bias_vs_ref": ratio,
               "gate_A_pass": j["gate_A"][arm]["pass"], "gate_A_max": max(j["gate_A"][arm]["var_over_floor"]),
               "gate_B": {k: b.get("verdict") for k, b in gb["bands"].items()},
               "max_bias_over_T": max(max(abs(b["B_x"]), abs(b["B_d"])) / b["T_b"] for b in gb["bands"].values()
                                      if b.get("verdict") != "NO_DATA"),
               "acc_vs_ref160_max_ratio": acc_ratio, "fractions": f, "f16_mean": f16, "prize_ms": prize,
               "prize_pct": 100 * prize / STEP_MS}
        out["arms"][arm] = row
        print(f"MIXPREC_ARM {arm} {cls} gateA_max={row['gate_A_max']:.2e} bias_vs_ref="
              f"{'nan' if ratio is None else f'{ratio:.3f}'} max|B|/T={row['max_bias_over_T']:.2e} "
              f"acc_ratio={'nan' if acc_ratio is None else f'{acc_ratio:.3f}'} f16={f16:.3f} "
              f"prize<={prize:.3f} ms ({row['prize_pct']:.1f}%)", flush=True)
    passing = [r for r in out["arms"].values() if r.get("class") == "PASS"]
    best = max(passing, key=lambda r: r["prize_ms"]) if passing else None
    out["best_pass_prize_ms"] = best["prize_ms"] if best else 0.0
    with open(os.path.join(a.res, "mixprec_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"MIXPREC_DONE V_wrap={out['V_wrap']['ok']} best PASS prize <= {out['best_pass_prize_ms']:.3f} ms "
          f"-> {os.path.join(a.res, 'mixprec_summary.json')}", flush=True)


if __name__ == "__main__":
    main()
