"""analyze_timestep.py -- PREDICTIONS_timestep.md's decision rule over timestep_probe.py's worker_<w>.json.
Standard library only (re-runnable on a login node).

  python3 analyze_timestep.py --in <dir with worker_*.json> --out <results dir>

Rule (fixed before the run):
  VALID       every ladder worker: control prod40 vs lean_u1 rel <= 1e-12, ref finite, and the reference adequate:
              e(if80) <= 0.1 e(prod40) in every band and frame kind (then ref's own error is <= ~1.4% of prod40's).
  ACC_OK(n)   every ladder worker ran if<n> finite and e_b(if<n>) <= e_b(prod40) in every band b, sub and af.
  CFL_OK(n)   CFL_max * 40 / n <= 0.72 (A-SQG_data.py's abort), CFL_max = the largest CFL at DT over the archived
              states (aux), the ladder states, and the reference runs' samples.
  n_adm       the smallest n in (32, 28, 25, 20) with ACC_OK and CFL_OK (fewest steps per frame), or none.
Projection per optimizer step (o1.2: 12 frames of n steps + the 0.40 s ML step, research round 2's formula):
  s(n) = 12 n t_step + 0.40 s, speedup = s(40) / s(n), for lean32ek_fast (2.698 ms) and lean_u1 (6.409 ms).
Prints TS_VALID, TS_ORDER, TS_ACC, TS_CFL, TS_VERDICT n_adm=<n|none> valid=<bool>, TS_ANALYSIS_DONE.
"""
import argparse
import glob
import json
import math
import os

CANDIDATES = (32, 28, 25, 20)
CFL_ABORT = 0.72
T_STEP_MS = {"lean32ek_fast": 2.698, "lean_u1": 6.409}               # A100, notes Parts X / XIII
ML_S = 0.40


def num(x):
    return float(x) if not isinstance(x, str) else float("nan")


def le(a, b):
    """a <= b elementwise, nan -> False."""
    return all(num(x) <= num(y) for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    W = {}
    for p in sorted(glob.glob(os.path.join(args.inp, "worker_*.json"))):
        with open(p) as f:
            d = json.load(f)
        W[d["worker"]] = d
    ladders = [d for d in W.values() if d["mode"] == "ladder"]
    aux = [d for d in W.values() if d["mode"] == "aux"]
    out = {"workers": sorted(W), "rule": __doc__.split("Rule (fixed before the run):")[1].split("Projection")[0]}
    if not ladders:
        print("ERROR TS_NO_LADDER_WORKERS")
        raise SystemExit(1)

    # ---- validity -----------------------------------------------------------------------------------------------
    valid, why = True, []
    for d in ladders:
        w = d["worker"]
        c = d.get("control", {})
        if not (num(c.get("prod40_vs_lean_u1_rel", "nan")) <= 1e-12):
            valid = False
            why.append(f"w{w} control {c}")
        if not d.get("runs", {}).get("ref", {}).get("finite"):
            valid = False
            why.append(f"w{w} ref not finite")
        e = d.get("errors", {})
        if "if80" in e and "prod40" in e:
            for k in ("sub", "af"):
                if not le(e["if80"][k], [0.1 * num(v) for v in e["prod40"][k]]):
                    valid = False
                    why.append(f"w{w} reference not adequate ({k}): if80 {e['if80'][k]} vs 0.1 prod40")
        else:
            valid = False
            why.append(f"w{w} missing if80/prod40")
    out["valid"], out["valid_why"] = valid, why
    print(f"TS_VALID {valid}" + ("" if valid else " " + "; ".join(why)[:600]))

    # ---- order checks (P3, P4) -------------------------------------------------------------------------------------
    order = {}
    for d in ladders:
        e = d.get("errors", {})
        w = d["worker"]
        r = {}
        for k in ("sub", "af"):
            if "prod80" in e and "prod40" in e:
                r[f"prod80/prod40_{k}"] = [num(a) / num(b) for a, b in zip(e["prod80"][k], e["prod40"][k])]
            if "if80" in e and "if40" in e:
                r[f"if80/if40_{k}"] = [num(a) / num(b) for a, b in zip(e["if80"][k], e["if40"][k])]
            if "if40" in e and "prod40" in e:
                r[f"if40/prod40_{k}"] = [num(a) / num(b) for a, b in zip(e["if40"][k], e["prod40"][k])]
            for n in (32, 28, 25, 20):
                if f"prod{n}" in e and "prod40" in e:
                    r[f"prod{n}/prod40_{k}"] = [num(a) / num(b) for a, b in zip(e[f"prod{n}"][k], e["prod40"][k])]
                if f"if{n}" in e and "prod40" in e:
                    r[f"if{n}/prod40_{k}"] = [num(a) / num(b) for a, b in zip(e[f"if{n}"][k], e["prod40"][k])]
            if "if40_vs_prod40" in e and "prod40" in e:
                r[f"change_if40/prod40err_{k}"] = [num(a) / num(b) for a, b in zip(e["if40_vs_prod40"][k],
                                                                                 e["prod40"][k])]
        order[w] = r
    out["ratios"] = order

    def rng(key):
        vals = [v for r in order.values() for v in r.get(key, []) if math.isfinite(v)]
        return (min(vals), max(vals)) if vals else (float("nan"), float("nan"))

    for key in ("prod80/prod40", "if80/if40", "if40/prod40", "if28/prod40", "if25/prod40", "if20/prod40",
                "change_if40/prod40err"):
        for k in ("sub", "af"):
            lo, hi = rng(f"{key}_{k}")
            print(f"TS_ORDER {key:>22s} {k:>3s}: {lo:.3g} .. {hi:.3g}")

    # ---- CFL --------------------------------------------------------------------------------------------------------
    cfl = [d["state"]["cfl"] for d in ladders if "state" in d]
    cfl += [d["cfl_ref"]["max"] for d in ladders if d.get("cfl_ref", {}).get("max") is not None]
    for d in aux:
        cfl += [s["cfl"] for s in d.get("cfl_states", [])]
    cfl_max = max(cfl) if cfl else float("nan")
    out["cfl_max_at_DT"] = cfl_max
    out["cfl_at_n"] = {n: cfl_max * 40 / n for n in (40,) + CANDIDATES}
    print(f"TS_CFL max at DT {cfl_max:.4f} over {len(cfl)} samples; at n: "
          + " ".join(f"{n}:{cfl_max * 40 / n:.3f}" for n in (40,) + CANDIDATES))

    # ---- admissibility ------------------------------------------------------------------------------------------------
    acc = {}
    for n in CANDIDATES:
        ok, notes = True, []
        for d in ladders:
            e, runs, w = d.get("errors", {}), d.get("runs", {}), d["worker"]
            if not runs.get(f"if{n}", {}).get("finite") or f"if{n}" not in e or "prod40" not in e:
                ok = False
                notes.append(f"w{w} if{n} missing or not finite")
                continue
            for k in ("sub", "af"):
                bad = [i for i, (a, b) in enumerate(zip(e[f"if{n}"][k], e["prod40"][k])) if not num(a) <= num(b)]
                if bad:
                    ok = False
                    notes.append(f"w{w} {k} bands {bad}")
        acc[n] = {"acc_ok": ok, "cfl_ok": bool(cfl_max * 40 / n <= CFL_ABORT), "notes": notes}
        print(f"TS_ACC n={n} acc_ok={ok} cfl_ok={acc[n]['cfl_ok']}" + (f" ({'; '.join(notes)[:300]})" if notes else ""))
    out["admissibility"] = acc
    adm = [n for n in CANDIDATES if acc[n]["acc_ok"] and acc[n]["cfl_ok"]]
    n_adm = min(adm) if adm else None
    out["n_adm"] = n_adm

    # ---- projection -----------------------------------------------------------------------------------------------------
    proj = {}
    for name, t in T_STEP_MS.items():
        s40 = 12 * 40 * t / 1e3 + ML_S
        proj[name] = {n: {"s_per_opt_step": 12 * n * t / 1e3 + ML_S, "speedup": s40 / (12 * n * t / 1e3 + ML_S)}
                      for n in (40,) + CANDIDATES}
    out["projection"] = proj
    if n_adm:
        print(f"TS_PROJECTION n={n_adm}: lean32ek_fast {proj['lean32ek_fast'][n_adm]['speedup']:.3f}x, "
              f"lean_u1 {proj['lean_u1'][n_adm]['speedup']:.3f}x per optimizer step")

    # ---- aux facts --------------------------------------------------------------------------------------------------------
    for d in aux:
        out["ms_per_step"] = d.get("ms_per_step")
        out["f32_if_accuracy"] = d.get("f32_if_accuracy")
        out["batched_rfft"] = d.get("batched_rfft")
        ms = d.get("ms_per_step") or {}
        if ms:
            print("TS_COST " + " ".join(f"{k}={v:.4f}" for k, v in ms.items())
                  + (f" | if_rt/prod_rt {ms['if_rt'] / ms['prod_rt']:.4f}" if "if_rt" in ms and "prod_rt" in ms else "")
                  + (f" | if32/lean32ek_fast {ms['if32'] / ms['lean32ek_fast']:.4f}"
                     if "if32" in ms and "lean32ek_fast" in ms else ""))
        fa = d.get("f32_if_accuracy") or {}
        if "if32_vs_if64" in fa and "fast_vs_u1" in fa:
            for k in ("sub", "af"):
                r = [num(a) / num(b) for a, b in zip(fa["if32_vs_if64"][k], fa["fast_vs_u1"][k])]
                print(f"TS_F32_RATIO {k}: " + " ".join(f"{x:.3g}" for x in r))
    print(f"TS_VERDICT n_adm={n_adm if n_adm else 'none'} valid={valid}")
    with open(os.path.join(args.out, "timestep_summary.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("TS_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
