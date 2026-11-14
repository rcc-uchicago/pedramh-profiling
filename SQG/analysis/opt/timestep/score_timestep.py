"""score_timestep.py <results dir> -- score PREDICTIONS_timestep.md's 16 predictions mechanically (stdlib).
Reads timestep_summary.json and worker_*.json from the results dir."""
import glob
import json
import math
import os
import sys


def num(x):
    return float(x) if not isinstance(x, str) else float("nan")


def main():
    d = sys.argv[1]
    S = json.load(open(os.path.join(d, "timestep_summary.json")))
    W = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "worker_*.json")))]
    L = [w for w in W if w["mode"] == "ladder"]
    A = [w for w in W if w["mode"] == "aux"]
    ratios = S.get("ratios", {})
    out = []

    def say(n, v, detail):
        out.append((n, v, detail))

    def vals(key):
        return [v for r in ratios.values() for v in r.get(key, [])]

    def within(key, lo, hi):
        v = [x for x in vals(key)]
        ok = bool(v) and all(math.isfinite(x) and lo <= x <= hi for x in v)
        return ok, (min(v), max(v)) if v else None

    rel = [num(w.get("control", {}).get("prod40_vs_lean_u1_rel", "nan")) for w in L]
    bit = [w.get("control", {}).get("bitwise") for w in L]
    say(1, "confirmed" if rel and all(r <= 1e-12 for r in rel) else "refuted", f"rel {rel}, bitwise {bit}")
    say(2, "confirmed" if S.get("valid") else "refuted", "; ".join(S.get("valid_why", []))[:300])
    a = [within(f"prod80/prod40_{k}", 0.4, 0.6) for k in ("sub", "af")]
    b = []
    for n in (32, 28, 25, 20):
        for k in ("sub", "af"):
            b.append(within(f"prod{n}/prod40_{k}", 0.8 * 40 / n, 1.2 * 40 / n))
    ok_a, ok_b = all(x[0] for x in a), all(x[0] for x in b)
    say(3, "confirmed" if ok_a and ok_b else ("half" if ok_a or ok_b else "refuted"),
        f"prod80/prod40 sub {a[0][1]} af {a[1][1]}; ladder ok={ok_b}")
    c = [within(f"if40/prod40_{k}", 0, 0.1) for k in ("sub", "af")]
    say(4, "confirmed" if all(x[0] for x in c) else "refuted", f"sub {c[0][1]} af {c[1][1]}")
    c = [within(f"if80/if40_{k}", 0.08, 0.25) for k in ("sub", "af")]
    say(5, "confirmed" if all(x[0] for x in c) else ("half" if any(x[0] for x in c) else "refuted"),
        f"sub {c[0][1]} af {c[1][1]}")
    adm = S.get("admissibility", {})
    for p, n in ((6, 28), (7, 25), (8, 20)):
        x = adm.get(str(n), {})
        say(p, "confirmed" if x.get("acc_ok") else "refuted", "; ".join(x.get("notes", []))[:200])
    fin = {name: all(w.get("runs", {}).get(name, {}).get("finite") for w in L)
           for name in [f"{s}{n}" for s in ("prod", "if") for n in (80, 40, 32, 28, 25, 20)]}
    ge25 = all(v for k, v in fin.items() if int(k.lstrip("prodif")) >= 25)
    n20 = fin.get("prod20") and fin.get("if20")
    say(9, "confirmed" if ge25 and n20 else ("half" if ge25 else "refuted"), f"n>=25 finite {ge25}; n=20 finite {n20}")
    cm = S.get("cfl_max_at_DT")
    say(10, "confirmed" if cm is not None and cm <= 0.45 and adm.get("25", {}).get("cfl_ok")
        and not adm.get("20", {}).get("cfl_ok") else "refuted", f"CFL_max {cm}; at n {S.get('cfl_at_n')}")
    say(11, "confirmed" if S.get("n_adm") == 25 else "refuted", f"n_adm = {S.get('n_adm')}")
    c = [within(f"change_if40/prod40err_{k}", 0.8, 1.2) for k in ("sub", "af")]
    say(12, "confirmed" if all(x[0] for x in c) else ("half" if any(x[0] for x in c) else "refuted"),
        f"sub {c[0][1]} af {c[1][1]}")
    ms = (A[0].get("ms_per_step") if A else None) or {}
    r1 = ms["if_rt"] / ms["prod_rt"] if "if_rt" in ms and "prod_rt" in ms else None
    r2 = ms["if32"] / ms["lean32ek_fast"] if "if32" in ms and "lean32ek_fast" in ms else None
    ok1, ok2 = r1 is not None and r1 <= 1.02, r2 is not None and r2 <= 1.02
    say(13, "confirmed" if ok1 and ok2 else ("half" if ok1 or ok2 else "refuted"), f"if_rt/prod_rt {r1}, if32/fast {r2}")
    fa = (A[0].get("f32_if_accuracy") if A else None) or {}
    if fa:
        rr = [num(x) / num(y) for k in ("sub", "af") for x, y in zip(fa["if32_vs_if64"][k], fa["fast_vs_u1"][k])]
        say(14, "confirmed" if all(0.33 <= x <= 3 for x in rr) else "refuted", f"ratio {min(rr):.3g} .. {max(rr):.3g}")
    else:
        say(14, "refuted", "no aux data")
    rb = ((A[0].get("batched_rfft") if A else None) or {}).get("f32", {})
    ops = rb.get("batched_fft_ops", [])
    say(15, "confirmed" if len(ops) == 1 and ops[0].startswith("c64[2,") else "refuted", f"batched ops {ops}")
    sv = rb.get("separate_us", 0) - rb.get("batched_us", 0) if rb else None
    say(16, "confirmed" if sv is not None and 20 <= sv <= 80 else "refuted",
        f"saving {sv} us (separate {rb.get('separate_us')}, batched {rb.get('batched_us')}), bitwise {rb.get('bitwise')}")
    for n, v, dd in out:
        print(f"P{n:<2d} {v:9s} {dd}")
    tally = {v: sum(1 for _, x, _ in out if x == v) for v in ("confirmed", "half", "refuted")}
    print("TS_SCORE " + " ".join(f"{k}={v}" for k, v in tally.items()) + f" | n_adm {S.get('n_adm')} valid {S.get('valid')}")


if __name__ == "__main__":
    main()
