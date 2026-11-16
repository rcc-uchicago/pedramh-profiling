"""score_midway3_timing.py -- score the Midway3 H100 timing jobs against PREDICTIONS_midway3_timing.md (its JSON
block is the only copy of the numbers). Stdlib only: runs on a login node.

  python3 SQG/midway3/score_midway3_timing.py [--probe <midway3_probe_dir>] [--inrow <midway3_inrow_dir>] \
      [--o12 <midway3_o12_dir>] [--o12tr <midway3_o12tr_dir> ...] [--o12b32 <midway3_o12tr_dir of Job 3C>] [--out <json>]

Every input is optional; what is not given scores MISSING. Lines: `GATE <id> PASS|FAIL|MISSING ...`,
`PRED <id> CONFIRMED|REFUTED|MISSING ...`, then `MIDWAY3_SCORE gates_failed=N gates_missing=M confirmed=a refuted=b
missing=c`. A failed gate means the run is invalid; a refuted prediction is a result.
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PRED_MD = os.path.join(HERE, "PREDICTIONS_midway3_timing.md")
ARMS = ("ref", "lean_u1", "lean32ek_fast", "if32_25", "tilerows_if25", "tilerows_inrow_if25")
TR_ARMS = ("if32_25", "tilerows_if25", "tilerows_inrow_if25")
# arm -> (file, key) of its ms / solver step, as table_o12_solvers.py
MS_KEY = {"ref": ("seed1", "ref"), "lean_u1": ("probe", "lean_u1"), "lean32ek_fast": ("probe", "lean32ek_fast"),
          "if32_25": ("probe", "if32"), "tilerows_if25": ("probe", "tilerows32_if_eager"),
          "tilerows_inrow_if25": ("probe", "inrow32_if_eager")}


def load_predictions(path=PRED_MD):
    text = open(path).read()
    m = re.search(r"```json PREDICTIONS\n(.*?)\n```", text, re.S)
    if not m:
        raise SystemExit(f"ERROR MIDWAY3_NO_PREDICTIONS_BLOCK in {path}")
    return json.loads(m.group(1))


def jload(d, name):
    p = os.path.join(d, name) if d else None
    return json.load(open(p)) if p and os.path.exists(p) else None


def o12_arms(dirs):
    """arm -> (solver entry of o12_extrapolation.json, the dir it came from), over every given o1.2 result dir."""
    out = {}
    for d in dirs:
        ex = jload(d, "o12_extrapolation.json")
        for arm, v in (ex or {}).get("solvers", {}).items():
            out[arm] = (v, d)
    return out


def one_node(v, batch):
    for k, pr in v["projections"].items():
        if k.startswith("1 node") and f"batch {batch}" in k and pr:
            return pr
    return None


def f4(x):
    return "None" if x is None else f"{x:.4f}"


def interval(point, lo, hi):
    return f"{point:.4g} [{lo:.4g}, {hi:.4g}]"


class Score:
    def __init__(self):
        self.gates, self.preds = [], []

    def gate(self, gid, ok, detail):
        v = "MISSING" if ok is None else ("PASS" if ok else "FAIL")
        self.gates.append({"id": gid, "verdict": v, "detail": detail})
        print(f"GATE {gid} {v} {detail}")

    def pred(self, gid, ok, detail):
        v = "MISSING" if ok is None else ("CONFIRMED" if ok else "REFUTED")
        self.preds.append({"id": gid, "verdict": v, "detail": detail})
        print(f"PRED {gid} {v} {detail}")


def score(P, probe=None, inrow=None, o12=None, o12tr=(), o12b32=None):
    S = Score()
    A = P["a100"]
    seed1 = jload(inrow, "seed1.json")
    ir = jload(inrow, "inrow_probe_prod.json")
    env = jload(probe, "midway_env_extra.json")
    runs = o12_arms(([o12] if o12 else []) + list(o12tr))
    runs32 = o12_arms([o12b32] if o12b32 else [])

    # ---- gates ------------------------------------------------------------------------------------------------------
    if env is None:
        S.gate("E1", None, "no midway_env_extra.json")
    else:
        g, j = env.get("torch_gpus", []), env.get("jax_devices", [])
        rel = (env.get("tile_fft") or {}).get("rel_l2_vs_torch")
        ok = (len(g) == 4 and all("H100" in x for x in g) and len(j) == 4 and all("H100" in x for x in j)
              and (env.get("warp") or {}).get("mathdx") == 1 and rel is not None
              and rel <= P["gates"]["tile_fft_rel_max"])
        S.gate("E1", ok, f"torch={g} jax={len(j)} mathdx={(env.get('warp') or {}).get('mathdx')} tile_fft_rel={rel}")
    if seed1 is None:
        S.gate("S1", None, "no seed1.json")
        S.gate("S2", None, "no seed1.json")
    else:
        gv = {v: seed1.get("gate", {}).get(v, {}).get("pass") for v in ("lean", "lean_u1", "lean_split")}
        S.gate("S1", all(x is True for x in gv.values()), f"pass={gv}")
        drift = seed1.get("bench", {}).get("ref_drift_pct")
        S.gate("S2", None if drift is None else abs(drift) <= P["gates"]["ref_drift_pct_max"], f"ref_drift_pct={drift}")
    if ir is None:
        S.gate("I1", None, "no inrow_probe_prod.json")
    else:
        vd = ir.get("verdict", {})
        flags = {k: vd.get(k) for k in ("valid", "G32", "GIF", "TR_G32", "TR_GIF")}
        ok = all((all(x) if isinstance(x, list) else x is True) for x in flags.values()) and ir.get("failures") == 0
        S.gate("I1", ok, f"{flags} failures={ir.get('failures')}")
    if seed1 is not None and ir is not None:
        a, b = seed1["bench"]["ms_per_step"].get("lean_u1"), ir["bench_ms_per_step"].get("lean_u1")
        rel = abs(a - b) / b if a and b else None
        S.gate("I2", None if rel is None else rel <= P["gates"]["lean_u1_agree_max"],
               f"seed1={a} probe={b} rel={rel if rel is None else round(rel, 5)}")
    else:
        S.gate("I2", None, "needs seed1.json and inrow_probe_prod.json")
    for arm in ARMS:
        if arm not in runs:
            S.gate(f"O1.{arm}", None, "no o1.2 run")
            continue
        v, d = runs[arm]
        S.gate(f"O1.{arm}", bool(v["parsed"].get("run_done")) and one_node(v, 16) is not None,
               f"run_done={v['parsed'].get('run_done', False)} dir={os.path.basename(d)}")
    for arm in TR_ARMS + tuple(a + "_b32" for a in ("tilerows_inrow_if25",)):
        src = runs32 if arm.endswith("_b32") else runs
        if arm not in src:
            S.gate(f"O2.{arm}", None, "no o1.2 run")
            continue
        log = os.path.join(src[arm][1], f"{arm}.log")
        text = open(log).read() if os.path.exists(log) else ""
        verdicts = re.findall(r"^O12_TR_GATE_VERDICT \S+ pass=(\w+)", text, re.M)
        S.gate(f"O2.{arm}", (bool(verdicts) and all(x == "True" for x in verdicts)) if text else None,
               f"verdicts={verdicts}")
    for arm in ("ref_b32", "tilerows_inrow_if25_b32"):
        if arm not in runs32:
            S.gate(f"O3.{arm}", None, "no batch-32 run")
            continue
        v = runs32[arm][0]
        peak = v["parsed"].get("peak")
        S.gate(f"O3.{arm}", None if peak is None else peak <= P["gates"]["b32_peak_gib_max"] and
               bool(v["parsed"].get("run_done")), f"peak={peak} GiB run_done={v['parsed'].get('run_done', False)}")

    # ---- Level 1: ms / solver step ----------------------------------------------------------------------------------
    h100 = {"ms": {}, "s_step": {}, "s_frame": {}, "hours_b16": {}, "hours_b32": {}, "s_step_b32": {}}
    for arm in ARMS:
        src, key = MS_KEY[arm]
        ms = None
        if src == "seed1" and seed1 is not None:
            ms = seed1["bench"]["ms_per_step"].get(key)
        elif src == "probe" and ir is not None:
            ms = ir["bench_ms_per_step"].get(key)
        h100["ms"][arm] = ms
        p, lo, hi = P["solver_speedup"]["f64" if arm in P["f64_arms"] else "f32"]
        a = A["ms"][arm]
        if ms is None:
            S.pred(f"P1.{arm}", None, f"predicted {interval(a / p, a / hi, a / lo)} ms")
        else:
            sp = a / ms
            S.pred(f"P1.{arm}", lo <= sp <= hi, f"H100 {ms:.4f} ms, A100 {a:.4f} ms, speedup {sp:.2f} "
                   f"(predicted {interval(p, lo, hi)})")
    split = (ir or {}).get("bench_ms_per_step", {}).get("inrow32_split_ms")
    eager = (ir or {}).get("bench_ms_per_step", {}).get("inrow32_eager")
    if split and eager:
        tot = sum(split.values())
        S.pred("P1.split_sum", abs(tot - eager) / eager <= P["split_sum_vs_eager_max"],
               f"sum {tot:.4f} vs eager {eager:.4f} ms; per op H100/A100 speedup: "
               + " ".join(f"{k}={A['inrow32_split_ms'][k] / v:.2f}" for k, v in split.items()
                          if k in A["inrow32_split_ms"] and v))
    else:
        S.pred("P1.split_sum", None, "no inrow32_split_ms")
    cv = (ir or {}).get("env", {}).get("cufft_version")
    if cv is None and env is not None:
        cv = (env.get("nvmath") or {}).get("cufft_version")
    S.pred("P1.cufft_version", None if cv is None else cv == P["cufft_version"], f"cufft_version={cv}")

    # ---- Level 2 + 3: o1.2 ------------------------------------------------------------------------------------------
    pm, lm, hm = P["ml_speedup"]
    for arm in ARMS:
        p, lo, hi = P["solver_speedup"]["f64" if arm in P["f64_arms"] else "f32"]
        frames, ml = 12 * A["s_frame"][arm], A["s_step"][arm] - 12 * A["s_frame"][arm]
        pt, fast, slow = frames / p + ml / pm, frames / hi + ml / hm, frames / lo + ml / lm
        steps = P["steps_1node_b16"]
        if arm not in runs:
            S.pred(f"P2.{arm}", None, f"predicted {interval(pt, fast, slow)} s/step")
            S.pred(f"P3.{arm}", None, "no o1.2 run")
            continue
        v = runs[arm][0]
        s, sf = v["pieces"]["s_step"], v["pieces"]["s_frame"]
        h100["s_step"][arm], h100["s_frame"][arm] = s, sf
        S.pred(f"P2.{arm}", None if s is None else fast <= s <= slow,
               f"H100 {s:.4f} s/step, A100 {A['s_step'][arm]}, speedup {A['s_step'][arm] / s:.2f} "
               f"(predicted {interval(pt, fast, slow)} s)" if s else "no s_step")
        pr = one_node(v, 16)
        h = pr["hours"] if pr else None
        h100["hours_b16"][arm] = h
        lo_h, hi_h = steps * fast / 3600, steps * slow / 3600 + P["hours_overhead_max"]
        S.pred(f"P3.{arm}", None if h is None else lo_h <= h <= hi_h,
               f"1 node b16 {h if h is None else round(h, 1)} h, A100 {A['hours_1node_b16'][arm]} "
               f"(predicted [{lo_h:.0f}, {hi_h:.0f}])")
    arm = "tilerows_inrow_if25"
    if arm in runs and runs[arm][0]["pieces"]["s_frame"] and runs[arm][0]["pieces"]["s_step"]:
        sf, s = runs[arm][0]["pieces"]["s_frame"], runs[arm][0]["pieces"]["s_step"]
        ml_h = s - 12 * sf
        ml_a = A["s_step"][arm] - 12 * A["s_frame"][arm]
        sp_ml, sp_fr = (ml_a / ml_h if ml_h > 0 else None), A["s_frame"][arm] / sf
        S.pred("P2.ml_speedup", None if sp_ml is None else lm <= sp_ml <= hm,
               f"ML step {ml_h:.4f} s (A100 {ml_a:.4f}): speedup {sp_ml if sp_ml is None else round(sp_ml, 2)} (predicted {interval(pm, lm, hm)})")
        S.pred("P2.direction", None if sp_ml is None else sp_ml < sp_fr,
               f"ML speedup {sp_ml if sp_ml is None else round(sp_ml, 2)} vs frame speedup {sp_fr:.2f} "
               "(predicted: ML < frames)")
    else:
        S.pred("P2.ml_speedup", None, "no tilerows_inrow_if25 run")
        S.pred("P2.direction", None, "no tilerows_inrow_if25 run")

    # ---- batch 32 -----------------------------------------------------------------------------------------------------
    for arm in ("ref", "tilerows_inrow_if25"):
        a32 = arm + "_b32"
        if a32 not in runs32 or arm not in runs:
            S.pred(f"P3.b32_ratio.{arm}", None, "needs the b16 and b32 runs")
            S.pred(f"P3.b32_hours.{arm}", None, "needs the b16 and b32 runs")
            continue
        v32, v16 = runs32[a32][0], runs[arm][0]
        s32, s16 = v32["pieces"]["s_step"], v16["pieces"]["s_step"]
        h100["s_step_b32"][arm] = s32
        lo, hi = P["b32_ratio"][arm]
        r = s32 / s16 if s32 and s16 else None
        S.pred(f"P3.b32_ratio.{arm}", None if r is None else lo <= r <= hi,
               f"s_step b32 {f4(s32)} / b16 {f4(s16)} = {r if r is None else round(r, 3)} (predicted [{lo}, {hi}])")
        p32, p16 = one_node(v32, 32), one_node(v16, 16)
        h100["hours_b32"][arm] = p32["hours"] if p32 else None
        S.pred(f"P3.b32_hours.{arm}", None if not (p32 and p16) else p32["hours"] < p16["hours"],
               f"1 node b32 {p32 and round(p32['hours'], 1)} h vs b16 {p16 and round(p16['hours'], 1)} h")
    peaks = [runs32[a][0]["parsed"].get("peak") for a in ("ref_b32", "tilerows_inrow_if25_b32") if a in runs32]
    peaks = [x for x in peaks if x is not None]
    lo, hi = P["b32_peak_gib"]
    S.pred("P3.b32_peak", (all(lo <= x <= hi for x in peaks)) if peaks else None, f"peaks={peaks} GiB "
           f"(predicted [{lo}, {hi}])")

    c = {k: sum(1 for x in S.preds if x["verdict"] == k) for k in ("CONFIRMED", "REFUTED", "MISSING")}
    gf = sum(1 for x in S.gates if x["verdict"] == "FAIL")
    gm = sum(1 for x in S.gates if x["verdict"] == "MISSING")
    print(f"MIDWAY3_SCORE gates_failed={gf} gates_missing={gm} confirmed={c['CONFIRMED']} refuted={c['REFUTED']} "
          f"missing={c['MISSING']}")
    return {"gates": S.gates, "predictions": S.preds, "h100": h100, "gates_failed": gf, "gates_missing": gm,
            "counts": c}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe")
    ap.add_argument("--inrow")
    ap.add_argument("--o12")
    ap.add_argument("--o12tr", action="append", default=[])
    ap.add_argument("--o12b32")
    ap.add_argument("--pred", default=PRED_MD)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = score(load_predictions(a.pred), a.probe, a.inrow, a.o12, a.o12tr, a.o12b32)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
