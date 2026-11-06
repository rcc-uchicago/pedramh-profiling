"""rank_round1.py -- one ranking table over the round-1 solver-step candidates (seed 1 and seed 2 JSONs). Stdlib only.

    python3 SQG/analysis/opt/rank_round1.py --seed1 <dir>/seed1.json --seed2 <dir>/seed2.json [--out ranking.txt]

Ranking rule (the user's gate policy): a variant that FAILED its pre-set tolerance gate (or never ran it) is never
ranked as a winner; among the passing variants the fastest ms/step wins. Bitwise equality is a column of information
only and is never a penalty: a non-bitwise passing variant ranks purely on speed. Columns: variant, seed, precision,
gate (PASS/FAIL + the measured 1-step relative L2 error and its threshold), bitwise after 480 steps, 4,000-step coarse
field relative L2 difference and energy relative difference vs the reference (synthetic start state; ungated,
chaotic), ms/step, speedup vs that seed's own production reference, optimizer-step seconds (12 x 40 steps + coarse).
Also printed: the best float64-only passing variant (a float32 solver is a science decision, DESIGN 4).
Prints OPT_RANK_OK on success (at least one JSON read), ERROR OPT_RANK_NO_INPUT otherwise.
"""
import argparse
import json
import math
import os
import sys

INF = float("inf")
PRECISION_SEED2 = {"lean64": "f64", "mixed": "f32-nonlin/f64-state", "lean32": "f32"}


def _f(x):
    """A float or NaN (JSON may hold strings like 'nan' / 'inf', None, or be missing)."""
    if isinstance(x, bool) or x is None:
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _nanmax(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    return max(vals) if vals else float("nan")


def load(path):
    if not path or not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def rows_seed1(j):
    """Rows from run_seed1.py's seed1.json."""
    rows = []
    bench = j.get("bench", {}) or {}
    ms_all = bench.get("ms_per_step", {}) or {}
    ref_ms = _f(ms_all.get("ref"))
    thr = _f((j.get("thresholds", {}) or {}).get("G1_1step_rel_l2_per_field"))
    gates = j.get("gate", {}) or {}
    long_v = ((j.get("long", {}) or {}).get("variants", {}) or {})
    opt = j.get("optstep", {}) or {}
    names = list((j.get("variants") or gates or {}).keys())
    for v in names:
        g = gates.get(v, {}) or {}
        err1 = _nanmax([_f(c.get("max_rel_l2")) for c in (g.get("1step", {}) or {}).values()])
        b480 = [c.get("bits_equal") for c in (g.get("480step", {}) or {}).values()]
        bitwise = (all(bool(x) for x in b480) if b480 else None)
        lg = long_v.get(v, {}) or {}
        ms = _f(ms_all.get(v))
        rows.append({
            "variant": v, "seed": 1, "precision": "f64",
            "gate_pass": bool(g.get("pass", False)), "gate_reason": g.get("reason") or "",
            "err1": err1, "thr": thr, "bitwise480": bitwise,
            "coarse4000": _f(lg.get("coarse_rel_l2")), "denergy4000": _f(lg.get("energy_rel_diff")),
            "ms": ms, "ref_ms": ref_ms, "speedup": (ref_ms / ms if ms > 0 and not math.isnan(ref_ms) else float("nan")),
            "optstep_s": _f((opt.get(v) or {}).get("s_median")) if opt.get("variant") == v else float("nan"),
            "ref_optstep_s": _f((opt.get("ref") or {}).get("s_median")),
        })
    return rows


def rows_seed2(j):
    """Rows from seed2_opt.py's seed2.json."""
    rows = []
    ref_ms = _f(j.get("ref_ms_per_step"))
    ref_opt = _f(j.get("ref_optstep_s"))
    for v, vr in (j.get("variants", {}) or {}).items():
        vr = vr or {}
        gate = vr.get("gate", {}) or {}
        err1 = _nanmax([_f(c.get("rel_l2_max")) for c in gate.values()])
        lg = vr.get("long", {}) or {}
        b480 = [c.get("bitwise_480") for c in lg.values() if isinstance(c, dict) and "bitwise_480" in c]
        bitwise = (all(bool(x) for x in b480) if b480 else None)
        syn = lg.get("synthetic", {}) or {}
        ms = _f(vr.get("ms_per_step"))
        reason = ";".join(vr.get("gate_reasons", []) or []) or (vr.get("error") or "")
        rows.append({
            "variant": v, "seed": 2, "precision": PRECISION_SEED2.get(v, vr.get("fft_precision_claimed", "?")),
            "gate_pass": bool(vr.get("gate_pass", False)), "gate_reason": reason[:120],
            "err1": err1, "thr": _f(vr.get("gate_threshold")), "bitwise480": bitwise,
            "coarse4000": _f(syn.get("coarse_rel_l2_end")), "denergy4000": _f(syn.get("spe_rel_diff_end")),
            "ms": ms, "ref_ms": ref_ms, "speedup": (ref_ms / ms if ms > 0 and not math.isnan(ref_ms) else float("nan")),
            "optstep_s": _f(vr.get("optstep_s")), "ref_optstep_s": ref_opt,
        })
    return rows


def rank(rows):
    """Sorted rows: passing variants first, fastest first; failed/unmeasured variants after, in the same order."""
    def key(r):
        ms = r["ms"] if (r["ms"] > 0 and not math.isnan(r["ms"])) else INF
        return (0 if r["gate_pass"] else 1, ms, r["variant"])
    return sorted(rows, key=key)


def winner(ranked, f64_only=False):
    for r in ranked:
        if r["gate_pass"] and r["ms"] > 0 and not math.isnan(r["ms"]) and (not f64_only or r["precision"] == "f64"):
            return r
    return None


def fmt(x, spec):
    return "-" if (x is None or (isinstance(x, float) and math.isnan(x))) else format(x, spec)


def table(ranked):
    head = ["#", "variant", "seed", "precision", "gate", "err1(thr)", "bitwise480", "coarse4000", "dE4000",
            "ms/step", "ref ms", "speedup", "optstep s", "note"]
    lines = [head]
    for i, r in enumerate(ranked, 1):
        lines.append([
            str(i), r["variant"], str(r["seed"]), r["precision"], "PASS" if r["gate_pass"] else "FAIL",
            f"{fmt(r['err1'], '.2e')}({fmt(r['thr'], '.0e')})",
            {True: "yes", False: "no", None: "-"}[r["bitwise480"]],
            fmt(r["coarse4000"], ".2e"), fmt(r["denergy4000"], "+.2e"),
            fmt(r["ms"], ".3f"), fmt(r["ref_ms"], ".3f"), fmt(r["speedup"], ".3f") + ("x" if not math.isnan(r["speedup"]) else ""),
            fmt(r["optstep_s"], ".3f"), ("" if r["gate_pass"] else r["gate_reason"]) or "",
        ])
    widths = [max(len(row[c]) for row in lines) for c in range(len(head))]
    out = []
    for n, row in enumerate(lines):
        out.append("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip())
        if n == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="rank the round-1 solver-step candidates")
    ap.add_argument("--seed1", default=None, help="seed1.json (run_seed1.py)")
    ap.add_argument("--seed2", default=None, help="seed2.json (seed2_opt.py)")
    ap.add_argument("--out", default=None, help="also write the table here")
    args = ap.parse_args(argv)
    rows, missing = [], []
    for path, fn, tag in ((args.seed1, rows_seed1, "seed1"), (args.seed2, rows_seed2, "seed2")):
        j = load(path)
        if j is None:
            missing.append(f"{tag}:{path}")
            continue
        try:
            rows.extend(fn(j))
        except Exception as exc:  # noqa: BLE001  (a malformed JSON must not hide the other seed)
            missing.append(f"{tag}:{type(exc).__name__}:{exc}")
    if not rows:
        print(f"ERROR OPT_RANK_NO_INPUT {missing}")
        return 1
    ranked = rank(rows)
    txt = table(ranked)
    best, best64 = winner(ranked), winner(ranked, f64_only=True)
    lines = ["Round 1 ranking: tolerance-gated (FAIL never wins); bitwise is information only; 4,000-step columns are",
             "ungated (chaotic flow, synthetic start state), speedup is vs the same seed's production reference.", "", txt, ""]
    if missing:
        lines.append(f"WARN OPT_RANK_MISSING_INPUT {missing}")
    if best is None:
        lines.append("WARN OPT_RANK_NO_PASSING_VARIANT: no variant passed its tolerance gate with a timing")
    else:
        lines.append(f"OPT_RANK_WINNER {best['variant']} (seed {best['seed']}, {best['precision']}): "
                     f"{best['ms']:.3f} ms/step, {best['speedup']:.3f}x vs ref {best['ref_ms']:.3f}, "
                     f"1-step rel L2 {fmt(best['err1'], '.2e')} <= {fmt(best['thr'], '.0e')}, bitwise480={best['bitwise480']}")
    if best64 is not None and (best is None or best64["variant"] != best["variant"]):
        lines.append(f"OPT_RANK_BEST_F64 {best64['variant']} (seed {best64['seed']}): {best64['ms']:.3f} ms/step, "
                     f"{best64['speedup']:.3f}x (float64-only: no science sign-off needed for precision)")
    lines.append("OPT_RANK_OK")
    out = "\n".join(lines)
    print(out)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(out + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
