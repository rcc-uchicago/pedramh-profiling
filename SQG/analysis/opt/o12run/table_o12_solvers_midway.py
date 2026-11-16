"""table_o12_solvers_midway.py -- the H100-vs-A100 table of every o1.2 solver arm (MIDWAY3_INROW_PROMPT.md §6 "The
table"). Beside table_o12_solvers.py, whose A100 rows (ROWS, the same Polaris jobs) it reuses unchanged; the H100
columns come from the Midway3 result dirs, read the same way. Stdlib only.

  python3 table_o12_solvers_midway.py --inrow <midway3_inrow_dir> --o12 <midway3_o12_dir> \
      --o12tr <midway3_o12tr_dir> [--o12tr ...] [--o12b32 <midway3_o12tr dir of Job 3C>]

Prints three markdown tables and TABLE_O12_SOLVERS_MIDWAY_OK:
  1. per solver: ms / solver step (1 GPU), o1.2 s / opt step, x ref, 1-node hours at batch 16 (and 32 where run),
     A100 | H100 | A100 / H100;
  2. o1.2's step split into its solver frames (12 x s_frame) and the rest (the strict-FP32 ML step), both GPUs:
     which part shrank more;
  3. the per-op split of the inrow32 step (CUDA events, median over steps 5-39), both GPUs.
A missing H100 input prints "—". The 14 / 16-GPU projections are Polaris-only and not shown (one node here).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import table_o12_solvers as T  # noqa: E402

FRAMES_PER_STEP = 12


def jl(*p):
    path = os.path.join(*p)
    return json.load(open(path)) if os.path.exists(path) else None


def one_node(v, batch):
    for k, pr in (v or {}).get("projections", {}).items():
        if k.startswith("1 node") and f"batch {batch}" in k and pr:
            return pr["hours"]
    return None


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def ratio(a, h, nd=2):
    return "—" if a is None or h is None or h == 0 else f"{a / h:.{nd}f}×"


def gather(seed1, probe, runs, runs32):
    """arm -> {ms, s_step, s_frame, h16, h32} for one GPU type."""
    ms_src = {"seed1": (seed1 or {}).get("bench", {}).get("ms_per_step", {}),
              "inrow": (probe or {}).get("bench_ms_per_step", {})}
    out = {}
    for arm, _job, spf, (src, key) in T.ROWS:
        v = runs.get(arm)
        p = (v or {}).get("pieces", {})
        out[arm] = {"spf": spf, "ms": ms_src[src].get(key), "s_step": p.get("s_step"), "s_frame": p.get("s_frame"),
                    "h16": one_node(v, 16), "h32": one_node(runs32.get(arm + "_b32"), 32),
                    "s_step32": (runs32.get(arm + "_b32") or {}).get("pieces", {}).get("s_step")}
    return out


def a100():
    seed1 = jl(T.RES, "polaris_opt_round1_7814300", "seed1.json")
    probe = jl(T.RES, "polaris_inrow_7853317", "inrow_probe_prod.json")
    runs = {arm: T.load(job, "o12_extrapolation.json")["solvers"][arm] for arm, job, _s, _k in T.ROWS}
    return gather(seed1, probe, runs, {}), probe


def h100(a):
    seed1 = jl(a.inrow, "seed1.json") if a.inrow else None
    probe = jl(a.inrow, "inrow_probe_prod.json") if a.inrow else None
    runs, runs32 = {}, {}
    for d in ([a.o12] if a.o12 else []) + a.o12tr:
        runs.update((jl(d, "o12_extrapolation.json") or {}).get("solvers", {}))
    if a.o12b32:
        runs32.update((jl(a.o12b32, "o12_extrapolation.json") or {}).get("solvers", {}))
    return gather(seed1, probe, runs, runs32), probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inrow")
    ap.add_argument("--o12")
    ap.add_argument("--o12tr", action="append", default=[])
    ap.add_argument("--o12b32")
    a = ap.parse_args()
    A, pa = a100()
    H, ph = h100(a)
    refA, refH = A["ref"]["s_step"], H["ref"]["s_step"]

    print("| solver | steps / frame | ms / solver step A100 | H100 | A100/H100 | o1.2 s / opt step A100 | H100 | "
          "A100/H100 | × ref A100 | × ref H100 | 1 node b16 (h) A100 | H100 | 1 node b32 (h) H100 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm, *_ in T.ROWS:
        x, y = A[arm], H[arm]
        print(f"| `{arm}` | {x['spf']} | {fmt(x['ms'])} | {fmt(y['ms'])} | {ratio(x['ms'], y['ms'])} | "
              f"{fmt(x['s_step'], 4)} | {fmt(y['s_step'], 4)} | {ratio(x['s_step'], y['s_step'])} | "
              f"{ratio(refA, x['s_step'])} | {ratio(refH, y['s_step'])} | {fmt(x['h16'], 1)} | {fmt(y['h16'], 1)} | "
              f"{fmt(y['h32'], 1)} |")
    print()
    print("o1.2 step = 12 solver frames + the rest (strict-FP32 ML step, data, DDP), 1 node, batch 16:")
    print()
    print("| solver | frames A100 (s) | frames H100 (s) | A100/H100 | ML A100 (s) | ML H100 (s) | A100/H100 | "
          "shrank more |")
    print("|---|---|---|---|---|---|---|---|")
    for arm, *_ in T.ROWS:
        x, y = A[arm], H[arm]
        fa = FRAMES_PER_STEP * x["s_frame"] if x["s_frame"] is not None else None
        fh = FRAMES_PER_STEP * y["s_frame"] if y["s_frame"] is not None else None
        ma = x["s_step"] - fa if fa is not None and x["s_step"] is not None else None
        mh = y["s_step"] - fh if fh is not None and y["s_step"] is not None else None
        more = "—"
        if None not in (fa, fh, ma, mh) and fh > 0 and mh > 0:
            rf, rm = fa / fh, ma / mh
            more = "same (±2%)" if abs(rf - rm) <= 0.02 * max(rf, rm) else ("solver" if rf > rm else "ML step")
        print(f"| `{arm}` | {fmt(fa, 4)} | {fmt(fh, 4)} | {ratio(fa, fh)} | {fmt(ma, 4)} | {fmt(mh, 4)} | "
              f"{ratio(ma, mh)} | {more} |")
    print()
    sa = (pa or {}).get("bench_ms_per_step", {}).get("inrow32_split_ms", {})
    sh = (ph or {}).get("bench_ms_per_step", {}).get("inrow32_split_ms", {})
    print("Per-op split of the inrow32 step (ms, CUDA events, median over steps 5-39):")
    print()
    print("| op | A100 | H100 | A100/H100 |")
    print("|---|---|---|---|")
    for op in list(sa) + [k for k in sh if k not in sa]:
        print(f"| {op} | {fmt(sa.get(op))} | {fmt(sh.get(op))} | {ratio(sa.get(op), sh.get(op))} |")
    ta, th = (sum(sa.values()) if sa else None), (sum(sh.values()) if sh else None)
    print(f"| **sum** | {fmt(ta)} | {fmt(th)} | {ratio(ta, th)} |")
    for k in ("inrow32_eager", "inrow32_graph"):
        va, vh = (pa or {}).get("bench_ms_per_step", {}).get(k), (ph or {}).get("bench_ms_per_step", {}).get(k)
        print(f"| {k} step | {fmt(va)} | {fmt(vh)} | {ratio(va, vh)} |")
    print("TABLE_O12_SOLVERS_MIDWAY_OK")


if __name__ == "__main__":
    main()
