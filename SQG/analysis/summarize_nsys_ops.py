"""summarize_nsys_ops.py -- GPU time per XLA op per solver step, from the SQLite export of an nsys capture.

  python summarize_nsys_ops.py <capture.sqlite> --range-regex '_rep_' --steps-per-range 480 --out <json>

Only GPU work launched inside the harness's NVTX ranges (domain 0, torch.cuda.nvtx) that match --range-regex counts.
Each kernel / memcpy / memset follows its CUDA API call (by correlationId). The call goes to the innermost range of
another NVTX domain open on its thread when it started: XLA's `Thunk:#name=...,hlo_op=<op>#` annotation, i.e. the
HLO op that launched it. An FFT op's thunk holds its input copy, the cuFFT kernels and the 1/N^2 scal. The op key
also carries the XLA module (`XlaModule:#...hlo_module=<m>,program_id=<id>#`), so two jitted functions that both
have an op of the same name stay apart.

Per op: GPU ms per solver step, share of the GPU-busy time, launches per step, its kernels (name, grid x block, count
and mean us) and its copies (kind, bytes per step), and its place in execution order (first start in the first
range). Also: the GPU-busy fraction of each range's span (first start to last end of the GPU work launched in it), and
GPU time per kernel name. Stdlib only (sqlite3; correlation is joined in Python dicts, never in SQL): runs inside the
job or on a login node. Prints NSYS_OPS_OK.
"""
import argparse
import bisect
import collections
import json
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_nsys_phases import COPY_KIND, api_name, enum, sweep, table   # noqa: E402

THUNK_OP = re.compile(r"hlo_op=([^#,]+)")
FFT_TYPE = re.compile(r"fft_type=(?:\w+\.)*(\w+)")
MODULE = re.compile(r"^XlaModule:.*?hlo_module=([^#,]+).*?program_id=(\d+)")


def op_label(stack):
    """XLA range stack (outer, ..., inner) -> (module, op, fft type or "")."""
    module = "-"
    for s in stack:
        m = MODULE.search(s)
        if m:
            module = f"{m.group(1)}#{m.group(2)}"
    inner = stack[-1] if stack else ""
    m = THUNK_OP.search(inner)
    if not m:
        return module, "(no thunk)", ""
    ft = FFT_TYPE.search(inner)
    return module, m.group(1), ft.group(1) if ft else ""


def columns(cur, name):
    return [r[1] for r in cur.execute(f"pragma table_info({name})")]


def summarize(path, range_regex, steps_per_range, domain=0):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    cur = db.cursor()
    strings = dict(cur.execute("select id, value from StringIds"))
    copy_label = enum(cur, "ENUM_CUDA_MEMCPY_OPER", None)
    if copy_label is None:
        copy_kind = COPY_KIND
    else:
        by_label = {"Host-to-Device": "htod", "Device-to-Host": "dtoh", "Device-to-Device": "dtod"}
        copy_kind = {i: by_label[lab] for i, lab in copy_label.items() if lab in by_label}

    harness, other = collections.defaultdict(list), collections.defaultdict(list)
    for start, end, text, text_id, tid, dom in cur.execute(
            "select start, end, text, textId, globalTid, domainId from NVTX_EVENTS "
            "where eventType = 59 and end is not null"):
        name = text if text is not None else strings.get(text_id, "?")
        (harness if (dom or 0) == domain else other)[tid].append((start, end, name))
    rre = re.compile(range_regex)
    windows = sorted((s, e, n) for rs in harness.values() for s, e, n in rs if rre.search(n))
    if not windows:
        raise SystemExit(f"ERROR NSYS_OPS_NO_RANGES: no domain-{domain} NVTX range matches {range_regex!r}")
    starts = [w[0] for w in windows]

    def window_of(t):
        i = bisect.bisect_right(starts, t) - 1
        return i if i >= 0 and t <= windows[i][1] else None

    # ---- API calls launched inside a window (any thread), and the XLA op open on their thread ----
    api, by_tid = {}, collections.defaultdict(list)
    for cid, start, name_id, tid in cur.execute(
            "select correlationId, start, nameId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"):
        w = window_of(start)
        if w is not None:
            api[cid] = (w, api_name(strings.get(name_id, "?")))
            by_tid[tid].append((start, cid))
    label = {}
    for tid, events in by_tid.items():
        paths, _ = sweep(other.get(tid, []), events)
        for cid, stack in paths.items():
            label[cid] = op_label(stack)

    # ---- GPU activities ----
    acts = []                                       # (start, end, cid, kind, name, shape, nbytes)
    kcols = columns(cur, "CUPTI_ACTIVITY_KIND_KERNEL") if table(cur, "CUPTI_ACTIVITY_KIND_KERNEL") else []
    if kcols:
        name_col = "shortName" if "shortName" in kcols else "demangledName"
        geo = [c for c in ("gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ") if c in kcols]
        sel = ", ".join(["start", "end", "correlationId", name_col] + geo)
        for row in cur.execute(f"select {sel} from CUPTI_ACTIVITY_KIND_KERNEL"):
            start, end, cid, nid = row[:4]
            g = row[4:]
            shape = (f"{g[0] * g[1] * g[2]}x{g[3] * g[4] * g[5]}" if len(g) == 6 else "")
            nm = strings.get(nid, nid) if isinstance(nid, int) else nid
            acts.append((start, end, cid, "kernel", str(nm), shape, 0))
    if table(cur, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        for start, end, cid, nbytes, kind in cur.execute(
                "select start, end, correlationId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"):
            acts.append((start, end, cid, "memcpy", f"memcpy_{copy_kind.get(kind, kind)}", "", nbytes))
    if table(cur, "CUPTI_ACTIVITY_KIND_MEMSET") and "correlationId" in columns(cur, "CUPTI_ACTIVITY_KIND_MEMSET"):
        for start, end, cid, nbytes in cur.execute(
                "select start, end, correlationId, bytes from CUPTI_ACTIVITY_KIND_MEMSET"):
            acts.append((start, end, cid, "memset", "memset", "", nbytes))

    nwin = len(windows)
    steps = nwin * steps_per_range
    ops = collections.defaultdict(lambda: {"gpu_ns": 0, "activities": 0, "first": None,
                                           "kernels": collections.defaultdict(lambda: [0, 0]),
                                           "copies": collections.defaultdict(lambda: [0, 0, 0])})
    by_kernel = collections.defaultdict(lambda: [0, 0])
    intervals = [[] for _ in range(nwin)]
    for start, end, cid, kind, name, shape, nbytes in acts:
        if cid not in api:
            continue
        w = api[cid][0]
        module, op, ftype = label.get(cid, ("-", "(no thunk)", ""))
        o = ops[(module, op, ftype)]
        dur = end - start
        o["gpu_ns"] += dur
        o["activities"] += 1
        if w == 0 and (o["first"] is None or start < o["first"]):
            o["first"] = start
        if kind == "kernel":
            k = o["kernels"][(name, shape)]
            by_kernel[name][0] += 1
            by_kernel[name][1] += dur
        else:
            k = o["copies"][name]
            k[2] += nbytes
        k[0] += 1
        k[1] += dur
        intervals[w].append((start, end))

    busy = span = 0
    for iv in intervals:
        if not iv:
            continue
        iv.sort()
        span += iv[-1][1] - iv[0][0]
        cs, ce = iv[0]
        for s, e in iv[1:]:
            if s > ce:
                busy += ce - cs
                cs, ce = s, e
            else:
                ce = max(ce, e)
        busy += ce - cs
    total_gpu = sum(o["gpu_ns"] for o in ops.values())
    modules = {m for m, _, _ in ops}
    t0 = windows[0][0]

    def us(ns):
        return round(ns / 1e3 / steps, 2)

    out_ops = []
    for (module, op, ftype), o in ops.items():
        out_ops.append({
            "op": op, "fft_type": ftype, "module": module,
            "us_per_step": us(o["gpu_ns"]), "share_of_gpu": round(o["gpu_ns"] / total_gpu, 4) if total_gpu else None,
            "activities_per_step": round(o["activities"] / steps, 3),
            "first_start_us_in_range0": None if o["first"] is None else round((o["first"] - t0) / 1e3, 1),
            "kernels": [{"name": n, "grid_x_block": sh, "per_step": round(c / steps, 3),
                         "mean_us": round(t / c / 1e3, 2), "us_per_step": us(t)}
                        for (n, sh), (c, t) in sorted(o["kernels"].items(), key=lambda kv: -kv[1][1])],
            "copies": [{"kind": n, "per_step": round(c / steps, 3), "bytes_per_step": round(b / steps, 1),
                        "us_per_step": us(t)} for n, (c, t, b) in o["copies"].items()]})
    out_ops.sort(key=lambda r: -r["us_per_step"])
    order = [r["op"] for r in sorted((r for r in out_ops if r["first_start_us_in_range0"] is not None),
                                     key=lambda r: r["first_start_us_in_range0"])]
    return {
        "sqlite": path, "range_regex": range_regex, "ranges": nwin, "steps_per_range": steps_per_range,
        "steps": steps, "modules": sorted(modules),
        "per_step": {"gpu_sum_us": us(total_gpu), "busy_us": us(busy), "span_us": us(span),
                     "idle_us": us(span - busy), "gpu_busy_fraction": round(busy / span, 4) if span else None,
                     "range_wall_us": us(sum(e - s for s, e, _ in windows))},
        "ops": out_ops, "execution_order_range0": order,
        "kernels_by_name": [{"name": n, "per_step": round(c / steps, 3), "us_per_step": us(t)}
                            for n, (c, t) in sorted(by_kernel.items(), key=lambda kv: -kv[1][1])],
        "api_calls_in_ranges": len(api), "activities_outside_ranges": sum(1 for a in acts if a[2] not in api)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--out", required=True)
    ap.add_argument("--range-regex", default=r"_rep_\d+$", help="harness NVTX ranges that hold the captured steps")
    ap.add_argument("--steps-per-range", type=int, required=True, help="solver steps inside each range")
    ap.add_argument("--domain", type=int, default=0, help="NVTX domain id of the harness ranges (torch: 0)")
    a = ap.parse_args()
    r = summarize(a.sqlite, a.range_regex, a.steps_per_range, a.domain)
    with open(a.out, "w") as fh:
        json.dump(r, fh, indent=1)
    p = r["per_step"]
    print(f"{os.path.basename(a.sqlite)}: {r['ranges']} ranges x {r['steps_per_range']} steps; per step: GPU busy "
          f"{p['busy_us']} of {p['span_us']} us ({p['gpu_busy_fraction']}), idle {p['idle_us']} us")
    multi = len(r["modules"]) > 1
    for o in r["ops"][:16]:
        ks = "; ".join(f"{k['name'][:28]} {k['grid_x_block']} x{k['per_step']:g} {k['mean_us']}us"
                       for k in o["kernels"][:3])
        cs = "; ".join(f"{c['kind']} {c['bytes_per_step'] / 1e6:.1f} MB {c['us_per_step']}us" for c in o["copies"])
        mod = f" [{o['module']}]" if multi else ""
        print(f"  {o['op'][:30] + (' ' + o['fft_type'] if o['fft_type'] else ''):38s}{mod} {o['us_per_step']:9.1f} us "
              f"{100 * (o['share_of_gpu'] or 0):5.1f}%  {ks}{' | ' + cs if cs else ''}")
    print(f"NSYS_OPS_OK {a.out}")


if __name__ == "__main__":
    main()
