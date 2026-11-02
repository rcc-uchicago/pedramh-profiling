"""summarize_nsys.py -- per-solver-step numbers from `nsys stats --format csv` exports of a profile_solver.py capture.

  python summarize_nsys.py <prefix> --steps 2400 --out <json>
reads <prefix>_cuda_gpu_trace.csv, <prefix>_cuda_api_sum.csv, <prefix>_cuda_gpu_kern_sum.csv and reports:
GPU busy fraction of the captured span (union of kernel + memcpy intervals), kernels / DtoH copies / sync calls per
solver step, and the top kernels by time. Note: captured with --cuda-graph-trace=node, so absolute times are
slightly inflated; take ms/step from profile_solver_bench.json.
"""
import argparse
import csv
import json
import re


def rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def col(r, *names):
    for n in names:
        if n in r and r[n] not in ("", None):
            return r[n]
    raise KeyError(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--steps", type=int, required=True, help="solver steps inside the capture")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    trace = rows(f"{a.prefix}_cuda_gpu_trace.csv")
    iv, n_kern, n_dtoh, n_htod, dtoh_bytes = [], 0, 0, 0, 0.0
    for r in trace:
        start, dur = float(col(r, "Start (ns)")), float(col(r, "Duration (ns)"))
        name = col(r, "Name")
        iv.append((start, start + dur))
        if "memcpy Device-to-Host" in name:
            n_dtoh += 1
            try:
                dtoh_bytes += float(col(r, "Bytes (MB)")) * 1e6
            except KeyError:
                pass
        elif "memcpy Host-to-Device" in name:
            n_htod += 1
        elif not name.startswith("[CUDA"):
            n_kern += 1
    iv.sort()
    busy, cur_s, cur_e = 0.0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    span = iv[-1][1] - iv[0][0] if iv else 0.0

    api = {col(r, "Name"): int(float(col(r, "Num Calls"))) for r in rows(f"{a.prefix}_cuda_api_sum.csv")}
    sync = {k: v for k, v in api.items() if re.search(r"Synchronize|EventQuery|StreamQuery", k)}
    memcpy_api = {k: v for k, v in api.items() if re.search(r"[Mm]emcpy", k)}
    launches = {k: v for k, v in api.items() if re.search(r"LaunchKernel|GraphLaunch|LaunchCooperative", k)}

    kern = rows(f"{a.prefix}_cuda_gpu_kern_sum.csv")
    top = [{"name": col(r, "Name")[:90], "time_pct": float(col(r, "Time (%)")),
            "instances_per_step": round(int(float(col(r, "Instances"))) / a.steps, 2),
            "avg_us": round(float(col(r, "Avg (ns)")) / 1e3, 1)} for r in kern[:15]]

    s = a.steps
    out = {"steps": s, "span_ms_per_step": round(span / 1e6 / s, 4), "busy_ms_per_step": round(busy / 1e6 / s, 4),
           "gpu_busy_fraction": round(busy / span, 4) if span else None,
           "kernels_per_step": round(n_kern / s, 2), "dtoh_copies_per_step": round(n_dtoh / s, 3),
           "dtoh_bytes_per_step": round(dtoh_bytes / s, 1), "htod_copies_per_step": round(n_htod / s, 3),
           "sync_api_calls_per_step": {k: round(v / s, 3) for k, v in sync.items()},
           "memcpy_api_calls_per_step": {k: round(v / s, 3) for k, v in memcpy_api.items()},
           "launch_api_calls_per_step": {k: round(v / s, 3) for k, v in launches.items()},
           "top_kernels": top}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"{a.prefix}: busy {out['busy_ms_per_step']} of {out['span_ms_per_step']} ms/step "
          f"({out['gpu_busy_fraction']}), {out['kernels_per_step']} kernels/step, "
          f"{out['dtoh_copies_per_step']} DtoH/step, syncs/step {out['sync_api_calls_per_step']}")
    for t in top[:6]:
        print(f"   {t['time_pct']:5.1f}%  x{t['instances_per_step']}/step  {t['avg_us']} us  {t['name']}")
    print(f"NSYS_SUMMARY_OK {a.out}")


if __name__ == "__main__":
    main()
