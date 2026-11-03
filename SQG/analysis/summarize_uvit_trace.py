"""summarize_uvit_trace.py -- kernel-only breakdown of the torch.profiler chrome trace written by profile_uvit.py (job C).

  python summarize_uvit_trace.py <out>/uvit_trace.json --out <out>/uvit_trace_summary.json

profile_uvit.py's `top_ops` divides each row's self CUDA time by the sum over ALL key_averages() rows, which mixes
aten ops, the kernels they launch, and the fwd/bwd/clip/opt record_function ranges -- kernel time is counted more than
once and a range ("fwd") can top the list. This reads only the GPU-side events (cat "kernel", "gpu_memcpy",
"gpu_memset") of the profiled steps, so every percentage is a share of actual GPU kernel time. Reports: kernel time and
GPU busy fraction per step, top kernels (share, calls/step, mean us), and kernel time by family (depthwise conv, cuDNN
conv, GEMM, layer norm, elementwise, copy, ...). Prints UVIT_TRACE_OK.
"""
import argparse
import json
import re
from collections import defaultdict

FAMILIES = [                                             # first match wins; order matters
    ("depthwise_conv_native", r"conv_depthwise2d"),
    ("cudnn_conv", r"cudnn|implicit_gemm|wgrad|dgrad|fprop|xmma|conv2d|winograd|precomputed_convolve|scudnn"),
    ("gemm", r"gemm|cutlass|cublas"),
    ("layer_norm", r"layer_norm|GammaBeta|LayerNorm"),
    ("softmax_attention", r"softmax|attention|fmha|flash|efficient"),
    ("optimizer", r"multi_tensor|foreach|adam|Adam"),
    ("copy", r"direct_copy|copy_kernel|CatArrayBatched|cat_"),
    ("reduce", r"reduce_kernel|Reduce"),
    ("elementwise", r"elementwise|vectorized|unrolled"),
]


def family(name):
    for fam, pat in FAMILIES:
        if re.search(pat, name):
            return fam
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.trace) as fh:
        ev = json.load(fh)["traceEvents"]

    steps = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev
                   if e.get("ph") == "X" and str(e.get("name", "")).startswith("ProfilerStep#")
                   and e.get("cat") == "user_annotation")
    n_steps = len(steps) or 1
    gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    by_name, calls, by_fam = defaultdict(float), defaultdict(int), defaultdict(float)
    kern_us, memcpy_us, iv = 0.0, 0.0, []
    for e in gpu:
        d = float(e.get("dur", 0.0))
        iv.append((float(e["ts"]), float(e["ts"]) + d))
        if e["cat"] != "kernel":
            memcpy_us += d
            by_fam["memcpy_memset"] += d
            continue
        kern_us += d
        by_name[e["name"]] += d
        calls[e["name"]] += 1
        by_fam[family(e["name"])] += d
    iv.sort()
    busy, cur_s, cur_e = 0.0, None, None                  # union of GPU intervals
    for s, t in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, t
        else:
            cur_e = max(cur_e, t)
    if cur_e is not None:
        busy += cur_e - cur_s
    span = iv[-1][1] - iv[0][0] if iv else 0.0
    gpu_us = kern_us + memcpy_us

    top = [{"name": n[:120], "family": family(n), "pct_of_gpu_time": round(100 * t / gpu_us, 2),
            "calls_per_step": round(calls[n] / n_steps, 1), "avg_us": round(t / calls[n], 1)}
           for n, t in sorted(by_name.items(), key=lambda kv: -kv[1])[:20]]
    out = {"trace": a.trace, "profiled_steps": len(steps),
           "gpu_ms_per_step": round(gpu_us / 1e3 / n_steps, 2), "kernel_ms_per_step": round(kern_us / 1e3 / n_steps, 2),
           "busy_ms_per_step": round(busy / 1e3 / n_steps, 2), "span_ms_per_step": round(span / 1e3 / n_steps, 2),
           "gpu_busy_fraction": round(busy / span, 4) if span else None,
           "kernels_per_step": round(sum(calls.values()) / n_steps, 1), "distinct_kernels": len(by_name),
           "families_pct_of_gpu_time": {f: round(100 * t / gpu_us, 2)
                                        for f, t in sorted(by_fam.items(), key=lambda kv: -kv[1])},
           "max_single_kernel_pct": top[0]["pct_of_gpu_time"] if top else None, "top_kernels": top}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"{len(steps)} steps: GPU {out['gpu_ms_per_step']} ms/step (busy {out['busy_ms_per_step']} of span "
          f"{out['span_ms_per_step']}), {out['kernels_per_step']} kernels/step, {out['distinct_kernels']} distinct")
    print("families % of GPU time: " + ", ".join(f"{k} {v}" for k, v in out["families_pct_of_gpu_time"].items()))
    for t in top[:8]:
        print(f"   {t['pct_of_gpu_time']:5.1f}%  x{t['calls_per_step']}/step  {t['avg_us']} us  [{t['family']}] {t['name'][:70]}")
    print(f"UVIT_TRACE_OK {a.out}")


if __name__ == "__main__":
    main()
