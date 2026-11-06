"""analyze_nsight.py <results dir> -- tables for notes Part X from the Nsight job's summaries (stdlib only).

  python3 SQG/analysis/opt/nsight/analyze_nsight.py SQG/analysis/results/polaris_nsight_lean_7819082
Note: ncu_window_lean_u1_xla.json's per-step numbers assume 3 steps per period (the job's setting); the period it
found (5 XLA kernels) is ONE step, because the 3-step body reuses one kernel per fusion -- multiply them by 3.

Per XLA op of lean32 (nsys): us/step, the op's ideal DRAM bytes from its HLO shapes (each operand read once, each
output written once; an FFT pass reads and writes its whole array), the implied GB/s, and the headroom at a practical
1.35 TB/s (what u*theta and the cuFFT row passes reach, job F). Then the ncu window, the probes and the cuFFT arm.
"""
import json
import os
import re
import sys

R = sys.argv[1]
PRACT = 1.35e12                                         # B/s: u*theta, scal, vector_fft rows reach ~1.3 TB/s (job F/E)
C64H, F32H = 4096 * 2049 * 8, 4096 * 2049 * 4           # half-plane complex64 / float32
C64F, F32F = 4096 * 4096 * 8, 4096 * 4096 * 4           # full-plane complex64 / float32
# ideal bytes per op instance (lean32), from hlo_lean32.txt shapes
IDEAL = [
    (r"^input_concatenate_fusion", "build_w", C64H + F32H + C64F),
    (r"^fft.*\bFFT$", "C2C (u - i v)", 2 * 2 * C64F),
    (r"^fft.*\bIRFFT$", "IRFFT (theta)", 2 * C64H + 2 * C64H + (C64H + F32F) + 2 * F32F),
    (r"^fft.*\bRFFT$", "RFFT (u theta, v theta)", (F32F + C64H) + 2 * C64H),
    (r"^loop_multiply_fusion$|^loop_multiply_fusion\.1$", "products", F32F + C64F + 2 * F32F),
    (r"^loop_add_multiply_fusion", "AB3", F32H + 5 * C64H + 2 * C64H),
    (r"^wrapped_copy\.[34]$", "tendency copy", 2 * C64H),
]


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def classify(op, ftype):
    key = f"{op} {ftype}".strip()
    for rx, label, b in IDEAL:
        if re.search(rx, key) or re.search(rx, op):
            return label, b
    return None, None


ops = load("nsys_ops_lean32.json")
if ops:
    p = ops["per_step"]
    print(f"lean32 nsys: GPU busy {p['busy_us']} of {p['span_us']} us/step ({p['gpu_busy_fraction']}), "
          f"idle {p['idle_us']} us, sum {p['gpu_sum_us']} us; modules {ops['modules']}")
    agg = {}
    for o in ops["ops"]:
        label, b = classify(o["op"], o["fft_type"])
        label = label or f"other: {o['op']}"
        a = agg.setdefault(label, {"us": 0.0, "n": 0.0, "bytes": 0.0, "ops": []})
        a["us"] += o["us_per_step"]
        a["ops"].append(o["op"])
        if b:
            # instances per step of this op: an op name appears once per 2-step body, i.e. 0.5 per step
            per = max([k["per_step"] for k in o["kernels"]] + [c["per_step"] for c in o["copies"]] + [0])
            a["n"] += per
            a["bytes"] += b * per
    tot_us = sum(a["us"] for a in agg.values())
    tot_head = 0.0
    print(f"{'op class':26s} {'us/step':>8s} {'share':>6s} {'ideal MB/step':>13s} {'GB/s':>7s} "
          f"{'at 1.35 TB/s':>12s} {'headroom us':>11s}")
    for label, a in sorted(agg.items(), key=lambda kv: -kv[1]["us"]):
        if a["bytes"]:
            gbs = a["bytes"] / (a["us"] * 1e-6) / 1e9
            floor = a["bytes"] / PRACT * 1e6
            head = a["us"] - floor
            tot_head += max(head, 0)
            print(f"{label:26s} {a['us']:8.1f} {a['us'] / tot_us:6.1%} {a['bytes'] / 1e6:13.1f} {gbs:7.0f} "
                  f"{floor:12.1f} {head:11.1f}")
        elif a["us"] >= 1:
            print(f"{label[:26]:26s} {a['us']:8.1f} {a['us'] / tot_us:6.1%}")
    ideal = sum(a["bytes"] for a in agg.values())
    print(f"total {tot_us:.1f} us/step; ideal {ideal / 1e6:.0f} MB/step -> {ideal / PRACT * 1e6:.0f} us at 1.35 TB/s; "
          f"headroom (sum of positive) {tot_head:.0f} us")
    print("kernels (lean32, nsys):")
    for o in ops["ops"][:12]:
        for k in o["kernels"]:
            print(f"   {o['op'][:28]:28s} {o['fft_type']:6s} {k['name'][:34]:34s} {k['grid_x_block']:>14s} "
                  f"x{k['per_step']:<5g} {k['mean_us']:8.1f} us")
        for c in o["copies"]:
            print(f"   {o['op'][:28]:28s} {o['fft_type']:6s} {c['kind']:34s} {'':>14s} x{c['per_step']:<5g} "
                  f"{c['us_per_step'] / max(c['per_step'], 1e-9):8.1f} us  {c['bytes_per_step'] / max(c['per_step'], 1e-9) / 1e6:.1f} MB")

for tag in ("lean32_window", "lean32_u1_xla", "lean_u1_xla"):
    w = load(f"ncu_window_{tag}.json")
    if not w:
        continue
    print(f"\nncu {tag}: {w['launches']} launches, period {w['period']}")
    for x in (w.get("last_period") or w["by_signature"]):
        name = x.get("name") or x.get("signature")
        print(f"   {name[:40]:40s} {str(x.get('grid', ''))[:16]:16s} {str(x.get('block', ''))[:12]:12s} "
              f"{x['duration_us']!s:>8s} us rd {x['dram_read_MB']!s:>7s} wr {x['dram_write_MB']!s:>7s} MB "
              f"{x['achieved_GBps']!s:>6s} GB/s {x['dram_pct_of_peak']!s:>5s}% L2 {x['l2_hit_pct']!s:>5s} "
              f"ld {x['ld_sector_eff_pct']!s:>5s} st {x['st_sector_eff_pct']!s:>5s} {x.get('top_stall')}")
    if w.get("period_totals"):
        print(f"   totals {w['period_totals']}")

for b in ("bench_default.json", "bench_cufft114.json"):
    s = load(b)
    if not s:
        continue
    print(f"\n{b}: libs {[os.path.basename(x) + ' @ ' + os.path.dirname(x)[-50:] for x in s.get('loaded_cuda_libs', [])]}")
    for n, v in s["variants"].items():
        print(f"   {n:16s} ms {v.get('ms_per_step')!s:>8s} vs_ref {v.get('speedup_vs_ref')!s:>6s} "
              f"vs_lean32 {v.get('speedup_vs_lean32')!s:>6s} check {v.get('check_pass', '-')!s:5s} "
              f"rel40 {v.get('check_40', {}).get('rel_l2_max', '-')!s:>9s} bit40 {v.get('check_40', {}).get('bitwise', '-')!s:5s} "
              f"bit480 {v.get('check_480', {}).get('bitwise', '-')!s:5s} copies {v.get('hlo', {}).get('full_size_copies')} "
              f"fusions {v.get('hlo', {}).get('fusion')}")
    print(f"   fft micro {s.get('fft_ms_per_call_4096')}")
    if s.get("frames"):
        print(f"   frames {s['frames']}")
