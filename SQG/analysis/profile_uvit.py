"""profile_uvit.py -- Job C: where the ~0.4 s UViT training step goes (no training data: random input).

Same model / optimizer / precision as the SQG trainings (o1.2 and the offline 1-TrainUViT_m16_tendency):
ConvNextUViT(n_feat=32, depths=[2,2,4,4,8], k=7, drop_path=0.1, n_attn=4, heads=8, residual), batch 32 of
(1, 256, 256), AdamW(5e-4, wd 0.05, betas 0.9/0.95), clip 1.0, strict fp32 (torch 2.3: allow_tf32 = False).
Cost does not depend on the data values, so x = randn, y = x + 0.1 randn.

  1. step time (CUDA events) split into fwd / bwd / clip / opt (record_function ranges);
  2. torch.profiler (schedule wait 2 / warmup 3 / active 10, CUDA + CPU, record_shapes): top ops by self CUDA time
     -> <out>/uvit_ops.txt, chrome trace <out>/uvit_trace.json;
  3. information only (NOT bitwise, so not adoptable without the equivalence gate): step time with
     cudnn.benchmark=True and with channels_last, and the max relative difference of their forward output vs the
     baseline -- an upper bound on what layout/algorithm tuning could buy.
Writes <out>/profile_uvit.json; prints UVIT_OK.
"""
import argparse
import json
import os
import statistics
import sys

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function, schedule

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from model_UViT import ConvNextUViT

BATCH, N = 32, 256


def build(dev, seed=0):
    torch.manual_seed(seed)
    model = ConvNextUViT(in_ch=1, out_ch=1, n_feat=32, depths=[2, 2, 4, 4, 8], kernel_size=7, drop_path=0.1,
                         n_attn=4, n_heads=8, residual=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05, betas=(0.9, 0.95))
    return model, opt


def step(model, opt, x, y, ev=None):
    if ev:
        ev[0].record()
    with record_function("fwd"):
        loss = F.mse_loss(model(x), y)
    if ev:
        ev[1].record()
    with record_function("bwd"):
        opt.zero_grad(set_to_none=True)
        loss.backward()
    if ev:
        ev[2].record()
    with record_function("clip"):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if ev:
        ev[3].record()
    with record_function("opt"):
        opt.step()
    if ev:
        ev[4].record()
    return loss


def timed(model, opt, x, y, n_warm=5, n=20):
    for _ in range(n_warm):
        step(model, opt, x, y)
    parts = {"fwd": [], "bwd": [], "clip": [], "opt": [], "total": []}
    for _ in range(n):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        step(model, opt, x, y, ev)
        ev[4].synchronize()
        for k, (a, b) in zip(("fwd", "bwd", "clip", "opt"), zip(ev[:-1], ev[1:])):
            parts[k].append(a.elapsed_time(b))
        parts["total"].append(ev[0].elapsed_time(ev[4]))
    return {k: round(statistics.median(v), 2) for k, v in parts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda:0")
    g = torch.Generator(device=dev).manual_seed(1)
    x = torch.randn(BATCH, 1, N, N, device=dev, generator=g)
    y = x + 0.1 * torch.randn(BATCH, 1, N, N, device=dev, generator=g)
    res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "batch": BATCH}

    model, opt = build(dev)
    res["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
    res["baseline_ms"] = timed(model, opt, x, y)
    res["peak_GiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    print(f"baseline ms (median of 20): {res['baseline_ms']}  peak {res['peak_GiB']} GiB", flush=True)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
                 schedule=schedule(wait=2, warmup=3, active=10)) as prof:
        for _ in range(15):
            step(model, opt, x, y)
            prof.step()
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
    with open(os.path.join(args.out, "uvit_ops.txt"), "w") as fh:
        fh.write(table)
    prof.export_chrome_trace(os.path.join(args.out, "uvit_trace.json"))
    tot = sum(e.self_cuda_time_total for e in prof.key_averages())
    res["top_ops"] = [{"op": e.key, "self_cuda_pct": round(100 * e.self_cuda_time_total / tot, 1),
                       "calls_per_step": round(e.count / 10, 1)}
                      for e in sorted(prof.key_averages(), key=lambda e: -e.self_cuda_time_total)[:15]]
    print("top ops by self CUDA time:", flush=True)
    for t in res["top_ops"][:10]:
        print(f"   {t['self_cuda_pct']:5.1f}%  x{t['calls_per_step']}/step  {t['op']}", flush=True)

    # ---- information only: tuning upper bounds (not bitwise) --------------------------------------------------
    with torch.no_grad():
        model.eval()
        ref_out = model(x)
    variants = {}
    torch.backends.cudnn.benchmark = True
    m2, o2 = build(dev)
    variants["cudnn_benchmark"] = (m2, o2, x, y)
    torch.backends.cudnn.benchmark = False
    m3, o3 = build(dev)
    m3 = m3.to(memory_format=torch.channels_last)
    variants["channels_last"] = (m3, o3, x.contiguous(memory_format=torch.channels_last),
                                 y.contiguous(memory_format=torch.channels_last))
    res["variants"] = {}
    for name, (m, o, xv, yv) in variants.items():
        torch.backends.cudnn.benchmark = name == "cudnn_benchmark"
        m.load_state_dict(model.state_dict())           # same weights as the profiled baseline model
        with torch.no_grad():
            m.eval()
            out = m(xv)
            rel = ((out - ref_out).abs().max() / ref_out.abs().max()).item()
        m.train()
        res["variants"][name] = {"ms": timed(m, o, xv, yv), "fwd_max_rel_diff_vs_baseline": rel}
        print(f"{name}: {res['variants'][name]}", flush=True)
    torch.backends.cudnn.benchmark = False

    with open(os.path.join(args.out, "profile_uvit.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"UVIT_OK -> {args.out}/profile_uvit.json")


if __name__ == "__main__":
    main()
