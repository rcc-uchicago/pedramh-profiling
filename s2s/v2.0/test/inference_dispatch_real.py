#!/usr/bin/env python3
"""
Real-PanguModel dispatch latency test.

Same gap-measurement methodology as inference_dispatch_smoke.py, but uses the
actual PanguModel_Plasim with production-shape random inputs instead of a
synthetic stand-in. Designed to answer: under the real model, does the
Midway test partition show the 10-50 ms inter-step gaps that DSI shows?

Key design choices:
- Random tensors (production shape from the YAML config). No HDF5 read.
  We are measuring the GPU-dispatch path, not the data-loading path.
- Random model init by default. Dispatch behaviour does not depend on
  trained weights. Use --load_checkpoint to verify with real weights.
- inference_mode + autocast(bfloat16), matching inference_optimized.py.
- torch.cuda.Event for gap timing. In-process, no profiler attach, no ptrace.
  Works on partitions where nsys kernel tracing is blocked.

Run modes:
- Single GPU:   python v2.0/test/inference_dispatch_real.py
- 4 GPUs:       torchrun --standalone --nproc_per_node=4 v2.0/test/inference_dispatch_real.py

The script prints a per-rank gap distribution into the same buckets as the
DSI nsys table (<=10ms, 10-50ms, 50-100ms, 100-500ms, >500ms).
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

REPO_V20 = Path(__file__).resolve().parents[1]
if str(REPO_V20) not in sys.path:
    sys.path.insert(0, str(REPO_V20))

from networks.pangu import PanguModel_Plasim
from utils.YParams import YParams


def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), local_rank, dist.get_world_size()
    torch.cuda.set_device(0)
    return 0, 0, 1


def make_random_inputs(params, device, batch):
    H, W = params.horizontal_resolution
    n_surf = len(params.surface_variables)
    n_upper = len(params.upper_air_variables)
    n_levels = len(params.levels)
    n_const = len(params.constant_boundary_variables)
    n_var = len(params.varying_boundary_variables)

    surface = torch.randn(batch, n_surf, H, W, device=device, dtype=torch.float32)
    upper_air = torch.randn(batch, n_upper, n_levels, H, W, device=device, dtype=torch.float32)
    constant_boundary = torch.randn(batch, n_const, H, W, device=device, dtype=torch.float32)
    varying_boundary = torch.randn(batch, n_var, H, W, device=device, dtype=torch.float32)
    return surface, upper_air, constant_boundary, varying_boundary


def run_autoregressive(model, surface, upper_air, const_bound, var_bound, steps, record):
    """One autoregressive run. Returns list of inter-step GPU-idle gaps (ms)."""
    gaps = []
    prev_end = None
    outputs_surf = [surface]

    for _ in range(steps):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        start_ev.record()
        out_s, out_u, _, _, _ = model(surface, const_bound, var_bound, upper_air)
        end_ev.record()

        outputs_surf.append(out_s.detach())
        surface = out_s
        upper_air = out_u

        if prev_end is not None and record:
            torch.cuda.synchronize()
            gaps.append(prev_end.elapsed_time(start_ev))

        prev_end = end_ev

    torch.cuda.synchronize()
    return gaps


def report_gaps(rank, label, all_gaps):
    flat = [g for run in all_gaps for g in run]
    if not flat:
        print(f"[rank {rank}] {label}: no gaps recorded", flush=True)
        return
    flat_sorted = sorted(flat)
    n = len(flat)
    buckets = {"<=10ms": 0, "10-50ms": 0, "50-100ms": 0, "100-500ms": 0, ">500ms": 0}
    for g in flat:
        if g > 500:
            buckets[">500ms"] += 1
        elif g > 100:
            buckets["100-500ms"] += 1
        elif g > 50:
            buckets["50-100ms"] += 1
        elif g > 10:
            buckets["10-50ms"] += 1
        else:
            buckets["<=10ms"] += 1

    median = flat_sorted[n // 2]
    p90 = flat_sorted[min(n - 1, int(n * 0.90))]
    p99 = flat_sorted[min(n - 1, int(n * 0.99))]
    mean = statistics.fmean(flat)
    print(
        f"[rank {rank}] {label}: n={n} "
        f"median={median:.3f}ms mean={mean:.3f}ms "
        f"p90={p90:.3f}ms p99={p99:.3f}ms max={max(flat):.3f}ms",
        flush=True,
    )
    for k, v in buckets.items():
        print(f"[rank {rank}]   {k}: {v}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default=str(REPO_V20 / "config" / "exp2.yaml"))
    parser.add_argument("--config", default="S2S")
    parser.add_argument("--steps", type=int, default=60, help="autoregressive steps per run")
    parser.add_argument("--warmup", type=int, default=3, help="warmup runs (not timed)")
    parser.add_argument("--reps", type=int, default=4, help="measured runs")
    parser.add_argument("--batch", type=int, default=0,
                        help="per-GPU batch (0 = derive from YAML batch_size / world_size)")
    parser.add_argument("--no_amp", action="store_true",
                        help="disable bfloat16 autocast (default matches production)")
    parser.add_argument("--load_checkpoint", action="store_true",
                        help="restore weights from params.checkpoint_path (default: random init)")
    args = parser.parse_args()

    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        props = torch.cuda.get_device_properties(device)
        print(f"Device: {props.name} ({props.total_memory/1e9:.1f} GB)", flush=True)
        print(f"World size: {world_size}, steps: {args.steps}, warmup: {args.warmup}, reps: {args.reps}", flush=True)

    params = YParams(args.yaml_config, args.config)
    # Disable HDF5-dependent paths so we can instantiate PanguModel_Plasim without data files
    params["land_variables"] = []
    params["ocean_variables"] = []
    params["mask_output"] = False
    # Some configs (e.g. exp2.yaml line 83) have a malformed num_levels; derive from the levels list instead
    params["num_levels"] = len(params.levels)

    per_gpu_batch = args.batch if args.batch > 0 else max(1, params.batch_size // world_size)
    params["batch_size"] = per_gpu_batch

    if rank == 0:
        print(
            f"Per-GPU batch={per_gpu_batch}, "
            f"H={params.horizontal_resolution[0]}, W={params.horizontal_resolution[1]}, "
            f"surface={len(params.surface_variables)}ch, "
            f"upper_air={len(params.upper_air_variables)}vars × {len(params.levels)}levels = {len(params.upper_air_variables)*len(params.levels)}ch, "
            f"const_bound={len(params.constant_boundary_variables)}ch, "
            f"var_bound={len(params.varying_boundary_variables)}ch",
            flush=True,
        )

    model = PanguModel_Plasim(params, land_mask=None, mask_fill=None).to(device).eval()

    if args.load_checkpoint and os.path.isfile(params.checkpoint_path):
        ckpt = torch.load(params.checkpoint_path, map_location=device)
        sd = ckpt.get("model_state", ckpt.get("model", ckpt))
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if rank == 0:
            print(f"Loaded checkpoint: {params.checkpoint_path}  (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    elif args.load_checkpoint and rank == 0:
        print(f"Checkpoint not found at {params.checkpoint_path} — using random init", flush=True)

    surface, upper_air, const_bound, var_bound = make_random_inputs(params, device, per_gpu_batch)

    if rank == 0:
        print(
            f"Input shapes: surface={tuple(surface.shape)}, "
            f"upper_air={tuple(upper_air.shape)}, "
            f"const_bound={tuple(const_bound.shape)}, "
            f"var_bound={tuple(var_bound.shape)}",
            flush=True,
        )

    amp_kwargs = dict(enabled=not args.no_amp, dtype=torch.bfloat16)

    # Single forward to confirm compute range
    with torch.inference_mode(), torch.amp.autocast("cuda", **amp_kwargs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            _ = model(surface, const_bound, var_bound, upper_air)
        torch.cuda.synchronize()
        single_ms = (time.perf_counter() - t0) / 5 * 1e3
    if rank == 0:
        print(f"Single forward pass: {single_ms:.2f} ms  (target ~14 ms to match production Pangu)", flush=True)

    # Warmup + measured runs
    all_gaps = []
    with torch.inference_mode(), torch.amp.autocast("cuda", **amp_kwargs):
        for _ in range(args.warmup):
            run_autoregressive(model, surface, upper_air, const_bound, var_bound, args.steps, record=False)
        for _ in range(args.reps):
            gaps = run_autoregressive(model, surface, upper_air, const_bound, var_bound, args.steps, record=True)
            all_gaps.append(gaps)

    if dist.is_initialized():
        dist.barrier()

    # Serialise per-rank reporting so output isn't interleaved
    for r in range(world_size):
        if r == rank:
            label = f"PanguModel_Plasim autoregressive ({args.steps} steps × {args.reps} reps)"
            report_gaps(rank, label, all_gaps)
        if dist.is_initialized():
            dist.barrier()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
