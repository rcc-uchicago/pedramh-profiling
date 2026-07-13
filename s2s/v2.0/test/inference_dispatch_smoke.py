#!/usr/bin/env python3
"""
Inference dispatch latency smoke test.

Background
----------
During S2S autoregressive inference, the model runs 60 sequential forward
passes in a Python loop — one per forecast time step. After each forward
pass the output becomes the input for the next step. Between every two
consecutive GPU kernel sequences there is a Python-level re-entry: list
appends, variable reassignment, and a new model() call that the CPU must
dispatch.

If the dispatch overhead between steps is large (10–50 ms), it directly
explains the inter-kernel idle gaps we observed in the DSI nsys profiles.
If it is sub-millisecond, the cause is hardware (NUMA dispatch latency or
PCIe contention) rather than Python overhead.

What this test measures
-----------------------
The GPU-side idle time between consecutive forward passes is measured using
paired torch.cuda.Events. An event is recorded on the GPU stream immediately
after each forward pass ends (end_event) and immediately before the next one
starts (start_event). The time between end_event[N] and start_event[N+1] is
the period when the GPU had no work queued — this is the pure dispatch gap.

Four patterns are compared:

  A. Python loop, list append       — matches inference.py / inference_optimized.py
                                      exactly: output appended to a list each step.

  B. Python loop, in-place write    — pre-allocated output tensor, no list append.
                                      Tests whether list append contributes to the gap.

  C. torch.compile (reduce-overhead) — JIT-compiled forward pass. Fuses small
                                      ops and reduces Python overhead per call.

  D. CUDA Graph                     — the entire 60-step loop captured as a single
                                      GPU command sequence. Python dispatch overhead
                                      is eliminated entirely; the CPU issues one
                                      replay command.

The synthetic forward pass
--------------------------
The real PanguModel forward pass is ~14 ms of GPU active time at batch=1.
We approximate this with a sequence of matmuls and layer norms at the same
memory footprint (surface + upper-air tensor sizes). The synthetic pass is
not architecturally identical to Pangu but reproduces the same dispatch
pattern: one Python function call → many GPU kernels → return tensor.

Run on any GPU node:
    PYTHONPATH=v2.0 python v2.0/test/inference_dispatch_smoke.py

The output shows per-step dispatch gap statistics. If pattern A has gaps
>>1 ms and pattern D has gaps ~0, Python dispatch is the bottleneck.
If A and D have similar gaps, the bottleneck is hardware (NUMA/PCIe).
"""

import time
import statistics
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Synthetic forward pass
# ---------------------------------------------------------------------------

# Tensor shapes matching the actual S2S inference tensors (batch=1).
# Surface and upper-air are the main inputs; we ignore diagnostic/boundary
# for simplicity since they don't change the dispatch pattern.
SURFACE_SHAPE   = (1, 16,  128, 256)   # ~12.5 MB
UPPER_AIR_SHAPE = (1, 104, 128, 256)   # ~81.8 MB

# Number of autoregressive time steps — matches inference_steps for 6h timestep
# over a 15-day forecast: (24 * 15) // 6 = 60
INFERENCE_STEPS = 60

WARMUP = 5    # steps discarded before timing starts
NREPS  = 3    # full 60-step runs to average over


class SyntheticForward(nn.Module):
    """
    Approximate the PanguModel forward pass compute without its architecture.

    The real forward pass takes ~14 ms on an H100 at batch=1. This module
    produces a GPU kernel sequence of similar duration using Conv2d layers
    operating on the actual spatial dimensions (128×256).

    The previous version used nn.Linear on the fully flattened tensors
    (e.g. nn.Linear(3_407_872, 512) for upper-air), which created ~7 GB
    weight matrices and caused OOM. Conv2d operates channel-wise and has
    negligible parameter memory: Conv2d(128, 128, 3) weights are only ~590 KB.

    The goal is not to replicate Pangu but to produce enough GPU work that
    the inter-step dispatch gap is measurable relative to real compute time.
    hidden=128 channels over a 128×256 spatial map gives ~5–15 ms per forward
    pass on H100/H200, which is the right order of magnitude.
    """

    def __init__(self, hidden: int = 128):
        super().__init__()
        surf_c  = SURFACE_SHAPE[1]    # 16
        upper_c = UPPER_AIR_SHAPE[1]  # 104

        # Surface branch: 16 → hidden → hidden → 16
        self.surf_net = nn.Sequential(
            nn.Conv2d(surf_c,  hidden, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden,  hidden, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden,  hidden, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden,  surf_c, 3, padding=1, bias=False),
        )
        # Upper-air branch: 104 → hidden → hidden → 104
        self.upper_net = nn.Sequential(
            nn.Conv2d(upper_c, hidden, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden,  hidden, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden,  hidden, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden,  upper_c, 3, padding=1, bias=False),
        )

    def forward(self, surface: torch.Tensor,
                upper_air: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Takes surface and upper-air tensors, returns updated versions.
        The autoregressive loop feeds each output back as the next input.
        """
        return self.surf_net(surface), self.upper_net(upper_air)


# ---------------------------------------------------------------------------
# Timing helpers using CUDA Events
# ---------------------------------------------------------------------------

def gpu_gap_ms(start_ev: torch.cuda.Event,
               end_ev:   torch.cuda.Event) -> float:
    """Return elapsed GPU time in ms between two recorded CUDA events."""
    torch.cuda.synchronize()
    return start_ev.elapsed_time(end_ev)


def run_loop(model, surface, upper_air, steps, record_events=True):
    """
    Run the autoregressive loop with list-append pattern (inference.py style).

    Returns (output_list, gap_ms_list) where gap_ms_list[i] is the GPU-idle
    time between the end of step i and the start of step i+1.
    """
    outputs = [surface]
    gaps    = []
    prev_end = None

    for step in range(steps):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev   = torch.cuda.Event(enable_timing=True)

        start_ev.record()
        out_s, out_u = model(surface, upper_air)
        end_ev.record()

        # List append — matches the real inference loop
        outputs.append(out_s.detach())
        surface   = out_s
        upper_air = out_u

        if prev_end is not None and record_events:
            # Gap = time from end of previous step to start of this step
            gaps.append(gpu_gap_ms(prev_end, start_ev))

        prev_end = end_ev

    torch.cuda.synchronize()
    return outputs, gaps


def run_loop_preallocated(model, surface, upper_air, steps):
    """
    Run with pre-allocated output tensor instead of list append.
    Tests whether list allocation contributes to the dispatch gap.
    """
    B = surface.shape[0]
    out_surf_buf = torch.empty(B, steps + 1, *surface.shape[1:],
                               device=surface.device)
    out_surf_buf[:, 0] = surface
    gaps     = []
    prev_end = None

    for step in range(steps):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev   = torch.cuda.Event(enable_timing=True)

        start_ev.record()
        out_s, upper_air = model(surface, upper_air)
        end_ev.record()

        # In-place write instead of list append
        out_surf_buf[:, step + 1] = out_s.detach()
        surface = out_s

        if prev_end is not None:
            gaps.append(gpu_gap_ms(prev_end, start_ev))
        prev_end = end_ev

    torch.cuda.synchronize()
    return gaps


def run_loop_cuda_graph(model, surface, upper_air, steps):
    """
    Capture the single-step forward pass as a CUDA Graph then replay it
    60 times. The GPU receives all work as one command sequence — Python
    only issues a single replay call per step rather than dispatching the
    full kernel sequence from Python.

    This is the upper bound on what software optimisation can achieve:
    if graph replay gaps are still 10–50 ms, the cause is hardware not Python.
    """
    # Warm up the graph capture with a few un-timed replays
    s_in  = surface.clone()
    u_in  = upper_air.clone()
    s_out = torch.empty_like(s_in)
    u_out = torch.empty_like(u_in)

    # Capture — all kernel launches inside this block are recorded
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        s_out, u_out = model(s_in, u_in)

    gaps     = []
    prev_end = None

    for step in range(steps):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev   = torch.cuda.Event(enable_timing=True)

        # Copy new inputs into the graph's input buffers
        s_in.copy_(surface)
        u_in.copy_(upper_air)

        start_ev.record()
        g.replay()   # single CPU call replays the entire captured kernel sequence
        end_ev.record()

        surface   = s_out.clone()
        upper_air = u_out.clone()

        if prev_end is not None:
            gaps.append(gpu_gap_ms(prev_end, start_ev))
        prev_end = end_ev

    torch.cuda.synchronize()
    return gaps


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(label: str, all_gaps: list[list[float]]) -> None:
    """
    Print gap statistics across all repetitions.
    all_gaps is a list of per-run gap lists (one list per NREPS run).
    """
    flat = [g for run in all_gaps for g in run]
    if not flat:
        print(f"  {label}: no gap data")
        return
    print(f"  {label:<45}"
          f"  median={statistics.median(flat):6.3f}ms"
          f"  mean={statistics.fmean(flat):6.3f}ms"
          f"  max={max(flat):6.3f}ms"
          f"  n={len(flat)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        print("No CUDA device — this test requires a GPU.")
        return

    device = torch.device("cuda")
    props  = torch.cuda.get_device_properties(device)
    print(f"\nDevice: {props.name}  ({props.total_memory/1e9:.1f} GB)")
    print(f"Steps per run: {INFERENCE_STEPS}  |  Warmup runs: {WARMUP}  "
          f"|  Measured runs: {NREPS}")
    print(f"Tensor sizes — surface: {SURFACE_SHAPE}, upper_air: {UPPER_AIR_SHAPE}")

    model = SyntheticForward().to(device).eval()

    # Measure a single forward pass to confirm we're in the right compute range
    surf  = torch.randn(SURFACE_SHAPE,   device=device)
    upper = torch.randn(UPPER_AIR_SHAPE, device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(20):
            _, _ = model(surf, upper)
    torch.cuda.synchronize()
    single_ms = (time.perf_counter() - t0) / 20 * 1e3
    print(f"\nSingle forward pass: ~{single_ms:.1f} ms  "
          f"(target ~14 ms to match real PanguModel)")

    print(f"\n{'='*75}")
    print(f"  Inter-step dispatch gaps  "
          f"(GPU idle between end of step N and start of step N+1)")
    print(f"  If gaps >> 0 ms: Python dispatch overhead is real.")
    print(f"  If gaps ≈ 0 ms with list-append but not CUDA Graph: Python is culprit.")
    print(f"  If gaps persist in CUDA Graph: hardware (NUMA/PCIe) is culprit.")
    print(f"{'='*75}")

    with torch.inference_mode():
        # Warmup
        for _ in range(WARMUP):
            surf  = torch.randn(SURFACE_SHAPE,   device=device)
            upper = torch.randn(UPPER_AIR_SHAPE, device=device)
            run_loop(model, surf, upper, INFERENCE_STEPS, record_events=False)

        # A: Python loop with list append (inference.py pattern)
        gaps_a = []
        for _ in range(NREPS):
            surf  = torch.randn(SURFACE_SHAPE,   device=device)
            upper = torch.randn(UPPER_AIR_SHAPE, device=device)
            _, gaps = run_loop(model, surf, upper, INFERENCE_STEPS)
            gaps_a.append(gaps)

        # B: Python loop with pre-allocated tensor (no list append)
        gaps_b = []
        for _ in range(NREPS):
            surf  = torch.randn(SURFACE_SHAPE,   device=device)
            upper = torch.randn(UPPER_AIR_SHAPE, device=device)
            gaps_b.append(run_loop_preallocated(model, surf, upper, INFERENCE_STEPS))

        # C: torch.compile
        try:
            compiled = torch.compile(model, mode="reduce-overhead")
            # warm up compile
            for _ in range(5):
                surf  = torch.randn(SURFACE_SHAPE,   device=device)
                upper = torch.randn(UPPER_AIR_SHAPE, device=device)
                run_loop(compiled, surf, upper, INFERENCE_STEPS, record_events=False)
            gaps_c = []
            for _ in range(NREPS):
                surf  = torch.randn(SURFACE_SHAPE,   device=device)
                upper = torch.randn(UPPER_AIR_SHAPE, device=device)
                _, gaps = run_loop(compiled, surf, upper, INFERENCE_STEPS)
                gaps_c.append(gaps)
        except Exception as e:
            gaps_c = None
            print(f"  torch.compile unavailable: {e}")

        # D: CUDA Graph
        try:
            surf  = torch.randn(SURFACE_SHAPE,   device=device)
            upper = torch.randn(UPPER_AIR_SHAPE, device=device)
            gaps_d = []
            for _ in range(NREPS):
                surf  = torch.randn(SURFACE_SHAPE,   device=device)
                upper = torch.randn(UPPER_AIR_SHAPE, device=device)
                gaps_d.append(run_cuda_graph := run_loop_cuda_graph(
                    model, surf, upper, INFERENCE_STEPS))
        except Exception as e:
            gaps_d = None
            print(f"  CUDA Graph unavailable: {e}")

    print()
    report("A: list append          [inference.py pattern]", gaps_a)
    report("B: pre-allocated tensor [no list append]",       gaps_b)
    if gaps_c:
        report("C: torch.compile reduce-overhead",           gaps_c)
    if gaps_d:
        report("D: CUDA Graph replay",                       [[g] for g in gaps_d]
               if isinstance(gaps_d[0], float) else gaps_d)

    print(f"\n  Interpretation:")
    if gaps_a and gaps_d:
        flat_a = [g for r in gaps_a for g in r]
        flat_d = gaps_d if isinstance(gaps_d[0], float) else \
                 [g for r in gaps_d for g in r]
        ratio = statistics.median(flat_a) / max(statistics.median(flat_d), 0.001)
        if ratio > 10:
            print(f"  Python dispatch overhead is dominant (A is {ratio:.0f}x worse than D).")
            print(f"  Pre-allocating outputs and/or CUDA Graphs would close most of the gap.")
        elif ratio > 2:
            print(f"  Python overhead is a partial contributor (A is {ratio:.1f}x worse than D).")
            print(f"  Both software and hardware factors are likely in play.")
        else:
            print(f"  Python overhead is NOT the dominant cause (A ≈ D).")
            print(f"  The gaps are structural — hardware topology (NUMA/PCIe) is the culprit.")
    print()


if __name__ == "__main__":
    main()
