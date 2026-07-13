#!/usr/bin/env python3
"""
Smoke test to isolate the D2H transfer patterns that differ between
inference.py and inference_optimized.py, without needing the model or a checkpoint.

Tests three patterns:
  A. times.item() x4 per sample  — inference.py style
  B. times.cpu().numpy()          — inference_optimized.py style (one bulk transfer)
  C. Synchronous .cpu().numpy() on large output tensors  — inference.py style
  D. Async pinned copy + single sync on large tensors    — inference_optimized.py style

Run on any GPU node:
    PYTHONPATH=v2.0 python v2.0/test/d2h_pattern_smoke.py

The test runs NSTEPS iterations of each pattern, reporting median time per step.
If hypothesis is correct, pattern A will be meaningfully slower than B, and
C meaningfully slower than D, because each .item() / .cpu() forces a GPU flush.
"""

import time
import statistics
import torch

NSTEPS   = 200    # number of measured iterations
WARMUP   = 20     # steps to skip before timing starts
BATCH    = 1      # match the inference batch size used in the DSI runs

# Output tensor shapes from a typical S2S inference run at the default resolution.
# Surface: (batch, n_surface_vars, lat, lon)  — approx 50 MB for batch=1
# Upper:   (batch, n_upper_vars, levels, lat, lon) — approx 250 MB for batch=1
# Diagnostic: same shape as surface roughly
SURF_SHAPE  = (BATCH, 16,  128, 256)
UPPER_SHAPE = (BATCH, 104, 128, 256)
DIAG_SHAPE  = (BATCH, 4,   128, 256)


def _median_ms(times_sec):
    return statistics.median(times_sec) * 1e3


def _queue_pending_work(device):
    """
    Simulate the async .to(device) transfers that are in-flight in the real
    inference loop when .item() is called. Queues a non-blocking H2D copy of a
    surface-sized tensor so the GPU stream has pending work to flush.
    """
    src = torch.randn(SURF_SHAPE, dtype=torch.float32, pin_memory=True)
    _ = src.to(device, non_blocking=True)


def bench_times_item(device, nsteps, warmup):
    """
    Pattern A — inference.py: 4 × .item() calls per sample in the batch.
    Each .item() is a cudaStreamSynchronize + scalar copy.

    Idle variant: GPU stream is empty before .item() — lower-bound estimate.
    Realistic variant: a pending async H2D transfer is in-flight before .item(),
    matching inference.py where .item() is called right after x.to(device).
    The realistic stall = scalar D2H round-trip + flushing the pending transfer.
    """
    times_gpu = torch.randint(1990, 2024, (BATCH, 4), device=device)
    idle, realistic = [], []

    for step in range(nsteps + warmup):
        # --- idle GPU ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(BATCH):
            _ = (times_gpu[i, 0].item(),
                 times_gpu[i, 1].item(),
                 times_gpu[i, 2].item(),
                 times_gpu[i, 3].item())
        torch.cuda.synchronize()
        if step >= warmup:
            idle.append(time.perf_counter() - t0)

        # --- pending work in stream before .item() ---
        torch.cuda.synchronize()
        _queue_pending_work(device)          # async H2D still in flight
        t0 = time.perf_counter()
        for i in range(BATCH):
            _ = (times_gpu[i, 0].item(),
                 times_gpu[i, 1].item(),
                 times_gpu[i, 2].item(),
                 times_gpu[i, 3].item())
        torch.cuda.synchronize()
        if step >= warmup:
            realistic.append(time.perf_counter() - t0)

    return idle, realistic


def bench_times_numpy(device, nsteps, warmup):
    """
    Pattern B — inference_optimized.py: one bulk .cpu().numpy() for the whole batch.
    One D2H transfer, no per-scalar GPU stalls.
    """
    times_gpu = torch.randint(1990, 2024, (BATCH, 4), device=device)
    results = []
    for step in range(nsteps + warmup):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        times_np = times_gpu.cpu().numpy().astype(int)
        for idx in range(times_np.shape[0]):
            _ = (times_np[idx, 0], times_np[idx, 1],
                 times_np[idx, 2], times_np[idx, 3])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if step >= warmup:
            results.append(elapsed)
    return results


def bench_sync_d2h(device, nsteps, warmup):
    """
    Pattern C — inference.py: three synchronous .cpu().numpy() calls on large tensors.
    Each blocks until the GPU has fully transferred that tensor before starting the next.
    """
    surf  = torch.randn(SURF_SHAPE,  device=device, dtype=torch.float32)
    upper = torch.randn(UPPER_SHAPE, device=device, dtype=torch.float32)
    diag  = torch.randn(DIAG_SHAPE,  device=device, dtype=torch.float32)
    results = []
    for step in range(nsteps + warmup):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _s = surf.cpu().numpy()
        _u = upper.cpu().numpy()
        _d = diag.cpu().numpy()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if step >= warmup:
            results.append(elapsed)
    return results


def bench_async_d2h(device, nsteps, warmup):
    """
    Pattern D — inference_optimized.py: persistent pinned buffers, three async copies,
    one synchronize covering all three.
    """
    surf  = torch.randn(SURF_SHAPE,  device=device, dtype=torch.float32)
    upper = torch.randn(UPPER_SHAPE, device=device, dtype=torch.float32)
    diag  = torch.randn(DIAG_SHAPE,  device=device, dtype=torch.float32)

    # Allocate pinned buffers once (mirrors the lazy-alloc in inference_optimized.py)
    buf_s = torch.empty(SURF_SHAPE,  dtype=torch.float32, pin_memory=True)
    buf_u = torch.empty(UPPER_SHAPE, dtype=torch.float32, pin_memory=True)
    buf_d = torch.empty(DIAG_SHAPE,  dtype=torch.float32, pin_memory=True)

    results = []
    for step in range(nsteps + warmup):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        buf_s.copy_(surf,  non_blocking=True)
        buf_u.copy_(upper, non_blocking=True)
        buf_d.copy_(diag,  non_blocking=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if step >= warmup:
            results.append(elapsed)
    return results


def fmt(label, results):
    med  = _median_ms(results)
    mn   = min(results) * 1e3
    mx   = max(results) * 1e3
    p95  = sorted(results)[int(len(results) * 0.95)] * 1e3
    print(f"  {label:<45}  median={med:6.2f}ms  min={mn:5.2f}ms  p95={p95:6.2f}ms  max={mx:6.2f}ms")


def main():
    if not torch.cuda.is_available():
        print("No CUDA device found — this test requires a GPU.")
        return

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"\nDevice: {props.name}  ({props.total_memory / 1e9:.1f} GB)")
    print(f"Tensor shapes — surf={SURF_SHAPE}  upper={UPPER_SHAPE}  diag={DIAG_SHAPE}")
    print(f"Batch size: {BATCH}   Warmup: {WARMUP}   Measured steps: {NSTEPS}\n")

    print("--- times extraction (per inference step) ---")
    item_idle, item_realistic = bench_times_item(device, NSTEPS, WARMUP)
    fmt("A (idle GPU):     .item() x4        [inference.py]",      item_idle)
    fmt("A (pending work): .item() x4        [inference.py]",      item_realistic)
    fmt("B:                bulk .cpu().numpy()[inference_optimized]", bench_times_numpy(device, NSTEPS, WARMUP))
    print("  ^ 'pending work' = async H2D in-flight before .item(), matching real inference loop")

    print("\n--- output D2H transfer (per inference step, all 3 tensors) ---")
    print(f"  Data volume: {(torch.tensor(SURF_SHAPE).prod() + torch.tensor(UPPER_SHAPE).prod() + torch.tensor(DIAG_SHAPE).prod()).item() * 4 / 1e6:.0f} MB")
    fmt("C: 3x synchronous .cpu().numpy()  [inference.py]",       bench_sync_d2h(device,  NSTEPS, WARMUP))
    fmt("D: async pinned copy + 1 sync     [inference_optimized]", bench_async_d2h(device, NSTEPS, WARMUP))

    print()


if __name__ == "__main__":
    main()
