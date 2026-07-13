#!/usr/bin/env python3
"""
PCIe host-to-device (H2D) and device-to-host (D2H) bandwidth test for multi-GPU nodes.

Background
----------
When the CPU sends a tensor to a GPU, data travels over the PCIe bus from
CPU DRAM into GPU HBM. The rate of this transfer is the H2D bandwidth. On a
server with multiple GPUs, all cards share the same PCIe root complex (or
separate root complexes depending on the node topology). If they all transfer
simultaneously, they may compete for the same upstream bandwidth — this
competition is PCIe contention.

Additionally, modern servers have multiple CPU sockets (NUMA nodes), each
with their own local DRAM and their own set of GPUs attached to that socket's
PCIe lanes. A GPU transferring data from CPU DRAM that belongs to the *other*
socket has to cross a slow inter-socket link (UPI/QPI) first — this is a NUMA
penalty. It typically shows up as one or two GPUs being significantly slower
than the others in a concurrent test.

What this script measures
--------------------------
1. Sequential bandwidth  — each GPU is tested one at a time, with all others
   idle. This is the best-case rate and is not affected by contention.
   It gives the per-GPU baseline.

2. Concurrent bandwidth  — all GPUs transfer simultaneously, each in its own
   OS process (mirroring torchrun's multi-process setup). This reveals whether
   bandwidth degrades under load, and whether the degradation is symmetric
   (PCIe contention) or asymmetric (NUMA penalty — only some GPUs slow down).

3. Contention delta  — the difference between concurrent and sequential
   bandwidth per GPU, as an absolute and percentage change. A large negative
   delta on GPU0 and GPU3 but not GPU1 and GPU2 (as seen on the DSI H200
   cluster) points to a NUMA topology issue rather than shared PCIe saturation.

Why pinned memory
-----------------
The test allocates memory in pinned (page-locked) CPU memory, which is what
PyTorch's DataLoader does when pin_memory=True. Pinned memory lets the PCIe
DMA engine transfer directly without staging through a bounce buffer, giving
the highest attainable rate.

Tensor sizes
------------
Derived from exp2.yaml (the default training/inference config):
  - 5 surface variables, 5 upper-air variables × 17 pressure levels = 85 channels
  - 2 diagnostic variables, 1 varying boundary variable
  - batch_size = 8 global / 4 GPUs = 2 per GPU
  - timedelta_hours = 24 → inference_steps = 15 (15-day forecast at 24h resolution)

Two groups of shapes are tested:

  H2D_SHAPES — tensors that move from the DataLoader (CPU) onto the GPU each
  inference step. These are the per-step inputs and are relatively small.

  D2H_SHAPES — tensors that move back from the GPU to CPU after all inference
  steps complete. These are the stacked multi-step output tensors and are much
  larger. The largest (stacked upper-air output) is ~178 MB, matching the
  ~165 MB per-transfer figure measured from the DSI nsys profile.

Usage
-----
Run on any GPU node:
    PYTHONPATH=v2.0 python v2.0/test/bandwidth_test.py

To isolate NUMA effects, run twice with numactl and compare:
    numactl --cpunodebind=0 --membind=0 python bandwidth_test.py
    numactl --cpunodebind=1 --membind=1 python bandwidth_test.py

A significant difference between the two runs means GPU processes are
sensitive to which NUMA node their CPU thread runs on.
"""

import time
import statistics
import torch
import torch.multiprocessing as mp
from typing import List

# ---------------------------------------------------------------------------
# Tensor shapes — derived from v2.0/config/exp2.yaml
# ---------------------------------------------------------------------------

# exp2.yaml config values that determine these shapes:
#   surface_variables: 5 variables
#   upper_air_variables: 5 vars × 17 pressure levels = 85 channels
#   diagnostic_variables: 2 variables
#   varying_boundary_variables: 1 variable
#   batch_size: 8 global / 4 GPUs = 2 per GPU (BATCH_PER_GPU)
#   timedelta_hours: 24  →  inference_steps = (24*15)//24 = 15

BATCH_PER_GPU   = 2    # 8 global / 4 GPUs
SURF_CHANNELS   = 5    # surface_variables count
UPPER_CHANNELS  = 85   # 5 upper_air_variables × 17 pressure levels
DIAG_CHANNELS   = 2    # diagnostic_variables count
VBND_CHANNELS   = 1    # varying_boundary_variables count
INFERENCE_STEPS = 15   # (24 × 15 days) // timedelta_hours=24
H, W            = 128, 256

# H2D shapes: per-step DataLoader output transferred to GPU each inference step.
# These are relatively small — the DataLoader moves one batch of inputs at a time.
H2D_SHAPES = {
    "surf_input":       (BATCH_PER_GPU, SURF_CHANNELS,  H, W),            #  1.3 MB
    "upper_air_input":  (BATCH_PER_GPU, UPPER_CHANNELS, H, W),            # 22.3 MB
    "varying_boundary": (BATCH_PER_GPU, INFERENCE_STEPS, VBND_CHANNELS, H, W),  #  3.9 MB
}

# D2H shapes: stacked multi-step outputs moved back to CPU after all inference
# steps complete. Much larger because all time steps are stacked together before
# the transfer. The upper-air stacked output at ~178 MB matches the ~165 MB
# per-transfer figure measured from the DSI H200 nsys profile.
T = INFERENCE_STEPS + 1   # initial state + 15 forecast steps = 16 time points
D2H_SHAPES = {
    "surf_output_stacked":  (BATCH_PER_GPU * T, SURF_CHANNELS,  H, W),  #  10.5 MB
    "upper_output_stacked": (BATCH_PER_GPU * T, UPPER_CHANNELS, H, W),  # 177.9 MB ← large
    "diag_output_stacked":  (BATCH_PER_GPU * T, DIAG_CHANNELS,  H, W),  #   4.2 MB
}

NREPS  = 50   # number of timed transfers per shape after warmup
WARMUP = 10   # transfers discarded at the start to let CUDA caches settle


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def _mb(shape: tuple) -> float:
    """Return the size in MB of a float32 tensor with the given shape."""
    n = 1
    for d in shape:
        n *= d
    return n * 4 / 1e6


def _measure(device_id: int, shape: tuple, nreps: int, warmup: int,
             direction: str = "h2d") -> List[float]:
    """
    Measure PCIe transfer bandwidth for a single GPU in one direction.

    direction='h2d': pinned CPU tensor → GPU  (DataLoader input path)
    direction='d2h': GPU tensor → pinned CPU  (inference output path)

    Uses synchronous blocking copies so wall-clock time equals actual transfer
    time, not just the time to enqueue the DMA command.
    """
    device  = torch.device(f"cuda:{device_id}")
    nbytes  = 1
    for d in shape: nbytes *= d
    nbytes *= 4  # float32

    if direction == "h2d":
        src = torch.randn(shape, dtype=torch.float32, pin_memory=True)
        dst = torch.empty(shape, dtype=torch.float32, device=device)
        def transfer(): dst.copy_(src, non_blocking=False)
    else:
        src = torch.randn(shape, dtype=torch.float32, device=device)
        dst = torch.empty(shape, dtype=torch.float32, pin_memory=True)
        def transfer(): dst.copy_(src, non_blocking=False)

    results = []
    for i in range(nreps + warmup):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        transfer()
        torch.cuda.synchronize(device)
        if i >= warmup:
            results.append(nbytes / (time.perf_counter() - t0) / 1e9)
    return results


# ---------------------------------------------------------------------------
# Sequential test
# ---------------------------------------------------------------------------

def sequential_test(num_gpus: int, shapes: dict, direction: str) -> dict:
    """
    Test each GPU independently with all others idle — no contention baseline.
    Returns {tensor_name: {gpu_id: [bandwidth_samples_GB/s]}}
    """
    results = {}
    for name, shape in shapes.items():
        results[name] = {}
        for gpu in range(num_gpus):
            results[name][gpu] = _measure(gpu, shape, NREPS, WARMUP, direction)
    return results


# ---------------------------------------------------------------------------
# Concurrent test
# ---------------------------------------------------------------------------

def _worker(rank: int, shape: tuple, nreps: int, warmup: int,
            direction: str, result_queue) -> None:
    """
    Child process for concurrent test — one process per GPU, each allocating
    its own pinned CPU buffer. Mirrors the torchrun multi-process setup.
    """
    samples = _measure(rank, shape, nreps, warmup, direction)
    result_queue.put((rank, samples))


def concurrent_test(num_gpus: int, shapes: dict, direction: str) -> dict:
    """
    All GPUs transfer simultaneously in separate processes.

    Reveals PCIe contention (symmetric drop across all GPUs) and NUMA penalty
    (asymmetric drop on GPUs whose CPU process is on the far socket).
    Returns {tensor_name: {gpu_id: [bandwidth_samples_GB/s]}}
    """
    results = {}
    for name, shape in shapes.items():
        q = mp.Queue()
        procs = [
            mp.Process(target=_worker,
                       args=(gpu, shape, NREPS, WARMUP, direction, q))
            for gpu in range(num_gpus)
        ]
        for p in procs: p.start()
        for p in procs: p.join()
        gpu_results = {}
        while not q.empty():
            gpu_id, samples = q.get()
            gpu_results[gpu_id] = samples
        results[name] = gpu_results
    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(label: str, results: dict, shapes: dict, num_gpus: int) -> None:
    """Print a per-GPU bandwidth table with one row per tensor type."""
    print(f"\n{'='*75}")
    print(f"  {label}")
    print(f"{'='*75}")
    header = f"  {'tensor':<26}  {'size_MB':>7}  " + \
             "  ".join(f"GPU{g} GB/s" for g in range(num_gpus))
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name, shape in shapes.items():
        mb = _mb(shape)
        row = f"  {name:<26}  {mb:>7.1f}  "
        for g in range(num_gpus):
            samples = results.get(name, {}).get(g, [])
            row += f"  {statistics.median(samples):>8.2f}  " if samples else f"  {'N/A':>8}  "
        print(row)

    # Median across all tensors per GPU — single summary number
    print()
    row = f"  {'AGGREGATE (median)':<26}  {'':>7}  "
    for g in range(num_gpus):
        all_s = [s for name in shapes for s in results.get(name, {}).get(g, [])]
        row += f"  {statistics.median(all_s):>8.2f}  " if all_s else f"  {'N/A':>8}  "
    print(row)


def contention_delta(seq: dict, con: dict, shapes: dict, num_gpus: int) -> None:
    """
    Per-GPU bandwidth change from sequential to concurrent.
    Symmetric drop → shared bus contention. Asymmetric drop → NUMA penalty.
    """
    print(f"\n  Contention delta (concurrent − sequential), median across all tensors:")
    for g in range(num_gpus):
        seq_bw = [s for name in shapes for s in seq.get(name, {}).get(g, [])]
        con_bw = [s for name in shapes for s in con.get(name, {}).get(g, [])]
        if seq_bw and con_bw:
            delta = statistics.median(con_bw) - statistics.median(seq_bw)
            pct   = delta / statistics.median(seq_bw) * 100
            sign  = "+" if delta >= 0 else ""
            print(f"    GPU{g}: {sign}{delta:.2f} GB/s  ({sign}{pct:.1f}%)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        print("No CUDA device — this test requires a GPU.")
        return

    num_gpus = torch.cuda.device_count()
    props    = [torch.cuda.get_device_properties(i) for i in range(num_gpus)]

    print(f"\nNode GPU summary:")
    for i, p in enumerate(props):
        print(f"  GPU{i}: {p.name}  ({p.total_memory/1e9:.1f} GB)")
    print(f"\nConfig: batch_per_gpu={BATCH_PER_GPU}  upper_channels={UPPER_CHANNELS}"
          f"  inference_steps={INFERENCE_STEPS}")
    print(f"Warmup={WARMUP}  Reps={NREPS}")

    print(f"\nH2D shapes (DataLoader inputs — CPU → GPU each inference step):")
    for name, shape in H2D_SHAPES.items():
        print(f"  {name:<26} {_mb(shape):6.1f} MB  {shape}")

    print(f"\nD2H shapes (stacked outputs — GPU → CPU after all steps complete):")
    for name, shape in D2H_SHAPES.items():
        print(f"  {name:<26} {_mb(shape):6.1f} MB  {shape}")

    # --- H2D (DataLoader → GPU) ---
    print("\n\nRunning H2D sequential test (one GPU at a time)...")
    h2d_seq = sequential_test(num_gpus, H2D_SHAPES, "h2d")
    print_results("H2D Sequential — no contention", h2d_seq, H2D_SHAPES, num_gpus)

    print("\nRunning H2D concurrent test (all GPUs simultaneously)...")
    h2d_con = concurrent_test(num_gpus, H2D_SHAPES, "h2d")
    print_results("H2D Concurrent — under load", h2d_con, H2D_SHAPES, num_gpus)
    contention_delta(h2d_seq, h2d_con, H2D_SHAPES, num_gpus)

    # --- D2H (GPU → CPU, stacked inference outputs) ---
    print("\n\nRunning D2H sequential test (one GPU at a time)...")
    d2h_seq = sequential_test(num_gpus, D2H_SHAPES, "d2h")
    print_results("D2H Sequential — no contention", d2h_seq, D2H_SHAPES, num_gpus)

    print("\nRunning D2H concurrent test (all GPUs simultaneously)...")
    d2h_con = concurrent_test(num_gpus, D2H_SHAPES, "d2h")
    print_results("D2H Concurrent — under load", d2h_con, D2H_SHAPES, num_gpus)
    contention_delta(d2h_seq, d2h_con, D2H_SHAPES, num_gpus)

    print()


if __name__ == "__main__":
    # spawn avoids CUDA context inheritance issues when forking.
    # Each child process initialises its own CUDA context cleanly.
    mp.set_start_method("spawn", force=True)
    main()
