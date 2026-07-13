# Benchmark Methodology

This document explains, in plain terms, how every number in `bench_results.csv` is produced by the instrumentation in `train.py`.

---

## Why we can't just use `time.time()`

A GPU works asynchronously. When Python calls `model.forward(...)`, the CPU queues the GPU kernels and returns immediately — the GPU is still running in the background. If you measure time with `time.time()` around the forward call, you capture only how long it took the CPU to *submit* the work, not how long the GPU took to *finish* it. On a fast CPU with a busy GPU, the submission takes microseconds while the actual computation takes hundreds of milliseconds. The original `tr_time` logs in the training loop had exactly this bug: they recorded queue time, not execution time.

The fix is `torch.cuda.synchronize()`. This call blocks the CPU until every outstanding GPU kernel on the current device has finished. By placing a `synchronize()` immediately before starting the clock and immediately before stopping it, we guarantee we're measuring real wall-clock GPU execution.

---

## The measurement window

For each training step, the code places three synchronization points:

```
synchronize() → t0
   [data preparation and host-to-device transfer]
synchronize() → t1
   [forward pass + loss + backward + optimizer step]
synchronize() → t2
```

- **t0**: the GPU is idle, nothing queued. Starting the clock from a clean slate.
- **t1**: data preparation is done. Everything has been moved to the GPU.
- **t2**: the optimizer step is done. No outstanding kernels.

The differences between these timestamps give us the raw material for all the timing metrics.

---

## Warmup period

The first `S2S_BENCH_WARMUP` steps (default 20) are discarded. This exists because:

- The first few steps pay one-time costs: CUDA kernel JIT compilation, cuDNN algorithm selection (autotuning), and NCCL communicator warm-up. These inflate step time significantly and are not representative of steady-state training.
- If `torch.compile` is enabled, the first steps also include Python-level graph capture and Triton kernel compilation, which can take tens of seconds.

Timing arrays are only populated starting at step `BENCH_WARMUP + 1`. The `bench_loop_t0` clock (used for the sanity check) also starts at the first measured step.

---

## Metric definitions

### `step_med` — median step time (seconds)

The middle value of all `(t2 - t0)` samples collected across the 80 measured steps. Median is used instead of mean because it is robust to outliers: an occasional long step from an OS scheduler interruption or a filesystem flush does not shift the median, whereas it would inflate the mean. This is the primary number to track across optimization runs.

### `step_p90` — 90th-percentile step time (seconds)

The step time that 90% of steps were faster than. It characterizes the tail of the distribution — how bad the worst 10% of steps are. A large gap between `step_med` and `step_p90` means the timing is noisy or there is periodic overhead (e.g., gradient accumulation, logging, checkpoint I/O happening every N steps).

For the baseline: `step_p90 = 0.643 s`, `step_med = 0.639 s`, a gap of only 4 ms — the distribution is very tight.

### `step_mean` and `step_std` — mean and standard deviation (seconds)

The arithmetic mean and population standard deviation of the same `(t2 - t0)` samples. Together they let you compute the coefficient of variation (`step_std / step_mean`) — a normalized measure of jitter. For the baseline: CV = 0.005 / 0.639 = 0.74%, indicating a very stable run.

### `cpu_prep_med` — median data-preparation time (seconds)

The middle value of all `(t1 - t0)` samples. This covers everything from when the CPU receives a new batch from the DataLoader to when every tensor has been transferred to GPU memory and is ready for the forward pass. Concretely: unpacking the HDF5 batch, applying normalisation with the mean/std arrays, and the `.to(device, non_blocking=False)` host-to-device copy.

For the baseline: `cpu_prep_med = 0.0027 s` — 2.7 milliseconds, or 0.43% of total step time.

### `compute_med` — median compute time (seconds)

The middle value of all `(t2 - t1)` samples. This covers the forward pass through PanguModel_Plasim (including the VAE reparameterisation and ensemble generation), the CRPS + KL loss computation, `loss.backward()`, and the Adam optimizer step. Everything that runs on the GPU.

For the baseline: `compute_med = 0.636 s`.

### `cpu_prep_frac` — fraction of step time spent on data preparation

Computed as `cpu_prep_med / step_med`. A simple ratio. Note the name: this measures the CPU-side preparation window, not true GPU idle time, because the host-to-device transfer may overlap with GPU execution from the previous step depending on CUDA stream scheduling. The name `cpu_prep_frac` (rather than `data_frac`) is intentional — it is honest about what is actually measured.

For the baseline: `0.0043` (0.43%). The training is overwhelmingly compute-bound.

### `samples_per_s` — global training throughput

Computed as `(batch_size_per_gpu × n_gpus) / step_med`. This is the end-to-end throughput: how many ERA5 timesteps the model processes per second of wall time, across all GPUs. It is the primary figure to optimise — doubling this number means the same model trains in half the time.

For the baseline: `(1 × 4) / 0.639 = 6.26 samples/s`.

Note that `batch_size_per_gpu` is taken from `params.batch_size` as configured in the YAML. Each rank receives that many samples from its `DistributedSampler` shard per step; the effective global batch size is `batch_size × world_size`.

### `peak_mem_gb_max_rank` — worst-case peak GPU memory across all ranks (GB)

Each rank measures `torch.cuda.max_memory_allocated()` — the highest point the PyTorch allocator reached during the entire run (in bytes). These per-rank values are reduced across all 4 GPUs using a max all-reduce, so the reported number is the worst-case rank. This is what determines whether a larger batch size will OOM.

For the baseline: `34.96 GB` on an H100 NVL with 94 GB HBM — 37% utilization.

### `scaler_skips` — number of AMP GradScaler-skipped steps

When using FP16 automatic mixed precision, PyTorch's `GradScaler` detects inf or NaN values in the scaled gradients after `backward()`. If any are found, it discards the optimizer step entirely (the weights are not updated), halves the loss scale, and carries on. These skipped steps are approximately twice as fast as normal steps because the optimizer kernel never runs — if they were included in the timing arrays, they would artificially deflate `step_med`.

The code detects a skip by comparing the GradScaler's loss scale before and after `scaler.update()`. If the scale decreased, the step was skipped and it is counted in `scaler_skips` but excluded from all timing arrays.

For the baseline: `0` — no skips. This is the key evidence that FP16 is numerically stable for this model and dataset, and that switching to BF16 is safe.

When BF16 is enabled (`S2S_AMP_DTYPE=bf16`), the `GradScaler` is constructed with `enabled=False`. It becomes a no-op: `scaler.scale(loss)` returns `loss` unchanged, `scaler.step(optimizer)` calls `optimizer.step()` directly, and `scaler_skips` will always be 0. There is no loss scaling and no skip mechanism because BF16's dynamic range (same exponent width as FP32) makes overflow essentially impossible.

### `n_steps_counted`

The number of non-skipped steps that contributed to the timing arrays. Under normal conditions this equals `S2S_BENCH_STEPS`. It can be lower if the dataset exhausted before the requested number of steps, which is caught by the `min(BENCH_STEPS, batches_per_loader - BENCH_WARMUP)` cap computed at runtime.

---

## Sanity check

Before writing the CSV row, the code asserts that the sum `step_med × n_steps_counted` agrees with the actual elapsed wall time (from `bench_loop_t0` to the end of the last measured step) within ±5%. If it doesn't, something is wrong with the timer — for example, a long pause between steps (garbage collection, OS scheduling, filesystem flush) that inflated `elapsed` without showing up in the per-step records. In that case the run aborts with exit code 3 and no row is written, so the CSV is never silently poisoned.

---

## How to read the CSV across multiple runs

Each row is one SLURM job. The primary comparison axis is `step_med` and `samples_per_s`. When comparing two rows:

- **Same `yaml_sha256_16`** means the config was identical — any difference in timing is attributable to the code change.
- **Different `ddp_find_unused`, `amp_dtype`, or `TORCH_COMPILE_MODE`** are the levers being pulled; the CSV records the state of each.
- **`scaler_skips > 0`** in a row means those steps were excluded from timing; the row is still valid but note the exclusion in any write-up.
- **Three runs with the same settings**: take the median of the three `step_med` values as the reported number. The `step_std` within each run tells you about within-job jitter; the spread across three jobs tells you about between-job variance (filesystem cold start, NCCL ring init, node-to-node variation).

---

## Nsight Systems profiling (midway_bench_nsys.sh)

`midway_bench_nsys.sh` runs the same benchmark with two additions:

1. `S2S_NVTX=1` activates NVTX range labels inside the step loop: `step_N` wraps the whole step; `data_prep`, `forward_loss`, `backward`, `optimizer` label sub-phases. These appear as coloured rows in the Nsight timeline.

2. The code calls `cudaProfilerStart()` at the first *measured* step and `cudaProfilerStop()` at the start of `_bench_finalize`. The nsys flag `--capture-range=cudaProfilerApi` means the profiler records nothing outside this window — warmup, NCCL init, and post-bench teardown are all excluded. The `.nsys-rep` file therefore contains only the 80 measured steps.

The script automatically exports a `.sqlite` file alongside the `.nsys-rep` using `nsys export --type=sqlite`. That SQLite file can be transferred here and queried directly.

### Analysing the SQLite export

Use `parse_nsys.py` (no extra installs — stdlib `sqlite3` only):

```bash
# On Midway after the job finishes, or locally after scp:
python v2.0/HPC_scripts/parse_nsys.py nsys_bench_<run_num>.sqlite
```

The script prints five sections: available tables, NVTX range breakdown, top-20 CUDA kernels by GPU time, NCCL all-reduce totals, and H2D memcpy totals.

### What to look for

- **`forward_loss` >> `backward`**: normal for a model with ensemble generation. If `backward` is much larger, gradient checkpointing is re-running a lot of work.
- **NCCL all-reduce > 10% of step time**: DDP communication is a bottleneck; consider `static_graph=True` or gradient compression.
- **`data_prep` avg_us > 5000 (5 ms)**: data loading is creeping up; check H2D copy sizes and whether `non_blocking=True` is set.
- **Top kernel is not a matmul or attention kernel**: something unexpected (e.g., a cast, a copy, or a reduction) is dominating — investigate that kernel.
- **High variance in `step_N` total duration**: OS jitter or NCCL retry; if std > 5% of mean, the node may be noisy.
