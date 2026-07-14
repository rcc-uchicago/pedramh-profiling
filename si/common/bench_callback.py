"""
Lightning callback for wall-clock throughput benchmarking.

Measurement window per step
───────────────────────────
    on_train_batch_start  →  torch.cuda.synchronize()  →  t0
        [batch already on GPU — Lightning transferred it before this hook]
        [training_step: preprocess + forward + loss]
        [Lightning: backward()]
        [Lightning: optimizer.step() + zero_grad()]
    on_train_batch_end    →  torch.cuda.synchronize()  →  t2

    step_time = t2 - t0  (GPU-accurate wall time for the full step)

No separate cpu_prep measurement: Lightning's DataLoader workers run
asynchronously, so the H2D transfer overlaps with the previous step's GPU
work and is already complete by the time on_train_batch_start fires.  The
step_time therefore represents pure compute (forward + backward + optimizer).
If the GPU has to stall waiting for data, that idle time is captured in
step_time too — a large step_time with a small model is the signature of a
data-loading bottleneck.

Environment knobs (set before launching)
──────────────────────────────────────────
    SI_BENCH_WARMUP   steps to discard before measuring  (default 20)
    SI_BENCH_STEPS    steps to measure                   (default 80)
    SI_BENCH_CSV      output CSV path                    (default bench_results.csv)
    SI_NVTX=1         emit NVTX step ranges + cudaProfilerStart/Stop for nsys
                        and (via this callback) backward / optimizer ranges
"""

import csv
import hashlib
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch
import lightning as L

BENCH_WARMUP = int(os.environ.get("SI_BENCH_WARMUP", "20"))
BENCH_STEPS  = int(os.environ.get("SI_BENCH_STEPS",  "80"))
BENCH_CSV    = os.environ.get("SI_BENCH_CSV", "bench_results.csv")
NVTX         = os.environ.get("SI_NVTX") == "1"


class BenchCallback(L.Callback):
    """Throughput benchmark callback; add as the sole callback in bench.py."""

    def __init__(self, *, n_gpus: int, batch_per_gpu: int,
                 config_path: str | None = None, run_num: str = "bench"):
        super().__init__()
        self.n_gpus        = n_gpus
        self.batch_per_gpu = batch_per_gpu
        self.config_path   = config_path
        self.run_num       = run_num

        self._step_times: list[float] = []
        self._iters       = 0
        self._t0: float | None = None
        self._bench_loop_t0: float | None = None
        self._done        = False

    # ------------------------------------------------------------------

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if NVTX:
            import torch.cuda.nvtx as nvtx
            nvtx.range_push(f"step_{self._iters}")
        torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    # --- NVTX backward / optimizer ranges (only when SI_NVTX=1) ----------
    # These let the next nsys profile separate "backward compute" from
    # "NCCL gradient-sync wait" and from "optimizer step", which the current
    # train_module.py ranges (preprocess / forward_loss) do not.
    def on_before_backward(self, trainer, pl_module, loss):
        if NVTX:
            import torch.cuda.nvtx as nvtx
            nvtx.range_push("backward")

    def on_after_backward(self, trainer, pl_module):
        if NVTX:
            import torch.cuda.nvtx as nvtx
            nvtx.range_pop()  # backward

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if NVTX:
            import torch.cuda.nvtx as nvtx
            nvtx.range_push("optimizer")
    # The 'optimizer' range is closed in on_train_batch_end below (Lightning
    # has no on_after_optimizer_step hook).  This means the optimizer range
    # also includes any post-step callback work, but that is negligible here.

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if NVTX:
            import torch.cuda.nvtx as nvtx
            nvtx.range_pop()  # optimizer (opened in on_before_optimizer_step)
            nvtx.range_pop()  # step_N (opened in on_train_batch_start)

        self._iters += 1
        step_time = t2 - self._t0  # type: ignore[operator]

        if trainer.global_rank == 0 and self._iters == 1:
            print(f"[BenchCallback] warmup={BENCH_WARMUP}  steps={BENCH_STEPS}  csv={BENCH_CSV}",
                  flush=True)

        if self._iters <= BENCH_WARMUP:
            if trainer.global_rank == 0:
                print(f"[BenchCallback] warmup {self._iters}/{BENCH_WARMUP}", flush=True)
            return

        # Start nsys capture window at the first measured step.
        if self._bench_loop_t0 is None:
            if NVTX:
                torch.cuda.cudart().cudaProfilerStart()
            self._bench_loop_t0 = t2

        self._step_times.append(step_time)
        if trainer.global_rank == 0:
            print(f"[BenchCallback] measured {len(self._step_times)}/{BENCH_STEPS}  "
                  f"step={step_time:.3f}s", flush=True)

        if len(self._step_times) >= BENCH_STEPS and not self._done:
            self._done = True
            self._finalize(trainer, pl_module)
            trainer.should_stop = True

    # ------------------------------------------------------------------

    def _finalize(self, trainer, pl_module):
        print(f"[BenchCallback] _finalize called on rank {trainer.global_rank}", flush=True)
        if NVTX:
            torch.cuda.cudart().cudaProfilerStop()

        steps   = self._step_times
        n       = len(steps)
        elapsed = time.perf_counter() - self._bench_loop_t0  # type: ignore[operator]

        step_sorted   = sorted(steps)
        step_med      = statistics.median(steps)
        step_p90      = step_sorted[int(0.9 * n)]
        step_mean     = statistics.mean(steps)
        step_std      = statistics.stdev(steps) if n > 1 else 0.0
        samples_per_s = (self.batch_per_gpu * self.n_gpus) / step_med

        # Fraction of wall-clock time the GPU spent idle between steps
        # (typically waiting on the dataloader). step_med×n is compute time;
        # elapsed is total wall time over the measurement window.
        compute_total = step_med * n
        data_idle_frac = max(0.0, (elapsed - compute_total)) / max(elapsed, 1e-9)
        # Effective throughput including idle time (what production actually sees).
        samples_per_s_wall = (self.batch_per_gpu * self.n_gpus * n) / max(elapsed, 1e-9)

        # If the GPU is starved more than half the time, the run is data-bound;
        # still write the row but make the bottleneck visible in the log.
        if data_idle_frac > 0.10:
            print(
                f"[BenchCallback] NOTE data-bound run "
                f"(elapsed={elapsed:.3f}s compute={compute_total:.3f}s "
                f"data_idle_frac={data_idle_frac:.2f}). "
                "step_med reflects compute only; samples_per_s_wall reflects effective throughput."
            )

        # Peak GPU memory — max across all ranks via all-reduce.
        peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
        if trainer.world_size > 1:
            peak_t = torch.tensor(peak_gb, device=pl_module.device)
            torch.distributed.all_reduce(peak_t, op=torch.distributed.ReduceOp.MAX)
            peak_gb = peak_t.item()

        if trainer.global_rank != 0:
            return  # only rank-0 writes

        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
            ).decode().strip()[:12]
        except Exception:
            git_sha = "unknown"

        config_sha = "unknown"
        if self.config_path:
            try:
                with open(self.config_path, "rb") as f:
                    config_sha = hashlib.sha256(f.read()).hexdigest()[:16]
            except Exception:
                pass

        row = {
            "timestamp":             time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_sha":               git_sha,
            "config_sha16":          config_sha,
            "run_num":               self.run_num,
            "n_gpus":                self.n_gpus,
            "batch_per_gpu":         self.batch_per_gpu,
            "precision":             str(trainer.precision),
            "step_med":              f"{step_med:.6f}",
            "step_p90":              f"{step_p90:.6f}",
            "step_mean":             f"{step_mean:.6f}",
            "step_std":              f"{step_std:.6f}",
            "samples_per_s":         f"{samples_per_s:.3f}",
            "samples_per_s_wall":    f"{samples_per_s_wall:.3f}",
            "data_idle_frac":        f"{data_idle_frac:.3f}",
            "peak_mem_gb_max_rank":  f"{peak_gb:.3f}",
            "n_steps_counted":       n,
        }

        csv_path = Path(BENCH_CSV)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with open(csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        print("\n" + "=" * 62)
        print(f"  BENCH RESULT  run_num={self.run_num}")
        print("=" * 62)
        for k, v in row.items():
            print(f"  {k:<26} {v}")
        print("=" * 62)
        print(f"[BenchCallback] Appended to {csv_path}\n")
