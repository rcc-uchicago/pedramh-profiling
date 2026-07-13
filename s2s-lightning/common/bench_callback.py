"""Lightning callback for wall-clock throughput benchmarking (S2S port).

This mirrors the structure of the SNFO template at
``$SNFO_DIR/common/bench_callback.py`` but preserves S2S's **S2S_BENCH**
framework: it reads the ``S2S_*`` environment knobs (not ``SNFO_*``) and writes
the CSV columns recognizable from ``v2.0/train.py::Trainer._bench_finalize``.
It moves the in-loop synchronisation / NVTX / CSV instrumentation that the
manual S2S training loop owned onto Lightning hooks, so the
:class:`modules.train_module.TrainModule` ``training_step`` stays clean (its
``forward_loss`` / ``data_prep`` NVTX ranges remain inside the step; the
``step_N`` / ``backward`` / ``optimizer`` ranges and the measured-window
synchronisation live here, exactly where the source's ``train_one_epoch``
emitted them).

Measurement window per step
    on_train_batch_start  ->  torch.cuda.synchronize()  ->  t0
        [batch already on GPU -- Lightning transferred it before this hook]
        [training_step: data_prep + forward + loss]
        [Lightning: backward()]
        [Lightning: optimizer.step() + zero_grad()]
    on_train_batch_end    ->  torch.cuda.synchronize()  ->  t2
    step_time = t2 - t0  (GPU-accurate wall time for the full step)

Environment knobs (set before launching; same names as ``v2.0/train.py``)
    S2S_BENCH_WARMUP   steps to discard before measuring  (default 20)
    S2S_BENCH_STEPS    steps to measure                   (default 80)
    S2S_BENCH_CSV      output CSV path                    (default bench_results.csv)
    S2S_NVTX=1         emit NVTX step ranges + cudaProfilerStart/Stop for nsys,
                       plus the backward / optimizer ranges added by this callback

See Also:
    modules.train_module.TrainModule.training_step: Emits the in-step
        ``data_prep`` / ``forward_loss`` NVTX ranges this callback brackets.
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

BENCH_WARMUP = int(os.environ.get("S2S_BENCH_WARMUP", "20"))
BENCH_STEPS = int(os.environ.get("S2S_BENCH_STEPS", "80"))
BENCH_CSV = os.environ.get("S2S_BENCH_CSV", "bench_results.csv")
NVTX = os.environ.get("S2S_NVTX") == "1"


class BenchCallback(L.Callback):
    """Throughput benchmark callback; add as the sole callback in ``bench.py``.

    Brackets each training step with ``torch.cuda.synchronize()`` so the
    recorded ``step_time`` is GPU-accurate, discards ``S2S_BENCH_WARMUP`` warmup
    steps, measures ``S2S_BENCH_STEPS`` steps, then (on rank 0) appends one row
    to ``S2S_BENCH_CSV`` and sets ``trainer.should_stop = True`` to end the run.
    When ``S2S_NVTX=1`` it also emits the ``step_N`` / ``backward`` /
    ``optimizer`` NVTX ranges and brackets the measured window with
    ``cudaProfilerStart`` / ``cudaProfilerStop`` so ``nsys
    --capture-range=cudaProfilerApi`` skips warmup.

    The CSV schema is kept recognizable versus
    ``v2.0/train.py::Trainer._bench_finalize`` (``git_sha``, ``run_num``,
    ``n_gpus``, ``batch_per_gpu``, ``amp_dtype``, ``ddp_find_unused``,
    ``step_med`` / ``step_p90`` / ``step_mean`` / ``step_std``,
    ``samples_per_s``, ``peak_mem_gb_max_rank``, ...). The data-loader timing
    columns the source measured by hand (``cpu_prep_med`` / ``compute_med`` /
    ``cpu_prep_frac``) are not separable on Lightning hooks -- the DataLoader
    workers run asynchronously and the H2D transfer has already completed by
    ``on_train_batch_start`` -- so they are reported as the all-up ``step_med``
    only and omitted from the row rather than emitted as misleading zeros.

    Args:
        n_gpus: Number of GPUs in the run (for the ``samples_per_s`` and
            ``n_gpus`` columns).
        batch_per_gpu: Per-GPU batch size (for ``samples_per_s`` and
            ``batch_per_gpu``).
        config_path (optional): Path to the YAML config; hashed into
            ``config_sha16`` when present.
        run_num: Run label written to the ``run_num`` column.

    Attributes:
        n_gpus (int): See ``n_gpus`` arg.
        batch_per_gpu (int): See ``batch_per_gpu`` arg.
        config_path (str | None): See ``config_path`` arg.
        run_num (str): See ``run_num`` arg.
    """

    def __init__(self, *, n_gpus: int, batch_per_gpu: int,
                 config_path: str | None = None, run_num: str = "bench") -> None:
        """Initialise step-timing state.

        Args:
            n_gpus: Number of GPUs in the run.
            batch_per_gpu: Per-GPU batch size.
            config_path (optional): Path to the YAML config (hashed for the row).
            run_num: Run label for the CSV row.
        """
        super().__init__()
        self.n_gpus = n_gpus
        self.batch_per_gpu = batch_per_gpu
        self.config_path = config_path
        self.run_num = run_num

        self._step_times: list[float] = []
        self._iters = 0
        self._t0: float | None = None
        self._bench_loop_t0: float | None = None
        self._done = False

    # ------------------------------------------------------------------

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        """Open the per-step NVTX range and start the GPU-synced step timer.

        Args:
            trainer: The Lightning ``Trainer``.
            pl_module: The Lightning module being trained.
            batch: The current batch (already on GPU; unused).
            batch_idx: Lightning's batch index (unused).
        """
        if NVTX:
            import torch.cuda.nvtx as nvtx
            nvtx.range_push(f"step_{self._iters}")
        torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    # --- NVTX backward / optimizer ranges (only when S2S_NVTX=1) -----------
    # These let an nsys profile separate "backward compute" from "NCCL
    # gradient-sync wait" and from "optimizer step", which the in-step
    # train_module.py ranges (data_prep / forward_loss) do not.
    def on_before_backward(self, trainer, pl_module, loss) -> None:
        """Open the ``backward`` NVTX range before Lightning calls ``backward``.

        Args:
            trainer: The Lightning ``Trainer`` (unused).
            pl_module: The Lightning module being trained (unused).
            loss: The loss tensor about to be backpropagated (unused).
        """
        if NVTX:
            import torch.cuda.nvtx as nvtx
            nvtx.range_push("backward")

    def on_after_backward(self, trainer, pl_module) -> None:
        """Close the ``backward`` NVTX range after Lightning's ``backward``.

        Args:
            trainer: The Lightning ``Trainer`` (unused).
            pl_module: The Lightning module being trained (unused).
        """
        if NVTX:
            import torch.cuda.nvtx as nvtx
            nvtx.range_pop()  # backward

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        """Open the ``optimizer`` NVTX range before the optimiser step.

        The range is closed in :meth:`on_train_batch_end` because Lightning has
        no ``on_after_optimizer_step`` hook; it therefore also spans any
        post-step callback work, which is negligible here.

        Args:
            trainer: The Lightning ``Trainer`` (unused).
            pl_module: The Lightning module being trained (unused).
            optimizer: The optimizer about to step (unused).
        """
        if NVTX:
            import torch.cuda.nvtx as nvtx
            nvtx.range_push("optimizer")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        """Close the step's NVTX ranges, record its time, finalise when done.

        Synchronises the GPU, computes ``step_time = t2 - t0``, and (after the
        warmup window) appends it to the measured list. On the first measured
        step it opens the nsys capture window (``cudaProfilerStart`` when
        ``S2S_NVTX=1``). Once ``S2S_BENCH_STEPS`` steps are recorded it calls
        :meth:`_finalize` and sets ``trainer.should_stop = True``.

        Args:
            trainer: The Lightning ``Trainer``.
            pl_module: The Lightning module being trained.
            outputs: The ``training_step`` output (unused).
            batch: The current batch (unused).
            batch_idx: Lightning's batch index (unused).
        """
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

    def _finalize(self, trainer, pl_module) -> None:
        """Aggregate the step timings and write one CSV row on rank 0.

        Closes the nsys capture window (``cudaProfilerStop`` when
        ``S2S_NVTX=1``), all-reduces peak GPU memory across ranks, computes the
        step-time statistics and effective throughput, and -- on rank 0 only --
        appends a row to ``S2S_BENCH_CSV`` (header written if the file is new).
        The CSV columns mirror ``v2.0/train.py::Trainer._bench_finalize``.

        Args:
            trainer: The Lightning ``Trainer`` (for rank / world size).
            pl_module: The Lightning module (for its ``device`` in the memory
                all-reduce).
        """
        print(f"[BenchCallback] _finalize called on rank {trainer.global_rank}", flush=True)
        if NVTX:
            torch.cuda.cudart().cudaProfilerStop()

        steps = self._step_times
        n = len(steps)
        elapsed = time.perf_counter() - self._bench_loop_t0  # type: ignore[operator]

        step_sorted = sorted(steps)
        step_med = statistics.median(steps)
        step_p90 = step_sorted[int(0.9 * n)] if n >= 10 else max(steps)
        step_mean = statistics.fmean(steps)
        step_std = statistics.pstdev(steps) if n > 1 else 0.0
        samples_per_s = (self.batch_per_gpu * self.n_gpus) / step_med if step_med > 0 else 0.0

        # Fraction of wall-clock time the GPU spent idle between measured steps
        # (typically waiting on the dataloader). step_med*n is compute time;
        # elapsed is total wall time over the measurement window.
        compute_total = step_med * n
        data_idle_frac = max(0.0, (elapsed - compute_total)) / max(elapsed, 1e-9)
        samples_per_s_wall = (self.batch_per_gpu * self.n_gpus * n) / max(elapsed, 1e-9)

        if data_idle_frac > 0.10 and trainer.global_rank == 0:
            print(
                f"[BenchCallback] NOTE data-bound run "
                f"(elapsed={elapsed:.3f}s compute={compute_total:.3f}s "
                f"data_idle_frac={data_idle_frac:.2f}). "
                "step_med reflects compute only; samples_per_s_wall reflects effective throughput.",
                flush=True,
            )

        # Peak GPU memory -- max across all ranks via all-reduce.
        peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
        if trainer.world_size > 1:
            peak_t = torch.tensor(peak_gb, device=pl_module.device)
            torch.distributed.all_reduce(peak_t, op=torch.distributed.ReduceOp.MAX)
            peak_gb = peak_t.item()

        if trainer.global_rank != 0:
            return  # only rank-0 writes

        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "--short=12", "HEAD"], stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            git_sha = "unknown"

        config_sha = "unknown"
        if self.config_path:
            try:
                with open(self.config_path, "rb") as f:
                    config_sha = hashlib.sha256(f.read()).hexdigest()[:16]
            except Exception:
                pass

        # amp_dtype mirrors v2.0/train.py: S2S_AMP_DTYPE selects the autocast
        # dtype (fp16 default). Under Lightning the Trainer's precision= owns AMP;
        # we record the env value for parity with the source's row.
        amp_dtype = os.environ.get("S2S_AMP_DTYPE", "fp16")

        row = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_sha": git_sha,
            "config_sha16": config_sha,
            "run_num": self.run_num,
            "n_gpus": self.n_gpus,
            "batch_per_gpu": self.batch_per_gpu,
            "precision": str(trainer.precision),
            "amp_dtype": amp_dtype,
            "ddp_find_unused": "false",
            "step_med": f"{step_med:.6f}",
            "step_p90": f"{step_p90:.6f}",
            "step_mean": f"{step_mean:.6f}",
            "step_std": f"{step_std:.6f}",
            "samples_per_s": f"{samples_per_s:.3f}",
            "samples_per_s_wall": f"{samples_per_s_wall:.3f}",
            "data_idle_frac": f"{data_idle_frac:.3f}",
            "peak_mem_gb_max_rank": f"{peak_gb:.3f}",
            "n_steps_counted": n,
        }

        csv_path = Path(BENCH_CSV)
        if csv_path.parent != Path(""):
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
        print(f"[BenchCallback] Appended to {csv_path}\n", flush=True)
