"""Per-epoch step-timing telemetry for the real training run. Opt-in, ~free.

WHY THIS IS NOT THE BENCH HARNESS
---------------------------------
``PANGU_BENCH``/``AI_ROSSBY_BENCH`` are steady-state instruments: a fixed
warmup + N timed steps, one aggregate CSV row, and on the PanguWeather side the
run *exits* afterwards with per-iteration diagnostics skipped. ai-rossby's
``profile_train.py`` additionally builds **no EMA** and builds its scheduler over
the bench window rather than the real schedule.

That makes them structurally unable to answer "what do the first N epochs of the
production run cost" — which is exactly the interesting window, because the
reference config has a **5-epoch LR warmup and a 6-epoch EMA warmup**, and EMA on
a 1.18 B-parameter model is an extra full sweep over ~4.7 GB of weights on every
step. Step cost genuinely changes inside epochs 1-6.

So this rides along on the production run instead: one CSV row per epoch, from
the loop that is actually training.

COST
----
Two ``cudaEvent``s per step (~microseconds each) and one all-reduce per *epoch*.
No ``torch.cuda.synchronize()`` in the step path — a per-step sync would remove
CPU/GPU overlap and change the very number being measured. Events are converted
to floats in chunks, so at most ``_CHUNK * 2`` stay live no matter how long the
epoch is.

Nothing here touches the model, the data, the loss, or the RNG. It is timing
only, and it is **off unless the env knob is set** — an unset knob leaves the
training path byte-identical.

CONTRACT (CLAUDE.md #10)
------------------------
``COLUMNS`` and their meaning are a CROSS-PROJECT contract, exactly like the
bench CSV's 19: the whole point is that a PanguWeather row and an ai-rossby row
are directly comparable. This file is DUPLICATED verbatim into
``physicsnemo_ai_rossby/examples/weather/ai_rossby/epoch_telemetry.py`` (the two
trees share no import path — DESIGN §2c), and
``epoch_telemetry_test.py`` asserts the copies have not drifted. Change one,
change both, in the same commit.

Knobs are per-project, as elsewhere:
    PANGU_EPOCH_TELEMETRY=1              enable
    PANGU_EPOCH_TELEMETRY_CSV=<path>     default: epoch_telemetry.csv in cwd
    PANGU_EPOCH_TELEMETRY_EPOCHS=<n>     record only the first n epochs (0 = all)
(ai-rossby: the same three with an AI_ROSSBY_ prefix.)
"""

from __future__ import annotations

import csv
import os
import socket
import statistics
import subprocess
import time
from pathlib import Path

import torch

# The per-epoch row. Column NAMES and ORDER are the cross-project contract —
# append only, never insert, never rename. `harness` is what lets the two
# projects' files be concatenated into one table.
COLUMNS = [
    "timestamp",        # ISO-8601, epoch end
    "harness",          # panguweather | ai_rossby
    "git_sha",
    "run_name",
    "host",
    "epoch",            # 1-based global epoch
    "n_gpus",
    "batch_per_gpu",
    "amp_dtype",
    "n_loaders",        # dataloader worker count — a known confound, so it is recorded
    "n_steps",          # optimizer steps counted this epoch
    "step_med_ms",      # GPU step time: data prep -> optimizer -> EMA -> scheduler
    "step_p90_ms",
    "step_mean_ms",
    "step_std_ms",
    "samples_per_s",    # global: n_steps * batch_per_gpu * n_gpus / epoch_wall_s
    "epoch_wall_s",
    "gpu_busy_frac",    # sum(step_ms)/1000 / epoch_wall_s; 1 - this is loader/other idle
    "peak_mem_gb",      # run-to-date, max over ranks (same semantic as the bench CSV)
    # 1 when the EMA parameter sweep RAN on every step this epoch. The two
    # harnesses gate this differently, and that is itself one of the findings:
    # PanguWeather skips `update_ema` entirely until `ema_warmup_epochs`, so its
    # step cost jumps at that epoch; ai-rossby calls `ema.update` from epoch 1
    # and warms up only the decay VALUE, so it pays the sweep throughout. On a
    # 1.18 B-param model that sweep is ~4.7 GB of elementwise traffic per step.
    "ema_active",
    "lr",
]

# How many event pairs stay live before being converted to floats. Bounded so a
# 10,950-step epoch does not hold ~22,000 CUDA events at once; large enough that
# the sync always lands on work finished long ago and never stalls.
_CHUNK = 512


def _git_sha(cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(cwd), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


class EpochTelemetry:
    """Records per-step GPU time and writes one row per epoch from rank 0.

    Usage in a training loop::

        tel = EpochTelemetry(prefix="PANGU", harness="panguweather", ...)
        for epoch in ...:
            tel.epoch_start(epoch)
            for batch in loader:
                tel.step_start()
                ...                      # data prep, forward, backward, opt, EMA
                tel.step_end()
            tel.epoch_end(lr=..., ema_active=...)

    Every method is a no-op when disabled, so the calls can sit unconditionally
    in the loop.
    """

    def __init__(self, *, prefix, harness, run_name, n_gpus, batch_per_gpu,
                 amp_dtype, n_loaders, rank, repo_dir):
        self.enabled = os.environ.get(f"{prefix}_EPOCH_TELEMETRY") == "1"
        self.csv_path = os.environ.get(
            f"{prefix}_EPOCH_TELEMETRY_CSV", "epoch_telemetry.csv"
        )
        self.max_epochs = int(os.environ.get(f"{prefix}_EPOCH_TELEMETRY_EPOCHS", "0"))
        self.harness = harness
        self.rank = rank
        self._meta = {
            "harness": harness,
            "git_sha": _git_sha(Path(repo_dir)),
            "run_name": run_name,
            "host": socket.gethostname(),
            "n_gpus": n_gpus,
            "batch_per_gpu": batch_per_gpu,
            "amp_dtype": amp_dtype,
            "n_loaders": n_loaders,
        }
        self.enabled = self.enabled and torch.cuda.is_available()
        self._reset()

    # --- lifecycle ---------------------------------------------------------

    def _reset(self):
        self._pending = []      # [(start_event, end_event)] not yet converted
        self._ms = []           # converted per-step durations
        self._t0 = None
        self._epoch = None
        self._start = None

    def _recording(self) -> bool:
        return (
            self.enabled
            and self._epoch is not None
            and (self.max_epochs <= 0 or self._epoch <= self.max_epochs)
        )

    def epoch_start(self, epoch: int) -> None:
        if not self.enabled:
            return
        self._reset()
        self._epoch = int(epoch)
        if not self._recording():
            return
        # Deliberately NO reset_peak_memory_stats() here. `peak_mem_gb` is
        # run-to-date, the same semantic as the bench CSV's
        # `peak_mem_gb_max_rank` — and resetting would silently lower the peak a
        # concurrently-enabled PANGU_BENCH row reports. The growth signal
        # survives anyway: a run-to-date peak is monotone, so a footprint jump
        # when the optimizer state or EMA shadow lands still shows as a step up.
        self._t0 = time.perf_counter()

    def step_start(self) -> None:
        if not self._recording():
            return
        self._start = torch.cuda.Event(enable_timing=True)
        self._start.record()

    def step_end(self) -> None:
        if not self._recording() or self._start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._pending.append((self._start, end))
        self._start = None
        if len(self._pending) >= _CHUNK:
            self._drain()

    def _drain(self) -> None:
        """Convert queued event pairs to milliseconds.

        Synchronizing on the LAST end event of the chunk is enough — CUDA events
        on one stream complete in order, so every earlier pair is ready too. By
        the time a chunk is full that work is hundreds of steps old, so this
        returns immediately rather than stalling the pipeline.
        """
        if not self._pending:
            return
        self._pending[-1][1].synchronize()
        self._ms.extend(s.elapsed_time(e) for s, e in self._pending)
        self._pending = []

    def epoch_end(self, *, lr=float("nan"), ema_active=False) -> None:
        """Finish the epoch and append a row. Must be called on ALL ranks.

        The peak-memory figure is reduced across ranks, which is a collective —
        a rank that skipped this call would hang the others.
        """
        if not self._recording():
            self._epoch = None
            return
        self._drain()
        wall = time.perf_counter() - self._t0
        n = len(self._ms)

        peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            t = torch.tensor([peak_gb], device="cuda")
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
            peak_gb = float(t.item())

        if self.rank == 0 and n > 0:
            self._write_row(n=n, wall=wall, peak_gb=peak_gb, lr=lr,
                            ema_active=ema_active)
        self._epoch = None

    # --- output ------------------------------------------------------------

    def _write_row(self, *, n, wall, peak_gb, lr, ema_active) -> None:
        ms = sorted(self._ms)
        row = {
            **self._meta,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "epoch": self._epoch,
            "n_steps": n,
            "step_med_ms": round(statistics.median(ms), 3),
            "step_p90_ms": round(ms[min(n - 1, int(0.90 * n))], 3),
            "step_mean_ms": round(statistics.fmean(ms), 3),
            "step_std_ms": round(statistics.pstdev(ms) if n > 1 else 0.0, 3),
            "samples_per_s": round(
                n * self._meta["batch_per_gpu"] * self._meta["n_gpus"] / wall, 4
            ) if wall > 0 else 0.0,
            "epoch_wall_s": round(wall, 3),
            "gpu_busy_frac": round(sum(ms) / 1000.0 / wall, 4) if wall > 0 else 0.0,
            "peak_mem_gb": round(peak_gb, 3),
            "ema_active": int(bool(ema_active)),
            "lr": lr,
        }
        path = Path(self.csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
        print(
            "EPOCH_TELEMETRY epoch=%d n=%d step_med=%.1fms gpu_busy=%.1f%% "
            "peak=%.2fGB ema=%d -> %s"
            % (row["epoch"], n, row["step_med_ms"], 100 * row["gpu_busy_frac"],
               row["peak_mem_gb"], row["ema_active"], path),
            flush=True,
        )
