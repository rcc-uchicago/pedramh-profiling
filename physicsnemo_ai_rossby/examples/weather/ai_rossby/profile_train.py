# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profiling-only entry point for the ai-rossby recipe. Trains nothing useful.

Runs ``AI_ROSSBY_BENCH_WARMUP`` untimed steps then ``AI_ROSSBY_BENCH_STEPS``
timed ones, and writes ONE CSV row. It is a *wrapper*, not a fork: the model,
datapipe, loss, optimizer and the training step itself are imported from
``train.py`` / ``train_loop.py``, so this file cannot drift from what actually
trains. (That drift is exactly what CLAUDE.md rule #4 exists about — see
``train.py`` vs ``train_optimized.py`` in the PanguWeather tree.)

Why this exists
---------------
ai-rossby has two timers today and neither can support an optimization ladder:

* ``cfg.bench.per_batch_tsv`` — ``perf_counter()`` deltas with **no CUDA sync**,
  so they measure kernel-launch return, not GPU completion.
* ``LaunchLogger`` — genuinely CUDA-event timed, but epoch-granular and a *mean*
  (``epoch_time / n_iters``). No median, no p90, no warmup split, no phase split.

Contract (do not drift — CLAUDE.md #10)
---------------------------------------
The CSV's first 19 columns and the NVTX range names are byte-identical to
``s2s/v2.0/train.py`` and ``PanguWeather/v2.0/train.py``. Renaming a range drops
it from ``parse_nsys.py``'s summary silently; reordering a column invalidates
every prior comparison. ``polaris/bench_instrumentation_test.py`` pins both.
Knobs are per-project (``AI_ROSSBY_*``); names and columns are shared.

Usage (COMPUTE NODE ONLY — see the login-node warning below)::

    AI_ROSSBY_BENCH=1 python -m torch.distributed.run --standalone \\
        --nnodes=1 --nproc_per_node=4 profile_train.py \\
        model=pangu_plasim_e3sm dataset=e3sm_pangu_parity \\
        training=pangu_plasim_legacy loss=mae

⚠ **Never run this on a login node, not even ``--help``.** It imports
``train.py``, which imports ``physicsnemo`` at module scope, and
``physicsnemo_ai_rossby/CLAUDE.md`` warns that CUDA/Warp init can core-dump
there. The drift-guard test is deliberately AST-only so *it* stays login-safe.
"""

from __future__ import annotations

import csv
import hashlib
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import torch
import torch.cuda.nvtx as nvtx
from omegaconf import DictConfig, OmegaConf

from physicsnemo.distributed import DistributedManager

# The trainer's own building blocks. Importing train.py is safe: main() is
# guarded by `if __name__ == "__main__"`, and the only module-level statement
# is a frozenset constant.
from train import (
    _flatten_optimizer_cfg,
    _flatten_scheduler_cfg,
    _resolve_path,
    build_datapipe,
    build_loss,
    build_model,
)
from train_loop import (
    _resolve_amp_dtype,
    make_optimizer,
    make_scheduler,
    multistep_train_step,
    train_step,
)

# --- knobs (per-project by convention; see the module docstring) ------------
BENCH = os.environ.get("AI_ROSSBY_BENCH") == "1"
BENCH_WARMUP = int(os.environ.get("AI_ROSSBY_BENCH_WARMUP", "20"))
BENCH_STEPS = int(os.environ.get("AI_ROSSBY_BENCH_STEPS", "80"))
BENCH_CSV = os.environ.get("AI_ROSSBY_BENCH_CSV", "bench_results.csv")
NVTX = os.environ.get("AI_ROSSBY_NVTX") == "1"

# The 19 canonical columns, in order, from s2s/v2.0/train.py. Anything
# ai-rossby adds is APPENDED after these — never inserted, never reordered.
S2S_COLUMNS = [
    "timestamp", "git_sha", "run_num", "n_gpus", "batch_per_gpu", "amp_dtype",
    "ddp_find_unused", "n_loaders", "step_med", "step_p90", "step_mean",
    "step_std", "cpu_prep_med", "compute_med", "cpu_prep_frac", "samples_per_s",
    "peak_mem_gb_max_rank", "scaler_skips", "n_steps_counted",
]
# ai-rossby additions. `n_params` is here because the SFNO/PanguPlasim step-time
# gap is currently a number with no explanation — SFNO is a measured 1.18 B
# params and ai-rossby's count has never been logged anywhere.
EXTRA_COLUMNS = ["samples_per_s_wall", "data_idle_frac", "config_sha16", "n_params"]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _config_sha16(cfg: DictConfig) -> str:
    """sha256 of the resolved config, truncated — pins WHAT was benched."""
    try:
        blob = OmegaConf.to_yaml(cfg, resolve=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
    except Exception:
        return "unknown"


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if not BENCH:
        print("ERROR BENCH_NOT_ENABLED: set AI_ROSSBY_BENCH=1 to run the bench.")
        sys.exit(2)

    # Same TF32/cudnn setup as train.py:792-796. Omitting it would bench a
    # different numeric path than the one that trains.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # --- bootstrap, replicated from train.py:798-804 ----------------------
    # NOT optional and NOT skippable: DistributedManager() RAISES if built
    # before initialize(), and ClimateDatapipe re-touches the same singleton
    # internally. This is the one piece of train.py's logic this wrapper must
    # copy rather than import.
    DistributedManager.initialize()
    dist = DistributedManager()
    torch.manual_seed(int(cfg.seed) + dist.rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed) + dist.rank)

    cfg_train = cfg.training

    # --- uniform steps ----------------------------------------------------
    # Every measured step must be the same shape, so we pin the FIRST stage's
    # unroll_steps and ignore the rest of the curriculum. This mirrors si/bench.py
    # forcing accumulate_grad_batches=1. It is an override: report it.
    stages = list(cfg_train.stages)
    unroll_steps = int(stages[0].get("unroll_steps", 1))
    if len(stages) > 1 and dist.rank == 0:
        print(f"bench: pinning stage 0 (unroll_steps={unroll_steps}); "
              f"{len(stages) - 1} later stage(s) ignored")

    # --- data / model / loss — all imported, none reimplemented ------------
    datapipe = build_datapipe(
        cfg,
        zarr_path=_resolve_path(cfg.dataset.zarr_path),
        distributed=(dist.world_size > 1),
        device=dist.device,
        shuffle=bool(cfg.dataset.shuffle),
        seed=int(cfg.seed),
        unroll_steps=unroll_steps,
    )
    # Random weights, no checkpoint: we are measuring speed, not quality
    # (si/bench.py's convention — it strips checkpoint paths for this reason).
    model = build_model(cfg.model).to(dist.device)
    inner_model = model
    n_params = sum(p.numel() for p in model.parameters())

    compile_mode = os.environ.get("TORCH_COMPILE_MODE", "")
    if compile_mode:
        model = torch.compile(model, mode=compile_mode, fullgraph=False)

    if dist.world_size > 1:
        from torch.nn.parallel import DistributedDataParallel

        ddp_kwargs: dict[str, Any] = {}
        bucket_cap_mb = cfg_train.get("ddp_bucket_cap_mb", None)
        if bucket_cap_mb is not None:
            ddp_kwargs["bucket_cap_mb"] = int(bucket_cap_mb)
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank] if dist.device.type == "cuda" else None,
            output_device=dist.device if dist.device.type == "cuda" else None,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
            gradient_as_bucket_view=True,
            static_graph=bool(cfg_train.get("ddp_static_graph", False)),
            **ddp_kwargs,
        )

    loss_fn = build_loss(cfg).to(dist.device)
    # Same flattening train.py:975,1088 applies — make_optimizer/make_scheduler
    # take flat keys, not the nested cfg. Calling them with the raw config
    # silently builds the wrong optimizer.
    optimizer = make_optimizer(inner_model, _flatten_optimizer_cfg(cfg_train.optimizer))
    total_steps = max(1, BENCH_WARMUP + BENCH_STEPS)
    scheduler = make_scheduler(
        optimizer,
        _flatten_scheduler_cfg(
            stages[0].scheduler,
            lr=float(cfg_train.optimizer.lr),
            steps_per_epoch=total_steps,
            num_epochs=1,
        ),
        total_steps=total_steps,
    )

    amp_dtype = _resolve_amp_dtype(cfg_train.get("amp", None))
    grad_scaler = (
        torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None
    )
    has_diagnostic = inner_model.has_diagnostic
    vae_kl_weight = float(cfg.loss.get("vae_kl_weight", 0.0))
    # Benched WITH the perturbation the production run pays (two randn draws
    # per step); negligible on GPU but measured rather than assumed.
    epsilon_factor = float(cfg_train.get("epsilon_factor", 0.0))

    # --- the measured loop ------------------------------------------------
    step_times: list[float] = []
    prep_times: list[float] = []
    compute_times: list[float] = []
    loader_waits: list[float] = []
    scaler_skips = 0
    iters = 0
    loop_t0: Optional[float] = None

    _sync()
    last_end = time.perf_counter()

    while len(step_times) < BENCH_STEPS:
        for batch in datapipe:
            if len(step_times) >= BENCH_STEPS:
                break
            measuring = iters >= BENCH_WARMUP
            if measuring and loop_t0 is None:
                # Start the profiler capture window at the first MEASURED step,
                # so nsys traces steady state and not warmup/autotune.
                #
                # The CLOCK starts at `last_end` (where the previous step's GPU
                # work finished), NOT at "now" — this step's loader_wait was
                # already measured from `last_end`, so anchoring here makes
                # `elapsed` and the accounted sum span the identical interval.
                loop_t0 = last_end
                if NVTX and torch.cuda.is_available():
                    torch.cuda.cudart().cudaProfilerStart()

            if NVTX and measuring:
                nvtx.range_push(f"step_{iters}")

            _sync()
            t0 = time.perf_counter()
            # Everything since the previous step's GPU work finished: the loader's
            # blocking fetch plus our own per-step overhead (range_push, the sync).
            # Measured OUTSIDE the step so it never inflates step_med, but INSIDE
            # the accounting so `elapsed == sum(loader_wait + step)` is an exact
            # identity and the self-check below can be trusted.
            loader_wait = t0 - last_end

            # H2D / batch prep. The datapipe already returns device tensors, so
            # this is the residual host-side work before compute begins.
            if NVTX:
                nvtx.range_push("data_prep")
            batch = {k: v for k, v in batch.items()}
            if NVTX:
                nvtx.range_pop()
            _sync()
            t1 = time.perf_counter()

            scale_before = grad_scaler.get_scale() if grad_scaler is not None else None
            step_fn = multistep_train_step if unroll_steps > 1 else train_step
            kwargs = dict(
                model=model, loss_fn=loss_fn, optimizer=optimizer,
                scheduler=scheduler, batch=batch, has_diagnostic=has_diagnostic,
                vae_kl_weight=vae_kl_weight, amp_dtype=amp_dtype,
                grad_scaler=grad_scaler,
                epsilon_factor=epsilon_factor,
            )
            if unroll_steps > 1:
                kwargs["unroll_steps"] = unroll_steps
            step_fn(**kwargs)

            _sync()
            t2 = time.perf_counter()

            if NVTX and measuring:
                nvtx.range_pop()  # step_{i}

            # A GradScaler-skipped step ran no optimizer update, so its timing
            # is not comparable. Count it, don't record it.
            skipped = (
                grad_scaler is not None
                and grad_scaler.get_scale() < scale_before
            )
            if skipped:
                scaler_skips += 1
            elif measuring:
                step_times.append(t2 - t0)
                prep_times.append(t1 - t0)
                compute_times.append(t2 - t1)
                loader_waits.append(loader_wait)

            iters += 1
            # Anchor the next step's loader_wait at t2 — the instant this step's
            # GPU work completed — so the bookkeeping above (range_pop, list
            # appends) lands INSIDE the next loader_wait rather than in a gap
            # that belongs to no bucket. Using perf_counter() here instead leaks
            # that time: invisible in an eager run (microseconds), but ~80 ms/step
            # under nsys, which tripped BENCH_CLOCK_MISMATCH on job 7352406.
            last_end = t2

    # Close the clock at the last recorded step boundary, not "now", so
    # `elapsed` is exactly the interval the accounted sum covers.
    loop_end = last_end
    if NVTX and torch.cuda.is_available():
        torch.cuda.cudart().cudaProfilerStop()

    _finalize(
        cfg=cfg, dist=dist, step_times=step_times, prep_times=prep_times,
        compute_times=compute_times, loader_waits=loader_waits,
        scaler_skips=scaler_skips, loop_t0=loop_t0, loop_end=loop_end,
        amp_dtype=amp_dtype, n_params=n_params, unroll_steps=unroll_steps,
    )


def _finalize(*, cfg, dist, step_times, prep_times, compute_times, loader_waits,
              scaler_skips, loop_t0, loop_end, amp_dtype, n_params,
              unroll_steps) -> None:
    """Reduce, self-check, and write one CSV row from rank 0."""
    n = len(step_times)
    if n == 0:
        print("ERROR BENCH_NO_STEPS_RECORDED")
        sys.exit(2)

    peak_gb = (
        torch.cuda.max_memory_allocated() / 1024 ** 3
        if torch.cuda.is_available() else 0.0
    )
    if dist.world_size > 1 and torch.cuda.is_available():
        t = torch.tensor(peak_gb, device=dist.device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        peak_gb = float(t.item())

    if dist.rank != 0:
        sys.exit(0)

    elapsed = loop_end - (loop_t0 if loop_t0 is not None else loop_end)
    step_sum = sum(step_times)
    loader_sum = sum(loader_waits)

    # Self-check: the measured windows must account for the wall time between
    # them. A large gap means the timer is measuring the wrong thing — a wrong
    # number is worse than no number, so fail loudly (s2s/v2.0/train.py:928-934).
    expected = step_sum + loader_sum
    if elapsed > 0 and abs(elapsed - expected) / elapsed > 0.10 and scaler_skips == 0:
        print(f"ERROR BENCH_CLOCK_MISMATCH elapsed={elapsed:.3f}s "
              f"accounted={expected:.3f}s (>10% apart)")
        sys.exit(3)

    step_med = statistics.median(step_times)
    samples_per_s = dist.world_size * int(cfg.dataset.batch_size) / step_med
    samples_per_s_wall = (
        dist.world_size * int(cfg.dataset.batch_size) * n / elapsed
        if elapsed > 0 else 0.0
    )
    data_idle_frac = (elapsed - step_sum) / elapsed if elapsed > 0 else 0.0

    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": _git_sha(),
        "run_num": str(cfg.get("run_name", "")),
        "n_gpus": dist.world_size,
        "batch_per_gpu": int(cfg.dataset.batch_size),
        "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype else "off",
        "ddp_find_unused": str(bool(dist.find_unused_parameters)).lower(),
        "n_loaders": int(cfg.dataset.get("num_workers", 0)),
        "step_med": round(step_med, 6),
        "step_p90": round(sorted(step_times)[int(0.9 * n)] if n >= 10 else max(step_times), 6),
        "step_mean": round(statistics.fmean(step_times), 6),
        "step_std": round(statistics.pstdev(step_times), 6),
        "cpu_prep_med": round(statistics.median(prep_times), 6),
        "compute_med": round(statistics.median(compute_times), 6),
        "cpu_prep_frac": round(statistics.median(prep_times) / step_med, 6),
        "samples_per_s": round(samples_per_s, 4),
        "peak_mem_gb_max_rank": round(peak_gb, 3),
        "scaler_skips": scaler_skips,
        "n_steps_counted": n,
        # --- appended after s2s's 19; never inserted ---
        "samples_per_s_wall": round(samples_per_s_wall, 4),
        "data_idle_frac": round(data_idle_frac, 4),
        "config_sha16": _config_sha16(cfg),
        "n_params": n_params,
    }

    path = Path(BENCH_CSV)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=S2S_COLUMNS + EXTRA_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(row)

    side = path.parent / f"bench_env_{row['run_num'] or row['git_sha']}.txt"
    with open(side, "w") as fh:
        fh.write(f"git_sha={row['git_sha']}\ntorch={torch.__version__}\n")
        fh.write(f"cuda={torch.version.cuda}\n")
        if torch.cuda.is_available():
            fh.write(f"gpu={torch.cuda.get_device_name(0)}\n")
        fh.write(f"world_size={dist.world_size}\nn_params={n_params}\n")
        fh.write(f"unroll_steps={unroll_steps}\nconfig_sha16={row['config_sha16']}\n")
        fh.write(f"warmup={BENCH_WARMUP}\nsteps={BENCH_STEPS}\n")
        fh.write(f"pbs_jobid={os.environ.get('PBS_JOBID', '')}\n")

    # Output hygiene (DESIGN §7): <=10 lines, max-error-and-where, no tensors.
    print(f"BENCH n={n} step_med={step_med * 1000:.1f}ms "
          f"p90={row['step_p90'] * 1000:.1f}ms samples/s={samples_per_s:.2f} "
          f"peak={peak_gb:.2f}GB params={n_params:,}")
    if data_idle_frac > 0.10:
        print(f"WARN DATA_BOUND data_idle_frac={data_idle_frac:.3f} (>0.10) — "
              f"the GPU is waiting on the loader; raise dataset.num_workers")
    print(f"BENCH_CSV_ROW_WRITTEN {path}")


if __name__ == "__main__":
    main()
