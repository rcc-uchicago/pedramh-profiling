#!/usr/bin/env python3
"""train_plasim.py — CLI entry for PlaSim → Makani SFNO training.

Mirrors stock ``makani/train.py::main`` for the runtime-injected params,
but instantiates :class:`sfno_training.trainer.PlasimTrainer` rather
than stock ``Trainer``. The PlasimTrainer's ``__init__`` installs the
three monkey-patches before stock ``Trainer.__init__`` runs.

Usage::

    python -m sfno_training.train_plasim \\
        --yaml_config /path/to/plasim_sim52_baseline.yaml \\
        --config plasim_sim52_astro_64x128 \\
        --run_num 0
"""

from __future__ import annotations

import hashlib
import logging
import os
from argparse import Namespace
from functools import partial
from math import prod

import torch

from makani.utils import argument_parser, comm, logging_utils, profiling
from makani.utils.parse_dataset_metada import parse_dataset_metadata
from makani.utils.profiling import Timer
from makani.utils.YParams import YParams

from sfno_training.trainer import PlasimTrainer


logger = logging.getLogger(__name__)


def _resolve_batch_sizes(params, data_parallel_size: int) -> int:
    """Resolve per-rank batch from a *global* ``params.batch_size``.

    Reads ``params.batch_size`` (already overwritten by an
    ``args.batch_size > 0`` CLI override at the call site) as the global
    batch, asserts divisibility by ``data_parallel_size``, stores both
    ``params['global_batch_size']`` and the per-rank ``params['batch_size']``,
    and returns the per-rank value. Behaviour-equivalent to the inline block
    that lived at this site before; lifted out so the divisibility contract
    is unit-testable without exercising the full CLI / ``comm.init`` path
    (docs/2026-05-05_ddp_throughput_fix_plan.md §I0).
    """
    global_batch = int(params.batch_size)
    assert global_batch % data_parallel_size == 0, (
        f"global_batch_size={global_batch} must be divisible "
        f"by data parallel size {data_parallel_size}"
    )
    params["global_batch_size"] = global_batch
    per_rank = global_batch // data_parallel_size
    params["batch_size"] = per_rank
    return per_rank


def _log_ddp_launch_summary(
    params,
    *,
    world_size: int,
    data_parallel_size: int,
) -> None:
    """Emit a single labelled block summarizing DDP / batch / dataloader config.

    Called once on rank 0 right after the file logger is configured so the
    output lands in both stdout and the per-experiment ``out.log``. Surfaces
    the inputs that drive A/B comparisons across the I1 sweep and the I2
    microbench (docs/2026-05-05_ddp_throughput_fix_plan.md §I0). All values
    come from ``params`` plus the ``world_size`` / ``data_parallel_size``
    that the caller derives from ``comm`` — nothing is hard-coded.
    """
    ema_cfg = params.get("ema", {}) or {}
    global_bs = int(params.get("global_batch_size", int(params.batch_size) * data_parallel_size))
    per_rank_bs = int(params.batch_size)
    lines = [
        "===== DDP launch summary =====",
        f"world_size                = {world_size}",
        f"data_parallel_size        = {data_parallel_size}",
        f"global_batch_size         = {global_bs}",
        f"per_rank_batch_size       = {per_rank_bs}",
        f"expected_train_steps_per_epoch  = floor(len(train) / {global_bs})",
        f"num_data_workers          = {params.get('num_data_workers', 0)}",
        f"prefetch_factor           = {params.get('prefetch_factor', None)}",
        f"persistent_workers        = {params.get('persistent_workers', None)}",
        f"multistep_count           = {params.get('multistep_count', 1)}",
        f"valid_autoreg_steps       = {params.get('valid_autoreg_steps', None)}",
        f"ema.enabled               = {bool(ema_cfg.get('enabled', False))}",
        f"ema_validation_period     = {params.get('ema_validation_period', 1)}",
        f"amp_mode                  = {params.get('amp_mode', 'none')}",
        f"checkpointing_level       = {params.get('checkpointing_level', 0)}",
        f"pretrained_checkpoint_path = {params.get('pretrained_checkpoint_path', None)}",
        f"resuming                  = {params.get('resuming', False)}",
        "==============================",
    ]
    for line in lines:
        logger.info(line)


def _write_warmstart_provenance(params, exp_dir: str) -> None:
    """Write warmstart_provenance.txt for a warm-started run.

    Documents the loaded source ckpt + the recipe knobs at launch time so
    eval-report rendering and post-hoc audits don't need to re-read SLURM
    logs. See docs/2026-05-14_v11_clip_warmstart_continuation_plan.md §6.1.
    """
    ckpt_path = params.get("pretrained_checkpoint_path")
    if not ckpt_path:
        return
    out_path = os.path.join(exp_dir, "warmstart_provenance.txt")

    flavor = "unknown"
    base = os.path.basename(str(ckpt_path))
    if "best_ckpt_ema" in base:
        flavor = "best_ckpt_ema_mp0 (ema)"
    elif "best_ckpt" in base:
        flavor = "best_ckpt_mp0 (raw)"
    elif "ckpt_mp" in base:
        flavor = base

    size_bytes: object = "missing"
    sha256_hex: object = "missing"
    try:
        st = os.stat(ckpt_path)
        size_bytes = int(st.st_size)
        h = hashlib.sha256()
        with open(ckpt_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sha256_hex = h.hexdigest()[:16]
    except OSError as exc:
        size_bytes = f"stat-failed: {exc}"
        sha256_hex = "n/a"

    input_noise = params.get("input_noise") or {}
    losses = params.get("losses") or []
    channel_weights = (losses[0].get("channel_weights") if losses else None)

    ema_cfg = params.get("ema", {}) or {}
    lines = [
        f"pretrained_checkpoint_path = {ckpt_path}",
        f"pretrained_checkpoint_flavor = {flavor}",
        f"pretrained_checkpoint_size_bytes = {size_bytes}",
        f"pretrained_checkpoint_sha256 = {sha256_hex}",
        "warmstart_load_order = after super().__init__, before EMAModel construction",
        f"lr_peak = {params.get('lr', None)}",
        (
            "lr_schedule = "
            f"{params.get('scheduler', None)}"
            f"(warmup={params.get('lr_warmup_steps', None)} epoch, "
            f"min={params.get('scheduler_min_lr', None)}, "
            f"T_max={params.get('scheduler_T_max', None)})"
        ),
        f"max_epochs = {params.get('max_epochs', None)}",
        f"batch_size_global = {params.get('global_batch_size', None)}",
        f"ema_decay = {ema_cfg.get('decay', None)}",
        f"optimizer_max_grad_norm = {params.get('optimizer_max_grad_norm', None)}",
        f"input_noise_sigma = {input_noise.get('sigma', None)}",
        f"channel_weights = {channel_weights}",
        f"n_history = {params.get('n_history', None)}",
        f"n_future = {params.get('n_future', None)}",
        f"multistep_count = {params.get('multistep_count', None)}",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("wrote warmstart provenance sidecar: %s", out_path)


def _world_size_from_env(names: tuple[str, ...], default: int = 1) -> int:
    """Return the largest positive world-size hint found in the environment."""
    world_size = default
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            parsed = int(value)
        except ValueError:
            continue
        if parsed > 0:
            world_size = max(world_size, parsed)
    return world_size


def _model_parallel_size(args: Namespace) -> int:
    return prod(
        [
            args.h_parallel_size,
            args.w_parallel_size,
            args.fin_parallel_size,
            args.fout_parallel_size,
        ]
    )


def _should_skip_distributed_init(args: Namespace) -> bool:
    """Single-rank ``--disable_ddp`` runs do not need torch.distributed.

    PhysicsNeMo's Slurm initializer expects launch-time variables that are
    absent in a plain ``sbatch`` shell on Stampede3. The SFNO smoke/tiny/short
    launchers are explicitly single-task jobs with ``--disable_ddp``, so avoid
    initializing a process group unless the user requested multiple ranks or
    model parallelism.
    """
    if not args.disable_ddp:
        return False
    if _model_parallel_size(args) != 1:
        return False
    world_size = _world_size_from_env(
        ("WORLD_SIZE", "SLURM_NTASKS", "SLURM_NPROCS", "OMPI_COMM_WORLD_SIZE"),
        default=1,
    )
    return world_size == 1


def main() -> None:
    parser = argument_parser.get_default_argument_parser()
    parser.add_argument(
        "--mode",
        default="train",
        type=str,
        choices=["train", "test"],
        help="Run training or perform a test.",
    )
    parser.add_argument(
        "--pretrained_checkpoint_path",
        default=None,
        type=str,
        help=(
            "Optional absolute path to a legacy makani checkpoint to warm-start "
            "model weights from (plan: 2026-05-14_v11_clip_warmstart_continuation). "
            "Single source of truth for the warm-start source — not a YAML field. "
            "Loads model_state ONLY; optimizer/scheduler/counters are kept fresh. "
            "Ignored when resuming from this run's own EXP_DIR."
        ),
    )
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)

    # distributed wireup
    params["fin_parallel_size"] = args.fin_parallel_size
    params["fout_parallel_size"] = args.fout_parallel_size
    params["h_parallel_size"] = args.h_parallel_size
    params["w_parallel_size"] = args.w_parallel_size
    params["model_parallel_sizes"] = [
        args.h_parallel_size,
        args.w_parallel_size,
        args.fin_parallel_size,
        args.fout_parallel_size,
    ]
    params["model_parallel_names"] = ["h", "w", "fin", "fout"]
    params["parameters_reduction_buffer_count"] = args.parameters_reduction_buffer_count

    params["load_checkpoint"] = args.load_checkpoint
    params["save_checkpoint"] = args.save_checkpoint

    distributed_initialized = False
    with Timer() as timer:
        if _should_skip_distributed_init(args):
            world_rank = 0
        else:
            comm.init(
                model_parallel_sizes=params["model_parallel_sizes"],
                model_parallel_names=params["model_parallel_names"],
                verbose=False,
            )
            distributed_initialized = True
            world_rank = comm.get_world_rank()
    if world_rank == 0:
        if distributed_initialized:
            print(f"Communicators wireup time: {timer.time:.2f}s")
        else:
            print(
                "Communicators wireup skipped for single-rank "
                f"--disable_ddp run: {timer.time:.2f}s"
            )

    params["world_size"] = comm.get_world_size()
    if args.batch_size > 0:
        params.batch_size = args.batch_size
    _resolve_batch_sizes(params, comm.get_size("data"))

    if "optimizer_max_grad_norm" not in params:
        params["optimizer_max_grad_norm"] = 1.0

    if torch.cuda.is_available():
        torch.cuda.set_device(comm.get_local_rank())
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.enable_grad_anomaly_detection:
        torch.autograd.set_detect_anomaly(True)

    expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    if world_rank == 0:
        logging.info(f"writing output to {expDir}")
        if not os.path.isdir(expDir):
            os.makedirs(expDir, exist_ok=True)
            os.makedirs(os.path.join(expDir, "training_checkpoints"), exist_ok=True)
            os.makedirs(os.path.join(expDir, "wandb"), exist_ok=True)

    params["experiment_dir"] = os.path.abspath(expDir)
    params["checkpoint_path"] = os.path.join(
        expDir, "training_checkpoints/ckpt_mp{mp_rank}_v{checkpoint_version}.tar"
    )
    params["best_checkpoint_path"] = os.path.join(
        expDir, "training_checkpoints/best_ckpt_mp{mp_rank}.tar"
    )

    resuming = True
    for mp_rank in range(comm.get_size("model")):
        checkpoint_fname = params.checkpoint_path.format(mp_rank=mp_rank, checkpoint_version=0)
        if params["load_checkpoint"] == "legacy" or mp_rank < 1:
            resuming = resuming and os.path.isfile(checkpoint_fname)
    params["resuming"] = resuming

    params["amp_mode"] = args.amp_mode
    params["jit_mode"] = args.jit_mode
    params["skip_validation"] = args.skip_validation
    params["skip_training"] = args.skip_training
    params["enable_odirect"] = args.enable_odirect
    params["enable_s3"] = args.enable_s3
    params["checkpointing_level"] = args.checkpointing_level
    params["enable_synthetic_data"] = args.enable_synthetic_data
    params["split_data_channels"] = args.split_data_channels
    params["print_timings_frequency"] = args.print_timings_frequency
    params["multistep_count"] = args.multistep_count
    params["n_future"] = args.multistep_count - 1
    params["disable_ddp"] = args.disable_ddp
    params["enable_grad_anomaly_detection"] = args.enable_grad_anomaly_detection
    if args.pretrained_checkpoint_path:
        params["pretrained_checkpoint_path"] = args.pretrained_checkpoint_path

    if not hasattr(params, "wandb_dir") or params["wandb_dir"] is None:
        params["wandb_dir"] = expDir

    if world_rank == 0:
        logging_utils.config_logger()
        logging_utils.log_to_file(
            logger_name=None,
            log_filename=os.path.join(expDir, "out.log"),
        )
        logging_utils.log_versions()
        params.log(logging.getLogger())
        _log_ddp_launch_summary(
            params,
            world_size=comm.get_world_size(),
            data_parallel_size=comm.get_size("data"),
        )

    params["log_to_wandb"] = (world_rank == 0) and params["log_to_wandb"]
    params["log_to_screen"] = (world_rank == 0) and params["log_to_screen"]

    if "metadata_json_path" in params:
        params, _ = parse_dataset_metadata(params["metadata_json_path"], params=params)
    else:
        raise RuntimeError(
            "params is missing 'metadata_json_path' — required for sfno_training "
            "(produced by plasim_makani_packager.metadata.write_outputs)."
        )

    if world_rank == 0:
        _write_warmstart_provenance(params, expDir)

    trainer = PlasimTrainer(params, world_rank)

    # torch.profiler / CUPTI capture branch ported from
    # makani-src/makani/train.py:147-169. Default behaviour
    # (no --capture_ranks) is identical to a bare trainer.train() call.
    if args.capture_prefix is not None:
        out_dir = os.path.dirname(args.capture_prefix)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    if params.get("skip_training", False):
        pass
    elif world_rank in args.capture_ranks:
        if args.capture_type == "torch":
            capture_prefix = (
                f"{args.capture_prefix}_rank{world_rank}"
                if args.capture_prefix is not None
                else None
            )
            trace_handler = partial(
                profiling.trace_handler,
                print_stats=True,
                export_trace_prefix=capture_prefix,
            )
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=args.capture_range_start - 1,
                    warmup=1,
                    active=args.capture_range_stop - args.capture_range_start,
                    repeat=1,
                ),
                on_trace_ready=trace_handler,
            ) as profiler_ctx:
                if args.capture_mode == "training":
                    trainer.train(training_profiler=profiler_ctx)
                elif args.capture_mode == "validation":
                    trainer.train(validation_profiler=profiler_ctx)
        elif args.capture_type == "cupti":
            with profiling.CUDAProfiler(
                capture_range_start=args.capture_range_start,
                capture_range_stop=args.capture_range_stop,
                enabled=True,
            ) as profiler_ctx:
                with torch.autograd.profiler.emit_nvtx(
                    enabled=True, record_shapes=False
                ):
                    if args.capture_mode == "training":
                        trainer.train(training_profiler=profiler_ctx)
                    elif args.capture_mode == "validation":
                        trainer.train(validation_profiler=profiler_ctx)
    else:
        trainer.train()

    if distributed_initialized:
        comm.cleanup()


if __name__ == "__main__":
    main()
