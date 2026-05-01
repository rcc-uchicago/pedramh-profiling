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

import logging
import os
from argparse import Namespace
from math import prod

import torch

from makani.utils import argument_parser, comm, logging_utils, profiling
from makani.utils.parse_dataset_metada import parse_dataset_metadata
from makani.utils.profiling import Timer
from makani.utils.YParams import YParams

from sfno_training.trainer import PlasimTrainer


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
    params["global_batch_size"] = params.batch_size
    assert params["global_batch_size"] % comm.get_size("data") == 0, (
        f"global_batch_size={params['global_batch_size']} must be divisible "
        f"by data parallel size {comm.get_size('data')}"
    )
    params["batch_size"] = int(params["global_batch_size"] // comm.get_size("data"))

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

    params["log_to_wandb"] = (world_rank == 0) and params["log_to_wandb"]
    params["log_to_screen"] = (world_rank == 0) and params["log_to_screen"]

    if "metadata_json_path" in params:
        params, _ = parse_dataset_metadata(params["metadata_json_path"], params=params)
    else:
        raise RuntimeError(
            "params is missing 'metadata_json_path' — required for sfno_training "
            "(produced by plasim_makani_packager.metadata.write_outputs)."
        )

    trainer = PlasimTrainer(params, world_rank)

    if not params.get("skip_training", False):
        trainer.train()

    if distributed_initialized:
        comm.cleanup()


if __name__ == "__main__":
    main()
