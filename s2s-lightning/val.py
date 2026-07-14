"""Lightning validation / inference entry point for S2S Pangu-PLASIM (Phase 4).

This is the inference-side counterpart of the Phase-3 training entry point
:mod:`train.py`. It mirrors the *shape* of the SI template at
``$SI_DIR/val.py`` (argparse -> config -> single-device ``Trainer`` ->
``trainer.validate(model, datamodule)``) but feeds it S2S's flat
:class:`utils.YParams.YParams` config rather than SI's nested
``model:`` / ``data:`` / ``training:`` dict, reusing :mod:`train.py`'s
``process_args`` / ``_resolve_devices`` helpers so the two entry points stay in
lockstep.

The validation/inference path is **reused, not rewritten**. It drives
:class:`modules.train_module.TrainModule` (whose
:meth:`~modules.train_module.TrainModule.validation_step` ports the per-batch
core of ``v2.0/train.py::Trainer.validate_one_epoch`` for scoring and
``v2.0/inference.py::Stepper.save_prediction`` for the netCDF write) against
:class:`data.datamodule.ClimateDataModule`'s validation loader -- the *same*
validate loader the canonical ``v2.0/inference.py::Stepper`` reads (via
``get_data_loader(..., validate=True)``).

Several inference contracts are enforced here rather than in the module, because
they are entry-point / Trainer-level concerns:

* **Single-device validation.** Like SI ``val.py`` this forces ``devices=1``
  and ``strategy="auto"`` (no DDP, no injected/distributed sampler), so the
  validate loader is read sequentially on one GPU. This sidesteps DDP val
  sharding entirely for the inference run.
* **Graceful no-checkpoint (random weights).** ``ckpt_path`` is passed to
  ``trainer.validate`` only when a checkpoint is supplied; without one the run
  proceeds on random weights, preserving the semantics of
  ``Stepper.__init__`` (which warns rather than crashes when the checkpoint file
  is absent -- "outputs are not meaningful unless this is a deliberate profiling
  run"). A no-checkpoint run is acceptable for a *structural* smoke.
* **Prediction saving.** When ``--save_predictions`` is set this resolves a
  writable ``predictions_dir`` under the run directory and injects it (plus a
  ``run_num``) into the flat config, which
  :meth:`modules.train_module.TrainModule.save_predictions` reads to write
  per-sample netCDF on rank 0 / batch 0.

The ``utils.*`` / ``networks.*`` imports reused transitively by the modules
resolve only when ``v2.0/`` is on ``PYTHONPATH``
(``PYTHONPATH=v2.0:.``).

Usage:
    PYTHONPATH=v2.0:. python val.py --yaml_config configs/test_midway.yaml \\
        --config S2S --checkpoint /path/to/ckpt.ckpt --save_predictions

See Also:
    train.py: The training-side sibling entry point (whose ``process_args`` /
        ``_resolve_devices`` this module reuses).
    modules.train_module.TrainModule: The LightningModule whose
        ``validation_step`` / ``save_predictions`` this drives.
    data.datamodule.ClimateDataModule: The DataModule supplying the validation
        loader and the normalizer.
    v2.0/inference.py: The canonical (bench-instrumented) inference source this
        path ports.
"""

import argparse
import os
from datetime import datetime

import torch

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger

from utils.YParams import YParams
from data.datamodule import ClimateDataModule
from modules.train_module import TrainModule
from train import process_args, _resolve_devices


def main(args):
    """Build the DataModule, module and single-device Trainer, then validate.

    Mirrors the SI ``val.py`` flow: load + override the config, force
    single-device validation, build the ``Trainer``, and call
    ``trainer.validate`` (passing ``ckpt_path`` only when a checkpoint is
    supplied). When ``--save_predictions`` is set, a writable ``predictions_dir``
    is injected into the config so
    :meth:`modules.train_module.TrainModule.save_predictions` can write netCDF.

    Args:
        args: The parsed argparse namespace (see :func:`build_parser`).
    """
    cfg = os.path.abspath(args.yaml_config)
    print(f"Loading config: {cfg} (section {args.config})", flush=True)
    params = YParams(cfg, args.config)

    # Reuse train.py's override + Trainer-knob resolution (devices/accelerator/
    # precision/wandb/has_diagnostic/...). process_args reads args.strategy etc.,
    # which build_parser below provides with the same names train.py uses.
    tk = process_args(args, params)

    seed = params["seed"] if "seed" in params else 42
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")

    # Single-device validation, exactly like SI val.py: no DDP, no distributed
    # sampler, sequential read of the validate loader on one GPU. We honour an
    # explicit --devices (a single id) but never run DDP here.
    devices = _resolve_devices(args.devices, params)
    if isinstance(devices, list):
        devices = devices[:1] if devices else 1
    elif isinstance(devices, int) and devices > 1:
        devices = 1
    strategy = "auto"

    # self.ddp on the module must be False for single-device validation so the
    # per-lead-time logs do not attempt cross-rank sync_dist reductions.
    params["_lightning_ddp"] = False

    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    name = f"{params['name'] if 'name' in params else 'S2S'}_val_{seed}_{now}"
    path = os.path.join(tk["log_dir"], name) + "/"
    os.makedirs(path, exist_ok=True)
    print(f"Logging to: {path}", flush=True)

    # Prediction saving: resolve a writable predictions dir under the run dir and
    # a run_num, then inject both into the flat config for TrainModule to read.
    if args.save_predictions:
        predictions_dir = args.predictions_dir or os.path.join(path, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        params["predictions_dir"] = predictions_dir
        params["save_predictions"] = True
        if "run_num" not in params:
            params["run_num"] = args.run_num
        print(f"Saving predictions to: {predictions_dir}", flush=True)

    wandb_logger = WandbLogger(project=tk["project"], name=name, mode=tk["wandb_mode"])

    # DataModule first: its train_dataset is the normalizer for the module (the
    # normalizer supplies the *_inv_transform stats and datetime_class used by
    # the netCDF save path).
    datamodule = ClimateDataModule(params)
    model = TrainModule(params, normalizer=datamodule.train_dataset)

    # Dev bound for the smoke: --limit_val_batches caps how many validation
    # batches are scored/saved without changing the production shape (mirrors
    # train.py's --max_steps / --fast_dev_run idiom). 0.0/None -> all batches.
    extra = {}
    if args.limit_val_batches is not None:
        extra["limit_val_batches"] = args.limit_val_batches
    if args.fast_dev_run:
        extra["fast_dev_run"] = args.fast_dev_run

    trainer = L.Trainer(
        devices=devices,
        num_nodes=1,
        accelerator=tk["accelerator"],
        strategy=strategy,
        precision=tk["precision"],
        log_every_n_steps=tk["log_every_n_steps"],
        default_root_dir=path,
        logger=wandb_logger,
        use_distributed_sampler=False,  # single-device; the loader owns ordering.
        **extra,
    )

    # Graceful no-checkpoint path: pass ckpt_path only when a checkpoint is
    # supplied. Without one, validation runs on random weights (a structural
    # run), mirroring Stepper.__init__'s warn-don't-crash behaviour.
    checkpoint = params["checkpoint"] if "checkpoint" in params else None
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
    if checkpoint is not None:
        print(f"Validating from checkpoint: {checkpoint}", flush=True)
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=checkpoint)
    else:
        print(
            "No checkpoint supplied; validating on RANDOM weights "
            "(structural run -- outputs are not scientifically meaningful).",
            flush=True,
        )
        trainer.validate(model=model, datamodule=datamodule)


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the validation/inference entry point.

    Shares the training entry point's ``--yaml_config`` / ``--config`` (section)
    pair and Lightning-knob flags (so :func:`train.process_args` /
    :func:`train._resolve_devices` can be reused unchanged), and adds the
    inference-specific flags: ``--checkpoint`` (optional; random weights
    otherwise), ``--save_predictions`` / ``--predictions_dir`` / ``--run_num``
    (netCDF saving), and ``--limit_val_batches`` / ``--fast_dev_run`` (smoke
    bounds).

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Validate / run inference for the S2S Pangu-PLASIM model (Lightning)")
    parser.add_argument("--yaml_config", default="configs/test_midway.yaml",
                        help="Path to the sectioned S2S YAML config")
    parser.add_argument("--config", default="S2S", help="YAML section name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--devices", nargs="+", default=[],
                        help="GPU device ids (only the first is used; validation is single-device)")
    parser.add_argument("--accelerator", default=None, help="Lightning accelerator (gpu/cpu/auto)")
    # --strategy/--precision are consumed by train.process_args; validation always
    # forces single-device strategy="auto" regardless of --strategy.
    parser.add_argument("--strategy", default=None, help="(ignored for validation; forced to auto)")
    parser.add_argument("--precision", default=None,
                        help="Lightning precision (16-mixed/bf16-mixed/32-true)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="(unused for validation; accepted for train.process_args parity)")
    parser.add_argument("--batch_size", type=int, default=None, help="Override per-GPU batch size")
    parser.add_argument("--wandb_mode", default=None, help="wandb mode (online/offline/disabled)")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint to validate from (random weights if omitted)")
    parser.add_argument("--run_num", default="lightning", help="Run id used in netCDF filenames")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Write per-sample netCDF predictions (rank0/batch0)")
    parser.add_argument("--predictions_dir", default=None,
                        help="Directory for netCDF predictions (default: <run_dir>/predictions)")
    parser.add_argument("--limit_val_batches", type=float, default=None,
                        help="Dev: cap validation batches (int count or float fraction; smoke bound)")
    parser.add_argument("--fast_dev_run", type=int, default=0,
                        help="Dev: Lightning fast_dev_run (N batches; 0 disables)")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    # --limit_val_batches takes an int count or a float fraction; argparse parses
    # it as float, so coerce whole numbers to int for Lightning's batch count.
    if parsed.limit_val_batches is not None and parsed.limit_val_batches.is_integer():
        parsed.limit_val_batches = int(parsed.limit_val_batches)
    main(parsed)
