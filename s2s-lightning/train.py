"""Lightning training entry point for the S2S Pangu-PLASIM model (Phase 3).

This is the production training entry point of the S2S -> PyTorch Lightning
port. It mirrors the *shape* of the SNFO template at ``$SNFO_DIR/train.py``
(argparse -> config -> callbacks -> :class:`lightning.Trainer` ->
``trainer.fit(model, datamodule)``) but feeds it S2S's flat
:class:`utils.YParams.YParams` config rather than SNFO's nested
``model:`` / ``data:`` / ``training:`` dict. The same flat ``params`` object is
handed to both :class:`data.datamodule.ClimateDataModule` and
:class:`modules.train_module.TrainModule`, matching what those committed modules
expect (``params.lr``, ``params.data_dir``, ``params.strategy``,
``params["_lightning_ddp"]``, ...).

Several hard correctness contracts of the port are enforced here rather than in
the modules, because they are Trainer-level concerns:

* **No injected sampler.** S2S's
  :func:`utils.data_loader_multifiles.get_data_loader` builds its own sampler,
  so the ``Trainer`` is constructed with ``use_distributed_sampler=False`` and
  the per-epoch ``set_epoch`` is restored via
  :class:`common.set_epoch_callback.SetEpochCallback`.
* **DDP invariants.** The DDP path uses
  ``DDPStrategy(find_unused_parameters=False, static_graph=True)`` -- safe
  because :meth:`modules.train_module.TrainModule._get_model` froze the dead
  modules. There is no manual ``dist.init_process_group``; Lightning's strategy
  owns the process group.
* **Precision.** The Trainer's ``precision=`` setting owns AMP (``"16-mixed"``
  reproduces S2S's fp16 default); no hand-rolled autocast / GradScaler.

The Trainer knobs (devices / accelerator / strategy / precision / ...) are
sourced from argparse plus optional flat keys in the YAML and built-in defaults
(there is no ``training:`` block in the S2S config). The ``utils.*`` /
``networks.*`` imports reused transitively by the modules resolve only when
``v2.0/`` is on ``PYTHONPATH`` (``PYTHONPATH=v2.0/:<repo-root>``).

Usage:
    PYTHONPATH=v2.0:. python train.py --yaml_config configs/test_midway.yaml \\
        --config S2S --devices 0 1 2 3

See Also:
    bench.py: The throughput-benchmark sibling entry point.
    modules.train_module.TrainModule: The LightningModule wrapping
        PanguModel_Plasim.
    data.datamodule.ClimateDataModule: The DataModule wrapping the HDF5 loaders.
"""

import argparse
import os
from datetime import datetime

import torch
from torch.optim.swa_utils import get_ema_avg_fn

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from utils.YParams import YParams
from data.datamodule import ClimateDataModule
from modules.train_module import TrainModule
from common.set_epoch_callback import SetEpochCallback

# WeightAveraging is the SNFO EMA base class, but it only exists in newer
# Lightning (>= 2.6); the pinned LPORT_ENV is 2.5.0.post0 and lacks it. Guard
# the import so train.py stays importable here, and only define
# EMAWeightAveraging when the base class is present. Building it when absent
# raises a clear error (env-unification with SNFO's newer Lightning is a Phase-5
# concern; until then EMA is opt-in via ema_decay > 0 and off by default in
# configs/test_midway.yaml).
try:
    from lightning.pytorch.callbacks import WeightAveraging as _WeightAveraging
except ImportError:  # Lightning < 2.6 (the pinned LPORT_ENV)
    _WeightAveraging = None


if _WeightAveraging is not None:

    class EMAWeightAveraging(_WeightAveraging):
        """Exponential-moving-average weight averaging (always-on update).

        Thin wrapper over ``lightning.pytorch.callbacks.WeightAveraging`` that
        uses an EMA averaging function and updates on every step. Mirrors the
        SNFO template's ``EMAWeightAveraging`` so the two codebases share
        callback shape. Defined only when ``WeightAveraging`` is importable (see
        the module-level guard); on the pinned LPORT_ENV (Lightning 2.5) it is
        absent and :func:`_build_ema_callback` raises if EMA is requested.

        Args:
            decay: EMA decay factor passed to
                :func:`torch.optim.swa_utils.get_ema_avg_fn`.
        """

        def __init__(self, decay: float = 0.99) -> None:
            """Build the EMA averaging callback.

            Args:
                decay: EMA decay factor.
            """
            super().__init__(avg_fn=get_ema_avg_fn(decay=decay))

        def should_update(self, step_idx=None, epoch_idx=None) -> bool:
            """Always update the averaged weights.

            Args:
                step_idx (optional): Global step index (unused).
                epoch_idx (optional): Epoch index (unused).

            Returns:
                bool: Always ``True``.
            """
            return True


def _build_ema_callback(decay: float):
    """Return an EMA weight-averaging callback, or raise if unsupported.

    Args:
        decay: EMA decay factor (``> 0`` to enable EMA).

    Returns:
        EMAWeightAveraging: The configured EMA callback.

    Raises:
        RuntimeError: If ``WeightAveraging`` is unavailable in the installed
            Lightning (the pinned LPORT_ENV, Lightning 2.5) but EMA was
            requested via ``ema_decay > 0``.
    """
    if _WeightAveraging is None:
        raise RuntimeError(
            "ema_decay > 0 requested but lightning.pytorch.callbacks.WeightAveraging "
            "is unavailable in this Lightning (needs >= 2.6; LPORT_ENV is 2.5). "
            "Set ema_decay=0 to disable EMA, or upgrade Lightning (Phase-5 env "
            "unification with SNFO)."
        )
    return EMAWeightAveraging(decay)


def _resolve_devices(args_devices, params):
    """Resolve the device specification from argparse and the config.

    Args:
        args_devices: The ``--devices`` argparse value (a list of strings, the
            GPU ids; empty when not passed).
        params: The :class:`utils.YParams.YParams` config (may carry a
            ``devices`` key).

    Returns:
        The device spec for ``L.Trainer(devices=...)``: a list of ints when
        ``--devices`` is given, else the config's ``devices`` value, else ``1``.
    """
    if args_devices:
        return [int(d) for d in args_devices]
    if "devices" in params:
        return params["devices"]
    return 1


def _is_ddp(strategy_name) -> bool:
    """Decide whether the run is a DDP run (drives ``_lightning_ddp``).

    Determined by the strategy name **alone**, not the device count: a
    ``--strategy ddp`` run must take the explicit
    ``DDPStrategy(find_unused_parameters=False, static_graph=True)`` path (C2)
    even on a single device, otherwise it silently falls through to the bare
    ``"ddp"`` string strategy (``static_graph=False``) and the C2 invariant is
    bypassed. A single-device default run uses ``--strategy auto`` (or no
    ``strategy`` in the config), which is not DDP, so the existing 1-GPU path is
    unaffected.

    Args:
        strategy_name: The resolved strategy string (e.g. ``"ddp"`` or
            ``"auto"``).

    Returns:
        bool: ``True`` when the strategy selects DDP, ``False`` otherwise.
        ``self.ddp`` on the module (which drives ``sync_dist`` on logging) is set
        from this.
    """
    return strategy_name in ("ddp", "ddp_find_unused_parameters_true")


def process_args(args, params):
    """Apply argparse overrides to the flat config and resolve Trainer knobs.

    Mirrors the SNFO template's ``process_args`` but operates on a flat
    :class:`utils.YParams.YParams` object (S2S has no nested ``training:``
    block). Argparse values take precedence over the config; the config's flat
    keys (or built-in defaults) fill the rest. Also injects ``has_diagnostic`` /
    ``num_ensemble_members`` defensively, exactly as ``v2.0/train.py::__main__``
    did, so the modules see the same derived keys.

    Args:
        args: The parsed argparse namespace.
        params: The :class:`utils.YParams.YParams` config (mutated in place).

    Returns:
        dict: A ``trainer_kwargs`` mapping of resolved Trainer knobs
        (``devices``, ``accelerator``, ``strategy_name``, ``precision``,
        ``max_epochs``, ``check_val_every_n_epoch``, ``log_every_n_steps``,
        ``ema_decay``, ``num_sanity_val_steps``, ``accumulate_grad_batches``,
        ``num_nodes``, ``wandb_mode``, ``project``, ``log_dir``).
    """
    if args.seed is not None:
        params["seed"] = args.seed
    if args.epochs is not None and args.epochs > 0:
        params["max_epochs"] = args.epochs
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
        print(f"[train] batch_size override: {args.batch_size}", flush=True)
    if args.wandb_mode is not None:
        params["wandb_mode"] = args.wandb_mode

    # has_diagnostic / num_ensemble_members: same defensive derivation as
    # v2.0/train.py::__main__.
    if "diagnostic_variables" in params and len(params.diagnostic_variables) > 0:
        params["has_diagnostic"] = True
    elif "has_diagnostic" not in params:
        params["has_diagnostic"] = False
    if "num_ensemble_members" not in params:
        params["num_ensemble_members"] = 1

    devices = _resolve_devices(args.devices, params)
    strategy_name = args.strategy or (params["strategy"] if "strategy" in params else "auto")
    precision = args.precision or (params["precision"] if "precision" in params else "16-mixed")

    trainer_kwargs = {
        "devices": devices,
        "accelerator": args.accelerator or (params["accelerator"] if "accelerator" in params else "gpu"),
        "strategy_name": strategy_name,
        "precision": precision,
        "max_epochs": params["max_epochs"] if "max_epochs" in params else 1,
        "check_val_every_n_epoch": params["check_val_every_n_epoch"] if "check_val_every_n_epoch" in params else 1,
        "log_every_n_steps": params["log_every_n_steps"] if "log_every_n_steps" in params else 50,
        "ema_decay": params["ema_decay"] if "ema_decay" in params else 0.0,
        "num_sanity_val_steps": params["num_sanity_val_steps"] if "num_sanity_val_steps" in params else 1,
        "accumulate_grad_batches": params["accumulate_grad_batches"] if "accumulate_grad_batches" in params else 1,
        "num_nodes": params["num_nodes"] if "num_nodes" in params else 1,
        "wandb_mode": params["wandb_mode"] if "wandb_mode" in params else "online",
        "project": params["project"] if "project" in params else "Pangu-S2S",
        "log_dir": params["log_dir"] if "log_dir" in params else "results/lightning/",
    }
    return trainer_kwargs


def main(args):
    """Build the DataModule, module, callbacks and Trainer, then fit.

    Args:
        args: The parsed argparse namespace (see :func:`build_parser`).
    """
    cfg = os.path.abspath(args.yaml_config)
    print(f"Loading config: {cfg} (section {args.config})", flush=True)
    params = YParams(cfg, args.config)
    tk = process_args(args, params)

    seed = params["seed"] if "seed" in params else 42
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")

    devices = tk["devices"]
    strategy_name = tk["strategy_name"]
    ddp = _is_ddp(strategy_name)

    # C6: set _lightning_ddp BEFORE constructing TrainModule so self.ddp (and
    # thus sync_dist on logging) matches the actual strategy.
    params["_lightning_ddp"] = ddp

    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    name = f"{params['name'] if 'name' in params else 'S2S'}_{seed}_{now}"
    path = os.path.join(tk["log_dir"], name) + "/"
    os.makedirs(path, exist_ok=True)
    print(f"Logging to: {path}", flush=True)

    wandb_logger = WandbLogger(project=tk["project"], name=name, mode=tk["wandb_mode"])

    # DataModule first: its train_dataset is the normalizer for the module (C5).
    datamodule = ClimateDataModule(params)
    model = TrainModule(params, normalizer=datamodule.train_dataset)

    epoch_checkpoint = ModelCheckpoint(
        dirpath=path,
        filename="model_{epoch:02d}",
        every_n_epochs=1,
        save_top_k=-1,
    )
    last_checkpoint = ModelCheckpoint(
        dirpath=path,
        every_n_train_steps=100,
        save_last=True,
        save_top_k=0,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # C4: restore per-epoch set_epoch on S2S's own DistributedSampler (Lightning
    # skips it under use_distributed_sampler=False).
    callbacks = [epoch_checkpoint, last_checkpoint, lr_monitor, SetEpochCallback()]
    if tk["ema_decay"] and tk["ema_decay"] > 0:
        callbacks.append(_build_ema_callback(tk["ema_decay"]))

    # C2: DDP via DDPStrategy(find_unused_parameters=False, static_graph=True).
    # The dead-module freeze in TrainModule._get_model keeps static_graph safe.
    if ddp:
        strategy = DDPStrategy(find_unused_parameters=False, static_graph=True)
    else:
        strategy = strategy_name  # "auto" / single-device string strategy

    # Dev bounds: --max_steps / --fast_dev_run keep a smoke short without
    # changing the production shape. Default max_steps=None -> run max_epochs.
    extra = {}
    if args.fast_dev_run:
        extra["fast_dev_run"] = args.fast_dev_run
    if args.max_steps is not None:
        extra["max_steps"] = args.max_steps

    trainer = L.Trainer(
        devices=devices,
        num_nodes=tk["num_nodes"],
        accelerator=tk["accelerator"],
        strategy=strategy,
        precision=tk["precision"],
        max_epochs=tk["max_epochs"],
        check_val_every_n_epoch=tk["check_val_every_n_epoch"],
        log_every_n_steps=tk["log_every_n_steps"],
        num_sanity_val_steps=tk["num_sanity_val_steps"],
        accumulate_grad_batches=tk["accumulate_grad_batches"],
        default_root_dir=path,
        callbacks=callbacks,
        logger=wandb_logger,
        use_distributed_sampler=False,  # C1: S2S builds its own sampler.
        **extra,
    )

    checkpoint = params["checkpoint"] if "checkpoint" in params else None
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
    if checkpoint is not None:
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint)
    else:
        trainer.fit(model=model, datamodule=datamodule)


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the training entry point.

    The S2S-style ``--yaml_config`` / ``--config`` (section) pair selects the
    sectioned YAML; the Lightning knobs (``--devices`` / ``--accelerator`` /
    ``--strategy`` / ``--precision``) override the config; ``--max_steps`` /
    ``--fast_dev_run`` bound a smoke run without changing the production shape.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(description="Train the S2S Pangu-PLASIM model (Lightning)")
    parser.add_argument("--yaml_config", default="configs/test_midway.yaml",
                        help="Path to the sectioned S2S YAML config")
    parser.add_argument("--config", default="S2S", help="YAML section name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--devices", nargs="+", default=[], help="GPU device ids")
    parser.add_argument("--accelerator", default=None, help="Lightning accelerator (gpu/cpu/auto)")
    parser.add_argument("--strategy", default=None, help="Lightning strategy (ddp/auto)")
    parser.add_argument("--precision", default=None,
                        help="Lightning precision (16-mixed/bf16-mixed/32-true)")
    parser.add_argument("--epochs", type=int, default=None, help="Override max_epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override per-GPU batch size")
    parser.add_argument("--wandb_mode", default=None, help="wandb mode (online/offline/disabled)")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Dev: cap total optimizer steps (smoke bound)")
    parser.add_argument("--fast_dev_run", type=int, default=0,
                        help="Dev: Lightning fast_dev_run (N batches; 0 disables)")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
