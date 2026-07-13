"""S2S training-throughput benchmark (Lightning port, Phase 3).

Runs the standard training loop with GPU-sync-accurate step timing, then writes
one CSV row and exits. Uses random model weights -- no checkpoint is loaded
because we are measuring speed, not quality. This mirrors the *shape* of the
SNFO template at ``$SNFO_DIR/bench.py`` but preserves S2S's **S2S_BENCH**
framework: it feeds S2S's flat :class:`utils.YParams.YParams` config to the
ported :class:`data.datamodule.ClimateDataModule` /
:class:`modules.train_module.TrainModule`, attaches the S2S-flavoured
:class:`common.bench_callback.BenchCallback` (which reads ``S2S_*`` env vars and
writes the ``v2.0/train.py::Trainer._bench_finalize`` CSV columns), and applies
``torch.compile`` to the *inner* PanguModel_Plasim (``model.model``), not the
LightningModule.

Key bench overrides applied on top of the YAML config:
    * ``wandb_mode = disabled`` -- no network traffic during the benchmark.
    * ``accumulate_grad_batches = 1`` -- every batch triggers an optimizer step
      so all step measurements are uniform.
    * ``num_sanity_val_steps = 0``, ``limit_val_batches = 0`` -- skip validation;
      we measure training throughput only.
    * No ModelCheckpoint / EMA / LRMonitor -- callbacks add synchronisation
      points that inflate step time.

DDP invariants (C2): when the config asks for DDP this builds an explicit
``DDPStrategy(find_unused_parameters=False, static_graph=True, ...)`` -- the
``find_unused_parameters=False`` / ``static_graph=True`` pair is mandatory for
S2S (the dead-module freeze in
:meth:`modules.train_module.TrainModule._get_model` makes ``static_graph`` safe)
and is kept alongside the SNFO bucket / bf16-compress knobs. The Trainer is also
built with ``use_distributed_sampler=False`` (C1) because S2S's
:func:`utils.data_loader_multifiles.get_data_loader` supplies its own sampler.

Usage:
    PYTHONPATH=v2.0:. S2S_BENCH=1 python bench.py \\
        --yaml_config configs/test_midway.yaml --config S2S --devices 0 1 2 3

Environment knobs:
    S2S_BENCH_WARMUP    warmup steps to discard   (default 20)
    S2S_BENCH_STEPS     steps to measure          (default 80)
    S2S_BENCH_CSV       output CSV path           (default bench_results.csv)
    S2S_NVTX=1          NVTX step ranges + nsys capture window
    S2S_AMP_DTYPE       fp16 (default) / bf16     (recorded in the CSV row)
    S2S_PRECISION       override Trainer precision (e.g. bf16-mixed)
    TORCH_COMPILE_MODE  torch.compile mode for model.model (off unless set)
    S2S_TORCH_COMPILE=1 force-enable torch.compile even without TORCH_COMPILE_MODE
    S2S_DDP_BUCKET_CAP_MB / S2S_DDP_BF16_COMPRESS / S2S_DDP_BUCKET_VIEW
                        DDP bucket / comm-hook knobs (mirror SNFO's defaults)

See Also:
    train.py: The production training entry point this benchmark shadows.
    common.bench_callback.BenchCallback: The S2S-flavoured timing callback.
"""

import argparse
import os
import time

import torch
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as ddp_hooks

from utils.YParams import YParams
from data.datamodule import ClimateDataModule
from modules.train_module import TrainModule
from common.bench_callback import BenchCallback, BENCH_WARMUP, BENCH_STEPS

# DDP knobs -- overridable via env so we can A/B without editing the file.
# Defaults mirror the SNFO bench template (200 MB bucket + bf16 gradient
# compression to cut per-call NCCL overhead).
DDP_BUCKET_CAP_MB = int(os.environ.get("S2S_DDP_BUCKET_CAP_MB", "200"))
DDP_BF16_COMPRESS = os.environ.get("S2S_DDP_BF16_COMPRESS", "1") == "1"
DDP_BUCKET_VIEW = os.environ.get("S2S_DDP_BUCKET_VIEW", "1") == "1"

# torch.compile knobs -- off by default (compile adds ~30-60 s to the first
# measured step; raise S2S_BENCH_WARMUP to ~40 when on). Enabled if either
# TORCH_COMPILE_MODE is set (the v2.0/train.py knob) or S2S_TORCH_COMPILE=1.
TORCH_COMPILE_MODE = os.environ.get("TORCH_COMPILE_MODE")
TORCH_COMPILE = (TORCH_COMPILE_MODE is not None) or (os.environ.get("S2S_TORCH_COMPILE") == "1")


def process_args(args, params):
    """Apply argparse overrides to the flat config (bench variant).

    Mirrors the SNFO bench template's ``process_args`` but on a flat
    :class:`utils.YParams.YParams` object. Also injects ``has_diagnostic`` /
    ``num_ensemble_members`` defensively, matching ``train.py::process_args``.

    Args:
        args: The parsed argparse namespace.
        params: The :class:`utils.YParams.YParams` config (mutated in place).
    """
    if args.seed is not None:
        params["seed"] = args.seed
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
        print(f"[bench] batch_size override: {args.batch_size}", flush=True)
    if "diagnostic_variables" in params and len(params.diagnostic_variables) > 0:
        params["has_diagnostic"] = True
    elif "has_diagnostic" not in params:
        params["has_diagnostic"] = False
    if "num_ensemble_members" not in params:
        params["num_ensemble_members"] = 1


def main(args):
    """Build the bench DataModule, module, callback and Trainer, then fit.

    Args:
        args: The parsed argparse namespace (see :func:`build_parser`).
    """
    cfg = os.path.abspath(args.yaml_config)
    print(f"[bench] config: {cfg} (section {args.config})", flush=True)
    params = YParams(cfg, args.config)
    process_args(args, params)

    seed_everything(params["seed"] if "seed" in params else 42)
    torch.set_float32_matmul_precision("high")

    # Bench overrides.
    params["wandb_mode"] = "disabled"
    params["accumulate_grad_batches"] = 1

    # Precision: S2S_PRECISION overrides the config (ablation runs); else the
    # config's precision; else "16-mixed" (S2S's fp16 default mapped to Lightning).
    precision = os.environ.get("S2S_PRECISION")
    if precision:
        print(f"[bench] S2S_PRECISION override: {precision}", flush=True)
    elif "precision" in params:
        precision = params["precision"]
    else:
        precision = "16-mixed"

    devices = [int(d) for d in args.devices] if args.devices else (
        params["devices"] if "devices" in params else 1
    )
    n_gpus = len(devices) if isinstance(devices, (list, tuple)) else int(devices)
    batch_per_gpu = params["batch_size"]
    strategy_name = args.strategy or (params["strategy"] if "strategy" in params else "auto")
    # C2 / P2-1: DDP is decided by the strategy name alone (not n_gpus), so a
    # single-device --strategy ddp still takes the explicit DDPStrategy path with
    # static_graph=True rather than the bare "ddp" string (static_graph=False).
    # Mirrors train.py::_is_ddp.
    ddp = strategy_name in ("ddp", "ddp_find_unused_parameters_true")

    # C6: set _lightning_ddp BEFORE constructing TrainModule.
    params["_lightning_ddp"] = ddp

    run_num = f"bench_{int(time.time())}"

    datamodule = ClimateDataModule(params)
    model = TrainModule(params, normalizer=datamodule.train_dataset)

    # Compile the inner PanguModel_Plasim, not the LightningModule, so the
    # callbacks and scheduler glue stay eager and Lightning's autocast wrapping
    # of the step is preserved. Off unless TORCH_COMPILE_MODE / S2S_TORCH_COMPILE.
    if TORCH_COMPILE:
        mode = TORCH_COMPILE_MODE or "default"
        print(f"[bench] torch.compile(model.model) mode={mode}", flush=True)
        model.model = torch.compile(model.model, mode=mode)

    bench_cb = BenchCallback(
        n_gpus=n_gpus,
        batch_per_gpu=batch_per_gpu,
        config_path=cfg,
        run_num=run_num,
    )

    # BenchCallback stops the trainer after BENCH_STEPS measured steps; +5 is a
    # small buffer so SLURM doesn't kill us mid-finalize.
    max_steps = BENCH_WARMUP + BENCH_STEPS + 5

    wandb_logger = WandbLogger(
        project=params["project"] if "project" in params else "Pangu-S2S",
        name=run_num,
        mode="disabled",
    )

    # C2: explicit DDPStrategy with the S2S invariants (find_unused_parameters=
    # False, static_graph=True) kept alongside the SNFO bucket / bf16 knobs.
    if ddp:
        ddp_kwargs = {
            "find_unused_parameters": False,
            "static_graph": True,
            "bucket_cap_mb": DDP_BUCKET_CAP_MB,
            "gradient_as_bucket_view": DDP_BUCKET_VIEW,
        }
        if DDP_BF16_COMPRESS:
            ddp_kwargs["ddp_comm_hook"] = ddp_hooks.bf16_compress_hook
        print(f"[bench] DDPStrategy kwargs: {ddp_kwargs}", flush=True)
        strategy = DDPStrategy(**ddp_kwargs)
    else:
        strategy = strategy_name

    trainer = L.Trainer(
        devices=devices,
        num_nodes=params["num_nodes"] if "num_nodes" in params else 1,
        accelerator=args.accelerator or (params["accelerator"] if "accelerator" in params else "gpu"),
        strategy=strategy,
        precision=precision,
        max_steps=max_steps,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        callbacks=[bench_cb],
        logger=wandb_logger,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        use_distributed_sampler=False,  # C1: S2S builds its own sampler.
        log_every_n_steps=max_steps + 1,  # suppress Lightning's internal logging
    )

    trainer.fit(model=model, datamodule=datamodule)


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the benchmark entry point.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(description="S2S training throughput benchmark (Lightning)")
    parser.add_argument("--yaml_config", default="configs/test_midway.yaml",
                        help="Path to the sectioned S2S YAML config")
    parser.add_argument("--config", default="S2S", help="YAML section name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--devices", nargs="+", default=[], help="GPU device ids")
    parser.add_argument("--accelerator", default=None, help="Lightning accelerator (gpu/cpu/auto)")
    parser.add_argument("--strategy", default=None, help="Lightning strategy (ddp/auto)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override per-GPU batch size (useful for low-memory GPUs)")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
