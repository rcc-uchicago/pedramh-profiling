"""SI training throughput benchmark.

Runs the standard training loop with GPU-sync-accurate step timing, then
writes one CSV row and exits.  Uses random model weights — no checkpoint is
loaded because we are measuring speed, not quality.

Key bench overrides applied on top of the YAML config
──────────────────────────────────────────────────────
    accumulate_grad_batches = 1
        Forces every batch to trigger an optimizer step so all step
        measurements are uniform.  The production SI_midway config uses 2;
        log the override in any report that compares these numbers.

    wandb_mode = disabled
        No network traffic during the benchmark.

    num_sanity_val_steps = 0, limit_val_batches = 0
        Skip all validation — we are measuring training throughput only.

    No ModelCheckpoint, no EMAWeightAveraging, no LRMonitor.
        Callbacks add synchronisation points that inflate step time.

Usage
─────
    python bench.py --config configs/SI_midway.yaml

    # Override GPU list (default: from config)
    python bench.py --config configs/SI_midway.yaml --devices 0 1 2 3

Environment knobs
──────────────────
    SI_BENCH_WARMUP   warmup steps to discard   (default 20)
    SI_BENCH_STEPS    steps to measure           (default 80)
    SI_BENCH_CSV      output CSV path            (default bench_results.csv)
    SI_NVTX=1         NVTX step ranges + nsys capture window
"""

import argparse
import os
import time

import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as ddp_hooks
import lightning as L

from common.utils import get_yaml
from data.datamodule import ClimateDataModule
from modules.train_module import TrainModule
from modules.ae_module import AutoencoderModule
from common.bench_callback import BenchCallback, BENCH_WARMUP, BENCH_STEPS

BENCH_CSV = os.environ.get("SI_BENCH_CSV", "bench_results.csv")

# DDP knobs — overridable via env so we can A/B without editing the file.
# Defaults are tuned against the 2026-05-20 NVTX profile (118 NCCL calls/step
# with the 25 MB Lightning default → bucket bump + bf16 compression).
DDP_BUCKET_CAP_MB    = int(os.environ.get("SI_DDP_BUCKET_CAP_MB", "200"))
DDP_BF16_COMPRESS    = os.environ.get("SI_DDP_BF16_COMPRESS", "1") == "1"
DDP_BUCKET_VIEW      = os.environ.get("SI_DDP_BUCKET_VIEW", "1") == "1"

# torch.compile knobs — off by default because compile times add ~30–60 s per
# rank to the first measured step.  Raise SI_BENCH_WARMUP to ~40 when on.
TORCH_COMPILE        = os.environ.get("SI_TORCH_COMPILE", "0") == "1"
TORCH_COMPILE_MODE   = os.environ.get("SI_COMPILE_MODE", "default")


def process_args(args, config):
    modelconfig = config["model"]
    trainconfig = config["training"]
    dataconfig  = config["data"]
    if args.devices:
        trainconfig["devices"] = [int(d) for d in args.devices]
    if args.seed is not None:
        trainconfig["seed"] = args.seed
    if args.batch_size is not None:
        dataconfig["batch_size"] = args.batch_size
        print(f"[bench] batch_size override: {args.batch_size}")
    return config, modelconfig, trainconfig, dataconfig


def main(args):
    config = get_yaml(args.config)
    config, modelconfig, trainconfig, dataconfig = process_args(args, config)

    seed_everything(trainconfig.get("seed", 42))
    torch.set_float32_matmul_precision("high")

    # Bench overrides.
    trainconfig["wandb_mode"]              = "disabled"
    trainconfig["accumulate_grad_batches"] = 1
    trainconfig["num_sanity_val_steps"]    = 0
    trainconfig.pop("checkpoint",          None)
    trainconfig.pop("partial_checkpoint",  None)

    # Allow precision to be overridden from the environment for ablation runs
    # without editing the config (e.g. export SI_PRECISION=bf16-mixed).
    precision_override = os.environ.get("SI_PRECISION")
    if precision_override:
        trainconfig["precision"] = precision_override
        print(f"[bench] SI_PRECISION override: {precision_override}")

    devices     = trainconfig["devices"]
    n_gpus      = len(devices) if isinstance(devices, list) else int(devices)
    batch_per_gpu = dataconfig["batch_size"]

    run_num = f"bench_{int(time.time())}"

    datamodule = ClimateDataModule(dataconfig=dataconfig)

    autoencoder = dataconfig.get("autoencoder", False)
    if autoencoder:
        model = AutoencoderModule(config, normalizer=datamodule.train_dataset)
    else:
        model = TrainModule(config, normalizer=datamodule.train_dataset)

    # Compile the inner model (DiT / UNet), not the LightningModule, so the
    # callbacks and scheduler glue stay eager.  Lightning's autocast wrapping
    # is preserved because torch.compile inspects the wrapped forward.
    if TORCH_COMPILE:
        print(f"[bench] torch.compile mode={TORCH_COMPILE_MODE}")
        model.model = torch.compile(model.model, mode=TORCH_COMPILE_MODE)

    bench_cb = BenchCallback(
        n_gpus=n_gpus,
        batch_per_gpu=batch_per_gpu,
        config_path=args.config,
        run_num=run_num,
    )

    # BenchCallback stops the trainer after BENCH_STEPS measured steps.
    # The +5 is a small buffer so SLURM doesn't kill us mid-finalize.
    max_steps = BENCH_WARMUP + BENCH_STEPS + 5

    wandb_logger = WandbLogger(
        project=trainconfig.get("project", "si_bench"),
        name=run_num,
        mode="disabled",
    )

    # Build an explicit DDPStrategy when the config asks for DDP so we can
    # set bucket size, gradient_as_bucket_view, and a bf16 gradient-compression
    # comm hook — these target the per-call NCCL overhead found in the
    # 2026-05-20 NVTX profile.
    strategy_cfg = trainconfig["strategy"]
    if strategy_cfg == "ddp":
        ddp_kwargs = {
            "bucket_cap_mb": DDP_BUCKET_CAP_MB,
            "gradient_as_bucket_view": DDP_BUCKET_VIEW,
        }
        if DDP_BF16_COMPRESS:
            ddp_kwargs["ddp_comm_hook"] = ddp_hooks.bf16_compress_hook
        print(f"[bench] DDPStrategy kwargs: {ddp_kwargs}")
        strategy_cfg = DDPStrategy(**ddp_kwargs)

    trainer = L.Trainer(
        devices          = trainconfig["devices"],
        num_nodes        = trainconfig.get("num_nodes", 1),
        accelerator      = trainconfig["accelerator"],
        strategy         = strategy_cfg,
        precision        = trainconfig["precision"],
        max_steps        = max_steps,
        limit_val_batches= 0,
        callbacks        = [bench_cb],
        logger           = wandb_logger,
        enable_progress_bar  = False,
        enable_model_summary = False,
        log_every_n_steps    = max_steps + 1,  # suppress Lightning's internal logging
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SI training throughput benchmark")
    parser.add_argument("--config",     required=True, help="Path to YAML config")
    parser.add_argument("--seed",       type=int, default=None)
    parser.add_argument("--devices",    nargs="+", default=[], help="GPU device IDs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size from config (useful for low-memory GPUs)")
    args = parser.parse_args()
    main(args)
