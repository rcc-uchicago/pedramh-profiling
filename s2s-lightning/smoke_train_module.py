"""Phase-2 smoke: 2-step Lightning fit of TrainModule on the real HDF5 data.

Builds ``ClimateDataModule`` + ``TrainModule`` from ``v2.0/config/test.yaml``
and runs a 2-step ``L.Trainer.fit`` on one GPU, asserting the training loss is
finite. Prints ``SMOKE_OK`` on success. Run under
``PYTHONPATH=v2.0/:<repo-root>`` -- ``v2.0/`` resolves ``utils``/``networks`` and
the repo root resolves ``data``/``modules`` -- via the nested sbatch wrapper
``midway_smoke_train_module.sh``.
"""

import math
import os
import sys

import lightning as L
import torch

from utils.YParams import YParams
from data.datamodule import ClimateDataModule
from modules.train_module import TrainModule


class _LossProbe(L.Callback):
    """Capture each step's training loss and assert finiteness.

    Attributes:
        losses (list[float]): The per-step training losses recorded so far.
    """

    def __init__(self):
        """Initialise the empty loss-history list."""
        self.losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Record the step's training loss and raise if it is non-finite.

        Args:
            trainer: The Lightning ``Trainer`` (unused).
            pl_module: The Lightning module being trained (unused).
            outputs: The ``training_step`` output (a loss tensor, or a dict with a
                ``"loss"`` key).
            batch: The current batch (unused).
            batch_idx: Lightning's batch index (used only for the log line).

        Raises:
            RuntimeError: If the recorded loss is NaN or infinite.
        """
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        val = float(loss.detach().cpu())
        self.losses.append(val)
        print(f"[probe] step {batch_idx} train loss = {val:.6f}", flush=True)
        if not math.isfinite(val):
            raise RuntimeError(f"NON-FINITE training loss at step {batch_idx}: {val}")


def main():
    """Run the Phase-2 2-step ``fit`` smoke and print ``SMOKE_OK`` on success.

    Builds :class:`data.datamodule.ClimateDataModule` +
    :class:`modules.train_module.TrainModule` from ``v2.0/config/test.yaml``
    (overriding ``batch_size=1`` so the ensemble/CRPS path fits a single GPU) and
    runs a 2-step ``L.Trainer.fit`` at ``precision="16-mixed"`` with a
    :class:`_LossProbe` callback asserting each step's loss is finite. Exits 2
    (via ``sys.exit``) if no step ran or any loss was non-finite, else prints
    ``SMOKE_OK`` (the commit gate token).

    Returns:
        None: Outcome is signalled by the printed token and the process exit code
        (0 on success, 2 on a recorded/finiteness failure).
    """
    # Resolve the config relative to THIS file, not cwd, so the smoke works from
    # any directory in the monorepo (s2s/v2.0/ is a sibling of s2s-lightning/).
    cfg = os.environ.get("S2S_SMOKE_CONFIG") or os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "s2s", "v2.0", "config", "test.yaml")
    )
    print(f"Loading config: {cfg} (section S2S)", flush=True)
    params = YParams(cfg, "S2S")

    # Single-GPU smoke: no process group, so disable DDP-style sync_dist logging.
    params["_lightning_ddp"] = False

    # Smoke-only batch-size override. test.yaml sets batch_size=4 and
    # num_ensemble_members=4, so the effective batch through the 79M-param model
    # is 4*4=16, which OOMs a single 93 GiB H100 (first attempt, job 51310287).
    # Drop to batch_size=1 -> effective batch 1*4=4, keeping the ensemble/CRPS
    # path exercised (B=1, ens=4) while cutting activations ~4x. Must be set
    # BEFORE building the datamodule, which reads batch_size at construction.
    params["batch_size"] = 1
    print(f"[smoke] batch_size override = {params['batch_size']} "
          f"(num_ensemble_members={params['num_ensemble_members']})", flush=True)

    print("Instantiating ClimateDataModule...", flush=True)
    dm = ClimateDataModule(params)
    print(
        f"train_dataset len={len(dm.train_dataset)} "
        f"val_dataset len={len(dm.val_dataset)}",
        flush=True,
    )

    print("Instantiating TrainModule (normalizer=train_dataset)...", flush=True)
    module = TrainModule(params, normalizer=dm.train_dataset)
    n_params = sum(p.numel() for p in module.model.parameters() if p.requires_grad)
    print(f"model trainable params = {n_params:,}", flush=True)

    # Precision: S2S defaults to fp16 + GradScaler (S2S_AMP_DTYPE=fp16); under
    # Lightning that maps to "16-mixed" (Lightning owns the GradScaler).
    precision = "16-mixed"
    print(f"precision = {precision}", flush=True)

    probe = _LossProbe()
    trainer = L.Trainer(
        max_steps=2,
        devices=1,
        accelerator="gpu",
        precision=precision,
        use_distributed_sampler=False,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        callbacks=[probe],
    )

    print("Starting trainer.fit (max_steps=2)...", flush=True)
    trainer.fit(module, datamodule=dm)

    if not probe.losses:
        print("SMOKE_FAIL: no training steps were recorded", flush=True)
        sys.exit(2)
    if not all(math.isfinite(x) for x in probe.losses):
        print(f"SMOKE_FAIL: non-finite loss in {probe.losses}", flush=True)
        sys.exit(2)

    print(f"recorded train losses = {probe.losses}", flush=True)
    print("SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
