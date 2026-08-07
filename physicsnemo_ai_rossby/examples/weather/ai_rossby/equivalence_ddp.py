# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-rank equivalence capture — the instrument ``equivalence.py`` cannot be.

``equivalence.py`` captures at **world size 1** and refuses anything else, by
design: §4.1 wants a deterministic baseline, and multi-rank gradient reduction
is not deterministic. Its own error message says *"add a separate 4-GPU baseline
when the change touches DDP"*. This is that.

Why ZeRO forces the issue
-------------------------
ZeRO Stage 1 is a **no-op at world size 1** — ``_wrap_zero`` falls back to plain
AdamW — so the single-rank gate can say nothing about it. Its entire effect is
the changed gradient reduction: DDP all-reduces then every rank runs the full
optimizer, whereas ZeRO reduce-scatters, each rank updates its shard, and the
parameters are all-gathered. Same arithmetic in exact math; different summation
ORDER in floating point, and float addition is not associative.

So a bitwise test is the wrong test — it would fail on a correct implementation.

The method: compare against a noise floor
-----------------------------------------
Run plain DDP **twice** at one seed. Any difference between those two runs is
the reduction's own run-to-run nondeterminism — the noise floor. Then run ZeRO
and ask whether its deviation from DDP sits **inside** that floor. If it does,
ZeRO is doing what DDP does, to the precision DDP itself offers. If ZeRO deviates
by much more, that is signal, not float noise.

A single DDP-vs-ZeRO comparison proves nothing, because DDP is not reproducible
against *itself*. That control is the whole point.

Two things this deliberately does differently from ``equivalence.py``:

* ``distributed=True`` on the datapipe. With ``distributed=False`` every rank
  would draw identical batches, the all-reduce would sum four equal values, and
  reduction order could not matter — a degenerate test that always passes.
* The model **is** DDP-wrapped. Unwrapped, no gradient sync happens at all and
  the thing under test never runs.

Emits the same JSON schema as ``equivalence.py`` (plus ``world_size`` and
``use_zero_optimizer``), so ``compare_baselines.py`` reads it unchanged.

Usage (COMPUTE NODE, 4 ranks)::

    CUBLAS_WORKSPACE_CONFIG=:4096:8 AI_ROSSBY_EQUIV_TAG=ddp_run1 \\
      python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \\
      equivalence_ddp.py <overrides>

PASS = ``EQUIV_DDP_CAPTURE_OK``. Rank 0 writes; other ranks exit silently.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel

from physicsnemo.distributed import DistributedManager

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
    train_step,
)

STEPS = int(os.environ.get("AI_ROSSBY_EQUIV_STEPS", "20"))
TAG = os.environ.get("AI_ROSSBY_EQUIV_TAG", "ddp_baseline")
MODE = os.environ.get("AI_ROSSBY_EQUIV_MODE", "train")
OUT_DIR = os.environ.get(
    "AI_ROSSBY_EQUIV_DIR",
    str(Path(__file__).resolve().parents[4] / "baselines" / "ai_rossby_sfno"),
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        print("ERROR CUBLAS_WORKSPACE_CONFIG_UNSET — must be set in the ENVIRONMENT "
              "before torch initialises cuBLAS; from Python it is too late.")
        sys.exit(2)

    # Determinism everywhere it is achievable. Reduction order is the one thing
    # that stays nondeterministic — which is exactly what is being measured.
    torch.backends.cudnn.benchmark = False
    # NOT use_deterministic_algorithms(True): several backward kernels here have
    # no deterministic implementation, and forcing it raises. The noise-floor
    # control absorbs that residual nondeterminism instead of pretending it away.

    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.world_size < 2:
        print(f"ERROR EQUIV_DDP_REQUIRES_MULTIRANK (got {dist.world_size}) — this "
              "tool exists to measure the DDP/ZeRO reduction. At world size 1 "
              "ZeRO is a no-op; use equivalence.py instead.")
        sys.exit(2)

    seed = int(cfg.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cfg_train = cfg.training
    stages = list(cfg_train.stages)
    if int(stages[0].get("unroll_steps", 1)) != 1:
        print("ERROR EQUIV_UNROLL_UNSUPPORTED — capture the single-step path.")
        sys.exit(2)

    # shuffle=False so the sample ORDER is fixed, but distributed=True so the
    # ranks hold DIFFERENT shards — without which the reduction is degenerate.
    datapipe = build_datapipe(
        cfg,
        zarr_path=_resolve_path(cfg.dataset.zarr_path),
        distributed=True,
        device=dist.device,
        shuffle=False,
        seed=seed,
        unroll_steps=1,
    )
    inner_model = build_model(cfg.model).to(dist.device)
    n_params = sum(p.numel() for p in inner_model.parameters())
    model = DistributedDataParallel(
        inner_model,
        device_ids=[dist.local_rank],
        output_device=dist.local_rank,
        find_unused_parameters=False,
    )

    loss_fn = build_loss(cfg).to(dist.device)
    optimizer = make_optimizer(inner_model, _flatten_optimizer_cfg(cfg_train.optimizer))
    use_zero = bool(cfg_train.optimizer.get("use_zero_optimizer", False))
    scheduler = make_scheduler(
        optimizer,
        _flatten_scheduler_cfg(stages[0].scheduler, lr=float(cfg_train.optimizer.lr),
                               steps_per_epoch=STEPS, num_epochs=1),
        total_steps=STEPS,
    )
    amp_dtype = _resolve_amp_dtype(cfg_train.get("amp", None))
    has_diagnostic = inner_model.has_diagnostic
    epsilon_factor = float(cfg_train.get("epsilon_factor", 0.0))

    traj: list[dict] = []
    for batch in datapipe:
        if len(traj) >= STEPS:
            break
        losses = train_step(
            model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
            batch=batch, has_diagnostic=has_diagnostic,
            vae_kl_weight=float(cfg.loss.get("vae_kl_weight", 0.0)),
            amp_dtype=amp_dtype, grad_scaler=None,
            epsilon_factor=epsilon_factor,
        )
        # Global, not rank-local: a per-rank loss would differ between runs
        # simply because the ranks hold different shards, drowning the signal.
        vec = torch.stack([losses[k].detach().float().reshape(())
                           for k in ("loss", "surface", "upper_air", "diagnostic")])
        torch.distributed.all_reduce(vec, op=torch.distributed.ReduceOp.AVG)
        gnorm = torch.sqrt(sum(p.grad.double().square().sum()
                               for p in inner_model.parameters() if p.grad is not None))
        torch.distributed.all_reduce(gnorm, op=torch.distributed.ReduceOp.AVG)
        traj.append({
            "loss": float(vec[0]), "surface": float(vec[1]),
            "upper_air": float(vec[2]), "diagnostic": float(vec[3]),
            "grad_norm": float(gnorm),
        })

    if dist.rank != 0:
        return

    record = {
        "tag": TAG, "git_sha": _git_sha(), "seed": seed, "steps": STEPS,
        "mode": MODE, "world_size": dist.world_size,
        "use_zero_optimizer": use_zero,
        "checkpointing": int(cfg.model.get("checkpointing", 0)),
        "epsilon_factor": epsilon_factor,
        "n_params": n_params,
        "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype else "off",
        "torch_compile_mode": None,
        "batch_size": int(cfg.dataset.batch_size),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "deterministic": False,   # reduction order is not — that is the point
        "cudnn_benchmark": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "config_yaml_sha256": hashlib.sha256(
            OmegaConf.to_yaml(cfg, resolve=True).encode()).hexdigest()[:16],
        "loss_trajectory": traj,
        "forward_output_stats": {},
    }
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{TAG}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"EQUIV_DDP_CAPTURE_OK {path}")
    print(f"  tag={TAG} world_size={dist.world_size} zero={use_zero} "
          f"steps={STEPS} seed={seed} loss[0]={traj[0]['loss']:.8e} "
          f"loss[-1]={traj[-1]['loss']:.8e}")


if __name__ == "__main__":
    main()
