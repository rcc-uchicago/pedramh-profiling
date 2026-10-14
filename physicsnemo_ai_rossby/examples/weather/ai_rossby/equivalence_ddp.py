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
AdamW — so the single-rank gate can say nothing about it.

⚠ CORRECTED 2026-08-07. An earlier version of this file claimed ZeRO
"reduce-scatters ... different summation ORDER", and concluded a bitwise test
would be wrong. **That was false.** ``ZeroRedundancyOptimizer`` does not touch
gradient reduction at all: DDP's all-reduce runs unchanged, each rank then runs
the local optimizer over its own shard and **broadcasts** the updated parameters
(``torch/distributed/optim/zero_redundancy_optimizer.py``: ``step()`` ->
``_local_step()`` + ``_sync_params()``). There is no reduce-scatter.

The consequence inverts the test. After DDP's all-reduce every rank holds
bit-identical gradients, and AdamW is elementwise — so which rank computes a
given parameter's update cannot change the result. **Correct ZeRO-1 is bitwise
identical to plain DDP.** Anything else is a defect, not float noise.

The method: hold everything else fixed and demand ZERO
------------------------------------------------------
Two plain-DDP runs still serve as a control — they establish that the harness
itself is reproducible (they came out bitwise equal, so it is). The real
requirement is that the ZeRO arm match them exactly.

⚠ THE KERNEL MUST BE HELD FIXED. ``_wrap_zero`` drops ``fused=True`` because
``ZeroRedundancyOptimizer`` rejects it, so a naive comparison runs FUSED AdamW
against EAGER AdamW and measures the kernel swap, not the sharding. The first
run of this test did exactly that and produced 6.675e-06 — entirely explained
by the kernel difference. The launcher now forces ``fused=false`` on every arm.

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
    # + rank, as production does (train.py:808-810). Seeding every rank
    # identically made all four draw the SAME epsilon_factor perturbation — a
    # per-rank correlation the real run does not have.
    torch.manual_seed(seed + dist.rank)
    torch.cuda.manual_seed_all(seed + dist.rank)

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
    # These kwargs MUST match production (train.py:934-943). In particular
    # gradient_as_bucket_view=True makes p.grad a view into the all-reduce
    # bucket — precisely the memory ZeRO's bucketing interacts with — so a
    # capture using the default would certify a configuration nobody runs.
    model = DistributedDataParallel(
        inner_model,
        device_ids=[dist.local_rank],
        output_device=dist.device,
        broadcast_buffers=dist.broadcast_buffers,
        find_unused_parameters=dist.find_unused_parameters,
        gradient_as_bucket_view=True,
        static_graph=bool(cfg_train.get("ddp_static_graph", False)),
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
    lr_trace: list[float] = []
    last_batch = None
    # ZeRO-1's characteristic failure is REPLICA DIVERGENCE: a missed or stale
    # parameter broadcast leaves ranks holding different weights. The averaged
    # scalars below cannot see it — averaging is exactly what hides cross-rank
    # disagreement — so check the replicas directly.
    max_rank_delta = 0.0

    def _cross_rank_param_delta() -> float:
        """Max spread of a per-rank parameter checksum. 0.0 == replicas agree.

        Cheap (one scalar, two all-reduces) and direct: if any rank's parameters
        differ after the optimizer step, MIN and MAX of the checksum diverge.
        """
        # SUM OF SQUARES, not a signed sum: a signed total is invariant under
        # permutation and under sign-cancelling differences, so it would miss a
        # wrong shard->parameter mapping. Squares are strictly positive, so the
        # normaliser is also well-conditioned (a signed sum is near zero).
        chk = torch.zeros((), dtype=torch.float64, device=dist.device)
        for prm in inner_model.parameters():
            chk += prm.detach().double().square().sum()
        lo, hi = chk.clone(), chk.clone()
        torch.distributed.all_reduce(lo, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(hi, op=torch.distributed.ReduceOp.MAX)
        denom = max(abs(float(hi)), 1e-30)
        return abs(float(hi) - float(lo)) / denom

    for batch in datapipe:
        if len(traj) >= STEPS:
            break
        last_batch = batch
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
        # Per-step LR, because the whole 20-step window can sit inside warmup:
        # with num_warmup_epochs=5 the warmup is 5*STEPS steps, so a 20-step
        # capture never leaves it. A discrepancy in the OPTIMIZER path scales
        # with step size, so a test run at 4% of peak LR understates it. Record
        # the LR so any reader can see which regime was actually exercised.
        lr_trace.append(float(optimizer.param_groups[0]["lr"]))
        max_rank_delta = max(max_rank_delta, _cross_rank_param_delta())
        traj.append({
            "loss": float(vec[0]), "surface": float(vec[1]),
            "upper_air": float(vec[2]), "diagnostic": float(vec[3]),
            "grad_norm": float(gnorm),
            "lr": lr_trace[-1],
        })

    # One extra forward on the last batch, recording summary stats only (never
    # tensors — DESIGN §4.2/§7), mirroring equivalence.py:286.
    final_out_stats = {}
    if last_batch is not None:
        with torch.no_grad():
            amp_ctx = (torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
                       if amp_dtype is not None else contextlib.nullcontext())
            with amp_ctx:
                o = inner_model(last_batch["surface_in"], last_batch["constant_boundary"],
                                last_batch["varying_boundary"], last_batch["upper_air_in"])
        for nm, t in (("surface", o[0]), ("upper_air", o[1])):
            f = t.detach().float()
            final_out_stats[nm] = {"mean": float(f.mean()), "std": float(f.std()),
                                   "min": float(f.min()), "max": float(f.max()),
                                   "shape": list(t.shape)}

    if dist.rank != 0:
        return

    record = {
        "tag": TAG, "git_sha": _git_sha(), "seed": seed, "steps": STEPS,
        # Witness the ACTUAL optimizer, not what the config asked for:
        # _wrap_zero has a silent fallback to plain AdamW, and it drops `fused`.
        # Without these the JSON cannot show which kernel or wrapper ran.
        "optimizer_class": type(optimizer).__name__,
        "optimizer_fused": bool(optimizer.param_groups[0].get("fused", False)),
        "lr_first": lr_trace[0] if lr_trace else None,
        "lr_last": lr_trace[-1] if lr_trace else None,
        "max_cross_rank_param_delta": max_rank_delta,
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
        # NOT {} — an empty dict makes compare_baselines.py silently compare
        # nothing (it iterates set(base) & set(cand)). It also adds the one
        # thing the scalars cannot give: grad_norm is a Euclidean norm over the
        # whole parameter vector, hence invariant under permutation or sign
        # flip, so a wrong shard->parameter mapping barely moves it. Output
        # tensor stats are not invariant that way.
        "forward_output_stats": final_out_stats,
    }
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{TAG}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"  optimizer={type(optimizer).__name__} "
          f"fused={bool(optimizer.param_groups[0].get('fused', False))} "
          f"lr[0]={lr_trace[0]:.3e} lr[-1]={lr_trace[-1]:.3e} "
          f"cross_rank_delta={max_rank_delta:.3e}")

    # (a) ZeRO must actually be a ZeRO optimizer. _wrap_zero silently falls back
    #     to plain AdamW when distributed is not initialised, and under a BITWISE
    #     bar that fallback reproduces the control arms EXACTLY — so the strictest
    #     possible test would pass the one case where ZeRO never ran. Assert it.
    if use_zero and type(optimizer).__name__ != "ZeroRedundancyOptimizer":
        print(f"ERROR ZERO_NOT_ACTUALLY_USED — use_zero_optimizer=True but the "
              f"optimizer is {type(optimizer).__name__}. _wrap_zero fell back "
              f"silently; this run proves nothing about ZeRO.")
        sys.exit(5)
    # (b) Replica divergence must FAIL, not merely print — the launcher keys on
    #     the exit code, so a bare print would have been invisible to the gate.
    if max_rank_delta > 0.0:
        print(f"ERROR REPLICA_DIVERGENCE max_cross_rank_param_delta="
              f"{max_rank_delta:.3e} — ranks hold DIFFERENT weights. That is a "
              f"ZeRO sharding/broadcast defect, not float noise.")
        sys.exit(6)
    print(f"EQUIV_DDP_CAPTURE_OK {path}")
    print(f"  tag={TAG} world_size={dist.world_size} zero={use_zero} "
          f"steps={STEPS} seed={seed} loss[0]={traj[0]['loss']:.8e} "
          f"loss[-1]={traj[-1]['loss']:.8e}")


if __name__ == "__main__":
    main()
