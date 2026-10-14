# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DESIGN §4.1 equivalence capture for the ai-rossby recipe.

Records a **deterministic** K-step loss trajectory plus forward-output summary
stats to JSON, so a hot-path change (`torch.compile`, FlexAttention, a fused
optimizer) can be proven to compute the same thing before it is adopted. Pairs
with ``compare_baselines.py``, which applies §4.1's metric and tolerance.

This is the correctness oracle — the thing DESIGN §5 means by "each rung = one
small commit with its own §4 check". `profile_train.py` answers *how fast*; this
answers *is it still the same model*. They are separate on purpose: the bench
wants `cudnn.benchmark=True` and 4 ranks, this wants determinism and 1 rank.

Usage (COMPUTE NODE — imports physicsnemo; see profile_train.py's warning)::

    # 1. baseline, before the change
    CUBLAS_WORKSPACE_CONFIG=:4096:8 AI_ROSSBY_EQUIV_TAG=eager \\
        python equivalence.py model=pangu_plasim_e3sm dataset=e3sm_pangu_parity \\
            training=pangu_plasim_legacy loss=mae

    # 2. one change, same seed and config
    CUBLAS_WORKSPACE_CONFIG=:4096:8 AI_ROSSBY_EQUIV_TAG=compiled \\
        TORCH_COMPILE_MODE=default python equivalence.py <same overrides>

    # 3. gate it
    python compare_baselines.py <baseline>.json <candidate>.json --tolerance 1e-2

`CUBLAS_WORKSPACE_CONFIG` must be set in the ENVIRONMENT before torch
initialises cuBLAS — setting it from Python here would be too late and would
silently give a non-deterministic GEMM, so this script refuses to run without it.

Output is a JSON text summary only (DESIGN §4.2 — no tensors in git).
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

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
    _model_accepts_train_kwarg,
    _optional_model_kwargs,
    _resolve_amp_dtype,
    make_optimizer,
    make_scheduler,
    train_step,
)

STEPS = int(os.environ.get("AI_ROSSBY_EQUIV_STEPS", "20"))  # K=20 per §4.1
TAG = os.environ.get("AI_ROSSBY_EQUIV_TAG", "baseline")

# "train" — §4.1 as written: K optimizer steps, so the weights evolve.
# "fixed" — K forward+backward passes with NO optimizer step, so the weights
#           never change and differences CANNOT compound between steps.
#
# Why "fixed" exists: a training trajectory in bf16 amplifies any bit-level
# perturbation, so it cannot distinguish a correct-but-bit-different kernel
# (e.g. a fusion) from a genuinely wrong one — both drift. Holding the weights
# fixed isolates the per-step kernel difference, which is the quantity a fusion
# change should actually be judged on. Backward still runs (it is where most of
# the fusion is), it just does not feed back into the weights.
MODE = os.environ.get("AI_ROSSBY_EQUIV_MODE", "train")
OUT_DIR = os.environ.get(
    "AI_ROSSBY_EQUIV_DIR",
    str(Path(__file__).resolve().parents[4] / "baselines" / "ai_rossby_pangu_plasim"),
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _stats(t: torch.Tensor) -> dict:
    """Summary stats of a tensor — never the tensor itself (DESIGN §4.2/§7)."""
    f = t.detach().float()
    return {
        "mean": float(f.mean()), "std": float(f.std()),
        "min": float(f.min()), "max": float(f.max()),
        "shape": list(t.shape),
    }


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        print("ERROR CUBLAS_WORKSPACE_CONFIG_UNSET — §4.1 requires it to be set in "
              "the ENVIRONMENT (:4096:8) before torch initialises cuBLAS; setting "
              "it from Python is too late and yields non-deterministic GEMM.")
        sys.exit(2)

    # Determinism, NOT speed — the inverse of profile_train.py's setup. TF32 is
    # left OFF here: it changes mantissa precision, and a baseline should pin the
    # arithmetic rather than the throughput.
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.world_size != 1:
        print(f"ERROR EQUIV_REQUIRES_WORLD_SIZE_1 (got {dist.world_size}) — §4.1 "
              "captures at world size 1; add a separate 4-GPU baseline when the "
              "change touches DDP.")
        sys.exit(2)

    seed = int(cfg.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg_train = cfg.training
    stages = list(cfg_train.stages)
    unroll_steps = int(stages[0].get("unroll_steps", 1))
    if unroll_steps != 1:
        print(f"ERROR EQUIV_UNROLL_UNSUPPORTED ({unroll_steps}) — capture the "
              "single-step path; multistep needs its own baseline.")
        sys.exit(2)

    # shuffle=False: a baseline must replay the SAME samples in the same order,
    # or the trajectory differs for reasons that have nothing to do with the
    # change under test.
    datapipe = build_datapipe(
        cfg,
        zarr_path=_resolve_path(cfg.dataset.zarr_path),
        distributed=False,
        device=dist.device,
        shuffle=False,
        seed=seed,
        unroll_steps=1,
    )
    model = build_model(cfg.model).to(dist.device)
    inner_model = model
    n_params = sum(p.numel() for p in model.parameters())

    compile_mode = os.environ.get("TORCH_COMPILE_MODE", "")
    if compile_mode:
        model = torch.compile(model, mode=compile_mode, fullgraph=False)

    loss_fn = build_loss(cfg).to(dist.device)
    optimizer = make_optimizer(inner_model, _flatten_optimizer_cfg(cfg_train.optimizer))
    scheduler = make_scheduler(
        optimizer,
        _flatten_scheduler_cfg(
            stages[0].scheduler, lr=float(cfg_train.optimizer.lr),
            steps_per_epoch=STEPS, num_epochs=1,
        ),
        total_steps=STEPS,
    )
    amp_dtype = _resolve_amp_dtype(cfg_train.get("amp", None))
    grad_scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None
    has_diagnostic = inner_model.has_diagnostic
    vae_kl_weight = float(cfg.loss.get("vae_kl_weight", 0.0))

    # --- the trajectory ---------------------------------------------------
    traj: list[dict] = []
    probe_batch = None
    for batch in datapipe:
        if len(traj) >= STEPS:
            break
        if probe_batch is None:
            probe_batch = batch  # kept for the post-training forward probe

        if MODE == "fixed":
            # Forward + backward, but NO optimizer.step(): weights are identical
            # at every iteration, so step i's difference is purely the kernels',
            # never inherited from step i-1. `grad_norm` is recorded because it
            # summarises the BACKWARD pass, which a forward-only check would miss
            # — and backward is 63% of this model's step time.
            optimizer.zero_grad(set_to_none=True)
            amp_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_dtype is not None else contextlib.nullcontext()
            )
            with amp_ctx:
                kw = _optional_model_kwargs(model, batch)
                if _model_accepts_train_kwarg(model):
                    out = model(
                        batch["surface_in"], batch["constant_boundary"],
                        batch["varying_boundary"], batch["upper_air_in"],
                        target_surface=batch.get("target_surface"),
                        target_upper_air=batch.get("target_upper_air"),
                        train=True, **kw,
                    )
                else:
                    out = model(
                        batch["surface_in"], batch["constant_boundary"],
                        batch["varying_boundary"], batch["upper_air_in"], **kw,
                    )
                o_s, o_u = out[0], out[1]
                o_d = out[2] if has_diagnostic else None
                losses = loss_fn(
                    o_s, o_u, batch["target_surface"], batch["target_upper_air"],
                    out_diagnostic=o_d,
                    target_diagnostic=batch.get("diagnostic") if has_diagnostic else None,
                )
            losses["loss"].backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
                inner_model.parameters(), float("inf")
            )  # inf max_norm => measures, never clips
            rec = {k: float(v) for k, v in losses.items()}
            rec["grad_norm"] = float(gnorm)
            traj.append(rec)
            optimizer.zero_grad(set_to_none=True)
        else:
            losses = train_step(
                model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
                batch=batch, has_diagnostic=has_diagnostic, vae_kl_weight=vae_kl_weight,
                amp_dtype=amp_dtype, grad_scaler=grad_scaler,
            )
            traj.append({k: float(v) for k, v in losses.items()})

    if len(traj) < STEPS:
        print(f"ERROR EQUIV_SHORT_TRAJECTORY ({len(traj)}/{STEPS} steps)")
        sys.exit(2)

    # --- forward-output stats (§4.1 asks for these alongside the losses) ---
    # A separate eval forward on a FIXED batch, so the numbers describe the
    # trained weights rather than whatever the last training step happened to see.
    model_out = {}
    with torch.no_grad():
        kw = _optional_model_kwargs(model, probe_batch)
        if _model_accepts_train_kwarg(model):
            out = model(
                probe_batch["surface_in"], probe_batch["constant_boundary"],
                probe_batch["varying_boundary"], probe_batch["upper_air_in"],
                target_surface=probe_batch.get("target_surface"),
                target_upper_air=probe_batch.get("target_upper_air"),
                train=False, **kw,
            )
        else:
            out = model(
                probe_batch["surface_in"], probe_batch["constant_boundary"],
                probe_batch["varying_boundary"], probe_batch["upper_air_in"], **kw,
            )
        model_out["surface"] = _stats(out[0])
        model_out["upper_air"] = _stats(out[1])

    record = {
        "tag": TAG,
        "git_sha": _git_sha(),
        "seed": seed,
        "steps": STEPS,
        "mode": MODE,
        "world_size": 1,
        "n_params": n_params,
        "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype else "off",
        "torch_compile_mode": compile_mode or None,
        "batch_size": int(cfg.dataset.batch_size),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "python": platform.python_version(),
        "deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "config_yaml_sha256": None,
        "loss_trajectory": traj,
        "forward_output_stats": model_out,
    }
    import hashlib
    record["config_yaml_sha256"] = hashlib.sha256(
        OmegaConf.to_yaml(cfg, resolve=True).encode()
    ).hexdigest()[:16]

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{TAG}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")

    first, last = traj[0]["loss"], traj[-1]["loss"]
    print(f"EQUIV_CAPTURE_OK {path}")
    print(f"  tag={TAG} mode={MODE} steps={STEPS} seed={seed} compile={compile_mode or 'eager'}")
    print(f"  loss {first:.6f} -> {last:.6f}   params={n_params:,}")


if __name__ == "__main__":
    main()
