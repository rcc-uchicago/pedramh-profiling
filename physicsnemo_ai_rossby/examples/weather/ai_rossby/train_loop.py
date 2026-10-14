# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-epoch training step + optimizer/scheduler factories for Pangu_Plasim.

Phase 3 v1: PanguPlasimLegacy (deterministic, no VAE-KL). The optimizer +
scheduler choices come from PanguWeather v2.0 config conventions
(AdamW + OneCycleLR for the legacy variant; AdamW + LinearWarmupCosineAnnealingLR
for the future PanguPlasim with VAE).
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Optional

import torch
import torch.cuda.nvtx as nvtx
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    OneCycleLR,
    SequentialLR,
)

# --- profiling instrumentation (inert unless AI_ROSSBY_NVTX=1) --------------
# Range NAMES are a cross-project contract, not a local choice: `parse_nsys.py`
# selects on them by literal string, so renaming one silently drops it from every
# profile summary and invalidates comparison against PanguWeather/s2s
# (CLAUDE.md #10). Knobs are per-project (`AI_ROSSBY_*`), names are shared.
#
# `ema` is deliberately NOT emitted here: ai-rossby's EMA update lives in
# train.py's batch loop, not in either step function below. parse_nsys.py
# tolerates an absent range; a MISNAMED one is the failure mode that matters.
NVTX = os.environ.get("AI_ROSSBY_NVTX") == "1"


@contextlib.contextmanager
def _nvtx_range(name: str):
    """Push an NVTX range, popping it even if the body raises.

    A context manager rather than bare push/pop pairs because an exception
    between them leaves the range open and corrupts every subsequent range in
    the trace. No-op — and no CUDA call at all — when AI_ROSSBY_NVTX is unset.
    """
    if NVTX:
        nvtx.range_push(name)
    try:
        yield
    finally:
        if NVTX:
            nvtx.range_pop()


def make_optimizer(model: torch.nn.Module, cfg: Any) -> torch.optim.Optimizer:
    """Build an optimizer from a config dict-like.

    Recognized keys:

    * ``optimizer_type`` — ``"AdamW"`` or ``"Muon"``.
    * ``lr`` — base learning rate.
    * ``weight_decay`` — default 0.
    * ``fused`` — when True, requests the fused CUDA kernel for AdamW
      (``torch.optim.AdamW(..., fused=True)``). Requires CUDA; falls back to
      the eager AdamW with a warning if the runtime can't honor it. Defaults
      to True on CUDA (matches PanguWeather's reference SFNO trainer), False
      otherwise.

    ``optimizer_type="Muon"`` requires ``model`` to expose a
    ``muon_param_groups(lr, weight_decay)`` method (the amip_si wrappers —
    :class:`AmipDiTWrapper` / :class:`RollingDiTWrapper` / :class:`ERDMWrapper`
    — all do) and the ``Muon`` package
    (``pip install git+https://github.com/KellerJordan/Muon``, or the
    ``muon-optimizers`` extra in ``pyproject.toml``). The two param groups it
    returns are handed verbatim to ``muon.MuonWithAuxAdam``.
    """
    name = getattr(cfg, "optimizer_type", "AdamW")
    if name == "Muon":
        return _make_muon_optimizer(model, cfg)
    if name != "AdamW":
        raise ValueError(
            f"Unsupported optimizer_type={name!r} (supported: 'AdamW', 'Muon')."
        )
    fused = bool(getattr(cfg, "fused", torch.cuda.is_available()))
    wd = float(getattr(cfg, "weight_decay", 0.0))
    betas = getattr(cfg, "betas", None)
    kwargs = dict(lr=float(cfg.lr), weight_decay=wd)
    if betas is not None:
        kwargs["betas"] = tuple(float(b) for b in betas)
    if fused:
        if not torch.cuda.is_available():
            import warnings as _warnings

            _warnings.warn(
                "cfg.fused=True requested but CUDA is not available; falling "
                "back to eager AdamW.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            kwargs["fused"] = True

    # Selective weight decay (ArchesWeather): apply wd ONLY to params whose name
    # contains 'weight' and not 'norm' (i.e. Linear/Conv weights, not biases or
    # LayerNorm/pos-bias params). Matches geoarches' configure_optimizers.
    if bool(getattr(cfg, "selective_weight_decay", False)) and wd > 0.0:
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "weight" in name and "norm" not in name:
                decay.append(p)
            else:
                no_decay.append(p)
        params = [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        kwargs.pop("weight_decay")
        return _wrap_zero(params, kwargs, cfg)

    return _wrap_zero(model.parameters(), kwargs, cfg)


def _wrap_zero(params, kwargs: dict, cfg: Any) -> torch.optim.Optimizer:
    """Build AdamW, optionally sharded across ranks as ZeRO Stage 1.

    ``use_zero_optimizer: True`` shards the optimizer STATE (Adam's ``exp_avg``
    and ``exp_avg_sq``) across the data-parallel ranks instead of every rank
    holding an identical full copy. Same arithmetic, same result — it trades
    memory for one all-gather of the updated parameters per step.

    Why it matters here: this model's optimizer state is
    2 x 1,182,108,160 x 4 B = **8.8 GB per GPU**, replicated 4 times over. At 4
    ranks ZeRO-1 leaves ~2.2 GB, freeing ~6.6 GB — which is the difference
    between ``checkpointing: 1`` (the 1.307x config, measured at 36.1 GB +
    4.4 GB of EMA = OOM on a 40 GB A100) and it fitting.

    Note it shards a DIFFERENT pool than activation checkpointing: ZeRO touches
    optimizer state, ``model.checkpointing`` touches activations. They compose.

    ⚠ Two things this changes that callers must honor:

    * **``fused=True`` is invalid on the wrapper** and is dropped with a
      warning. PanguWeather's trainer documents the same constraint
      (``train.py:713-717``). Costs ~1-2%, against ~6.6 GB gained.
    * **``state_dict()`` returns only the LOCAL SHARD** until
      ``consolidate_state_dict(to=0)`` is called. Saving without it produces a
      checkpoint that silently resumes with a fraction of the optimizer state —
      a corruption that shows up as a training-curve discontinuity days later,
      not as an error. ``train.py`` consolidates before every save.

    Falls back to the plain optimizer (with a warning) when torch.distributed is
    not initialized, so single-GPU runs and the bench keep working unchanged.
    """
    if not bool(getattr(cfg, "use_zero_optimizer", False)):
        return torch.optim.AdamW(params, **kwargs)

    import warnings as _warnings

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        _warnings.warn(
            "use_zero_optimizer=True but torch.distributed is not initialized; "
            "falling back to the unsharded AdamW.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.optim.AdamW(params, **kwargs)

    from torch.distributed.optim import ZeroRedundancyOptimizer

    if kwargs.pop("fused", False):
        _warnings.warn(
            "fused=True is not supported by ZeroRedundancyOptimizer; using the "
            "eager AdamW kernel. (~1-2% slower, ~6.6 GB/GPU saved.)",
            RuntimeWarning,
            stacklevel=2,
        )
    return ZeroRedundancyOptimizer(
        params, optimizer_class=torch.optim.AdamW, **kwargs
    )


def _make_muon_optimizer(model: torch.nn.Module, cfg: Any) -> torch.optim.Optimizer:
    """Build ``muon.MuonWithAuxAdam`` from ``model.muon_param_groups()``.

    ``cfg.weight_decay`` is forwarded to both the Muon and aux-AdamW
    groups (matches upstream amip, which applies a single weight_decay
    across both). The Muon group's LR multiplier defaults to the
    wrapper method's own default (10x, per upstream).
    """
    if not hasattr(model, "muon_param_groups"):
        raise ValueError(
            f"optimizer_type='Muon' requires a model exposing "
            f"muon_param_groups(); {type(model).__name__} does not. "
            "Use one of the amip_si wrappers (AmipDiTWrapper / "
            "RollingDiTWrapper / ERDMWrapper) or add Muon support to "
            "the wrapper."
        )
    try:
        from muon import MuonWithAuxAdam
    except ImportError as exc:
        raise ImportError(
            "optimizer_type='Muon' requires the `muon` package: "
            "`pip install git+https://github.com/KellerJordan/Muon` "
            "(or `pip install nvidia-physicsnemo[muon-optimizers]`)."
        ) from exc
    param_groups = model.muon_param_groups(
        lr=float(cfg.lr),
        weight_decay=float(getattr(cfg, "weight_decay", 0.01)),
    )
    return MuonWithAuxAdam(param_groups)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Any,
    *,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build a scheduler from a config dict-like.

    Supported ``scheduler``:

    * ``"OneCycleLR"`` — uses ``oc_pct_start``, ``oc_div_factor``,
      ``oc_final_div_factor`` (PanguWeather PANGU_PLASIM_H5_DERECHO_0514 keys).
      ``max_lr`` defaults to ``lr``.
    * ``"LinearWarmupCosineAnnealingLR"`` — composes a linear warmup
      (``num_warmup_steps``) with cosine annealing to ``eta_min``.
    """
    name = getattr(cfg, "scheduler", "OneCycleLR")
    if name == "OneCycleLR":
        return OneCycleLR(
            optimizer,
            max_lr=float(cfg.lr),
            total_steps=total_steps,
            pct_start=float(getattr(cfg, "oc_pct_start", 0.1)),
            div_factor=float(getattr(cfg, "oc_div_factor", 1e5)),
            final_div_factor=float(getattr(cfg, "oc_final_div_factor", 0.00025)),
            anneal_strategy="cos",
        )
    if name == "LinearWarmupCosineAnnealingLR":
        warmup_steps = int(getattr(cfg, "num_warmup_steps", 0) or 0)
        warmup_start_lr = float(getattr(cfg, "warmup_start_lr", 1e-8))
        eta_min = float(getattr(cfg, "eta_min", 0.0))
        if warmup_steps <= 0:
            return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)
        warmup = LinearLR(
            optimizer,
            start_factor=warmup_start_lr / float(cfg.lr),
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=eta_min
        )
        return SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
    if name == "CosineAnnealingLR":
        # Plain CosineAnnealingLR — used by the AMIP diffusion recipe.
        # ``T_max`` defaults to ``total_steps`` (the per-stage budget the
        # caller supplies); ``cosine_eta_min`` mirrors the yaml key name
        # used in conf/training/amip_diffusion.yaml. ``eta_min`` is
        # accepted as a synonym so the Phase 3 LinearWarmupCosineAnnealingLR
        # config keys also work here.
        T_max = int(getattr(cfg, "T_max", total_steps))
        eta_min = float(
            getattr(cfg, "cosine_eta_min", getattr(cfg, "eta_min", 0.0))
        )
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    raise ValueError(f"Unknown scheduler {name!r}")


_AMP_DTYPES = {
    "none": None,
    "off": None,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


def _resolve_amp_dtype(amp: str | bool | None) -> Optional[torch.dtype]:
    """Map ``cfg.amp`` (string or bool) to a torch dtype or ``None`` for off."""
    if amp is None or amp is False:
        return None
    if amp is True:
        return torch.bfloat16  # default-on AMP picks bf16 (matches PanguWeather)
    return _AMP_DTYPES.get(str(amp).lower())


def perturb_inputs(surface, upper_air, epsilon_factor: float):
    """PanguWeather-parity Gaussian perturbation of the INPUT state.

    PanguWeather adds this in its data loader
    (``data_loader_multifiles.py:1102-1112``), gated on ``epsilon_factor > 0``:

        surface_t   += randn(surface_t.shape)   * eps * (surface_ff_std / surface_std)
        upper_air_t += randn(upper_air_t.shape) * eps * (upper_air_ff_std / upper_air_std)

    Both std files are the SAME file in the E3SM configs
    (``data_2015-2050_std_corr.nc``), so those ratios are exactly 1.0 and the
    perturbation reduces to ``randn * epsilon_factor`` in NORMALIZED units.
    That is why this needs no stats and can live here.

    Applied to the input timestep only — never the targets, boundaries or
    diagnostics, matching PanguWeather.

    Deliberately done here (on device, per batch) rather than in the dataset:
    PanguWeather draws it inside DataLoader workers from the global RNG with no
    ``worker_init_fn``, which makes its ``num_data_workers`` an output-CHANGING
    knob (CHANGELOG next-action #4). Drawing it here uses the per-rank-seeded
    global RNG instead, so the noise is reproducible from ``cfg.seed`` and
    independent of worker count. Same distribution, no trap.
    """
    if not epsilon_factor or epsilon_factor <= 0.0:
        return surface, upper_air
    return (
        surface + torch.randn_like(surface) * epsilon_factor,
        upper_air + torch.randn_like(upper_air) * epsilon_factor,
    )


def clip_and_measure_grads(
    model: torch.nn.Module,
    *,
    grad_clip_norm: float = 0.0,
    grad_scaler: Optional["torch.amp.GradScaler"] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_stats: Optional[dict] = None,
) -> None:
    """Clip gradients (and/or record their norm) BETWEEN backward and step.

    ⚠ WHY THIS FUNCTION EXISTS — a silent no-op that cost a 50-node sweep.

    ``train.py`` used to apply ``clip_grad_norm_`` at its own call site, AFTER
    ``train_step``/``multistep_train_step`` returned. But both of those run
    ``optimizer.step()`` internally (train_loop.py:476, :618), so the clip ran
    on gradients that had ALREADY been applied to the weights, and the next
    iteration's ``optimizer.zero_grad(set_to_none=True)`` then discarded them.
    ``training.grad_clip_norm`` therefore did nothing at all in the eager path.

    It failed silently in the worst way: the knob parsed, logged, and appeared
    in the run config, so job 7575680 (``HP_ARM=clip``) looked like a clean test
    of gradient clipping. It wasn't — it was a rerun of the unclipped config,
    which is exactly why it matched the unclipped run to within noise at the
    epoch-12 divergence (0.1376 vs 0.1618).

    Clipping must happen after backward and before the step, which is only
    reachable from inside those functions. Hence this helper.

    ``grad_stats``: when a dict is given it is populated in-place with
    ``{"grad_norm": float, "clipped": bool}``. Requesting stats with clipping
    OFF takes a READ-ONLY norm (``vector_norm``, never ``clip_grad_norm_``,
    which would write ``grad.mul_(1.0)`` back) so the diagnostic cannot perturb
    what it measures. Costs one extra pass over the gradients (~4.7 GB on this
    model, ~3 ms) — hence opt-in, not always-on.
    """
    want_clip = bool(grad_clip_norm and grad_clip_norm > 0.0)
    if not want_clip and grad_stats is None:
        return

    # fp16: gradients are still multiplied by the loss scale here, so both the
    # clip threshold and the reported norm would be meaningless. unscale_ is
    # idempotent per step, and grad_scaler.step() below detects it already ran.
    if grad_scaler is not None and optimizer is not None:
        grad_scaler.unscale_(optimizer)

    if want_clip:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    else:
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        total_norm = (
            torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(g.detach(), 2) for g in grads]), 2
            )
            if grads
            else torch.zeros(())
        )

    if grad_stats is not None:
        norm = float(total_norm)
        grad_stats["grad_norm"] = norm
        grad_stats["clipped"] = bool(want_clip and norm > grad_clip_norm)


def train_step(
    *,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    batch: dict[str, torch.Tensor],
    has_diagnostic: bool,
    vae_kl_weight: float = 0.0,
    amp_dtype: Optional[torch.dtype] = None,
    grad_scaler: Optional["torch.amp.GradScaler"] = None,
    epsilon_factor: float = 0.0,
    capture_outputs: Optional[dict] = None,
    grad_clip_norm: float = 0.0,
    grad_stats: Optional[dict] = None,
) -> dict[str, torch.Tensor]:
    """One optimizer step: forward + backward + clip + step + scheduler tick.

    Returns the loss dict from :class:`PanguPlasimLoss` plus a ``"vae_kl"``
    entry. Compatible with both PanguPlasimLegacy (5- or 7-tuple output with
    zero latent placeholders — `vae_kl` stays ~0) and PanguPlasim with VAE
    (6- or 7-tuple with real ``mu``/``logvar``).

    ``capture_outputs``: when given a dict, it is populated in-place with the
    detached (normalized-space) prediction/target tensors — ``out_surface``,
    ``out_upper_air``, ``out_diagnostic``, ``target_surface``,
    ``target_upper_air``, ``target_diagnostic`` (the last of each pair,
    ``out_diagnostic``/``target_diagnostic``, is ``None`` when the model has
    no diagnostic head) — for a caller-side per-variable diagnostic (e.g.
    PanguWeather-style per-var/per-level wandb logging). ``None`` (default)
    skips this — existing callers are unaffected.

    Mixed-precision support
    -----------------------
    When ``amp_dtype`` is not ``None``, the forward + loss computation runs
    under ``torch.amp.autocast(device_type="cuda", dtype=amp_dtype)``. For
    ``bf16`` (matches PanguWeather v2.0's default for SFNO_PLASIM) no
    :class:`GradScaler` is needed. For ``fp16`` pass an externally-managed
    ``grad_scaler`` so the trainer can also persist its state across
    checkpoints. The optimizer step is wrapped in
    ``grad_scaler.step`` + ``grad_scaler.update`` when present.

    When ``vae_kl_weight > 0`` and the model emits real ``(mu, logvar,
    mu_e2, logvar_e2)`` tuples, the KL divergence between the two encoder
    posteriors is computed and added: ``total = task_loss + vae_kl_weight * kl``.
    For PanguPlasimLegacy the model returns zero placeholders for the latent
    fields, so the KL evaluates to 0 and the task loss is unchanged
    regardless of ``vae_kl_weight``.
    """
    from loss import vae_kl_loss  # local import keeps train_loop / loss decoupled at import time

    optimizer.zero_grad(set_to_none=True)

    # PanguWeather-parity input perturbation. Before autocast so it is drawn in
    # fp32, matching PanguWeather (which perturbs in its fp32 loader). A shallow
    # copy so the caller's batch is untouched; no-op at the default 0.0.
    if epsilon_factor and epsilon_factor > 0.0:
        s, u = perturb_inputs(batch["surface_in"], batch["upper_air_in"], epsilon_factor)
        batch = {**batch, "surface_in": s, "upper_air_in": u}

    # Autocast context — no-op when amp_dtype is None.
    if amp_dtype is None:
        amp_ctx = contextlib.nullcontext()
    else:
        device_type = "cuda" if batch["surface_in"].is_cuda else "cpu"
        amp_ctx = torch.amp.autocast(device_type=device_type, dtype=amp_dtype)

    extra_kwargs = _optional_model_kwargs(model, batch)
    with _nvtx_range("forward_loss"), amp_ctx:
        out = model(
            batch["surface_in"],
            batch["constant_boundary"],
            batch["varying_boundary"],
            batch["upper_air_in"],
            target_surface=batch.get("target_surface"),
            target_upper_air=batch.get("target_upper_air"),
            train=True,
            **extra_kwargs,
        ) if _model_accepts_train_kwarg(model) else model(
            batch["surface_in"],
            batch["constant_boundary"],
            batch["varying_boundary"],
            batch["upper_air_in"],
            **extra_kwargs,
        )

        # Output tuple layout:
        # * PanguPlasimLegacy (no diag): (surface, upper_air, 0, 0, 0, 0)
        # * PanguPlasimLegacy (diag):    (surface, upper_air, diag, 0, 0, 0, 0)
        # * PanguPlasim (no diag, train=True): (surface, upper_air, mu, logvar, mu_e2, logvar_e2)
        # * PanguPlasim (diag, train=True):    (surface, upper_air, diag, mu, logvar, mu_e2, logvar_e2)
        if has_diagnostic:
            out_surface, out_upper_air, out_diag = out[0], out[1], out[2]
            latent_offset = 3
        else:
            out_surface, out_upper_air = out[0], out[1]
            out_diag = None
            latent_offset = 2

        target_diag = batch.get("diagnostic") if has_diagnostic else None
        losses = loss_fn(
            out_surface,
            out_upper_air,
            batch["target_surface"],
            batch["target_upper_air"],
            out_diagnostic=out_diag,
            target_diagnostic=target_diag,
        )

        if capture_outputs is not None:
            capture_outputs.update(
                out_surface=out_surface.detach(),
                out_upper_air=out_upper_air.detach(),
                out_diagnostic=out_diag.detach() if out_diag is not None else None,
                target_surface=batch["target_surface"].detach(),
                target_upper_air=batch["target_upper_air"].detach(),
                target_diagnostic=target_diag.detach() if target_diag is not None else None,
            )

        # The VAE-KL branch fires only when (a) KL weight > 0, (b) the model
        # returned at least four latent slots, AND (c) those slots are torch
        # Tensors (the legacy port emits Python int `0` placeholders, not
        # tensors — easy sentinel for "no VAE here").
        latent_slots = out[latent_offset : latent_offset + 4] if len(out) >= latent_offset + 4 else ()
        has_real_latents = (
            len(latent_slots) == 4
            and all(isinstance(x, torch.Tensor) and x.numel() > 0 for x in latent_slots)
        )
        if vae_kl_weight > 0.0 and has_real_latents:
            mu, logvar, mu_e2, logvar_e2 = latent_slots
            kl = vae_kl_loss(mu, logvar, mu_e2, logvar_e2)
            losses["vae_kl"] = kl.detach()
            losses["loss"] = losses["loss"] + vae_kl_weight * kl
        else:
            # VAE disabled or model emits placeholders. Keep the key for logger uniformity.
            losses["vae_kl"] = torch.zeros((), device=out_surface.device, dtype=out_surface.dtype)

    # Backward + step. GradScaler is required for fp16 (underflow protection);
    # bf16 retains enough dynamic range that no scaling is needed.
    if grad_scaler is not None:
        with _nvtx_range("backward"):
            grad_scaler.scale(losses["loss"]).backward()
        clip_and_measure_grads(
            model, grad_clip_norm=grad_clip_norm, grad_scaler=grad_scaler,
            optimizer=optimizer, grad_stats=grad_stats,
        )
        with _nvtx_range("optimizer"):
            grad_scaler.step(optimizer)
            grad_scaler.update()
    else:
        with _nvtx_range("backward"):
            losses["loss"].backward()
        clip_and_measure_grads(
            model, grad_clip_norm=grad_clip_norm, grad_stats=grad_stats,
        )
        with _nvtx_range("optimizer"):
            optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return losses


def multistep_train_step(
    *,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    batch: dict[str, torch.Tensor],
    has_diagnostic: bool,
    unroll_steps: int,
    vae_kl_weight: float = 0.0,
    amp_dtype: Optional[torch.dtype] = None,
    grad_scaler: Optional["torch.amp.GradScaler"] = None,
    epsilon_factor: float = 0.0,
    grad_clip_norm: float = 0.0,
    grad_stats: Optional[dict] = None,
) -> dict[str, torch.Tensor]:
    r"""K-step rollout training with per-step loss accumulation.

    Expects ``batch`` to carry sequence keys produced by
    :class:`physicsnemo.experimental.datapipes.plasim.SequenceDataset`:

    * ``surface_in_seq``:        ``(B, T+1, C_s, H, W)``
    * ``upper_air_in_seq``:      ``(B, T+1, C_u, L, H, W)``
    * ``varying_boundary_seq``:  ``(B, T+1, C_b, H, W)``
    * ``diagnostic_seq``:        ``(B, T+1, C_d, H, W)`` (when has_diagnostic)
    * ``constant_boundary``:     ``(C_b^c, H, W)`` or ``(B, C_b^c, H, W)``

    The model is unrolled K times (K = ``unroll_steps``); the prediction
    at step k is fed back as the input state for step k+1. Per-step
    losses are summed then divided by K — the resulting scalar is in the
    same scale as the single-step loss for direct LR/EMA comparability.

    VAE-KL is not supported in this code path (the multi-step rollout
    averages predictions away from the latent encoder semantics); pass
    ``vae_kl_weight=0`` (default) or use single-step
    :func:`train_step` for the VAE variant.
    """
    if "surface_in_seq" not in batch or "upper_air_in_seq" not in batch:
        raise KeyError(
            "multistep_train_step requires sequence batch keys "
            "(`*_seq`). Use the datapipe in unroll_steps>1 mode."
        )
    if int(unroll_steps) < 1:
        raise ValueError(f"unroll_steps must be ≥ 1, got {unroll_steps}")

    optimizer.zero_grad(set_to_none=True)

    if amp_dtype is None:
        amp_ctx = contextlib.nullcontext()
    else:
        device_type = "cuda" if batch["surface_in_seq"].is_cuda else "cpu"
        amp_ctx = torch.amp.autocast(device_type=device_type, dtype=amp_dtype)

    surface_seq = batch["surface_in_seq"]               # (B, T+1, C_s, H, W)
    upper_seq = batch["upper_air_in_seq"]               # (B, T+1, C_u, L, H, W)
    varying_seq = batch["varying_boundary_seq"]         # (B, T+1, C_b, H, W)
    diag_seq = batch.get("diagnostic_seq") if has_diagnostic else None
    const_boundary = batch.get("constant_boundary")     # (C, H, W) or (B, C, H, W)

    # Initial state = first frame. Perturbed like the single-step path — and
    # ONLY here: PanguWeather perturbs the loaded input sample, not each
    # autoregressive state the model produces during the rollout.
    state_surface = surface_seq[:, 0]
    state_upper = upper_seq[:, 0]
    state_surface, state_upper = perturb_inputs(state_surface, state_upper, epsilon_factor)

    accum_components = {
        "surface": torch.zeros((), device=state_surface.device, dtype=state_surface.dtype),
        "upper_air": torch.zeros((), device=state_surface.device, dtype=state_surface.dtype),
        "diagnostic": torch.zeros((), device=state_surface.device, dtype=state_surface.dtype),
    }
    accum_loss = torch.zeros((), device=state_surface.device, dtype=state_surface.dtype)

    # One range spans the whole rollout: the K sub-steps are not separable phases
    # and per-sub-step ranges would multiply `forward_loss` entries by K, which
    # parse_nsys.py would average into a meaningless per-range median.
    with _nvtx_range("forward_loss"), amp_ctx:
        for k in range(int(unroll_steps)):
            boundary_in = varying_seq[:, k]
            out = model(
                state_surface,
                const_boundary,
                boundary_in,
                state_upper,
            )
            if has_diagnostic:
                next_surface, next_upper, next_diag = out[0], out[1], out[2]
            else:
                next_surface, next_upper = out[0], out[1]
                next_diag = None

            target_surface_k = surface_seq[:, k + 1]
            target_upper_k = upper_seq[:, k + 1]
            target_diag_k = diag_seq[:, k + 1] if diag_seq is not None else None

            losses_k = loss_fn(
                next_surface,
                next_upper,
                target_surface_k,
                target_upper_k,
                out_diagnostic=next_diag,
                target_diagnostic=target_diag_k,
            )
            accum_loss = accum_loss + losses_k["loss"]
            for comp in ("surface", "upper_air", "diagnostic"):
                if comp in losses_k:
                    val = losses_k[comp]
                    if not isinstance(val, torch.Tensor):
                        val = torch.tensor(float(val), device=accum_loss.device, dtype=accum_loss.dtype)
                    accum_components[comp] = accum_components[comp] + val

            # Detach the boundary path (no grad through it) but keep the
            # state path so per-step gradients flow back through the rollout.
            state_surface = next_surface
            state_upper = next_upper

    total = accum_loss / float(unroll_steps)
    avg_components = {
        k: (v / float(unroll_steps)).detach() for k, v in accum_components.items()
    }
    losses_out: dict[str, torch.Tensor] = {
        "loss": total,
        "surface": avg_components["surface"],
        "upper_air": avg_components["upper_air"],
        "diagnostic": avg_components["diagnostic"],
        "vae_kl": torch.zeros((), device=total.device, dtype=total.dtype),
    }

    if grad_scaler is not None:
        with _nvtx_range("backward"):
            grad_scaler.scale(total).backward()
        clip_and_measure_grads(
            model, grad_clip_norm=grad_clip_norm, grad_scaler=grad_scaler,
            optimizer=optimizer, grad_stats=grad_stats,
        )
        with _nvtx_range("optimizer"):
            grad_scaler.step(optimizer)
            grad_scaler.update()
    else:
        with _nvtx_range("backward"):
            total.backward()
        clip_and_measure_grads(
            model, grad_clip_norm=grad_clip_norm, grad_stats=grad_stats,
        )
        with _nvtx_range("optimizer"):
            optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return losses_out


_OPTIONAL_MODEL_BATCH_KEYS = ("surface_prev_in", "upper_air_prev_in", "calendar")


def _optional_model_kwargs(
    model: torch.nn.Module, batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Extra forward kwargs a model wants that also exist in the batch.

    Returns the subset of ``{surface_prev_in, upper_air_prev_in, calendar}``
    that are BOTH present in ``batch`` AND named parameters of the model's
    ``forward``. SFNO/Pangu forwards don't name these, so the result is empty
    and their call is byte-for-byte unchanged; ArchesWeather names all three.
    """
    inner = _unwrap_model(model)
    varnames = getattr(
        inner.forward, "__code__", type("_x", (), {"co_varnames": ()})()
    ).co_varnames
    return {k: batch[k] for k in _OPTIONAL_MODEL_BATCH_KEYS if k in batch and k in varnames}


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Peel DDP and ``torch.compile`` wrappers to reach the real module.

    Both helpers below read the *forward signature* to decide how to call the
    model. ``torch.compile`` returns an ``OptimizedModule`` whose ``forward`` is
    dynamo's wrapper, so introspecting it reports the WRONG signature: measured
    on a model whose forward takes ``train=``, the eager object answers True and
    the compiled object answers False. That would silently drop ``train=True``
    and the target kwargs under compile — flipping activation checkpointing off
    and changing the forward path, i.e. compiling would change what the model
    computes (CLAUDE.md #1). Unwrapping first keeps compiled and eager identical.

    ``_orig_mod`` is checked before ``module`` because ``OptimizedModule``
    forwards unknown attributes to the wrapped model. Bounded so a pathological
    wrapper chain cannot spin.
    """
    m = model
    for _ in range(4):
        if hasattr(m, "_orig_mod"):  # torch.compile
            m = m._orig_mod
        elif isinstance(getattr(m, "module", None), torch.nn.Module):  # DDP
            m = m.module
        else:
            break
    return m


def _model_accepts_train_kwarg(model: torch.nn.Module) -> bool:
    """Detect whether the model's forward signature accepts ``train=`` + targets.

    The faithful PanguPlasim port takes a ``train`` flag plus optional
    ``target_*`` kwargs (it routes them through the VAE's second encoder when
    ``train=True``). PanguPlasimLegacy doesn't — its forward only takes the
    four input tensors.
    """
    inner = _unwrap_model(model)
    return getattr(inner, "has_vae", False) or "train" in getattr(
        inner.forward, "__code__", type("_x", (), {"co_varnames": ()})()
    ).co_varnames
