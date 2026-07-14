"""EMA (Exponential Moving Average) helper for the PlaSim → Makani SFNO trainer.

Public surface (per docs/2026-05-02_ema_implementation_plan.md §7.1):

* ``EMAModel`` — per-rank shadow of a model's trainable parameters with a
  Karras-style warmup schedule (``decay_t = min(decay_max, (1+t)/(10+t))``).
  Shadow dtype is dtype-aware (§3.2): complex parameters keep their complex
  dtype (so SFNO's ``complex64`` spectral weights do not lose imaginary
  components), half-precision real parameters shadow in ``float32`` for
  accumulation precision, and other real parameters shadow in their native
  dtype.

* ``EMAModel.update(model)`` — invoked from a post-step optimizer hook;
  fires only on non-skipped ``GradScaler.step`` invocations.

* ``EMAModel.applied_to(model)`` — context manager: snapshot live params,
  copy the EMA shadow into ``model.parameters()`` in-place, yield, and
  restore the snapshot in ``finally``. Used by ``PlasimTrainer.validate_one_epoch``
  for the second (EMA) validation pass.

* ``EMAModel.export_model_state(model)`` — produce a FULL canonical
  state_dict (model.state_dict with wrapper prefixes stripped) where the
  trainable-parameter values are replaced with the EMA shadow (cast to the
  parameter's dtype). Buffers and any untracked params are copied through
  unchanged. ``sharded_dims_mp`` is mirrored from live tensors. Drives
  ``best_ckpt_ema_mp{mp_rank}.tar`` so that
  ``Driver.restore_from_checkpoint(..., strict=True)`` succeeds at
  inference time without an inference-side change.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn


def _get_model_state_dict_prefix(model: nn.Module) -> str:
    """Wrapper-prefix detection mirroring ``makani.utils.checkpoint_helpers.get_model_state_dict_prefix``.

    Inlined here so this module is importable without the full Makani
    dependency chain (helpful for unit tests that run on a CPU-only node).
    Tracks ``_orig_mod.`` (``torch.compile``) and ``module.`` (DDP) layers
    in any nested combination.
    """
    prefix = ""
    m = model
    while True:
        if hasattr(m, "_orig_mod"):
            prefix += "_orig_mod."
            m = m._orig_mod
        elif isinstance(m, nn.parallel.DistributedDataParallel):
            prefix += "module."
            m = m.module
        else:
            break
    return prefix


def _shadow_dtype_for(param: torch.Tensor) -> torch.dtype:
    """Pick the shadow dtype for ``param`` (per plan §3.2).

    * Complex parameters keep their complex dtype — casting ``complex64``
      to ``float32`` silently drops the imaginary component, which would
      corrupt SFNO's spectral weights.
    * fp16 / bf16 parameters shadow in fp32 for accumulation precision
      (the classic AMP-EMA pattern).
    * Other real parameters shadow in their native dtype.
    """
    if param.is_complex():
        return param.dtype
    if param.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return param.dtype


class EMAModel:
    """Per-rank Exponential Moving Average of a model's trainable parameters.

    Parameters
    ----------
    model : nn.Module
        Live model. Wrapper prefixes (``_orig_mod.`` from ``torch.compile``,
        ``module.`` from DDP) are stripped from shadow keys so they match
        the canonical state_dict form used by ``Driver._save_checkpoint_legacy``.
    decay : float
        Asymptotic decay (``decay_max``).
    warmup : bool
        If True, ``decay_t = min(decay_max, (1+t)/(10+t))`` with ``t``
        being the post-update count (first update sees ``t=1`` ⇒ decay
        ≈ ``2/11 ≈ 0.182``). If False, ``decay_t == decay_max`` from t=1.
    """

    CONFIG_VERSION: int = 1

    def __init__(self, model: nn.Module, *, decay: float = 0.999, warmup: bool = True) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EMAModel decay must lie in (0, 1); got {decay!r}")
        self.decay_max = float(decay)
        self.warmup = bool(warmup)
        self._step: int = 0

        self._prefix = _get_model_state_dict_prefix(model)
        self._shadow: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            stripped = name.removeprefix(self._prefix)
            shadow = torch.empty_like(p, dtype=_shadow_dtype_for(p), device=p.device)
            shadow.copy_(p.detach().to(shadow.dtype))
            self._shadow[stripped] = shadow

    # ------------------------------------------------------------------
    # Step accounting
    # ------------------------------------------------------------------
    @property
    def step(self) -> int:
        return self._step

    @step.setter
    def step(self, value: int) -> None:
        self._step = int(value)

    @property
    def decay_t(self) -> float:
        """Effective decay at the current ``self._step``.

        Caller convention: ``update`` increments ``self._step`` BEFORE
        reading ``decay_t``, so the first update observes ``t=1`` and
        the warmup formula yields ``2/11 ≈ 0.182``.
        """
        if not self.warmup:
            return self.decay_max
        t = self._step
        return min(self.decay_max, (1.0 + t) / (10.0 + t))

    # ------------------------------------------------------------------
    # Update — called from optimizer post-step hook
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Increment ``self._step`` and EMA-blend each shadow with its live param.

        Step indexing is increment-then-compute: ``self._step += 1`` runs
        before ``self.decay_t`` is read, so the very first update observes
        ``t=1`` (decay ``2/11`` under warmup).
        """
        self._step += 1
        d = self.decay_t
        one_minus_d = 1.0 - d
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            stripped = name.removeprefix(self._prefix)
            shadow = self._shadow.get(stripped)
            if shadow is None:
                # New trainable parameter appeared after construction — refuse
                # silently extending: surfaces as a clean error rather than
                # quietly desyncing the shadow dict.
                raise KeyError(
                    f"EMAModel.update: live param {name!r} (stripped: {stripped!r}) "
                    f"has no shadow. Reconstruct EMAModel after model surgery."
                )
            shadow.mul_(d).add_(p.detach().to(shadow.dtype), alpha=one_minus_d)

    # ------------------------------------------------------------------
    # State_dict / load_state_dict
    # ------------------------------------------------------------------
    def state_dict(self) -> "OrderedDict[str, torch.Tensor]":
        # Return cloned tensors so callers can mutate without touching
        # live shadows; ``torch.save`` does not require this but the
        # cost is negligible vs. parameter count and avoids surprises.
        return OrderedDict((k, v.detach().clone()) for k, v in self._shadow.items())

    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, torch.Tensor]",
        *,
        strict: bool = True,
    ) -> None:
        """In-place restore of shadow tensors from a saved EMA state dict.

        Validations on the **raw** incoming tensor (BEFORE any device/dtype
        cast) — a real → complex cast silently produces a zero-imaginary
        complex tensor and would defeat the complex-vs-real check.
        """
        incoming_keys = set(state_dict.keys())
        shadow_keys = set(self._shadow.keys())
        missing = sorted(shadow_keys - incoming_keys)
        unexpected = sorted(incoming_keys - shadow_keys)

        if strict and (missing or unexpected):
            raise RuntimeError(
                "EMAModel.load_state_dict: key mismatch.\n"
                f"  missing_keys: {missing}\n"
                f"  unexpected_keys: {unexpected}"
            )
        if missing:
            import logging

            logging.getLogger("sfno_training.ema").warning(
                "EMAModel.load_state_dict: missing keys (kept fresh): %s", missing
            )
        if unexpected:
            import logging

            logging.getLogger("sfno_training.ema").warning(
                "EMAModel.load_state_dict: unexpected keys (ignored): %s", unexpected
            )

        for name, shadow in self._shadow.items():
            if name not in state_dict:
                continue
            raw = state_dict[name]

            # Complex-vs-real check on the RAW tensor — must precede the cast.
            if raw.is_complex() != shadow.is_complex():
                raise RuntimeError(
                    f"EMAModel.load_state_dict: complex/real mismatch for {name!r}: "
                    f"incoming dtype={raw.dtype}, shadow dtype={shadow.dtype}. "
                    f"Refusing to cast (would corrupt complex spectral weights)."
                )

            # Shape check on the RAW tensor.
            if raw.shape != shadow.shape:
                raise RuntimeError(
                    f"EMAModel.load_state_dict: shape mismatch for {name!r}: "
                    f"got {tuple(raw.shape)}, expected {tuple(shadow.shape)}"
                )

            incoming = raw.to(device=shadow.device, dtype=shadow.dtype, non_blocking=False)
            shadow.copy_(incoming)

    # ------------------------------------------------------------------
    # Validation swap helpers
    # ------------------------------------------------------------------
    @contextmanager
    def applied_to(self, model: nn.Module) -> Iterator[None]:
        """Temporarily install EMA shadow values into ``model.parameters()``.

        Snapshots live params on entry, copies ``shadow.to(p.dtype)`` into
        each trainable param in place, yields, then restores the snapshot
        in ``finally``. No CPU↔GPU transfer; memory cost during the swap
        is one extra copy of trainable params worth of storage.
        """
        snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
        try:
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                stripped = name.removeprefix(self._prefix)
                shadow = self._shadow.get(stripped)
                if shadow is None:
                    continue
                snapshots.append((p, p.detach().clone()))
                p.data.copy_(shadow.to(p.dtype))
            yield
        finally:
            for p, snap in snapshots:
                p.data.copy_(snap)

    # ------------------------------------------------------------------
    # Export — drives best_ckpt_ema_mp{mp_rank}.tar
    # ------------------------------------------------------------------
    def export_model_state(self, model: nn.Module) -> "OrderedDict[str, torch.Tensor]":
        """Return the FULL canonical state_dict with EMA values substituted.

        Starts from ``model.state_dict()`` (wrapper prefix stripped),
        replaces tracked trainable-parameter entries with their EMA shadow
        cast to the parameter's native dtype, and copies through buffers
        and any untracked params. ``sharded_dims_mp`` is mirrored from the
        live module so model-parallel restore works (mirrors
        ``driver.py:540-543``).

        Cloned tensors so the caller can ``torch.save`` without aliasing
        the live model state.
        """
        live = model.state_dict()
        if self._prefix:
            nn.modules.utils.consume_prefix_in_state_dict_if_present(live, self._prefix)

        param_map = {name.removeprefix(self._prefix): param for name, param in model.named_parameters()}
        buffer_map = {name.removeprefix(self._prefix): buf for name, buf in model.named_buffers()}

        out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for key, tensor in live.items():
            p = param_map.get(key)
            if p is not None and p.requires_grad and key in self._shadow:
                exported = self._shadow[key].to(p.dtype).detach().clone()
            else:
                exported = tensor.detach().clone()

            src = param_map.get(key, buffer_map.get(key))
            if src is not None and hasattr(src, "sharded_dims_mp"):
                exported.sharded_dims_mp = src.sharded_dims_mp
            out[key] = exported

        return out
