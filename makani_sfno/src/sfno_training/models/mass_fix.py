"""Dry-air mass fix — ACE2's ``conserve_dry_air`` adapted to the makani E3SM channels.

⚠ DIAGNOSTIC ARM ONLY. This changes what the model computes (CLAUDE.md "science is
jesswan's"). It is off unless ``params.conserve_dry_air`` is true, and it may not
become a default without jesswan's sign-off
(``polaris_makani_finetune_stability_handoff.md`` §7).

ACE2 (``fme/core/corrector/atmosphere.py:_force_conserve_dry_air``) holds the
area-weighted global mean of dry-air surface pressure,
``ps - Σ_k (Δa_k + Δb_k·ps)·q_k`` (hybrid levels, specific total water), equal to
the input's, by a globally uniform shift of each column's dry-air pressure. The
makani pack has no per-level specific humidity, but it has the column total
``TMQ`` (kg m⁻²), and the water column's surface-pressure share is ``g·TMQ``:

    dry = PS - g·TMQ
    err = gm(dry_pred) - gm(dry_input)          (area-weighted, equiangular)
    PS_pred ← PS_pred - err                     (uniform; TMQ untouched)

``Σ_k (Δa_k + Δb_k·ps)·q_k = Σ_k Δp_k·q_k = g·(column water)``, so the dry-air
quantity is the same. The update differs slightly: ACE2 holds the mixing ratios
``q_k`` fixed, so its column water scales with the corrected ``ps`` (the ``Δb``
term in its solve). Here the column water ``TMQ`` is held fixed instead, which
is the natural choice when water is a column mass and changes the result only
by ``err·TMQ·g/ps`` ≈ 1e-3 of the (hPa-scale) correction.

Everything is done in the model's z-space: ``PS = z·σ + μ`` is affine per channel,
so a uniform physical shift ``err`` is a uniform z shift ``err/σ_PS``. The
correction is computed in float64 and returned in the prediction's dtype.
"""
from __future__ import annotations

import math

import numpy as np
import torch

GRAVITY = 9.80665  # m s-2, the value fme.core.constants uses


def equiangular_weights(nlat: int) -> np.ndarray:
    """cos-lat cell weights for a cell-centred equiangular grid, summing to 1.

    Identical to ``sfno_ensemble.scores.equiangular_weights`` (duplicated so the
    model package does not import the evaluation package).
    """
    lat = np.deg2rad(90.0 - (np.arange(nlat) + 0.5) * (180.0 / nlat))
    w = np.cos(lat)
    return w / w.sum()


class DryAirFix:
    """Callable ``(inp_z, pred_z) -> pred_z`` with the dry-air constraint applied.

    ``inp_z``  ``(B, C_in, H, W)``: the step's input state, z-space; channel order
               starts with the state channels.
    ``pred_z`` ``(B, C_out, H, W)``: the prediction, z-space, state channels first.
    """

    def __init__(self, *, channel_names, global_means, global_stds, nlat: int,
                 ps_name: str = "PS", tmq_name: str = "TMQ"):
        names = list(channel_names)
        for n in (ps_name, tmq_name):
            if n not in names:
                raise ValueError(f"DRY_AIR_FIX: channel {n} not in channel_names")
        self.i_ps, self.i_q = names.index(ps_name), names.index(tmq_name)
        mu = np.asarray(global_means, dtype=np.float64).reshape(-1)
        sd = np.asarray(global_stds, dtype=np.float64).reshape(-1)
        if mu.size < len(names) or sd.size < len(names):
            raise ValueError(f"DRY_AIR_FIX: stats have {mu.size}/{sd.size} channels, "
                             f"need {len(names)}")
        self.mu_ps, self.sd_ps = float(mu[self.i_ps]), float(sd[self.i_ps])
        self.mu_q, self.sd_q = float(mu[self.i_q]), float(sd[self.i_q])
        self.w = torch.as_tensor(equiangular_weights(nlat), dtype=torch.float64)
        if not (math.isfinite(self.sd_ps) and self.sd_ps > 0):
            raise ValueError(f"DRY_AIR_FIX: global std of {ps_name} is {self.sd_ps}")

    def _gm_dry(self, x: torch.Tensor) -> torch.Tensor:
        """Area-weighted global mean of PS - g·TMQ, physical Pa, shape (B,)."""
        x = x.to(torch.float64)
        ps = x[:, self.i_ps] * self.sd_ps + self.mu_ps
        q = x[:, self.i_q] * self.sd_q + self.mu_q
        dry = ps - GRAVITY * q                                   # (B, H, W)
        w = self.w.to(dry.device)
        return dry.mean(dim=-1) @ w

    def __call__(self, inp_z: torch.Tensor, pred_z: torch.Tensor) -> torch.Tensor:
        if pred_z.shape[-2] != self.w.numel():
            raise ValueError(f"DRY_AIR_FIX: nlat {pred_z.shape[-2]} != weights {self.w.numel()}")
        err = self._gm_dry(pred_z) - self._gm_dry(inp_z)          # (B,) Pa
        shift = (err / self.sd_ps)[:, None, None]                  # float64, z units
        # ⚠ Promote bf16/fp16 to float32. The per-step dry-air error is a few Pa,
        # far below one bf16 step of PS (σ_PS/256 at |z|~1); a bf16 value shifted by
        # that and rounded back to bf16 is returned UNCHANGED, so a bf16 result would
        # silently undo the fix (fix-on screen 7650512: B_e01 dry drift -30.9 hPa
        # with the fix on vs -31.0 off). Promotion is lossless for the other channels.
        out_dtype = torch.promote_types(pred_z.dtype, torch.float32)
        out = pred_z.to(out_dtype).clone()
        out[:, self.i_ps] = (pred_z[:, self.i_ps].to(torch.float64) - shift).to(out_dtype)
        return out
