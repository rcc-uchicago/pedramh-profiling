"""Frozen-soil moisture fix — ACE2's land ``soil_moisture_corrector`` concept adapted
to the makani E3SM channels.

⚠ DIAGNOSTIC ARM ONLY. This changes what the model computes (CLAUDE.md "science is
jesswan's"). It is off unless ``params.conserve_soil_moisture`` is true, and it may
not become a default without jesswan's sign-off
(``polaris_makani_finetune_stability_handoff.md`` §7).

ACE2's land-surface experiment config (``ACE2_retrain/ace_exp/configs/experiments/
2025-04-20-lsm-offline/ace-train-config-lsmv0.yaml``) specifies a
``soil_moisture_corrector`` (``soil_moisture_pattern``, ``soil_temperature_pattern``,
``temperature_threshold: 272.0``) alongside ``force_positive_names`` on its soil-
moisture and snow channels. Checked before writing this: that corrector's Python
implementation is not in this repo's vendored ``ace_exp`` tree —
``fme/core/corrector/atmosphere.py:AtmosphereCorrectorConfig`` has no
``soil_moisture_corrector`` field, and its loader (``dacite.Config(strict=True)``,
``fme/core/corrector/registry.py``) would refuse the config's own key outright — nor
is it on the one sibling location checked (``worktree-ace2-fme-perf-handoff``'s
``ACE2_retrain/`` is untracked there, so unreadable without checking out another
worktree). Either the config targets a newer ACE2 than what's vendored here, or it
was never run against this exact code. This is therefore a FRESH REIMPLEMENTATION of
the documented physical intent, in the same spirit as ``mass_fix.py``'s dry-air fix —
not a code port.

Physical intent: liquid soil moisture should not change via normal transport while
the soil is frozen, and soil moisture is never negative. The makani E3SM pack has one
10cm-depth soil pair (``SOILWATER_10CM``, ``TSOI_10CM``) rather than ACE2's 3-layer
soil model. Applied pointwise, per gridcell, using the step's INPUT temperature (the
state the frozen/unfrozen condition is causally known from at the start of the step):

    frozen = TSOI_10CM_input < threshold_k           (272.0 K, ACE2's own value)
    SOILWATER_10CM_pred <- where(frozen, SOILWATER_10CM_input, SOILWATER_10CM_pred)
    SOILWATER_10CM_pred <- max(SOILWATER_10CM_pred, 0)

Unlike the dry-air fix, this is a purely pointwise/local correction — no spatial
reduction, so (unlike ``DryAirFix``) it needs no area weights and no restriction on
``h_parallel_size``/``w_parallel_size``. It also doesn't need the dry-air fix's
bf16-ulp promotion care: both operations here fully REPLACE the cell's value (with
the cached input, or with the floor) rather than adding a small shift, so there is no
"shift rounds away" failure mode. Promotion to at least float32 is kept anyway, for
the same reason every other model-state arithmetic in this codebase does it.
"""
from __future__ import annotations

import numpy as np
import torch

from sfno_training.models.mass_fix import _stats_rows


class SoilMoistureFix:
    """Callable ``(inp_z, pred_z) -> pred_z`` with the frozen-soil + positivity
    constraint applied to the soil-moisture channel.

    ``inp_z``  ``(B, C_in, H, W)``: the step's input state, z-space; channel order
               starts with the state channels.
    ``pred_z`` ``(B, C_out, H, W)``: the prediction, z-space, state channels first.
    """

    def __init__(self, *, channel_names, global_means, global_stds,
                 moisture_name: str = "SOILWATER_10CM",
                 temperature_name: str = "TSOI_10CM",
                 temperature_threshold_k: float = 272.0, stats_index=None):
        """``channel_names``: the model's output channels, in tensor order.
        ``stats_index``: the run's ``out_channels`` — the stats row of each of those
        channels. Needed when the stats are wider than ``channel_names`` (a run on a
        channel subset of its pack keeps the pack's full-width stats; same rule as
        ``DryAirFix``, reusing its ``_stats_rows`` helper)."""
        names = list(channel_names)
        for n in (moisture_name, temperature_name):
            if n not in names:
                raise ValueError(f"SOIL_MOISTURE_FIX: channel {n} not in channel_names")
        self.i_soil = names.index(moisture_name)
        self.i_tsoi = names.index(temperature_name)
        mu = np.asarray(global_means, dtype=np.float64).reshape(-1)
        sd = np.asarray(global_stds, dtype=np.float64).reshape(-1)
        if mu.size != sd.size:
            raise ValueError(f"SOIL_MOISTURE_FIX: means have {mu.size} channels, stds {sd.size}")
        rows = _stats_rows(len(names), mu.size, stats_index)
        self.mu_soil, self.sd_soil = float(mu[rows[self.i_soil]]), float(sd[rows[self.i_soil]])
        self.mu_tsoi, self.sd_tsoi = float(mu[rows[self.i_tsoi]]), float(sd[rows[self.i_tsoi]])
        self.threshold_k = float(temperature_threshold_k)
        if not (np.isfinite(self.sd_soil) and self.sd_soil > 0):
            raise ValueError(f"SOIL_MOISTURE_FIX: global std of {moisture_name} is {self.sd_soil}")
        if not (np.isfinite(self.sd_tsoi) and self.sd_tsoi > 0):
            raise ValueError(f"SOIL_MOISTURE_FIX: global std of {temperature_name} is {self.sd_tsoi}")
        # physical 0 for the moisture channel, in z units (affine: z = (phys - mu) / sd)
        self._zero_z = (0.0 - self.mu_soil) / self.sd_soil
        # the frozen threshold, in z units of the temperature channel
        self._threshold_z = (self.threshold_k - self.mu_tsoi) / self.sd_tsoi

    def __call__(self, inp_z: torch.Tensor, pred_z: torch.Tensor) -> torch.Tensor:
        out_dtype = torch.promote_types(pred_z.dtype, torch.float32)
        out = pred_z.to(out_dtype).clone()
        tsoi_inp = inp_z[:, self.i_tsoi].to(out_dtype)
        soil_inp = inp_z[:, self.i_soil].to(out_dtype)
        frozen = tsoi_inp < self._threshold_z
        soil = torch.where(frozen, soil_inp, out[:, self.i_soil])
        soil = torch.clamp(soil, min=self._zero_z)
        out[:, self.i_soil] = soil
        return out
