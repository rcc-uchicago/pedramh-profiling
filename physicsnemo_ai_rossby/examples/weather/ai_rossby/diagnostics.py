# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-iteration per-variable/per-level wandb diagnostics, at PanguWeather parity.

Gives the SFNO-E3SM parity run the same per-variable/per-level wandb detail as
PanguWeather v2.0's ``train.py`` (``weighted_rmse_torch_channels`` /
``weighted_rmse_torch_3D`` + ``diagnostic_log_per_iter``,
``PanguWeather/v2.0/train.py:141-157,1657-1684``), so the two harnesses' panels
overlay in the same wandb project.

Deliberately separate from :mod:`loss` — :func:`loss.lat_weighted_residual` /
:func:`loss.per_var_lat_weighted_residual` reduce to a scalar (or per-channel
scalar) by contract, and callers rely on that; this module keeps the
channel/level axis instead.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

# PanguWeather's E3SM production config carries TWO level lists for the same
# 18 hybrid ("sigma") levels (`E3SM_SFNO_H5_POLARIS_ALLDATA.yaml`):
#   `levels:`       [5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500,
#                    600, 700, 850, 925, 1000]        <- rounded nominal hPa labels
#   `sigma_levels:` [4.714998332947841, 10.655023096474308, ...]  <- true hybrid values
# `data_loader_multifiles.py:527-528` sets `self.levels = np.array(params['levels'])`
# UNCONDITIONALLY (`use_sigma_levels` only controls which list `load_mean_std`
# indexes stats with) -- so the wandb key PanguWeather actually emits
# (`train.py:1683-1684`, `f'train_{var}_level{level:.4f}_lwrmse'`) is built from
# the ROUNDED list, e.g. "train_T_level5.0000_lwrmse", never from `sigma_levels`.
#
# ai-rossby's `conf/model/sfno_e3sm_parity.yaml` `levels:` field holds the
# OTHER list -- the full-precision hybrid values (verified identical, same
# order, to PanguWeather's `sigma_levels`; see the handoff's §3 lat-grid check
# for the same kind of cross-config verification) -- because that is what
# `SfnoPlasim`/the normalizer need for by-value level matching. Formatting
# keys from `cfg.model.levels` directly would silently emit
# "train_T_level4.7150_lwrmse", which would NEVER merge with PanguWeather's
# panel. This list exists so the two harnesses key on the same rounded label
# while the *physical* std lookup still goes through the true hybrid values.
#
# Ordered top-to-bottom, positionally aligned with `cfg.model.levels` (both
# configs list the 18 levels in the same order -- checked numerically, not
# assumed).
PANGU_UPPER_AIR_LEVEL_LABELS: tuple[float, ...] = (
    5.0, 10.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 250.0,
    300.0, 400.0, 500.0, 600.0, 700.0, 850.0, 925.0, 1000.0,
)


def per_channel_lat_weighted_rmse(
    pred: torch.Tensor, target: torch.Tensor, lat_weights: torch.Tensor
) -> torch.Tensor:
    r"""Lat-weighted RMSE, reduced over (lat, lon) only -- channel/level kept.

    Mirrors PanguWeather v2.0's ``weighted_rmse_torch_channels`` /
    ``weighted_rmse_torch_3D`` (``train.py:147-157``) exactly -- same
    ``sqrt(mean(weight * (pred - target)**2, dim=(-1, -2)))`` -- generalized to
    one function since the shapes differ only by an extra level axis.

    Parameters
    ----------
    pred, target : torch.Tensor
        Shape ``(N, C, H, W)`` (surface / diagnostic) or ``(N, C, L, H, W)``
        (upper-air). ``H`` is always the second-to-last dim.
    lat_weights : torch.Tensor
        Shape ``(H,)`` -- e.g. from :func:`loss.cos_lat_weights`. Broadcast
        over every other dim.

    Returns
    -------
    torch.Tensor
        Shape ``(N, C)`` or ``(N, C, L)``.

    Notes
    -----
    Reduces in float32 regardless of ``pred``/``target``'s dtype (they may be
    bf16 under autocast) -- an RMSE diagnostic accumulated in bf16 would be
    dominated by rounding rather than signal.
    """
    shape = [1] * pred.ndim
    shape[-2] = lat_weights.shape[0]
    weight = lat_weights.view(shape).float()
    resid_sq = (pred.float() - target.float()).pow(2)
    return torch.sqrt(torch.mean(weight * resid_sq, dim=(-1, -2)))


def pangu_style_lwrmse_logs(
    *,
    surface_lwrmse: torch.Tensor,
    upper_air_lwrmse: torch.Tensor,
    diagnostic_lwrmse: Optional[torch.Tensor],
    surface_variables: Sequence[str],
    upper_air_variables: Sequence[str],
    diagnostic_variables: Sequence[str],
    surface_std: torch.Tensor,
    upper_air_std: torch.Tensor,
    diagnostic_std: Optional[torch.Tensor],
    level_labels: Sequence[float] = PANGU_UPPER_AIR_LEVEL_LABELS,
) -> dict[str, torch.Tensor]:
    r"""Denormalize + key-format per-channel RMSE to PanguWeather's exact schema.

    De-normalization is the scalar identity PanguWeather relies on
    (``train.py:1677,1680,1684``): for z-scored ``pred_z = (pred - mean) /
    std``, ``pred_z - target_z = (pred - target) / std``, so
    ``std * RMSE(pred_z, target_z) == RMSE(pred, target)`` exactly -- no need
    to denormalize the full tensors first, just scale the scalar result.

    Parameters
    ----------
    surface_lwrmse : torch.Tensor
        Shape ``(N, C_s)``, normalized-space, from :func:`per_channel_lat_weighted_rmse`.
    upper_air_lwrmse : torch.Tensor
        Shape ``(N, C_u, L)``, normalized-space.
    diagnostic_lwrmse : torch.Tensor or None
        Shape ``(N, C_d)``, normalized-space; ``None`` when the recipe has no
        diagnostic head.
    surface_variables, upper_air_variables, diagnostic_variables : sequence of str
        Channel-order names (upper-air already sigma-then-pressure order, same
        as the model's forward output).
    surface_std, upper_air_std, diagnostic_std : torch.Tensor or None
        :class:`~physicsnemo.experimental.datapipes.climate.transforms.ClimateNormalizer`
        buffers -- shapes ``(C_s, 1, 1)``, ``(C_u, L, 1, 1)``, ``(C_d, 1, 1)``.
        ``diagnostic_std`` is ``None`` when ``normalize_diagnostic=False``; keys
        for that group are skipped in that case (nothing to de-normalize by).
    level_labels : sequence of float, optional
        Nominal per-level labels for the wandb key string -- see
        :data:`PANGU_UPPER_AIR_LEVEL_LABELS`. Positional, must be the same
        length as the upper-air level axis.

    Returns
    -------
    dict of str -> torch.Tensor
        0-d (scalar) tensors, keyed exactly as PanguWeather's
        ``diagnostic_log_per_iter``: ``train_{var}_lwrmse`` (surface,
        diagnostic) and ``train_{var}_level{level:.4f}_lwrmse`` (upper-air).
    """
    if len(level_labels) != upper_air_lwrmse.shape[-1]:
        raise ValueError(
            f"level_labels has {len(level_labels)} entries but upper_air_lwrmse's "
            f"level axis has {upper_air_lwrmse.shape[-1]}"
        )

    logs: dict[str, torch.Tensor] = {}

    for j, var in enumerate(surface_variables):
        logs[f"train_{var}_lwrmse"] = (
            torch.mean(surface_lwrmse[:, j]) * surface_std[j].reshape(())
        ).reshape(())

    if diagnostic_lwrmse is not None and diagnostic_std is not None:
        for j, var in enumerate(diagnostic_variables):
            logs[f"train_{var}_lwrmse"] = (
                torch.mean(diagnostic_lwrmse[:, j]) * diagnostic_std[j].reshape(())
            ).reshape(())

    for j, var in enumerate(upper_air_variables):
        for k, level in enumerate(level_labels):
            logs[f"train_{var}_level{level:.4f}_lwrmse"] = (
                torch.mean(upper_air_lwrmse[:, j, k]) * upper_air_std[j, k].reshape(())
            ).reshape(())

    return logs
