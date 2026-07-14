"""metrics — lat-weighted RMSE, ACC, bias maps.

Implements docs/sfno_eval_plan.md §D.1, §D.2, §D.3, §B.5.

Manual implementations rather than ``earth2studio.statistics`` because
``earth2studio.statistics.acc`` imports ``earth2studio.data`` which is
not in our sparse clone (Codex round-1 fix 3). The implementations are
short enough (~40 lines each) that owning them outright is cleaner than
a hybrid dependency.

All functions accept torch tensors. Inputs may be on any device; outputs
match the input device.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Lat weights — Legendre-Gauss quadrature
# ---------------------------------------------------------------------------

def legendre_gauss_lat_weights(nlat: int) -> torch.Tensor:
    """Return Gauss-Legendre quadrature weights normalised to sum to 1.

    Tries ``torch_harmonics.quadrature.legendre_gauss_weights(nlat, -1, 1)``
    first (Makani's bundled dep). Falls back to ``np.polynomial.legendre.leggauss``
    if torch_harmonics is unavailable.

    The function returns the **weights** only (length ``nlat``); the
    associated cosines are discarded.
    """
    try:
        import torch_harmonics as th
        nodes_weights = th.quadrature.legendre_gauss_weights(nlat, -1.0, 1.0)
        # API returns (cos_thetas, weights) — see verified-API note in §B.5.
        weights = nodes_weights[1]
        if not isinstance(weights, torch.Tensor):
            weights = torch.as_tensor(weights, dtype=torch.float64)
    except (ImportError, AttributeError):
        # Numpy fallback — same nodes/weights up to f64 precision.
        _, w = np.polynomial.legendre.leggauss(nlat)
        weights = torch.as_tensor(w, dtype=torch.float64)
    weights = weights / weights.sum()
    return weights.to(torch.float32)


def cache_lat_weights(out_path, nlat: int = 64) -> Path:
    """Compute and cache Gauss-Legendre weights to ``out_path`` as ``.npy``.

    Idempotent — overwrites if the file exists. Returns the resolved path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weights = legendre_gauss_lat_weights(nlat).numpy()
    np.save(out_path, weights)
    return out_path


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------

def rmse_lat_weighted(
    pred: torch.Tensor,
    truth: torch.Tensor,
    lat_weights: torch.Tensor,
) -> torch.Tensor:
    """Latitude-weighted RMSE.

    Parameters
    ----------
    pred, truth : torch.Tensor
        Both shaped ``(..., lat, lon)`` in physical units.
    lat_weights : torch.Tensor
        Shape ``(lat,)``, summing to 1.

    Returns
    -------
    torch.Tensor
        Shape ``(...)`` (the leading dims of pred/truth, with lat & lon
        reduced).
    """
    if pred.shape != truth.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(truth.shape)}")
    if pred.shape[-2] != lat_weights.shape[0]:
        raise ValueError(
            f"lat dim {pred.shape[-2]} does not match lat_weights {lat_weights.shape[0]}"
        )
    err2 = (pred - truth) ** 2                  # (..., lat, lon)
    err2_lon = err2.mean(dim=-1)                # (..., lat)
    w = lat_weights.to(err2_lon.dtype).to(err2_lon.device)
    err2_w = (err2_lon * w).sum(dim=-1)         # (...,)
    return err2_w.sqrt()


def rmse_lat_weighted_masked(
    pred: torch.Tensor,
    truth: torch.Tensor,
    lat_weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Latitude-weighted RMSE over unmasked cells.

    Parameters
    ----------
    pred, truth : torch.Tensor
        ``(..., lat, lon)`` in physical units.
    lat_weights : torch.Tensor
        ``(lat,)``.
    mask : torch.Tensor
        ``(..., lat, lon)`` bool. True = keep, False = drop.

    Returns
    -------
    torch.Tensor
        ``(...,)``. NaN where every cell is masked out.
    """
    if pred.shape != truth.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(truth.shape)}")
    if mask.shape != pred.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} != pred shape {tuple(pred.shape)}"
        )
    lat_w = lat_weights.to(pred.dtype).to(pred.device).unsqueeze(-1)  # (lat, 1)
    w = lat_w * mask.to(pred.dtype)                                   # (..., lat, lon)
    w_sum = w.sum(dim=(-2, -1))
    err2 = (pred - truth) ** 2
    num = (err2 * w).sum(dim=(-2, -1))
    return torch.where(
        w_sum > 0,
        (num / w_sum).sqrt(),
        torch.full_like(w_sum, float("nan")),
    )


# ---------------------------------------------------------------------------
# ACC — anomaly correlation coefficient
# ---------------------------------------------------------------------------

def acc(
    pred: torch.Tensor,
    truth: torch.Tensor,
    clim_mean: torch.Tensor,
    lat_weights: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Anomaly correlation coefficient, lat-weighted.

    Parameters
    ----------
    pred, truth, clim_mean : torch.Tensor
        Each shape-broadcast-compatible at ``(..., lat, lon)``.
        ``clim_mean`` may be a single map (lat, lon) broadcast against
        any leading ``...`` dims.
    lat_weights : torch.Tensor
        Shape ``(lat,)``.
    eps : float
        Numerical floor on the denominator.

    Returns
    -------
    torch.Tensor
        Shape ``(...,)``, values in ``[-1, 1]``.
    """
    pred_anom = pred - clim_mean
    truth_anom = truth - clim_mean
    w = lat_weights.to(pred_anom.dtype).to(pred_anom.device)  # (lat,)
    w = w.unsqueeze(-1)                                       # (lat, 1) for broadcast
    num = (pred_anom * truth_anom * w).sum(dim=(-2, -1))
    den_p = ((pred_anom ** 2) * w).sum(dim=(-2, -1)).sqrt()
    den_t = ((truth_anom ** 2) * w).sum(dim=(-2, -1)).sqrt()
    return num / (den_p * den_t + eps)


def acc_masked(
    pred: torch.Tensor,
    truth: torch.Tensor,
    clim_mean: torch.Tensor,
    lat_weights: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Anomaly correlation coefficient over unmasked cells, lat-weighted.

    Same masking rule as ``rmse_lat_weighted_masked``: cells where
    ``mask`` is False are excluded from numerator, both denominators,
    and the weight sum. Returns NaN where every cell is masked out.
    """
    if mask.shape != pred.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} != pred shape {tuple(pred.shape)}"
        )
    pred_anom = pred - clim_mean
    truth_anom = truth - clim_mean
    lat_w = lat_weights.to(pred.dtype).to(pred.device).unsqueeze(-1)  # (lat, 1)
    w = lat_w * mask.to(pred.dtype)                                   # (..., lat, lon)
    w_sum = w.sum(dim=(-2, -1))
    num = (pred_anom * truth_anom * w).sum(dim=(-2, -1))
    den_p = ((pred_anom ** 2) * w).sum(dim=(-2, -1)).sqrt()
    den_t = ((truth_anom ** 2) * w).sum(dim=(-2, -1)).sqrt()
    return torch.where(
        w_sum > 0,
        num / (den_p * den_t + eps),
        torch.full_like(w_sum, float("nan")),
    )


# ---------------------------------------------------------------------------
# Bias maps
# ---------------------------------------------------------------------------

def bias_map(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Mean error field: ``mean over IC of (pred - truth)``.

    Parameters
    ----------
    pred, truth : torch.Tensor
        Shape ``(n_ic, ..., lat, lon)``. The leading IC dim is reduced.

    Returns
    -------
    torch.Tensor
        Shape ``(..., lat, lon)``.
    """
    if pred.shape != truth.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(truth.shape)}")
    return (pred - truth).mean(dim=0)
