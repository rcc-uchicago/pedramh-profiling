# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``examples/weather/ai_rossby/diagnostics.py``.

Covers ``ai_rossby_finegrained_wandb_handoff.md``: the per-var/per-level wandb
metrics that give the SFNO-E3SM parity run the same wandb detail as
PanguWeather v2.0 (``PanguWeather/v2.0/train.py``'s ``weighted_rmse_torch_channels``
/ ``weighted_rmse_torch_3D`` + ``diagnostic_log_per_iter``).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

_RECIPE_DIR = Path(__file__).resolve().parents[3] / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_RECIPE_DIR))

from diagnostics import (  # noqa: E402
    PANGU_UPPER_AIR_LEVEL_LABELS,
    pangu_style_lwrmse_logs,
    per_channel_lat_weighted_rmse,
)
from loss import PanguPlasimLoss, cos_lat_weights  # noqa: E402
from train_loop import train_step  # noqa: E402


# --------------------------------------------------------------------------- #
# per_channel_lat_weighted_rmse
# --------------------------------------------------------------------------- #


def _pangu_weighted_rmse_torch_channels(pred, target, latitudes):
    """Literal re-implementation of PanguWeather v2.0's `train.py:147-151`
    (not imported — importing across the PanguWeather/ai-rossby copy boundary
    would need both on PYTHONPATH at once, which CLAUDE.md's "Project
    independence" box says never to do)."""
    lat_weights_unweighted = torch.cos(3.1416 / 180.0 * latitudes)
    weight_1d = latitudes.size()[0] * lat_weights_unweighted / torch.sum(lat_weights_unweighted)
    weight = torch.reshape(weight_1d, (1, 1, -1, 1))
    return torch.sqrt(torch.mean(weight * (pred - target) ** 2.0, dim=(-1, -2)))


def _pangu_weighted_rmse_torch_3d(pred, target, latitudes):
    """Literal re-implementation of PanguWeather v2.0's `train.py:153-157`."""
    lat_weights_unweighted = torch.cos(3.1416 / 180.0 * latitudes)
    weight_1d = latitudes.size()[0] * lat_weights_unweighted / torch.sum(lat_weights_unweighted)
    weight = torch.reshape(weight_1d, (1, 1, 1, -1, 1))
    return torch.sqrt(torch.mean(weight * (pred - target) ** 2.0, dim=(-1, -2)))


def _phi_degrees_like_cos_lat_weights(num_lat: int) -> torch.Tensor:
    """The same cell-centered equiangular grid `cos_lat_weights` synthesizes
    internally, in degrees (float64). Deriving `latitudes` this way (rather
    than hardcoding the H=180 production grid's 89.5-degree offset) is what
    makes the tests below valid for an arbitrary H, not just 180."""
    phi = torch.linspace(
        math.pi / 2 - math.pi / (2 * num_lat),
        -math.pi / 2 + math.pi / (2 * num_lat),
        num_lat,
        dtype=torch.float64,
    )
    return phi * 180.0 / math.pi


def test_per_channel_lat_weighted_rmse_shapes():
    N, C, L, H, W = 2, 3, 4, 6, 8
    lat_weights = cos_lat_weights(H, "cpu", torch.float32)

    pred4 = torch.randn(N, C, H, W)
    tgt4 = torch.randn(N, C, H, W)
    out4 = per_channel_lat_weighted_rmse(pred4, tgt4, lat_weights)
    assert out4.shape == (N, C)

    pred5 = torch.randn(N, C, L, H, W)
    tgt5 = torch.randn(N, C, L, H, W)
    out5 = per_channel_lat_weighted_rmse(pred5, tgt5, lat_weights)
    assert out5.shape == (N, C, L)


def test_per_channel_lat_weighted_rmse_matches_pangu_formula_channels():
    torch.manual_seed(0)
    N, C, H, W = 3, 4, 10, 12
    # PanguWeather's own weight formula takes raw latitudes in DEGREES (its
    # `latitude_weighting_factor_torch`), whereas `cos_lat_weights` returns an
    # already-normalized weight vector. Feeding it the SAME grid
    # `cos_lat_weights` synthesizes internally (§3 test below proves this is
    # the real E3SM grid, not just an arbitrary choice) makes this a
    # same-weights, reduction-formula-only comparison.
    latitudes = _phi_degrees_like_cos_lat_weights(H)
    pred = torch.randn(N, C, H, W, dtype=torch.float64)
    tgt = torch.randn(N, C, H, W, dtype=torch.float64)

    expected = _pangu_weighted_rmse_torch_channels(pred, tgt, latitudes)
    lat_weights = cos_lat_weights(H, "cpu", torch.float64)
    got = per_channel_lat_weighted_rmse(pred, tgt, lat_weights)
    # `per_channel_lat_weighted_rmse` always reduces in float32 by design (its
    # own docstring) regardless of the input dtype, so compare values only at
    # float32 precision, not dtype.
    torch.testing.assert_close(got, expected.float(), rtol=1e-5, atol=1e-6)


def test_per_channel_lat_weighted_rmse_matches_pangu_formula_3d():
    torch.manual_seed(1)
    N, C, L, H, W = 2, 3, 5, 8, 9
    latitudes = _phi_degrees_like_cos_lat_weights(H)
    pred = torch.randn(N, C, L, H, W, dtype=torch.float64)
    tgt = torch.randn(N, C, L, H, W, dtype=torch.float64)

    expected = _pangu_weighted_rmse_torch_3d(pred, tgt, latitudes)
    lat_weights = cos_lat_weights(H, "cpu", torch.float64)
    got = per_channel_lat_weighted_rmse(pred, tgt, lat_weights)
    torch.testing.assert_close(got, expected.float(), rtol=1e-5, atol=1e-6)


def test_per_channel_lat_weighted_rmse_zero_when_pred_equals_target():
    lat_weights = cos_lat_weights(6, "cpu", torch.float32)
    x = torch.randn(2, 3, 6, 7)
    out = per_channel_lat_weighted_rmse(x, x, lat_weights)
    assert torch.all(out == 0)


# --------------------------------------------------------------------------- #
# §3 — cos_lat_weights vs. PanguWeather's real E3SM grid
# --------------------------------------------------------------------------- #


def test_cos_lat_weights_grid_matches_pangu_e3sm_lat_array():
    """PanguWeather's production `lat:` array
    (`PanguWeather/v2.0/config/E3SM_SFNO_H5_POLARIS_ALLDATA.yaml:213-228`,
    confirmed against the rendered production config on disk) is 180
    cell-centered points, -89.5 to 89.5, 1-degree spacing, ascending (south to
    north) — the exact mirror of `cos_lat_weights`'s synthesized `phi` grid,
    which is descending (north to south). Verified numerically here (not
    assumed) — handoff §3's required check before trusting any new number.
    """
    num_lat = 180
    pangu_lat_ascending = torch.tensor(
        [-89.5 + i for i in range(num_lat)], dtype=torch.float64
    )

    phi_deg_ascending = torch.flip(_phi_degrees_like_cos_lat_weights(num_lat), dims=[0])

    torch.testing.assert_close(
        pangu_lat_ascending, phi_deg_ascending, rtol=0, atol=1e-12
    )

    # And the normalization formulas agree too (`N*w/sum(w) == w/mean(w)`,
    # handoff §3's second claim, verified rather than assumed).
    w_pangu = torch.cos(math.pi / 180.0 * pangu_lat_ascending)
    w_pangu = num_lat * w_pangu / torch.sum(w_pangu)
    w_rossby = torch.flip(cos_lat_weights(num_lat, "cpu", torch.float64), dims=[0])
    torch.testing.assert_close(w_pangu, w_rossby, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- #
# pangu_style_lwrmse_logs — key format + de-normalization
# --------------------------------------------------------------------------- #


def test_pangu_upper_air_level_labels_match_pangu_e3sm_nominal_list():
    """Pins `PANGU_UPPER_AIR_LEVEL_LABELS` to PanguWeather's literal
    `SFNO.levels:` list (`E3SM_SFNO_H5_POLARIS_ALLDATA.yaml:158`) — the ROUNDED
    labels `data_loader_multifiles.py:527-528` actually keys wandb with, NOT
    the full-precision `sigma_levels:` ai-rossby's own `cfg.model.levels`
    holds. A future edit that "fixes" this to the full-precision values would
    silently stop merging with PanguWeather's panel.
    """
    expected = (
        5.0, 10.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 250.0,
        300.0, 400.0, 500.0, 600.0, 700.0, 850.0, 925.0, 1000.0,
    )
    assert PANGU_UPPER_AIR_LEVEL_LABELS == expected
    assert len(PANGU_UPPER_AIR_LEVEL_LABELS) == 18


def test_pangu_style_lwrmse_logs_key_format_exact_strings():
    N, Cs, Cu, L, Cd = 2, 2, 1, 3, 1
    surface_variables = ["TREFHT", "TSOI_10CM"]
    upper_air_variables = ["T"]
    diagnostic_variables = ["PRECT"]
    level_labels = (5.0, 10.0, 850.0)

    logs = pangu_style_lwrmse_logs(
        surface_lwrmse=torch.ones(N, Cs),
        upper_air_lwrmse=torch.ones(N, Cu, L),
        diagnostic_lwrmse=torch.ones(N, Cd),
        surface_variables=surface_variables,
        upper_air_variables=upper_air_variables,
        diagnostic_variables=diagnostic_variables,
        surface_std=torch.ones(Cs, 1, 1),
        upper_air_std=torch.ones(Cu, L, 1, 1),
        diagnostic_std=torch.ones(Cd, 1, 1),
        level_labels=level_labels,
    )

    expected_keys = {
        "train_TREFHT_lwrmse",
        "train_TSOI_10CM_lwrmse",
        "train_PRECT_lwrmse",
        "train_T_level5.0000_lwrmse",
        "train_T_level10.0000_lwrmse",
        "train_T_level850.0000_lwrmse",
    }
    assert set(logs.keys()) == expected_keys
    for v in logs.values():
        assert torch.is_tensor(v) and v.ndim == 0


# The full E3SM-parity variable contract (handoff §2a) — identical order in
# both harnesses' rendered configs, verified there against
# `PanguWeather/v2.0/config/E3SM_SFNO_H5_POLARIS_ALLDATA.yaml` and
# `physicsnemo_ai_rossby/.../conf/model/sfno_e3sm_parity.yaml`, NOT re-derived
# from memory here.
E3SM_SURFACE_VARIABLES = [
    "TREFHT", "U10", "RHREFHT", "PS", "PSL", "TMQ", "SOILWATER_10CM", "TSOI_10CM",
]
E3SM_UPPER_AIR_VARIABLES = ["T", "U", "V", "Z3", "RELHUM"]
E3SM_DIAGNOSTIC_VARIABLES = ["FSNTOA", "FSNT", "PRECT"]


def test_which_variables_are_tracked_and_key_count_matches_parity_gate():
    """Shows exactly which variables the per-iteration wandb block tracks, and
    asserts the total matches the SFNO-E3SM parity gate's own channel
    arithmetic: 8 surface + 5*18 upper-air + 3 diagnostic = 101
    (`out_chans`, `compare_sfno_parity.py` / handoff §2a). A dropped or
    permuted variable here would silently produce fewer/different keys
    without this test ever failing on shape alone.
    """
    assert len(E3SM_SURFACE_VARIABLES) == 8
    assert len(E3SM_UPPER_AIR_VARIABLES) == 5
    assert len(E3SM_DIAGNOSTIC_VARIABLES) == 3
    assert len(PANGU_UPPER_AIR_LEVEL_LABELS) == 18

    N = 2
    Cs, Cu, L, Cd = (
        len(E3SM_SURFACE_VARIABLES),
        len(E3SM_UPPER_AIR_VARIABLES),
        len(PANGU_UPPER_AIR_LEVEL_LABELS),
        len(E3SM_DIAGNOSTIC_VARIABLES),
    )
    logs = pangu_style_lwrmse_logs(
        surface_lwrmse=torch.ones(N, Cs),
        upper_air_lwrmse=torch.ones(N, Cu, L),
        diagnostic_lwrmse=torch.ones(N, Cd),
        surface_variables=E3SM_SURFACE_VARIABLES,
        upper_air_variables=E3SM_UPPER_AIR_VARIABLES,
        diagnostic_variables=E3SM_DIAGNOSTIC_VARIABLES,
        surface_std=torch.ones(Cs, 1, 1),
        upper_air_std=torch.ones(Cu, L, 1, 1),
        diagnostic_std=torch.ones(Cd, 1, 1),
    )

    EXPECTED_TOTAL = 8 + 5 * 18 + 3  # == 101, matches out_chans
    assert EXPECTED_TOTAL == 101
    assert len(logs) == EXPECTED_TOTAL

    # Spot-check one key per group, exact string, so a formatting regression
    # (not just a count regression) fails loudly too.
    for expected_key in (
        "train_TREFHT_lwrmse",
        "train_TSOI_10CM_lwrmse",
        "train_PRECT_lwrmse",
        "train_T_level5.0000_lwrmse",
        "train_RELHUM_level1000.0000_lwrmse",
    ):
        assert expected_key in logs

    # No accidental collisions between the three groups' key namespaces.
    assert len(set(logs.keys())) == EXPECTED_TOTAL


def test_pangu_style_lwrmse_logs_denormalizes_correctly():
    """Std multiply is the scalar identity `std * RMSE(pred_z, target_z) ==
    RMSE(pred, target)` (handoff §2b) — check it isn't missed or doubled."""
    N, Cs = 4, 1
    known_rmse_normalized = 2.0
    known_std = 3.5

    logs = pangu_style_lwrmse_logs(
        surface_lwrmse=torch.full((N, Cs), known_rmse_normalized),
        upper_air_lwrmse=torch.zeros(N, 0, 0),
        diagnostic_lwrmse=None,
        surface_variables=["X"],
        upper_air_variables=[],
        diagnostic_variables=[],
        surface_std=torch.full((Cs, 1, 1), known_std),
        upper_air_std=torch.zeros(0, 0, 1, 1),
        diagnostic_std=None,
        level_labels=(),
    )
    assert torch.allclose(logs["train_X_lwrmse"], torch.tensor(known_rmse_normalized * known_std))


def test_pangu_style_lwrmse_logs_skips_diagnostic_when_std_none():
    """`diagnostic_std` is None when `normalize_diagnostic=False` — there is
    nothing to de-normalize by, so those keys must be omitted, not crash."""
    N, Cs = 2, 1
    logs = pangu_style_lwrmse_logs(
        surface_lwrmse=torch.ones(N, Cs),
        upper_air_lwrmse=torch.zeros(N, 0, 0),
        diagnostic_lwrmse=torch.ones(N, 1),
        surface_variables=["X"],
        upper_air_variables=[],
        diagnostic_variables=["PRECT"],
        surface_std=torch.ones(Cs, 1, 1),
        upper_air_std=torch.zeros(0, 0, 1, 1),
        diagnostic_std=None,
        level_labels=(),
    )
    assert "train_PRECT_lwrmse" not in logs
    assert "train_X_lwrmse" in logs


def test_pangu_style_lwrmse_logs_rejects_mismatched_level_labels():
    with pytest.raises(ValueError, match="level_labels"):
        pangu_style_lwrmse_logs(
            surface_lwrmse=torch.zeros(1, 0),
            upper_air_lwrmse=torch.zeros(1, 1, 2),
            diagnostic_lwrmse=None,
            surface_variables=[],
            upper_air_variables=["T"],
            diagnostic_variables=[],
            surface_std=torch.zeros(0, 1, 1),
            upper_air_std=torch.ones(1, 2, 1, 1),
            diagnostic_std=None,
            level_labels=(5.0,),  # length 1, but the level axis has 2
        )


# --------------------------------------------------------------------------- #
# train_step(capture_outputs=...) wiring
# --------------------------------------------------------------------------- #


class _TinyModel(torch.nn.Module):
    """Minimal stand-in with SfnoPlasim's forward shape (no `train=` kwarg,
    no VAE) — enough for `_model_accepts_train_kwarg`/`_optional_model_kwargs`
    to route `train_step` down the plain (non-VAE) call path."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, surface_in, constant_boundary, varying_boundary, upper_air_in):
        return surface_in * self.scale, upper_air_in * self.scale


def _tiny_batch(N=2, Cs=3, Cu=2, L=4, H=6, W=8):
    return {
        "surface_in": torch.randn(N, Cs, H, W),
        "constant_boundary": torch.zeros(N, 1, H, W),
        "varying_boundary": torch.zeros(N, 1, H, W),
        "upper_air_in": torch.randn(N, Cu, L, H, W),
        "target_surface": torch.randn(N, Cs, H, W),
        "target_upper_air": torch.randn(N, Cu, L, H, W),
    }


def _tiny_loss(H, Cs=3, Cu=2):
    return PanguPlasimLoss(
        surface_variables=[f"s{i}" for i in range(Cs)],
        upper_air_variable_names=[f"u{i}" for i in range(Cu)],
        diagnostic_variables=[],
        num_lat=H,
        loss_type="l2",
        latitude_weighted=False,
    )


def test_train_step_capture_outputs_populates_expected_tensors():
    torch.manual_seed(0)
    N, Cs, Cu, L, H, W = 2, 3, 2, 4, 6, 8
    model = _TinyModel()
    loss_fn = _tiny_loss(H, Cs, Cu)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = _tiny_batch(N, Cs, Cu, L, H, W)

    capture: dict = {}
    losses = train_step(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=None,
        batch=batch,
        has_diagnostic=False,
        capture_outputs=capture,
    )

    assert set(losses) == {"loss", "surface", "upper_air", "diagnostic", "vae_kl"}
    assert set(capture) == {
        "out_surface", "out_upper_air", "out_diagnostic",
        "target_surface", "target_upper_air", "target_diagnostic",
    }
    assert capture["out_surface"].shape == (N, Cs, H, W)
    assert capture["out_upper_air"].shape == (N, Cu, L, H, W)
    assert capture["out_diagnostic"] is None
    assert capture["target_diagnostic"] is None
    assert capture["target_surface"].shape == (N, Cs, H, W)
    assert not capture["out_surface"].requires_grad


def test_train_step_without_capture_outputs_is_unaffected():
    """Existing callers (profile_train.py, equivalence.py, equivalence_ddp.py)
    don't pass `capture_outputs` — confirm the default stays a true no-op."""
    torch.manual_seed(0)
    N, Cs, Cu, L, H, W = 2, 3, 2, 4, 6, 8
    model = _TinyModel()
    loss_fn = _tiny_loss(H, Cs, Cu)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = _tiny_batch(N, Cs, Cu, L, H, W)

    losses = train_step(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=None,
        batch=batch,
        has_diagnostic=False,
    )
    assert set(losses) == {"loss", "surface", "upper_air", "diagnostic", "vae_kl"}


def test_train_step_capture_then_diagnostics_end_to_end():
    """The full path: train_step's capture -> per_channel_lat_weighted_rmse ->
    pangu_style_lwrmse_logs, exactly as wired in train.py, with sane
    physical-unit magnitudes (handoff §5's smoke check) — surface_std scales
    a known residual to a known physical value, not an O(1) normalized one."""
    torch.manual_seed(0)
    N, Cs, Cu, L, H, W = 2, 2, 1, 3, 6, 8
    surface_names = ["s0", "s1"]
    upper_names = ["u0"]
    model = _TinyModel()
    loss_fn = _tiny_loss(H, Cs, Cu)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = _tiny_batch(N, Cs, Cu, L, H, W)

    capture: dict = {}
    train_step(
        model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=None,
        batch=batch, has_diagnostic=False, capture_outputs=capture,
    )

    lat_weights = cos_lat_weights(H, "cpu", torch.float32)
    surface_lwrmse = per_channel_lat_weighted_rmse(
        capture["out_surface"], capture["target_surface"], lat_weights
    )
    upper_air_lwrmse = per_channel_lat_weighted_rmse(
        capture["out_upper_air"], capture["target_upper_air"], lat_weights
    )
    assert surface_lwrmse.shape == (N, Cs)
    assert upper_air_lwrmse.shape == (N, Cu, L)

    # A large surface_std (e.g. temperature-Kelvin-scale) must scale the
    # O(1) normalized RMSE up accordingly -- catches a missed `* std`.
    surface_std = torch.tensor([300.0, 1.0]).view(Cs, 1, 1)
    upper_air_std = torch.ones(Cu, L, 1, 1)
    logs = pangu_style_lwrmse_logs(
        surface_lwrmse=surface_lwrmse,
        upper_air_lwrmse=upper_air_lwrmse,
        diagnostic_lwrmse=None,
        surface_variables=surface_names,
        upper_air_variables=upper_names,
        diagnostic_variables=[],
        surface_std=surface_std,
        upper_air_std=upper_air_std,
        diagnostic_std=None,
        level_labels=(5.0, 10.0, 850.0),
    )
    normalized_s0 = float(torch.mean(surface_lwrmse[:, 0]))
    assert math.isclose(float(logs["train_s0_lwrmse"]), normalized_s0 * 300.0, rel_tol=1e-5)
    assert set(logs.keys()) == {
        "train_s0_lwrmse", "train_s1_lwrmse",
        "train_u0_level5.0000_lwrmse",
        "train_u0_level10.0000_lwrmse",
        "train_u0_level850.0000_lwrmse",
    }
