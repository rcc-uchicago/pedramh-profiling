"""Tier 1 of the 5410 yaml regression net (per docs plan v3.1).

Allowlist presence + pinned-value test for the per-Y yaml override.
Designed to run on a Stampede3 login node in <1 s, without GPU and
without filesystem reads under ``$SCRATCH/data/...``.

Two modes
---------
1. **Default** (no ``RUN_ROOT`` env var): each test builds its yaml
   into a ``tmp_path`` fixture via ``build_per_y_yaml``. Used for
   routine ``pytest`` runs.

2. **Live** (``RUN_ROOT=<run_root>``): tests load yamls directly from
   ``<run_root>/inference/SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y<Y>.yaml``
   instead of building them. Used by verification step 3 in the plan
   to confirm the live run-root yamls match the test contract.

What is asserted (for Y ∈ {121..128})
-------------------------------------
- Presence of every yaml-derived attribute the smoke path reads
  unguardedly (allowlist below; sourced from the v3 audit).
- Pinned values (under canonical eval-track K=60):
    * ``epsilon_factor == 0.0`` (deterministic NWP, 2026-05-08 user decision)
    * ``len(save_basenames) == 1`` (single-IC invariant)
    * ``ensemble_inference_hours == (K + 1) * 6 == 366`` (Y-independent;
      year-long sentinels 8760/8784 must NOT appear)
    * ``prediction_duration_days == (K + 1) * 6 / 24 == 15.25`` (BCS
      single_ic loader span; required key per partial-horizon plan)
    * ``val_year_start == Y``, ``val_year_end == Y + 1``
    * ``save_forecasts is True``, ``log_to_wandb is False``
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_REPO = Path(
    "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0"
)

_TEST_YEARS = tuple(range(121, 129))
_LEAP_YEARS = (124, 128)

# Canonical eval-track forecast-leads horizon used by these tests.
_K = 60
_EXPECTED_HOURS = (_K + 1) * 6        # 366
_EXPECTED_DAYS = (_K + 1) * 6 / 24.0  # 15.25


# Allowlist of yaml-derived attrs the smoke path reads unguardedly.
# Sourced from the audit; grouped for readability. Anything not in
# this list either has a hasattr guard, is set dynamically by main()
# / Stepper, or is unused on the smoke path.
_REQUIRED_FROM_YAML: tuple[str, ...] = (
    # Path / filesystem
    "data_dir", "bias_data_dir", "climatology_file", "exp_dir", "load_exp_dir",
    # Calendar
    "calendar", "has_year_zero", "leap_year", "no_leap_year",
    # Time-stepping
    "timedelta_hours", "data_timedelta_hours",
    # Validation window
    "val_year_start", "val_year_end",
    # Geometry
    "horizontal_resolution", "lat", "lon", "lev",
    "num_levels", "use_sigma_levels", "sigma_levels", "levels",
    # Variable lists
    "upper_air_variables", "surface_variables", "diagnostic_variables",
    "constant_boundary_variables", "varying_boundary_variables",
    # Forecast / loss
    "predict_delta", "epsilon_factor", "forecast_lead_times",
    # Inference flags
    "save_forecasts", "save_basenames", "ensemble_inference_hours",
    "prediction_duration_days",
    # Logging
    "log_to_wandb", "log_to_screen",
    # Mean/std file basenames (read by load_mean_std)
    "surface_mean", "surface_std", "surface_ff_std",
    "upper_air_mean", "upper_air_std", "upper_air_ff_std",
    "boundary_mean", "boundary_std",
    "diagnostic_mean", "diagnostic_std",
    # SFNO architecture
    "nettype", "spectral_transform", "filter_type", "operator_type",
    "scale_factor", "embed_dim", "num_layers", "num_blocks",
    "use_mlp", "mlp_ratio", "activation_function", "encoder_layers",
    "pos_embed", "drop_rate", "drop_path_rate", "sparsity_threshold",
    "normalization_layer", "hard_thresholding_fraction",
    "use_complex_kernels", "big_skip", "rank", "factorization",
    "separable", "complex_network", "complex_activation",
    "spectral_layers", "checkpointing", "sync_norm",
    # AMP / TE
    "enable_fp8", "use_transformer_engine",
)


def _yparams_load(yaml_path: Path):
    """Load YParams under --config=SFNO. Skip cleanly off-Stampede3."""
    if not _UPSTREAM_REPO.is_dir():
        pytest.skip(f"upstream repo not present: {_UPSTREAM_REPO}")
    p = str(_UPSTREAM_REPO)
    if p not in sys.path:
        sys.path.insert(0, p)
    from utils.YParams import YParams  # type: ignore

    return YParams(str(yaml_path), "SFNO")


@pytest.fixture(scope="module")
def yamls_by_year(tmp_path_factory):
    """Yield ``{Y: yaml_path}`` for all 8 years.

    - If ``RUN_ROOT`` env var is set, return live yamls under
      ``<run_root>/inference/...``. Skips if the file is missing.
    - Otherwise build into ``tmp_path`` via ``build_per_y_yaml``.
    """
    pytest.importorskip("ruamel.yaml")
    from sfno_inference_5410.stampede3_yaml_override import (
        UPSTREAM_YAML_PATH,
        UPSTREAM_CKPT_PATH,
        build_per_y_yaml,
        _yaml_name_for_year,
    )

    rr = os.environ.get("RUN_ROOT")
    if rr:
        run_root = Path(rr)
        out: dict[int, Path] = {}
        for Y in _TEST_YEARS:
            yp = run_root / "inference" / _yaml_name_for_year(Y)
            if not yp.is_file():
                pytest.skip(f"live yaml missing under RUN_ROOT: {yp}")
            out[Y] = yp
        return out

    # Default: build into tmp_path.
    if not UPSTREAM_YAML_PATH.is_file():
        pytest.skip(f"upstream yaml not present: {UPSTREAM_YAML_PATH}")
    if not UPSTREAM_CKPT_PATH.is_file():
        pytest.skip(f"upstream ckpt not present: {UPSTREAM_CKPT_PATH}")

    root = tmp_path_factory.mktemp("required_attrs")
    config_dir = root / "config"
    exp_dir = root / "exp"
    out = {Y: build_per_y_yaml(Y, config_dir, exp_dir, K=_K) for Y in _TEST_YEARS}
    return out


class TestRequiredAttrsPresent:
    @pytest.mark.parametrize("Y", _TEST_YEARS)
    def test_every_required_attr_present(self, yamls_by_year, Y):
        p = _yparams_load(yamls_by_year[Y])
        missing = [k for k in _REQUIRED_FROM_YAML if not hasattr(p, k)]
        assert not missing, (
            f"Y={Y} yaml is missing required attrs: {missing}\n"
            f"yaml path: {yamls_by_year[Y]}"
        )


class TestPinnedValues:
    @pytest.mark.parametrize("Y", _TEST_YEARS)
    def test_epsilon_factor_zero(self, yamls_by_year, Y):
        p = _yparams_load(yamls_by_year[Y])
        assert p.epsilon_factor == 0.0, (
            f"Y={Y}: epsilon_factor must be 0.0 for deterministic NWP "
            f"(2026-05-08 user decision); got {p.epsilon_factor!r}"
        )

    @pytest.mark.parametrize("Y", _TEST_YEARS)
    def test_save_basenames_length_one(self, yamls_by_year, Y):
        p = _yparams_load(yamls_by_year[Y])
        assert isinstance(p.save_basenames, (list, tuple)), (
            f"Y={Y}: save_basenames must be a list, got {type(p.save_basenames)}"
        )
        assert len(p.save_basenames) == 1, (
            f"Y={Y}: save_basenames must be length-1 (single-IC invariant); "
            f"got len={len(p.save_basenames)}"
        )

    @pytest.mark.parametrize("Y", _TEST_YEARS)
    def test_ensemble_inference_hours(self, yamls_by_year, Y):
        p = _yparams_load(yamls_by_year[Y])
        # Under partial-horizon eval (K=60), the per-Y horizon is
        # Y-independent: (K + 1) * 6 = 366. Year-long sentinels
        # 8760/8784 must NOT appear (they would indicate the pre-fix
        # year-long override leaked through).
        assert p.ensemble_inference_hours == _EXPECTED_HOURS, (
            f"Y={Y}: ensemble_inference_hours expected {_EXPECTED_HOURS} "
            f"(K={_K}); got {p.ensemble_inference_hours}"
        )
        assert p.ensemble_inference_hours not in (8760, 8784), (
            f"Y={Y}: ensemble_inference_hours == {p.ensemble_inference_hours} "
            f"is a year-long sentinel — partial-horizon override leaked"
        )

    @pytest.mark.parametrize("Y", _TEST_YEARS)
    def test_prediction_duration_days(self, yamls_by_year, Y):
        p = _yparams_load(yamls_by_year[Y])
        # Required by the BCS single_ic loader at
        # data_loader_multifiles.py:818-823 — without it, the date
        # range collapses for sub-year rollouts.
        assert hasattr(p, "prediction_duration_days"), (
            f"Y={Y}: prediction_duration_days missing — BCS loader will collapse"
        )
        assert abs(float(p.prediction_duration_days) - _EXPECTED_DAYS) < 1e-9, (
            f"Y={Y}: prediction_duration_days expected {_EXPECTED_DAYS} "
            f"(K={_K}); got {p.prediction_duration_days}"
        )

    @pytest.mark.parametrize("Y", _TEST_YEARS)
    def test_val_year_window(self, yamls_by_year, Y):
        p = _yparams_load(yamls_by_year[Y])
        assert p.val_year_start == Y
        assert p.val_year_end == Y + 1

    @pytest.mark.parametrize("Y", _TEST_YEARS)
    def test_save_forecasts_and_log_flags(self, yamls_by_year, Y):
        p = _yparams_load(yamls_by_year[Y])
        assert p.save_forecasts is True, (
            f"Y={Y}: save_forecasts must be True so upstream writes NetCDF "
            f"outputs; got {p.save_forecasts!r}"
        )
        assert p.log_to_wandb is False, (
            f"Y={Y}: log_to_wandb must be False (offline eval); "
            f"got {p.log_to_wandb!r}"
        )

    @pytest.mark.parametrize("Y", _TEST_YEARS)
    def test_leap_year_pinned_to_Y(self, yamls_by_year, Y):
        # Upstream `data_loader_multifiles.py:931-934` builds the varying-
        # boundary h5 path from `leap_year`/`no_leap_year` (template year),
        # not the actual data year. Stampede3 data tree is year-keyed
        # (`121_*.h5` … `128_*.h5`), so both must equal Y or the loader
        # tries `<dir>/11_<idx>.h5` (upstream default) and FileNotFoundErrors.
        p = _yparams_load(yamls_by_year[Y])
        assert p.leap_year == Y, (
            f"Y={Y}: leap_year must be pinned to Y for the year-keyed "
            f"Stampede3 data tree; got {p.leap_year!r}"
        )
        assert p.no_leap_year == Y, (
            f"Y={Y}: no_leap_year must be pinned to Y for the year-keyed "
            f"Stampede3 data tree; got {p.no_leap_year!r}"
        )
