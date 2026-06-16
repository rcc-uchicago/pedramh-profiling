"""Tests for ``src/sfno_inference_5410/upstream_hydration.py``.

Codex round-2 fix #1: split allowlist by helper boundary so each helper
is tested at its own surface, not at a misleading union.

Three helpers, three allowlists:
  * ``hydrate_static_params`` → ``_STATIC_HYDRATION_ATTRS`` (17 fields)
  * ``set_per_y_params``      → ``_PER_Y_HYDRATION_ATTRS`` (4 fields)
  * ``set_per_ic_params``     → ``_PER_IC_HYDRATION_ATTRS`` (6 fields)

Plus a ``test_full_main_equivalence`` that asserts the union — i.e., the
params object after all three run matches what upstream main() produces
just before ``Stepper(params_list, ...)``.

The static-attrs test is the load-bearing one. The NWP eval pins
``nc_bc_offset = 0`` so the single-IC BCS loader matches the model's
training/validation autoregression boundary phase.

These tests require the upstream PanguWeather/v2.0 tree to be present.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import cftime
import pytest


_UPSTREAM_REPO = Path(
    "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0"
)


def _need_upstream():
    if not _UPSTREAM_REPO.is_dir():
        pytest.skip(f"upstream repo not present: {_UPSTREAM_REPO}")


@pytest.fixture(scope="module")
def y121_yaml(tmp_path_factory):
    """Build a Y=121 yaml override at K=60 + ckpt symlink shim in a tmp dir.

    hydrate_static_params follows upstream main()'s ckpt resolution
    (long_inference.py:1407-1424): looks under
    ``<exp_dir>/SFNO/5410/checkpoints/`` for best_ckpt.tar, ckpt_latest.tar,
    or ckpt_epoch_*.tar. Without the symlink shim the ckpt-resolution
    raises FileNotFoundError. This fixture mirrors what
    build_5410_yaml_override.py does at runtime.
    """
    pytest.importorskip("ruamel.yaml")
    _need_upstream()
    from sfno_inference_5410.stampede3_yaml_override import (
        UPSTREAM_CKPT_PATH,
        build_ckpt_symlink_shim,
        build_per_y_yaml,
    )
    if not UPSTREAM_CKPT_PATH.is_file():
        pytest.skip(f"upstream ckpt not present: {UPSTREAM_CKPT_PATH}")
    root = tmp_path_factory.mktemp("hydration")
    yaml_path = build_per_y_yaml(121, root / "config", root / "exp", K=60)
    build_ckpt_symlink_shim(121, root / "exp")
    return yaml_path


def test_static_attrs_present_after_hydrate(y121_yaml):
    """After hydrate_static_params, every _STATIC_HYDRATION_ATTRS field is set."""
    pytest.importorskip("ruamel.yaml")
    _need_upstream()
    from sfno_inference_5410.upstream_hydration import (
        _STATIC_HYDRATION_ATTRS,
        hydrate_static_params,
    )

    params = hydrate_static_params(
        y121_yaml, K=60, upstream_repo=_UPSTREAM_REPO,
    )
    missing = [k for k in _STATIC_HYDRATION_ATTRS if not hasattr(params, k)]
    assert not missing, (
        f"hydrate_static_params missing static attrs: {missing}. "
        f"Expected: {_STATIC_HYDRATION_ATTRS}"
    )


def test_static_pinned_values(y121_yaml):
    """Pinned values upstream main() injects."""
    pytest.importorskip("ruamel.yaml")
    _need_upstream()
    from sfno_inference_5410.upstream_hydration import hydrate_static_params

    params = hydrate_static_params(
        y121_yaml, K=60, upstream_repo=_UPSTREAM_REPO,
    )
    # The upstream standalone script hard-codes 18, but the validation
    # autoregression path uses current-step boundary forcing. The 5410
    # NWP eval must match validation, so the corrected value is 0.
    assert params.nc_bc_offset == 0, (
        f"nc_bc_offset must be 0 for 5410 NWP eval; got {params.nc_bc_offset}"
    )
    assert params.world_size == 1
    assert params.batch_size == 1
    assert params.local_rank == 0
    assert params.enable_amp is True
    assert params.log_to_wandb is False
    assert params.resuming is True
    assert params.run_iter == 1
    # has_diagnostic from the yaml's diagnostic_variables list.
    assert params.has_diagnostic is True  # SFNO_PLASIM has pr_6h
    assert params.num_ensemble_members == 1
    assert params.ensemble_members_per_pred == 1


def test_per_y_attrs_present_after_set_per_y(y121_yaml):
    """After set_per_y_params(Y=121), all four per-Y fields are set."""
    pytest.importorskip("ruamel.yaml")
    _need_upstream()
    from sfno_inference_5410.upstream_hydration import (
        _PER_Y_HYDRATION_ATTRS,
        hydrate_static_params,
        set_per_y_params,
    )

    params = hydrate_static_params(
        y121_yaml, K=60, upstream_repo=_UPSTREAM_REPO,
    )
    set_per_y_params(params, Y=121)
    for k in _PER_Y_HYDRATION_ATTRS:
        assert hasattr(params, k), f"missing per-Y attr {k}"
    assert params.val_year_start == 121
    assert params.val_year_end == 122
    assert params.leap_year == 121
    assert params.no_leap_year == 121


def test_set_per_y_rejects_bool():
    """isinstance(True, int) is True in Python — must be rejected explicitly."""
    pytest.importorskip("ruamel.yaml")
    _need_upstream()
    from sfno_inference_5410.upstream_hydration import set_per_y_params

    class FakeParams(dict):
        def __setattr__(self, k, v): self[k] = v

    fake = FakeParams()
    for bad in (True, False, "121", 1.5):
        with pytest.raises(ValueError):
            set_per_y_params(fake, Y=bad)
    with pytest.raises(ValueError):
        set_per_y_params(fake, Y=0)
    with pytest.raises(ValueError):
        set_per_y_params(fake, Y=-1)


def test_per_ic_attrs_present_after_set_per_ic(tmp_path, y121_yaml):
    """After set_per_ic_params, all six per-IC fields are set including
    init_nc_timestep_offset (recomputed from the IC NC's time index)."""
    pytest.importorskip("ruamel.yaml")
    _need_upstream()
    from sfno_inference_5410.ic_source import resolve_ic_nc_path
    from sfno_inference_5410.upstream_hydration import (
        _PER_IC_HYDRATION_ATTRS,
        hydrate_static_params,
        set_per_ic_params,
    )

    # Need a real IC NC for set_per_ic_params (it opens the file).
    run_root = Path(
        "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval_5410/20260507_phase1_gate"
    )
    ic_nc = run_root / "inference" / "ic_nc" / "121_0000.nc"
    if not ic_nc.is_file():
        pytest.skip(f"IC NC not present: {ic_nc}")

    params = hydrate_static_params(
        y121_yaml, K=60, upstream_repo=_UPSTREAM_REPO,
    )
    init_dt = cftime.DatetimeProlepticGregorian(121, 1, 1, 0, has_year_zero=True)
    final_dt = init_dt + dt.timedelta(hours=61 * 6)
    set_per_ic_params(
        params,
        init_datetime=init_dt,
        final_datetime=final_dt,
        init_nc_filepaths=[ic_nc],
        save_basename="Y121_s0000",
        output_dir=tmp_path,
    )
    for k in _PER_IC_HYDRATION_ATTRS:
        assert hasattr(params, k), f"missing per-IC attr {k}"
    assert params.init_datetime == init_dt
    assert params.final_datetime == final_dt
    assert list(params.init_nc_filepaths) == [str(ic_nc)]
    # init_nc_timestep_offset must be a list of ints (one per filepath).
    # For Y=121 s=0 with init_datetime = 0121-01-01 00:00, the offset is 0.
    assert params.init_nc_timestep_offset == [0]
    assert params.save_basename == "Y121_s0000"
    assert params.output_dir == str(tmp_path)


def test_set_per_ic_rejects_multi_ic(tmp_path, y121_yaml):
    """Single-IC invariant: init_nc_filepaths must be length 1."""
    pytest.importorskip("ruamel.yaml")
    _need_upstream()
    from sfno_inference_5410.upstream_hydration import (
        hydrate_static_params,
        set_per_ic_params,
    )

    params = hydrate_static_params(
        y121_yaml, K=60, upstream_repo=_UPSTREAM_REPO,
    )
    init_dt = cftime.DatetimeProlepticGregorian(121, 1, 1, 0, has_year_zero=True)
    final_dt = init_dt + dt.timedelta(hours=61 * 6)
    with pytest.raises(ValueError):
        set_per_ic_params(
            params,
            init_datetime=init_dt, final_datetime=final_dt,
            init_nc_filepaths=["a.nc", "b.nc"],
            save_basename="Y121_s0000",
            output_dir=tmp_path,
        )


def test_full_main_equivalence(tmp_path, y121_yaml):
    """After hydrate + set_per_y + set_per_ic, every union-attr is present.

    This is the union of all three allowlists. After running all three
    helpers, the params object should have every field upstream main()
    sets between argparse and Stepper(...).
    """
    pytest.importorskip("ruamel.yaml")
    _need_upstream()
    from sfno_inference_5410.upstream_hydration import (
        _PER_IC_HYDRATION_ATTRS,
        _PER_Y_HYDRATION_ATTRS,
        _STATIC_HYDRATION_ATTRS,
        hydrate_static_params,
        set_per_ic_params,
        set_per_y_params,
    )

    run_root = Path(
        "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval_5410/20260507_phase1_gate"
    )
    ic_nc = run_root / "inference" / "ic_nc" / "121_0000.nc"
    if not ic_nc.is_file():
        pytest.skip(f"IC NC not present: {ic_nc}")

    params = hydrate_static_params(
        y121_yaml, K=60, upstream_repo=_UPSTREAM_REPO,
    )
    set_per_y_params(params, Y=121)
    init_dt = cftime.DatetimeProlepticGregorian(121, 1, 1, 0, has_year_zero=True)
    final_dt = init_dt + dt.timedelta(hours=61 * 6)
    set_per_ic_params(
        params,
        init_datetime=init_dt, final_datetime=final_dt,
        init_nc_filepaths=[ic_nc],
        save_basename="Y121_s0000",
        output_dir=tmp_path,
    )

    union = _STATIC_HYDRATION_ATTRS + _PER_Y_HYDRATION_ATTRS + _PER_IC_HYDRATION_ATTRS
    missing = [k for k in union if not hasattr(params, k)]
    assert not missing, f"union allowlist missing: {missing}"
