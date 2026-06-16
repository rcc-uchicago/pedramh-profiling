"""Tier 2 of the 5410 yaml regression net (per docs plan v3.1).

Real upstream-loader exercise on the smoke path's ``save_basenames``
read site at ``data_loader_multifiles.py:829``, with all NetCDF/h5
filesystem I/O monkeypatched out so the test runs in <1 s on a
Stampede3 login node.

Why this test exists
--------------------
Tier 1 (``test_required_attrs.py``) is an allowlist — it only catches
keys we already know to require. This test calls upstream
``get_data_loader(... ensemble=True, init_from_nc=True)`` (the same
call ``long_inference.py:192`` makes) so a future upstream resync that
adds an unguarded ``params.<x>`` access on the construction path
fails here, before a SLURM submit, even if we forget to update the
allowlist.

Two modes
---------
Same convention as Tier 1: ``RUN_ROOT=<run_root>`` switches to live
yamls; otherwise build per-Y yamls into ``tmp_path`` via the override
generator.

Monkeypatches (applied unconditionally — hermetic test)
-------------------------------------------------------
- ``GetDataset._load_constant_boundary_data`` → returns a 2-tuple of
  zero tensors. Original opens NetCDF at ``data_loader_multifiles.py:741``.
- ``GetDataset._load_varying_boundary_data`` → returns zero tensor.
  Original loops 1460 6-hour boundary timesteps reading h5 from
  ``$SCRATCH``.
- ``GetDataset.load_mean_std`` → returns 2-tuple of zero tensors.
  Original opens NetCDF at ``data_loader_multifiles.py:767``. Codex
  v3 review specifically requested unconditional monkeypatching here.

Monkeypatching does NOT weaken the missing-key check: the call sites
of all three methods read ``params.<x>`` attrs from YParams *before*
calling the (now-stubbed) method, so any unguarded missing key still
surfaces as an AttributeError.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import cftime
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_REPO = Path(
    "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0"
)

_TEST_YEARS = tuple(range(121, 129))
_LEAP_YEARS = (124, 128)

# Canonical eval-track forecast-leads horizon used by these tests.
_K = 60


def _add_upstream_to_path():
    if not _UPSTREAM_REPO.is_dir():
        pytest.skip(f"upstream repo not present: {_UPSTREAM_REPO}")
    p = str(_UPSTREAM_REPO)
    if p not in sys.path:
        sys.path.insert(0, p)


def _yparams_load(yaml_path: Path):
    _add_upstream_to_path()
    from utils.YParams import YParams  # type: ignore

    return YParams(str(yaml_path), "SFNO")


@pytest.fixture(scope="module")
def yamls_by_year(tmp_path_factory):
    """Two-mode yaml resolver (same as Tier 1)."""
    pytest.importorskip("ruamel.yaml")
    pytest.importorskip("torch")
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

    if not _UPSTREAM_REPO.is_dir():
        pytest.skip(f"upstream repo not present: {_UPSTREAM_REPO}")
    if not UPSTREAM_YAML_PATH.is_file():
        pytest.skip(f"upstream yaml not present: {UPSTREAM_YAML_PATH}")
    if not UPSTREAM_CKPT_PATH.is_file():
        pytest.skip(f"upstream ckpt not present: {UPSTREAM_CKPT_PATH}")

    root = tmp_path_factory.mktemp("contract")
    config_dir = root / "config"
    exp_dir = root / "exp"
    out = {Y: build_per_y_yaml(Y, config_dir, exp_dir, K=_K) for Y in _TEST_YEARS}
    return out


def _hydrate_main_params(p, *, Y: int) -> None:
    """Replicate ``long_inference.py:main()`` mutations up to Stepper.

    Sets the dynamic params that ``main()`` injects before the first
    ``get_data_loader`` call so the data loader sees a hydrated
    YParams object identical to the one upstream sees on the smoke
    path.
    """
    p["run_iter"] = 1
    p["has_diagnostic"] = bool(getattr(p, "diagnostic_variables", []) or False)
    if not hasattr(p, "num_ensemble_members"):
        p["num_ensemble_members"] = 1
    p["init_nc_filepaths"] = ["/dev/null"]  # value unused with monkeypatched loaders
    # 5410 NWP eval matches the validation autoregression boundary phase:
    # step 1 consumes boundary fields from the IC time, not init+18h.
    p["nc_bc_offset"] = 0
    p["ensemble_members_per_pred"] = p.num_ensemble_members
    p["world_size"] = 1
    p["batch_size"] = 1
    p["local_rank"] = 0
    p["enable_amp"] = True

    # Partial-horizon: final_datetime = init + (K + 1) * 6h, not Jan 1 (Y+1).
    init_dt = cftime.DatetimeProlepticGregorian(Y, 1, 1, 0, has_year_zero=True)
    final_dt = init_dt + dt.timedelta(hours=(_K + 1) * 6)
    p["init_datetime"] = init_dt
    p["final_datetime"] = final_dt
    p["init_nc_timestep_offset"] = [0]


@pytest.fixture
def stub_io(monkeypatch):
    """Unconditionally monkeypatch the three NetCDF/h5-reading methods.

    Returns a no-op so the test body can use it as a fixture marker.
    """
    pytest.importorskip("torch")
    import torch
    _add_upstream_to_path()
    from utils.data_loader_multifiles import GetDataset  # type: ignore

    def _fake_load_constant(self):
        n_const = max(len(getattr(self.params, "constant_boundary_variables", []) or []), 1)
        H, W = self.params.horizontal_resolution
        return (torch.zeros(n_const, H, W, dtype=torch.float32),
                torch.zeros(1, H, W, dtype=torch.float32))

    def _fake_load_varying(self, batch_idx=None):
        n_var = max(len(getattr(self.params, "varying_boundary_variables", []) or []), 1)
        steps = self.params.ensemble_inference_hours // self.params.timedelta_hours
        H, W = self.params.horizontal_resolution
        return torch.zeros(steps, n_var, H, W, dtype=torch.float32)

    def _fake_load_mean_std(self, mean_file, std_file, datavars,
                            upper_air=True, use_sigma_levels=False, level_delta=1e-4):
        n_vars = max(len(datavars), 1)
        H, W = self.params.horizontal_resolution
        if upper_air:
            n_lev = self.params.num_levels
            return (torch.zeros(n_vars, n_lev, dtype=torch.float32),
                    torch.ones(n_vars, n_lev, dtype=torch.float32))
        return (torch.zeros(n_vars, dtype=torch.float32),
                torch.ones(n_vars, dtype=torch.float32))

    monkeypatch.setattr(GetDataset, "_load_constant_boundary_data", _fake_load_constant)
    monkeypatch.setattr(GetDataset, "_load_varying_boundary_data", _fake_load_varying)
    monkeypatch.setattr(GetDataset, "load_mean_std", _fake_load_mean_std)
    return None


class TestGetDataLoaderEnsembleBranch:
    """Exercise the same call long_inference.py:192 makes.

    This is the call site whose constructor reaches
    ``data_loader_multifiles.py:829`` and reads
    ``self.params.save_basenames``.
    """

    @pytest.mark.parametrize("Y", _TEST_YEARS)
    def test_ensemble_init_from_nc_construction(self, yamls_by_year, stub_io, Y):
        _add_upstream_to_path()
        from utils.data_loader_multifiles import get_data_loader  # type: ignore

        p = _yparams_load(yamls_by_year[Y])
        _hydrate_main_params(p, Y=Y)

        # The smoke call: ensemble=True, init_from_nc=True, train=False.
        data_loader, dataset = get_data_loader(
            p, p.data_dir, distributed=False,
            year_start=p.val_year_start, year_end=p.val_year_end,
            train=False, ensemble=True, init_from_nc=True,
        )

        # Construction succeeded => save_basenames was readable and
        # ensemble_inference_hours / horizontal_resolution / etc are
        # all present. Sanity-check the date_range size matches the
        # single-IC invariant.
        assert len(dataset.date_range_sizes) == 1
        assert dataset.date_range_sizes[0] == 1, (
            f"Y={Y}: expected length-1 date_range (single IC), "
            f"got {dataset.date_range_sizes!r}"
        )
        assert len(dataset.init_datetimes) == 1


class TestGetDataLoaderSingleIcBranch:
    """Second get_data_loader call from long_inference.py:199.

    ``single_ic=True`` hits a different branch in ``_get_dates`` than
    the ensemble branch — covers a separate set of unguarded reads.
    """

    @pytest.mark.parametrize("Y", _TEST_YEARS)
    def test_single_ic_construction(self, yamls_by_year, stub_io, Y):
        _add_upstream_to_path()
        from utils.data_loader_multifiles import get_data_loader  # type: ignore

        p = _yparams_load(yamls_by_year[Y])
        _hydrate_main_params(p, Y=Y)
        # Stepper.__init__:197 sets single_ic_offset before the second
        # get_data_loader call. Replicate.
        p["single_ic_offset"] = 0

        data_loader, dataset = get_data_loader(
            p, p.data_dir, distributed=False,
            year_start=p.init_datetime.year, year_end=p.final_datetime.year,
            train=False, single_ic=True,
        )
        assert dataset.single_ic is True


class TestPerturberGate:
    """epsilon_factor=0 must keep the perturber gate False.

    Verifies the *intent* of forcing ``epsilon_factor=0`` —
    long_inference.py:204 will not construct ``Perturber`` for these
    yamls, so the missing ``perturbation_type`` key in upstream is
    moot.
    """

    @pytest.mark.parametrize("Y", _TEST_YEARS)
    def test_gate_is_false(self, yamls_by_year, Y):
        p = _yparams_load(yamls_by_year[Y])
        # Mirror the gate at long_inference.py:204
        assert (p.epsilon_factor > 0.) is False, (
            f"Y={Y}: epsilon_factor={p.epsilon_factor!r} would make "
            f"long_inference.py:204 construct Perturber, which would "
            f"then fail on missing perturbation_type."
        )
