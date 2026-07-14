"""Tests for src/sfno_inference_5410/stampede3_yaml_override.py + scripts/build_5410_yaml_override.py.

Coverage (per docs/2026-05-06_group_sfno_5410_eval_plan.md §H, §B.1, §3 P-2):

For each ``Y ∈ {121..128}``:
  - generated yaml has **zero** ``/glade/`` substrings;
  - YParams (upstream) loads it without error;
  - ``val_year_start == Y``, ``val_year_end == Y + 1``;
  - ``save_forecasts == True``, ``log_to_wandb == False``;
  - all 8 yamls share identical model architecture (only the per-Y
    int fields differ);
  - the **single-file** symlink shim resolves:
      ``os.path.islink(...ckpt_epoch_50.tar)`` and
      ``os.path.realpath(...) == "/work2/.../v2.0/.../ckpt_epoch_50.tar"`` and
      ``natsorted(glob(...ckpt_epoch_*.tar))[-1]`` is exactly that file;
  - ``latitudes`` and ``longitudes`` lists are unchanged from the source yaml.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import pytest


_UPSTREAM_REPO = Path(
    "/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0"
)
_UPSTREAM_CKPT = (
    _UPSTREAM_REPO / "results" / "SFNO" / "5410" / "checkpoints" / "ckpt_epoch_50.tar"
)
_UPSTREAM_YAML = _UPSTREAM_REPO / "config" / "SFNO_PLASIM_H5_DERECHO_5410.yaml"

_TEST_YEARS = tuple(range(121, 129))

# Canonical eval-track forecast-leads horizon used by these tests.
_K = 60

# YParams + natsort live in the venv.
pytest.importorskip("ruamel.yaml")


def _add_upstream_to_path():
    """Make `utils.YParams` importable for the duration of the call."""
    p = str(_UPSTREAM_REPO)
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="module")
def built_artifacts(tmp_path_factory):
    """Run `build_all` once for the whole test module."""
    pytest.importorskip("ruamel.yaml")
    if not _UPSTREAM_YAML.is_file():
        pytest.skip(f"upstream yaml not present: {_UPSTREAM_YAML}")
    if not _UPSTREAM_CKPT.is_file():
        pytest.skip(f"upstream ckpt not present: {_UPSTREAM_CKPT}")

    from sfno_inference_5410.stampede3_yaml_override import build_all

    root = tmp_path_factory.mktemp("yaml_shim")
    config_dir = root / "config"
    exp_dir = root / "exp"
    out = build_all(config_dir, exp_dir, K=_K)
    return {"out": out, "config_dir": config_dir, "exp_dir": exp_dir}


class TestPerYearYaml:
    @pytest.mark.parametrize("Y", _TEST_YEARS)
    def test_no_glade_substrings(self, built_artifacts, Y):
        yaml_path = built_artifacts["out"][Y]["yaml"]
        text = yaml_path.read_text()
        assert "/glade/" not in text, (
            f"yaml for Y={Y} still contains /glade/ paths"
        )

    @pytest.mark.parametrize("Y", _TEST_YEARS)
    def test_yparams_loads(self, built_artifacts, Y):
        _add_upstream_to_path()
        from utils.YParams import YParams  # type: ignore

        yaml_path = built_artifacts["out"][Y]["yaml"]
        p = YParams(str(yaml_path), "SFNO")
        assert p.val_year_start == Y
        assert p.val_year_end == Y + 1
        assert p.save_forecasts is True
        assert p.log_to_wandb is False

    def test_identical_model_architecture(self, built_artifacts):
        """All 8 yamls must share the same model arch — only val_year_*
        differ. Compare the SFNO section sans val_year_*."""
        _add_upstream_to_path()
        from utils.YParams import YParams  # type: ignore

        # Architecture-relevant keys (sample; not exhaustive but sufficient
        # to detect drift). Keys that intentionally vary per Y are excluded.
        arch_keys = (
            "embed_dim", "num_layers", "num_blocks", "operator_type",
            "scale_factor", "spectral_layers", "use_mlp", "mlp_ratio",
            "horizontal_resolution", "num_levels", "sigma_levels", "levels",
            "upper_air_variables", "surface_variables",
            "diagnostic_variables", "varying_boundary_variables",
            "constant_boundary_variables",
        )
        ref = None
        for Y in _TEST_YEARS:
            p = YParams(str(built_artifacts["out"][Y]["yaml"]), "SFNO")
            row = {k: getattr(p, k, None) for k in arch_keys}
            if ref is None:
                ref = row
            else:
                assert row == ref, f"Y={Y} yaml diverges from Y={_TEST_YEARS[0]}"

    def test_lat_lon_unchanged(self, built_artifacts):
        """The 64-entry lat list and 128-entry lon list must round-trip
        unchanged from the source yaml."""
        _add_upstream_to_path()
        from utils.YParams import YParams  # type: ignore

        # Source values.
        ref = YParams(str(_UPSTREAM_YAML), "SFNO")
        for Y in _TEST_YEARS:
            p = YParams(str(built_artifacts["out"][Y]["yaml"]), "SFNO")
            assert list(p.lat) == list(ref.lat)
            assert list(p.lon) == list(ref.lon)


class TestCheckpointShim:
    @pytest.mark.parametrize("Y", _TEST_YEARS)
    def test_shim_is_file_symlink(self, built_artifacts, Y):
        shim = built_artifacts["out"][Y]["shim"]
        assert os.path.islink(shim), f"shim for Y={Y} is not a symlink"

    @pytest.mark.parametrize("Y", _TEST_YEARS)
    def test_shim_resolves_to_upstream(self, built_artifacts, Y):
        shim = built_artifacts["out"][Y]["shim"]
        assert os.path.realpath(shim) == str(_UPSTREAM_CKPT.resolve())

    @pytest.mark.parametrize("Y", _TEST_YEARS)
    def test_natsort_picks_shim(self, built_artifacts, Y):
        natsort = pytest.importorskip("natsort")
        shim = built_artifacts["out"][Y]["shim"]
        ckpt_glob = os.path.join(os.path.dirname(shim), "ckpt_epoch_*.tar")
        matches = natsort.natsorted(glob.glob(ckpt_glob))
        assert matches, f"no matches for {ckpt_glob}"
        assert matches[-1] == str(shim)


class TestIdempotence:
    def test_double_build_no_overwrite_error(self, built_artifacts):
        """Building twice with the same target must not raise."""
        from sfno_inference_5410.stampede3_yaml_override import (
            build_ckpt_symlink_shim,
        )
        Y = 121
        exp_dir = built_artifacts["exp_dir"]
        # Already built by the fixture; second call must succeed.
        shim = build_ckpt_symlink_shim(Y, exp_dir)
        assert os.path.islink(shim)


class TestShimYInvariant:
    """All 8 Y values must produce the same shim path (one shim per
    exp_dir; upstream's expDir = exp_dir/args.config/run_num is itself
    Y-invariant when --config=SFNO)."""

    def test_all_years_share_one_shim_path(self, built_artifacts):
        shims = {Y: built_artifacts["out"][Y]["shim"] for Y in _TEST_YEARS}
        unique = set(str(p) for p in shims.values())
        assert len(unique) == 1, (
            f"shim paths must be Y-invariant; got {len(unique)} distinct: {unique}"
        )

    def test_shim_under_sfno_section_dir(self, built_artifacts):
        from sfno_inference_5410.stampede3_yaml_override import CONFIG_SECTION
        shim = built_artifacts["out"][_TEST_YEARS[0]]["shim"]
        # Layout: <exp_dir>/SFNO/5410/checkpoints/ckpt_epoch_50.tar
        parts = Path(shim).parts
        assert "SFNO" == CONFIG_SECTION
        assert parts[-4] == CONFIG_SECTION
        assert parts[-3] == "5410"
        assert parts[-2] == "checkpoints"
