"""Tests for scripts/eval_inference_5410.py argv constructor.

Coverage (per docs/2026-05-06_group_sfno_5410_eval_plan.md §H, §B.2):

For every ``(Y, s)`` in the run plan:
  - ``--init_nc_filepaths == resolve_ic_nc_path(Y, s, run_root)`` and the
    file exists;
  - ``--init_datetime`` parses to
    ``cftime.DatetimeProlepticGregorian(Y, ...)`` such that
    ``start_date.year == Y == params.val_year_start``;
  - ``--final_datetime`` parses to
    ``cftime.DatetimeProlepticGregorian(Y+1, 1, 1, 0)``;
  - ``--output_dir`` exists / is writable;
  - ``--save_basename == f"Y{Y}_s{s:04d}"``;
  - ``--config`` matches the per-Y yaml filename (without ``.yaml``);
  - ``--run_num == "5410"``;
  - ``<run_root>/inference/ic_source.json`` exists with a recognized
    ``ic_source`` value.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cftime
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ORCH_PATH = _REPO_ROOT / "scripts" / "eval_inference_5410.py"


def _load_orchestrator():
    """Load `scripts/eval_inference_5410.py` as a module (it lives in scripts/, not src/)."""
    spec = importlib.util.spec_from_file_location("eval_inference_5410", _ORCH_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fake_run_root(tmp_path_factory):
    """Stand up a fake run_root with ic_source.json + IC NC stubs.

    Uses ic_source = "ic_nc_built_from_h5" so we can place the IC NC
    files inside ``tmp_path`` (no Stampede3-mount dependence).
    """
    root = tmp_path_factory.mktemp("run_root")
    inference = root / "inference"
    inference.mkdir()
    (inference / "upstream_raw").mkdir()
    ic_nc_dir = inference / "ic_nc"
    ic_nc_dir.mkdir()

    # Touch one IC NC per (Y, s) so the existence check passes.
    for Y in range(121, 129):
        for s in [0, 122, 244, 366, 488, 610, 732, 854, 976, 1098, 1220, 1342]:
            (ic_nc_dir / f"{Y}_{s:04d}.nc").write_bytes(b"")

    (inference / "ic_source.json").write_text(
        json.dumps({
            "ic_source": "ic_nc_built_from_h5",
            "resolved_at": "2026-05-07T00:00:00Z",
            "gate_pass_sha256": "0" * 64,
        })
    )
    return root


@pytest.fixture(scope="module")
def config_dir(tmp_path_factory):
    """Stand up a config dir with the 8 per-Y yaml stubs."""
    cfg = tmp_path_factory.mktemp("config")
    for Y in range(121, 129):
        (cfg / f"SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y{Y}.yaml").write_text("# stub\n")
    return cfg


_K = 60  # canonical eval-track forecast-leads horizon


@pytest.fixture(scope="module")
def run_plan(fake_run_root, config_dir):
    orch = _load_orchestrator()
    return orch.build_run_plan(fake_run_root, config_dir, K=_K)


_OFFSETS = (0, 122, 244, 366, 488, 610, 732, 854, 976, 1098, 1220, 1342)


class TestRunPlanShape:
    def test_96_entries(self, run_plan):
        assert len(run_plan) == 96

    def test_8_years_x_12_offsets(self, run_plan):
        years = sorted(set(int(e["save_basename"].split("_")[0][1:]) for e in run_plan))
        assert years == list(range(121, 129))
        for entry in run_plan:
            s = int(entry["save_basename"].split("_s")[1])
            assert s in _OFFSETS


class TestArgvComponents:
    @pytest.mark.parametrize("Y", list(range(121, 129)))
    def test_init_datetime_year_matches_Y(self, run_plan, Y):
        entries = [e for e in run_plan if e["init_datetime"].year == Y]
        assert len(entries) == 12, f"Y={Y} should have 12 entries"
        for e in entries:
            assert isinstance(e["init_datetime"], cftime.DatetimeProlepticGregorian)
            assert e["init_datetime"].year == Y

    @pytest.mark.parametrize("Y", list(range(121, 129)))
    def test_final_datetime_is_init_plus_K_plus_1_steps(self, run_plan, Y):
        """Partial-horizon contract: final = init + (K+1)*6h.

        Replaces the pre-2026-05-08 "final = Jan 1 (Y+1)" contract.
        Under K=60 the last IC of each year (s=1342) ends mid-January
        of the SAME year (s+(K+1) = 1403 < 1460), so final.year is Y,
        not Y+1.
        """
        import datetime as _dt
        entries = [e for e in run_plan if e["init_datetime"].year == Y]
        for e in entries:
            expected = e["init_datetime"] + _dt.timedelta(hours=(_K + 1) * 6)
            assert e["final_datetime"] == expected, (
                f"Y={Y} s={(e['init_datetime'] - cftime.DatetimeProlepticGregorian(Y,1,1,0,has_year_zero=True)).total_seconds()/21600}: "
                f"got {e['final_datetime']}, expected {expected}"
            )

    def test_save_basename_format(self, run_plan):
        for e in run_plan:
            sb = e["save_basename"]
            Y = e["init_datetime"].year
            # save_basename = f"Y{Y}_s{s:04d}"; recompute s from init_datetime.
            secs = (e["init_datetime"] - cftime.DatetimeProlepticGregorian(
                Y, 1, 1, 0, has_year_zero=True
            )).total_seconds()
            s = int(round(secs / (6 * 3600)))
            assert sb == f"Y{Y}_s{s:04d}"

    def test_config_is_sfno_section(self, run_plan):
        """``--config`` selects a YAML top-level key, not the file basename.

        The upstream yaml has anchors ``base_config / PLASIM / SFNO /
        modified_1``; SFNO is the architecture section. Per-Y
        differentiation lives in ``--yaml_config``'s SFNO section
        (val_year_start), not in ``--config``.
        """
        for e in run_plan:
            assert e["config"] == "SFNO"
            yaml_name = e["yaml"].name
            assert yaml_name.startswith("SFNO_PLASIM_H5_DERECHO_5410_stampede3_Y")
            assert yaml_name.endswith(".yaml")

    def test_no_argv_in_v2_1_orchestrator(self, run_plan):
        """As of v2.1 the orchestrator runs in-process (no per-IC argv).

        The legacy subprocess argv is captured separately in
        tests/sfno_inference_5410/integration/_legacy_subprocess_launcher.py
        as a reference path for A/B equivalence tests; it is no longer
        part of the per-IC entry dict returned by build_argv_for_ic.
        """
        for e in run_plan:
            assert "argv" not in e, (
                f"build_argv_for_ic must not return 'argv' as of v2.1; "
                f"got {sorted(e.keys())}"
            )

    def test_entry_has_explicit_Y_s(self, run_plan):
        """v2.1 entry dict carries Y and s as explicit ints."""
        for e in run_plan:
            assert isinstance(e["Y"], int)
            assert isinstance(e["s"], int)
            assert e["Y"] in range(121, 129)
            assert e["s"] in _OFFSETS

    def test_ic_nc_path_resolved_via_dispatcher(self, run_plan, fake_run_root):
        """Path matches `resolve_ic_nc_path` (i.e. comes from ic_source.json)."""
        from sfno_inference_5410.ic_source import resolve_ic_nc_path
        for e in run_plan:
            Y = e["init_datetime"].year
            secs = (e["init_datetime"] - cftime.DatetimeProlepticGregorian(
                Y, 1, 1, 0, has_year_zero=True
            )).total_seconds()
            s = int(round(secs / (6 * 3600)))
            expected = resolve_ic_nc_path(Y, s, fake_run_root)
            assert e["ic_nc"] == expected
            assert e["ic_nc"].exists(), f"missing IC NC: {e['ic_nc']}"

    def test_output_dir_exists(self, run_plan):
        for e in run_plan:
            assert e["output_dir"].is_dir()


class TestDatetimesRoundtrip:
    """Spot-check that init/final datetimes round-trip through %Y-%m-%d_%H:%M:%S.

    The legacy subprocess launcher (and any A/B tooling) uses this format
    string; the in-process orchestrator passes cftime objects directly
    but the format must remain compatible so the legacy reference path
    in tests/sfno_inference_5410/integration/_legacy_subprocess_launcher.py
    can faithfully reproduce the old behavior.
    """

    def test_init_datetime_roundtrip(self, run_plan):
        for e in run_plan[:5]:
            s = e["init_datetime"].strftime("%Y-%m-%d_%H:%M:%S")
            t = cftime.DatetimeProlepticGregorian.strptime(
                s, "%Y-%m-%d_%H:%M:%S", calendar="proleptic_gregorian",
            )
            assert t == e["init_datetime"]

    def test_final_datetime_roundtrip(self, run_plan):
        for e in run_plan[:5]:
            s = e["final_datetime"].strftime("%Y-%m-%d_%H:%M:%S")
            t = cftime.DatetimeProlepticGregorian.strptime(
                s, "%Y-%m-%d_%H:%M:%S", calendar="proleptic_gregorian",
            )
            assert t == e["final_datetime"]


class TestIcSourceJson:
    """Verify run_root has a valid ic_source.json (precondition for run plan)."""

    def test_ic_source_json_exists(self, fake_run_root):
        cfg_path = fake_run_root / "inference" / "ic_source.json"
        assert cfg_path.is_file()
        cfg = json.loads(cfg_path.read_text())
        assert cfg["ic_source"] in (
            "plev_data", "sigma_data_transferred", "ic_nc_built_from_h5",
        )

    def test_dispatcher_raises_when_missing(self, tmp_path):
        from sfno_inference_5410.ic_source import resolve_ic_nc_path
        with pytest.raises(FileNotFoundError, match="ic_source.json"):
            resolve_ic_nc_path(121, 0, tmp_path)
