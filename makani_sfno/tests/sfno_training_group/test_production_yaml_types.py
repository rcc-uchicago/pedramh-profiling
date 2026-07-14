"""T.2 — production YAML type/value contracts (Phase F.E).

Verifies the rendered production YAML satisfies long_inference and train.py
contracts:
  - ensemble_inference_hours is SCALAR int (NOT a list — line 483 uses //).
  - save_forecasts is bool True.
  - save_basenames is a non-empty list (defensive — plan v5 §B4).
  - long_rollout_years == 1 (per-IC year-bounded save).
  - prediction_duration_days is ABSENT (would break leap years).
  - varying_boundary_variables == ['z0', 'sst', 'rsdt', 'sic'] (z0 stays varying).
  - constant_boundary_variables == ['lsm', 'sg'].
  - SFNO override sets nettype 'sfno_plasim', batch_size 8, max_epochs 50.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

PROD_YAML = REPO_ROOT / "src" / "sfno_training_group" / "config" / "plasim_sim52_sigma10_sfno_full.yaml"


def _make_fake_manifest(tmp_path: Path, years: list[int]) -> Path:
    """Build a minimal manifest with `years` entries good enough for render_yaml."""
    entries = []
    for y in years:
        entries.append({
            "year": y,
            "n_timesteps": 1455,
            "n_timesteps_native": 1455,
            "synthetic_start_dt": f"{y:04d}-01-01 00:00:00",
            "synthetic_last_idx_dt": f"{y:04d}-12-30 12:00:00",
            "last_train_init_idx": 1453,
            "last_train_init_dt": f"{y:04d}-12-30 06:00:00",
            "train_end_exclusive_dt": f"{y:04d}-12-30 12:00:00",
            "last_val_init_idx_for_max_lead_K": 1394,
            "last_val_init_dt_for_max_lead_K": f"{y:04d}-12-15 12:00:00",
            "val_end_exclusive_dt_for_max_lead_K": f"{y:04d}-12-15 18:00:00",
            "z0_temporal_std_mean": 0.0001,
            "src_path": "/fake.h5",
        })
    manifest = {
        "calendar": "proleptic_gregorian",
        "has_year_zero": True,
        "data_timedelta_hours": 6,
        "max_forecast_lead_steps": 60,
        "src_root": "/fake",
        "dst": str(tmp_path),
        "expected_state_channels": [],
        "expected_diagnostic_channels": [],
        "expected_forcing_channels": [],
        "sigma_levels_pl_native": [],
        "zg_levels_pa": [],
        "years": entries,
    }
    p = tmp_path / "_v10_calendar_manifest.json"
    p.write_text(json.dumps(manifest))
    return p


@pytest.fixture(scope="module")
def rendered_production_yaml(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Render production YAML against a 100-year manifest and parse it."""
    tmp = tmp_path_factory.mktemp("yaml_types")
    train_years = list(range(12, 112))   # 100 years
    val_years = [11]
    manifest = _make_fake_manifest(tmp, train_years + val_years)
    out = tmp / "rendered.yaml"

    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'src'}:{existing}" if existing else str(REPO_ROOT / "src")
    )
    res = subprocess.run(
        [sys.executable, "-m", "sfno_training_group.tools.render_yaml",
         "--tpl", str(PROD_YAML),
         "--out", str(out),
         "--data-dir", "/fake/data",
         "--exp-dir", "/fake/exp",
         "--manifest", str(manifest),
         "--train-years", *(str(y) for y in train_years),
         "--val-years", *(str(y) for y in val_years)],
        capture_output=True, text=True, env=env,
    )
    if res.returncode != 0:
        pytest.fail(f"render_yaml failed: stdout={res.stdout}\nstderr={res.stderr}")
    text = out.read_text()
    parsed = yaml.safe_load(text)
    return parsed


def test_train_data_sets_has_100_entries(rendered_production_yaml: dict) -> None:
    sfno = rendered_production_yaml["SFNO"]
    train_block = sfno["train_data_sets"]["/fake/data"]
    assert len(train_block) == 100, f"expected 100 train entries, got {len(train_block)}"
    # First and last entries cover years 12 and 111.
    assert train_block[0][0] == "0012-01-01 00:00:00"
    assert train_block[-1][0] == "0111-01-01 00:00:00"


def test_ensemble_inference_hours_is_scalar(rendered_production_yaml: dict) -> None:
    sfno = rendered_production_yaml["SFNO"]
    val = sfno["ensemble_inference_hours"]
    assert isinstance(val, (int, float)), (
        f"ensemble_inference_hours must be scalar; got {type(val).__name__} = {val!r}. "
        f"Loader at data_loader_multifiles.py:483 does `// timedelta_hours`."
    )
    assert val == 8760, f"expected 8760 (one year), got {val}"


def test_no_prediction_duration_days(rendered_production_yaml: dict) -> None:
    sfno = rendered_production_yaml["SFNO"]
    assert "prediction_duration_days" not in sfno, (
        "prediction_duration_days breaks leap years (loader sets end=start+365 days; "
        "leap year buffer is 1464). Use long_rollout_years: 1 instead."
    )


def test_long_rollout_years_eq_1(rendered_production_yaml: dict) -> None:
    assert rendered_production_yaml["SFNO"]["long_rollout_years"] == 1


def test_save_forecasts_true(rendered_production_yaml: dict) -> None:
    assert rendered_production_yaml["SFNO"]["save_forecasts"] is True


def test_save_basenames_present_and_nonempty(rendered_production_yaml: dict) -> None:
    sb = rendered_production_yaml["SFNO"]["save_basenames"]
    assert isinstance(sb, list)
    assert len(sb) >= 1


def test_boundary_variable_layout(rendered_production_yaml: dict) -> None:
    sfno = rendered_production_yaml["SFNO"]
    assert sfno["varying_boundary_variables"] == ["z0", "sst", "rsdt", "sic"]
    assert sfno["constant_boundary_variables"] == ["lsm", "sg"]


def test_sfno_overrides(rendered_production_yaml: dict) -> None:
    sfno = rendered_production_yaml["SFNO"]
    assert sfno["nettype"] == "sfno_plasim"
    assert sfno["max_epochs"] == 50
    assert sfno["batch_size"] == 8
    # 5410 SFNO sigma reference uses lr=2e-6. PyYAML may parse `2e-06` as a
    # string (YAML 1.1 scientific notation requires a decimal point); group's
    # YParams coerces to float, so we mirror that here.
    assert abs(float(sfno["lr"]) - 2e-6) < 1e-12
    assert sfno["num_inferences"] == 128
    assert sfno["use_sigma_levels"] is True
    assert sfno["fresh_start"] is False


def test_validation_uses_year_11(rendered_production_yaml: dict) -> None:
    sfno = rendered_production_yaml["SFNO"]
    val_block = sfno["validation_data_sets"]["/fake/data"]
    assert len(val_block) == 1
    assert val_block[0][0] == "0011-01-01 00:00:00"
    # End-exclusive from manifest's val_end_exclusive_dt_for_max_lead_K.
    assert val_block[0][1] == "0011-12-15 18:00:00"
