"""Tests for render_yaml.py — verify end-exclusive substitution."""

from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sfno_training_group.tools.render_yaml import render  # noqa: E402


def test_render_substitutes_and_detects_leftovers() -> None:
    tpl = "data: {{DATA_DIR}}\nexp: {{EXP_DIR}}\nend12: {{TRAIN_END_EXCL_12}}\n"
    out = render(tpl, {
        "{{DATA_DIR}}": "/scratch/x",
        "{{EXP_DIR}}": "/scratch/exp",
        "{{TRAIN_END_EXCL_12}}": "0012-12-30 18:00:00",
    })
    assert "{{" not in out
    assert "/scratch/x" in out
    assert "0012-12-30 18:00:00" in out


def test_render_raises_on_leftover() -> None:
    import pytest
    tpl = "data: {{DATA_DIR}}\nmissing: {{NOT_PROVIDED}}\n"
    with pytest.raises(RuntimeError, match="Unrendered"):
        render(tpl, {"{{DATA_DIR}}": "/x"})


def test_render_yaml_cli_emits_end_exclusive(tmp_path: Path) -> None:
    """Round-trip via the CLI: build a tiny manifest, render the smoke tpl, assert outputs."""
    # Tiny manifest with year 12 train + year 11 val.
    manifest = {
        "calendar": "proleptic_gregorian",
        "has_year_zero": True,
        "data_timedelta_hours": 6,
        "max_forecast_lead_steps": 60,
        "years": [
            {
                "year": 11,
                "n_timesteps": 1455,
                "synthetic_start_dt": "0011-01-01 00:00:00",
                "synthetic_last_idx_dt": "0011-12-30 12:00:00",
                "last_train_init_idx": 1453,
                "last_train_init_dt": "0011-12-30 06:00:00",
                "train_end_exclusive_dt": "0011-12-30 12:00:00",
                "last_val_init_idx_for_max_lead_K": 1394,
                "last_val_init_dt_for_max_lead_K": "0011-12-15 12:00:00",
                "val_end_exclusive_dt_for_max_lead_K": "0011-12-15 18:00:00",
                "z0_temporal_std_mean": 1e-4,
            },
            {
                "year": 12,
                "n_timesteps": 1455,
                "synthetic_start_dt": "0012-01-01 00:00:00",
                "synthetic_last_idx_dt": "0012-12-30 12:00:00",
                "last_train_init_idx": 1453,
                "last_train_init_dt": "0012-12-30 06:00:00",
                "train_end_exclusive_dt": "0012-12-30 12:00:00",
                "last_val_init_idx_for_max_lead_K": 1394,
                "last_val_init_dt_for_max_lead_K": "0012-12-15 12:00:00",
                "val_end_exclusive_dt_for_max_lead_K": "0012-12-15 18:00:00",
                "z0_temporal_std_mean": 1e-4,
            },
        ],
    }
    manifest_path = tmp_path / "_v10_calendar_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    tpl_path = tmp_path / "tpl.yaml"
    tpl_path.write_text(
        "data_dir: \"{{DATA_DIR}}\"\n"
        "exp_dir: \"{{EXP_DIR}}\"\n"
        "train_data_sets:\n"
        "  \"{{DATA_DIR}}\":\n"
        "    - ['0012-01-01 00:00:00', '{{TRAIN_END_EXCL_12}}']\n"
        "validation_data_sets:\n"
        "  \"{{DATA_DIR}}\":\n"
        "    - ['0011-01-01 00:00:00', '{{VAL_END_EXCL_11}}']\n"
    )
    out_path = tmp_path / "rendered.yaml"

    result = subprocess.run(
        [
            sys.executable, "-m", "sfno_training_group.tools.render_yaml",
            "--tpl", str(tpl_path), "--out", str(out_path),
            "--data-dir", "/data", "--exp-dir", "/exp",
            "--manifest", str(manifest_path),
            "--train-years", "12",
            "--val-years", "11",
        ],
        env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"render_yaml CLI failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    out_text = out_path.read_text()
    assert "{{" not in out_text
    assert "0012-12-30 12:00:00" in out_text          # train end exclusive
    assert "0011-12-15 18:00:00" in out_text          # val end exclusive
    assert "/data" in out_text and "/exp" in out_text


def test_render_smoke_yaml_against_real_manifest(tmp_path: Path) -> None:
    """Smoke: render the actual smoke YAML against the real on-disk manifest."""
    manifest_path = Path("/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/group_sfno/sim52_smoke/_v10_calendar_manifest.json")
    if not manifest_path.is_file():
        import pytest
        pytest.skip("on-disk manifest not present (Phase B.1 has not run on this checkout)")
    tpl_path = REPO_ROOT / "src/sfno_training_group/config/plasim_sim52_sigma10_sfno_smoke.yaml"
    out_path = tmp_path / "rendered.yaml"

    result = subprocess.run(
        [
            sys.executable, "-m", "sfno_training_group.tools.render_yaml",
            "--tpl", str(tpl_path), "--out", str(out_path),
            "--data-dir", "/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/group_sfno/sim52_smoke",
            "--exp-dir", "/scratch/11114/zhixingliu/SFNO_Climate_Emulator/runs/sfno_group_sigma10_smoke",
            "--manifest", str(manifest_path),
            "--train-years", "12", "13",
            "--val-years", "11",
        ],
        env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"render_yaml CLI failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    out_text = out_path.read_text()
    assert "{{" not in out_text, "leftover placeholders in rendered YAML"
    # End-exclusive sanity: the val end string must equal last_val_init_dt + 6h.
    manifest = json.loads(manifest_path.read_text())
    y11 = next(y for y in manifest["years"] if y["year"] == 11)
    assert y11["val_end_exclusive_dt_for_max_lead_K"] in out_text
