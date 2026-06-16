"""CLI flag-parsing + validation tests for scripts/infer_sfno5410_byo_ic.py.

Tier 1: pure argparse, no GPU, no upstream import. We cherry-pick
``parse_init_datetime`` and run argparse with --help / known-bad combos
to validate the CLI's invariants.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "infer_sfno5410_byo_ic.py"


def _load_script_module():
    """Import the script as a module without running main()."""
    spec = importlib.util.spec_from_file_location("infer_sfno5410_byo_ic", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_init_datetime_happy():
    mod = _load_script_module()
    assert mod.parse_init_datetime("2026-05-09_06:00:00") == (2026, 5, 9, 6)
    assert mod.parse_init_datetime("0001-01-01_00:00:00") == (1, 1, 1, 0)


def test_parse_init_datetime_rejects_subhour():
    mod = _load_script_module()
    import argparse
    with pytest.raises(argparse.ArgumentTypeError, match="sub-hour"):
        mod.parse_init_datetime("2026-05-09_06:30:00")
    with pytest.raises(argparse.ArgumentTypeError, match="sub-hour"):
        mod.parse_init_datetime("2026-05-09_06:00:30")


def test_parse_init_datetime_rejects_off_grid_hour():
    mod = _load_script_module()
    import argparse
    with pytest.raises(argparse.ArgumentTypeError, match="multiple of 6"):
        mod.parse_init_datetime("2026-05-09_03:00:00")
    with pytest.raises(argparse.ArgumentTypeError, match="multiple of 6"):
        mod.parse_init_datetime("2026-05-09_15:00:00")


def test_parse_init_datetime_rejects_garbage():
    mod = _load_script_module()
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        mod.parse_init_datetime("not-a-date")


def _run_cli(*args, expect_fail: bool = True) -> subprocess.CompletedProcess:
    """Run the script with argv; we expect argparse failure BEFORE upstream import."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_cli_help_exits_zero():
    p = _run_cli("--help", expect_fail=False)
    assert p.returncode == 0, p.stderr
    assert "BYO" in p.stdout or "user" in p.stdout.lower()
    assert "--ic-nc" in p.stdout
    assert "--num-members" in p.stdout
    assert "--epsilon-factor" in p.stdout


def test_cli_rejects_ensemble_without_perturbation(tmp_path):
    """num-members>1 + epsilon=0 must fail at argparse, before upstream import."""
    p = _run_cli(
        "--ic-nc", str(tmp_path / "no_such.nc"),  # never read; argparse fails first
        "--init-datetime", "0001-01-01_00:00:00",
        "--horizon-days", "1.0",
        "--num-members", "4",
        "--epsilon-factor", "0.0",
        "--output-dir", str(tmp_path / "out"),
    )
    assert p.returncode != 0
    assert "bit-identical" in p.stderr or "epsilon" in p.stderr.lower()


def test_cli_rejects_epsilon_without_perturbation_type(tmp_path):
    p = _run_cli(
        "--ic-nc", str(tmp_path / "no_such.nc"),
        "--init-datetime", "0001-01-01_00:00:00",
        "--horizon-days", "1.0",
        "--num-members", "4",
        "--epsilon-factor", "1e-3",
        # --perturbation-type omitted
        "--output-dir", str(tmp_path / "out"),
    )
    assert p.returncode != 0
    assert "perturbation-type" in p.stderr.lower()


def test_cli_rejects_horizon_over_cap(tmp_path):
    p = _run_cli(
        "--ic-nc", str(tmp_path / "no_such.nc"),
        "--init-datetime", "0001-01-01_00:00:00",
        "--horizon-days", "366.0",
        "--output-dir", str(tmp_path / "out"),
    )
    assert p.returncode != 0
    assert "exceeds cap" in p.stderr or "365" in p.stderr


def test_cli_rejects_negative_horizon(tmp_path):
    p = _run_cli(
        "--ic-nc", str(tmp_path / "no_such.nc"),
        "--init-datetime", "0001-01-01_00:00:00",
        "--horizon-days", "-1",
        "--output-dir", str(tmp_path / "out"),
    )
    assert p.returncode != 0
    assert "must be positive" in p.stderr.lower() or "positive" in p.stderr.lower()


def test_cli_rejects_invalid_perturbation_type(tmp_path):
    p = _run_cli(
        "--ic-nc", str(tmp_path / "no_such.nc"),
        "--init-datetime", "0001-01-01_00:00:00",
        "--horizon-days", "1.0",
        "--num-members", "4",
        "--epsilon-factor", "1e-3",
        "--perturbation-type", "uniform_noise",  # not in choices
        "--output-dir", str(tmp_path / "out"),
    )
    assert p.returncode != 0
    assert "perturbation-type" in p.stderr.lower() or "choose from" in p.stderr.lower()
