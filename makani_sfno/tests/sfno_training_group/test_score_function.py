"""Tests for the score-function wrapper (Phase G).

These exercise the cheap pieces (DatasetShim channel counts, EMA-vs-model_state
preference) on CPU. Full forward-pass tests require the group conda env + the
heavy SFNO_v2 model and are guarded by a marker that skips when the env is
absent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sfno_training_group.score_function._dataset_shim import DatasetShim  # noqa: E402


def test_dataset_shim_channel_counts() -> None:
    shim = DatasetShim(
        upper_air_variables=["ta", "ua", "va", "hus", "zg"],
        surface_variables=["pl", "tas"],
        diagnostic_variables=["pr_6h"],
        varying_boundary_variables=["z0", "sst", "rsdt", "sic"],
        constant_boundary_variables=["lsm", "sg"],
        sigma_levels=[0.0383, 0.1191, 0.21085, 0.31685, 0.4368, 0.5668, 0.69935, 0.82335, 0.9241, 0.9833],
        levels=[20000, 25000, 30000, 40000, 50000, 60000, 70000, 85000, 92500, 100000],
        use_sigma_levels=True,
    )
    # 50 upper-air per-level + 2 surface + 4 varying-boundary = 56
    assert len(shim.variable_list_in) == 56
    # 50 upper-air per-level + 2 surface + 1 diag = 53
    assert len(shim.variable_list_out) == 53
    # SFNO line 749: in_chans = variable_list_in + constant_boundary
    assert shim.in_chans == 58
    assert shim.out_chans == 53
    # Verify ordering / structure of upper-air keys.
    assert shim.upper_air_keys[0].startswith("ta_")
    assert shim.upper_air_keys[-1].startswith("zg_")
    assert "zg_50000.0" in shim.upper_air_keys
    # last in: varying_boundary at the end of variable_list_in
    assert shim.variable_list_in[-1] == "sic"
    # last out: pr_6h at the end of variable_list_out
    assert shim.variable_list_out[-1] == "pr_6h"


def test_dataset_shim_no_sigma_for_zg() -> None:
    """Even when use_sigma_levels=True, zg keys use pressure levels."""
    shim = DatasetShim(
        upper_air_variables=["zg"],
        surface_variables=["pl"],
        diagnostic_variables=["pr_6h"],
        varying_boundary_variables=["sst"],
        constant_boundary_variables=["lsm"],
        sigma_levels=[0.5],
        levels=[50000],
        use_sigma_levels=True,
    )
    # zg should NOT use sigma; key should be zg_50000.0
    assert shim.upper_air_keys == ["zg_50000.0"]


_GROUP_ENV = Path("/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/envs/group_pangu_sfno_v2/bin/python")


@pytest.mark.skipif(not _GROUP_ENV.is_file(),
                    reason="group conda env not built (Phase A not run)")
def test_group_emulator_module_imports_in_group_env() -> None:
    """Smoke: GroupEmulator module imports cleanly under the group env."""
    import subprocess
    result = subprocess.run(
        [str(_GROUP_ENV), "-c",
         "import sys; sys.path.insert(0, '/work2/09979/awikner/stampede3/PanguWeather/v2.0');"
         "sys.path.insert(0, '" + str(REPO_ROOT / "src") + "');"
         "from sfno_training_group.score_function.group_emulator import GroupEmulator;"
         "print('OK')"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr={result.stderr}\nstdout={result.stdout}"
    assert "OK" in result.stdout


def test_ema_state_preference_logic() -> None:
    """Mock-only: verify the prefer_ema branch picks ema_state when present.

    We don't construct GroupEmulator (heavy SFNO model build), but we test the
    contract by simulating the ckpt-load logic in isolation.
    """
    # Mirror the wrapper's state-selection logic:
    ckpt_with_ema = {"model_state": {"a": 1}, "ema_state": {"a": 2}}
    ckpt_without_ema = {"model_state": {"a": 1}, "ema_state": None}
    ckpt_no_ema_key = {"model_state": {"a": 1}}

    def select(ckpt: dict, prefer: bool) -> tuple[str, dict]:
        if prefer and ckpt.get("ema_state") is not None:
            return "ema_state", ckpt["ema_state"]
        return "model_state", ckpt["model_state"]

    assert select(ckpt_with_ema, True) == ("ema_state", {"a": 2})
    assert select(ckpt_with_ema, False) == ("model_state", {"a": 1})
    assert select(ckpt_without_ema, True) == ("model_state", {"a": 1})
    assert select(ckpt_no_ema_key, True) == ("model_state", {"a": 1})


def test_module_prefix_strip_logic() -> None:
    """Verify DDP 'module.' prefix is stripped if present."""
    # Mimic the wrapper's strip step.
    state_dict_ddp = {"module.layer.weight": "w", "module.layer.bias": "b"}
    state_dict_plain = {"layer.weight": "w", "layer.bias": "b"}

    def maybe_strip(sd: dict) -> dict:
        if any(k.startswith("module.") for k in sd.keys()):
            return {k.removeprefix("module."): v for k, v in sd.items()}
        return sd

    assert maybe_strip(state_dict_ddp) == state_dict_plain
    assert maybe_strip(state_dict_plain) == state_dict_plain
