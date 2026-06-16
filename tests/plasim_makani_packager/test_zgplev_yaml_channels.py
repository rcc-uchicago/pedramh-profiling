"""Drift guard: every zgplev YAML must carry exactly TARGET_CHANNELS.

The SFNO training YAMLs and the packager template hard-code the full
53-element ``channel_names`` list rather than deriving it from
``TARGET_CHANNELS``. They are easy to forget when the pressure-level
subset changes (see docs/2026-05-04_zg1000hpa_migration_plan.md §2).

This test is the canary: any YAML whose channel_names disagrees with
the tuple-of-truth fails this case loudly, before the packager runs or
training launches.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from plasim_makani_packager.channels import TARGET_CHANNELS


_REPO_ROOT = Path(__file__).resolve().parents[2]


# Each entry: (yaml path relative to repo root, top-level config key).
ZGPLEV_YAMLS: tuple[tuple[str, str], ...] = (
    (
        "src/plasim_makani_packager/templates/plasim_64x128_zgplev.yaml",
        "plasim_sim52_astro_64x128_zgplev",
    ),
    (
        "src/sfno_training/config/plasim_sim52_zgplev_full.yaml",
        "plasim_sim52_zgplev_full",
    ),
    (
        "src/sfno_training/config/plasim_sim52_zgplev_baseline.yaml",
        "plasim_sim52_zgplev_baseline",
    ),
    (
        "src/sfno_training/config/plasim_sim52_zgplev_tiny.yaml",
        "plasim_sim52_zgplev_tiny",
    ),
    (
        "src/sfno_training/config/plasim_sim52_zgplev_short.yaml",
        "plasim_sim52_zgplev_short",
    ),
    (
        "src/sfno_training/config/plasim_sim52_zgplev_smoke.yaml",
        "plasim_sim52_zgplev_smoke",
    ),
)


@pytest.mark.parametrize("rel_path,config_key", ZGPLEV_YAMLS)
def test_yaml_channel_names_matches_target_channels(rel_path: str, config_key: str):
    path = _REPO_ROOT / rel_path
    assert path.exists(), f"missing zgplev YAML: {path}"

    blocks = yaml.safe_load(path.read_text())
    assert config_key in blocks, (
        f"{path}: top-level key {config_key!r} not found "
        f"(got {list(blocks.keys())!r})"
    )

    cfg = blocks[config_key]
    assert "channel_names" in cfg, f"{path}[{config_key}]: missing channel_names"

    assert cfg["channel_names"] == list(TARGET_CHANNELS), (
        f"{path}[{config_key}]: channel_names disagrees with TARGET_CHANNELS.\n"
        f"  yaml: {cfg['channel_names']}\n"
        f"  truth: {list(TARGET_CHANNELS)}"
    )
