"""metadata.build_metadata + render_yaml: 53 target names, absolute paths."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from plasim_makani_packager.channels import (
    DIAGNOSTIC_CHANNELS,
    FORCING_CHANNELS,
    STATE_CHANNELS,
    TARGET_CHANNELS,
)
from plasim_makani_packager.metadata import (
    DEFAULT_CONFIG_NAME,
    build_metadata,
    render_yaml,
    write_outputs,
)


def _minimal_packaged_file(path: Path, *, H: int = 64, W: int = 128) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "fields_state", data=np.zeros((1, 52, H, W), dtype=np.float32)
        )
        f.create_dataset(
            "fields_diagnostic", data=np.zeros((1, 1, H, W), dtype=np.float32)
        )
        f.create_dataset(
            "forcing", data=np.zeros((1, 6, H, W), dtype=np.float32)
        )
        f.create_dataset("lat", data=np.linspace(-80, 80, H, dtype=np.float64))
        f.create_dataset("lon", data=np.linspace(0, 357.1875, W, dtype=np.float64))
        f.create_dataset(
            "timestamp", data=np.zeros((1,), dtype=np.int64)
        )
        f.create_dataset(
            "time_plasim", data=np.zeros((1,), dtype=np.float64)
        )
        f.create_dataset(
            "channel_state", data=np.array(STATE_CHANNELS, dtype="S16")
        )
        f.create_dataset(
            "channel_diagnostic", data=np.array(DIAGNOSTIC_CHANNELS, dtype="S16")
        )
        f.create_dataset(
            "channel_forcing", data=np.array(FORCING_CHANNELS, dtype="S16")
        )


def test_build_metadata_content(tmp_path: Path):
    train = tmp_path / "train"
    train.mkdir()
    _minimal_packaged_file(train / "MOST.0003.h5")

    md = build_metadata(
        tmp_path,
        dataset_name="plasim-sim52-astro-64x128",
        train_years=(3, 100),
        valid_years=(101, 120),
        test_years=(121, 128),
        sst_land_fill_k=271.35,
        rsdt_method="astronomical",
        packager_version="test",
    )
    assert md["h5_path"] == "fields_state"
    assert md["diagnostic_h5_path"] == "fields_diagnostic"
    assert md["forcing_h5_path"] == "forcing"
    assert md["dhours"] == 6

    # 53 target names, ordering locked
    assert md["coords"]["channel"] == list(TARGET_CHANNELS)
    assert md["coords"]["channel_state"] == list(STATE_CHANNELS)
    assert md["coords"]["channel_diagnostic"] == list(DIAGNOSTIC_CHANNELS)
    assert md["coords"]["channel_forcing"] == list(FORCING_CHANNELS)

    assert md["attrs"]["rsdt_method"] == "astronomical"
    assert md["attrs"]["requires_patched_makani"] is True
    assert md["attrs"]["train_years"] == [3, 100]


def test_render_yaml_substitutes_output_root_and_exp_dir(tmp_path: Path):
    template = tmp_path / "template.yaml"
    template.write_text(
        "plasim_sim52_astro_64x128:\n"
        "  train_data_path: \"{{OUTPUT_ROOT}}/train\"\n"
        "  exp_dir: \"{{EXP_DIR}}\"\n"
    )
    out_root = tmp_path / "output"
    exp_dir = tmp_path / "runs"
    rendered = render_yaml(
        template,
        output_root=out_root,
        exp_dir=exp_dir,
        config_name=DEFAULT_CONFIG_NAME,
    )
    assert "{{OUTPUT_ROOT}}" not in rendered
    assert "{{EXP_DIR}}" not in rendered
    assert str(out_root.resolve()) in rendered
    assert str(exp_dir.resolve()) in rendered


def test_write_outputs_produces_files(tmp_path: Path):
    # Stage a packaged file so build_metadata can sample lat/lon.
    train = tmp_path / "train"
    train.mkdir()
    _minimal_packaged_file(train / "MOST.0003.h5")

    md = build_metadata(
        tmp_path,
        dataset_name="d",
        train_years=(3, 100),
        valid_years=(101, 120),
        test_years=(121, 128),
        sst_land_fill_k=271.35,
        rsdt_method="astronomical",
        packager_version="v",
    )
    tpl = tmp_path / "tpl.yaml"
    tpl.write_text("plasim_sim52_astro_64x128:\n  x: \"{{OUTPUT_ROOT}}\"\n")
    rendered = render_yaml(
        tpl,
        output_root=tmp_path,
        exp_dir=tmp_path / "runs",
        config_name=DEFAULT_CONFIG_NAME,
    )
    meta_path, cfg_path = write_outputs(
        tmp_path,
        metadata=md,
        rendered_yaml=rendered,
        config_name=DEFAULT_CONFIG_NAME,
    )
    assert meta_path.exists() and cfg_path.exists()
    loaded = json.loads(meta_path.read_text())
    assert loaded["coords"]["channel"] == list(TARGET_CHANNELS)
