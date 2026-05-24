"""Makani parse_dataset_metadata: produce in==out==[0..52] from data.json.

Exercises the parser end-to-end with the metadata.json our packager
writes. The smoke test also asserts this, but this focused test runs
without needing full torch / stats / dataset construction, so failures
are easier to localize.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

makani = pytest.importorskip("makani")
from makani.utils.YParams import YParams  # noqa: E402
from makani.utils.parse_dataset_metada import parse_dataset_metadata  # noqa: E402

from plasim_makani_packager import metadata as meta_module
from plasim_makani_packager.channels import (
    DIAGNOSTIC_CHANNELS,
    FORCING_CHANNELS,
    STATE_CHANNELS,
    TARGET_CHANNELS,
)


def _minimal_packaged_file(path: Path) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "fields_state", data=np.zeros((1, 52, 64, 128), dtype=np.float32)
        )
        f.create_dataset(
            "fields_diagnostic", data=np.zeros((1, 1, 64, 128), dtype=np.float32)
        )
        f.create_dataset(
            "forcing", data=np.zeros((1, 6, 64, 128), dtype=np.float32)
        )
        f.create_dataset("lat", data=np.linspace(-80, 80, 64, dtype=np.float64))
        f.create_dataset(
            "lon", data=np.linspace(0.0, 357.1875, 128, dtype=np.float64)
        )
        f.create_dataset("timestamp", data=np.zeros((1,), dtype=np.int64))
        f.create_dataset("time_plasim", data=np.zeros((1,), dtype=np.float64))
        f.create_dataset(
            "channel_state", data=np.array(STATE_CHANNELS, dtype="S16")
        )
        f.create_dataset(
            "channel_diagnostic",
            data=np.array(DIAGNOSTIC_CHANNELS, dtype="S16"),
        )
        f.create_dataset(
            "channel_forcing", data=np.array(FORCING_CHANNELS, dtype="S16")
        )
        # Provenance attrs the packager stamps in production; metadata.py
        # self-consistency check reads these.
        f.attrs["rsdt_method"] = "astronomical"
        f.attrs["sst_mode"] = "ocean_era5"


def test_parser_sees_53_channels(tmp_path: Path):
    (tmp_path / "train").mkdir()
    _minimal_packaged_file(tmp_path / "train" / "MOST.0003.h5")

    md = meta_module.build_metadata(
        tmp_path,
        dataset_name="plasim-test",
        train_years=(3, 100),
        valid_years=(101, 120),
        test_years=(121, 128),
        sst_land_fill_k=271.35,
        rsdt_method="astronomical",
        packager_version="test",
    )
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir()
    (meta_dir / "data.json").write_text(json.dumps(md))

    # Minimal YAML with just the keys the parser touches.
    cfg_text = (
        "plasim_test:\n"
        f"  metadata_json_path: \"{meta_dir / 'data.json'}\"\n"
        f"  channel_names: {list(TARGET_CHANNELS)}\n"
    )
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_text)

    params = YParams(str(cfg_path), "plasim_test", print_params=False)
    parse_dataset_metadata(params.metadata_json_path, params=params)

    assert list(params.in_channels) == list(range(53))
    assert list(params.out_channels) == list(range(53))
