"""PR-A unit test: PlasimForcingDataset shapes / dtypes / channel order /
relative-timestamp wiring on synthetic packaged data.

Extracted from tests/plasim_makani_packager/test_multifile_loader_smoke.py
steps 4-5, 10 per docs/sfno_training_implementation_plan.md §7.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
makani = pytest.importorskip("makani")

from helpers import build_dataset, load_params  # noqa: E402

from sfno_training.data import PlasimForcingDataset  # noqa: E402


def test_dataset_subclass(packaged_dataset: Path):
    """PlasimForcingDataset subclasses MultifilesDataset (so the trainer's
    monkey-patch surface stays correct)."""
    from makani.utils.dataloaders.data_loader_multifiles import MultifilesDataset

    params = load_params(packaged_dataset)
    ds = build_dataset(params, packaged_dataset, n_future=0)
    assert isinstance(ds, MultifilesDataset)


def test_channel_order_matches_metadata(packaged_dataset: Path):
    """metadata.data.json's coords.channel must equal state ‖ diagnostic;
    [:52] is state, [52] is diagnostic."""
    meta = json.loads((packaged_dataset / "metadata" / "data.json").read_text())
    assert meta["coords"]["channel"][:52] == meta["coords"]["channel_state"]
    assert meta["coords"]["channel"][52] == meta["coords"]["channel_diagnostic"][0]


def test_dataset_shapes_single_step(packaged_dataset: Path):
    """n_future=0 returns 4-tuple with stock (T, C, H, W) shapes."""
    params = load_params(packaged_dataset)
    ds = build_dataset(params, packaged_dataset, n_future=0)
    assert ds.in_channels.tolist() == list(range(52))
    assert ds.out_channels.tolist() == list(range(53))

    inp_state, tar, inp_forcing, tar_forcing = ds[0]
    assert inp_state.shape == (1, 52, 64, 128)
    assert tar.shape == (1, 53, 64, 128)
    assert inp_forcing.shape == (1, 6, 64, 128)
    assert tar_forcing.shape == (1, 6, 64, 128)


def test_dataset_shapes_two_step(packaged_dataset: Path):
    """n_future=1 -> tar / tar_forcing get T=2 along the time axis."""
    params = load_params(packaged_dataset)
    ds = build_dataset(params, packaged_dataset, n_future=1)
    inp_state, tar, inp_forcing, tar_forcing = ds[0]
    assert inp_state.shape == (1, 52, 64, 128)
    assert tar.shape == (2, 53, 64, 128)
    assert inp_forcing.shape == (1, 6, 64, 128)
    assert tar_forcing.shape == (2, 6, 64, 128)


def test_dataset_dtypes(packaged_dataset: Path):
    """All four returned tensors are float32."""
    params = load_params(packaged_dataset)
    ds = build_dataset(params, packaged_dataset, n_future=1)
    for arr in ds[0]:
        assert arr.dtype == torch.float32


def test_stats_files_shapes(packaged_dataset: Path):
    """Stats produced by compute_stats have the shapes the dataloader and
    the loss expect."""
    stats_dir = packaged_dataset / "stats"
    expected = {
        "global_means.npy": (1, 53, 1, 1),
        "global_stds.npy": (1, 53, 1, 1),
        "time_means.npy": (1, 53, 64, 128),
        "forcing_global_means.npy": (1, 6, 1, 1),
        "forcing_global_stds.npy": (1, 6, 1, 1),
        "forcing_time_means.npy": (1, 6, 64, 128),
    }
    for name, shape in expected.items():
        arr = np.load(stats_dir / name)
        assert arr.shape == shape, f"{name}: {arr.shape} != {shape}"
        assert arr.dtype == np.float32


def test_relative_timestamp_path(packaged_dataset: Path):
    """relative_timestamp=True -> the dataset's date_fn produces a
    timedelta (compat shim covers Python 3.12 numpy-int64). Construction
    must not raise."""
    params = load_params(packaged_dataset)
    ds = build_dataset(params, packaged_dataset, n_future=0)
    assert ds.relative_timestamp is True
    # Smoke: timestamps array is non-empty, datestamps round-tripped via
    # the patched timedelta cast in sfno_training.compat.
    assert len(ds.timestamps) > 0
    import datetime as dt
    assert isinstance(ds.datestamps[0], dt.timedelta)


def test_timestamp_reset_across_source_splits_uses_file_order(
    packaged_dataset: Path, tmp_path: Path
):
    """A full-training subset can symlink files from multiple original
    packager splits. Those source splits each start /timestamp at zero, so
    the dataloader must keep MOST.xxxx order and synthesize a continuous
    internal time axis.
    """
    params = load_params(packaged_dataset)
    mixed = tmp_path / "mixed_train"
    mixed.mkdir()
    (mixed / "MOST.0005.h5").symlink_to(packaged_dataset / "train" / "MOST.0005.h5")
    (mixed / "MOST.0101.h5").symlink_to(packaged_dataset / "valid" / "MOST.0101.h5")

    ds = PlasimForcingDataset(
        location=str(mixed),
        dt=1,
        in_channels=list(range(params.n_state_channels)),
        out_channels=list(range(params.n_state_channels + params.n_diagnostic_channels)),
        n_forcing_channels=params.n_forcing_channels,
        n_history=0,
        n_future=0,
        diagnostic_dataset_path=params.diagnostic_h5_path,
        forcing_dataset_path=params.forcing_h5_path,
        relative_timestamp=True,
        data_grid_type=params.data_grid_type,
        model_grid_type=params.model_grid_type,
        bias=np.load(params.global_means_path),
        scale=np.load(params.global_stds_path),
        forcing_bias=np.load(params.forcing_global_means_path),
        forcing_scale=np.load(params.forcing_global_stds_path),
    )

    assert [Path(p).name for p in ds.files_paths] == ["MOST.0005.h5", "MOST.0101.h5"]
    assert set(np.diff(ds.timestamps).tolist()) == {21600}

    # Index 9 reads input from MOST.0005 and target from MOST.0101, crossing
    # the original split boundary that used to fail at dataset construction.
    inp_state, tar, inp_forcing, tar_forcing = ds[9]
    assert inp_state.shape == (1, 52, 64, 128)
    assert tar.shape == (1, 53, 64, 128)
    assert inp_forcing.shape == (1, 6, 64, 128)
    assert tar_forcing.shape == (1, 6, 64, 128)


def test_dataset_inference_mode_unsupported_via_loader(packaged_dataset: Path):
    """PlasimForcingDataset itself accepts return_target=False (used by
    stock inference path) -- but PR-B's _plasim_get_dataloader gates
    inference at the trainer level. PR-A only checks the dataset can build
    in the eval / no-target shape."""
    params = load_params(packaged_dataset)
    ds = build_dataset(params, packaged_dataset, n_future=0)
    sample = ds.get_sample_at_index(0, return_target=False)
    # 2-tuple: (inp_state, inp_forcing)
    assert len(sample) == 2
    assert sample[0].shape == (1, 52, 64, 128)
    assert sample[1].shape == (1, 6, 64, 128)
