"""Phase 4b Makani smoke: YParams → PlasimForcingDataset → PlasimSingleStepWrapper.

This test asserts the full chain from rendered YAML through dataloader
through patched preprocessor to (stock) LossHandler, on synthetic
one-year HDF5 data + stats shipped by packager.py. Positive + negative
multistep rollout probes also run so the trainer-patch PR cannot silently
regress the strip invariants.

Skipped when torch / makani / physicsnemo aren't importable (e.g. on the
Stampede3 login node). Actual execution happens in CI on a node with the
full Makani dependency set.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
makani = pytest.importorskip("makani")
from torch import nn  # noqa: E402  -- after importorskip

from makani.models.stepper import MultiStepWrapper
from makani.utils.YParams import YParams
from makani.utils.loss import LossHandler
from makani.utils.parse_dataset_metada import parse_dataset_metadata

from plasim_makani_packager import metadata as meta_module
from plasim_makani_packager.packager import process_one
from plasim_makani_packager.stats import compute_stats

from stub_forcing_loader import (  # noqa: E402 -- conftest adjusts sys.path
    PlasimForcingDataset,
    PlasimMultiStepWrapper,
    PlasimPreprocessor,
    PlasimSingleStepWrapper,
)
from test_hdf5_writer import (  # noqa: E402
    _make_boundary_file,
    _make_most_file,
)


@pytest.fixture(scope="module")
def packaged_dataset(tmp_path_factory) -> Path:
    """Build synthetic single-year dataset + stats + metadata."""
    root = tmp_path_factory.mktemp("packaged")
    sim = 52
    years = (3, 4, 5)  # three training years
    postproc_root = root / "postproc"
    boundary_root = root / "boundary"
    output_root = root / "out"

    (postproc_root / f"sim{sim}").mkdir(parents=True)
    (boundary_root / f"sim{sim}").mkdir(parents=True)

    T_small = 10
    for y in years:
        most_path = postproc_root / f"sim{sim}" / f"MOST.{y:04d}.nc"
        bnd_path = boundary_root / f"sim{sim}" / f"boundary.{y:04d}.nc"
        _make_most_file(most_path, T=T_small)
        _make_boundary_file(bnd_path, most_path)

    opts = Namespace(
        sims=[sim],
        train_years=[3, 100],
        valid_years=[101, 120],
        test_years=[121, 128],
        postproc_root=postproc_root,
        boundary_root=boundary_root,
        output_root=output_root,
        sst_land_fill_k=271.35,
        postprocessor_git_sha="testsha0123456789abcdef",
        task_index=None,
        count_tasks=False,
        overwrite=False,
        dry_run=False,
        verbose=False,
    )
    for y in years:
        process_one(sim, y, opts)

    compute_stats(output_root, train_years=(3, 5))

    tpl = (
        Path(meta_module.__file__).resolve().parent
        / "templates"
        / "plasim_64x128_zgplev.yaml"
    )
    md = meta_module.build_metadata(
        output_root,
        dataset_name="plasim-sim52-astro-64x128-zgplev",
        train_years=(3, 5),
        valid_years=(101, 120),
        test_years=(121, 128),
        sst_land_fill_k=271.35,
        rsdt_method="astronomical",
        packager_version="test",
    )
    rendered = meta_module.render_yaml(
        tpl,
        output_root=output_root,
        exp_dir=root / "runs",
        config_name="plasim_sim52_astro_64x128_zgplev",
    )
    meta_module.write_outputs(
        output_root,
        metadata=md,
        rendered_yaml=rendered,
        config_name="plasim_sim52_astro_64x128_zgplev",
    )
    return output_root


def _load_params(output_root: Path) -> YParams:
    cfg = output_root / "config" / "plasim_sim52_astro_64x128_zgplev.yaml"
    params = YParams(str(cfg), "plasim_sim52_astro_64x128_zgplev", print_params=False)
    parse_dataset_metadata(params.metadata_json_path, params=params)
    # Preprocessor2D-required shape fields (normally set by driver._set_data_shapes)
    params.img_shape_x_resampled = params.img_shape_x
    params.img_shape_y_resampled = params.img_shape_y
    params.img_crop_shape_x = params.img_shape_x
    params.img_crop_shape_y = params.img_shape_y
    params.img_crop_offset_x = 0
    params.img_crop_offset_y = 0
    params.img_local_shape_x = params.img_shape_x
    params.img_local_shape_y = params.img_shape_y
    params.img_local_offset_x = 0
    params.img_local_offset_y = 0
    params.img_local_shape_x_resampled = params.img_shape_x
    params.img_local_shape_y_resampled = params.img_shape_y
    params.subsampling_factor = 1
    params.n_history = 0
    params.n_future = 0
    params.history_normalization_mode = "none"
    # Preprocessor2D reads these (params.target for residual learning branch)
    params.target = "tendency"
    params.normalize_residual = False
    return params


def _build_dataset(params: YParams, n_future: int = 0) -> PlasimForcingDataset:
    return PlasimForcingDataset(
        location=params.train_data_path,
        dt=1,
        in_channels=list(range(params.n_state_channels)),
        out_channels=list(range(params.n_state_channels + params.n_diagnostic_channels)),
        n_forcing_channels=params.n_forcing_channels,
        n_history=0,
        n_future=n_future,
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


def test_metadata_consistency(packaged_dataset: Path):
    meta = json.loads((packaged_dataset / "metadata" / "data.json").read_text())
    # coords.channel[:52] must equal coords.channel_state; [52] must equal diag[0]
    assert meta["coords"]["channel"][:52] == meta["coords"]["channel_state"]
    assert meta["coords"]["channel"][52] == meta["coords"]["channel_diagnostic"][0]


def test_yparams_and_parse_dataset_metadata(packaged_dataset: Path):
    params = _load_params(packaged_dataset)
    assert list(params.in_channels) == list(range(53))
    assert list(params.out_channels) == list(range(53))
    assert params.data_grid_type == "legendre-gauss"
    assert params.dhours == 6


def test_dataset_returns_expected_shapes_single_step(packaged_dataset: Path):
    params = _load_params(packaged_dataset)
    ds = _build_dataset(params, n_future=0)
    assert ds.in_channels.tolist() == list(range(52))
    assert ds.out_channels.tolist() == list(range(53))

    inp_state, tar, inp_forcing, tar_forcing = ds[0]
    # Stock MultifilesDataset contract: (n_history+1, C, H, W) / (n_future+1, C, H, W)
    assert inp_state.shape == (1, 52, 64, 128)
    assert tar.shape == (1, 53, 64, 128)
    assert inp_forcing.shape == (1, 6, 64, 128)
    assert tar_forcing.shape == (1, 6, 64, 128)


def test_dataset_returns_expected_shapes_two_step(packaged_dataset: Path):
    params = _load_params(packaged_dataset)
    ds = _build_dataset(params, n_future=1)
    inp_state, tar, inp_forcing, tar_forcing = ds[0]
    assert inp_state.shape == (1, 52, 64, 128)
    assert tar.shape == (2, 53, 64, 128)
    assert inp_forcing.shape == (1, 6, 64, 128)
    assert tar_forcing.shape == (2, 6, 64, 128)


def test_single_step_wrapper_positive(packaged_dataset: Path):
    params = _load_params(packaged_dataset)
    ds = _build_dataset(params, n_future=0)
    inp_state, tar, inp_forcing, tar_forcing = ds[0]

    # _set_data_shapes + PlasimTrainer override
    params.N_in_channels = params.n_state_channels + params.n_forcing_channels
    params.N_out_channels = 53

    # Fresh Conv2d per wrapper ctor (plan v9 fix #6 — shared instance re-registers params)
    model_handle = lambda: nn.Conv2d(in_channels=58, out_channels=53, kernel_size=1)

    wrapper = PlasimSingleStepWrapper(params, model_handle)
    wrapper.train()
    assert isinstance(wrapper.preprocessor, PlasimPreprocessor)

    inp_b = inp_state.unsqueeze(0)    # (1, 1, 52, 64, 128)
    tar_b = tar.unsqueeze(0)          # (1, 1, 53, 64, 128)
    xz = inp_forcing.unsqueeze(0)     # (1, 1, 6, 64, 128)
    yz = tar_forcing.unsqueeze(0)     # (1, 1, 6, 64, 128)

    inp_b, tar_b = wrapper.preprocessor.cache_unpredicted_features(inp_b, tar_b, xz, yz)
    inp_b = wrapper.preprocessor.flatten_history(inp_b)
    tar_b = wrapper.preprocessor.flatten_history(tar_b)

    pred = wrapper(inp_b)
    assert pred.shape == (1, 53, 64, 128)

    loss_fn = LossHandler(params)
    assert loss_fn.channel_weights.shape[1] == 53
    loss_val = loss_fn(pred, tar_b)
    assert torch.isfinite(loss_val).item()


def test_preprocessor_append_history_strip(packaged_dataset: Path):
    params = _load_params(packaged_dataset)
    # Build wrapper to install the preprocessor with the right n_channels.
    model_handle = lambda: nn.Conv2d(58, 53, 1)
    wrapper = PlasimSingleStepWrapper(params, model_handle)
    wrapper.train()

    # 53 input gets stripped to 52
    x1 = torch.zeros(1, 52, 64, 128)
    x2 = torch.randn(1, 53, 64, 128)
    x_out = wrapper.preprocessor.append_history(x1, x2, step=0, update_state=False)
    assert x_out.shape == (1, 52, 64, 128)

    # Hard-reject unexpected channel count (plan v9 #4)
    with pytest.raises(AssertionError, match="channels must be"):
        wrapper.preprocessor.append_history(x1, torch.zeros(1, 60, 64, 128), step=0)
    with pytest.raises(AssertionError, match="4D"):
        wrapper.preprocessor.append_history(x1, torch.zeros(1, 53, 64), step=0)


def test_multi_step_positive_and_negative(packaged_dataset: Path):
    params = _load_params(packaged_dataset)
    ds = _build_dataset(params, n_future=1)
    inp_state2, tar2, inp_forcing2, tar_forcing2 = ds[0]

    params.N_in_channels = params.n_state_channels + params.n_forcing_channels
    params.N_out_channels = 53
    params.n_future = 1

    # Rebuild LossHandler after n_future flip (plan v9 #4 — multistep_weight
    # is 106-wide, channel_weights stays 53-wide; forward tiles).
    loss_fn_multi = LossHandler(params)
    assert loss_fn_multi.channel_weights.shape[1] == 53
    assert loss_fn_multi.multistep_weight.shape[1] == 53 * 2

    model_handle = lambda: nn.Conv2d(58, 53, 1)

    # Positive: PlasimMultiStepWrapper produces (B, 53*(n_future+1), H, W)
    ms_patched = PlasimMultiStepWrapper(params, model_handle)
    ms_patched.train()
    assert isinstance(ms_patched.preprocessor, PlasimPreprocessor)

    inp_b = inp_state2.unsqueeze(0)       # (1, 1, 52, 64, 128)
    tar_b = tar2.unsqueeze(0)             # (1, 2, 53, 64, 128)
    xz = inp_forcing2.unsqueeze(0)        # (1, 1, 6, 64, 128)
    yz = tar_forcing2.unsqueeze(0)        # (1, 2, 6, 64, 128)

    inp_b, tar_b = ms_patched.preprocessor.cache_unpredicted_features(inp_b, tar_b, xz, yz)
    inp_flat = ms_patched.preprocessor.flatten_history(inp_b)
    tar_flat = ms_patched.preprocessor.flatten_history(tar_b)

    pred_ms = ms_patched(inp_flat)
    assert pred_ms.shape == (1, 53 * 2, 64, 128)
    loss_ms = loss_fn_multi(pred_ms, tar_flat)
    assert torch.isfinite(loss_ms).item()

    # Negative regression: stock MultiStepWrapper (no PlasimPreprocessor) must FAIL.
    ms_stock = MultiStepWrapper(params, model_handle)
    ms_stock.train()
    inp_b3 = inp_state2.unsqueeze(0)
    tar_b3 = tar2.unsqueeze(0)
    xz3 = inp_forcing2.unsqueeze(0)
    yz3 = tar_forcing2.unsqueeze(0)
    inp_b3, tar_b3 = ms_stock.preprocessor.cache_unpredicted_features(inp_b3, tar_b3, xz3, yz3)
    inp_flat3 = ms_stock.preprocessor.flatten_history(inp_b3)
    with pytest.raises((RuntimeError, AssertionError)):
        _ = ms_stock(inp_flat3)
