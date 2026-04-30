"""Shared helpers for tests/sfno_training/.

Kept separate from conftest.py because Python's import system can't
disambiguate ``from conftest import X`` between sibling test
directories (the repo also has ``tests/plasim_makani_packager/conftest.py``).
Tests here import these as ``from helpers import ...``; the local
conftest puts ``tests/sfno_training`` on ``sys.path``.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

# Heavy deps gate -- mirrors conftest. The conftest's importorskip already
# covers the package-level skip; this is a defensive secondary gate so
# `from helpers import X` at the top of a test module doesn't error out
# of the importer's call before pytest's own skip logic runs.
torch = pytest.importorskip("torch")
makani = pytest.importorskip("makani")

import torch.nn as nn  # noqa: E402  -- after importorskip


class RecordingDummyModel(nn.Module):
    """Minimal nettype that:

    - records every input tensor it sees on ``self.inputs_seen`` (so
      content sentinels can verify channel layout per step);
    - emits a sentinel pr_6h channel value (``-9999.0``) — if that ever
      shows up in the next-step state input, the strip is broken;
    - threads a single trainable ``dummy_param`` through the output so
      the loss has a grad path and ``optimizer.step`` actually fires
      (Codex round 2 #2).

    Parameter signature matches what stock ``model_registry.get_model``
    passes through ``functools.partial`` at ``model_registry.py:187``::

        partial(model_handle, inp_shape=..., out_shape=..., inp_chans=...,
                out_chans=..., **model_kwargs)

    The ``**kw`` swallow absorbs SFNO-specific knobs in ``model_kwargs``
    that this dummy doesn't need.
    """

    PR_6H_SENTINEL: float = -9999.0

    def __init__(self, inp_shape, out_shape, inp_chans, out_chans, **kw):
        super().__init__()
        self.inp_chans = inp_chans
        self.out_chans = out_chans
        self.inputs_seen: list[torch.Tensor] = []
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.inputs_seen.append(x.detach().clone())
        out = torch.zeros(
            x.shape[0],
            self.out_chans,
            x.shape[2],
            x.shape[3],
            device=x.device,
            dtype=x.dtype,
        )
        # Sentinel on the trailing diagnostic slot. If the strip is broken,
        # this leaks into the next forward's state portion (channels 0..51).
        out[:, -1:, ...] = self.PR_6H_SENTINEL
        # Grad path through the (zero-effect) trainable parameter.
        return out + 0.0 * self.dummy_param.sum()


def load_params(output_root: Path):
    """Load YParams from the synthetic packaged-dataset config and
    populate Preprocessor2D-required shape attrs (normally set by
    ``Driver._set_data_shapes``)."""
    from makani.utils.YParams import YParams
    from makani.utils.parse_dataset_metada import parse_dataset_metadata

    cfg = output_root / "config" / "plasim_sim52_astro_64x128_zgplev.yaml"
    params = YParams(str(cfg), "plasim_sim52_astro_64x128_zgplev", print_params=False)
    parse_dataset_metadata(params.metadata_json_path, params=params)

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
    params.target = "tendency"
    params.normalize_residual = False
    return params


def build_dataset(params, output_root: Path, *, n_future: int = 0, split: str = "train"):
    """Build a PlasimForcingDataset against the packaged synthetic data."""
    from sfno_training.data import PlasimForcingDataset

    location = {
        "train": params.train_data_path,
        "valid": params.valid_data_path,
    }[split]

    return PlasimForcingDataset(
        location=location,
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


def build_packaged_dataset(tmp_path_factory, *, fixture_name: str = "packaged_sfno") -> Path:
    """Build a tiny packaged dataset (3 train years, sim52 layout, T=10/year)
    on which all sfno_training tests run.

    Mirrors ``tests/plasim_makani_packager/test_multifile_loader_smoke.py``
    so PR-A and PR-B don't need their own packager wiring.
    """
    from plasim_makani_packager import metadata as meta_module
    from plasim_makani_packager.packager import process_one
    from plasim_makani_packager.stats import compute_stats

    from test_hdf5_writer import _make_boundary_file, _make_most_file

    root = tmp_path_factory.mktemp(fixture_name)
    sim = 52
    train_years = (3, 4, 5)
    valid_years = (101,)        # at least one valid file so PlasimTrainer init succeeds
    test_years = (121,)         # ditto for test (cheap; some Driver paths may peek)
    all_years = (*train_years, *valid_years, *test_years)
    postproc_root = root / "postproc"
    boundary_root = root / "boundary"
    output_root = root / "out"

    (postproc_root / f"sim{sim}").mkdir(parents=True)
    (boundary_root / f"sim{sim}").mkdir(parents=True)

    T_small = 10
    for y in all_years:
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
    for y in all_years:
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
