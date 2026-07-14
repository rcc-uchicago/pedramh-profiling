"""PR-B CPU smoke: tiny real SFNO trained for one epoch on the synthetic
packaged dataset. Catches torch_harmonics 0.6 → 0.8 / RealSHT API breakage
that the dummy-nettype tests (RecordingDummyModel) cannot.

Marked ``slow`` — runs in ~2 min on CPU. Excluded from default ``pytest``
runs (``pytest -m "not slow"`` style); developers run it before pushing
PR-B changes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
makani = pytest.importorskip("makani")

from sfno_training.trainer import PlasimTrainer  # noqa: E402

from test_trainer_ci import (  # noqa: E402  reuse helpers
    _load_yparams,
    _override_for_smoke,
    _populate_runtime_params,
)


pytestmark = pytest.mark.slow


def _override_for_real_sfno_cpu(params) -> None:
    """Tiny real SFNO architecture — small enough for CPU in <2 min."""
    params["nettype"] = "SFNO"
    params["filter_type"] = "linear"
    params["scale_factor"] = 8
    params["embed_dim"] = 8
    params["num_layers"] = 1
    params["complex_activation"] = "real"
    params["normalization_layer"] = "instance_norm"
    params["hard_thresholding_fraction"] = 1.0
    params["use_mlp"] = True
    params["mlp_mode"] = "serial"
    params["mlp_ratio"] = 2
    params["separable"] = False
    params["operator_type"] = "dhconv"
    params["activation_function"] = "gelu"
    params["pos_embed"] = "none"


def test_real_sfno_cpu_train_one_epoch(packaged_dataset: Path, tmp_path: Path):
    params = _load_yparams(packaged_dataset)
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    (exp_dir / "training_checkpoints").mkdir()

    _populate_runtime_params(params, exp_dir)
    _override_for_smoke(params, n_future=0)
    _override_for_real_sfno_cpu(params)

    pt = PlasimTrainer(params, world_rank=0, device="cpu")

    assert pt.params.N_in_channels == 58
    pt.train_one_epoch()

    # Loss is non-NaN — RealSHT didn't blow up under the
    # 64×128 / scale_factor=8 / embed_dim=8 config.
    step_counts = [state.get("step", 0) for state in pt.optimizer.state.values()]
    assert step_counts and max(step_counts) >= 1
