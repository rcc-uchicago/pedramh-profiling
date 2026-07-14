"""PR-B trainer-CI integration test: PlasimTrainer.train_one_epoch().

End-to-end exercise of:
  - `_install_plasim_patches` (3 module-attribute rebindings)
  - `_plasim_get_dataloader` (constructs PlasimForcingDataset + DataLoader)
  - `_set_data_shapes` (overrides N_in_channels = 58 + asserts aux flags off)
  - one full epoch of `train_one_epoch` on the synthetic packaged dataset
    using `params.nettype = "plasim_test_recording_dummy"` (registered in
    conftest)

Asserts the four wrapper isinstance contracts (dataset / wrapper /
preprocessor / N_in_channels), the optimizer-step contract (relies on
RecordingDummyModel.dummy_param's grad path), and a content sentinel
that every captured model input has shape[1] == 58.

Plan v9 hard gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
makani = pytest.importorskip("makani")

from makani.utils.YParams import YParams  # noqa: E402

from helpers import RecordingDummyModel  # noqa: E402

from sfno_training.data import PlasimForcingDataset  # noqa: E402
from sfno_training.models import (  # noqa: E402
    PlasimMultiStepWrapper,
    PlasimPreprocessor,
    PlasimSingleStepWrapper,
)
from sfno_training.trainer import PlasimTrainer  # noqa: E402


def _populate_runtime_params(params, exp_dir: Path) -> None:
    """Set the params that ``train_plasim.py::main`` populates from CLI args.

    Driver.__init__ + Trainer.__init__ read these directly. We default
    everything to the minimum needed to drive ``train_one_epoch()``
    without touching wandb, dist, or AMP.
    """
    params["experiment_dir"] = str(exp_dir)
    params["checkpoint_path"] = str(
        exp_dir / "training_checkpoints" / "ckpt_mp{mp_rank}_v{checkpoint_version}.tar"
    )
    params["best_checkpoint_path"] = str(
        exp_dir / "training_checkpoints" / "best_ckpt_mp{mp_rank}.tar"
    )
    params["resuming"] = False
    params["amp_mode"] = "none"
    params["jit_mode"] = "none"
    params["skip_validation"] = True
    params["skip_training"] = False
    params["enable_synthetic_data"] = False
    params["enable_s3"] = False
    params["enable_odirect"] = False
    params["checkpointing_level"] = 0
    params["multistep_count"] = 1
    params["n_future"] = 0
    params["disable_ddp"] = True
    params["enable_grad_anomaly_detection"] = False
    params["split_data_channels"] = False
    params["print_timings_frequency"] = 0
    params["load_checkpoint"] = "legacy"
    params["save_checkpoint"] = "none"
    params["pretrained"] = False

    # World / data parallelism
    params["world_size"] = 1
    params["global_batch_size"] = 1
    params["batch_size"] = 1
    params["data_num_shards"] = 1
    params["data_shard_id"] = 0
    params["fin_parallel_size"] = 1
    params["fout_parallel_size"] = 1
    params["h_parallel_size"] = 1
    params["w_parallel_size"] = 1
    params["model_parallel_sizes"] = [1, 1, 1, 1]
    params["model_parallel_names"] = ["h", "w", "fin", "fout"]
    params["parameters_reduction_buffer_count"] = 1
    params["optimizer_max_grad_norm"] = 1.0

    # Logging — keep noise minimal, no wandb in CI.
    # NB: `log_to_screen=True` is required because Driver.__init__ only assigns
    # `self.logger` when log_to_screen is true, and Trainer.__init__ on main now
    # has an unconditional `self.logger.info(...)` in the visualizer-init branch
    # (makani-src/makani/utils/training/deterministic_trainer.py:232).
    params["log_to_screen"] = True
    params["log_to_wandb"] = False
    params["log_video"] = 0
    params["wandb_dir"] = str(exp_dir)
    params["verbose"] = False
    params["num_data_workers"] = 0
    # Trainer's visualizer construction uses ProcessPoolExecutor with this
    # value as max_workers; must be >= 1 even if log_to_wandb=False.
    params["num_visualization_workers"] = 1


def _override_for_smoke(params, *, n_future: int = 0) -> None:
    """Overrides on top of the packager-rendered YAML to drive a single
    train epoch with the recording dummy nettype."""
    params["nettype"] = "plasim_test_recording_dummy"
    params["n_history"] = 0
    params["n_future"] = n_future
    params["multistep_count"] = n_future + 1
    params["max_epochs"] = 1
    params["batch_size"] = 1
    params["n_train_samples_per_epoch"] = 2
    params["n_eval_samples"] = 2
    params["valid_autoreg_steps"] = max(n_future, 1)
    # ReduceLROnPlateau requires validation — swap to a step-based scheduler
    # so train_one_epoch can run with skip_validation=True.
    params["scheduler"] = "CosineAnnealingLR"
    params["scheduler_T_max"] = 1
    params["history_normalization_mode"] = "none"
    # Aux features must all be off — PlasimTrainer asserts this.
    params["add_zenith"] = False
    params["add_grid"] = False
    params["add_orography"] = False
    params["add_landmask"] = False
    params["add_soiltype"] = False
    # Preprocessor2D-required attrs (also set by load_params in helpers.py).
    params["target"] = "tendency"
    params["normalize_residual"] = False


def _load_yparams(packaged_dataset: Path) -> YParams:
    from makani.utils.parse_dataset_metada import parse_dataset_metadata

    cfg = packaged_dataset / "config" / "plasim_sim52_astro_64x128_zgplev.yaml"
    params = YParams(str(cfg), "plasim_sim52_astro_64x128_zgplev", print_params=False)
    parse_dataset_metadata(params.metadata_json_path, params=params)
    return params


def test_trainer_ci_train_one_epoch(packaged_dataset: Path, tmp_path: Path):
    params = _load_yparams(packaged_dataset)
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    (exp_dir / "training_checkpoints").mkdir()

    _populate_runtime_params(params, exp_dir)
    _override_for_smoke(params, n_future=0)

    pt = PlasimTrainer(params, world_rank=0, device="cpu")

    # (1) — (5) wrapper + dataset isinstance contracts
    assert isinstance(pt, PlasimTrainer)
    assert isinstance(pt.train_dataset, PlasimForcingDataset)
    assert isinstance(pt.model, PlasimSingleStepWrapper)
    assert isinstance(pt.model.preprocessor, PlasimPreprocessor)
    assert pt.params.N_in_channels == 58, (
        f"PlasimTrainer should set N_in_channels=58, got {pt.params.N_in_channels}"
    )
    assert len(pt.train_dataloader) == 2
    assert len(pt.valid_dataloader) == 2

    # Drive one epoch of training. Sidesteps full Trainer.train()'s
    # outer-loop behavior (validation, scheduler step, log_epoch,
    # checkpoint write — all skipped per Codex round 2 fix #6).
    _, _, train_logs = pt.train_one_epoch()
    assert train_logs["train_steps"] == 2

    # (6) optimizer-step contract — RecordingDummyModel routes loss through
    # dummy_param so backward + step actually fire.
    step_counts = [
        state.get("step", 0)
        for state in pt.optimizer.state.values()
    ]
    assert step_counts and max(step_counts) >= 1, (
        f"expected optimizer to have stepped at least once; step counts {step_counts}"
    )

    # (7) content sentinel: every recorded model input had 58 channels.
    seen = pt.model.model.inputs_seen
    assert len(seen) >= 1, "RecordingDummyModel was never invoked"
    for k, x in enumerate(seen):
        assert x.shape[1] == 58, f"step {k}: expected 58 channels, got {x.shape[1]}"
        # State portion must never carry the pr_6h sentinel — single-step
        # training only invokes the model once, so this is mostly a smoke
        # check; the multistep version is the real test for diagnostic
        # leakage (test_validation_rollout.py).
        assert not torch.any(x[:, :52] == RecordingDummyModel.PR_6H_SENTINEL), (
            f"step {k}: pr_6h sentinel leaked into state input"
        )
