"""Integration tests for the EMA path through ``PlasimTrainer``.

Mirrors the construction style of ``test_trainer_ci.py`` (synthetic
packaged dataset + ``RecordingDummyModel`` nettype + cpu-only) but exercises
the EMA-on path:

- EMA-on trainer construction wires up the optimizer post-step hook and
  seeds the shadow.
- ``train_one_epoch`` drives the hook; ``ema.step`` matches the number of
  optimizer steps that fired.
- ``save_checkpoint`` reopens the per-epoch file and appends the four EMA
  keys (``ema_state``, ``ema_step``, ``ema_config``, ``ema_best_loss``).
- The §6 logging routing contract: EMA scalars land in
  ``valid_logs["metrics"]`` (not ``["base"]``) so they print on screen
  even when ``log_to_wandb=False``.
- The §3.5 / Goal #7 hard-error path: ``__init__`` rejects EMA + flexible
  on either ``save_checkpoint`` or ``load_checkpoint``.

These tests share the ``packaged_dataset`` fixture from conftest.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
makani = pytest.importorskip("makani")

from makani.utils.YParams import YParams  # noqa: E402

from sfno_training.trainer import EMAModel, PlasimTrainer  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers — copied/condensed from test_trainer_ci.py to keep this module
# self-contained (those helpers are private to that test).
# ---------------------------------------------------------------------------
def _populate_runtime_params(params, exp_dir: Path) -> None:
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
    params["save_checkpoint"] = "legacy"
    params["pretrained"] = False
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
    params["log_to_screen"] = True
    params["log_to_wandb"] = False
    params["log_video"] = 0
    params["wandb_dir"] = str(exp_dir)
    params["verbose"] = False
    params["num_data_workers"] = 0
    params["num_visualization_workers"] = 1


def _override_for_smoke(params) -> None:
    params["nettype"] = "plasim_test_recording_dummy"
    params["n_history"] = 0
    params["n_future"] = 0
    params["multistep_count"] = 1
    params["max_epochs"] = 1
    params["batch_size"] = 1
    params["n_train_samples_per_epoch"] = 2
    params["n_eval_samples"] = 2
    params["valid_autoreg_steps"] = 1
    params["scheduler"] = "CosineAnnealingLR"
    params["scheduler_T_max"] = 1
    params["history_normalization_mode"] = "none"
    params["add_zenith"] = False
    params["add_grid"] = False
    params["add_orography"] = False
    params["add_landmask"] = False
    params["add_soiltype"] = False
    params["target"] = "tendency"
    params["normalize_residual"] = False


def _load_yparams(packaged_dataset: Path) -> YParams:
    from makani.utils.parse_dataset_metada import parse_dataset_metadata

    cfg = packaged_dataset / "config" / "plasim_sim52_astro_64x128_zgplev.yaml"
    params = YParams(str(cfg), "plasim_sim52_astro_64x128_zgplev", print_params=False)
    parse_dataset_metadata(params.metadata_json_path, params=params)
    return params


def _build_params(packaged_dataset: Path, exp_dir: Path, *, ema_block):
    params = _load_yparams(packaged_dataset)
    exp_dir.mkdir(exist_ok=True)
    (exp_dir / "training_checkpoints").mkdir(exist_ok=True)
    _populate_runtime_params(params, exp_dir)
    _override_for_smoke(params)
    if ema_block is not None:
        params["ema"] = ema_block
    return params


# ---------------------------------------------------------------------------
# Happy path: EMA on, full ckpt round-trip
# ---------------------------------------------------------------------------
def test_ema_enabled_construction_and_save_round_trip(
    packaged_dataset: Path, tmp_path: Path
):
    """EMA-enabled trainer constructs, train_one_epoch drives the post-step
    hook, save_checkpoint persists the four ema_* keys, validate_one_epoch
    routes EMA scalars to ``valid_logs["metrics"]``."""
    exp_dir = tmp_path / "exp"
    params = _build_params(
        packaged_dataset,
        exp_dir,
        ema_block={
            "enabled": True,
            "decay": 0.9,
            "warmup": True,
            "allow_config_change": False,
        },
    )

    pt = PlasimTrainer(params, world_rank=0, device="cpu")

    # Construction wires the EMA shadow + post-step hook.
    assert pt.ema_enabled is True
    assert isinstance(pt.ema, EMAModel)
    assert pt.ema.step == 0
    assert pt.ema.decay_max == pytest.approx(0.9)
    assert pt.ema.warmup is True
    assert pt.best_ema_loss == float("inf")

    # Drive one epoch — the post-step hook fires once per actual optimizer
    # step (= 2 train samples / batch_size 1).
    _, _, _ = pt.train_one_epoch()
    optimizer_steps = max(state.get("step", 0) for state in pt.optimizer.state.values())
    assert pt.ema.step == optimizer_steps, (
        f"EMA step ({pt.ema.step}) must equal optimizer steps ({optimizer_steps})"
    )
    assert pt.ema.step >= 1

    # validate_one_epoch routes EMA scalars to valid_logs["metrics"] (plan §6).
    _, _, valid_logs = pt.validate_one_epoch(epoch=0, profiler=None)
    metrics = valid_logs.get("metrics", {})
    base = valid_logs.get("base", {})
    for key in ("validation loss ema", "ema decay effective", "ema step", "ema best loss"):
        assert key in metrics, f"§6 routing: {key!r} missing from valid_logs['metrics']"
        assert key not in base, f"§6 routing: {key!r} leaked into valid_logs['base']"
    # The raw "validation loss" key stays under ["base"] (drives stock raw-best save).
    assert "validation loss" in base
    assert "validation loss ema" not in base

    # Save a per-epoch checkpoint and verify the four EMA keys land.
    ckpt_path = params.checkpoint_path.format(mp_rank=0, checkpoint_version=0)
    counters = {"iters": pt.iters, "epoch": pt.epoch}
    pt.save_checkpoint(
        params.checkpoint_path.replace("{checkpoint_version}", "0"),
        pt.model,
        loss=pt.loss_obj,
        optimizer=pt.optimizer,
        scheduler=pt.scheduler,
        counters=counters,
        checkpoint_mode="legacy",
    )

    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "ema_state" in saved
    assert "ema_step" in saved
    assert "ema_config" in saved
    assert "ema_best_loss" in saved
    assert saved["ema_step"] == pt.ema.step
    assert saved["ema_config"] == {
        "decay": 0.9,
        "warmup": True,
        "version": EMAModel.CONFIG_VERSION,
    }
    # ema_state must be loadable back into a fresh EMA seeded from the same model.
    pt.ema.load_state_dict(saved["ema_state"], strict=True)


# ---------------------------------------------------------------------------
# EMA-off path stays inert (regression guard)
# ---------------------------------------------------------------------------
def test_ema_disabled_path_unchanged(packaged_dataset: Path, tmp_path: Path):
    """When EMA is absent / disabled, PlasimTrainer behaves exactly like
    before: no shadow, no extra ckpt keys, no metrics routing."""
    exp_dir = tmp_path / "exp"
    params = _build_params(packaged_dataset, exp_dir, ema_block=None)

    pt = PlasimTrainer(params, world_rank=0, device="cpu")
    assert pt.ema_enabled is False
    assert pt.ema is None

    pt.train_one_epoch()

    ckpt_template = params.checkpoint_path.replace("{checkpoint_version}", "0")
    pt.save_checkpoint(
        ckpt_template,
        pt.model,
        loss=pt.loss_obj,
        optimizer=pt.optimizer,
        scheduler=pt.scheduler,
        counters={"iters": pt.iters, "epoch": pt.epoch},
        checkpoint_mode="legacy",
    )
    saved = torch.load(
        ckpt_template.format(mp_rank=0), map_location="cpu", weights_only=False
    )
    for key in ("ema_state", "ema_step", "ema_config", "ema_best_loss"):
        assert key not in saved, f"{key!r} should not appear when EMA is disabled"

    _, _, valid_logs = pt.validate_one_epoch(epoch=0, profiler=None)
    for key in ("validation loss ema", "ema decay effective", "ema step", "ema best loss"):
        assert key not in valid_logs.get("metrics", {})


# ---------------------------------------------------------------------------
# Hard-error reject: EMA + flexible mode on save
# ---------------------------------------------------------------------------
def test_ema_with_flexible_save_rejected(packaged_dataset: Path, tmp_path: Path):
    exp_dir = tmp_path / "exp"
    params = _build_params(
        packaged_dataset,
        exp_dir,
        ema_block={"enabled": True, "decay": 0.999, "warmup": True},
    )
    params["save_checkpoint"] = "flexible"

    with pytest.raises(NotImplementedError, match="legacy save AND load"):
        PlasimTrainer(params, world_rank=0, device="cpu")


# ---------------------------------------------------------------------------
# Hard-error reject: EMA + flexible mode on load
# ---------------------------------------------------------------------------
def test_ema_with_flexible_load_rejected(packaged_dataset: Path, tmp_path: Path):
    exp_dir = tmp_path / "exp"
    params = _build_params(
        packaged_dataset,
        exp_dir,
        ema_block={"enabled": True, "decay": 0.999, "warmup": True},
    )
    params["load_checkpoint"] = "flexible"

    with pytest.raises(NotImplementedError, match="legacy save AND load"):
        PlasimTrainer(params, world_rank=0, device="cpu")
