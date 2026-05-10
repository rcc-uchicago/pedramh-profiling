"""PlasimTrainer + monkey-patch installer for the PlaSim → Makani contract.

Wires three pieces into stock Makani **without editing core**:

1. ``_install_plasim_patches()`` — rebinds three module attributes:
     - ``makani.models.model_registry.SingleStepWrapper`` → ``PlasimSingleStepWrapper``
     - ``makani.models.model_registry.MultiStepWrapper``  → ``PlasimMultiStepWrapper``
     - ``makani.utils.training.deterministic_trainer.get_dataloader`` → ``_plasim_get_dataloader``

   These are ``from X import Y`` module-scope bindings in their importers,
   which are mutable Python module attributes. Rebinding takes effect on
   subsequent accesses. The integration test guards against any upstream
   import-form change.

2. ``_plasim_get_dataloader(params, files_pattern, device, mode)`` —
   constructs a :class:`~sfno_training.data.PlasimForcingDataset`,
   wraps it in a torch ``DataLoader`` (with optional ``DistributedSampler``),
   attaches the stock-compat attrs (``lat_lon``, ``get_output_normalization``,
   ``get_input_normalization``) and returns ``(dataloader, dataset, sampler)``.
   Hard-fails on ``mode == "inference"`` — inference is out of scope until
   the follow-up ``src/sfno_inference/`` PR ships.

3. ``PlasimTrainer(Trainer)`` — calls ``_install_plasim_patches()`` BEFORE
   ``super().__init__()``. Overrides ``_set_data_shapes`` to override
   ``params.N_in_channels = 58`` after stock population, with hard
   asserts on every aux-feature flag that would inject extra channels.

See docs/sfno_training_implementation_plan.md §6 for the full spec.
"""

from __future__ import annotations

import logging
import os
import time
import types
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

# Import installs the Python 3.12 timedelta shim before any MultifilesDataset
# instantiation reads /timestamp.
from sfno_training import compat  # noqa: F401  side-effect: shim install

from sfno_training.data import PlasimForcingDataset
from sfno_training.models import PlasimMultiStepWrapper, PlasimSingleStepWrapper
from sfno_training.trainer.ema import EMAModel

from makani.models import model_registry
from makani.utils import comm
from makani.utils.dataloader import init_distributed_io
from makani.utils.dataloaders.data_helpers import get_data_normalization
from makani.utils.driver import Driver
from makani.utils.training import deterministic_trainer
from makani.utils.training.deterministic_trainer import Trainer

logger = logging.getLogger("sfno_training.trainer")


# ---------------------------------------------------------------------------
# Sample limits
# ---------------------------------------------------------------------------
def _sample_limit(params, mode: str) -> Optional[int]:
    if mode == "train":
        value = params.get("n_train_samples_per_epoch", params.get("n_train_samples", None))
    elif mode == "eval":
        value = params.get("n_eval_samples_per_epoch", params.get("n_eval_samples", None))
    else:
        value = None
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


# ---------------------------------------------------------------------------
# Sampler factory (factored out for unit-testability — P1 §"Tests")
# ---------------------------------------------------------------------------
def _make_train_eval_sampler(
    loader_dataset,
    mode: str,
    num_replicas: int,
    rank: int,
) -> Optional[DistributedSampler]:
    """Construct the DistributedSampler for train/eval, or ``None`` for the
    single-rank fast path.

    ``drop_last=True`` is set explicitly so the sampler trims the tail to a
    multiple of ``num_replicas`` instead of padding-by-duplicating. The
    DataLoader's ``drop_last=True`` is then a no-op except on the genuine
    final partial batch — symmetric across both ends.
    """
    if num_replicas <= 1:
        return None
    return DistributedSampler(
        loader_dataset,
        shuffle=(mode == "train"),
        num_replicas=num_replicas,
        rank=rank,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Custom dataloader
# ---------------------------------------------------------------------------
def _plasim_get_dataloader(params, files_pattern, device, mode: str = "train"):
    """Drop-in replacement for ``makani.utils.dataloader.get_dataloader``.

    Mirrors the stock multifiles branch but instantiates
    :class:`PlasimForcingDataset`, slices the 53-channel target stats into
    52-channel input stats, and reads forcing stats from
    ``params.forcing_global_means_path`` / ``forcing_global_stds_path``.

    Inference mode is rejected outright (plan §"Hard gate on full
    emulator rollout"): stock ``Inferencer._inference_indexlist`` has no
    slot for our 6 forcing channels and would silently produce
    physically-wrong predictions.
    """
    init_distributed_io(params)

    assert mode != "inference", (
        "PlaSim inference is out of scope until src/sfno_inference/ ships "
        "(see docs/sfno_training_implementation_plan.md §'Hard gate on full "
        "emulator rollout')."
    )

    bias, scale = get_data_normalization(params)
    forcing_bias = np.load(params.forcing_global_means_path).astype(np.float32)
    forcing_scale = np.load(params.forcing_global_stds_path).astype(np.float32)

    n_future = (
        params.get("valid_autoreg_steps") if (mode == "eval") else params.get("n_future", 0)
    )

    # NB: stock parse_dataset_metadata sets params.in_channels = params.out_channels =
    # list(range(53)) (the full 53-channel target space). PlasimForcingDataset reads
    # /fields_state (52 channels) via in_channels and (state ‖ diagnostic) (53 channels)
    # via out_channels — so we slice in_channels to the first n_state_channels.
    n_state = params.get("n_state_channels", 52)
    n_target = params.get("n_state_channels", 52) + params.get("n_diagnostic_channels", 1)
    in_channels = list(range(n_state))
    out_channels = list(range(n_target))

    dataset = PlasimForcingDataset(
        location=files_pattern,
        dt=params.get("dt"),
        in_channels=in_channels,
        out_channels=out_channels,
        n_history=params.get("n_history", 0),
        n_future=n_future,
        diagnostic_dataset_path=params.get("diagnostic_h5_path", "fields_diagnostic"),
        forcing_dataset_path=params.get("forcing_h5_path", "forcing"),
        n_forcing_channels=params.get("n_forcing_channels", 6),
        forcing_bias=forcing_bias,
        forcing_scale=forcing_scale,
        add_zenith=params.get("add_zenith", False),
        data_grid_type=params.get("data_grid_type", "equiangular"),
        model_grid_type=params.get("model_grid_type", "equiangular"),
        bias=bias,
        scale=scale,
        crop_size=(params.get("crop_size_x", None), params.get("crop_size_y", None)),
        crop_anchor=(params.get("crop_anchor_x", 0), params.get("crop_anchor_y", 0)),
        subsampling_factor=params.get("subsampling_factor", 1),
        return_timestamp=False,
        return_target=True,
        relative_timestamp=True,
        file_suffix=params.get("dataset_file_suffix", "h5"),
        enable_s3=params.get("enable_s3", False),
        io_grid=params.get("io_grid", [1, 1, 1]),
        io_rank=params.get("io_rank", [0, 0, 0]),
    )

    if mode in ("train", "eval"):
        limit = _sample_limit(params, mode)
        loader_dataset = (
            Subset(dataset, range(min(limit, len(dataset)))) if limit is not None else dataset
        )
        sampler = _make_train_eval_sampler(
            loader_dataset,
            mode=mode,
            num_replicas=params.data_num_shards,
            rank=params.data_shard_id,
        )
        # Phase-1 P2 DataLoader knobs (default-on; both kwargs require
        # num_workers > 0, so guard the workerless path).
        loader_kwargs = dict(
            batch_size=int(params.batch_size),
            num_workers=params.num_data_workers,
            shuffle=((sampler is None) and (mode == "train")),
            sampler=sampler,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )
        if params.num_data_workers > 0:
            loader_kwargs["persistent_workers"] = bool(
                params.get("persistent_workers", True)
            )
            loader_kwargs["prefetch_factor"] = int(params.get("prefetch_factor", 4))
        dataloader = DataLoader(loader_dataset, **loader_kwargs)
    else:
        sampler = None
        dataloader = types.SimpleNamespace()

    # Stock-compat attrs (Trainer reads these directly off the dataloader).
    dataloader.lat_lon = dataset.lat_lon
    dataloader.get_output_normalization = dataset.get_output_normalization
    dataloader.get_input_normalization = dataset.get_input_normalization

    return dataloader, dataset, sampler


# ---------------------------------------------------------------------------
# Patch installer
# ---------------------------------------------------------------------------
_PATCHES_INSTALLED: bool = False


def _install_plasim_patches() -> None:
    """Rebind the three Makani module attributes. Idempotent."""
    global _PATCHES_INSTALLED
    if _PATCHES_INSTALLED:
        return

    model_registry.SingleStepWrapper = PlasimSingleStepWrapper
    model_registry.MultiStepWrapper = PlasimMultiStepWrapper
    deterministic_trainer.get_dataloader = _plasim_get_dataloader

    _PATCHES_INSTALLED = True
    logger.info("installed PlaSim trainer patches")


# ---------------------------------------------------------------------------
# Trainer subclass
# ---------------------------------------------------------------------------
class PlasimTrainer(Trainer):
    """Stock :class:`Trainer` with two delta points:

    1. ``__init__`` calls :func:`_install_plasim_patches` BEFORE
       ``super().__init__()`` so the wrapper-class lookup at model build
       time and the dataloader factory call both resolve to PlaSim
       implementations.

    2. ``_set_data_shapes`` overrides ``params.N_in_channels`` to the
       locked 58 (= 52 state + 6 forcing) after the stock population,
       and hard-asserts every aux-feature flag that would otherwise
       inject extra channels — anything else is a config-drift bug.
    """

    EXPECTED_N_IN_CHANNELS: int = 58

    def __init__(
        self,
        params=None,
        world_rank: int = 0,
        device: Optional[str] = None,
    ):
        # Order matters: patches must land BEFORE super().__init__ runs the
        # dataloader factory and constructs the model wrapper.
        _install_plasim_patches()
        super().__init__(params, world_rank, device)

        # ---- EMA setup (plan §7.2(a)) ----
        ema_cfg = self.params.get("ema", {}) or {} if self.params is not None else {}
        self.ema_enabled = bool(ema_cfg.get("enabled", False))
        self.ema: Optional[EMAModel] = None
        self.best_ema_loss: float = float("inf")

        # Validate the EMA-period knob at init time so a misconfig surfaces
        # before the first epoch starts. period == 1 is the default and
        # reproduces pre-P4 behaviour.
        ema_validation_period = int(self.params.get("ema_validation_period", 1))
        assert ema_validation_period >= 1, (
            "ema_validation_period must be >= 1 "
            f"(got {ema_validation_period})"
        )
        self._ema_validation_period: int = ema_validation_period

        if self.ema_enabled:
            # Goal #7: legacy save AND legacy load only. Flexible save lacks
            # gather/scatter for EMA shadows, and flexible load leaves
            # per-rank EMA files unrecoverable for the current MP topology.
            flex_save = self.params.save_checkpoint == "flexible"
            flex_load = self.params.load_checkpoint == "flexible"
            if flex_save or flex_load:
                raise NotImplementedError(
                    "ema.enabled=True is currently scoped to legacy save AND load "
                    f"(got save_checkpoint={self.params.save_checkpoint!r}, "
                    f"load_checkpoint={self.params.load_checkpoint!r}). "
                    "Flexible-mode EMA support requires gather/scatter of EMA "
                    "shadows on save and a flex-aware EMA restore path; deferred "
                    "to a follow-up. Either switch both to 'legacy' or set "
                    "ema.enabled: false."
                )

            self.ema = EMAModel(
                self.model,
                decay=float(ema_cfg.get("decay", 0.999)),
                warmup=bool(ema_cfg.get("warmup", True)),
            )
            # Optimizer post-step hook: fires only on actual optimizer.step()
            # invocations, so GradScaler-skipped batches do not trigger an
            # EMA update (plan §3.3).
            self.optimizer.register_step_post_hook(
                lambda *_args, **_kwargs: self.ema.update(self.model)
            )
            if self.log_to_screen:
                logger.info(
                    "EMA enabled: decay=%s, warmup=%s, shadowed_params=%d",
                    self.ema.decay_max,
                    self.ema.warmup,
                    len(self.ema._shadow),
                )
            # Resume: super().__init__ already loaded model_state and set
            # self.checkpoint_version_current; pull ema_* from the SAME version.
            self._maybe_restore_ema_state()

    def _set_data_shapes(self, params, dataset):
        # Stock population first — sets img_*, N_in_channels=52, etc.
        super()._set_data_shapes(params, dataset)

        # Hard-assert the aux-feature flags off — any of these would inject
        # extra channels via Driver._set_data_shapes (driver.py:178-219) and
        # break the locked 58-channel input contract.
        assert params.n_history == 0, (
            "PlasimTrainer requires n_history == 0 (the 58-channel override "
            f"only holds at history=0); got params.n_history={params.n_history}"
        )
        assert params.history_normalization_mode == "none", (
            "PlasimTrainer requires history_normalization_mode == 'none' "
            "(stock history-normalization would compute stats on the "
            "58-channel post-concat input and try to denormalize a 53-channel "
            f"target with the first 53 input stats); got "
            f"{params.history_normalization_mode!r}"
        )
        assert not params.get("add_zenith", False), (
            "PlasimTrainer requires add_zenith=False — solar insolation is "
            "already in /forcing[rsdt]"
        )
        # input_noise: only concatenate-mode breaks the 58-channel contract
        # (driver.py:194-198 adds n_channels to N_dynamic_channels). perturb-
        # mode adds noise in place via preprocessor._append_channels (line
        # 255-258) and leaves n_noise_chan=0, so the channel-count math
        # stays correct. Used by group-recipe clones that match upstream's
        # epsilon_factor=0.05 input perturbation.
        _input_noise = params.get("input_noise")
        if _input_noise is not None:
            assert _input_noise.get("mode") == "perturb", (
                "PlasimTrainer requires input_noise.mode == 'perturb' when "
                "set — concatenate-mode injects extra input channels and "
                f"breaks the locked 58-channel contract; got mode="
                f"{_input_noise.get('mode')!r}"
            )
        assert not params.get("add_grid", False), (
            "PlasimTrainer requires add_grid=False — would inject 2+ static channels"
        )
        assert not params.get("add_orography", False), (
            "PlasimTrainer requires add_orography=False — orography is already "
            "available via /forcing[sg]"
        )
        assert not params.get("add_landmask", False), (
            "PlasimTrainer requires add_landmask=False — land/sea info is "
            "already in /forcing[lsm]"
        )
        assert not params.get("add_soiltype", False), (
            "PlasimTrainer requires add_soiltype=False — would inject 8 static channels"
        )

        # Override N_in_channels to the locked 58. The model is built right
        # after this in Trainer.__init__ at deterministic_trainer.py:132,
        # so the model_registry partial sees the right inp_chans.
        n_state = params.get("n_state_channels", 52)
        n_forcing = params.get("n_forcing_channels", 6)
        params.N_in_channels = n_state + n_forcing

        if params.N_in_channels != self.EXPECTED_N_IN_CHANNELS:
            logger.warning(
                "PlasimTrainer: N_in_channels=%d (n_state=%d + n_forcing=%d), "
                "expected %d — verify this is intentional",
                params.N_in_channels,
                n_state,
                n_forcing,
                self.EXPECTED_N_IN_CHANNELS,
            )

    # ------------------------------------------------------------------
    # EMA — resume, save, validation, EMA-best emission (plan §7.2)
    # ------------------------------------------------------------------
    def _maybe_restore_ema_state(self) -> None:
        """Restore ``ema_state`` / ``ema_step`` / ``ema_best_loss`` from the
        per-rank legacy checkpoint that ``super().__init__`` just loaded.

        Resume contract (plan §4.4):
        - Skipped when not resuming (fresh-start: live params are the seed).
        - Read at ``checkpoint_version=self.checkpoint_version_current`` —
          v1 hard-coded 0 and would silently pair the latest raw weights
          with a stale shadow after rotation.
        - ``ema_config`` mismatch raises unless ``allow_config_change=True``.
        """
        if not self.params.resuming:
            return
        if self.ema is None:
            return

        path = self.params.checkpoint_path.format(
            checkpoint_version=self.checkpoint_version_current,
            mp_rank=comm.get_rank("model"),
        )
        if not os.path.isfile(path):
            logger.warning(
                "EMA resume: checkpoint file %s not found (version=%d). "
                "Keeping fresh-seeded EMA shadow.",
                path,
                self.checkpoint_version_current,
            )
            return

        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # YParams attribute access is top-level only; nested blocks remain
        # dict-like, so access via dict .get().
        ema_cfg = self.params.get("ema", {}) or {}
        allow_config_change = bool(ema_cfg.get("allow_config_change", False))

        ckpt_ema_cfg = ckpt.get("ema_config")
        current_cfg = {
            "decay": float(self.ema.decay_max),
            "warmup": bool(self.ema.warmup),
            "version": int(EMAModel.CONFIG_VERSION),
        }
        if ckpt_ema_cfg is None:
            logger.warning(
                "EMA resume: checkpoint %s has no ema_config — treating as "
                "pre-EMA checkpoint. Shadow stays freshly seeded; effective "
                "re-warmup from t=0.",
                path,
            )
        else:
            differing = [
                k for k in ("decay", "warmup", "version")
                if ckpt_ema_cfg.get(k) != current_cfg.get(k)
            ]
            if differing:
                msg = (
                    "EMA resume: ema_config mismatch on keys "
                    f"{differing}: checkpoint={ckpt_ema_cfg}, "
                    f"current={current_cfg}"
                )
                if allow_config_change:
                    logger.warning(
                        "%s — accepting because ema.allow_config_change=True. "
                        "Shadow may carry stale dynamics until decay catches up.",
                        msg,
                    )
                else:
                    raise RuntimeError(
                        f"{msg}. Either revert the config change or set "
                        "ema.allow_config_change: true in the YAML."
                    )

        ema_state = ckpt.get("ema_state")
        if ema_state is not None:
            self.ema.load_state_dict(ema_state, strict=True)
        else:
            logger.warning(
                "EMA resume: checkpoint %s missing ema_state — keeping fresh "
                "shadow (effective re-warmup).",
                path,
            )

        ema_step = ckpt.get("ema_step")
        if ema_step is not None:
            self.ema.step = int(ema_step)
        else:
            logger.warning("EMA resume: checkpoint %s missing ema_step.", path)

        ema_best_loss = ckpt.get("ema_best_loss")
        if ema_best_loss is not None:
            self.best_ema_loss = float(ema_best_loss)
        else:
            logger.warning(
                "EMA resume: checkpoint %s missing ema_best_loss — leaving at "
                "+inf (first post-resume epoch may overwrite EMA-best).",
                path,
            )

        if self.log_to_screen:
            logger.info(
                "EMA resume: loaded shadow from %s (step=%d, best_ema_loss=%s).",
                path,
                self.ema.step,
                self.best_ema_loss,
            )

    def save_checkpoint(
        self,
        checkpoint_path,
        model,
        loss=None,
        optimizer=None,
        scheduler=None,
        counters=None,
        checkpoint_mode: str = "legacy",
    ) -> None:
        """Override stock ``Driver.save_checkpoint`` to append EMA keys.

        Defers to ``Driver.save_checkpoint`` first (writes the canonical
        legacy file), then — on data-parallel rank 0 within the current
        MP rank — reopens the file and appends ``ema_state``, ``ema_step``,
        ``ema_config``, ``ema_best_loss``. Read-then-rewrite costs one
        extra ~O(model size) of disk I/O per epoch; cheaper than
        duplicating ``_save_checkpoint_legacy`` internals.
        """
        Driver.save_checkpoint(
            checkpoint_path,
            model,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            counters=counters,
            checkpoint_mode=checkpoint_mode,
        )

        if not self.ema_enabled:
            return

        # Defense in depth — __init__ should have prevented this.
        if checkpoint_mode == "flexible":
            raise NotImplementedError(
                "EMA + flexible-mode save is not supported (see plan §3.5, "
                "goal #7). The __init__ guard should have rejected this."
            )

        if self.data_parallel_rank != 0:
            return

        checkpoint_fname = checkpoint_path.format(mp_rank=comm.get_rank("model"))
        if not os.path.isfile(checkpoint_fname):
            logger.warning(
                "EMA save: stock checkpoint %s not found after Driver.save_checkpoint; "
                "skipping ema_* append.",
                checkpoint_fname,
            )
            return

        # Explicit weights_only=False matches stock's load semantics
        # (driver.py:388, 453) and silences PyTorch ≥ 2.4's default-flip warning.
        ckpt = torch.load(checkpoint_fname, map_location="cpu", weights_only=False)
        ckpt["ema_state"] = self.ema.state_dict()
        ckpt["ema_step"] = int(self.ema.step)
        ckpt["ema_config"] = {
            "decay": float(self.ema.decay_max),
            "warmup": bool(self.ema.warmup),
            "version": int(EMAModel.CONFIG_VERSION),
        }
        ckpt["ema_best_loss"] = float(self.best_ema_loss)
        torch.save(ckpt, checkpoint_fname)

    def _should_run_ema_validation(self, epoch: int) -> bool:
        """Decide whether the EMA validation pass runs this epoch.

        Always runs when ``ema_validation_period == 1`` (default;
        behaviour-equivalent to pre-P4). With period K > 1, runs on
        epochs satisfying ``epoch % K == 0`` and unconditionally on the
        final epoch (``max_epochs - 1``) so the post-training EMA-best
        checkpoint never lags by more than K-1 epochs.
        """
        if not self.ema_enabled:
            return False
        period = self._ema_validation_period
        if period == 1:
            return True
        if (epoch % period) == 0:
            return True
        max_epochs = int(self.params.max_epochs)
        return epoch == (max_epochs - 1)

    def validate_one_epoch(self, epoch, profiler=None):
        """Two-pass validation: raw weights + EMA weights.

        Pass 2 runs on **all ranks** — ``validate_one_epoch`` contains
        ``dist.barrier`` and metric all-reduces; subset-rank execution
        would deadlock. Visualization is suppressed on the EMA pass by
        stashing ``params.log_video=0`` (restored in ``finally``).

        With ``ema_validation_period > 1`` the EMA pass is skipped on
        non-EMA epochs: the raw validation tuple is returned unchanged,
        ``params.log_video`` is not touched, the best-EMA checkpoint
        path is bypassed, and the four EMA metric keys
        (``validation loss ema``, ``ema decay effective``, ``ema step``,
        ``ema best loss``) are absent from ``valid_logs["metrics"]`` for
        that epoch. Makani's screen logger and wandb upload at
        ``deterministic_trainer.py:709,731`` iterate the metrics dict
        dynamically, so missing keys are tolerated downstream.
        """
        raw = super().validate_one_epoch(epoch, profiler=profiler)
        valid_time, viz_time, valid_logs = raw

        if not self._should_run_ema_validation(epoch):
            return valid_time, viz_time, valid_logs

        ema_t0 = time.perf_counter()
        saved_log_video = self.params.get("log_video", 0)
        try:
            self.params["log_video"] = 0  # falsifies visualize_data predicate
            with self.ema.applied_to(self.model):
                _, _, valid_logs_ema = super().validate_one_epoch(epoch, profiler=None)
        finally:
            self.params["log_video"] = saved_log_video
        ema_t = time.perf_counter() - ema_t0

        # Routing per plan §6: EMA scalars go in valid_logs["metrics"] so they
        # print on screen even when log_to_wandb=False. ["base"]["validation
        # loss"] stays raw (drives the stock raw-best save flow at
        # deterministic_trainer.py:404-406 — must not be overwritten).
        ema_loss = valid_logs_ema["base"]["validation loss"]
        metrics = valid_logs.setdefault("metrics", {})
        metrics["validation loss ema"] = ema_loss
        for k, v in valid_logs_ema.get("metrics", {}).items():
            metrics[f"{k} ema"] = v

        # Rank-gated EMA-best write — file write only on data_parallel_rank 0,
        # but EMA validation ran on all ranks above.
        if (self.data_parallel_rank == 0) and (ema_loss < self.best_ema_loss):
            self._save_best_ema_checkpoint(epoch, float(ema_loss))
            self.best_ema_loss = float(ema_loss)

        # Populate the remaining §6 metrics AFTER the EMA-best update so that
        # "ema best loss" reflects the just-written value when an improvement
        # happened this epoch.
        metrics["ema decay effective"] = float(self.ema.decay_t)
        metrics["ema step"] = int(self.ema.step)
        metrics["ema best loss"] = float(self.best_ema_loss)

        return valid_time + ema_t, viz_time, valid_logs

    def log_epoch(self, train_logs, valid_logs, timing_logs):
        """Inject ``samples/sec`` into both timing/train logs before deferring.

        ``samples/sec = per_rank_batch * world_size / (training step time / 1000)``.
        Writing the key into ``timing_logs`` makes Makani's screen logger emit
        it — its loop iterates ``timing_logs.keys()`` dynamically at
        ``deterministic_trainer.py:712-713``. Writing the same key into
        ``train_logs`` puts it on wandb via Makani's
        ``wandb.log(train_logs, step=self.epoch)`` at ``:733`` (which fires
        before the ``commit=True`` at ``:738``), so no extra ``wandb.log``
        call is needed from this override. See
        docs/2026-05-05_ddp_throughput_fix_plan.md §I0.

        Backfill keys: when ``--skip_validation`` is set, the upstream
        training loop (deterministic_trainer.py:374-376) initializes
        ``valid_logs = {"base": {}, "metrics": {}}``, but
        ``log_epoch`` reads ``valid_logs["base"]["validation steps"]``
        (line 709) and ``["validation loss"]`` (line 724) unconditionally
        on the rank-0 screen path. Without backfill, rank 0 raises
        ``KeyError`` mid-print and the other ranks block on the next
        AllReduce until NCCL times out (10 min). Insert sentinel values
        so the upstream printer remains a pure formatting path.
        """
        step_time_ms = float(timing_logs.get("training step time [ms]", 0.0) or 0.0)
        if step_time_ms > 0.0:
            per_rank_bs = int(self.params.batch_size)
            world_size = int(self.params.get("world_size", comm.get_world_size()))
            samples_per_sec = per_rank_bs * world_size / (step_time_ms / 1000.0)
        else:
            samples_per_sec = 0.0
        timing_logs["samples/sec"] = samples_per_sec
        train_logs["samples/sec"] = samples_per_sec
        base = valid_logs.setdefault("base", {})
        base.setdefault("validation steps", 0)
        base.setdefault("validation loss", float("nan"))
        valid_logs.setdefault("metrics", {})
        return super().log_epoch(train_logs, valid_logs, timing_logs)

    def _save_best_ema_checkpoint(self, epoch: int, ema_loss: float) -> None:
        """Emit ``best_ckpt_ema_mp{mp_rank}.tar`` (plan §7.2(e)).

        ``model_state`` is the FULL canonical state_dict with EMA values
        substituted only where we have a shadow — so
        ``Driver.restore_from_checkpoint(..., strict=True)`` succeeds at
        inference time without an inference-side change.
        """
        # Two-step path derivation:
        #   1. Insert _ema before _mp{mp_rank} on the raw template.
        #   2. Format with mp_rank=comm.get_rank("model").
        raw_template = self.params.best_checkpoint_path  # ".../best_ckpt_mp{mp_rank}.tar"
        ema_template = raw_template.replace("_mp{mp_rank}", "_ema_mp{mp_rank}")
        path = ema_template.format(mp_rank=comm.get_rank("model"))

        store_dict: "OrderedDict" = OrderedDict()
        store_dict["model_state"] = self.ema.export_model_state(self.model)

        if self.loss_obj is not None:
            store_dict["loss_state_dict"] = self.loss_obj.state_dict()

        comm_names = comm.get_model_comm_names()
        comm_dict: "OrderedDict" = OrderedDict()
        for cname in comm_names:
            comm_dict[cname] = {
                "size": comm.get_size(cname),
                "rank": comm.get_rank(cname),
            }
        store_dict["comm_grid"] = comm_dict

        store_dict["iters"] = int(self.iters)
        store_dict["epoch"] = int(self.epoch)

        with torch.no_grad():
            torch.save(store_dict, path)

        if self.log_to_screen:
            logger.info(
                "saved EMA-best checkpoint @ epoch %d to %s, val/loss_ema=%.4e",
                epoch,
                path,
                ema_loss,
            )
