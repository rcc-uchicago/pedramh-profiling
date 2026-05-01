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
import types
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

from makani.models import model_registry
from makani.utils.dataloader import init_distributed_io
from makani.utils.dataloaders.data_helpers import get_data_normalization
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
        sampler = (
            DistributedSampler(
                loader_dataset,
                shuffle=(mode == "train"),
                num_replicas=params.data_num_shards,
                rank=params.data_shard_id,
            )
            if (params.data_num_shards > 1)
            else None
        )
        dataloader = DataLoader(
            loader_dataset,
            batch_size=int(params.batch_size),
            num_workers=params.num_data_workers,
            shuffle=((sampler is None) and (mode == "train")),
            sampler=sampler,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )
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
        assert params.get("input_noise") is None, (
            "PlasimTrainer requires input_noise to be unset — "
            "concatenate-mode noise injects extra input channels"
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
