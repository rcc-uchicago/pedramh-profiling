"""Lightning ``LightningModule`` wrapping S2S's PanguModel_Plasim + losses.

This module ports the model/loss/optimisation half of
``v2.0/train.py::Trainer`` onto a :class:`lightning.LightningModule`, mirroring
the SNFO template at ``$SNFO_DIR/modules/train_module.py``. It **reuses** the
existing S2S architecture and loss functions rather than reimplementing them:

* :class:`networks.pangu.PanguModel_Plasim` is held as ``self.model`` — the one
  S2S-specific component that distinguishes this codebase from SNFO.
* The latitude-weighted CRPS / MSE / L1 losses, their masked variants, and the
  VAE KL term from :mod:`utils.losses` are instantiated in
  :meth:`TrainModule._setup_loss_fun` (a faithful port of
  ``v2.0/train.py::Trainer.setup_loss_fun``) and called from
  :meth:`TrainModule.training_step`.

The per-step body of :meth:`TrainModule.training_step` is a faithful port of
``v2.0/train.py::Trainer.cal_loss`` plus the input-preparation and ensemble
handling from ``Trainer._prepare_inputs_batch``. Two behaviours that the manual
S2S loop owned are delegated to Lightning here:

* **AMP / precision.** ``cal_loss`` ran the forward+loss under an explicit
  ``torch.amp.autocast`` and the loop drove a ``GradScaler``. Under Lightning
  automatic optimisation the Trainer owns both, selected via its ``precision=``
  setting (``"16-mixed"`` reproduces S2S's fp16 default, ``"bf16-mixed"`` the
  bf16 path). No hand-rolled autocast / scaler lives here.
* **Scheduler stepping.** ``setup_scheduler`` built the LR scheduler and the
  loop called ``scheduler.step()``; here :meth:`configure_optimizers` returns
  ``[optimizer], [scheduler]`` and Lightning steps it.

The DDP-safety **dead-module freeze** (``layer_perturbation2``,
``layer_purturbation_e2`` — modules defined in :mod:`networks.pangu` with no
forward call) is reproduced in :meth:`__init__` so that the
``DDPStrategy(find_unused_parameters=False, static_graph=True)`` wired in the
Phase-3 entry point stays safe.

The ``networks.*`` / ``utils.*`` imports resolve only when ``v2.0/`` is on
``PYTHONPATH`` (``PYTHONPATH=v2.0/``), matching the rest of the ported tree.

See Also:
    networks.pangu.PanguModel_Plasim: The reused architecture, held as
        ``self.model``; returns the 7-tuple consumed by
        :meth:`TrainModule.training_step`.
    utils.losses.Latitude_weighted_CRPSLoss: The primary ``weightedCRPS`` loss.
    utils.losses.Kl_divergence_gaussians: The VAE KL-divergence term.
    data.datamodule.ClimateDataModule: Supplies the batches and the
        ``normalizer`` (its ``train_dataset``) consumed here.
"""

import os
from datetime import timedelta

import cf_xarray as cfxr  # noqa: F401  (registers the .cf accessor used in save_predictions)
import lightning as L
import numpy as np
import torch
import torch.cuda.nvtx as nvtx
import xarray as xr

from networks.pangu import PanguModel_Plasim
from utils.losses import (
    Latitude_weighted_MSELoss,
    Latitude_weighted_L1Loss,
    Masked_L1Loss,
    Masked_MSELoss,
    Latitude_weighted_masked_L1Loss,
    Latitude_weighted_masked_MSELoss,
    Latitude_weighted_CRPSLoss,
    Kl_divergence_gaussians,
)

# Mirrors S2S_NVTX in v2.0/train.py and the module-level _NVTX in
# networks.pangu — set the same env var to activate the in-step ranges here.
_NVTX = os.environ.get("S2S_NVTX") == "1"


def to_ensemble_batch(data: torch.Tensor, ens_members: int) -> torch.Tensor:
    """Tile a batch of ``M`` samples into ``M * ens_members`` samples.

    Faithful port of ``v2.0/train.py::to_ensemble_batch``. Each sample is
    repeated ``ens_members`` times along a new dimension and flattened back into
    the batch dimension, so the CRPS loss (which reshapes ``B * ens_members``
    back to ``(B, ens_members, ...)``) sees a contiguous ensemble per sample.

    The NVTX ``to_ensemble_batch`` range is emitted only when ``S2S_NVTX=1``
    (the original always pushed it; gating it keeps a no-op fast path during
    normal training, consistent with the rest of this module).

    Args:
        data: Input tensor of shape ``(M, ...)``.
        ens_members: Number of ensemble members to tile per sample.

    Returns:
        torch.Tensor: Tensor of shape ``(M * ens_members, ...)``.
    """
    if _NVTX:
        nvtx.range_push("to_ensemble_batch")
    data = data.unsqueeze(1).expand(-1, ens_members, *data.shape[1:]).reshape(
        -1, *data.shape[1:]
    )
    if _NVTX:
        nvtx.range_pop()  # to_ensemble_batch
    return data


class TrainModule(L.LightningModule):
    """Train :class:`networks.pangu.PanguModel_Plasim` under Lightning.

    Wraps the S2S Pangu-PLASIM architecture and its loss functions into a
    :class:`lightning.LightningModule`. The model is held as ``self.model`` and
    the losses (selected by ``params.loss``, plus the optional VAE KL term) are
    instantiated in :meth:`_setup_loss_fun`. :meth:`training_step` ports
    ``v2.0/train.py::Trainer.cal_loss``: it runs the model (returning the
    7-tuple ``(output_surface, output_upper_air, output_diagnostic, mu, sigma,
    mu2, sigma2)`` -- where ``mu2``/``sigma2`` are the second-encoder
    ``mu_e2``/``sigma_e2`` of :class:`networks.pangu.PanguModel_Plasim`, consumed
    positionally), combines the surface / upper-air / diagnostic losses with the
    original weights, and adds the VAE term when enabled.

    The constant-boundary tensor and the CRPS latitudes are registered as
    buffers so Lightning places them on the right device — no manual
    ``.to(self.device)`` / ``.cuda()`` is performed in :meth:`__init__`.

    Args:
        params: S2S parameter object (attribute- and item-accessible, e.g. a
            :class:`utils.YParams.YParams` instance) carrying the model, loss,
            optimiser, scheduler and variable configuration. The keys consumed
            here mirror those read by ``v2.0/train.py::Trainer``
            (``loss``, ``vae_loss``, ``vae_loss_weight``, ``num_ensemble_members``,
            ``optimizer_type``, ``scheduler``, ``lr``, ``weight_decay``, ``lat``,
            ``mask_output``, ``predict_delta``, ``max_epochs``, the variable
            groups, ...). ``has_diagnostic`` and ``num_ensemble_members`` are
            derived defensively here when absent (matching ``v2.0/train.py``'s
            ``__main__``), so the module is usable without the full entry-point
            wiring (which lands in Phase 3).
        normalizer (optional): The training dataset, doubling as the
            normalization / statistics source (see
            :class:`data.datamodule.ClimateDataModule`, watch-point (b)). It
            supplies ``constant_boundary_data``, ``land_mask`` and ``mask_fill``.
            Required for full operation; only optional so the class can be
            imported/introspected without a dataset.

    Attributes:
        model (PanguModel_Plasim): The reused S2S architecture.
        ddp (bool): Whether the configured strategy is a DDP variant; controls
            ``sync_dist`` on logging calls.
        has_diagnostic (bool): Whether diagnostic variables are present (drives
            the diagnostic-loss term, the batch unpacking, and the eval-forward
            tuple arity in :meth:`predict`).
        num_ensemble_members (int): Ensemble size for the CRPS loss; ``> 1``
            activates :func:`to_ensemble_batch`.
        inference_steps (int): Full autoregressive horizon for the netCDF save
            path in :meth:`save_predictions` (derived as
            ``max(forecast_lead_times)`` like ``v2.0/inference.py::__main__``).
        predictions_dir (str | None): Directory the validation netCDF
            predictions are written to (set by the entry point ``val.py`` via the
            ``predictions_dir`` param key); ``None`` disables saving.
        do_save_predictions (bool): Whether :meth:`validation_step` writes
            netCDF predictions on rank 0 / batch 0 (requires both
            ``params.save_predictions`` and a ``predictions_dir``).

    See Also:
        networks.pangu.PanguModel_Plasim: The wrapped architecture.
        data.datamodule.ClimateDataModule: Provides batches and the normalizer.
        val.py: The Lightning validation/inference entry point that drives
            ``trainer.validate`` against this module.
    """

    def __init__(self, params, normalizer=None) -> None:
        """Build the model, freeze dead modules, and instantiate the losses.

        Args:
            params: S2S parameter object (see the class docstring).
            normalizer (optional): The training dataset / statistics source (see
                the class docstring).
        """
        super().__init__()
        self.params = params
        self.n = normalizer

        # has_diagnostic / num_ensemble_members are normally injected by
        # v2.0/train.py's __main__ (lines 1825-1835). Derive them defensively
        # here so the module works whether or not the (Phase-3) entry point
        # has set them.
        if "has_diagnostic" in params:
            self.has_diagnostic = bool(params.has_diagnostic)
        else:
            self.has_diagnostic = (
                hasattr(params, "diagnostic_variables")
                and len(params.diagnostic_variables) > 0
            )
            params["has_diagnostic"] = self.has_diagnostic
        if "num_ensemble_members" not in params:
            params["num_ensemble_members"] = 1
        self.num_ensemble_members = int(params.num_ensemble_members)

        # mask_output gating, mirroring Trainer.check_land_ocean_variables.
        self.mask_output = bool(getattr(params, "mask_output", False))
        self.has_land = (
            hasattr(params, "land_variables") and len(params.land_variables) > 0
        )
        self.has_ocean = (
            hasattr(params, "ocean_variables") and len(params.ocean_variables) > 0
        )

        # CRPS latitudes (registered as a buffer so Lightning places it). The
        # masked-loss / CRPS objects reference it via self.latitudes below.
        latitudes = torch.from_numpy(np.array(params.lat)).to(torch.float32)
        self.register_buffer("latitudes", latitudes, persistent=False)

        # --- model (reused; NOT relocated — Phase 5 owns physical relocation) ---
        self.model = self._get_model()

        # --- losses (faithful port of Trainer.setup_loss_fun) ---
        self.loss_obj_pl, self.loss_obj_sfc, self.loss_obj_diagnostic, self.loss_vae = (
            self._setup_loss_fun()
        )

        # --- constant-boundary buffer (built from the normalizer's dataset) ---
        # Trainer.get_dataset builds this as
        #   train_dataset.constant_boundary_data.unsqueeze(0) * ones(B,1,1,1)
        # and (when ensembling) tiles it. We register the *unbatched* (c,h,w)
        # tensor and expand/tile it per-step in training_step, so the buffer is
        # batch-size-agnostic and Lightning still owns its device placement.
        if self.n is not None:
            self.register_buffer(
                "constant_boundary_data",
                self.n.constant_boundary_data.to(torch.float32),
                persistent=False,
            )
        else:
            self.constant_boundary_data = None

        # SNFO reads ``config['training']['strategy']`` to set ``self.ddp``. S2S
        # configs do not carry a Lightning-style ``training.strategy`` block yet
        # (that convergence is Phase 3/5), so honour a ``strategy`` key if one is
        # present, else default to the DDP-safe ``True`` used by v2.0/train.py
        # (which always ran under DistributedDataParallel). The Phase-3 entry
        # point can override via the ``_lightning_ddp`` param key.
        if "_lightning_ddp" in params:
            self.ddp = bool(params["_lightning_ddp"])
        elif "strategy" in params:
            self.ddp = params["strategy"] in (
                "ddp", "ddp_find_unused_parameters_true",
            )
        else:
            self.ddp = True

        # --- inference / prediction-saving config (Phase 4) ---
        # inference_steps is the full autoregressive horizon for the netCDF save
        # path. The S2S config does not carry it (it is derived in
        # v2.0/inference.py::__main__): with forecast_lead_times present it is
        # max(forecast_lead_times), else (24*15)//timedelta_hours. We derive it
        # the same way so save_predictions rolls out the full horizon the source
        # saved, independent of the (shorter) per-lead-time scoring in predict().
        if hasattr(params, "forecast_lead_times"):
            self.inference_steps = int(max(params.forecast_lead_times))
        else:
            self.inference_steps = (24 * 15) // int(params.timedelta_hours)
        # Where validation_step writes netCDF predictions (rank0/batch0). The
        # entry point (val.py) sets a writable run dir via params['predictions_dir'];
        # when absent, save_predictions is a no-op (so the Phase-2 training smoke,
        # which has no predictions dir, never tries to write).
        self.predictions_dir = (
            params["predictions_dir"] if "predictions_dir" in params else None
        )
        # save_predictions toggle: only saves when explicitly enabled AND a
        # predictions_dir is set (mirrors SNFO gating save off a config flag).
        self.do_save_predictions = bool(
            getattr(params, "save_predictions", False) and self.predictions_dir
        )

        # The normalizer holds open HDF5 handles and is not picklable; exclude it
        # from the hparams snapshot (SNFO ignores its normalizer the same way).
        self.save_hyperparameters(ignore=["normalizer"])

    def _move_losses_to_device(self) -> None:
        """Point the loss objects' device-resident tensors at this device.

        The :mod:`utils.losses` objects store ``latitudes`` (and any ``mask``) as
        **plain attributes**, not registered buffers, so ``LightningModule.to``
        does not migrate them when Lightning moves the module to the GPU. In the
        original ``v2.0/train.py::Trainer.setup_loss_fun`` this never surfaced
        because the latitudes were built with ``.to(self.device)`` already on the
        GPU. Here we re-point each loss's ``latitudes`` at the module's
        ``self.latitudes`` buffer (which Lightning has already moved) and move any
        non-``None`` ``mask`` to the module device, so the in-loss
        ``weight * (pred - target)`` math sees same-device tensors.

        Idempotent and safe to call from :meth:`setup`, :meth:`on_fit_start`,
        and :meth:`on_validation_start` (the last is required for the
        ``trainer.validate`` / inference path, where ``on_fit_start`` never
        fires and ``setup`` runs before the module is moved to the GPU).
        """
        for loss_obj in (self.loss_obj_pl, self.loss_obj_sfc, self.loss_obj_diagnostic):
            if hasattr(loss_obj, "latitudes") and isinstance(
                getattr(loss_obj, "latitudes"), torch.Tensor
            ):
                loss_obj.latitudes = self.latitudes
            mask = getattr(loss_obj, "mask", None)
            if isinstance(mask, torch.Tensor):
                loss_obj.mask = mask.to(self.device)

    def setup(self, stage: str) -> None:
        """Migrate loss-object tensors onto the module device for every stage.

        Args:
            stage: The Lightning stage (``"fit"``, ``"validate"``, ``"test"``, or
                ``"predict"``). Used only to scope the call; the migration is the
                same for all stages.
        """
        self._move_losses_to_device()

    def on_fit_start(self) -> None:
        """Re-assert loss-tensor device placement at the start of ``fit``.

        Belt-and-suspenders alongside :meth:`setup`: by ``on_fit_start`` the
        module (and its buffers) are guaranteed to be on the training device, so
        re-pointing the loss tensors here is exact even if a strategy moved the
        module after :meth:`setup`.
        """
        self._move_losses_to_device()

    def on_validation_start(self) -> None:
        """Re-assert loss-tensor device placement at the start of validation.

        Required for the ``trainer.validate`` / inference path (:mod:`val.py`):
        :meth:`on_fit_start` does not fire there, and :meth:`setup` runs *before*
        the module is moved to the GPU, so its re-pointing of ``loss.latitudes``
        would capture the still-CPU buffer. ``on_validation_start`` fires after
        the module (and its ``latitudes`` buffer) are on the validation device,
        so the in-loss ``weight * (pred - target)`` math in
        :class:`utils.losses.Latitude_weighted_CRPSLoss` sees same-device
        tensors. Also fires during ``fit``'s validation phase (a harmless
        idempotent re-assert).
        """
        self._move_losses_to_device()

    def _get_model(self) -> PanguModel_Plasim:
        """Build PanguModel_Plasim and freeze its dead modules.

        Faithful port of the model-construction portion of
        ``v2.0/train.py::Trainer.get_model`` for the ``pangu_plasim`` /
        non-``predict_delta`` path (the only path exercised by ``test.yaml``).
        The ``mask_fill`` source mirrors ``get_model``: ``params.mask_fill`` when
        present, else the normalizer dataset's ``mask_fill``. The ``land_mask``
        is left ``None`` here (matching ``get_land_mask_bool`` when
        ``mask_output`` is False); the masked-output land-mask path is a Phase-5
        concern and is not reached by the smoke config.

        The two dead modules (``layer_perturbation2``, defined at
        ``networks/pangu.py`` line 363 with its forward call commented out, and
        ``layer_purturbation_e2``, defined at line 408 and never called) are
        frozen so the Phase-3
        ``DDPStrategy(find_unused_parameters=False, static_graph=True)`` is safe.

        Returns:
            PanguModel_Plasim: The constructed, dead-module-frozen model.

        Raises:
            NotImplementedError: If ``params.nettype`` is not ``"pangu_plasim"``.
        """
        params = self.params
        if getattr(params, "nettype", None) != "pangu_plasim":
            raise NotImplementedError(
                f"nettype {getattr(params, 'nettype', None)!r} not implemented "
                "(only 'pangu_plasim' is ported)"
            )

        land_mask = self.n.land_mask if (self.n is not None and self.mask_output) else None
        if hasattr(params, "mask_fill"):
            mask_fill = params.mask_fill
        elif self.n is not None:
            mask_fill = self.n.mask_fill
        else:
            mask_fill = None

        model = PanguModel_Plasim(params, land_mask=land_mask, mask_fill=mask_fill)

        # Freeze dead-code modules so DDP static_graph=True is safe.
        # NOTE: torch.compile (TORCH_COMPILE_MODE) and the DDP wrap that
        # v2.0/train.py::get_model applies here are NOT done in the module —
        # under Lightning the strategy owns the DDP wrap and bench.py owns
        # torch.compile (Phase 3). The freeze MUST live with the model, so it
        # stays here.
        _dead_modules = {"layer_perturbation2", "layer_purturbation_e2"}
        for mod_name, mod in model.named_modules():
            if mod_name in _dead_modules:
                mod.requires_grad_(False)
        return model

    def _setup_loss_fun(self):
        """Instantiate the loss objects exactly as ``Trainer.setup_loss_fun``.

        Faithful port of ``v2.0/train.py::Trainer.setup_loss_fun``. Selects the
        plevel / surface / diagnostic loss objects from ``params.loss`` (one of
        ``l1`` / ``l2`` / ``weightedl1`` / ``weightedl2`` / ``weightedCRPS``) and
        builds the optional VAE KL term when ``params.vae_loss`` is set. The
        masked surface variants are used only when land/ocean variables are
        present *and* (for the MSE/L1 families) ``mask_output`` is set — matching
        the source's branching. The land/ocean mask tensor (``mask_bool``) is
        built here only for those masked families; ``test.yaml`` uses
        ``weightedCRPS`` with ``mask_output=False``, so the unmasked CRPS path is
        taken and no mask is constructed.

        Returns:
            tuple: ``(loss_obj_pl, loss_obj_sfc, loss_obj_diagnostic,
            loss_vae)`` where each loss object is a callable
            ``(pred, target) -> scalar`` (the KL term is callable
            ``(mu, logvar, mu2, logvar2) -> scalar``), and unused slots are
            ``0`` exactly as in the source.

        Raises:
            NotImplementedError: If ``params.loss`` is not one of the supported
                values.
        """
        params = self.params
        loss_obj_pl = 0
        loss_obj_sfc = 0
        loss_obj_diagnostic = 0
        loss_vae = 0

        if getattr(params, "vae_loss", False):
            loss_vae = Kl_divergence_gaussians()

        # Build the land/ocean mask only for the masked MSE/L1 families. For
        # weightedCRPS the mask is passed straight to the CRPS object (and is
        # None when mask_output is False / no land-ocean vars), so we resolve a
        # mask tensor lazily below where needed.
        mask_bool = self._build_mask_bool()
        lat = self.latitudes

        if params.loss == "l1":
            loss_obj_pl = torch.nn.L1Loss()
            if (self.has_land or self.has_ocean) and self.mask_output:
                loss_obj_sfc = Masked_L1Loss(mask_bool)
            else:
                loss_obj_sfc = torch.nn.L1Loss()
            if self.has_diagnostic:
                loss_obj_diagnostic = torch.nn.L1Loss()
        elif params.loss == "l2":
            loss_obj_pl = torch.nn.MSELoss()
            if (self.has_land or self.has_ocean) and self.mask_output:
                loss_obj_sfc = Masked_MSELoss(mask_bool)
            else:
                loss_obj_sfc = torch.nn.MSELoss()
            if self.has_diagnostic:
                loss_obj_diagnostic = torch.nn.MSELoss()
        elif params.loss == "weightedl1":
            loss_obj_pl = Latitude_weighted_L1Loss(lat)
            if (self.has_land or self.has_ocean) and self.mask_output:
                loss_obj_sfc = Latitude_weighted_masked_L1Loss(lat, mask_bool)
            else:
                loss_obj_sfc = Latitude_weighted_L1Loss(lat)
            if self.has_diagnostic:
                loss_obj_diagnostic = Latitude_weighted_L1Loss(lat)
        elif params.loss == "weightedl2":
            loss_obj_pl = Latitude_weighted_MSELoss(lat)
            if (self.has_land or self.has_ocean) and self.mask_output:
                loss_obj_sfc = Latitude_weighted_masked_MSELoss(lat, mask_bool)
            else:
                loss_obj_sfc = Latitude_weighted_MSELoss(lat)
            if self.has_diagnostic:
                loss_obj_diagnostic = Latitude_weighted_MSELoss(lat)
        elif params.loss == "weightedCRPS":
            n_ens = params.num_ensemble_members
            loss_obj_pl = Latitude_weighted_CRPSLoss(lat, n_ens)
            if self.has_land or self.has_ocean:
                loss_obj_sfc = Latitude_weighted_CRPSLoss(lat, n_ens, mask_bool)
            else:
                loss_obj_sfc = Latitude_weighted_CRPSLoss(lat, n_ens)
            if self.has_diagnostic:
                loss_obj_diagnostic = Latitude_weighted_CRPSLoss(lat, n_ens)
        else:
            raise NotImplementedError(f"loss {params.loss!r} not implemented")

        return loss_obj_pl, loss_obj_sfc, loss_obj_diagnostic, loss_vae

    def _build_mask_bool(self):
        """Build the per-surface-variable land/ocean boolean mask, or ``None``.

        Faithful port of ``v2.0/train.py::Trainer.get_land_mask_bool`` for the
        ``pangu_plasim`` net. Returns a stacked boolean mask (one ``(h, w)``
        plane per surface variable: the land mask for land variables, its
        complement for ocean variables, all-ones otherwise) when land/ocean
        variables are present and ``mask_output`` is set; otherwise ``None``.
        For ``test.yaml`` (``mask_output=False``) this returns ``None``.

        Returns:
            torch.Tensor | None: The stacked boolean mask, or ``None`` when no
            mask is needed (or no normalizer is available to source the land
            mask from).
        """
        if not ((self.has_land or self.has_ocean) and self.mask_output):
            return None
        if self.n is None:
            return None
        params = self.params
        land_mask = torch.clone(self.n.land_mask.detach())
        mask_bool = []
        for var in params.surface_variables:
            if var in params.land_variables:
                mask_bool.append(torch.clone(land_mask).to(torch.bool))
            elif var in params.ocean_variables:
                mask_bool.append(torch.logical_not(torch.clone(land_mask).to(torch.bool)))
            else:
                mask_bool.append(torch.ones(land_mask.shape, dtype=torch.bool))
        return torch.stack(mask_bool)

    def _prepare_inputs(self, data):
        """Unpack a training batch, tile for the ensemble, set channels-last.

        Faithful port of ``v2.0/train.py::Trainer._prepare_inputs_batch`` minus
        the manual ``.to(self.device)`` (Lightning has already moved the batch).
        The training batch ordering is the one returned by
        :meth:`utils.data_loader_multifiles.GetDataset.__getitem__` (train
        branch, line 594 when diagnostics are present):
        ``(input_surface, input_upper_air, target_surface, target_upper_air,
        target_diagnostic, varying_boundary_data)``. When ``has_diagnostic`` is
        False the batch is the 5-element variant (line 596) and
        ``target_diagnostic`` is left as ``0``.

        When ``num_ensemble_members > 1`` every tensor is tiled via
        :func:`to_ensemble_batch`. Tensors are then converted to channels-last
        (``channels_last`` for 4-D surface/boundary, ``channels_last_3d`` for 5-D
        upper-air) to keep cuDNN in NHWC, exactly as the source does.

        Args:
            data: The batch list from the training dataloader (5 or 6 tensors).

        Returns:
            tuple: ``(input_surface, input_upper_air, target_surface,
            target_upper_air, target_diagnostic, varying_boundary_data)``.
        """
        input_surface = 0
        input_upper_air = 0
        target_surface = 0
        target_upper_air = 0
        target_diagnostic = 0
        varying_boundary_data = 0
        if self.has_diagnostic:
            (
                input_surface,
                input_upper_air,
                target_surface,
                target_upper_air,
                target_diagnostic,
                varying_boundary_data,
            ) = data
        else:
            (
                input_surface,
                input_upper_air,
                target_surface,
                target_upper_air,
                varying_boundary_data,
            ) = data

        if self.num_ensemble_members > 1:
            tiled = [
                to_ensemble_batch(t, self.num_ensemble_members)
                for t in (
                    input_surface,
                    input_upper_air,
                    target_surface,
                    target_upper_air,
                    target_diagnostic,
                    varying_boundary_data,
                )
            ]
            (
                input_surface,
                input_upper_air,
                target_surface,
                target_upper_air,
                target_diagnostic,
                varying_boundary_data,
            ) = tiled

        cl, cl3 = torch.channels_last, torch.channels_last_3d
        if isinstance(input_surface, torch.Tensor):
            input_surface = input_surface.to(memory_format=cl)
        if isinstance(input_upper_air, torch.Tensor):
            input_upper_air = input_upper_air.to(memory_format=cl3)
        if isinstance(target_surface, torch.Tensor):
            target_surface = target_surface.to(memory_format=cl)
        if isinstance(target_upper_air, torch.Tensor):
            target_upper_air = target_upper_air.to(memory_format=cl3)
        if isinstance(target_diagnostic, torch.Tensor):
            target_diagnostic = target_diagnostic.to(memory_format=cl)
        if isinstance(varying_boundary_data, torch.Tensor):
            varying_boundary_data = varying_boundary_data.to(
                memory_format=cl if varying_boundary_data.dim() == 4 else cl3
            )

        return (
            input_surface,
            input_upper_air,
            target_surface,
            target_upper_air,
            target_diagnostic,
            varying_boundary_data,
        )

    def _constant_boundary_for(self, batch_size: int) -> torch.Tensor:
        """Expand (and ensemble-tile) the constant-boundary buffer to a batch.

        ``Trainer.get_dataset`` materialised the constant boundary as
        ``cbd.unsqueeze(0) * ones(B, 1, 1, 1)`` and (when ensembling) tiled it
        with :func:`to_ensemble_batch`. We keep the unbatched ``(c, h, w)``
        buffer and reproduce that expansion per step so the buffer is
        batch-size-agnostic (and Lightning owns its device placement).

        Args:
            batch_size: The *pre-ensemble* batch size ``B`` (the model input is
                ``B * num_ensemble_members`` rows once tiled).

        Returns:
            torch.Tensor: Constant-boundary tensor of shape
            ``(B * num_ensemble_members, c, h, w)`` on the module's device.

        Raises:
            RuntimeError: If no normalizer was supplied (so no constant-boundary
                buffer exists).
        """
        if self.constant_boundary_data is None:
            raise RuntimeError(
                "TrainModule has no constant_boundary_data; pass "
                "normalizer=datamodule.train_dataset when constructing it."
            )
        cbd = self.constant_boundary_data.unsqueeze(0).expand(batch_size, -1, -1, -1)
        if self.num_ensemble_members > 1:
            cbd = to_ensemble_batch(cbd, self.num_ensemble_members)
        return cbd

    def training_step(self, batch, batch_idx):
        """Run one training step: forward, combine losses, log, return loss.

        Faithful port of ``v2.0/train.py::Trainer.cal_loss`` and the per-step
        body of ``Trainer.train_one_epoch``. The model returns the 7-tuple
        ``(output_surface, output_upper_air, output_diagnostic, mu, sigma, mu2,
        sigma2)``; the total loss is, matching the source exactly:

        * ``(loss_sfc + loss_diagnostic) * 0.25 + loss_pl`` when diagnostics are
          present, else ``loss_sfc * 0.25 + loss_pl``; plus
        * ``vae_loss_weight * Kl_divergence_gaussians(mu, sigma, mu2, sigma2)``
          when ``params.vae_loss`` is set.

        Autocast and the GradScaler are intentionally **not** applied here:
        under Lightning automatic optimisation the Trainer's ``precision=``
        setting owns AMP. Lightning also drives ``backward`` and the optimiser
        step, so the NVTX ``backward`` / ``optimizer`` ranges that bracketed
        them in the source move to the Phase-3 ``BenchCallback`` (markers
        below) rather than being dropped.

        Args:
            batch: The training batch (5 or 6 tensors; see :meth:`_prepare_inputs`).
            batch_idx: Lightning's batch index (unused).

        Returns:
            torch.Tensor: The scalar training loss.
        """
        if _NVTX:
            nvtx.range_push("data_prep")
        (
            input_surface,
            input_upper_air,
            target_surface,
            target_upper_air,
            target_diagnostic,
            varying_boundary_data,
        ) = self._prepare_inputs(batch)
        if _NVTX:
            nvtx.range_pop()  # data_prep

        # Pre-ensemble batch size: rows added by ensembling are already in
        # input_surface, so derive B from it divided by the ensemble factor.
        batch_rows = input_surface.shape[0]
        pre_ens_b = batch_rows // self.num_ensemble_members
        constant_boundary_data = self._constant_boundary_for(pre_ens_b)

        # Phase 3 -> BenchCallback: on_train_batch_start torch.cuda.synchronize()
        #   + cudaProfilerStart and the step_{i} NVTX range bracket the whole
        #   step; they live on Lightning hooks, not inside training_step.
        if _NVTX:
            nvtx.range_push("forward_loss")
        (
            output_surface,
            output_upper_air,
            output_diagnostic,
            mu,
            sigma,
            mu2,
            sigma2,
        ) = self.model(
            input_surface,
            constant_boundary_data,
            varying_boundary_data,
            input_upper_air,
            target_surface,
            target_upper_air,
            train=True,
        )

        loss_diagnostic = 0
        if self.has_diagnostic:
            loss_diagnostic = self.loss_obj_diagnostic(output_diagnostic, target_diagnostic)
        loss_sfc = self.loss_obj_sfc(output_surface, target_surface)
        loss_pl = self.loss_obj_pl(output_upper_air, target_upper_air)

        if self.has_diagnostic:
            loss = (loss_sfc + loss_diagnostic) * 0.25 + loss_pl
        else:
            loss = (loss_sfc * 0.25) + loss_pl

        loss_vae = 0
        if getattr(self.params, "vae_loss", False):
            loss_vae = self.loss_vae(mu, sigma, mu2, sigma2)
            loss = loss + self.params.vae_loss_weight * loss_vae
        if _NVTX:
            nvtx.range_pop()  # forward_loss
        # Phase 3 -> BenchCallback: on_before_backward / on_after_backward push
        #   the "backward" NVTX range; on_before_optimizer_step pushes
        #   "optimizer"; on_train_batch_end syncs + records the step timing and,
        #   on the final measured step, cudaProfilerStop + writes the CSV row.

        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=self.ddp)
        self.log("train/loss_sfc", loss_sfc, on_step=True, on_epoch=False, sync_dist=self.ddp)
        self.log("train/loss_pl", loss_pl, on_step=True, on_epoch=False, sync_dist=self.ddp)
        if self.has_diagnostic:
            self.log(
                "train/loss_diagnostic", loss_diagnostic,
                on_step=True, on_epoch=False, sync_dist=self.ddp,
            )
        if getattr(self.params, "vae_loss", False):
            self.log("train/loss_vae", loss_vae, on_step=True, on_epoch=False, sync_dist=self.ddp)

        return loss

    @torch.no_grad()
    def predict(self, batch):
        """Roll the model out over the validation lead times and score it.

        Ports the per-batch core of ``v2.0/train.py::Trainer.validate_one_epoch``
        (the autoregressive rollout and per-lead-time loss accumulation), with
        the heavy diagnostic/ACC/spectra/plotting branches omitted — those write
        files and call wandb and belong to the Phase-4 inference path, not the
        Phase-2 module smoke. The validation batch ordering is the one returned
        by :meth:`utils.data_loader_multifiles.GetDataset.__getitem__` (validate
        branch, line 607 when diagnostics are present):
        ``(input_surface, input_upper_air, targets_surface, targets_upper_air,
        targets_diagnostic, varying_boundary_data, start_time_tensor)``.

        For each forecast lead time the model is stepped autoregressively
        (the surface/upper-air prediction feeds the next step's input, as in the
        source), and the surface / upper-air / diagnostic losses are combined
        with the same ``(loss_sfc + loss_diag) * 0.25 + loss_pl`` weighting used
        in training. The model is called in inference mode (5-tuple return:
        ``output_surface, output_upper_air, output_diagnostic, mu, sigma``),
        matching the validation call in the source.

        Args:
            batch: A validation batch (see ordering above).

        Returns:
            dict: Mapping ``"valid_loss_{step}step" -> scalar tensor`` of the
            combined loss at each configured forecast lead time.
        """
        params = self.params
        if self.has_diagnostic:
            (
                val_input_surface,
                val_input_upper_air,
                val_target_surface,
                val_target_upper_air,
                val_target_diagnostic,
                val_varying_boundary_data,
                _times,
            ) = batch
        else:
            (
                val_input_surface,
                val_input_upper_air,
                val_target_surface,
                val_target_upper_air,
                val_varying_boundary_data,
                _times,
            ) = batch
            val_target_diagnostic = None

        if self.num_ensemble_members > 1:
            tensors = [
                val_input_surface,
                val_input_upper_air,
                val_target_surface,
                val_target_upper_air,
                val_varying_boundary_data,
            ]
            if self.has_diagnostic:
                tensors.insert(4, val_target_diagnostic)
            tiled = [to_ensemble_batch(t, self.num_ensemble_members) for t in tensors]
            if self.has_diagnostic:
                (
                    val_input_surface,
                    val_input_upper_air,
                    val_target_surface,
                    val_target_upper_air,
                    val_target_diagnostic,
                    val_varying_boundary_data,
                ) = tiled
            else:
                (
                    val_input_surface,
                    val_input_upper_air,
                    val_target_surface,
                    val_target_upper_air,
                    val_varying_boundary_data,
                ) = tiled

        lead_times_steps = params.forecast_lead_times
        max_lead_time = max(lead_times_steps)
        pre_ens_b = val_input_surface.shape[0] // self.num_ensemble_members
        constant_boundary_data = self._constant_boundary_for(pre_ens_b)

        loss_dict = {}
        for step in range(max_lead_time):
            if _NVTX:
                nvtx.range_push(f"val_model_forward_step{step}")
            # Eval-forward arity branches on has_diagnostic (Phase-4 resolution of
            # the former TODO): PanguModel_Plasim.forward returns a 5-tuple
            # ``(surface, upper_air, diagnostic, mu, sigma)`` from its
            # ``if self.num_diagnostic_vars > 0`` branch and a 4-tuple
            # ``(surface, upper_air, mu, sigma)`` from the ``else`` branch (no
            # diagnostics). The diagnostic path (the one the smoke exercises,
            # test_midway.yaml has diagnostic_variables) is unchanged.
            if self.has_diagnostic:
                val_output_surface, val_output_upper_air, val_output_diagnostic, _, _ = self.model(
                    val_input_surface,
                    constant_boundary_data,
                    val_varying_boundary_data[:, step],
                    val_input_upper_air,
                )
            else:
                val_output_surface, val_output_upper_air, _, _ = self.model(
                    val_input_surface,
                    constant_boundary_data,
                    val_varying_boundary_data[:, step],
                    val_input_upper_air,
                )
                val_output_diagnostic = None
            if _NVTX:
                nvtx.range_pop()  # val_model_forward_step

            if (step + 1) in lead_times_steps:
                if _NVTX:
                    nvtx.range_push(f"val_loss_step{step}")
                target_index = step
                loss_sfc = self.loss_obj_sfc(val_output_surface, val_target_surface[:, target_index])
                loss_pl = self.loss_obj_pl(val_output_upper_air, val_target_upper_air[:, target_index])
                if self.has_diagnostic:
                    loss_diag = self.loss_obj_diagnostic(
                        val_output_diagnostic, val_target_diagnostic[:, target_index]
                    )
                    loss = (loss_sfc + loss_diag) * 0.25 + loss_pl
                else:
                    loss = loss_sfc * 0.25 + loss_pl
                loss_dict[f"valid_loss_{step + 1}step"] = loss
                if _NVTX:
                    nvtx.range_pop()  # val_loss_step

            val_input_surface, val_input_upper_air = val_output_surface, val_output_upper_air

        return loss_dict

    def validation_step(self, batch, batch_idx):
        """Score a validation batch, log per-lead-time losses, optionally save.

        Mirrors the SNFO ``validation_step`` public shape: delegate the rollout
        and scoring to :meth:`predict`, then log each per-lead-time loss with
        ``sync_dist=self.ddp``. When :attr:`do_save_predictions` is set, the
        source's prediction-saving behaviour (``Stepper.validate_one_epoch`` ->
        ``save_prediction``) is reproduced on rank 0 / batch 0 only -- matching
        SNFO's ``if batch_idx == 0 and (not self.ddp or self.global_rank == 0)``
        gate -- via :meth:`save_predictions`.

        Args:
            batch: A validation batch (see :meth:`predict`).
            batch_idx: Lightning's batch index (used to gate saving to batch 0).
        """
        loss_dict = self.predict(batch)
        for key, value in loss_dict.items():
            self.log(f"val/{key}", value, on_step=False, on_epoch=True, sync_dist=self.ddp)

        if self.do_save_predictions and batch_idx == 0:
            if not self.ddp or self.global_rank == 0:
                self.save_predictions(batch)

    @torch.no_grad()
    def _rollout_for_save(self, batch):
        """Full-horizon ensemble rollout + denormalisation for netCDF saving.

        Ports the save-preparation core of
        ``v2.0/inference.py::Stepper.validate_one_epoch`` (distinct from the
        per-lead-time *scoring* rollout in :meth:`predict`): for each of the two
        ensemble members it rolls the model out over the full
        :attr:`inference_steps` horizon, stacking every step (the input state
        plus each forecast) along a time axis, then **denormalises** the stacked
        surface / upper-air / diagnostic trajectories with the dataset's
        ``*_inv_transform`` (preserving normalize<->inverse symmetry exactly as
        the source) and moves them to CPU numpy. The autoregressive feed-forward
        (each step's prediction becomes the next step's input) is preserved.

        The model is called in inference mode, so its tuple arity branches on
        :attr:`has_diagnostic` (5-tuple with diagnostics / 4-tuple without; see
        :meth:`predict`).

        Note:
            The saved ensemble count is fixed at 2 (faithful to the source's
            ``for ens_id in range(2)``), intentionally independent of
            :attr:`num_ensemble_members`, which sizes only the *scoring*
            ensemble in :meth:`predict`. Re-coupling them would silently diverge
            from the canonical inference output.

        Args:
            batch: A validation batch (see :meth:`predict` for the ordering).

        Returns:
            list[dict]: One dict per ensemble member, each with keys
            ``"ens_id"`` (int), ``"surface"`` / ``"upper_air"`` /
            ``"diagnostic"`` (numpy arrays of shape ``(B, T, ...)``; the
            diagnostic entry is ``None`` when :attr:`has_diagnostic` is False)
            and ``"start_times"`` (a list of ``datetime_class`` start times, one
            per sample in the batch).

        Raises:
            RuntimeError: If no normalizer was supplied (no dataset to source the
                ``*_inv_transform`` statistics / ``datetime_class`` from).
        """
        if self.n is None:
            raise RuntimeError(
                "TrainModule has no normalizer; pass "
                "normalizer=datamodule.train_dataset to enable save_predictions."
            )
        if self.has_diagnostic:
            (
                val_input_surface,
                val_input_upper_air,
                _val_target_surface,
                _val_target_upper_air,
                _val_target_diagnostic,
                val_varying_boundary_data,
                times,
            ) = batch
        else:
            (
                val_input_surface,
                val_input_upper_air,
                _val_target_surface,
                _val_target_upper_air,
                val_varying_boundary_data,
                times,
            ) = batch

        # Reconstruct the per-sample start times from the times tensor
        # (year, month, day, hour), exactly as Stepper.validate_one_epoch does.
        start_times = []
        for i in range(times.shape[0]):
            start_times.append(
                self.n.datetime_class(
                    int(times[i, 0].item()),
                    int(times[i, 1].item()),
                    int(times[i, 2].item()),
                    hour=int(times[i, 3].item()),
                )
            )

        # The save path does NOT ensemble-tile the inputs (unlike the scoring
        # rollout in predict()): v2.0/inference.py::Stepper.validate_one_epoch
        # keeps the raw batch and draws the ensemble by re-running the stochastic
        # VAE model twice (the `for ens_id in range(2)` loop below). So build the
        # constant boundary at the *raw* input batch size, WITHOUT the ensemble
        # tiling _constant_boundary_for applies -- matching Stepper, which built
        # constant_boundary_data at params.batch_size and never tiled it.
        if self.constant_boundary_data is None:
            raise RuntimeError(
                "TrainModule has no constant_boundary_data; pass "
                "normalizer=datamodule.train_dataset when constructing it."
            )
        batch_b = val_input_surface.shape[0]
        constant_boundary_data = self.constant_boundary_data.unsqueeze(0).expand(
            batch_b, -1, -1, -1
        )

        results = []
        for ens_id in range(2):
            if _NVTX:
                nvtx.range_push(f"inference_save_ens_{ens_id}")
            in_surface = val_input_surface
            in_upper_air = val_input_upper_air

            # Initialise the trajectory with the input state (matches the source,
            # which seeds the lists with the input surface/upper-air and a zero
            # diagnostic frame).
            diag0 = torch.zeros(
                (in_surface.shape[0], self.model.num_diagnostic_vars,
                 in_surface.shape[2], in_surface.shape[3]),
                dtype=in_surface.dtype, device=in_surface.device,
            )
            surf_traj = [in_surface]
            upper_traj = [in_upper_air]
            diag_traj = [diag0]

            for time_step in range(self.inference_steps):
                if _NVTX:
                    nvtx.range_push(f"val_save_forward_step{time_step}")
                if self.has_diagnostic:
                    out_surface, out_upper_air, out_diagnostic, _, _ = self.model(
                        in_surface,
                        constant_boundary_data,
                        val_varying_boundary_data[:, time_step],
                        in_upper_air,
                    )
                else:
                    out_surface, out_upper_air, _, _ = self.model(
                        in_surface,
                        constant_boundary_data,
                        val_varying_boundary_data[:, time_step],
                        in_upper_air,
                    )
                    out_diagnostic = diag0
                surf_traj.append(out_surface)
                upper_traj.append(out_upper_air)
                diag_traj.append(out_diagnostic)
                in_surface, in_upper_air = out_surface, out_upper_air
                if _NVTX:
                    nvtx.range_pop()  # val_save_forward_step

            surf = torch.stack(surf_traj, dim=1)
            upper = torch.stack(upper_traj, dim=1)
            diag = torch.stack(diag_traj, dim=1)
            B, T = surf.shape[:2]

            # Denormalise on the flattened (B*T, ...) view, exactly as the source,
            # then reshape back to (B, T, ...) and move to CPU numpy.
            surf_np = (
                self.n.surface_inv_transform(surf.view(B * T, *surf.shape[2:]))
                .cpu().numpy().reshape(B, T, *surf.shape[2:])
            )
            upper_np = (
                self.n.upper_air_inv_transform(upper.view(B * T, *upper.shape[2:]))
                .cpu().numpy().reshape(B, T, *upper.shape[2:])
            )
            if self.has_diagnostic:
                diag_np = (
                    self.n.diagnostic_inv_transform(diag.view(B * T, *diag.shape[2:]))
                    .cpu().numpy().reshape(B, T, *diag.shape[2:])
                )
            else:
                diag_np = None

            results.append({
                "ens_id": ens_id,
                "surface": surf_np,
                "upper_air": upper_np,
                "diagnostic": diag_np,
                "start_times": start_times,
            })
            if _NVTX:
                nvtx.range_pop()  # inference_save_ens
        return results

    def save_predictions(self, batch) -> None:
        """Write per-sample netCDF predictions for a validation batch.

        Faithful port of ``v2.0/inference.py::Stepper.save_prediction``: for each
        ensemble member (from :meth:`_rollout_for_save`) and each sample whose
        forecast start time is at 00 UTC, it builds an :class:`xarray.Dataset`
        over the ``(time, level, latitude, longitude)`` coordinates and writes it
        to ``{predictions_dir}/{nettype}_{run_num}_{dt}h_{steps}step_{YYYYMMDDHH}_ens_{ens}.nc``.
        The 00 UTC filter, the CF axis attributes, and the per-variable
        ``DataArray`` construction match the source.

        Unlike ``Stepper.save_prediction`` this takes no ``experiment_dir`` /
        ``config_filepath`` bookkeeping (that was entry-point state, not climate
        logic); it writes straight to :attr:`predictions_dir`, which the entry
        point sets and creates. Saving the dataset to disk is heavy
        (xarray/cftime), so the smoke exercises this path on a small number of
        batches; whether any file is actually written depends on the data start
        hours (only 00 UTC samples are saved), which it logs.

        Args:
            batch: A validation batch (see :meth:`predict` for the ordering).

        Raises:
            RuntimeError: If :attr:`predictions_dir` is unset (the entry point
                must set ``params['predictions_dir']`` before saving).
        """
        if self.predictions_dir is None:
            raise RuntimeError(
                "save_predictions called without predictions_dir set; the entry "
                "point (val.py) must set params['predictions_dir']."
            )
        params = self.params
        savedir = self.predictions_dir
        os.makedirs(savedir, exist_ok=True)

        if _NVTX:
            nvtx.range_push("saving predictions...")
        ens_results = self._rollout_for_save(batch)

        n_written = 0
        for res in ens_results:
            ens_id = res["ens_id"]
            surface_prediction = res["surface"]
            upper_air_prediction = res["upper_air"]
            diagnostic_prediction = res["diagnostic"]
            start_times = res["start_times"]

            for sample in range(surface_prediction.shape[0]):
                time_range = xr.cftime_range(
                    start_times[sample]
                    + timedelta(hours=params["timedelta_hours"] * sample),
                    start_times[sample]
                    + timedelta(hours=params["timedelta_hours"]
                                * (sample + self.inference_steps)),
                    freq="%dh" % params["timedelta_hours"],
                    inclusive="both",
                )
                coordinates = {
                    "time": time_range,
                    "level": params.levels,
                    "latitude": params.lat,
                    "longitude": params.lon,
                }
                if start_times[sample].strftime("%H") != "00":
                    print(
                        f"[save_predictions] skipping start time {start_times[sample]} "
                        "(not 00UTC)", flush=True,
                    )
                    continue

                run_num = params["run_num"] if "run_num" in params else "lightning"
                filename = "%s_%s_%dh_%dstep_%s_ens_%s.nc" % (
                    params.nettype, run_num, params["timedelta_hours"],
                    self.inference_steps, start_times[sample].strftime("%Y%m%d%H"),
                    ens_id,
                )
                dataset = xr.Dataset(
                    data_vars=dict(),
                    coords=coordinates,
                    attrs=dict(description=(
                        f"Prediction from {params.nettype} model run {run_num}")),
                )
                dataset["level"].attrs["axis"] = "Z"
                dataset["latitude"].attrs["axis"] = "Y"
                dataset["longitude"].attrs["axis"] = "X"
                # This single line is required for guess_coord_axis to work.
                dataset["level"].attrs["positive"] = "down"
                dataset = dataset.cf.guess_coord_axis()

                for idx, var in enumerate(self.n.surface_variables):
                    dataset[var] = xr.DataArray(
                        data=surface_prediction[sample, :, idx],
                        dims=["time", "latitude", "longitude"],
                        coords={
                            "time": time_range,
                            "latitude": dataset.latitude.values,
                            "longitude": dataset.longitude.values,
                        },
                    )
                for idx, var in enumerate(self.n.upper_air_variables):
                    dataset[var] = xr.DataArray(
                        data=upper_air_prediction[sample, :, idx],
                        dims=["time", "level", "latitude", "longitude"],
                        coords=coordinates,
                    )
                if self.has_diagnostic and diagnostic_prediction is not None:
                    for idx, var in enumerate(self.n.diagnostic_variables):
                        dataset[var] = xr.DataArray(
                            data=diagnostic_prediction[sample, :, idx],
                            dims=["time", "latitude", "longitude"],
                            coords={
                                "time": time_range,
                                "latitude": dataset.latitude.values,
                                "longitude": dataset.longitude.values,
                            },
                        )

                dataset["latitude"] = dataset["latitude"].astype("float32").assign_attrs(
                    {"long_name": "Latitude", "unit": "degrees_north"})
                dataset["longitude"] = dataset["longitude"].astype("float32").assign_attrs(
                    {"long_name": "Longitude", "unit": "degrees_east"})
                dataset["time"] = dataset["time"].assign_attrs(
                    {"long_name": "Forecast Valid Time"})
                dataset["level"] = dataset["level"].astype("float32").assign_attrs(
                    {"long_name": "Level", "unit": "hPa"})
                dataset = dataset.chunk({"time": 1, "level": 1})
                out_path = os.path.join(savedir, filename)
                dataset.to_netcdf(out_path, "w")
                n_written += 1
                print(f"[save_predictions] wrote {out_path}", flush=True)
        if _NVTX:
            nvtx.range_pop()  # saving predictions...
        print(
            f"[save_predictions] {n_written} netCDF file(s) written to {savedir} "
            f"(00UTC samples only)", flush=True,
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Roll out and save predictions for one batch under ``trainer.predict``.

        The ``trainer.predict`` counterpart of the validation save path: it
        writes per-sample netCDF predictions via :meth:`save_predictions` for
        every batch (not just batch 0). The source
        ``v2.0/inference.py::Stepper.predict`` delegates to
        ``Stepper.validate_one_epoch``, whose batch loop both rolls out and
        saves (capped there at ``i > 5``); Lightning owns batch iteration here,
        so that per-batch cap is expressed via the Trainer's
        ``limit_predict_batches`` instead. The validation entry point ``val.py``
        drives saving through
        :meth:`validation_step` (``trainer.validate``); this method is the path
        used when a caller drives ``trainer.predict`` against the
        :meth:`data.datamodule.ClimateDataModule.predict_dataloader` instead.

        Args:
            batch: A validation/inference batch (see :meth:`predict`).
            batch_idx: Lightning's batch index (unused; predict saves every batch).
            dataloader_idx: Lightning's dataloader index (unused).

        Returns:
            None: Predictions are written to :attr:`predictions_dir` as a side
            effect; there is no in-memory return.
        """
        self.save_predictions(batch)
        return None

    def configure_optimizers(self):
        """Build the optimiser and LR scheduler, faithful to the source.

        Ports ``v2.0/train.py::Trainer.get_optimizer`` (Adam, with
        ``fused=True`` when ``optimizer_type == "FusedAdam"``) and
        ``Trainer.setup_scheduler`` (``ReduceLROnPlateau`` / ``CosineAnnealingLR``
        / ``OneCycleLR`` per ``params.scheduler``). Unlike the manual loop,
        Lightning steps the scheduler — there is no ``scheduler.step()`` here.

        For ``OneCycleLR`` the total step count needs ``steps_per_epoch``; under
        Lightning that is taken from the trainer's
        ``estimated_stepping_batches`` (``max_epochs * steps_per_epoch``) so the
        schedule spans the configured run. ``ReduceLROnPlateau`` monitors
        ``val/valid_loss_1step`` (the 1-step lead-time validation loss), which is
        faithful to the source: ``v2.0/train.py`` stepped the plateau scheduler on
        ``valid_logs['valid_loss']`` (``train.py:623``), and that ``valid_loss`` is
        accumulated only at the first lead time (``if step == 0`` in
        ``validate_one_epoch``, ``train.py:1333-1334``) -- i.e. the 1-step loss.
        The all-lead-times ``multi_step_losses`` accumulator is never monitored.

        Returns:
            tuple | torch.optim.Optimizer: ``([optimizer], [scheduler])`` when a
            scheduler is configured (a dict-wrapped scheduler for the plateau /
            one-cycle cases that need a monitor or per-step interval), else just
            the optimizer when ``params.scheduler`` selects no scheduler.

        Raises:
            ValueError: Never raised directly; an unrecognised
                ``params.scheduler`` falls through to "no scheduler" exactly as
                the source's ``setup_scheduler`` does.
        """
        params = self.params
        if getattr(params, "optimizer_type", "Adam") == "FusedAdam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=params.lr,
                weight_decay=params.weight_decay, fused=True,
            )
        else:
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=params.lr, weight_decay=params.weight_decay,
            )

        scheduler_name = getattr(params, "scheduler", None)
        if scheduler_name == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.2, patience=5, mode="min",
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val/valid_loss_1step"},
            }
        elif scheduler_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=params.max_epochs,
            )
            return [optimizer], [scheduler]
        elif scheduler_name == "OneCycleLR":
            total_steps = int(self.trainer.estimated_stepping_batches)
            pct_start = getattr(params, "oc_pct_start", 0.3)
            div_factor = getattr(params, "oc_div_factor", 25)
            final_div_factor = getattr(params, "oc_final_div_factor", 1e4)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=params.lr,
                total_steps=total_steps,
                pct_start=pct_start,
                div_factor=div_factor,
                final_div_factor=final_div_factor,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        else:
            return optimizer
