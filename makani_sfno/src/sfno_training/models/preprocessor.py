"""PlasimPreprocessor — strip the diagnostic channel(s) from ``pred``
before the autoregressive feedback copy.

Covers two in-scope rollout call sites (inference is explicitly out of
scope per docs/sfno_training_implementation_plan.md §"Hard gate on full
emulator rollout"):

  - ``makani.models.stepper.MultiStepWrapper._forward_train`` at
    ``makani/makani/models/stepper.py:112`` -- training rollout.
  - ``makani.utils.training.deterministic_trainer.Trainer.validate_one_epoch``
    at ``makani/makani/utils/training/deterministic_trainer.py:661`` --
    validation rollout (shares the wrapper's preprocessor instance via
    ``self.preprocessor = self.model.preprocessor``).
"""

from __future__ import annotations

import numpy as np
from makani.models.preprocessor import Preprocessor2D


def _pget(params, key, default=None):
    """``params.get`` for makani ParamsBase / dict, ``getattr`` otherwise."""
    get = getattr(params, "get", None)
    if callable(get):
        try:
            return get(key, default)
        except TypeError:
            pass
    return getattr(params, key, default)


def _build_dry_air_fix(params):
    from sfno_training.models.mass_fix import DryAirFix

    for key in ("h_parallel_size", "w_parallel_size"):
        if int(_pget(params, key, 1) or 1) != 1:
            raise ValueError(f"DRY_AIR_FIX: {key}={_pget(params, key)}; the global mean "
                             "needs a spatial all-reduce, not implemented")
    return DryAirFix(
        channel_names=list(params.channel_names),
        global_means=np.load(params.global_means_path),
        global_stds=np.load(params.global_stds_path),
        nlat=int(_pget(params, "img_crop_shape_x", None) or params.img_shape_x),
    )


class PlasimPreprocessor(Preprocessor2D):
    """Auto-strip diagnostic channels from ``pred`` before feedback.

    ``params.n_state_channels``      -- state-feedback channels (52 in the v9 contract).
    ``params.n_diagnostic_channels`` -- loss-only channels (1 in the v9 contract).

    :meth:`append_history` strips the diagnostic tail. The optional dry-air fix
    adds two pass-through hooks (:meth:`append_unpredicted_features`,
    :meth:`history_denormalize`) that do nothing unless ``conserve_dry_air`` is on.
    All other behavior (forcing caching, history_normalize, ...) is inherited unchanged.
    """

    def __init__(self, params):
        super().__init__(params)
        self.n_state_channels = params.n_state_channels
        self.n_full_out_channels = (
            params.n_state_channels + params.n_diagnostic_channels
        )
        # Dry-air mass fix (models/mass_fix.py). DIAGNOSTIC ARM ONLY: off unless
        # params.conserve_dry_air is true; with it off nothing below runs and the
        # step is bitwise unchanged.
        self.dry_air_fix = None
        self._fix_inp_state = None
        if bool(_pget(params, "conserve_dry_air", False)):
            self.dry_air_fix = _build_dry_air_fix(params)

    # --- dry-air fix hooks -------------------------------------------------
    # makani's stepper calls, once per predicted step and in this order,
    #   append_unpredicted_features(inp)  ->  model  ->  history_denormalize(yn, target=True)
    # (stepper.py:34/50, :82/100, :125/143; no other callers). The first hook
    # remembers the step's input state; the second corrects the prediction.
    # getattr: instances built without __init__ (the climate-driver tests do this)
    # must behave exactly as before the fix existed.
    def append_unpredicted_features(self, inp, target=False):
        if getattr(self, "dry_air_fix", None) is not None and not target:
            self._fix_inp_state = inp[:, : self.n_state_channels]
        return super().append_unpredicted_features(inp, target=target)

    def history_denormalize(self, xn, target=False):
        x = super().history_denormalize(xn, target=target)
        if getattr(self, "dry_air_fix", None) is not None and target:
            if self._fix_inp_state is None:
                raise RuntimeError("DRY_AIR_FIX: prediction without a cached input state")
            x = self.dry_air_fix(self._fix_inp_state, x)
            self._fix_inp_state = None
        return x

    def append_history(self, x1, x2, step, update_state=True):
        # Hard-fail on any shape drift -- plan §2 + Codex round 1 fix #4.
        # x2 must be either:
        #   - n_state_channels      (already-stripped state-only tensor; pass through)
        #   - n_full_out_channels   (full pred from model; strip diagnostic tail)
        # Anything else is a bug somewhere upstream and must crash loudly.
        assert x2.dim() == 4, (
            f"PlasimPreprocessor.append_history expected x2 4D (B, C, H, W), "
            f"got {x2.dim()}D shape {tuple(x2.shape)}"
        )
        assert x2.shape[1] in (self.n_state_channels, self.n_full_out_channels), (
            f"PlasimPreprocessor.append_history: x2 channels must be "
            f"{self.n_state_channels} or {self.n_full_out_channels}, "
            f"got {x2.shape[1]}"
        )
        if x2.shape[1] == self.n_full_out_channels:
            x2 = x2[:, : self.n_state_channels, ...]
        return super().append_history(x1, x2, step, update_state=update_state)
