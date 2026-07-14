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

from makani.models.preprocessor import Preprocessor2D


class PlasimPreprocessor(Preprocessor2D):
    """Auto-strip diagnostic channels from ``pred`` before feedback.

    ``params.n_state_channels``      -- state-feedback channels (52 in the v9 contract).
    ``params.n_diagnostic_channels`` -- loss-only channels (1 in the v9 contract).

    Only :meth:`append_history` is overridden; all other behavior
    (forcing caching, append_unpredicted_features, history_normalize,
    ...) is inherited unchanged.
    """

    def __init__(self, params):
        super().__init__(params)
        self.n_state_channels = params.n_state_channels
        self.n_full_out_channels = (
            params.n_state_channels + params.n_diagnostic_channels
        )

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
