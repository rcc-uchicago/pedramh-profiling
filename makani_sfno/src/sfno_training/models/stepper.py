"""PlasimSingleStepWrapper / PlasimMultiStepWrapper — install
:class:`PlasimPreprocessor` in place of stock :class:`Preprocessor2D`.

The stock ``_forward_train`` / ``_forward_eval`` loop bodies in
:class:`MultiStepWrapper` reach into ``self.preprocessor.append_history``
and ``self.preprocessor.cache_unpredicted_features``, so swapping the
preprocessor in ``__init__`` is enough to redirect those calls through
the patched class.

Trainer wires these in via the two monkey-patches on
``makani.models.model_registry.{SingleStepWrapper, MultiStepWrapper}``
(see ``sfno_training.trainer.plasim_trainer._install_plasim_patches``).
"""

from __future__ import annotations

from makani.models.stepper import MultiStepWrapper, SingleStepWrapper

from sfno_training.models.preprocessor import PlasimPreprocessor


class PlasimSingleStepWrapper(SingleStepWrapper):
    def __init__(self, params, model_handle):
        super().__init__(params, model_handle)
        # Replace stock Preprocessor2D with the PlasimPreprocessor subclass.
        # nn.Module.__setattr__ correctly re-registers the submodule.
        self.preprocessor = PlasimPreprocessor(params)


class PlasimMultiStepWrapper(MultiStepWrapper):
    def __init__(self, params, model_handle):
        super().__init__(params, model_handle)
        self.preprocessor = PlasimPreprocessor(params)
