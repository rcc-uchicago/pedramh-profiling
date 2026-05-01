"""sfno_training — PlaSim → Makani SFNO training/validation wrapper.

Subclasses + monkey-patches stock Makani so it consumes the asymmetric
PlaSim dataset contract (52 state + 6 forcing → 58 input; 52 state + 1
diagnostic = 53 output; pr_6h is loss-only and never fed back).

Inference is OUT OF SCOPE in this PR — see
docs/sfno_training_implementation_plan.md §"Hard gate on full emulator
rollout". Stock makani.utils.inference.inferencer.Inferencer would
silently drop forcing and produce physically wrong predictions on this
contract; gated at sfno_training.trainer.plasim_trainer._plasim_get_dataloader.
"""
