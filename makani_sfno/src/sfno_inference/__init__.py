"""sfno_inference — eval-time rollout driver for the PlaSim SFNO emulator.

This package implements the v2 inference path required by the hard gate
in ``src/sfno_training/trainer/plasim_trainer.py:93`` (which blocks the
stock Makani Inferencer because it drops the 6 PlaSim forcing channels).

Modules:
  - ``checkpoint_loader`` — load eval params from the run-dir
    ``config.json`` and build a PlasimMultiStepWrapper from the saved
    checkpoint. Asserts the 58→53 channel contract and the AMP/device
    contracts (see ``docs/sfno_eval_plan.md`` §B.0).
  - ``rollout_driver`` — arbitrary-K rollout that mirrors
    ``validate_one_epoch`` step-by-step (see §B.1-§B.3).
  - ``nc_writer`` — physical-units NetCDF output with the lead-time
    schema from §B.4.
"""
from sfno_inference.checkpoint_loader import (
    load_eval_params,
    build_wrapper_from_checkpoint,
)
from sfno_inference.rollout_driver import (
    RolloutResult,
    rollout_one_ic,
    nwp_ic_offsets,
)
from sfno_inference.nc_writer import write_rollout_nc

__all__ = [
    "load_eval_params",
    "build_wrapper_from_checkpoint",
    "RolloutResult",
    "rollout_one_ic",
    "nwp_ic_offsets",
    "write_rollout_nc",
]
