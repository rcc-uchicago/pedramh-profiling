"""Phase 4b smoke shim — re-export the production sfno_training classes.

This module used to define four classes itself (PlasimForcingDataset,
PlasimPreprocessor, PlasimSingleStepWrapper, PlasimMultiStepWrapper).
The production versions now live under ``src/sfno_training/`` (PR-A of
docs/sfno_training_implementation_plan.md). Phase 4b's
``test_multifile_loader_smoke.py`` continues to import from this module
— kept as a thin re-export so a single source of truth lives in
``src/sfno_training/`` while the existing packager test suite stays
green with zero test changes.

The Python 3.12 ``get_timedelta_from_timestamp`` shim is installed as a
side effect of importing ``sfno_training.compat`` (transitively imported
via ``sfno_training.data``), exactly as it was before.
"""

from __future__ import annotations

from sfno_training.data import PlasimForcingDataset
from sfno_training.models import (
    PlasimMultiStepWrapper,
    PlasimPreprocessor,
    PlasimSingleStepWrapper,
)

__all__ = [
    "PlasimForcingDataset",
    "PlasimPreprocessor",
    "PlasimSingleStepWrapper",
    "PlasimMultiStepWrapper",
]
