"""Pytest config for tests/sfno_training/.

Responsibilities:
1. sys.path: prepend ``{repo}/src`` so ``sfno_training.*`` imports work,
   plus this directory and ``tests/plasim_makani_packager`` (we reuse
   ``test_hdf5_writer._make_*`` to build synthetic input data).
2. Register the ``RecordingDummyModel`` nettype with Makani's model
   registry so PR-A's wrapper test and PR-B's trainer-CI test can both
   request ``params.nettype = "plasim_test_recording_dummy"``.
3. Provide the ``packaged_dataset`` fixture (module-scoped).

Heavy deps (torch, makani) are gated via ``importorskip`` — on a node
without the full Makani dependency set (e.g. Stampede3 login), the
whole module is silently skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
_TESTS_DIR = Path(__file__).resolve().parent
_PACKAGER_TESTS = _REPO_ROOT / "tests" / "plasim_makani_packager"

for entry in (_SRC, _TESTS_DIR, _PACKAGER_TESTS):
    s = str(entry)
    if s not in sys.path:
        sys.path.insert(0, s)


# Heavy deps gate
pytest.importorskip("torch")
pytest.importorskip("makani")


# ---------------------------------------------------------------------------
# Custom 'slow' marker — registered here to silence the
# PytestUnknownMarkWarning on test_smoke_sfno_cpu.py. Slow tests are NOT
# excluded by default: developers run them via ``pytest -m slow`` per
# docs/sfno_training_implementation_plan.md §"Test commands". CI-style
# runs use ``--ignore=tests/sfno_training/test_smoke_sfno_cpu.py``.
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: mark test as slow (CPU SFNO smoke etc.)"
    )


# ---------------------------------------------------------------------------
# Register RecordingDummyModel as a nettype
# ---------------------------------------------------------------------------
# Idempotent: pytest can re-import conftest in the same process (e.g.
# ``pytest --collect-only`` followed by ``pytest``); Makani's
# ``_register_from_module`` raises on duplicate names, so drop-and-rewrite.
from makani.models.model_registry import register_model, _model_registry  # noqa: E402

from helpers import RecordingDummyModel  # noqa: E402

_REGISTERED_NAME = "plasim_test_recording_dummy"
if _REGISTERED_NAME in _model_registry:
    del _model_registry[_REGISTERED_NAME]
register_model(RecordingDummyModel, name=_REGISTERED_NAME)


# ---------------------------------------------------------------------------
# Synthetic packaged-dataset fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def packaged_dataset(tmp_path_factory) -> Path:
    """Tiny packaged dataset: 3 train years, sim52 layout, T=10 per file."""
    from helpers import build_packaged_dataset

    return build_packaged_dataset(tmp_path_factory)
