"""PR-A unit test: PlasimPreprocessor.append_history strips diagnostic
and hard-rejects unexpected shapes.

Plan §7 step 9.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
makani = pytest.importorskip("makani")

import torch.nn as nn  # noqa: E402

from helpers import load_params  # noqa: E402

from sfno_training.models import PlasimPreprocessor, PlasimSingleStepWrapper  # noqa: E402


def _make_wrapper(packaged_dataset: Path) -> PlasimSingleStepWrapper:
    """Wrapper construction is the easiest path to a fully-initialized
    PlasimPreprocessor (Preprocessor2D needs a populated params object)."""
    params = load_params(packaged_dataset)
    model_handle = lambda: nn.Conv2d(58, 53, kernel_size=1)
    return PlasimSingleStepWrapper(params, model_handle)


def test_append_history_strips_full_pred(packaged_dataset: Path):
    """53-channel x2 (full pred) -> 52-channel output (diagnostic stripped)."""
    wrapper = _make_wrapper(packaged_dataset)
    wrapper.train()

    x1 = torch.zeros(1, 52, 64, 128)
    x2 = torch.randn(1, 53, 64, 128)
    out = wrapper.preprocessor.append_history(x1, x2, step=0, update_state=False)
    assert out.shape == (1, 52, 64, 128)


def test_append_history_passes_through_state_only(packaged_dataset: Path):
    """52-channel x2 (already state-only) -> 52-channel output (no slice)."""
    wrapper = _make_wrapper(packaged_dataset)
    wrapper.train()

    x1 = torch.zeros(1, 52, 64, 128)
    x2 = torch.randn(1, 52, 64, 128)
    out = wrapper.preprocessor.append_history(x1, x2, step=0, update_state=False)
    assert out.shape == (1, 52, 64, 128)
    # And the values must be identical to x2 (no silent slicing)
    assert torch.equal(out, x2)


def test_append_history_rejects_unexpected_channel_count(packaged_dataset: Path):
    """Anything other than n_state_channels or n_full_out_channels must
    crash loudly. plan §7 + Codex round 1 fix #4."""
    wrapper = _make_wrapper(packaged_dataset)
    wrapper.train()

    x1 = torch.zeros(1, 52, 64, 128)
    with pytest.raises(AssertionError, match="channels must be"):
        wrapper.preprocessor.append_history(
            x1, torch.zeros(1, 60, 64, 128), step=0, update_state=False
        )
    with pytest.raises(AssertionError, match="channels must be"):
        wrapper.preprocessor.append_history(
            x1, torch.zeros(1, 1, 64, 128), step=0, update_state=False
        )


def test_append_history_rejects_non_4d(packaged_dataset: Path):
    """3D x2 (e.g. accidentally squeezed) must crash loudly."""
    wrapper = _make_wrapper(packaged_dataset)
    wrapper.train()

    x1 = torch.zeros(1, 52, 64, 128)
    with pytest.raises(AssertionError, match="4D"):
        wrapper.preprocessor.append_history(
            x1, torch.zeros(1, 53, 64), step=0, update_state=False
        )
