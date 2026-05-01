"""PR-A integration test: single-step + two-step rollout through
Plasim{Single,Multi}StepWrapper, with content sentinels via
RecordingDummyModel.

Asserts:
  (a) model.inputs_seen[k].shape[1] == 58 for all k
  (b) state portion inputs_seen[k][:, :52] never contains the pr_6h sentinel
  (c) forcing portion inputs_seen[k][:, 52:58] matches tar_forcing_normalized[:, k-1:k]
      exactly for k >= 1 (Codex round 2 fix #3 — k-1:k, not k:k+1)
  (d) negative regression: stock MultiStepWrapper fails on two-step

See docs/sfno_training_implementation_plan.md §4 for the autoregressive
rollout invariants.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
makani = pytest.importorskip("makani")

from makani.models.stepper import MultiStepWrapper  # noqa: E402
from makani.utils.loss import LossHandler  # noqa: E402

from helpers import RecordingDummyModel, build_dataset, load_params  # noqa: E402

from sfno_training.models import (  # noqa: E402
    PlasimMultiStepWrapper,
    PlasimPreprocessor,
    PlasimSingleStepWrapper,
)


def _recording_model_handle(inp_shape, out_shape, inp_chans=58, out_chans=53):
    """Factory matching stock get_model's partial signature."""
    return partial(
        RecordingDummyModel,
        inp_shape=inp_shape,
        out_shape=out_shape,
        inp_chans=inp_chans,
        out_chans=out_chans,
    )


def _shape(params):
    return (params.img_shape_x, params.img_shape_y)


def test_single_step_wrapper_content(packaged_dataset: Path):
    """Single-step wrapper sees 58-channel input; loss is finite; content
    sentinels (a)-(b) hold for k=0."""
    params = load_params(packaged_dataset)
    ds = build_dataset(params, packaged_dataset, n_future=0)
    inp_state, tar, inp_forcing, tar_forcing = ds[0]

    params.N_in_channels = params.n_state_channels + params.n_forcing_channels
    params.N_out_channels = 53

    wrapper = PlasimSingleStepWrapper(
        params, _recording_model_handle(_shape(params), _shape(params))
    )
    wrapper.train()
    assert isinstance(wrapper.preprocessor, PlasimPreprocessor)

    inp_b = inp_state.unsqueeze(0)        # (1, 1, 52, 64, 128)
    tar_b = tar.unsqueeze(0)              # (1, 1, 53, 64, 128)
    xz = inp_forcing.unsqueeze(0)         # (1, 1, 6, 64, 128)
    yz = tar_forcing.unsqueeze(0)         # (1, 1, 6, 64, 128)

    # Defensive snapshot of expected forcing BEFORE the forward. Makani 0.2.0
    # (released wheel) had `cache_unpredicted_features` aliasing xz directly:
    # `self.unpredicted_inp_train = xz` at preprocessor.py:400. The in-place
    # `.copy_(utar)` inside append_history then mutated the caller's xz. The
    # repo `makani-src/` clone (upstream main, commit c970430) ships the fix
    # as `xz.clone()`. We're now pinned to the editable install of that clone
    # so this snapshot is no longer load-bearing — kept as a defensive guard
    # in case the env drifts back to a wheel that doesn't have the fix.
    expected_xz = xz.clone()

    inp_b, tar_b = wrapper.preprocessor.cache_unpredicted_features(inp_b, tar_b, xz, yz)
    inp_flat = wrapper.preprocessor.flatten_history(inp_b)
    tar_flat = wrapper.preprocessor.flatten_history(tar_b)

    pred = wrapper(inp_flat)
    assert pred.shape == (1, 53, 64, 128)

    # Content sentinel (a): the model saw exactly 58 channels at step 0.
    seen = wrapper.model.inputs_seen
    assert len(seen) == 1
    assert seen[0].shape == (1, 58, 64, 128)
    # Content sentinel (b): step 0 state portion is the original input
    # state (pre-rollout), so no sentinel can be there.
    assert not torch.any(seen[0][:, :52] == RecordingDummyModel.PR_6H_SENTINEL)
    # Bonus check: the forcing portion is exactly the (normalized) input forcing.
    assert torch.allclose(seen[0][:, 52:58], expected_xz.squeeze(1))

    # Loss is finite (RecordingDummyModel emits a sentinel pr_6h, but that's
    # still a finite float, so the loss handler accepts it).
    loss_fn = LossHandler(params)
    assert loss_fn.channel_weights.shape[1] == 53
    loss_val = loss_fn(pred, tar_flat)
    assert torch.isfinite(loss_val).item()


def test_multistep_wrapper_content(packaged_dataset: Path):
    """Two-step training rollout: every step sees 58 input channels; the
    pr_6h sentinel never leaks into the state portion of subsequent
    steps; the forcing portion at step k >= 1 matches tar_forcing[k-1:k]."""
    params = load_params(packaged_dataset)
    ds = build_dataset(params, packaged_dataset, n_future=1)
    inp_state, tar, inp_forcing, tar_forcing = ds[0]

    params.N_in_channels = params.n_state_channels + params.n_forcing_channels
    params.N_out_channels = 53
    params.n_future = 1

    loss_fn_multi = LossHandler(params)
    assert loss_fn_multi.channel_weights.shape[1] == 53
    assert loss_fn_multi.multistep_weight.shape[1] == 53 * 2

    wrapper = PlasimMultiStepWrapper(
        params, _recording_model_handle(_shape(params), _shape(params))
    )
    wrapper.train()
    assert isinstance(wrapper.preprocessor, PlasimPreprocessor)

    inp_b = inp_state.unsqueeze(0)        # (1, 1, 52, 64, 128)
    tar_b = tar.unsqueeze(0)              # (1, 2, 53, 64, 128)
    xz = inp_forcing.unsqueeze(0)         # (1, 1, 6, 64, 128)
    yz = tar_forcing.unsqueeze(0)         # (1, 2, 6, 64, 128)

    # Defensive snapshots of expected forcing BEFORE the forward. Makani 0.2.0
    # (released wheel) aliased xz in `cache_unpredicted_features`; the in-place
    # `.copy_(utar)` inside append_history then mutated the caller's xz. Upstream
    # main (commit c970430, the repo `makani-src/` clone we now run editable)
    # ships the fix as `xz.clone()`. Snapshots kept as a defensive guard in case
    # the env drifts back to a wheel without the fix.
    expected_xz = xz.clone()
    expected_yz = yz.clone()

    inp_b, tar_b = wrapper.preprocessor.cache_unpredicted_features(inp_b, tar_b, xz, yz)
    inp_flat = wrapper.preprocessor.flatten_history(inp_b)
    tar_flat = wrapper.preprocessor.flatten_history(tar_b)

    pred_ms = wrapper(inp_flat)
    assert pred_ms.shape == (1, 53 * 2, 64, 128)
    loss_ms = loss_fn_multi(pred_ms, tar_flat)
    assert torch.isfinite(loss_ms).item()

    seen = wrapper.model.inputs_seen
    assert len(seen) == 2, f"expected 2 model invocations for n_future=1, got {len(seen)}"

    # (a) every step has 58 input channels
    for k, x in enumerate(seen):
        assert x.shape == (1, 58, 64, 128), f"step {k} shape {x.shape}"

    # (b) the pr_6h sentinel never leaks into the state portion of any step
    for k, x in enumerate(seen):
        leaked = (x[:, :52] == RecordingDummyModel.PR_6H_SENTINEL).any().item()
        assert not leaked, f"pr_6h sentinel leaked into state input at step {k}"

    # (c) forcing-content alignment (Codex round 2 fix #3 — k-1:k indexing)
    # step 0: forcing portion == input forcing (expected_xz)
    assert torch.allclose(seen[0][:, 52:58], expected_xz.squeeze(1))
    # step 1: forcing portion == tar_forcing[:, 0:1] (yz[:, k-1:k] with k=1)
    assert torch.allclose(seen[1][:, 52:58], expected_yz[:, 0])


def test_multistep_stock_wrapper_fails(packaged_dataset: Path):
    """Negative regression (d): without PlasimPreprocessor, stock
    MultiStepWrapper feeds the full 53-channel pred back into the next
    step's append_unpredicted_features (which then yields 59 channels),
    breaking the model's 58-channel input. Must crash.

    Uses a strict Conv2d (locked to 58 in-channels) — RecordingDummyModel
    silently zero-fills regardless of input shape, so it can't catch this.
    """
    import torch.nn as nn

    params = load_params(packaged_dataset)
    ds = build_dataset(params, packaged_dataset, n_future=1)
    inp_state, tar, inp_forcing, tar_forcing = ds[0]

    params.N_in_channels = params.n_state_channels + params.n_forcing_channels
    params.N_out_channels = 53
    params.n_future = 1

    ms_stock = MultiStepWrapper(params, lambda: nn.Conv2d(58, 53, kernel_size=1))
    ms_stock.train()

    inp_b = inp_state.unsqueeze(0)
    tar_b = tar.unsqueeze(0)
    xz = inp_forcing.unsqueeze(0)
    yz = tar_forcing.unsqueeze(0)

    inp_b, tar_b = ms_stock.preprocessor.cache_unpredicted_features(inp_b, tar_b, xz, yz)
    inp_flat = ms_stock.preprocessor.flatten_history(inp_b)
    with pytest.raises((RuntimeError, AssertionError)):
        _ = ms_stock(inp_flat)
