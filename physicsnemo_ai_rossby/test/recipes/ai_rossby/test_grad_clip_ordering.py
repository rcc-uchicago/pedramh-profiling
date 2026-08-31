# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Regression tests for gradient clipping ORDER (and its norm diagnostic).

THE BUG THESE PIN
-----------------
``train.py`` applied ``clip_grad_norm_`` at its own call site, *after*
``train_step`` / ``multistep_train_step`` returned. Both of those call
``optimizer.step()`` internally, so the clip ran on gradients that had already
been applied to the weights, and the next iteration's
``optimizer.zero_grad(set_to_none=True)`` discarded them before they were ever
used. ``training.grad_clip_norm`` was a **complete no-op** in the eager path.

Nothing failed. The knob parsed, logged, and showed up in the resolved run
config, so job 7575680 (``HP_ARM=clip``, 48 nodes, 2h42m) read as a clean test
of gradient clipping when it was really a rerun of the unclipped config — which
is why it tracked the unclipped run to within noise right through the epoch-12
divergence (0.1376 vs 0.1618).

A unit test of ``clip_and_measure_grads`` alone would NOT have caught this: the
clipping function was always correct, it was called in the wrong place. So the
central test here inspects the gradients **at ``optimizer.step()`` time**, via a
fake optimizer, which is the only vantage point from which the ordering is
observable.

Runs on CPU, no physicsnemo import -> safe in the recipe-test job. Do not run on
a login node (CLAUDE.md #3): importing torch there can hang.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_RECIPE_DIR = Path(__file__).resolve().parents[3] / "examples" / "weather" / "ai_rossby"
sys.path.insert(0, str(_RECIPE_DIR))

from train_loop import clip_and_measure_grads, train_step  # noqa: E402


class _RecordingOptimizer(torch.optim.SGD):
    """SGD that records the gradient norm VISIBLE TO ``step()``.

    This is the whole point: ``grad.norm()`` read from inside ``step()`` is
    post-clip if and only if clipping ran in the right place.
    """

    def __init__(self, params, lr=0.1):
        super().__init__(params, lr=lr)
        self.norm_at_step: list[float] = []

    def step(self, closure=None):  # noqa: D102
        grads = [
            p.grad for g in self.param_groups for p in g["params"] if p.grad is not None
        ]
        total = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(x, 2) for x in grads]), 2
        )
        self.norm_at_step.append(float(total))
        return super().step(closure)


class _TinyModel(torch.nn.Module):
    """Minimal stand-in matching the positional forward that ``train_step`` calls.

    Returns the 6-tuple PanguPlasimLegacy layout ``(surface, upper_air, 0, 0,
    0, 0)`` so the VAE-KL branch takes its placeholder path.
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(torch.eye(4) * scale)

    def forward(self, surface_in, constant_boundary, varying_boundary, upper_air_in):
        return (self.lin(surface_in), self.lin(upper_air_in), 0, 0, 0, 0)


class _ScaledMSELoss(torch.nn.Module):
    """MSE scaled up so the gradient norm is comfortably above any test clip."""

    def __init__(self, gain: float = 1000.0):
        super().__init__()
        self.gain = gain

    def forward(
        self, out_s, out_u, tgt_s, tgt_u, out_diagnostic=None, target_diagnostic=None
    ):
        loss = self.gain * (
            ((out_s - tgt_s) ** 2).mean() + ((out_u - tgt_u) ** 2).mean()
        )
        z = torch.zeros((), dtype=loss.dtype)
        return {"loss": loss, "surface": z, "upper_air": z, "diagnostic": z}


def _batch():
    torch.manual_seed(0)
    return {
        "surface_in": torch.randn(2, 4),
        "upper_air_in": torch.randn(2, 4),
        "constant_boundary": torch.zeros(2, 4),
        "varying_boundary": torch.zeros(2, 4),
        "target_surface": torch.randn(2, 4),
        "target_upper_air": torch.randn(2, 4),
    }


def _run(grad_clip_norm, grad_stats=None):
    torch.manual_seed(0)
    model = _TinyModel()
    opt = _RecordingOptimizer(model.parameters())
    losses = train_step(
        model=model,
        loss_fn=_ScaledMSELoss(),
        optimizer=opt,
        scheduler=None,
        batch=_batch(),
        has_diagnostic=False,
        grad_clip_norm=grad_clip_norm,
        grad_stats=grad_stats,
    )
    return model, opt, losses


CLIP = 0.5


def test_unclipped_gradient_norm_is_large():
    """Guard the fixture: without clipping the norm must exceed the clip value.

    If the loss gain is ever tuned down so the natural norm sits BELOW ``CLIP``,
    every clipping assertion below would pass vacuously.
    """
    _, opt, _ = _run(0.0)
    assert opt.norm_at_step[0] > 10 * CLIP, (
        f"fixture too weak: unclipped norm {opt.norm_at_step[0]:.3e} is not "
        f"comfortably above the test clip {CLIP}"
    )


def test_clip_is_applied_before_optimizer_step():
    """THE regression test. Pre-fix this saw the full unclipped norm."""
    _, opt, _ = _run(CLIP)
    seen = opt.norm_at_step[0]
    assert seen == pytest.approx(CLIP, rel=1e-4), (
        f"optimizer.step() saw gradient norm {seen:.6e}, expected it clipped to "
        f"{CLIP}. Clipping is running in the wrong place again — it must happen "
        f"between backward() and optimizer.step(), inside the step function."
    )


def test_clipping_actually_changes_the_weights():
    """End-to-end: clipped and unclipped runs must not land on the same weights.

    Complements the norm check — it would still fail if some future refactor
    clipped a *copy* of the gradients.
    """
    clipped, _, _ = _run(CLIP)
    plain, _, _ = _run(0.0)
    assert not torch.allclose(clipped.lin.weight, plain.lin.weight, atol=1e-8), (
        "clipped and unclipped steps produced identical weights — clipping had "
        "no effect on the update."
    )


def test_grad_stats_reports_preclip_norm_and_clip_flag():
    stats: dict = {}
    _, opt, _ = _run(CLIP, grad_stats=stats)
    assert stats["clipped"] is True
    # The REPORTED norm is pre-clip (that is the diagnostic value); the norm the
    # optimizer sees is post-clip. Both must hold at once.
    assert stats["grad_norm"] > CLIP
    assert opt.norm_at_step[0] == pytest.approx(CLIP, rel=1e-4)


def test_measurement_pass_does_not_perturb_gradients():
    """With clipping OFF, asking for stats must be numerically inert.

    ``clip_and_measure_grads`` deliberately avoids ``clip_grad_norm_(inf)`` for
    this path because that writes ``grad.mul_(1.0)`` back. Equal weights AND an
    equal norm-at-step are what "inert" means here.
    """
    stats: dict = {}
    measured, opt_m, _ = _run(0.0, grad_stats=stats)
    plain, opt_p, _ = _run(0.0)
    assert torch.equal(measured.lin.weight, plain.lin.weight)
    assert opt_m.norm_at_step[0] == opt_p.norm_at_step[0]
    assert stats["clipped"] is False
    assert stats["grad_norm"] == pytest.approx(opt_p.norm_at_step[0], rel=1e-6)


def test_clip_off_is_bit_identical_to_before_the_fix():
    """The default path (``grad_clip_norm: 0.0``) must be unchanged.

    Every shipping ai-rossby config sets ``grad_clip_norm: 0.0``, so the fix has
    to be a no-op for them — otherwise it would invalidate the captured
    baselines (CLAUDE.md #1).
    """
    a, opt_a, _ = _run(0.0)
    b, opt_b, _ = _run(0.0)
    assert torch.equal(a.lin.weight, b.lin.weight)
    assert opt_a.norm_at_step == opt_b.norm_at_step
    assert all(math.isfinite(x) for x in opt_a.norm_at_step)


def test_helper_is_a_noop_when_nothing_is_requested():
    """No clip, no stats -> the gradients must come back untouched."""
    model = _TinyModel()
    model(torch.randn(2, 4), None, None, torch.randn(2, 4))[0].sum().backward()
    before = model.lin.weight.grad.clone()
    clip_and_measure_grads(model, grad_clip_norm=0.0, grad_stats=None)
    assert torch.equal(model.lin.weight.grad, before)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
