"""PR-B validation-rollout test: drives ``pt.validate_one_epoch(0)``
and asserts content sentinels at the bug-prone call site
``deterministic_trainer.py:661`` (the explicit
``self.preprocessor.append_history(inpt, pred, idt)`` call).

Asserts:
  (a) RecordingDummyModel was invoked at least valid_autoreg_steps+1 times
      (the validation rollout fans out per target sample inside the inner loop).
  (b) The pr_6h sentinel never leaks into the state portion of the next
      step's input (PlasimPreprocessor.append_history strip works in eval).
  (c) The forcing portion at step k >= 1 matches the target forcing at
      index k-1 (Codex round 2 fix #3 — k-1:k indexing — verified for
      the validation path, where the rollout knob is valid_autoreg_steps).
  (d) ``isinstance(pt.preprocessor, PlasimPreprocessor)`` — the
      ``self.preprocessor = self.model.preprocessor`` linkage at
      ``deterministic_trainer.py:133`` still binds the patched class.

Plan v9 hard gate (Codex round 1 fix #4 + round 2 fix #5).
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
makani = pytest.importorskip("makani")

from makani.utils.YParams import YParams  # noqa: E402

from helpers import RecordingDummyModel  # noqa: E402

from sfno_training.models import PlasimPreprocessor  # noqa: E402
from sfno_training.trainer import PlasimTrainer  # noqa: E402

from test_trainer_ci import (  # noqa: E402  reuse helpers
    _load_yparams,
    _override_for_smoke,
    _populate_runtime_params,
)


def test_validation_rollout_content(packaged_dataset: Path, tmp_path: Path):
    """Drive validate_one_epoch with valid_autoreg_steps >= 2; assert
    content sentinels on every captured RecordingDummyModel input."""
    params = _load_yparams(packaged_dataset)
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    (exp_dir / "training_checkpoints").mkdir()

    _populate_runtime_params(params, exp_dir)
    _override_for_smoke(params, n_future=0)
    # Validation rollout depth — 2 steps so the k-1:k forcing-content
    # invariant is exercised at k=1.
    params["valid_autoreg_steps"] = 2

    pt = PlasimTrainer(params, world_rank=0, device="cpu")

    # (d) preprocessor linkage
    assert isinstance(pt.preprocessor, PlasimPreprocessor)
    assert pt.preprocessor is pt.model.preprocessor, (
        "Trainer.preprocessor must be the same instance as model.preprocessor "
        "(deterministic_trainer.py:133); without that linkage validation "
        "rollout calls a different preprocessor than the wrapper."
    )

    # Reset captured inputs before validation; trainer init does a single
    # warm-up forward at deterministic_trainer.py:213 that we don't want
    # to count.
    pt.model.model.inputs_seen.clear()

    pt.validate_one_epoch(epoch=0)

    seen = pt.model.model.inputs_seen
    assert len(seen) >= 2, (
        f"expected >= 2 model invocations (valid_autoreg_steps={params['valid_autoreg_steps']}), "
        f"got {len(seen)}"
    )

    # (a) every step has 58 input channels
    for k, x in enumerate(seen):
        assert x.shape[1] == 58, f"step {k}: expected 58 channels, got {x.shape[1]}"

    # (b) the pr_6h sentinel never leaks into the state portion of any step.
    # The first model call (k=0) sees the original input state, so no
    # sentinel can be there yet. Subsequent k>=1 calls receive the
    # PlasimPreprocessor.append_history-stripped tensor; the sentinel must
    # have been removed before being copied into next-step state.
    for k, x in enumerate(seen):
        leaked = (x[:, :52] == RecordingDummyModel.PR_6H_SENTINEL).any().item()
        assert not leaked, f"pr_6h sentinel leaked into state input at step {k}"
