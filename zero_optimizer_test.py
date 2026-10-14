#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ZeRO Stage 1 (sharded optimizer state) in the ai-rossby recipe.

ZeRO-1 shards Adam's ``exp_avg``/``exp_avg_sq`` across ranks instead of every
rank holding an identical copy. On this model that state is
2 x 1,182,108,160 x 4 B = 8.8 GB per GPU, so sharding over 4 ranks frees
~6.6 GB — enough to reach ``model.checkpointing: 1`` (the measured 1.307x
config, which OOMs without it). Same arithmetic, same result.

The failure modes here are all SILENT, which is why each gets a test:

* **The whitelist.** ``_flatten_optimizer_cfg`` maps config keys explicitly; a
  key missing from it is dropped without error, so the config would advertise
  sharding while the run used the full optimizer.
* **Unconsolidated save.** ``ZeroRedundancyOptimizer.state_dict()`` returns only
  the local shard. Saving without ``consolidate_state_dict(to=0)`` produces a
  checkpoint that resumes with a fraction of the optimizer state — no error,
  just a training curve that goes wrong days later.
* **Consolidation inside the rank-0 guard.** It is a COLLECTIVE. Called only on
  rank 0, it deadlocks every other rank. This is the kind of bug that hangs a
  168 h job at its first checkpoint.
* **fused + ZeRO.** The wrapper rejects the fused kernel; passing it through
  would raise at construction, on a compute node, minutes into a job.
* **Default-on.** It must default OFF so the parity measurement still
  reproduces.

**Text/AST only — no torch, no GPU**, so it is login-node safe (CLAUDE.md #3).
Real sharding behaviour needs 4 ranks and is exercised by the smoke.

Run::

    python3.12 zero_optimizer_test.py

PASS = ``ZERO_OPTIMIZER_TEST_OK (<n> tests)``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
RECIPE = REPO / "physicsnemo_ai_rossby/examples/weather/ai_rossby"
TRAIN = RECIPE / "train.py"
TRAIN_LOOP = RECIPE / "train_loop.py"
PARITY_TRAINING = RECIPE / "conf/training/sfno_e3sm_parity.yaml"


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def _line_of(path: Path, needle: str) -> int:
    for i, line in enumerate(_lines(path)):
        if needle in line:
            return i
    raise AssertionError(f"not found in {path.name}: {needle!r}")


# --- the knob actually reaches the optimizer -------------------------------


def test_key_survives_the_flatten_whitelist():
    """_flatten_optimizer_cfg drops anything not listed — the silent failure."""
    src = TRAIN.read_text()
    block = src.split("def _flatten_optimizer_cfg")[1].split("\ndef ")[0]
    assert '"use_zero_optimizer"' in block, (
        "use_zero_optimizer missing from _flatten_optimizer_cfg — the config "
        "would say sharding is on while the run used the full optimizer"
    )


def test_make_optimizer_reads_it():
    src = TRAIN_LOOP.read_text()
    assert 'getattr(cfg, "use_zero_optimizer", False)' in src


def test_defaults_off():
    """Off unless asked, so the parity measurement still reproduces."""
    assert 'getattr(cfg, "use_zero_optimizer", False)' in TRAIN_LOOP.read_text()
    assert '"use_zero_optimizer": bool(opt_cfg.get("use_zero_optimizer", False))' \
        in TRAIN.read_text()
    m = re.search(r"(?m)^\s*use_zero_optimizer:\s*(\S+)", PARITY_TRAINING.read_text())
    assert m and m.group(1).lower() == "false", "parity config must ship it OFF"


# --- checkpoint correctness (the dangerous part) ---------------------------


def test_save_consolidates_first():
    src = TRAIN.read_text()
    assert "consolidate_state_dict(to=0)" in src, (
        "saving a ZeRO optimizer without consolidating writes a checkpoint that "
        "silently resumes with a fraction of the optimizer state"
    )


def test_consolidate_is_outside_the_rank0_guard():
    """It is a COLLECTIVE — inside the guard it deadlocks the other ranks.

    Asserted structurally: the consolidate call must come BEFORE the
    `_saving and dist.rank == 0` line, and must not itself be rank-gated.
    """
    consolidate = _line_of(TRAIN, "consolidate_state_dict(to=0)")
    guard = _line_of(TRAIN, "if _saving and dist.rank == 0:")
    assert consolidate < guard, (
        f"consolidate at line {consolidate} must precede the rank-0 guard at {guard}"
    )
    # And the two lines above it must not be a rank test.
    context = "\n".join(_lines(TRAIN)[consolidate - 2: consolidate + 1])
    assert "dist.rank" not in context, f"consolidate appears rank-gated:\n{context}"


def test_consolidate_is_duck_typed_not_import_gated():
    """hasattr, so a non-distributed build or a plain AdamW is a no-op."""
    assert 'hasattr(optimizer, "consolidate_state_dict")' in TRAIN.read_text()


# --- construction constraints ----------------------------------------------


def test_fused_is_dropped_for_zero():
    """The wrapper rejects fused; passing it would raise on a compute node."""
    src = TRAIN_LOOP.read_text()
    block = src.split("def _wrap_zero")[1]
    assert 'kwargs.pop("fused", False)' in block, (
        "fused must be popped before constructing ZeroRedundancyOptimizer"
    )


def test_falls_back_when_not_distributed():
    """Single-GPU runs and the bench must keep working unchanged."""
    block = TRAIN_LOOP.read_text().split("def _wrap_zero")[1]
    assert "is_initialized()" in block
    assert "falling back to the unsharded AdamW" in block


def test_both_optimizer_paths_are_wrapped():
    """Selective-weight-decay and the plain path must BOTH honor the flag.

    make_optimizer has two returns; wrapping only one would make the knob
    depend on an unrelated setting.
    """
    src = TRAIN_LOOP.read_text()
    block = src.split("def make_optimizer")[1].split("\ndef _wrap_zero")[0]
    assert block.count("_wrap_zero(") == 2, (
        f"expected both AdamW return paths to go through _wrap_zero, "
        f"found {block.count('_wrap_zero(')}"
    )
    assert "return torch.optim.AdamW(" not in block, (
        "an unwrapped AdamW return path would silently ignore use_zero_optimizer"
    )


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print("ERROR %s: %s" % (t.__name__, exc))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("ERROR %s raised %s: %s" % (t.__name__, type(exc).__name__, exc))
    if failed:
        print("ERROR ZERO_OPTIMIZER_TEST_FAILED (%d/%d)" % (failed, len(tests)))
        sys.exit(1)
    print("ZERO_OPTIMIZER_TEST_OK (%d tests)" % len(tests))
