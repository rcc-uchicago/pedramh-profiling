#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drift guard for the per-epoch telemetry. Static analysis only.

``epoch_telemetry.py`` exists in TWO trees — ``PanguWeather/v2.0/utils/`` and
``physicsnemo_ai_rossby/examples/weather/ai_rossby/`` — because the two projects
share code by COPY, not by import (DESIGN §2c), and putting the repo root on
either job's ``PYTHONPATH`` would risk the ``utils``/``networks``/``config``
collision CLAUDE.md warns about. Duplication is the safe choice here; silent
divergence between the copies is not.

What this pins, and why each one is a real failure mode:

* **The copies are identical.** If one drifts, the two harnesses' rows stop
  meaning the same thing while still looking comparable — the exact failure the
  CHANGELOG 2026-08-05 post-mortem is about.
* **The column list.** Same cross-project contract as the bench CSV's 19
  columns (CLAUDE.md #10): a renamed or reordered column silently invalidates
  every row already collected.
* **All four hooks are wired into BOTH production trainers.** A telemetry object
  that is constructed but never stepped writes nothing and still exits 0.
* **The measured window spans the same work on both sides.** PanguWeather closes
  it right after ``scheduler.step()`` and BEFORE the per-iteration RMSE/wandb
  diagnostics, which have no counterpart inside ai-rossby's window; ai-rossby
  closes it after ``ema.update``. Get this wrong and the comparison measures
  different things at identical column names.
* **It is opt-in.** Unset knob => the training path is byte-identical.

**Text/AST only — no torch, no physicsnemo, no GPU**, so it is safe on a login
node (CLAUDE.md #3). Runtime behaviour is proven by the smoke job's
``EPOCH_TELEMETRY epoch=...`` line, not here.

Run::

    python3.12 epoch_telemetry_test.py

PASS = ``EPOCH_TELEMETRY_TEST_OK (<n> tests)``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
PANGU_MOD = REPO / "PanguWeather/v2.0/utils/epoch_telemetry.py"
AR_MOD = REPO / "physicsnemo_ai_rossby/examples/weather/ai_rossby/epoch_telemetry.py"
PANGU_TRAIN = REPO / "PanguWeather/v2.0/train.py"
AR_TRAIN = REPO / "physicsnemo_ai_rossby/examples/weather/ai_rossby/train.py"

# The contract. Written out literally rather than imported from the module, so a
# rename shows up as a diff in THIS file too and cannot be nodded through.
EXPECTED_COLUMNS = [
    "timestamp", "harness", "git_sha", "run_name", "host", "epoch", "n_gpus",
    "batch_per_gpu", "amp_dtype", "n_loaders", "n_steps", "step_med_ms",
    "step_p90_ms", "step_mean_ms", "step_std_ms", "samples_per_s",
    "epoch_wall_s", "gpu_busy_frac", "peak_mem_gb", "ema_active", "lr",
]

HOOKS = ("epoch_start", "step_start", "step_end", "epoch_end")


def _read(p: Path) -> str:
    assert p.exists(), f"missing: {p}"
    return p.read_text()


def _columns(src: str) -> list:
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "COLUMNS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("COLUMNS not found")


def _line_of(src: str, needle: str) -> int:
    for i, line in enumerate(src.splitlines()):
        if needle in line:
            return i
    raise AssertionError(f"not found: {needle!r}")


# --- the two copies -------------------------------------------------------


def test_copies_are_identical():
    assert _read(PANGU_MOD) == _read(AR_MOD), (
        "the two epoch_telemetry.py copies have diverged; change both in one commit"
    )


def test_columns_match_the_contract():
    assert _columns(_read(PANGU_MOD)) == EXPECTED_COLUMNS


def test_module_is_env_prefixed_not_hardcoded():
    """The module must take its knob prefix as an argument.

    A hardcoded PANGU_/AI_ROSSBY_ prefix is what would force the two copies to
    differ, which is what test_copies_are_identical is protecting.
    """
    src = _read(PANGU_MOD)
    assert '{prefix}_EPOCH_TELEMETRY' in src
    assert "PANGU_EPOCH_TELEMETRY\"" not in src


# --- wiring into both trainers --------------------------------------------


def test_pangu_wires_all_hooks():
    src = _read(PANGU_TRAIN)
    assert "from utils.epoch_telemetry import EpochTelemetry" in src
    for h in HOOKS:
        assert f"_epoch_telemetry.{h}(" in src, f"PanguWeather never calls {h}"


def test_ai_rossby_wires_all_hooks():
    src = _read(AR_TRAIN)
    assert "from epoch_telemetry import EpochTelemetry" in src
    for h in HOOKS:
        assert f"telemetry.{h}(" in src, f"ai-rossby never calls {h}"


def test_harness_labels_are_distinct():
    assert 'harness="panguweather"' in _read(PANGU_TRAIN)
    assert 'harness="ai_rossby"' in _read(AR_TRAIN)


# --- the measured window spans the same work ------------------------------


def test_pangu_window_excludes_per_iteration_diagnostics():
    """step_end must close before the RMSE/wandb block, after scheduler.step().

    That block is PanguWeather-only per-iteration diagnostics with no
    counterpart inside ai-rossby's window. Inside the window it would inflate
    Pangu's step time against a harness that never pays it — a difference that
    would read as "ai-rossby is faster".
    """
    src = _read(PANGU_TRAIN)
    end = _line_of(src, "self._epoch_telemetry.step_end()")
    sched = _line_of(src, "if self.params.scheduler in ['OneCycleLR'")
    diagnostics = _line_of(src, "# Skipped under BENCH, mirroring s2s")
    assert sched < end < diagnostics, (
        f"step_end at line {end} must sit between scheduler.step() ({sched}) "
        f"and the diagnostics block ({diagnostics})"
    )


def test_ai_rossby_window_includes_the_ema_sweep():
    """step_end must close AFTER ema.update, before the logging call.

    On a 1.18 B-param model the EMA sweep is ~4.7 GB of elementwise traffic per
    step — real hot-path work, and one of the things the epoch-1..6 window
    exists to measure.
    """
    src = _read(AR_TRAIN)
    ema = _line_of(src, "ema.update(inner_model, epoch=global_epoch)")
    end = _line_of(src, "telemetry.step_end()")
    log = _line_of(src, "log.log_minibatch(")
    assert ema < end < log, (
        f"step_end at line {end} must sit between ema.update ({ema}) and "
        f"log_minibatch ({log})"
    )


def test_epoch_end_precedes_validation_on_both():
    """The row must cover TRAINING only — validation scope differs by harness."""
    ar = _read(AR_TRAIN)
    assert _line_of(ar, "telemetry.epoch_end(") < _line_of(
        ar, "# --- Validation (optional) ---"
    )
    pg = _read(PANGU_TRAIN)
    # PanguWeather's epoch_end sits in train_one_epoch, which returns before the
    # caller runs validation at all — assert it is inside that method.
    assert _line_of(pg, "self._epoch_telemetry.epoch_end(") > _line_of(
        pg, "def train_one_epoch"
    )


# --- opt-in ---------------------------------------------------------------


def test_disabled_unless_knob_is_set():
    """An unset knob must mean 'off', and off must reach every method."""
    src = _read(PANGU_MOD)
    assert 'os.environ.get(f"{prefix}_EPOCH_TELEMETRY") == "1"' in src
    # Every public method routes through the enabled/_recording guard.
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "EpochTelemetry")
    for name in HOOKS:
        fn = next(n for n in cls.body
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        guards = [
            n for n in ast.walk(fn)
            if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "_recording"
        ] + [
            n for n in ast.walk(fn)
            if isinstance(n, ast.Attribute) and n.attr == "enabled"
        ]
        assert guards, f"{name} does not check the enabled/_recording guard"


def test_no_per_step_cuda_synchronize():
    """A sync in the step path would change the number being measured.

    It removes CPU/GPU overlap across the step boundary — the timing equivalent
    of a Heisenbug. Events are drained in chunks instead; the only
    ``synchronize`` allowed is on an event, in ``_drain``.
    """
    tree = ast.parse(_read(PANGU_MOD))
    # AST, not a substring scan: the module docstring says the words
    # "torch.cuda.synchronize()" while explaining why it does not call it.
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and ast.unparse(n.func) == "torch.cuda.synchronize"
    ]
    assert not calls, "torch.cuda.synchronize must not appear anywhere here"

    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "EpochTelemetry")
    for name in ("step_start", "step_end"):
        fn = next(n for n in cls.body
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        syncs = [
            n for n in ast.walk(fn)
            if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "synchronize"
        ]
        assert not syncs, f"{name} must not synchronize"


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
        print("ERROR EPOCH_TELEMETRY_TEST_FAILED (%d/%d)" % (failed, len(tests)))
        sys.exit(1)
    print("EPOCH_TELEMETRY_TEST_OK (%d tests)" % len(tests))
