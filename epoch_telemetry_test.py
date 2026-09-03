#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drift guard for the per-epoch telemetry. Static analysis only.

``epoch_telemetry.py`` exists in THREE trees — ``PanguWeather/v2.0/utils/``,
``physicsnemo_ai_rossby/examples/weather/ai_rossby/`` and ``ACE2_retrain/`` —
because the projects share code by COPY, not by import (DESIGN §2c), and putting
the repo root on either job's ``PYTHONPATH`` would risk the
``utils``/``networks``/``config`` collision CLAUDE.md warns about. Duplication is
the safe choice here; silent divergence between the copies is not.

ACE2 is wired differently from the other two and deliberately so: ``fme`` is a
vendored upstream tree that this repo does not edit, so there is no ``train.py``
to call the four hooks from. ``ACE2_retrain/ace2_telemetry.py`` monkeypatches
them onto fme at import time instead (the ``ace2_nvtx.py`` pattern). The tests
below therefore assert ACE2's hooks against the INJECTOR, and additionally pin
the two discriminations that injection makes possible to get wrong.

What this pins, and why each one is a real failure mode:

* **The copies are identical.** If one drifts, the two harnesses' rows stop
  meaning the same thing while still looking comparable — the exact failure the
  CHANGELOG 2026-08-05 post-mortem is about.
* **The column list.** Same cross-project contract as the bench CSV's 19
  columns (CLAUDE.md #10): a renamed or reordered column silently invalidates
  every row already collected.
* **All four hooks are wired into all three harnesses.** A telemetry object
  that is constructed but never stepped writes nothing and still exits 0.
* **The measured window spans the same work in all three.** PanguWeather closes
  it right after ``scheduler.step()`` and BEFORE the per-iteration RMSE/wandb
  diagnostics, which have no counterpart inside ai-rossby's window; ai-rossby
  closes it after ``ema.update``; ACE2 closes it after the per-iteration
  ``step_scheduler``, which fme calls after EMA. Get this wrong and the
  comparison measures different things at identical column names.
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
ACE2_MOD = REPO / "ACE2_retrain/epoch_telemetry.py"
PANGU_TRAIN = REPO / "PanguWeather/v2.0/train.py"
AR_TRAIN = REPO / "physicsnemo_ai_rossby/examples/weather/ai_rossby/train.py"
ACE2_INJECTOR = REPO / "ACE2_retrain/ace2_telemetry.py"

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


def _line_of(src: str, needle: str, code_only: bool = False) -> int:
    """First line containing ``needle``; with ``code_only``, skip comment lines.

    ``code_only`` is needed wherever the needle is a statement the surrounding
    prose also names: ai-rossby's train.py explains the timed window in a comment
    that spells ``telemetry.step_end()`` seven lines BEFORE ``step_start``, so a
    naive scan reported step_end at line 1276 and
    ``test_ai_rossby_window_includes_the_ema_sweep`` failed against correctly
    ordered code. A drift guard that fires on a comment is worse than none — it
    trains the reader to ignore it. The section-marker lookups below are the
    opposite case and must keep matching comments.
    """
    for i, line in enumerate(src.splitlines()):
        if code_only and line.lstrip().startswith("#"):
            continue
        if needle in line:
            return i
    raise AssertionError(f"not found: {needle!r}")


# --- the two copies -------------------------------------------------------


def test_copies_are_identical():
    ref = _read(PANGU_MOD)
    for other in (AR_MOD, ACE2_MOD):
        assert ref == _read(other), (
            f"{other.relative_to(REPO)} has diverged from {PANGU_MOD.relative_to(REPO)}; "
            "change every copy in one commit"
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


def test_ace2_wires_all_hooks():
    """ACE2's four hooks live in the injector, not in a train.py it does not own."""
    src = _read(ACE2_INJECTOR)
    assert "from epoch_telemetry import EpochTelemetry" in src
    for h in HOOKS:
        assert f"tel.{h}(" in src, f"ACE2 never calls {h}"


def test_harness_labels_are_distinct():
    assert 'harness="panguweather"' in _read(PANGU_TRAIN)
    assert 'harness="ai_rossby"' in _read(AR_TRAIN)
    assert 'harness="ace2"' in _read(ACE2_INJECTOR)


# --- the measured window spans the same work ------------------------------


def test_pangu_window_excludes_per_iteration_diagnostics():
    """step_end must close before the RMSE/wandb block, after scheduler.step().

    That block is PanguWeather-only per-iteration diagnostics with no
    counterpart inside ai-rossby's window. Inside the window it would inflate
    Pangu's step time against a harness that never pays it — a difference that
    would read as "ai-rossby is faster".
    """
    src = _read(PANGU_TRAIN)
    end = _line_of(src, "self._epoch_telemetry.step_end()", code_only=True)
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
    ema = _line_of(src, "ema.update(inner_model, epoch=global_epoch)", code_only=True)
    end = _line_of(src, "telemetry.step_end()", code_only=True)
    log = _line_of(src, "log.log_minibatch(")
    assert ema < end < log, (
        f"step_end at line {end} must sit between ema.update ({ema}) and "
        f"log_minibatch ({log})"
    )


def _ace2_hook(name: str) -> ast.FunctionDef:
    """The nested wrapper `install()` patches onto fme under `name`."""
    tree = ast.parse(_read(ACE2_INJECTOR))
    install = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "install")
    return next(n for n in ast.walk(install)
                if isinstance(n, ast.FunctionDef) and n.name == name)


def test_ace2_window_excludes_no_grad_batches():
    """`train_on_batch` runs three times per epoch for reasons that are not steps.

    fme calls it from the training loop, from `_log_first_batch_metrics`, and
    from the post-epoch train-evaluation pass; the last two pass
    `NullOptimization` and are forward-only under `no_grad`. Timing them would
    mix a much cheaper batch into the step distribution and pull `step_med_ms`
    down by an amount that moves with `train_evaluation_samples` — a config knob,
    not a property of the hardware.
    """
    src = ast.unparse(_ace2_hook("train_on_batch"))
    assert "NullOptimization" in src, (
        "ACE2's step_start must be gated on the optimization argument, not on a "
        "call counter — the counter breaks when train_evaluation_samples changes"
    )
    assert "tel.step_start()" in src


def test_ace2_window_closes_after_the_scheduler_step():
    """step_end must fire AFTER the wrapped call and only on the per-iteration one.

    After: the window then spans optimizer + EMA + scheduler, matching
    ai-rossby's (which closes after `ema.update`) and PanguWeather's (after
    `scheduler.step()`). Only per-iteration: `step_scheduler` is also called once
    per epoch with the validation loss, and firing there would append a spurious
    step whose duration is a whole validation pass.
    """
    fn = _ace2_hook("step_scheduler")
    src = ast.unparse(fn)
    assert "is_iteration" in src, "step_end is not gated on is_iteration"
    orig = next(i for i, n in enumerate(fn.body)
                if "_orig_step_scheduler" in ast.unparse(n))
    end = next(i for i, n in enumerate(fn.body) if "tel.step_end()" in ast.unparse(n))
    assert orig < end, (
        "step_end fires before the wrapped scheduler call, so the scheduler step "
        "falls outside the window that the other two harnesses include"
    )


def test_ace2_epoch_opens_after_the_first_batch_probe():
    """epoch_start hangs off `subset_loader`, not off the top of the epoch.

    On the first epoch of a run fme calls `_log_first_batch_metrics()`
    (trainer.py:526-528) before the training loop: it pulls one batch off a
    2.4 TB NetCDF and runs a forward pass. Tens of seconds of one-off cost with
    no timed step in it, against a 60-step arm of ~30 s — starting the wall clock
    above it could roughly halve `gpu_busy_frac`, which is the number the whole
    port turns on.
    """
    assert "tel.epoch_start(" in ast.unparse(_ace2_hook("subset_loader"))
    assert "epoch_start(" not in ast.unparse(_ace2_hook("train_one_epoch")), (
        "epoch_start must not fire at the top of train_one_epoch — that puts "
        "_log_first_batch_metrics inside epoch_wall_s"
    )


def test_ace2_epoch_closes_before_the_train_evaluation_pass():
    """epoch_end hangs off `alternate_shuffle`, not off the end of the epoch.

    `Trainer.train_one_epoch` runs a second, forward-only pass over the data
    after training (trainer.py:586-592) to build the train aggregator.
    `alternate_shuffle` (trainer.py:583) is the single production call site
    between the two. Closing the epoch at the end of `train_one_epoch` instead
    would fold that pass into `epoch_wall_s`, so `gpu_busy_frac` — the number
    this harness exists to produce — would drop by a config-dependent amount
    while still carrying the contract's column name.
    """
    src = ast.unparse(_ace2_hook("alternate_shuffle"))
    assert "tel.epoch_end(" in src
    assert "tel.epoch_end(" not in ast.unparse(_ace2_hook("train_one_epoch")), (
        "epoch_end must not fire from train_one_epoch — that includes the "
        "train-evaluation pass in epoch_wall_s"
    )


def test_ace2_banner_is_a_print_not_a_log():
    """Every rank must emit the banner, and fme's logging is rank-0 only.

    Both world-size guards in the scaling parser read that one line, and the
    `ranks_reporting` guard counts how many distinct ranks emitted it. Routing it
    through `logging` would leave that guard reporting 1 on every arm — passing
    by construction on a 1-node run and failing every multi-node one for a reason
    that has nothing to do with the run.
    """
    tree = ast.parse(_read(ACE2_INJECTOR))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_banner")
    src = ast.unparse(fn)
    assert "print(" in src and "ACE2_BANNER" in src
    assert "world_size=%d" in src and "steps_per_epoch=%d" in src
    # AST, not a substring scan: the docstring says the word "logging" while
    # explaining why the banner does not use it.
    calls = [ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)]
    assert not [c for c in calls if c.startswith(("logging", "logger"))], (
        "the banner must not go through logging — fme routes it to rank 0 only"
    )


def test_epoch_end_precedes_validation_on_both():
    """The row must cover TRAINING only — validation scope differs by harness."""
    ar = _read(AR_TRAIN)
    assert _line_of(ar, "telemetry.epoch_end(", code_only=True) < _line_of(
        ar, "# --- Validation (optional) ---"
    )
    pg = _read(PANGU_TRAIN)
    # PanguWeather's epoch_end sits in train_one_epoch, which returns before the
    # caller runs validation at all — assert it is inside that method.
    assert _line_of(pg, "self._epoch_telemetry.epoch_end(", code_only=True) > _line_of(
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
