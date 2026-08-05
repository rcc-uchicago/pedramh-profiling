#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drift guard for ai-rossby's bench instrumentation. Static analysis only.

ai-rossby's counterpart to ``PanguWeather/v2.0/test/bench_instrumentation_test.py``.
It asserts that ``profile_train.py`` and ``train_loop.py`` keep emitting the CSV
columns and NVTX range names that ``s2s/v2.0/HPC_scripts/parse_nsys.py`` and every
prior bench row depend on (CLAUDE.md #10). A renamed range is invisible: it still
traces, ``parse_nsys.py`` just silently reports nothing for it, and the comparison
against PanguWeather quietly becomes meaningless.

**AST only — no torch, no physicsnemo, no GPU.** That is deliberate twice over:
it runs in milliseconds in CI, and it is the one piece of this harness that is safe
to run on a login node (``physicsnemo_ai_rossby/CLAUDE.md``: importing physicsnemo
there can core-dump).

Run::

    python polaris/bench_instrumentation_test.py

PASS = ``AI_ROSSBY_BENCH_INSTR_OK (<n> tests)``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_RECIPE = _REPO / "physicsnemo_ai_rossby" / "examples" / "weather" / "ai_rossby"
_PROFILE = _RECIPE / "profile_train.py"
_TRAIN_LOOP = _RECIPE / "train_loop.py"
_S2S_TRAIN = _REPO / "s2s" / "v2.0" / "train.py"
_PARSE_NSYS = _REPO / "s2s" / "v2.0" / "HPC_scripts" / "parse_nsys.py"

# s2s's 19, in order. The anchor for every cross-project comparison.
S2S_COLUMNS = [
    "timestamp", "git_sha", "run_num", "n_gpus", "batch_per_gpu", "amp_dtype",
    "ddp_find_unused", "n_loaders", "step_med", "step_p90", "step_mean",
    "step_std", "cpu_prep_med", "compute_med", "cpu_prep_frac", "samples_per_s",
    "peak_mem_gb_max_rank", "scaler_skips", "n_steps_counted",
]
APPENDED_COLUMNS = ["samples_per_s_wall", "data_idle_frac", "config_sha16", "n_params"]

# Every range ai-rossby emits. Exact-equality anchor, matching PanguWeather's
# `assert pangu == EXPECTED_RANGES` — a subset check alone would let someone add
# an unparsed range without noticing.
EXPECTED_RANGES = {"data_prep", "forward_loss", "backward", "optimizer"}


def _read(p: Path) -> str:
    return p.read_text()


def _list_literal(src: str, name: str) -> list[str]:
    """The string elements of a module-level ``name = [...]`` assignment."""
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return [
                        e.value for e in node.value.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    ]
    raise AssertionError(f"no module-level list literal named {name}")


def _row_keys(src: str) -> list[str]:
    """Keys of the `row = {...}` dict literal, in source order."""
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "row" and isinstance(node.value, ast.Dict):
                    return [
                        k.value for k in node.value.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)
                    ]
    raise AssertionError("no `row = {...}` dict literal found")


def _emitted_ranges(src: str) -> tuple[set[str], int]:
    """(literal range names, count of f-string ranges).

    Covers BOTH forms ai-rossby uses: `nvtx.range_push("x")` in profile_train.py
    and the `_nvtx_range("x")` context manager in train_loop.py.
    """
    names: set[str] = set()
    fstrings = 0
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        is_push = (isinstance(fn, ast.Attribute) and fn.attr == "range_push") or (
            isinstance(fn, ast.Name) and fn.id == "_nvtx_range"
        )
        if is_push and node.args:
            a = node.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                names.add(a.value)
            elif isinstance(a, ast.JoinedStr):
                fstrings += 1
    return names, fstrings


# --------------------------------------------------------------------------


def test_csv_columns_match_s2s_exactly():
    got = _row_keys(_read(_PROFILE))[:len(S2S_COLUMNS)]
    assert got == S2S_COLUMNS, (
        "CSV schema drifted from s2s.\n  got:      %s\n  expected: %s" % (got, S2S_COLUMNS))


def test_declared_column_constant_matches_the_row():
    """The DictWriter fieldnames and the row dict must not disagree.

    They are written in two places in profile_train.py; if they drift the writer
    raises at the very end of a bench — after the GPU time is already spent.
    """
    src = _read(_PROFILE)
    assert _list_literal(src, "S2S_COLUMNS") == S2S_COLUMNS
    assert _list_literal(src, "EXTRA_COLUMNS") == APPENDED_COLUMNS
    assert _row_keys(src) == S2S_COLUMNS + APPENDED_COLUMNS, (
        "row dict and S2S_COLUMNS+EXTRA_COLUMNS disagree — DictWriter would raise")


def test_s2s_itself_still_writes_those_columns():
    """Guards the other direction: if s2s's schema moves, our anchor is stale."""
    for col in S2S_COLUMNS:
        assert '"%s"' % col in _read(_S2S_TRAIN), (
            "s2s/v2.0/train.py no longer writes '%s' — S2S_COLUMNS here is stale" % col)


def test_new_columns_are_appended_not_inserted():
    got = _row_keys(_read(_PROFILE))[len(S2S_COLUMNS):]
    assert got == APPENDED_COLUMNS, (
        "ai-rossby's extra columns must be APPENDED after s2s's 19 so positional "
        "readers survive; got %s" % (got,))


def test_nvtx_range_names_are_exactly_the_expected_set():
    """Exact equality, not subset — an extra unparsed range is also drift."""
    loop_names, _ = _emitted_ranges(_read(_TRAIN_LOOP))
    prof_names, prof_f = _emitted_ranges(_read(_PROFILE))
    combined = loop_names | prof_names
    assert combined == EXPECTED_RANGES, (
        "NVTX ranges drifted.\n  got:      %s\n  expected: %s"
        % (sorted(combined), sorted(EXPECTED_RANGES)))
    assert prof_f >= 1, "the per-step f-string range (step_{N}) is missing"


def test_parse_nsys_parses_every_range_we_emit():
    src = _read(_PARSE_NSYS)
    for name in sorted(EXPECTED_RANGES):
        assert "'%s'" % name in src, (
            "parse_nsys.py does not know the '%s' range — it would be emitted into "
            "every trace and summarised in none of them." % name)
    assert "step_%" in src, "parse_nsys.py lost its step_% prefix query"


def test_every_nvtx_statement_is_gated():
    """The safety property: with the env unset, training is byte-identical.

    This is what lets the train_loop.py edit land without re-validating the green
    smoke (job 7341412). An ungated range_push would add a real CUDA call to the
    production path.
    """
    gated = 0
    for path in (_PROFILE, _TRAIN_LOOP):
        tree = ast.parse(_read(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            test_src = ast.dump(node.test)
            if "NVTX" not in test_src and "BENCH" not in test_src:
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    if isinstance(fn, ast.Attribute) and fn.attr in (
                        "range_push", "range_pop", "cudaProfilerStart", "cudaProfilerStop",
                    ):
                        gated += 1
        # Every push/pop in the file must be inside such an `if`.
        total = sum(
            1 for n in ast.walk(ast.parse(_read(path)))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in ("range_push", "range_pop", "cudaProfilerStart", "cudaProfilerStop")
        )
        assert gated >= total or total == 0, (
            "%s has %d nvtx/profiler calls but only %d are gated on NVTX/BENCH"
            % (path.name, total, gated))
    assert gated >= 6, "expected at least 6 gated nvtx/profiler calls, found %d" % gated


def test_bench_defaults_match_the_other_projects():
    src = _read(_PROFILE)
    assert '"AI_ROSSBY_BENCH_WARMUP", "20"' in src, "warmup default drifted from 20"
    assert '"AI_ROSSBY_BENCH_STEPS", "80"' in src, "steps default drifted from 80"


def test_torch_compile_knob_is_actually_wired():
    """A knob that is read but never applied is worse than no knob."""
    src = _read(_PROFILE)
    assert 'os.environ.get("TORCH_COMPILE_MODE"' in src
    assert "torch.compile(" in src, "TORCH_COMPILE_MODE is read but never applied"


def test_clock_self_check_is_present():
    """A wrong number is worse than no number — the reconciliation must stay."""
    src = _read(_PROFILE)
    assert "BENCH_CLOCK_MISMATCH" in src
    assert "step_sum + loader_sum" in src or "expected = step_sum" in src


def test_bench_is_opt_in():
    """Without AI_ROSSBY_BENCH=1 the script must refuse, not silently train."""
    src = _read(_PROFILE)
    assert 'AI_ROSSBY_BENCH") == "1"' in src
    assert "BENCH_NOT_ENABLED" in src


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
        print("ERROR AI_ROSSBY_BENCH_INSTR_FAILED (%d/%d)" % (failed, len(tests)))
        sys.exit(1)
    print("AI_ROSSBY_BENCH_INSTR_OK (%d tests)" % len(tests))
