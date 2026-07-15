"""Tests for the S2S_BENCH / NVTX harness ported into PanguWeather (DESIGN.md §2c).

Runs anywhere: no E3SM data, no cluster, no GPU, no torch import of train.py (which would
need the whole model stack). Everything here is static analysis of the source, which is the
right tool for the failure mode being guarded: **instrumentation drift**. A dropped or
renamed NVTX range, or a reordered CSV column, does not crash anything — it silently
invalidates every comparison and breaks parse_nsys.py (CLAUDE.md #10). A runtime smoke
cannot see that; a schema test can.

    python PanguWeather/v2.0/test/bench_instrumentation_test.py   # PASS = "BENCH_INSTR_OK"
    pytest -q PanguWeather/v2.0/test/bench_instrumentation_test.py

What it pins down:
  1. PanguWeather's CSV columns == s2s's 19, same names, same ORDER, with the two
     loader_wait_* columns APPENDED (never inserted) — so a positional reader still works
  2. the NVTX range names are byte-identical to s2s's (the names ARE the contract)
  3. parse_nsys.py knows every range the harness emits — an emitted-but-unparsed range is
     a measurement nobody ever sees
  4. every NVTX/BENCH statement is GATED — this is the safety property that lets the port
     ship without re-validating the GREEN 0.3411 smoke: knobs unset => legacy path
  5. the bench defaults match s2s's (20/80), so the METHOD is comparable even though A100
     and H100-NVL numbers are not
"""

import ast
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PANGU_TRAIN = os.path.join(_HERE, os.pardir, "train.py")
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir))
_S2S_TRAIN = os.path.join(_REPO, "s2s", "v2.0", "train.py")
_PARSE_NSYS = os.path.join(_REPO, "s2s", "v2.0", "HPC_scripts", "parse_nsys.py")

# The 19 columns s2s/v2.0/train.py has always written, in order. Duplicated here on
# purpose: if someone edits BOTH trees to drop a column, test 1 (which diffs the two
# sources) would still pass. This literal is the independent anchor.
S2S_COLUMNS = [
    "timestamp", "git_sha", "run_num", "n_gpus", "batch_per_gpu", "amp_dtype",
    "ddp_find_unused", "n_loaders", "step_med", "step_p90", "step_mean", "step_std",
    "cpu_prep_med", "compute_med", "cpu_prep_frac", "samples_per_s",
    "peak_mem_gb_max_rank", "scaler_skips", "n_steps_counted",
]
APPENDED_COLUMNS = ["loader_wait_med", "loader_wait_frac"]

# Ranges the harness emits with a literal (non-f-string) name. `step_{N}` is an f-string
# and is matched by prefix in parse_nsys.py, so it is handled separately.
EXPECTED_RANGES = {
    "to_ensemble_batch", "data_prep", "forward_loss", "backward", "optimizer", "ema",
}
# 'ema' is PanguWeather-only (s2s has no EMA); the rest must match s2s byte-for-byte.
SHARED_RANGES = EXPECTED_RANGES - {"ema"}


def _read(path):
    with open(path) as fh:
        return fh.read()


def _tree(path):
    return ast.parse(_read(path))


def _find_func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError("function %s not found" % name)


def _row_columns(path):
    """The key order of the `row = {...}` dict literal inside _bench_finalize."""
    fn = _find_func(_tree(path), "_bench_finalize")
    for node in ast.walk(fn):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and any(getattr(t, "id", None) == "row" for t in node.targets)):
            keys = [k.value for k in node.value.keys]
            assert all(isinstance(k, str) for k in keys), "non-literal CSV column key"
            return keys
    raise AssertionError("no `row = {...}` dict found in _bench_finalize of %s" % path)


def _pushed_ranges(path):
    """Literal names passed to nvtx.range_push(...) — f-strings excluded."""
    names = set()
    fstrings = 0
    for node in ast.walk(_tree(path)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "range_push"):
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            names.add(arg.value)
        elif isinstance(arg, ast.JoinedStr):
            fstrings += 1
    return names, fstrings


def test_csv_columns_match_s2s_exactly():
    pangu = _row_columns(_PANGU_TRAIN)
    assert pangu[:len(S2S_COLUMNS)] == S2S_COLUMNS, (
        "CSV schema drifted from s2s. Expected the first %d columns to be s2s's, in order.\n"
        "  got:      %s\n  expected: %s" % (len(S2S_COLUMNS), pangu[:len(S2S_COLUMNS)], S2S_COLUMNS))


def test_s2s_itself_still_writes_those_columns():
    # Guards the other direction: if s2s's schema changes, the anchor above is stale and
    # this port's "identical columns" claim quietly stops being true.
    assert _row_columns(_S2S_TRAIN) == S2S_COLUMNS, (
        "s2s/v2.0/train.py's CSV schema changed — update S2S_COLUMNS here AND check every "
        "committed bench_results.csv still parses.")


def test_new_columns_are_appended_not_inserted():
    pangu = _row_columns(_PANGU_TRAIN)
    assert pangu[len(S2S_COLUMNS):] == APPENDED_COLUMNS, (
        "loader_wait_* must be APPENDED after s2s's 19 so positional readers survive; got %s"
        % (pangu[len(S2S_COLUMNS):],))


def test_nvtx_range_names_are_byte_identical_to_s2s():
    pangu, pangu_f = _pushed_ranges(_PANGU_TRAIN)
    s2s, _ = _pushed_ranges(_S2S_TRAIN)
    assert pangu == EXPECTED_RANGES, (
        "NVTX ranges drifted.\n  got:      %s\n  expected: %s" % (sorted(pangu), sorted(EXPECTED_RANGES)))
    missing = SHARED_RANGES - s2s
    assert not missing, (
        "these ranges are supposed to be shared with s2s but s2s no longer emits them: %s"
        % sorted(missing))
    assert pangu_f >= 1, "the per-step f-string range (step_{N}) is missing"


def test_parse_nsys_parses_every_range_we_emit():
    src = _read(_PARSE_NSYS)
    # to_ensemble_batch is emitted but not summarised by the parser in EITHER tree (it is
    # init-only in PanguWeather), so it is not required here.
    for name in sorted(EXPECTED_RANGES - {"to_ensemble_batch"}):
        assert "'%s'" % name in src, (
            "parse_nsys.py does not know the '%s' range — it would be emitted into every "
            "trace and summarised in none of them." % name)
    assert "step_%" in src, "parse_nsys.py lost its step_%% prefix query"


def test_every_nvtx_and_bench_statement_is_gated():
    """The safety property: with S2S_BENCH/S2S_NVTX unset the legacy path is unchanged.

    This is what lets the harness land without re-validating the GREEN loss-0.3411 smoke
    (job 7252271 / 7253591). An ungated nvtx.range_push or an ungated timing sync would
    silently put profiling overhead into every production run.
    """
    tree = _tree(_PANGU_TRAIN)
    gated, ungated = 0, []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.guards = []

        def visit_If(self, node):
            self.guards.append(ast.unparse(node.test))
            for child in node.body:
                self.visit(child)
            self.guards.pop()
            # `else` of `if BENCH:` is NOT a bench guard — walk it with the guard removed.
            for child in node.orelse:
                self.visit(child)

        def visit_Call(self, node):
            nonlocal gated
            f = node.func
            is_nvtx = isinstance(f, ast.Attribute) and f.attr in ("range_push", "range_pop")
            is_prof = isinstance(f, ast.Attribute) and f.attr in (
                "cudaProfilerStart", "cudaProfilerStop")
            if is_nvtx or is_prof:
                if any(g in ("NVTX", "BENCH") or "NVTX" in g or "BENCH" in g
                       for g in self.guards):
                    gated += 1
                else:
                    ungated.append("line %d: %s" % (node.lineno, ast.unparse(node)[:60]))
            self.generic_visit(node)

    for fn_name in ("to_ensemble_batch", "train_one_epoch", "_bench_finalize"):
        Visitor().visit(_find_func(tree, fn_name))

    assert not ungated, (
        "these profiling statements run on the LEGACY path (no S2S_NVTX/S2S_BENCH):\n  %s"
        % "\n  ".join(ungated))
    assert gated >= 8, "expected >=8 gated nvtx/profiler calls, found %d" % gated


def _env_default(path, knob):
    """The literal default in `os.environ.get("<knob>", "<default>")`, via AST.

    Not a string match: s2s writes `os.environ.get("S2S_BENCH_STEPS",  "80")` with padding
    spaces, so a literal search tests formatting rather than the value.
    """
    for node in ast.walk(_tree(path)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == knob):
            return node.args[1].value
    return None


def test_torch_compile_knob_is_actually_wired():
    """The bench scripts advertise TORCH_COMPILE_MODE — it must reach torch.compile.

    It did not. PanguWeather had only a commented-out `torch.compile(..., mode='default')`
    and no env read (DESIGN §2c's table: s2s 2, PanguWeather 0), so the commented-out
    `export TORCH_COMPILE_MODE=...` in the bench scripts was a live trap: uncomment it and
    you get no compile, no error, and a "torch.compile doesn't help this model" conclusion.
    """
    assert _env_default(_PANGU_TRAIN, "TORCH_COMPILE_MODE") is not None, (
        "TORCH_COMPILE_MODE is not read from the environment — the knob the bench scripts "
        "document does nothing")
    fn = _find_func(_tree(_PANGU_TRAIN), "get_model")
    compiles = [n for n in ast.walk(fn)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "compile"]
    assert compiles, "no live torch.compile call in get_model()"
    src = _read(_PANGU_TRAIN)
    assert 'torch.compile(self.model, mode=_compile_mode, fullgraph=False)' in src, (
        "the compile call no longer uses the env-resolved mode / fullgraph=False")


def test_bench_defaults_match_s2s():
    for knob, default in (("S2S_BENCH_WARMUP", "20"), ("S2S_BENCH_STEPS", "80")):
        got = _env_default(_PANGU_TRAIN, knob)
        assert got == default, (
            "%s defaults to %r here vs s2s's %r — the two trees would bench different "
            "windows and their CSVs would not be method-comparable." % (knob, got, default))
        assert _env_default(_S2S_TRAIN, knob) == default, (
            "s2s's %s default changed; re-check this port" % knob)


def test_loop_clock_stops_before_the_profiler_teardown():
    """`elapsed` must be sampled BEFORE cudaProfilerStop()/all_reduce.

    s2s samples it after both, which folds the profiler's buffer flush into the measured
    wall time. Under nsys that flush dwarfs the loop: job 7255503 reported elapsed=51.8s vs
    sum(steps)+sum(waits)=25.7s and the self-check discarded the row — on every profiled
    run. The timers were correct; the clock was stopped in the wrong place.
    """
    fn = _find_func(_tree(_PANGU_TRAIN), "_bench_finalize")
    body = list(ast.walk(fn))

    def line_of(pred):
        return min((n.lineno for n in body if pred(n)), default=None)

    loop_end = line_of(lambda n: isinstance(n, ast.Assign)
                       and any(getattr(t, "id", None) == "loop_end" for t in n.targets))
    prof_stop = line_of(lambda n: isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute)
                        and n.func.attr == "cudaProfilerStop")
    assert loop_end is not None, "_bench_finalize no longer samples loop_end"
    assert prof_stop is not None, "cudaProfilerStop vanished from _bench_finalize"
    assert loop_end < prof_stop, (
        "loop_end (line %d) is sampled AFTER cudaProfilerStop (line %d) — every nsys run "
        "will fold the profiler flush into `elapsed` and refuse its bench row"
        % (loop_end, prof_stop))
    assert "elapsed = loop_end - loop_t0" in _read(_PANGU_TRAIN), (
        "`elapsed` is not derived from loop_end — the ordering above is then pointless")


def test_loader_wait_is_reconciled_not_ignored():
    """The elapsed-vs-sum self-check must account for loader idle.

    s2s reconciles `elapsed` against sum(step_times) alone. That holds only while the
    loader keeps ahead of the GPU; on an input-bound run the between-step fetch is in
    `elapsed` and in no step window, so the check fires and the row is REFUSED — the
    harness would abort exactly when the loader is the finding.
    """
    src = _read(_PANGU_TRAIN)
    assert "expected = sum(step_times) + loader_wait_sum" in src, (
        "the self-check no longer includes loader_wait_sum — an input-bound run will exit 3 "
        "instead of reporting its loader idle")
    assert "sys.exit(3)" in src, "the timer self-disagreement guard was removed"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print("  ok    %s" % t.__name__)
        except AssertionError as e:
            print("  FAIL  %s: %s" % (t.__name__, e)); failed += 1
        except Exception as e:  # noqa: BLE001
            print("  ERROR %s: %s: %s" % (t.__name__, type(e).__name__, e)); failed += 1
    print()
    if failed:
        print("ERROR %d/%d bench-instrumentation tests failed" % (failed, len(tests)))
        sys.exit(1)
    print("BENCH_INSTR_OK (%d tests)" % len(tests))
