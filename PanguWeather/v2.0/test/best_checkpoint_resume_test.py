"""Regression test for the best_valid_loss / early_stopping_counter resume bug.

Runs anywhere: no E3SM data, no cluster, no GPU, no torch import of train.py (which would
need the whole model stack — networks.pangu, apex, etc.). Everything here is static
analysis of the source, matching this project's own convention for exactly this situation
(see test/bench_instrumentation_test.py) — the failure mode being guarded is silent
STATE LOSS across a resume, which a runtime smoke on a single (non-resumed) launch cannot
see at all, but a structural check on the source can.

The bug (found 2026-08-14 on the live production run, job 7368237 -> 7368539 -> ...):
`best_valid_loss` and `early_stopping_counter` were plain local variables inside train(),
re-initialized to 1.e6 / 0 on every call — i.e. on every process resume, since
restore_checkpoint() runs once in __init__, before train() is ever called, and had no way
to hand these two values back to train()'s local scope. Confirmed on-disk: best_ckpt.tar's
mtime matched the FIRST validated epoch after the most recent resume (epoch 42,
val_loss=0.02690), not the true minimum over the whole run (epoch 27, val_loss=0.026548,
~15 epochs and several resumes earlier) — whose own numbered checkpoint had already
rotated out of the 10-file retention window by the time this was noticed, i.e. unrecoverable.

The fix: both become `self.` attributes, seeded via `getattr(self, ..., default)` in
train() (so a fresh run still starts at 1.e6/0, but a value already set by
restore_checkpoint() survives), persisted in save_checkpoint()'s checkpoint_data dict, and
restored via `checkpoint.get(..., default)` (not `checkpoint[...]`, so a checkpoint saved
BEFORE this fix — which lacks the key entirely — degrades to the old default instead of
raising KeyError) in restore_checkpoint().

    python PanguWeather/v2.0/test/best_checkpoint_resume_test.py   # PASS = "BEST_CKPT_RESUME_OK"
    pytest -q PanguWeather/v2.0/test/best_checkpoint_resume_test.py
"""

import ast
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PANGU_TRAIN = os.path.join(_HERE, os.pardir, "train.py")


def _read(path):
    with open(path) as fh:
        return fh.read()


def _tree(path):
    return ast.parse(_read(path), filename=path)


def _find_method(tree, class_name, method_name):
    """Return the ast.FunctionDef for `class_name.method_name`, or raise."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"{class_name}.{method_name} not found in {_PANGU_TRAIN}")


def _assign_targets(fn_node):
    """Every assignment target name/attribute-path in a function body, as strings.

    'best_valid_loss = 1.e6' -> 'best_valid_loss'
    'self.best_valid_loss = ...' -> 'self.best_valid_loss'
    Only handles the simple Name/Attribute-on-self shapes this file actually uses.
    """
    out = []
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    out.append(tgt.id)
                elif (
                    isinstance(tgt, ast.Attribute)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self"
                ):
                    out.append(f"self.{tgt.attr}")
    return out


def _const_value(node):
    """Literal value of a constant node across Python versions.

    Python >=3.8 represents every literal as ast.Constant(value=...). This login
    node's default python3 is 3.6, which still uses the pre-3.8 split grammar
    (ast.Str for strings, ast.Num for numbers) -- so isinstance(node, ast.Constant)
    is always False there even for a plain 'best_valid_loss' string literal. Returns
    a sentinel object (never equal to any real value) for anything else, so callers
    can compare directly without a separate isinstance guard.
    """
    if isinstance(node, ast.Constant):  # 3.8+
        return node.value
    if isinstance(node, ast.Str):  # <=3.7, string literals
        return node.s
    if isinstance(node, ast.Num):  # <=3.7, numeric literals
        return node.n
    return _NOT_A_CONSTANT


_NOT_A_CONSTANT = object()


def _source_segment(src, node):
    """Best-effort single-line source for an error message.

    Not `ast.get_source_segment` (needs Python >=3.8; this login node's default
    `python3` is 3.6) -- just the line the node starts on, which is enough context
    for the assertion messages below.
    """
    lines = src.splitlines()
    lineno = getattr(node, "lineno", None)
    if lineno is None or lineno < 1 or lineno > len(lines):
        return ""
    return lines[lineno - 1].strip()


def test_train_no_longer_has_a_bare_local_reset():
    """The exact bug: a plain-Name assignment target wipes any restored value.

    If `best_valid_loss` (no `self.` prefix) is EVER an assignment target inside
    train(), a resumed run's restored value is shadowed by a fresh local variable the
    moment train() runs its own init lines, regardless of what restore_checkpoint() did
    in __init__ two calls earlier.
    """
    tree = _tree(_PANGU_TRAIN)
    fn = _find_method(tree, "Trainer", "train")
    targets = _assign_targets(fn)
    assert "best_valid_loss" not in targets, (
        "train() assigns to a bare `best_valid_loss` local -- this is exactly the bug: "
        "it shadows anything restore_checkpoint() set on self, every time train() runs"
    )
    assert "early_stopping_counter" not in targets, (
        "train() assigns to a bare `early_stopping_counter` local -- same bug, second "
        "variable"
    )


def test_train_seeds_best_state_from_self_with_getattr_fallback():
    """The fix's init lines: self.best_valid_loss = getattr(self, 'best_valid_loss', 1.e6).

    train() legitimately assigns to self.best_valid_loss a SECOND time too, inside the
    per-epoch `if is_best:` block (`self.best_valid_loss = valid_logs['valid_loss']`) --
    that one is correct as a plain value assignment and must NOT be flagged. Only the
    FIRST (lowest-lineno) assignment -- the one-time seed at the top of the method -- is
    required to be the getattr-with-self-fallback shape.
    """
    tree = _tree(_PANGU_TRAIN)
    fn = _find_method(tree, "Trainer", "train")
    src = _read(_PANGU_TRAIN)

    first_assign = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not (
            isinstance(tgt, ast.Attribute)
            and isinstance(tgt.value, ast.Name)
            and tgt.value.id == "self"
            and tgt.attr in ("best_valid_loss", "early_stopping_counter")
        ):
            continue
        if tgt.attr not in first_assign or node.lineno < first_assign[tgt.attr].lineno:
            first_assign[tgt.attr] = node

    for attr, node in first_assign.items():
        value = node.value
        is_getattr_on_self = (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "getattr"
            and len(value.args) == 3
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id == "self"
            and _const_value(value.args[1]) == attr
        )
        assert is_getattr_on_self, (
            f"self.{attr} = ... at line {node.lineno} (the first assignment in train(), "
            f"expected to be the resume-safe seed) is not `getattr(self, '{attr}', "
            f"<default>)` -- got: {_source_segment(src, node)!r} -- a resumed value would "
            "not survive re-entering train()"
        )

    assert "best_valid_loss" in first_assign, "train() never assigns self.best_valid_loss at all"
    assert "early_stopping_counter" in first_assign, (
        "train() never assigns self.early_stopping_counter at all"
    )


def test_is_best_comparison_reads_self_attribute():
    """`is_best = valid_logs['valid_loss'] <= self.best_valid_loss`, not a bare local."""
    tree = _tree(_PANGU_TRAIN)
    fn = _find_method(tree, "Trainer", "train")
    src = _read(_PANGU_TRAIN)
    found = False
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "is_best"
        ):
            found = True
            segment = _source_segment(src, node)
            assert "self.best_valid_loss" in segment, (
                f"is_best comparison no longer reads self.best_valid_loss: {segment!r}"
            )
    assert found, "no `is_best = ...` assignment found in train() -- did it get renamed?"


def test_save_checkpoint_persists_both_fields():
    """checkpoint_data must carry 'best_valid_loss' and 'early_stopping_counter' so a
    LATER resume has something to read back."""
    tree = _tree(_PANGU_TRAIN)
    fn = _find_method(tree, "Trainer", "save_checkpoint")
    src = _read(_PANGU_TRAIN)

    dict_node = None
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "checkpoint_data"
            and isinstance(node.value, ast.Dict)
        ):
            dict_node = node.value
            break
    assert dict_node is not None, "save_checkpoint has no `checkpoint_data = {...}` dict literal"

    keys = {_const_value(k) for k in dict_node.keys} - {_NOT_A_CONSTANT}
    assert "best_valid_loss" in keys, (
        f"checkpoint_data is missing 'best_valid_loss' -- keys found: {sorted(keys)} -- "
        f"source: {_source_segment(src, dict_node)!r}"
    )
    assert "early_stopping_counter" in keys, (
        f"checkpoint_data is missing 'early_stopping_counter' -- keys found: {sorted(keys)}"
    )


def test_restore_checkpoint_uses_get_not_subscript():
    """Must be `checkpoint.get('best_valid_loss', 1.e6)`, never `checkpoint['best_valid_loss']`
    -- every checkpoint written before this fix lacks the key entirely, and a bare subscript
    would turn every pre-fix checkpoint's resume into a KeyError crash instead of a graceful
    fallback to the old (buggy but non-fatal) behaviour."""
    tree = _tree(_PANGU_TRAIN)
    fn = _find_method(tree, "Trainer", "restore_checkpoint")
    src = _read(_PANGU_TRAIN)

    seen = {"best_valid_loss": False, "early_stopping_counter": False}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not (
            isinstance(tgt, ast.Attribute)
            and isinstance(tgt.value, ast.Name)
            and tgt.value.id == "self"
            and tgt.attr in seen
        ):
            continue
        value = node.value
        is_dict_get = (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "get"
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "checkpoint"
            and len(value.args) == 2
            and _const_value(value.args[0]) == tgt.attr
        )
        assert is_dict_get, (
            f"self.{tgt.attr} = ... in restore_checkpoint() is not "
            f"checkpoint.get('{tgt.attr}', <default>) -- got: {_source_segment(src, node)!r} "
            "-- a pre-fix checkpoint (no such key) would raise KeyError instead of falling back"
        )
        seen[tgt.attr] = True

    assert seen["best_valid_loss"], "restore_checkpoint() never restores self.best_valid_loss"
    assert seen["early_stopping_counter"], (
        "restore_checkpoint() never restores self.early_stopping_counter"
    )


def test_train_py_still_parses_and_no_orphaned_bare_names_elsewhere():
    """Whole-file sanity: the edited file is syntactically valid, and neither bare name
    survives as an assignment target ANYWHERE in the file (not just inside train()) --
    catches a stray leftover local-variable assignment outside the two methods checked
    above (e.g. in a helper this test doesn't otherwise look at)."""
    tree = _tree(_PANGU_TRAIN)  # ast.parse already raises SyntaxError -> test fails loudly
    bare_targets = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in (
                    "best_valid_loss",
                    "early_stopping_counter",
                ):
                    bare_targets.append((tgt.id, tgt.lineno))
    assert not bare_targets, f"bare (non-self) assignment(s) found outside expected scope: {bare_targets}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print("  ok    %s" % t.__name__)
        except AssertionError as e:
            print("  FAIL  %s: %s" % (t.__name__, e))
            failed += 1
        except Exception as e:  # noqa: BLE001
            print("  ERROR %s: %s: %s" % (t.__name__, type(e).__name__, e))
            failed += 1
    print()
    if failed:
        print("ERROR %d/%d best-checkpoint-resume tests failed" % (failed, len(tests)))
        sys.exit(1)
    print("BEST_CKPT_RESUME_OK (%d tests)" % len(tests))
