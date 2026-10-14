#!/usr/bin/env python3
"""Structural tests for §4.9's dhconv layout fix (`PANGU_DHCONV_XIO`).

AST-only — no torch import, no GPU, so it runs on a login node (CLAUDE.md #3). It pins
the two traps that were actually hit while writing the change, both of which fail
*silently* or *late*:

1. **TorchScript cannot read `os.environ`.** `_contract_dhconv` is `@torch.jit.script`,
   so a knob check inside it fails at import — not at first use. Hence two scripted
   variants selected by an unscripted caller.
2. **`torch.randn` into a reordered shape changes the initial weights.** It consumes the
   same RNG draws either way but places them at different positions, so the §4.12 gate
   would fail for a reason that has nothing to do with layout — and would read as "the
   optimization changes what the model computes".

Run: python3 test/dhconv_layout_test.py      PASS token: DHCONV_LAYOUT_OK
"""
import ast
import sys
from pathlib import Path

# `factorizations.py` contains `f"{implementation=}"` — the f-string debug specifier,
# **Python 3.8+** — so it cannot even be PARSED by the Polaris login node's python3
# (3.6.15). Say so plainly instead of dying with a confusing SyntaxError three frames
# deep. This is the fifth 3.6 landmine recorded in this repo.
if sys.version_info < (3, 8):
    sys.exit("ERROR DHCONV_LAYOUT_NEEDS_PY38: this test AST-parses factorizations.py, "
             "which uses a 3.8+ f-string. Run it with the ai-rossby venv:\n"
             "  $AI_ROSSBY_VENV/bin/python test/dhconv_layout_test.py")

NET = Path(__file__).resolve().parents[1] / "networks" / "modulus_sfno"


def _tree(name):
    return ast.parse((NET / name).read_text())


def _scripted_fns(tree):
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef):
            for d in n.decorator_list:
                if "jit.script" in ast.dump(d):
                    out[n.name] = n
    return out


def test_both_einsum_variants_exist_and_differ_only_in_operand_order():
    src = (NET / "contractions.py").read_text()
    assert '"bixy,iox->boxy"' in src, "original contraction missing"
    assert '"bixy,xio->boxy"' in src, "x-major variant missing"


def test_knob_is_never_read_inside_a_scripted_function():
    """TorchScript cannot compile os.environ — this fails at IMPORT, not at first use."""
    for fname in ("contractions.py", "factorizations.py"):
        tree = _tree(fname)
        for name, fn in _scripted_fns(tree).items():
            body = ast.dump(fn)
            assert "dhconv_weight_is_xio" not in body, (
                "%s.%s is @torch.jit.script and reads the knob — TorchScript cannot "
                "compile os.environ" % (fname, name))
            assert "environ" not in body, "%s.%s reads os.environ under jit.script" % (
                fname, name)


def test_dispatch_happens_in_an_UNscripted_caller():
    tree = _tree("factorizations.py")
    scripted = _scripted_fns(tree)
    callers = [n.name for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef)
               and "dhconv_weight_is_xio" in ast.dump(n)]
    assert callers, "nothing selects between the two variants"
    for c in callers:
        assert c not in scripted, "%s dispatches but is scripted" % c


def test_randn_is_never_called_with_the_reordered_shape():
    """The init trap: same draws, different positions => different initial weights."""
    tree = _tree("s2convolutions.py")
    randn_calls = [n for n in ast.walk(tree)
                   if isinstance(n, ast.Call)
                   and "randn" in ast.dump(n.func)]
    assert randn_calls, "expected a torch.randn for the weight"
    for call in randn_calls:
        dumped = ast.dump(call)
        assert "permute" not in dumped, (
            "torch.randn is being called on a permuted/reordered shape — the draw must "
            "happen in the ORIGINAL order and be permuted afterwards")


def test_the_permute_is_applied_and_made_contiguous():
    src = (NET / "s2convolutions.py").read_text()
    assert ".permute(2, 0, 1, 3).contiguous()" in src, (
        "the [i,o,x,2] -> [x,i,o,2] permute must be materialised contiguous, or the "
        "stored parameter is still i-major and the fix does nothing")


def test_default_is_OFF():
    """Flipping it changes the shape of 95.8% of parameters; every checkpoint breaks."""
    src = (NET / "contractions.py").read_text()
    assert 'os.environ.get("PANGU_DHCONV_XIO", "0")' in src, "default must be off"


if __name__ == "__main__":
    fails = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("  ok    %s" % name)
            except AssertionError as e:
                fails.append(name)
                print("  FAIL  %s: %s" % (name, e))
    print("DHCONV_LAYOUT_OK (6 tests)" if not fails
          else "ERROR DHCONV_LAYOUT %d/6 failed" % len(fails))
    sys.exit(1 if fails else 0)
