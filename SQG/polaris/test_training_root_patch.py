"""Login-node string test of the SQG_TRAINING_ROOT patch (PROFILING_PROMPT.md §3.1a) -- no jax, no torch, no GPU.

The solver itself cannot be imported on a CPU (it raises "JAX is not using the allocated GPU"), so this checks the text
transform only; the GPU half (§3.1b: the patched import, its self-hash, AB3 level 2) is sqg_env_probe.py.
Checks, for BOTH solver harnesses (profile_solver.py and profile_overlap.py are standalone copies):
  * the patch changes exactly one line, the TRAINING_ROOT line, and nothing else (same line count);
  * the two copies of patch_training_root give identical output, and the result still compiles;
  * with the SQG_* env vars unset, the module-level defaults are Midway's (solver path, batch 32, buffer 8192);
  * a source without exactly one TRAINING_ROOT line is rejected.
  python3 SQG/polaris/test_training_root_patch.py      # any python >= 3.7 (compile needs `from __future__ import annotations`)
Prints PATCH_TEST_OK.
"""
import hashlib
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SQG = os.path.dirname(HERE)
SOLVER = os.path.join(SQG, "fixed_ring", "A-SQG_data.py")
MIDWAY_SOLVER = "/project/pedramh/weidong/SQG/fixed_ring/A-SQG_data.py"
ROOT = "/eagle/projects/lighthouse-uchicago/members/someone/sqg_stage/AllTrainData"


def load(name):
    """A harness as a module, with the SQG_* overrides unset (its top level imports only the stdlib)."""
    saved = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith("SQG_")}
    try:
        spec = importlib.util.spec_from_file_location(name, os.path.join(SQG, "analysis", name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.environ.update(saved)
    return mod


def main():
    with open(SOLVER, "rb") as fh:
        raw = fh.read()
    with open(os.path.join(SQG, "fixed_ring", "SHA256SUMS")) as fh:
        sums = dict(line.split()[::-1] for line in fh if line.strip())
    assert hashlib.sha256(raw).hexdigest() == sums["A-SQG_data.py"], "vendored solver does not match SHA256SUMS"
    src = importlib.util.decode_source(raw)
    before = src.split("\n")

    outs = {}
    for name in ("profile_solver", "profile_overlap"):
        mod = load(name)
        assert mod.ASQG_PY == MIDWAY_SOLVER, f"{name}: default solver path {mod.ASQG_PY}"
        patched = mod.patch_training_root(src, ROOT)
        after = patched.split("\n")
        assert len(after) == len(before), f"{name}: line count {len(before)} -> {len(after)}"
        changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
        assert len(changed) == 1, f"{name}: {len(changed)} lines changed"
        i = changed[0]
        assert before[i].startswith('TRAINING_ROOT = Path("/project/pedramh/'), f"{name}: changed {before[i]!r}"
        assert after[i] == f'TRAINING_ROOT = Path(r"{ROOT}")', f"{name}: wrote {after[i]!r}"
        compile(patched, SOLVER, "exec", dont_inherit=True)
        try:
            mod.patch_training_root(src.replace("TRAINING_ROOT = ", "TRAINING_ROOT_X = ", 1), ROOT)
            raise AssertionError(f"{name}: accepted a source with no TRAINING_ROOT line")
        except RuntimeError:
            pass
        outs[name] = (patched, i + 1)
        if name == "profile_overlap":
            assert (mod.BATCH, mod.BUF) == (32, 8192), f"defaults batch {mod.BATCH} buf {mod.BUF}"
    assert outs["profile_solver"] == outs["profile_overlap"], "the two harness copies disagree"
    print(f"PATCH_TEST_OK: one line changed (line {outs['profile_solver'][1]}, TRAINING_ROOT), both harnesses agree, "
          f"patched source compiles; Midway defaults intact (python {sys.version.split()[0]})")


if __name__ == "__main__":
    main()
