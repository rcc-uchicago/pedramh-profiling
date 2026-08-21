#!/usr/bin/env python3
"""Tests for utils/equivalence.py — plan item 18's capture side.

Runs on a login node: `utils.equivalence` imports torch lazily, and only
`test_tensor_stats_*` needs it (skipped when absent). CLAUDE.md #3 forbids importing
torch on a login node for a "quick check", so nothing here does at module scope.

The load-bearing test is `test_end_to_end_compare_baselines_accepts_our_records`: it
feeds two recorder-produced records to the REAL `compare_baselines.py`. A capture
format that our own tests bless but the comparison tool rejects would be worthless,
and that tool is in a different subtree with its own lifecycle.

Run: python3 test/equivalence_test.py     PASS token: EQUIV_RECORDER_OK
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.equivalence import (EquivalenceRecorder, REQUIRED_CONFIG,  # noqa: E402
                               config_sha256, config_sha256_from_env,
                               effective_seed, tensor_stats, validate_record)

REPO = Path(__file__).resolve().parents[3]
COMPARE = REPO / "physicsnemo_ai_rossby" / "polaris" / "compare_baselines.py"


def _py37():
    """An interpreter that can run compare_baselines.py.

    It opens with `from __future__ import annotations`, which is **3.7+**, while the
    Polaris login node's `python3` is **3.6.15** — so the gate tool cannot be run with
    the default interpreter there. Prefer the ai-rossby venv (3.12); it needs no torch,
    since compare_baselines is pure json/argparse.
    """
    if sys.version_info >= (3, 7):
        return sys.executable
    venv = os.environ.get("AI_ROSSBY_VENV")
    for c in ([os.path.join(venv, "bin", "python")] if venv else []) + [
            "/eagle/projects/lighthouse-uchicago/members/mehta5/conda-envs"
            "/ai-rossby-venv/bin/python"]:
        if os.path.exists(c):
            return c
    return None

CFG = dict(seed=0, world_size=1, n_params=1182108160, batch_size=1,
           amp_dtype="bfloat16", config_yaml_sha256="a" * 64, mode="train")


def _rec(path, losses, **over):
    r = EquivalenceRecorder(path=str(path))
    for i, v in enumerate(losses):
        r.record_step(train_batch_loss=v, batch_grad_norm=1.0 + i, batch_grad_max=2.0)
    cfg = dict(CFG)
    cfg.update(over)
    r.finalize(**cfg)
    return json.loads(Path(path).read_text())


class _FakeTensor:
    """Duck-types just enough of a tensor. Keeps this test off torch entirely, which
    CLAUDE.md #3 forbids importing on a login node — even for a quick check."""

    def __init__(self, vals, shape):
        self.vals, self.shape = vals, shape

    def detach(self):
        return self

    def float(self):
        return self

    def numel(self):
        return len(self.vals)

    def _s(self, v):
        return type("S", (), {"item": lambda _s, _v=v: _v})()

    def mean(self):
        return self._s(sum(self.vals) / len(self.vals))

    def std(self):
        m = sum(self.vals) / len(self.vals)
        return self._s((sum((x - m) ** 2 for x in self.vals) / (len(self.vals) - 1)) ** 0.5)

    def min(self):
        return self._s(min(self.vals))

    def max(self):
        return self._s(max(self.vals))


class TestRecorder(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_disabled_unless_env_or_path_given(self):
        # It rides inside the real training step; always-on is instrumentation drift.
        os.environ.pop("PANGU_EQUIV_JSON", None)
        r = EquivalenceRecorder()
        self.assertFalse(r.enabled)
        r.record_step(train_batch_loss=1.0)
        self.assertIsNone(r.finalize(**CFG))
        self.assertEqual(r.trajectory, [])

    def test_env_var_enables_it(self):
        p = os.path.join(self.d, "e.json")
        os.environ["PANGU_EQUIV_JSON"] = p
        try:
            self.assertTrue(EquivalenceRecorder().enabled)
        finally:
            os.environ.pop("PANGU_EQUIV_JSON")

    def test_record_has_every_MUST_MATCH_key(self):
        rec = _rec(os.path.join(self.d, "b.json"), [3.0, 2.5, 2.0])
        for k in REQUIRED_CONFIG:
            self.assertIn(k, rec, "compare_baselines aborts without %s" % k)
        self.assertEqual(rec["steps"], 3)

    def test_unstable_key_set_is_rejected(self):
        # compare_baselines intersects step key sets, so a record that drops a metric
        # mid-run still compares clean on the remainder and prints PASS. A baseline
        # that can pass by omission is worse than no baseline at all.
        r = EquivalenceRecorder(path=os.path.join(self.d, "u.json"))
        r.record_step(train_batch_loss=1.0, batch_grad_norm=1.0)
        r.record_step(train_batch_loss=0.9)                       # dropped a metric
        with self.assertRaises(ValueError) as cm:
            r.finalize(**CFG)
        self.assertIn("EQUIV_UNSTABLE_KEYS", str(cm.exception))

    def test_missing_config_key_is_rejected(self):
        r = EquivalenceRecorder(path=os.path.join(self.d, "m.json"))
        r.record_step(train_batch_loss=1.0)
        bad = dict(CFG)
        del bad["seed"]
        with self.assertRaises(ValueError) as cm:
            r.finalize(**bad)
        self.assertIn("EQUIV_RECORD_INCOMPLETE", str(cm.exception))
        self.assertIn("seed", str(cm.exception))

    def test_non_finite_loss_is_rejected(self):
        r = EquivalenceRecorder(path=os.path.join(self.d, "n.json"))
        r.record_step(train_batch_loss=float("nan"))
        with self.assertRaises(ValueError) as cm:
            r.finalize(**CFG)
        self.assertIn("EQUIV_NON_FINITE", str(cm.exception))

    def test_empty_trajectory_is_rejected(self):
        r = EquivalenceRecorder(path=os.path.join(self.d, "z.json"))
        with self.assertRaises(ValueError) as cm:
            r.finalize(**CFG)
        self.assertIn("EQUIV_RECORD_EMPTY_TRAJECTORY", str(cm.exception))

    def test_missing_config_hash_is_fatal_not_defaulted(self):
        # Two records both carrying "unknown" would compare CLEAN across two genuinely
        # different configs — the apples-to-oranges case MUST_MATCH exists to prevent.
        os.environ.pop("S2S_YAML", None)
        with self.assertRaises(ValueError) as cm:
            config_sha256_from_env()
        self.assertIn("EQUIV_NO_CONFIG", str(cm.exception))

    def test_effective_seed_comes_from_what_seed_torch_actually_set(self):
        os.environ["PYTHONHASHSEED"] = "7"
        try:
            self.assertEqual(effective_seed(), 7)
        finally:
            os.environ.pop("PYTHONHASHSEED")
        with self.assertRaises(ValueError) as cm:
            effective_seed()
        self.assertIn("EQUIV_NO_SEED", str(cm.exception))

    def test_output_stats_are_recorded_and_complete(self):
        # item 18 asks for a trajectory AND output stats; the first baseline
        # (job 7551401) shipped with forward_output_stats EMPTY because the hook
        # recorded only the trajectory. compare_baselines silently compares zero
        # output quantities in that case.
        r = EquivalenceRecorder(path=os.path.join(self.d, "o.json"))
        r.record_step(train_batch_loss=1.0)
        r.record_output("output_surface", _FakeTensor([1.0, 2.0, 3.0, 4.0], [2, 2]))
        r.finalize(**CFG)
        rec = json.loads(Path(os.path.join(self.d, "o.json")).read_text())
        st = rec["forward_output_stats"]["output_surface"]
        self.assertEqual(st["shape"], [2, 2])
        self.assertAlmostEqual(st["mean"], 2.5)
        self.assertEqual(st["min"], 1.0)
        self.assertEqual(st["max"], 4.0)
        for k in ("shape", "mean", "std", "min", "max"):
            self.assertIn(k, st, "compare_baselines reads %s" % k)

    def test_tensor_stats_needs_no_torch(self):
        self.assertNotIn("import torch",
                         (Path(__file__).resolve().parents[1]
                          / "utils" / "equivalence.py").read_text())
        s = tensor_stats(_FakeTensor([0.0, 2.0], [2]))
        self.assertEqual(s["mean"], 1.0)

    def test_config_sha256_changes_with_the_file(self):
        a = os.path.join(self.d, "a.yaml")
        Path(a).write_text("lr: 1\n")
        h1 = config_sha256(a)
        Path(a).write_text("lr: 2\n")
        self.assertNotEqual(h1, config_sha256(a))

    @unittest.skipUnless(COMPARE.exists() and _py37(),
                         "needs compare_baselines.py and a 3.7+ interpreter")
    def test_end_to_end_compare_baselines_accepts_our_records(self):
        base = os.path.join(self.d, "base.json")
        cand = os.path.join(self.d, "cand.json")
        _rec(base, [3.0, 2.5, 2.0])
        _rec(cand, [3.0, 2.5, 2.0])                                # identical
        r = subprocess.run([_py37(), str(COMPARE), base, cand,
                            "--tolerance", "1e-5"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("EQUIVALENCE_OK", r.stdout)
        # ...and that it actually compares a non-trivial number of quantities, not 0.
        self.assertNotIn("quantities  : 0", r.stdout)

    @unittest.skipUnless(COMPARE.exists() and _py37(),
                         "needs compare_baselines.py and a 3.7+ interpreter")
    def test_end_to_end_a_real_drift_is_caught(self):
        base = os.path.join(self.d, "b2.json")
        cand = os.path.join(self.d, "c2.json")
        _rec(base, [3.0, 2.5, 2.0])
        _rec(cand, [3.0, 2.5, 2.02])                               # 1% drift
        r = subprocess.run([_py37(), str(COMPARE), base, cand,
                            "--tolerance", "1e-5"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("EQUIVALENCE_FAILED", r.stdout)


if __name__ == "__main__":
    res = unittest.main(exit=False, verbosity=2).result
    print("EQUIV_RECORDER_OK" if res.wasSuccessful() else "ERROR EQUIV_RECORDER_FAILED")
    sys.exit(0 if res.wasSuccessful() else 1)
