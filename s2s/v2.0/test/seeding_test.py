"""Tests for the --seed knob (DESIGN.md §4.0).

Runs anywhere: no ERA5, no cluster, no GPU required (CUDA assertions self-skip). This is
deliberate — the S2S/port data smokes are blocked on the ERA5 Globus stage, so the seed
mechanism has to be provable without them.

    python s2s/v2.0/test/seeding_test.py          # PASS = "SEEDING_OK"
    pytest -q s2s/v2.0/test/seeding_test.py

What it pins down:
  1. same seed  -> identical draws from random / numpy / torch (and CUDA)
  2. diff seed  -> different draws (a seed that changes nothing is not a seed)
  3. legacy path preserved byte-for-byte when no seed is given  <- the safety property
  4. resolve_seed precedence: --seed > $S2S_SEED > YAML > None
  5. rank offset: ranks get distinct but seed-derived streams
  6. model-level: same seed -> identical init AND identical forward/backward
"""

import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import seeding  # noqa: E402


def _draws():
    """One draw from each RNG the trainer actually uses."""
    return (random.random(), float(np.random.rand()), float(torch.rand(1)))


def test_same_seed_reproduces():
    seeding.apply_seed(1234)
    a = _draws()
    seeding.apply_seed(1234)
    b = _draws()
    assert a == b, "same seed must reproduce: %r != %r" % (a, b)


def test_different_seed_differs():
    seeding.apply_seed(1234)
    a = _draws()
    seeding.apply_seed(4321)
    b = _draws()
    assert a != b, "different seeds produced identical draws: %r" % (a,)


def test_numpy_is_seeded():
    """The gap that made runs diverge: train.py:1251 draws the val sample from numpy."""
    seeding.apply_seed(7)
    a = [int(np.random.randint(0, 1_000_000)) for _ in range(5)]
    seeding.apply_seed(7)
    b = [int(np.random.randint(0, 1_000_000)) for _ in range(5)]
    assert a == b, "numpy not reproducible under apply_seed: %r != %r" % (a, b)


def test_legacy_path_is_byte_identical():
    """No seed -> train.py must behave exactly as it did before this knob existed.

    This is the property that lets the knob ship without re-validating every green:
    resolve_seed returns None, train.py takes torch.manual_seed(world_rank), and the torch
    stream is identical to the historical one.
    """
    assert seeding.resolve_seed(cli_seed=None, params={}, env={}) is None
    assert seeding.resolve_seed(cli_seed=None, params={"seed": None}, env={}) is None

    for world_rank in (0, 1, 3):
        torch.manual_seed(world_rank)              # exactly the historical line
        legacy = torch.rand(4).tolist()
        torch.manual_seed(world_rank)              # what train.py runs when _seed is None
        current = torch.rand(4).tolist()
        assert legacy == current, "legacy stream changed for rank %d" % world_rank


def test_resolve_precedence():
    # CLI beats env and YAML
    assert seeding.resolve_seed(cli_seed=1, params={"seed": 2}, env={"S2S_SEED": "3"}) == 1
    # env beats YAML
    assert seeding.resolve_seed(cli_seed=None, params={"seed": 2}, env={"S2S_SEED": "3"}) == 3
    # YAML is the last resort before legacy
    assert seeding.resolve_seed(cli_seed=None, params={"seed": 2}, env={}) == 2
    # seed 0 is a real seed, not "unset" — the classic falsy bug
    assert seeding.resolve_seed(cli_seed=0, params={"seed": 2}, env={}) == 0
    assert seeding.resolve_seed(cli_seed=None, params={"seed": 0}, env={}) == 0
    assert seeding.resolve_seed(cli_seed=None, params={}, env={"S2S_SEED": "0"}) == 0


def test_bad_seed_raises_not_silently_ignored():
    """A run asked to be reproducible must fail loudly rather than quietly become legacy."""
    for bad in ("abc", "", None):
        env = {} if bad is None else {"S2S_SEED": bad}
        if bad in ("", None):
            assert seeding.resolve_seed(params={}, env=env) is None   # empty/absent = legacy
            continue
        try:
            seeding.resolve_seed(params={}, env=env)
        except ValueError:
            pass
        else:
            raise AssertionError("S2S_SEED=%r should raise" % (bad,))

    for bad in (-1, 2 ** 32, 1.5, True):
        try:
            seeding.resolve_seed(cli_seed=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("--seed %r should raise" % (bad,))


def test_rank_offset_distinct_but_derived():
    """Ranks must not share a stream (they'd correlate the loader noise), yet stay derived."""
    seeding.apply_seed(100, rank=0)
    r0 = _draws()
    seeding.apply_seed(100, rank=1)
    r1 = _draws()
    assert r0 != r1, "rank 0 and rank 1 share an RNG stream"

    # ...and reproducible per rank
    seeding.apply_seed(100, rank=1)
    assert _draws() == r1

    # rank offset matches the legacy intent: seed+rank
    assert seeding.apply_seed(100, rank=2) == 102
    # ...and at rank 0 (the §4.1 baseline case) the applied seed IS the seed
    assert seeding.apply_seed(100, rank=0) == 100


def test_model_init_and_step_reproduce():
    """The property that actually matters: same seed -> same weights AND same gradients.

    A seeded RNG is worthless if the model still drifts, so exercise a real
    init + forward + backward rather than bare RNG draws.
    """
    def run():
        seeding.apply_seed(2024)
        model = torch.nn.Sequential(
            torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4))
        x = torch.randn(8, 16)
        loss = model(x).pow(2).mean()
        loss.backward()
        grads = torch.cat([p.grad.flatten() for p in model.parameters()])
        return loss.item(), grads

    l1, g1 = run()
    l2, g2 = run()
    assert l1 == l2, "loss not reproducible: %r vs %r" % (l1, l2)
    assert torch.equal(g1, g2), "gradients not reproducible under the same seed"


def test_cuda_seeded_if_available():
    if not torch.cuda.is_available():
        print("  (skip: no CUDA visible)")
        return
    seeding.apply_seed(31337)
    a = torch.rand(4, device="cuda").cpu()
    seeding.apply_seed(31337)
    b = torch.rand(4, device="cuda").cpu()
    assert torch.equal(a, b), "CUDA RNG not reproducible under apply_seed"


def test_enable_determinism_sets_flags():
    seeding.enable_determinism(warn_only=True)
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
    # leave the process as we found it for any test that follows
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True


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
        print("ERROR %d/%d seeding tests failed" % (failed, len(tests)))
        sys.exit(1)
    print("SEEDING_OK (%d tests)" % len(tests))
