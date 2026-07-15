"""Tests for the VAE noise-fixing hook (DESIGN.md §4.0).

Runs anywhere: no E3SM data, no cluster, no GPU (CUDA assertions self-skip). Deliberate —
the pangu_plasim path this hook serves is blocked on PLASIM data that is not staged on
Polaris, so the mechanism has to be provable without it.

    python PanguWeather/v2.0/test/vae_noise_test.py     # PASS = "VAE_NOISE_OK"
    pytest -q PanguWeather/v2.0/test/vae_noise_test.py

What it pins down:
  1. legacy is the DEFAULT and is byte-for-byte torch.randn_like  <- the safety property
  2. FIXED really is fixed: repeated draws are identical, across resets and processes
  3. FIXED does NOT collapse the ensemble — members still differ (this is the failure that
     would silently turn CRPS spread into 0 and make the gate meaningless)
  4. the two encoder sites get INDEPENDENT eps even at identical shape (keyed on site);
     a shape-only key would correlate two draws that are independent in the real model
  5. FIXED survives a changed global RNG stream — which is the entire point: torch.compile
     can consume the global stream differently, and that must not move our eps
  6. GENERATOR gives fresh-but-reproducible draws, also stream-independent
  7. a seed of 0 is honoured (the classic falsy-seed bug)
  8. bad input fails loudly rather than silently falling back to legacy
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import vae_noise  # noqa: E402


def _std(shape=(4, 3)):
    return torch.ones(shape)


def teardown():
    vae_noise.disable()


def test_default_is_legacy_and_matches_randn_like():
    """No mode enabled => the historical global-RNG draw, byte for byte.

    This is what lets the hook ship without re-validating any existing green run.
    """
    vae_noise.disable()
    assert vae_noise.mode() == vae_noise.LEGACY
    std = _std()
    torch.manual_seed(1234)
    got = vae_noise.draw_eps(std)
    torch.manual_seed(1234)
    want = torch.randn_like(std)
    assert torch.equal(got, want), "the default path is no longer torch.randn_like"


def test_legacy_is_stochastic():
    """A 'noise hook' whose legacy path stopped being random would be a silent science
    change. Two draws must differ."""
    vae_noise.disable()
    std = _std()
    assert not torch.equal(vae_noise.draw_eps(std), vae_noise.draw_eps(std))


def test_fixed_repeats_identically():
    vae_noise.enable_fixed(seed=0)
    std = _std()
    a = vae_noise.draw_eps(std, site="encoder1")
    b = vae_noise.draw_eps(std, site="encoder1")
    assert torch.equal(a, b), "FIXED eps changed between draws — it is not fixed"


def test_fixed_survives_reset_and_reseed():
    """reset() drops the cache; the same seed must regenerate the SAME eps. If this fails,
    a baseline and its comparison run would use different noise."""
    vae_noise.enable_fixed(seed=7)
    std = _std()
    a = vae_noise.draw_eps(std, site="encoder1").clone()
    vae_noise.reset()
    b = vae_noise.draw_eps(std, site="encoder1")
    assert torch.equal(a, b)


def test_fixed_is_independent_of_the_global_rng_stream():
    """THE point of the module. torch.compile / FlexAttention can change how many draws the
    global stream serves and in what order; a correct optimization must not move our eps.
    Seeding the global RNG alone does NOT give this property."""
    std = _std()
    vae_noise.enable_fixed(seed=3)
    torch.manual_seed(1)
    a = vae_noise.draw_eps(std, site="encoder1").clone()

    vae_noise.reset()
    torch.manual_seed(999)          # a different global stream ...
    _ = torch.randn(1000)           # ... consumed a different amount
    b = vae_noise.draw_eps(std, site="encoder1")
    assert torch.equal(a, b), "FIXED eps moved when the global RNG stream changed"


def test_fixed_does_not_collapse_the_ensemble():
    """Freezing eps must not make every ensemble member identical.

    Members are folded into the BATCH by to_ensemble_batch, so one frozen tensor still
    gives each member its own value. If this ever failed, CRPS spread would go to 0 and the
    equivalence gate would be comparing a degenerate model while looking perfectly green.
    """
    vae_noise.enable_fixed(seed=0)
    eps = vae_noise.draw_eps(_std((8, 16)), site="encoder1")
    rows = [eps[i] for i in range(eps.shape[0])]
    for i in range(1, len(rows)):
        assert not torch.equal(rows[0], rows[i]), (
            "ensemble member %d got the same eps as member 0 — spread collapsed" % i)


def test_sites_are_independent_at_identical_shape():
    """encoder1 and encoder2 can have the SAME shape. Keyed on shape alone they would share
    eps, correlating draws that are independent in the real model and changing what the KL
    term measures."""
    vae_noise.enable_fixed(seed=0)
    std = _std()
    a = vae_noise.draw_eps(std, site="encoder1")
    b = vae_noise.draw_eps(std, site="encoder2")
    assert a.shape == b.shape
    assert not torch.equal(a, b), "encoder1 and encoder2 received identical eps"


def test_site_seed_is_stable_across_processes():
    """crc32, not hash(): PYTHONHASHSEED randomizes str hashing per process, so a hash()-
    derived seed would give a different 'fixed' eps on every run — silently."""
    vae_noise.enable_fixed(seed=5)
    assert vae_noise._site_seed("encoder1") == vae_noise._site_seed("encoder1")
    assert vae_noise._site_seed("encoder1") != vae_noise._site_seed("encoder2")
    # Pinned literal: recomputed independently of the module's arithmetic.
    import zlib
    assert vae_noise._site_seed("encoder1") == (5 * 1_000_003 + zlib.crc32(b"encoder1")) % (2**63 - 1)


def test_generator_is_fresh_but_reproducible():
    std = _std()
    vae_noise.enable_generator(seed=11)
    first = [vae_noise.draw_eps(std, site="encoder1").clone() for _ in range(3)]
    assert not torch.equal(first[0], first[1]), "GENERATOR must draw fresh noise each call"

    vae_noise.enable_generator(seed=11)      # re-enable => reset => same sequence
    second = [vae_noise.draw_eps(std, site="encoder1").clone() for _ in range(3)]
    for a, b in zip(first, second):
        assert torch.equal(a, b), "GENERATOR sequence is not reproducible for a fixed seed"


def test_generator_is_independent_of_the_global_stream():
    std = _std()
    vae_noise.enable_generator(seed=11)
    torch.manual_seed(1)
    a = vae_noise.draw_eps(std, site="encoder1").clone()

    vae_noise.enable_generator(seed=11)
    torch.manual_seed(4242)
    _ = torch.randn(500)
    b = vae_noise.draw_eps(std, site="encoder1")
    assert torch.equal(a, b), "GENERATOR eps moved with the global RNG stream"


def test_seed_zero_is_honoured():
    """seed=0 is falsy — the classic bug is `if not seed: use_legacy()`."""
    vae_noise.enable_fixed(seed=0)
    assert vae_noise.mode() == vae_noise.FIXED, "seed 0 fell back to legacy"
    assert vae_noise.seed() == 0


def test_different_seeds_give_different_noise():
    """A seed that changes nothing is not a seed."""
    std = _std()
    vae_noise.enable_fixed(seed=0)
    a = vae_noise.draw_eps(std, site="encoder1").clone()
    vae_noise.enable_fixed(seed=1)
    b = vae_noise.draw_eps(std, site="encoder1")
    assert not torch.equal(a, b)


def test_scoped_fixed_restores_previous_mode():
    vae_noise.disable()
    with vae_noise.fixed(seed=2):
        assert vae_noise.mode() == vae_noise.FIXED
    assert vae_noise.mode() == vae_noise.LEGACY, "the context manager leaked FIXED"


def test_bad_input_fails_loudly():
    for bad in ("nonsense", "fixed"):        # unknown mode; and a mode with no seed
        os.environ["PANGU_VAE_NOISE"] = bad
        try:
            vae_noise._resolve_from_env()
        except ValueError:
            pass
        else:
            raise AssertionError("PANGU_VAE_NOISE=%r was accepted silently" % bad)
        finally:
            del os.environ["PANGU_VAE_NOISE"]
    try:
        vae_noise._enable("fixed", None)
    except ValueError:
        pass
    else:
        raise AssertionError("a None seed was accepted")


def test_env_knob_configures_the_mode():
    os.environ["PANGU_VAE_NOISE"] = "fixed:9"
    try:
        vae_noise._resolve_from_env()
        assert vae_noise.mode() == vae_noise.FIXED and vae_noise.seed() == 9
    finally:
        del os.environ["PANGU_VAE_NOISE"]
        vae_noise.disable()


def test_cuda_fixed_matches_cpu_fixed():
    """eps is drawn on the CPU then moved, so a baseline captured on one backend is
    comparable with a re-run on another. Self-skips without CUDA."""
    if not torch.cuda.is_available():
        print("      (no CUDA visible — device-parity assertion skipped)")
        return
    vae_noise.enable_fixed(seed=0)
    cpu = vae_noise.draw_eps(torch.ones(4, 3), site="encoder1")
    vae_noise.reset()
    gpu = vae_noise.draw_eps(torch.ones(4, 3, device="cuda"), site="encoder1")
    assert torch.equal(cpu, gpu.cpu()), "fixed eps differs between CPU and CUDA"


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
        finally:
            teardown()
    print()
    if failed:
        print("ERROR %d/%d vae_noise tests failed" % (failed, len(tests)))
        sys.exit(1)
    print("VAE_NOISE_OK (%d tests)" % len(tests))
