"""Unit tests for ``sfno_training.trainer.ema.EMAModel``.

Pure-CPU, torch-only. Verifies the spec in
``docs/2026-05-02_ema_implementation_plan.md`` §7.1 — in particular:

- Karras-style warmup schedule (closed form at t=1, 10, 100, 1000).
- Update math on real and complex parameters (regression guard against
  the v1 bug where ``.float()`` would have dropped the imaginary
  component of ``complex64`` SFNO spectral weights).
- Dtype-aware shadows: complex stays complex, fp16/bf16 ⇒ fp32 shadow.
- ``state_dict``/``load_state_dict`` round-trip + strict-mode key
  mismatches + complex-vs-real cast guard.
- ``applied_to`` swap-and-restore preserves complex weights bit-exactly.
- ``export_model_state`` emits a FULL canonical state_dict with EMA
  values for tracked params and live values for buffers.
- ``sharded_dims_mp`` mirrored from live tensors onto exported tensors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Surface the conftest gate first; if torch is missing the rest aborts cleanly.
torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

# Allow this file to be exercised standalone (e.g. python -m pytest
# tests/sfno_training/test_ema.py from the repo root) without going
# through tests/sfno_training/conftest.py — handy when makani is not
# installed locally. conftest.py prepends src/ to sys.path; replicate.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from sfno_training.trainer.ema import (  # noqa: E402
    EMAModel,
    _get_model_state_dict_prefix,
    _shadow_dtype_for,
)


# ---------------------------------------------------------------------------
# Toy modules
# ---------------------------------------------------------------------------
class _RealModel(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([1.0, 2.0, 3.0], dtype=dtype))


class _ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(
            torch.tensor([1.0 + 1.0j, 2.0 + 2.0j], dtype=torch.complex64)
        )


class _MixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.r = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.c = nn.Parameter(
            torch.tensor([1.0 + 1.0j, 2.0 + 2.0j], dtype=torch.complex64)
        )
        self.register_buffer("buf", torch.tensor([99.0, 100.0]))


# ---------------------------------------------------------------------------
# Decay schedule
# ---------------------------------------------------------------------------
def test_decay_warmup_curve():
    m = _RealModel()
    ema = EMAModel(m, decay=0.999, warmup=True)

    assert ema.decay_t == pytest.approx(1.0 / 10.0)  # step=0
    ema.step = 1
    assert ema.decay_t == pytest.approx(2.0 / 11.0)
    ema.step = 10
    assert ema.decay_t == pytest.approx(11.0 / 20.0)
    ema.step = 100
    assert ema.decay_t == pytest.approx(101.0 / 110.0)
    ema.step = 1000
    assert ema.decay_t == pytest.approx(min(0.999, 1001.0 / 1010.0))
    # Asymptote
    ema.step = 10**8
    assert ema.decay_t == pytest.approx(0.999)


def test_decay_no_warmup():
    m = _RealModel()
    ema = EMAModel(m, decay=0.95, warmup=False)
    for t in (0, 1, 100, 10**6):
        ema.step = t
        assert ema.decay_t == pytest.approx(0.95)


def test_decay_validation_rejects_out_of_range():
    m = _RealModel()
    with pytest.raises(ValueError):
        EMAModel(m, decay=0.0)
    with pytest.raises(ValueError):
        EMAModel(m, decay=1.0)


# ---------------------------------------------------------------------------
# Update math
# ---------------------------------------------------------------------------
def test_update_math_real():
    m = _RealModel()
    ema = EMAModel(m, decay=0.9, warmup=False)  # decay constant = 0.9

    initial = m.w.detach().clone()
    m.w.data.copy_(torch.tensor([5.0, 10.0, 15.0]))
    ema.update(m)

    expected = 0.9 * initial + 0.1 * torch.tensor([5.0, 10.0, 15.0])
    assert torch.allclose(ema._shadow["w"], expected, atol=1e-7)
    assert ema.step == 1

    m.w.data.copy_(torch.tensor([7.0, 14.0, 21.0]))
    ema.update(m)
    expected = 0.9 * expected + 0.1 * torch.tensor([7.0, 14.0, 21.0])
    assert torch.allclose(ema._shadow["w"], expected, atol=1e-7)
    assert ema.step == 2


def test_update_math_complex():
    """Regression guard: ``.float()`` would have dropped imaginary parts.

    The shadow must average BOTH real and imaginary components.
    """
    m = _ComplexModel()
    ema = EMAModel(m, decay=0.8, warmup=False)
    assert ema._shadow["w"].dtype == torch.complex64

    initial = m.w.detach().clone()
    new_val = torch.tensor([3.0 + 4.0j, 5.0 - 2.0j], dtype=torch.complex64)
    m.w.data.copy_(new_val)
    ema.update(m)

    expected = 0.8 * initial + 0.2 * new_val
    assert torch.allclose(ema._shadow["w"], expected, atol=1e-7)
    # Imaginary preserved on both sides.
    assert ema._shadow["w"].imag.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# Dtype invariants
# ---------------------------------------------------------------------------
def test_dtype_invariant_complex_stays_complex():
    m = _ComplexModel()
    ema = EMAModel(m, decay=0.9)
    assert ema._shadow["w"].dtype == torch.complex64
    assert _shadow_dtype_for(m.w) == torch.complex64


def test_dtype_invariant_half_promotes_to_fp32():
    m_fp16 = _RealModel(dtype=torch.float16)
    ema = EMAModel(m_fp16, decay=0.9)
    assert ema._shadow["w"].dtype == torch.float32

    m_bf16 = _RealModel(dtype=torch.bfloat16)
    ema = EMAModel(m_bf16, decay=0.9)
    assert ema._shadow["w"].dtype == torch.float32


def test_dtype_invariant_fp32_stays_fp32():
    m = _RealModel(dtype=torch.float32)
    ema = EMAModel(m, decay=0.9)
    assert ema._shadow["w"].dtype == torch.float32


# ---------------------------------------------------------------------------
# State dict roundtrip
# ---------------------------------------------------------------------------
def test_state_dict_roundtrip_real_and_complex():
    m = _MixedModel()
    ema = EMAModel(m, decay=0.7, warmup=True)

    # Drift the shadow off the seed.
    m.r.data.copy_(torch.tensor([100.0, 200.0]))
    m.c.data.copy_(torch.tensor([10.0 + 5.0j, -4.0 + 8.0j], dtype=torch.complex64))
    ema.update(m)
    ema.update(m)

    sd = ema.state_dict()

    # Round-trip into a fresh EMA on a clean model — all shadow tensors
    # bit-exactly match.
    m2 = _MixedModel()
    ema2 = EMAModel(m2, decay=0.7, warmup=True)
    ema2.load_state_dict(sd, strict=True)
    for k in ema._shadow:
        assert torch.equal(ema._shadow[k], ema2._shadow[k]), f"mismatch at {k!r}"


def test_load_state_dict_strict_key_mismatch():
    m = _MixedModel()
    ema = EMAModel(m, decay=0.9)
    sd = ema.state_dict()
    sd["bogus"] = torch.zeros(2)
    with pytest.raises(RuntimeError, match="unexpected_keys"):
        ema.load_state_dict(sd, strict=True)

    # And a missing key:
    sd2 = ema.state_dict()
    del sd2["r"]
    with pytest.raises(RuntimeError, match="missing_keys"):
        ema.load_state_dict(sd2, strict=True)


def test_load_state_dict_complex_real_mismatch_rejected():
    """Real tensor cannot silently cast into a complex shadow.

    PyTorch's ``.to(complex64)`` succeeds on a real tensor by zeroing
    the imaginary part — that would silently corrupt the shadow.
    EMAModel must reject this BEFORE the cast.
    """
    m = _ComplexModel()
    ema = EMAModel(m, decay=0.9)
    sd = ema.state_dict()
    # Inject a real tensor of identical shape.
    sd["w"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
    with pytest.raises(RuntimeError, match="complex/real mismatch"):
        ema.load_state_dict(sd, strict=True)


def test_load_state_dict_shape_mismatch_rejected():
    m = _RealModel()
    ema = EMAModel(m, decay=0.9)
    sd = ema.state_dict()
    sd["w"] = torch.tensor([1.0, 2.0])  # ndim matches but size differs
    with pytest.raises(RuntimeError, match="shape mismatch"):
        ema.load_state_dict(sd, strict=True)


# ---------------------------------------------------------------------------
# applied_to context manager
# ---------------------------------------------------------------------------
def test_applied_to_swaps_and_restores_real():
    m = _RealModel()
    ema = EMAModel(m, decay=0.5, warmup=False)
    m.w.data.copy_(torch.tensor([10.0, 20.0, 30.0]))
    ema.update(m)
    shadow_snapshot = ema._shadow["w"].clone()

    live_before = m.w.detach().clone()
    with ema.applied_to(m):
        # During the context, params hold the EMA shadow.
        assert torch.allclose(m.w.data, shadow_snapshot.to(m.w.dtype))
    # After the context, live params restored.
    assert torch.equal(m.w.data, live_before)


def test_applied_to_restores_complex_bitexact():
    m = _ComplexModel()
    ema = EMAModel(m, decay=0.5, warmup=False)
    m.w.data.copy_(torch.tensor([10.0 + 1.0j, 20.0 - 4.0j], dtype=torch.complex64))
    ema.update(m)

    live_before = m.w.detach().clone()
    with ema.applied_to(m):
        # In the EMA pass, params hold the shadow values.
        assert m.w.data.is_complex()
    assert torch.equal(m.w.data, live_before)


def test_applied_to_restores_on_exception():
    m = _RealModel()
    ema = EMAModel(m, decay=0.5, warmup=False)
    m.w.data.copy_(torch.tensor([5.0, 10.0, 15.0]))
    ema.update(m)

    live_before = m.w.detach().clone()
    with pytest.raises(RuntimeError, match="boom"):
        with ema.applied_to(m):
            raise RuntimeError("boom")
    assert torch.equal(m.w.data, live_before)


# ---------------------------------------------------------------------------
# export_model_state
# ---------------------------------------------------------------------------
def test_export_model_state_full_dict():
    """The exported dict must contain all live state_dict keys (params +
    buffers), with EMA values substituted for tracked trainable params and
    buffers copied through unchanged. The result must be loadable
    via ``model.load_state_dict(..., strict=True)`` (the §4.3 contract).
    """
    m = _MixedModel()
    ema = EMAModel(m, decay=0.5, warmup=False)
    # Drift the shadow.
    m.r.data.copy_(torch.tensor([100.0, 200.0]))
    m.c.data.copy_(torch.tensor([5.0 + 2.0j, 6.0 - 3.0j], dtype=torch.complex64))
    ema.update(m)

    exported = ema.export_model_state(m)
    live_keys = set(m.state_dict().keys())

    assert set(exported.keys()) == live_keys, (
        f"export_model_state must return all live state_dict keys "
        f"(missing: {live_keys - set(exported.keys())}, "
        f"extra: {set(exported.keys()) - live_keys})"
    )

    # Buffers copied through (live values).
    assert torch.equal(exported["buf"], m.buf)

    # Tracked params replaced with EMA shadow (cast to param's dtype).
    assert torch.allclose(exported["r"], ema._shadow["r"].to(m.r.dtype))
    assert torch.allclose(exported["c"], ema._shadow["c"].to(m.c.dtype))

    # Strict-load contract.
    m2 = _MixedModel()
    m2.load_state_dict(exported, strict=True)


def test_export_model_state_does_not_alias_live_state():
    """Caller may ``torch.save`` the result; it must not be a view onto
    the live module's tensors (mutating the live module after export
    must not change the exported dict)."""
    m = _MixedModel()
    ema = EMAModel(m, decay=0.5, warmup=False)
    exported = ema.export_model_state(m)

    buf_snapshot = exported["buf"].clone()
    m.buf.fill_(-1.0)
    assert torch.equal(exported["buf"], buf_snapshot), (
        "exported buffer aliases live buffer — would corrupt saved checkpoint"
    )


def test_sharded_dims_mp_preserved():
    """When live params/buffers carry ``sharded_dims_mp``, exported
    tensors must too (mirrors driver.py:540-543)."""
    m = _MixedModel()
    # Simulate model parallel: stamp sharded_dims_mp onto the live tensors.
    m.r.sharded_dims_mp = (0,)
    m.c.sharded_dims_mp = (None,)
    m.buf.sharded_dims_mp = (None,)

    ema = EMAModel(m, decay=0.9)
    exported = ema.export_model_state(m)

    assert getattr(exported["r"], "sharded_dims_mp", None) == (0,)
    assert getattr(exported["c"], "sharded_dims_mp", None) == (None,)
    assert getattr(exported["buf"], "sharded_dims_mp", None) == (None,)


# ---------------------------------------------------------------------------
# Wrapper-prefix detection
# ---------------------------------------------------------------------------
def test_prefix_detection_no_wrappers():
    m = _RealModel()
    assert _get_model_state_dict_prefix(m) == ""


def test_prefix_detection_torch_compile_simulated():
    """Simulate ``torch.compile`` by attaching ``_orig_mod`` (the attribute
    pytorch's OptimizedModule uses)."""
    m = _RealModel()

    class _Wrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self._orig_mod = inner

    w = _Wrap(m)
    assert _get_model_state_dict_prefix(w) == "_orig_mod."


# ---------------------------------------------------------------------------
# Update on missing-shadow param raises
# ---------------------------------------------------------------------------
def test_update_with_new_param_raises():
    """If a model adds a new trainable parameter after EMAModel construction,
    update() must raise rather than silently desync — the user is
    expected to reconstruct EMAModel after model surgery."""
    m = _RealModel()
    ema = EMAModel(m, decay=0.9)
    # Hot-add a new param.
    m.extra = nn.Parameter(torch.tensor([1.0, 2.0]))
    with pytest.raises(KeyError, match="extra"):
        ema.update(m)
