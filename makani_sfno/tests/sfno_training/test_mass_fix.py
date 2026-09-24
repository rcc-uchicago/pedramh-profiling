"""Tests for the dry-air mass fix (src/sfno_training/models/mass_fix.py) and its
wiring into PlasimPreprocessor. DIAGNOSTIC feature; see the module docstring.

Needs torch + makani: run on a compute node (polaris_climate_screen_test.pbs).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("makani")

from sfno_training.models import mass_fix as M  # noqa: E402
from sfno_training.models import preprocessor as P  # noqa: E402

NAMES = ["PS", "T", "TMQ", "U", "PRECT"]        # 4 state + 1 diagnostic
H, W = 6, 12
MU = np.array([98000.0, 280.0, 25.0, 0.0, 1e-5])
SD = np.array([5000.0, 20.0, 15.0, 10.0, 1e-5])


def _fix():
    return M.DryAirFix(channel_names=NAMES, global_means=MU, global_stds=SD, nlat=H)


def _gm_dry(x_z):
    """Independent reference: area mean of PS - g*TMQ in Pa, per batch member."""
    w = M.equiangular_weights(H)
    x = x_z.detach().to(torch.float64).numpy()
    ps = x[:, 0] * SD[0] + MU[0]
    q = x[:, 2] * SD[2] + MU[2]
    return (ps - M.GRAVITY * q).mean(axis=-1) @ w


def _pair(seed=0, B=2, c_out=5):
    g = torch.Generator().manual_seed(seed)
    inp = torch.randn(B, 4, H, W, generator=g, dtype=torch.float64)
    pred = inp.clone()
    pred = torch.cat([pred, torch.zeros(B, 1, H, W, dtype=torch.float64)], 1)[:, :c_out]
    pred = pred + 0.1 * torch.randn(pred.shape, generator=g, dtype=torch.float64)
    pred[:, 0] -= 0.2        # a 1000 Pa mass loss the fix must remove
    return inp, pred


def test_fix_restores_input_dry_air_exactly():
    inp, pred = _pair()
    assert np.abs(_gm_dry(pred) - _gm_dry(inp)).max() > 500       # the seeded loss is there
    out = _fix()(inp, pred)
    np.testing.assert_allclose(_gm_dry(out), _gm_dry(inp), atol=1e-6, rtol=0)


def test_fix_moves_only_ps_and_uniformly():
    inp, pred = _pair(1)
    out = _fix()(inp, pred)
    diff = (out - pred)
    assert torch.equal(diff[:, 1:], torch.zeros_like(diff[:, 1:]))   # TMQ, T, U, PRECT untouched
    d = diff[:, 0]
    assert torch.allclose(d, d[:, :1, :1].expand_as(d), atol=1e-12)  # one shift per member
    assert not torch.allclose(d[0], d[1])                             # computed per member


def test_fix_sign_adds_mass_back_when_prediction_lost_it():
    inp, pred = _pair(2)
    out = _fix()(inp, pred)
    assert bool((out[:, 0] > pred[:, 0]).all())


def test_fix_keeps_bf16_and_stays_close():
    inp, pred = _pair(3)
    out = _fix()(inp.to(torch.bfloat16), pred.to(torch.bfloat16))
    assert out.dtype == torch.bfloat16
    # bf16 PS spacing at |z|~1 is ~2^-8 * 5000 Pa ~ 20 Pa per point; the global
    # mean after rounding is far tighter than the 1000 Pa the fix removes
    assert np.abs(_gm_dry(out) - _gm_dry(inp.to(torch.bfloat16))).max() < 20.0


def test_fix_is_differentiable():
    inp, pred = _pair(4)
    pred.requires_grad_(True)
    _fix()(inp, pred)[:, 0].sum().backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_fix_refuses_missing_channel_and_bad_grid():
    with pytest.raises(ValueError, match="TMQ"):
        M.DryAirFix(channel_names=["PS", "T"], global_means=MU, global_stds=SD, nlat=H)
    inp, pred = _pair(5)
    with pytest.raises(ValueError, match="nlat"):
        M.DryAirFix(channel_names=NAMES, global_means=MU, global_stds=SD, nlat=H + 1)(inp, pred)


def test_weights_match_the_evaluation_package():
    from sfno_ensemble.scores import equiangular_weights
    np.testing.assert_array_equal(M.equiangular_weights(180), equiangular_weights(180))


# ---------------------------------------------------------------------------
# PlasimPreprocessor wiring (stock Preprocessor2D methods stubbed to identity)
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_base(monkeypatch):
    base = P.Preprocessor2D
    monkeypatch.setattr(base, "append_unpredicted_features", lambda self, inp, target=False: inp)
    monkeypatch.setattr(base, "history_denormalize", lambda self, xn, target=False: xn)


def _pp(fix):
    pp = P.PlasimPreprocessor.__new__(P.PlasimPreprocessor)
    torch.nn.Module.__init__(pp)
    pp.n_state_channels = 4
    pp.n_full_out_channels = 5
    pp.dry_air_fix = fix
    pp._fix_inp_state = None
    return pp


def test_preprocessor_flag_off_is_a_pass_through(stub_base):
    pp = _pp(None)
    inp, pred = _pair(6)
    assert pp.append_unpredicted_features(inp) is inp
    assert pp.history_denormalize(pred, target=True) is pred      # bitwise: same object
    assert pp._fix_inp_state is None


def test_preprocessor_flag_on_uses_the_cached_step_input(stub_base):
    pp = _pp(_fix())
    inp, pred = _pair(7)
    pp.append_unpredicted_features(inp)
    out = pp.history_denormalize(pred, target=True)
    np.testing.assert_allclose(_gm_dry(out), _gm_dry(inp), atol=1e-6, rtol=0)
    assert pp._fix_inp_state is None                               # consumed once
    # input-side denormalize (target=False) is never corrected
    assert pp.history_denormalize(pred, target=False) is pred


def test_preprocessor_flag_on_without_input_fails_loudly(stub_base):
    pp = _pp(_fix())
    _, pred = _pair(8)
    with pytest.raises(RuntimeError, match="cached input"):
        pp.history_denormalize(pred, target=True)


def test_build_from_params(tmp_path):
    np.save(tmp_path / "m.npy", MU.reshape(1, -1, 1, 1))
    np.save(tmp_path / "s.npy", SD.reshape(1, -1, 1, 1))
    prm = SimpleNamespace(channel_names=NAMES, global_means_path=str(tmp_path / "m.npy"),
                          global_stds_path=str(tmp_path / "s.npy"), img_crop_shape_x=H,
                          img_shape_x=H, h_parallel_size=1, w_parallel_size=1)
    f = P._build_dry_air_fix(prm)
    assert (f.i_ps, f.i_q) == (0, 2) and f.sd_ps == 5000.0
    prm.h_parallel_size = 2
    with pytest.raises(ValueError, match="all-reduce"):
        P._build_dry_air_fix(prm)
    assert P._pget({"conserve_dry_air": True}, "conserve_dry_air") is True
    assert P._pget(SimpleNamespace(), "conserve_dry_air", False) is False
