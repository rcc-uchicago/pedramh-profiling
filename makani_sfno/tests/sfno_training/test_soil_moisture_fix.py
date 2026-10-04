"""Tests for the frozen-soil-moisture fix (src/sfno_training/models/soil_moisture_fix.py)
and its wiring into PlasimPreprocessor, including composition with the dry-air fix.
DIAGNOSTIC feature; see the module docstring.

Needs torch + makani: run on a compute node (polaris_climate_screen_test.pbs).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("makani")

from sfno_training.models import mass_fix as D  # noqa: E402
from sfno_training.models import preprocessor as P  # noqa: E402
from sfno_training.models import soil_moisture_fix as M  # noqa: E402

NAMES = ["PS", "SOILWATER_10CM", "TSOI_10CM", "TMQ", "PRECT"]   # 4 state + 1 diagnostic
H, W = 6, 12
MU = np.array([98000.0, 0.3, 280.0, 25.0, 1e-5])
SD = np.array([5000.0, 0.1, 20.0, 15.0, 1e-5])
THRESH_K = 272.0


def _fix(threshold_k=THRESH_K):
    return M.SoilMoistureFix(channel_names=NAMES, global_means=MU, global_stds=SD,
                              temperature_threshold_k=threshold_k)


def _phys(x_z, i):
    return x_z[:, i] * SD[i] + MU[i]


def _pair(seed=0, B=2, c_out=5):
    g = torch.Generator().manual_seed(seed)
    inp = torch.randn(B, 4, H, W, generator=g, dtype=torch.float64)
    pred = inp.clone()
    pred = torch.cat([pred, torch.zeros(B, 1, H, W, dtype=torch.float64)], 1)[:, :c_out]
    pred = pred + 0.1 * torch.randn(pred.shape, generator=g, dtype=torch.float64)
    return inp, pred


def test_fix_freezes_moisture_where_input_is_below_threshold():
    inp, pred = _pair()
    # force a clean split: half the grid frozen in the input, half not
    tsoi_phys = _phys(inp, 2).clone()
    tsoi_phys[:, :, : W // 2] = THRESH_K - 5.0     # frozen
    tsoi_phys[:, :, W // 2:] = THRESH_K + 5.0      # unfrozen
    inp[:, 2] = (tsoi_phys - MU[2]) / SD[2]
    pred_soil_before = pred[:, 1].clone()
    zero_z = float(_fix()._zero_z)
    out = _fix()(inp, pred)
    frozen_half = out[:, 1, :, : W // 2]
    unfrozen_half = out[:, 1, :, W // 2:]
    # The positivity clamp applies everywhere, including frozen cells (the floor is a
    # physical constraint, not a freeze exemption), so the expectation is the INPUT
    # moisture clamped at the floor, not the raw input value.
    want_frozen = torch.clamp(inp[:, 1, :, : W // 2], min=zero_z)
    np.testing.assert_allclose(frozen_half.numpy(), want_frozen.numpy(), atol=1e-10)
    # unfrozen half is untouched by the freeze (still the pre-clamp prediction, unless negative)
    want_unfrozen = torch.clamp(pred_soil_before[:, :, W // 2:], min=zero_z)
    np.testing.assert_allclose(unfrozen_half.numpy(), want_unfrozen.numpy(), atol=1e-10)


def test_fix_clamps_negative_moisture_to_zero_everywhere():
    inp, pred = _pair(1)
    inp[:, 2] = 10.0   # input TSOI way above threshold in z-space: nothing frozen
    pred[:, 1] = -5.0  # deeply negative moisture prediction, z-space
    out = _fix()(inp, pred)
    phys = _phys(out, 1)
    assert bool((phys >= -1e-9).all())


def test_fix_touches_only_the_moisture_channel():
    inp, pred = _pair(2)
    out = _fix()(inp, pred)
    other = [0, 2, 3, 4]
    assert torch.equal(out[:, other], pred.to(out.dtype)[:, other])


def test_fix_promotes_bf16_to_float32():
    inp, pred = _pair(3)
    out = _fix()(inp.to(torch.bfloat16), pred.to(torch.bfloat16))
    assert out.dtype == torch.float32


def test_fix_keeps_float32_and_float64_dtypes():
    inp, pred = _pair(10)
    assert _fix()(inp.float(), pred.float()).dtype == torch.float32
    assert _fix()(inp, pred).dtype == torch.float64


def test_fix_is_differentiable():
    inp, pred = _pair(4)
    pred.requires_grad_(True)
    _fix()(inp, pred)[:, 1].sum().backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_fix_refuses_missing_channel():
    with pytest.raises(ValueError, match="SOILWATER_10CM"):
        M.SoilMoistureFix(channel_names=["PS", "TSOI_10CM"], global_means=MU, global_stds=SD)
    with pytest.raises(ValueError, match="TSOI_10CM"):
        M.SoilMoistureFix(channel_names=["PS", "SOILWATER_10CM"], global_means=MU, global_stds=SD)


# ---------------------------------------------------------------------------
# channel-subset runs (same pattern as test_mass_fix.py): full-width stats,
# indexed by out_channels
# ---------------------------------------------------------------------------

PACK_NAMES = ["Z3", "PS", "SOILWATER_10CM", "TSOI_10CM", "TMQ", "U", "PRECT"]
PACK_MU = np.array([5500.0, 98000.0, 0.3, 280.0, 25.0, 0.0, 1e-5])
PACK_SD = np.array([300.0, 5000.0, 0.1, 20.0, 15.0, 10.0, 1e-5])
OUT_CH = [1, 2, 3, 4, 6]
assert [PACK_NAMES[i] for i in OUT_CH] == NAMES
assert np.array_equal(PACK_MU[OUT_CH], MU) and np.array_equal(PACK_SD[OUT_CH], SD)


def test_subset_stats_are_read_by_out_channels():
    f = M.SoilMoistureFix(channel_names=NAMES, global_means=PACK_MU, global_stds=PACK_SD,
                           stats_index=OUT_CH)
    assert (f.mu_soil, f.sd_soil, f.mu_tsoi, f.sd_tsoi) == (0.3, 0.1, 280.0, 20.0)
    inp, pred = _pair(11)
    assert torch.equal(f(inp, pred), _fix()(inp, pred))


def test_means_and_stds_of_different_width_are_refused():
    with pytest.raises(ValueError, match="means have 7 channels, stds 5"):
        M.SoilMoistureFix(channel_names=NAMES, global_means=PACK_MU, global_stds=SD,
                          stats_index=OUT_CH)


# ---------------------------------------------------------------------------
# PlasimPreprocessor wiring (stock Preprocessor2D methods stubbed to identity)
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_base(monkeypatch):
    base = P.Preprocessor2D
    monkeypatch.setattr(base, "append_unpredicted_features", lambda self, inp, target=False: inp)
    monkeypatch.setattr(base, "history_denormalize", lambda self, xn, target=False: xn)


def _pp(dry_fix=None, soil_fix=None):
    pp = P.PlasimPreprocessor.__new__(P.PlasimPreprocessor)
    torch.nn.Module.__init__(pp)
    pp.n_state_channels = 4
    pp.n_full_out_channels = 5
    pp.dry_air_fix = dry_fix
    pp.soil_moisture_fix = soil_fix
    pp._fix_inp_state = None
    return pp


def test_preprocessor_both_flags_off_is_a_pass_through(stub_base):
    pp = _pp()
    inp, pred = _pair(6)
    assert pp.append_unpredicted_features(inp) is inp
    assert pp.history_denormalize(pred, target=True) is pred      # bitwise: same object
    assert pp._fix_inp_state is None


def test_preprocessor_soil_fix_alone_uses_the_cached_step_input(stub_base):
    pp = _pp(soil_fix=_fix())
    inp, pred = _pair(7)
    inp[:, 2] = 10.0   # nothing frozen, isolate the positivity clamp
    pp.append_unpredicted_features(inp)
    out = pp.history_denormalize(pred, target=True)
    phys = out[:, 1] * SD[1] + MU[1]
    assert bool((phys >= -1e-9).all())
    assert pp._fix_inp_state is None                               # consumed once
    assert pp.history_denormalize(pred, target=False) is pred


def test_preprocessor_composes_dry_air_and_soil_moisture_fixes(stub_base):
    """Both DIAGNOSTIC fixes active together must each apply, and the dry-air fix's
    own behavior must be unchanged by the soil fix's presence (independent channels)."""
    dry_names = ["PS", "T", "TMQ", "U", "PRECT"]
    dry_mu, dry_sd = np.array([98000., 280., 25., 0., 1e-5]), np.array([5000., 20., 15., 10., 1e-5])
    dry_only = D.DryAirFix(channel_names=dry_names, global_means=dry_mu, global_stds=dry_sd, nlat=H)

    pp = _pp(dry_fix=dry_only, soil_fix=_fix())
    inp, pred = _pair(8)
    inp[:, 2] = 10.0
    pred[:, 0] -= 0.2   # seeded mass loss, same as test_mass_fix.py's _pair
    pp.append_unpredicted_features(inp)
    out = pp.history_denormalize(pred, target=True)

    # dry-air behavior: identical to running DryAirFix alone on the same pair
    expected_dry = dry_only(inp, pred)
    assert torch.equal(out[:, 0], expected_dry[:, 0])
    # soil behavior: positivity clamp still applied
    phys_soil = out[:, 1] * SD[1] + MU[1]
    assert bool((phys_soil >= -1e-9).all())
    assert pp._fix_inp_state is None


def test_preprocessor_flag_on_without_input_fails_loudly(stub_base):
    pp = _pp(soil_fix=_fix())
    _, pred = _pair(9)
    with pytest.raises(RuntimeError, match="cached input"):
        pp.history_denormalize(pred, target=True)


def test_build_from_params(tmp_path):
    np.save(tmp_path / "m.npy", MU.reshape(1, -1, 1, 1))
    np.save(tmp_path / "s.npy", SD.reshape(1, -1, 1, 1))
    prm = SimpleNamespace(channel_names=NAMES, global_means_path=str(tmp_path / "m.npy"),
                          global_stds_path=str(tmp_path / "s.npy"), img_crop_shape_x=H,
                          img_shape_x=H)
    f = P._build_soil_moisture_fix(prm)
    assert (f.i_soil, f.i_tsoi) == (1, 2)
    assert f.sd_soil == 0.1 and f.sd_tsoi == 20.0
    assert P._pget({"conserve_soil_moisture": True}, "conserve_soil_moisture") is True
    assert P._pget(SimpleNamespace(), "conserve_soil_moisture", False) is False
