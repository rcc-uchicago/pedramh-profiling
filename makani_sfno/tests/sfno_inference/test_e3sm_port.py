"""Tests for the E3SM inference port (changes A-D and G).

Scope → docs/2026-09-10_e3sm_inference_port_scope.md. Every change had to
GENERALIZE rather than special-case E3SM, because `src/sfno_inference/` is a
`git subtree` shared with the Stampede3 `eval-sfno-own` path (CLAUDE.md #5).
So each test here asserts the PLaSim contract still works alongside the E3SM
one — a test that only proves E3SM works would not show the constraint was met.

Run on a compute node: `qsub polaris/polaris_e3sm_port_test.pbs`
(CLAUDE.md #3 — importing torch on a login node hangs or core-dumps).
PASS = E3SM_PORT_OK.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from sfno_eval.metrics import (
    equiangular_lat_weights,
    lat_weights,
    legendre_gauss_lat_weights,
)
from sfno_inference.rollout_driver import _extract_truth_sic


# The two contracts this module has to serve, from the real configs.
PLASIM_FORCING = ["lsm", "sg", "z0", "sst", "rsdt", "sic"]
E3SM_FORCING = ["lsm", "topo", "glacier", "natveg", "sst", "solin", "ice"]


class _FakeDataset:
    """Only the two attributes `_extract_truth_sic` reads."""

    def __init__(self, n_forcing: int, bias=None, scale=None):
        self.forcing_bias = (
            np.arange(n_forcing, dtype=np.float32).reshape(1, n_forcing, 1, 1)
            if bias is None else np.asarray(bias, dtype=np.float32)
        )
        self.forcing_scale = (
            np.full((1, n_forcing, 1, 1), 2.0, dtype=np.float32)
            if scale is None else np.asarray(scale, dtype=np.float32)
        )


class _FakeParams:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _forcing_tensor(names, H=3, W=4, K=2):
    """`(K, n_forcing, H, W)` where channel c is filled with the value c."""
    n = len(names)
    t = torch.zeros(K, n, H, W, dtype=torch.float32)
    for c in range(n):
        t[:, c, :, :] = float(c)
    return t


# ---------------------------------------------------------------------------
# Change D — sea ice located by NAME, not by position
# ---------------------------------------------------------------------------

def test_sic_found_by_name_on_plasim_contract():
    """PLaSim: `sic` is at index 5 — the position the old code hardcoded."""
    tar = _forcing_tensor(PLASIM_FORCING)
    ds = _FakeDataset(len(PLASIM_FORCING))
    out = _extract_truth_sic(tar, ds, _FakeParams(forcing_channel_names=PLASIM_FORCING))
    assert out is not None
    # channel 5 holds the value 5.0; stats are bias=idx, scale=2.0
    assert torch.allclose(out, torch.full_like(out, 5.0 * 2.0 + 5.0))


def test_sic_found_by_name_on_e3sm_contract_and_is_not_solin():
    """🐛 The regression this whole change exists for.

    E3SM forcing is length 7, so the old `if n < 6: disable` guard PASSED,
    and positional index 5 is `solin`. Sea ice is at 6. The old code returned
    solar insolation labelled `truth_sic` with units="fraction".
    """
    tar = _forcing_tensor(E3SM_FORCING)
    ds = _FakeDataset(len(E3SM_FORCING))
    out = _extract_truth_sic(tar, ds, _FakeParams(forcing_channel_names=E3SM_FORCING))
    assert out is not None

    ice_idx, solin_idx = E3SM_FORCING.index("ice"), E3SM_FORCING.index("solin")
    assert (ice_idx, solin_idx) == (6, 5), "fixture no longer reproduces the bug"

    expected_ice = float(ice_idx) * 2.0 + float(ice_idx)
    expected_solin = float(solin_idx) * 2.0 + float(solin_idx)
    assert torch.allclose(out, torch.full_like(out, expected_ice))
    assert not torch.allclose(out, torch.full_like(out, expected_solin))


def test_sic_disabled_without_names_rather_than_guessing():
    """No names ⇒ None. Guessing a position is what produced wrong physics."""
    tar = _forcing_tensor(E3SM_FORCING)
    ds = _FakeDataset(len(E3SM_FORCING))
    assert _extract_truth_sic(tar, ds, _FakeParams()) is None
    assert _extract_truth_sic(tar, ds, None) is None


def test_sic_disabled_when_no_alias_matches():
    names = ["lsm", "topo", "sst", "solin"]        # no sea-ice channel at all
    out = _extract_truth_sic(
        _forcing_tensor(names), _FakeDataset(len(names)),
        _FakeParams(forcing_channel_names=names),
    )
    assert out is None


def test_sic_name_match_is_case_and_space_insensitive():
    names = [" LSM ", "topo", "glacier", "natveg", "sst", "solin", " ICE "]
    out = _extract_truth_sic(
        _forcing_tensor(names), _FakeDataset(len(names)),
        _FakeParams(forcing_channel_names=names),
    )
    assert out is not None
    assert torch.allclose(out, torch.full_like(out, 6.0 * 2.0 + 6.0))


def test_sic_disabled_when_stats_are_shorter_than_the_named_index(caplog):
    """Names say index 6; stats only cover 6 channels (0..5). Must not index 5."""
    tar = _forcing_tensor(E3SM_FORCING)
    ds = _FakeDataset(6)                            # one channel short
    with caplog.at_level(logging.WARNING):
        out = _extract_truth_sic(
            tar, ds, _FakeParams(forcing_channel_names=E3SM_FORCING))
    assert out is None
    assert "truth_sic disabled" in caplog.text


def test_sic_missing_dataset_stats_disables():
    class _NoStats:
        pass
    out = _extract_truth_sic(
        _forcing_tensor(E3SM_FORCING), _NoStats(),
        _FakeParams(forcing_channel_names=E3SM_FORCING),
    )
    assert out is None


# ---------------------------------------------------------------------------
# Change G — quadrature must follow the grid, and it fails SILENTLY
# ---------------------------------------------------------------------------

def test_equiangular_weights_sum_to_one_and_are_symmetric():
    w = equiangular_lat_weights(180)
    assert w.shape == (180,)
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)
    assert torch.allclose(w, torch.flip(w, dims=[0]), atol=1e-7)


def test_equiangular_weights_are_exactly_the_cell_areas():
    """Weight ∝ sin(top) - sin(bottom), which is the exact spherical cell area."""
    nlat = 180
    edges = np.linspace(np.pi / 2, -np.pi / 2, nlat + 1)
    exact = np.sin(edges[:-1]) - np.sin(edges[1:])
    exact = exact / exact.sum()
    assert np.allclose(equiangular_lat_weights(nlat).numpy(), exact, atol=1e-6)


def test_equiangular_differs_from_gauss_legendre_at_the_poles():
    """Why the bug is worth a test, and why it is invisible.

    The two weight vectors have the SAME shape, so the only downstream guard
    (`pred.shape[-2] != lat_weights.shape[0]`) cannot fire. In ABSOLUTE terms
    they differ by only ~4e-5, which is why nothing looks wrong. In RELATIVE
    terms Gauss-Legendre over-weights the polar row by ~1.5x — measured — and
    that is the error that lands in a polar score.
    """
    nlat = 180
    gl = legendre_gauss_lat_weights(nlat).numpy()
    eq = equiangular_lat_weights(nlat).numpy()
    assert gl.shape == eq.shape, "shape guard downstream cannot catch this"

    # Absolute difference is small enough to pass a casual allclose.
    assert np.abs(gl - eq).max() < 1e-4

    # Relative difference at the pole is not small at all.
    pole_ratio = float(gl[0] / eq[0])
    assert pole_ratio > 1.4, f"polar over-weighting collapsed to {pole_ratio}"

    # ...and it is a polar effect: mid-latitudes agree far better.
    mid = nlat // 2
    assert abs(gl[mid] / eq[mid] - 1.0) < 0.01


def test_lat_weights_dispatches_and_defaults_to_plasims_rule():
    nlat = 64                                        # PLaSim T21 is 64x128
    assert np.allclose(lat_weights(nlat).numpy(),
                       legendre_gauss_lat_weights(nlat).numpy())
    assert np.allclose(lat_weights(nlat, "legendre-gauss").numpy(),
                       legendre_gauss_lat_weights(nlat).numpy())
    assert np.allclose(lat_weights(180, "equiangular").numpy(),
                       equiangular_lat_weights(180).numpy())
    assert np.allclose(lat_weights(180, " EquiAngular ").numpy(),
                       equiangular_lat_weights(180).numpy())


def test_unknown_grid_type_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="UNKNOWN_GRID_TYPE"):
        lat_weights(180, "cubed-sphere")


# ---------------------------------------------------------------------------
# Changes A/B — the channel contract is derived, and still checked
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n_state,n_diag,n_forcing,n_in,n_out",
    [(52, 1, 6, 58, 53),        # PLaSim — the contract the literals encoded
     (100, 1, 7, 107, 101)],    # E3SM ALLDATA — what the literals rejected
)
def test_consistent_channel_contracts_are_accepted(tmp_path, n_state, n_diag,
                                                   n_forcing, n_in, n_out):
    from sfno_inference.checkpoint_loader import load_eval_params
    _write_run_dir(tmp_path, n_state, n_diag, n_forcing, n_in, n_out)
    p = load_eval_params(tmp_path, K=4)
    assert int(p.N_in_channels) == n_in
    assert int(p.N_out_channels) == n_out
    # Eval-only overrides still applied (K-1 = 3).
    assert int(p.valid_autoreg_steps) == 3
    assert int(p.n_history) == 0
    assert int(p.batch_size) == 1


@pytest.mark.parametrize(
    "n_state,n_diag,n_forcing,n_in,n_out",
    [(100, 1, 7, 106, 101),     # inputs do not add up
     (100, 1, 7, 107, 102)],    # outputs do not add up
)
def test_inconsistent_channel_contract_is_rejected(tmp_path, n_state, n_diag,
                                                   n_forcing, n_in, n_out):
    """The literals were replaced by a check, not deleted. A contract whose
    roles do not sum is exactly what would mis-slice the rollout."""
    from sfno_inference.checkpoint_loader import load_eval_params
    _write_run_dir(tmp_path, n_state, n_diag, n_forcing, n_in, n_out)
    with pytest.raises(ValueError, match="CHANNEL_CONTRACT_INCONSISTENT"):
        load_eval_params(tmp_path, K=4)


# ---------------------------------------------------------------------------
# Change E — init_time labelling: date it when you can, index it when you can't
# ---------------------------------------------------------------------------

def test_calendar_labelling_when_the_pack_carries_an_anchor():
    """PLaSim path, unchanged: anchor + time_plasim ⇒ a datetime64."""
    from sfno_inference.nc_writer import _resolve_init_time

    r = _FakeParams(file_anchor="0126-08-01 00:00:00", time_plasim_at_ic=2.0,
                    ic_global_idx=7)
    value, labelling = _resolve_init_time(r)
    assert labelling == "calendar"
    assert value == np.datetime64("0126-08-03T00:00:00", "s")   # +2 days


def test_step_index_labelling_when_the_anchor_is_absent():
    """E3SM path: no anchor ⇒ the IC index, and NOT a fabricated date.

    The pack is noleap with a split-cumulative day count, so adding those days
    to a proleptic-Gregorian datetime64 drifts one day per leap year crossed.
    """
    from sfno_inference.nc_writer import _resolve_init_time

    for absent in ("", "   ", None):
        value, labelling = _resolve_init_time(
            _FakeParams(file_anchor=absent, time_plasim_at_ic=365.0,
                        ic_global_idx=42))
        assert labelling == "step_index"
        assert int(value) == 42
        assert not isinstance(value, np.datetime64)


def test_malformed_anchor_still_raises_rather_than_downgrading():
    """A present-but-broken anchor is a packing bug, not a calendar choice.
    Silently falling back to an index would hide it."""
    from sfno_inference.nc_writer import _resolve_init_time

    with pytest.raises(ValueError, match="unparseable anchor"):
        _resolve_init_time(_FakeParams(file_anchor="not-a-date",
                                       time_plasim_at_ic=0.0, ic_global_idx=0))


def _write_run_dir(tmp_path, n_state, n_diag, n_forcing, n_in, n_out):
    """Minimal run dir: config.json plus the two stats files it requires."""
    import json
    cfg = {
        "N_in_channels": n_in,
        "N_out_channels": n_out,
        "n_state_channels": n_state,
        "n_diagnostic_channels": n_diag,
        "n_forcing_channels": n_forcing,
        "amp_mode": "none",
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    np.save(tmp_path / "global_means.npy", np.zeros((1, n_out, 1, 1), np.float32))
    np.save(tmp_path / "global_stds.npy", np.ones((1, n_out, 1, 1), np.float32))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
