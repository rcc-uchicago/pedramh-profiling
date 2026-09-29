"""Tests for channel_subset_gate.check_subset (port F, handoff §6a).

Pure python -- safe anywhere pytest runs:
    cd makani_sfno/polaris && python -m pytest -q test_channel_subset_gate.py
"""

from __future__ import annotations

import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from channel_subset_gate import check_subset  # noqa: E402

TARGET = ["PS", "TREFHT", "SOILWATER_10CM", "TSOI_10CM", "Z3_l17", "PRECT"]
N_STATE, N_DIAG = 5, 1
DROP = ["SOILWATER_10CM", "TSOI_10CM"]


def _cfg(**over):
    cfg = {"dropped_channel_names": DROP,
           "channel_names": ["PS", "TREFHT", "Z3_l17", "PRECT"],
           "n_state_channels": 3, "n_diagnostic_channels": 1}
    cfg.update(over)
    return cfg


def test_correct_subset_passes():
    assert check_subset(_cfg(), TARGET, N_STATE, N_DIAG) == []


def test_same_width_name_drift_is_caught():
    # the failure the full-width gate exists for: right length, wrong order
    bad = check_subset(_cfg(channel_names=["TREFHT", "PS", "Z3_l17", "PRECT"]),
                       TARGET, N_STATE, N_DIAG)
    assert any("first mismatch [0]" in b for b in bad)


def test_undeclared_drop_is_caught():
    # a third channel silently missing from channel_names
    bad = check_subset(_cfg(channel_names=["PS", "Z3_l17", "PRECT"]), TARGET, N_STATE, N_DIAG)
    assert any("channel_names" in b for b in bad)


def test_stale_n_state_is_caught():
    bad = check_subset(_cfg(n_state_channels=5), TARGET, N_STATE, N_DIAG)
    assert any("n_state_channels" in b for b in bad)


def test_unknown_drop_name_is_caught():
    bad = check_subset(_cfg(dropped_channel_names=["SOILWATER"]), TARGET, N_STATE, N_DIAG)
    assert any("not in TARGET_CHANNELS" in b for b in bad)


def test_dropping_the_diagnostic_is_refused():
    bad = check_subset(
        _cfg(dropped_channel_names=["PRECT"],
             channel_names=["PS", "TREFHT", "SOILWATER_10CM", "TSOI_10CM", "Z3_l17"],
             n_state_channels=5),
        TARGET, N_STATE, N_DIAG)
    assert any("diagnostic" in b for b in bad)


def test_shipped_nosoil_config_passes_against_the_real_converter():
    stem = "e3sm_alldata_nosoil"
    yaml = pytest.importorskip("yaml")
    pytest.importorskip("h5py")
    import convert_e3sm_to_makani_alldata as conv

    cfg = yaml.safe_load(open(HERE / f"{stem}.yaml"))[stem]
    assert cfg["dropped_channel_names"] == ["SOILWATER_10CM", "TSOI_10CM"]
    assert check_subset(cfg, list(conv.TARGET_CHANNELS), conv.N_STATE,
                        len(conv.DIAG_CHANNELS)) == []
    assert cfg["forcing_channel_names"] == conv.FORCING_CHANNELS


# Port G (operator, 2026-09-29): the ACE2-EAMv3 variable set.
G_DROP = (["U10", "RHREFHT", "PSL", "TMQ", "SOILWATER_10CM", "TSOI_10CM"]
          + [f"Z3_l{i:02d}" for i in range(18)])


def test_shipped_ace2vars_config_passes_against_the_real_converter():
    yaml = pytest.importorskip("yaml")
    pytest.importorskip("h5py")
    import convert_e3sm_to_makani_alldata as conv

    cfg = yaml.safe_load(open(HERE / "e3sm_alldata_ace2vars.yaml"))["e3sm_alldata_ace2vars"]
    assert sorted(cfg["dropped_channel_names"]) == sorted(G_DROP)
    assert check_subset(cfg, list(conv.TARGET_CHANNELS), conv.N_STATE,
                        len(conv.DIAG_CHANNELS)) == []
    names = cfg["channel_names"]
    assert len(names) == 77 and cfg["n_state_channels"] == 76 and names[-1] == "PRECT"
    # the two the operator kept on purpose, and nothing of the dropped families
    assert "TREFHT" in names and sum(n.startswith("RELHUM_l") for n in names) == 18
    assert not any(n.startswith("Z3_") for n in names)
    # the dry-air fix's inputs: PS survives, TMQ does not (mass_fix must refuse)
    assert "PS" in names and "TMQ" not in names


def test_ace2vars_differs_from_nosoil_only_in_the_declared_keys():
    """G is F's recipe with a different channel set -- nothing else may move."""
    yaml = pytest.importorskip("yaml")
    f = yaml.safe_load(open(HERE / "e3sm_alldata_nosoil.yaml"))["e3sm_alldata_nosoil"]
    g = yaml.safe_load(open(HERE / "e3sm_alldata_ace2vars.yaml"))["e3sm_alldata_ace2vars"]
    moved = sorted(k for k in set(f) | set(g) if f.get(k) != g.get(k))
    assert moved == ["channel_names", "dropped_channel_names", "n_state_channels"]
