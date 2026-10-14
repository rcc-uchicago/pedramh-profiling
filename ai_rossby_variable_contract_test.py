#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the two model-config shapes ``ai_rossby_variable_contract`` accepts.

The contract check is the only thing standing between us and a channel
permutation, which is a SILENT failure: ``ClimateZarrDataset`` stacks by
store-attrs order, so permuted channels are correctly-shaped and raise nothing.
That makes the checker itself worth testing — a checker that skips a group is
indistinguishable from one that passes it.

Two shapes exist because the two model classes differ, not by preference:

* **split** — ``PanguPlasimLegacy`` takes ``land_variables`` /
  ``ocean_variables`` and slices its surface tensor ``[surface|land|ocean]``.
* **folded** — ``SfnoPlasim.__init__`` accepts neither, and ``build_model``
  forwards every config key as a kwarg, so declaring them raises ``TypeError``.
  ``surface_variables`` is the concatenation instead.

Both describe the same tensor in the same channel order, so both must pass — and
a permutation or a dropped land variable must fail in *either* shape. That last
part is what these tests actually pin: the folded branch must not become a way
to skip the land channels.

**Standard library only — no torch, no physicsnemo, no GPU**, matching the
module under test (it has to run on a login node, where importing torch can
hang or core-dump; CLAUDE.md #3).

Run::

    python3.12 ai_rossby_variable_contract_test.py

PASS = ``VARIABLE_CONTRACT_TEST_OK (<n> tests)``.
"""

from __future__ import annotations

import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import ai_rossby_variable_contract as vc

_REPO = Path(__file__).resolve().parent
_RECIPE = _REPO / "physicsnemo_ai_rossby" / "examples" / "weather" / "ai_rossby"
_CONVERTER = _REPO / "physicsnemo_ai_rossby" / "tools" / "data" / "e3sm" / "pangu_h5_to_zarr.py"
_DATASET = _RECIPE / "conf" / "dataset" / "e3sm_pangu_parity.yaml"
_SPLIT_MODEL = _RECIPE / "conf" / "model" / "pangu_plasim_e3sm.yaml"
_FOLDED_MODEL = _RECIPE / "conf" / "model" / "sfno_e3sm_parity.yaml"

_LEVELS = "levels: %r\n" % (vc.PLANNED["levels"],)


def _model_yaml(groups: dict) -> str:
    """A minimal model config carrying exactly `groups` (plus levels)."""
    return "".join("%s: %r\n" % (k, v) for k, v in groups.items()) + _LEVELS


def _write(text: str) -> Path:
    fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    fh.write(text)
    fh.close()
    return Path(fh.name)


def _check(model: Path) -> int:
    """Run --check-artifacts against `model`, silently. Returns the exit code.

    No `--store`: the store checks are independent of the model shape, and this
    must run anywhere (a login node, a laptop) with no converted data on disk.
    """
    with redirect_stdout(io.StringIO()):
        return vc.check_artifacts(model, _DATASET, _CONVERTER, None)


_SPLIT_GROUPS = {g: vc.PLANNED[g] for g in vc.VAR_GROUPS}
_FOLDED_GROUPS = {
    g: (vc.STORE_SURFACE if g == "surface_variables" else vc.PLANNED[g])
    for g in vc.VAR_GROUPS
    if g not in vc.FOLDABLE_GROUPS
}


# --- The two shapes both pass, on the REAL configs -------------------------


def test_real_split_config_passes():
    assert _check(_SPLIT_MODEL) == 0, "pangu_plasim_e3sm.yaml (split) must pass"


def test_real_folded_config_passes():
    assert _check(_FOLDED_MODEL) == 0, "sfno_e3sm_parity.yaml (folded) must pass"


def test_shapes_are_detected():
    assert vc.parse_ai_rossby_model(_SPLIT_MODEL)["surface_folded"] is False
    assert vc.parse_ai_rossby_model(_FOLDED_MODEL)["surface_folded"] is True


# --- The folded branch must not be a way to skip the land channels ---------
#
# This is the whole risk of adding a second shape. Each case below is a real
# defect that a shape-blind checker would wave through.


def test_folded_rejects_unfolded_surface():
    """Folded config whose surface is the 6 prognostics — land silently gone."""
    g = dict(_FOLDED_GROUPS, surface_variables=vc.PLANNED["surface_variables"])
    assert _check(_write(_model_yaml(g))) != 0


def test_folded_rejects_permuted_land():
    """Right names, wrong order — the failure mode torch.cat cannot catch."""
    surface = list(vc.STORE_SURFACE)
    surface[-2], surface[-1] = surface[-1], surface[-2]  # swap the two land vars
    g = dict(_FOLDED_GROUPS, surface_variables=surface)
    assert _check(_write(_model_yaml(g))) != 0


def test_folded_rejects_dropped_land_variable():
    g = dict(_FOLDED_GROUPS, surface_variables=vc.STORE_SURFACE[:-1])
    assert _check(_write(_model_yaml(g))) != 0


def test_folded_still_checks_other_groups():
    """Folding surface must not relax any other group."""
    upper = list(vc.PLANNED["upper_air_variables"])
    upper[0], upper[1] = upper[1], upper[0]
    g = dict(_FOLDED_GROUPS, upper_air_variables=upper)
    assert _check(_write(_model_yaml(g))) != 0


def test_split_rejects_folded_surface():
    """The mirror: a split config must not carry land in `surface_variables`."""
    g = dict(_SPLIT_GROUPS, surface_variables=vc.STORE_SURFACE)
    assert _check(_write(_model_yaml(g))) != 0


# --- Half-declared configs are an error, not a shape -----------------------


def test_partial_foldable_groups_raise():
    """`land_variables` without `ocean_variables` is neither shape.

    Guessing which was meant is exactly how a permutation gets waved through, so
    the parser refuses rather than picking a branch.
    """
    g = {k: v for k, v in _SPLIT_GROUPS.items() if k != "ocean_variables"}
    try:
        vc.parse_ai_rossby_model(_write(_model_yaml(g)))
    except KeyError:
        return
    raise AssertionError("a half-declared config must raise, not pick a shape")


def test_missing_required_group_raises():
    """Only land/ocean are optional; anything else missing is an error."""
    g = {k: v for k, v in _FOLDED_GROUPS.items() if k != "diagnostic_variables"}
    try:
        vc.parse_ai_rossby_model(_write(_model_yaml(g)))
    except KeyError:
        return
    raise AssertionError("a missing required group must raise")


# --- The folded order is the one the model will actually be fed ------------


def test_store_surface_is_pangu_concatenation():
    """`STORE_SURFACE` must equal PanguWeather's own surface concatenation.

    `data_loader_multifiles.py:452-472` builds it as
    surface + land + ocean, and `SfnoPlasim` is handed that tensor. If this
    drifts, the folded config describes a tensor nobody produces.
    """
    assert vc.STORE_SURFACE == (
        vc.PLANNED["surface_variables"]
        + vc.PLANNED["land_variables"]
        + vc.PLANNED["ocean_variables"]
    )


def test_channel_arithmetic_matches_panguweather():
    """in_chans 105 / out_chans 101 — the numbers the architecture gate asserts."""
    n_up = len(vc.PLANNED["upper_air_variables"]) * len(vc.PLANNED["levels"])
    n_surf = len(vc.STORE_SURFACE)
    in_chans = n_surf + len(vc.PLANNED["constant_boundary_variables"]) \
        + len(vc.PLANNED["varying_boundary_variables"]) + n_up
    out_chans = n_surf + len(vc.PLANNED["diagnostic_variables"]) + n_up
    assert (in_chans, out_chans) == (105, 101), (in_chans, out_chans)
    assert vc.n_channels(vc.PLANNED) == 108


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print("ERROR %s: %s" % (t.__name__, exc))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("ERROR %s raised %s: %s" % (t.__name__, type(exc).__name__, exc))
    if failed:
        print("ERROR VARIABLE_CONTRACT_TEST_FAILED (%d/%d)" % (failed, len(tests)))
        sys.exit(1)
    print("VARIABLE_CONTRACT_TEST_OK (%d tests)" % len(tests))
