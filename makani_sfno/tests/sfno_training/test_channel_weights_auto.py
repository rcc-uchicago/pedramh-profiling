"""Port C finding (polaris_makani_ace2_ports_handoff.md §5): what does the INSTALLED
makani's `channel_weights: "auto"` do to the E3SM ALLDATA channel names?

`auto` is keyed on lowercase ERA5 names (u10m, t2m, z500, ...). Every E3SM name
(PS, T_l00, Z3_l17, RELHUM_l05, ...) falls through to the same default, so after
normalization `auto` IS `constant` on this pack -- switching to it would be a
silent no-op. This test pins that, so a makani upgrade that changes it is noticed.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("makani")

from makani.utils.losses.base_loss import _compute_channel_weighting_helper  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "polaris"))
conv = pytest.importorskip("convert_e3sm_to_makani_alldata")


def test_auto_is_constant_on_the_e3sm_alldata_names():
    names = list(conv.TARGET_CHANNELS)
    auto = _compute_channel_weighting_helper(names, "auto")
    const = _compute_channel_weighting_helper(names, "constant")
    assert torch.allclose(auto, const), (
        "installed makani's 'auto' now distinguishes E3SM channels -- port C's "
        "'auto is a no-op here' finding no longer holds; re-read base_loss.py")
