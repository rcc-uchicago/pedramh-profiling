"""Port F: train on a SUBSET of a pack's state channels without repacking.

`polaris_makani_ace2_ports_handoff.md` §6a drops SOILWATER_10CM and TSOI_10CM by
shortening `channel_names`. Stock `parse_dataset_metadata` turns the names into
pack-relative indices (`params.out_channels`), which the loss, the climatology
and the model package already use to index the full-width stats. These tests
pin the fork's dataset to the same indices:

* full-width configs are unchanged (the lists are exactly `range(n)`);
* a misplaced diagnostic fails loud instead of reading the wrong target;
* a subset reads AND normalizes each kept channel exactly as the full-width
  dataset does at that channel's position -- the silent failure this guards
  against is channel k normalized with channel j's mean/std.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
makani = pytest.importorskip("makani")

from helpers import build_dataset, load_params  # noqa: E402

from sfno_training.trainer.plasim_trainer import _plasim_channel_indices  # noqa: E402


class _P(dict):
    """Minimal stand-in for makani's ParamsBase: attribute + .get access."""

    __getattr__ = dict.__getitem__


def test_full_width_is_identity():
    p = _P(out_channels=list(range(53)), data_channel_names=[f"c{i}" for i in range(53)])
    ic, oc = _plasim_channel_indices(p, 52, 53)
    assert ic == list(range(52)) and oc == list(range(53))


def test_no_out_channels_falls_back_to_contiguous():
    ic, oc = _plasim_channel_indices(_P(), 52, 53)
    assert ic == list(range(52)) and oc == list(range(53))


def test_subset_drops_state_rows_keeps_diagnostic_last():
    keep = [c for c in range(100) if c not in (8, 9)] + [100]
    p = _P(out_channels=keep, data_channel_names=[f"c{i}" for i in range(101)])
    ic, oc = _plasim_channel_indices(p, 98, 99)
    assert oc == keep
    assert ic == keep[:98] and 100 not in ic


def test_count_mismatch_fails_loud():
    p = _P(out_channels=list(range(101)), data_channel_names=[f"c{i}" for i in range(101)])
    with pytest.raises(ValueError, match="CHANNEL_SUBSET_MISMATCH"):
        _plasim_channel_indices(p, 98, 99)


def test_diagnostic_not_last_fails_loud():
    # PRECT (pack index 100) moved to the front of channel_names.
    idx = [100] + [c for c in range(100) if c not in (8, 9)]
    p = _P(out_channels=idx, data_channel_names=[f"c{i}" for i in range(101)])
    with pytest.raises(ValueError, match="CHANNEL_SUBSET_MISMATCH"):
        _plasim_channel_indices(p, 98, 99)


def test_subset_reads_and_normalizes_the_kept_channels(packaged_dataset: Path):
    from makani.utils.parse_dataset_metada import parse_dataset_metadata

    full_params = load_params(packaged_dataset)
    full = build_dataset(full_params, packaged_dataset, n_future=1)

    # Drop two state channels in the middle, exactly as port F does.
    names = list(full_params.channel_names)
    dropped = {names[3], names[7]}
    sub_params = load_params(packaged_dataset)
    sub_params["channel_names"] = [n for n in names if n not in dropped]
    parse_dataset_metadata(sub_params.metadata_json_path, params=sub_params)
    n_state = full_params.n_state_channels - 2
    ic, oc = _plasim_channel_indices(sub_params, n_state, n_state + 1)
    sub = build_dataset(
        sub_params, packaged_dataset, n_future=1, in_channels=ic, out_channels=oc
    )

    keep_in = [c for c in range(full_params.n_state_channels) if c not in (3, 7)]
    keep_out = keep_in + [full_params.n_state_channels]
    f_inp, f_tar, f_fin, f_ftar = full[0]
    s_inp, s_tar, s_fin, s_ftar = sub[0]
    assert s_inp.shape[1] == n_state and s_tar.shape[1] == n_state + 1
    np.testing.assert_array_equal(np.asarray(s_inp), np.asarray(f_inp)[:, keep_in])
    np.testing.assert_array_equal(np.asarray(s_tar), np.asarray(f_tar)[:, keep_out])
    np.testing.assert_array_equal(np.asarray(s_fin), np.asarray(f_fin))
