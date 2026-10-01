"""sfno_training.compat: the helpers that let our code run on the pin and on makani main.

Expectations are read from the installed makani, so this file is valid in both
venvs (makani_port/HANDOFF_worker.md M2: identical pass counts old vs new).
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("torch")
pytest.importorskip("makani")

from makani.utils import comm  # noqa: E402

from sfno_training import compat  # noqa: E402

_DEFAULT_NAMES = list(inspect.signature(comm.init).parameters["model_parallel_names"].default)
_ON_PIN = "fin" in _DEFAULT_NAMES


def test_layout_is_the_installed_comm_layout():
    sizes, names = compat.model_parallel_layout(1, 1)
    assert names == _DEFAULT_NAMES
    assert sizes == [1] * len(names)


def test_layout_keeps_spatial_sizes():
    sizes, names = compat.model_parallel_layout(2, 4)
    assert sizes[:2] == [2, 4] and names[:2] == ["h", "w"]


def test_legacy_config_layout_is_normalised():
    # every checkpoint we keep was saved by the pin with this pair (api_delta §5)
    sizes, names = compat.normalize_model_parallel([1, 1, 1, 1], ["h", "w", "fin", "fout"])
    assert names == _DEFAULT_NAMES
    assert sizes == [1] * len(names)


def test_set_params_writes_one_size_per_group():
    params = {}
    compat.set_model_parallel_params(params, *compat.model_parallel_layout(1, 1))
    expected = {f"{n}_parallel_size": 1 for n in _DEFAULT_NAMES}
    expected.update(model_parallel_sizes=[1] * len(_DEFAULT_NAMES), model_parallel_names=_DEFAULT_NAMES)
    assert params == expected


def test_feature_split_on_the_installed_makani():
    if _ON_PIN:  # the pin has no matmul group to put a size in
        with pytest.raises(ValueError, match="needs makani main"):
            compat.model_parallel_layout(1, 1, matmul=2)
        assert compat.model_parallel_layout(1, 1, fin=2, fout=2) == ([1, 1, 2, 2], ["h", "w", "fin", "fout"])
    else:  # main maps fin x fout onto its one matmul group
        assert compat.model_parallel_layout(1, 1, fin=2, fout=2) == ([1, 1, 4], ["h", "w", "matmul"])
        assert compat.model_parallel_layout(1, 1, matmul=2) == ([1, 1, 2], ["h", "w", "matmul"])
