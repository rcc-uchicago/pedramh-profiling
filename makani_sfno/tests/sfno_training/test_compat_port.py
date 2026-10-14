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


def test_loss_handler_compile_off_only_changes_the_default():
    class Compiling:
        def __init__(self, params, compile=True):
            self.compile = compile

    class PinStyle:
        def __init__(self, params):
            pass

    eager = compat.loss_handler_compile_off(Compiling)
    assert issubclass(eager, Compiling)
    assert eager(None).compile is False
    assert eager(None, compile=True).compile is True
    assert compat.loss_handler_compile_off(PinStyle) is PinStyle


def test_both_trainer_modules_build_an_eager_loss():
    from makani.utils import loss as makani_loss
    from makani.utils.training import deterministic_trainer, ensemble_trainer

    from sfno_training.trainer.plasim_trainer import _install_plasim_patches

    _install_plasim_patches()
    has_compile = "compile" in inspect.signature(makani_loss.LossHandler.__init__).parameters
    for module in (deterministic_trainer, ensemble_trainer):
        if has_compile:  # makani main
            assert module.LossHandler is not makani_loss.LossHandler
            assert issubclass(module.LossHandler, makani_loss.LossHandler)
        else:  # the pin: untouched
            assert module.LossHandler is makani_loss.LossHandler


# --- legacy checkpoints (api_delta §3.1) ---
class _NotAllowlisted:
    pass


def test_ruamel_checkpoint_loads_and_nothing_else_is_allowed(tmp_path):
    import torch
    from makani.utils import checkpoint_helpers
    from ruamel.yaml import YAML

    on_main = hasattr(checkpoint_helpers, "load_checkpoint")
    assert compat.CHECKPOINT_SAFE_GLOBALS_REGISTERED is on_main
    if not on_main:  # the pin loads with weights_only=False
        return
    cfg = YAML().load("lr: &lr 2.0e-3\neps: 1.0e-8\nlr_again: *lr\n")
    assert type(cfg["lr"]).__name__ == "ScalarFloat" and cfg["lr"].anchor.value == "lr"
    path = tmp_path / "ckpt.tar"
    torch.save({"model_state": {"w": torch.arange(3.0)}, "optimizer_state_dict": {"lr": cfg["lr"], "eps": cfg["eps"]}}, path)
    ckpt = checkpoint_helpers.load_checkpoint(str(path))
    assert ckpt["optimizer_state_dict"]["lr"] == 2.0e-3 and ckpt["optimizer_state_dict"]["eps"] == 1.0e-8
    assert torch.equal(ckpt["model_state"]["w"], torch.arange(3.0))
    torch.save({"x": _NotAllowlisted()}, tmp_path / "other.tar")
    with pytest.raises(RuntimeError, match="allowlist"):
        checkpoint_helpers.load_checkpoint(str(tmp_path / "other.tar"))


# --- grid declaration (api_delta §3.2, makani_port/grid_declaration.md option 1) ---
_OUR_LAT = [89.5 - i for i in range(180)]


class _Params(dict):
    """Item and attribute access, as makani's ParamsBase gives parse_dataset_metadata."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None


def _data_json(tmp_path, lat, drop=()):
    import json

    meta = {
        "dataset_name": "e3sm_test",
        "h5_path": "fields",
        "dhours": 6,
        "coords": {
            "grid_type": "equiangular",
            "lat": lat,
            "lon": [float(i) for i in range(360)],
            "channel": ["T_l00", "PS"],
        },
        "attrs": {"description": "fixture"},
    }
    for key in drop:
        meta.pop(key)
    path = tmp_path / "data.json"
    path.write_text(json.dumps(meta))
    return _Params(metadata_json_path=str(path))


def _stock_check():
    from makani.utils import parse_dataset_metada as pdm

    return getattr(pdm, "verify_grid_type", None)


_HAS_GRID_CHECK = _stock_check() is not None  # main (798245b); the pin has no check


def test_cellcentred_predicate_is_exactly_our_grid():
    assert compat.is_our_cellcentred_grid("equiangular", _OUR_LAT)
    assert not compat.is_our_cellcentred_grid("legendre-gauss", _OUR_LAT)
    assert not compat.is_our_cellcentred_grid("equiangular", _OUR_LAT[::-1])
    assert not compat.is_our_cellcentred_grid("equiangular", _OUR_LAT + [-90.5])
    assert not compat.is_our_cellcentred_grid("equiangular", [x + 0.01 for x in _OUR_LAT])


def test_our_data_json_parses_and_the_check_is_restored(tmp_path, capsys, monkeypatch):
    stock = _stock_check()
    monkeypatch.setattr(compat, "_optout_logged", False)
    params = _data_json(tmp_path, _OUR_LAT)
    params, _ = compat.parse_dataset_metadata_scoped(params["metadata_json_path"], params=params)
    assert _stock_check() is stock
    assert params["data_grid_type"] == "equiangular" and params["lat"] == _OUR_LAT
    assert params["channel_names"] == ["T_l00", "PS"] and params["dataset"]["name"] == "e3sm_test"
    out = capsys.readouterr().out.splitlines()
    logged = [line for line in out if line.startswith(compat.GRID_VERIFY_OPTOUT)]
    assert len(logged) == (1 if _HAS_GRID_CHECK else 0)


def test_any_other_grid_is_still_checked(tmp_path):
    stock = _stock_check()
    params = _data_json(tmp_path, [80.0 - 160.0 * i / 179 for i in range(180)])
    if _HAS_GRID_CHECK:
        with pytest.raises(ValueError, match="not that grid"):
            compat.parse_dataset_metadata_scoped(params["metadata_json_path"], params=params)
    else:  # the pin takes any declaration
        compat.parse_dataset_metadata_scoped(params["metadata_json_path"], params=params)
    assert _stock_check() is stock


def test_check_is_restored_when_parsing_raises(tmp_path):
    stock = _stock_check()
    params = _data_json(tmp_path, _OUR_LAT, drop=("dataset_name",))
    with pytest.raises(KeyError, match="dataset_name"):
        compat.parse_dataset_metadata_scoped(params["metadata_json_path"], params=params)
    assert _stock_check() is stock
