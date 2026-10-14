"""Tests for finetune_base_check.py (polaris_makani_f_finetune_handoff.md §2.3).

Standard library + pytest; uses the SHIPPED nosoil / ace2vars yamls, so a drift in
their channel lists shows up here too.
"""

import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import finetune_base_check as B  # noqa: E402


def _names(stem, key="channel_names"):
    return B.yaml_list((HERE / f"{stem}.yaml").read_text(), key)


def _base(tmp_path, names, *, n_out=None, stats=True):
    """A fake base run: <pack>/{stats,metadata}, <run>/config.json, <run>/training_checkpoints/x.tar."""
    pack = tmp_path / "pack"
    (pack / "stats").mkdir(parents=True)
    (pack / "metadata").mkdir()
    (pack / "metadata" / "data.json").write_text("{}")
    run = tmp_path / "run"
    (run / "training_checkpoints").mkdir(parents=True)
    ckpt = run / "training_checkpoints" / "best_ckpt_mp0.tar"
    ckpt.write_bytes(b"")
    cfg = {"N_out_channels": len(names) if n_out is None else n_out, "channel_names": names}
    if stats:
        cfg["global_means_path"] = str(pack / "stats" / "global_means.npy")
    (run / "config.json").write_text(json.dumps(cfg))
    return ckpt, pack


@pytest.mark.parametrize("stem", ["e3sm_alldata_nosoil", "e3sm_alldata_ace2vars"])
def test_matching_base_passes_and_names_its_pack(tmp_path, stem, capsys):
    ckpt, pack = _base(tmp_path, _names(stem))
    assert B.main(["x", str(ckpt), str(HERE / f"{stem}.yaml"), stem]) == 0
    out = capsys.readouterr().out
    assert out.startswith("FINETUNE_BASE_OK n_out=%d pack=%s " % (len(_names(stem)), pack))


def test_a_101_channel_checkpoint_is_refused_for_the_99_channel_config(tmp_path, capsys):
    """The handoff's named case: A (101 out) must never start an F arm."""
    full = _names("e3sm_alldata_nosoil")
    full = full[:8] + ["SOILWATER_10CM", "TSOI_10CM"] + full[8:]
    ckpt, _ = _base(tmp_path, full)
    rc = B.main(["x", str(ckpt), str(HERE / "e3sm_alldata_nosoil.yaml"), "e3sm_alldata_nosoil"])
    out = capsys.readouterr().out
    assert rc == 2 and "N_out_channels=101, config has 99" in out
    assert "first mismatch [8]: SOILWATER_10CM vs T_l00" in out


def test_f_checkpoint_is_refused_for_the_g_config(tmp_path, capsys):
    ckpt, _ = _base(tmp_path, _names("e3sm_alldata_nosoil"))
    rc = B.main(["x", str(ckpt), str(HERE / "e3sm_alldata_ace2vars.yaml"), "e3sm_alldata_ace2vars"])
    assert rc == 2 and "N_out_channels=99, config has 77" in capsys.readouterr().out


def test_same_count_different_names_is_refused(tmp_path):
    g = _names("e3sm_alldata_ace2vars")
    swapped = [g[1], g[0]] + g[2:]
    bad, _ = B.check_base({"N_out_channels": 77, "channel_names": swapped,
                           "global_means_path": "/p/stats/global_means.npy"}, g)
    assert len(bad) == 1 and "first mismatch [0]" in bad[0]


def test_checkpoint_outside_a_run_dir_and_missing_stats_path_are_refused(tmp_path, capsys):
    stray = tmp_path / "x.tar"
    stray.write_bytes(b"")
    yml = str(HERE / "e3sm_alldata_ace2vars.yaml")
    assert B.main(["x", str(stray), yml, "e3sm_alldata_ace2vars"]) == 2
    assert "ERROR FINETUNE_BASE_UNREADABLE" in capsys.readouterr().out
    ckpt, _ = _base(tmp_path / "b", _names("e3sm_alldata_ace2vars"), stats=False)
    assert B.main(["x", str(ckpt), yml, "e3sm_alldata_ace2vars"]) == 2
    assert "no global_means_path" in capsys.readouterr().out


def test_yaml_list_does_not_confuse_the_three_channel_lists():
    text = (HERE / "e3sm_alldata_ace2vars.yaml").read_text()
    assert len(B.yaml_list(text, "channel_names")) == 77
    assert len(B.yaml_list(text, "dropped_channel_names")) == 24
    assert B.yaml_list(text, "forcing_channel_names")[0] == "lsm"
