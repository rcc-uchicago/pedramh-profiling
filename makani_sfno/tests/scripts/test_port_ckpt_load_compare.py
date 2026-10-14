"""port_ckpt_load_compare.py: the M3 identity chain (stdlib only)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "port_ckpt_load_compare", Path(__file__).resolve().parents[2] / "scripts" / "port_ckpt_load_compare.py")
lc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lc)


def _entry(seed):
    return {k: "%s%02d" % (k[:4], seed) for k in lc.FIELDS}


def _write(tmp, name, ckpts):
    p = tmp / name
    p.write_text(json.dumps({"checkpoints": ckpts}))
    return str(p)


def test_all_equal_across_two_refs(tmp_path, capsys):
    r1 = _write(tmp_path, "r1.json", {"A": _entry(1), "B": _entry(2)})
    r2 = _write(tmp_path, "r2.json", {"C": _entry(3)})
    load = _write(tmp_path, "load.json", {"A": _entry(1), "B": _entry(2), "C": _entry(3)})
    assert lc.main([load, r1, r2, "--expect", "3"]) == 0
    assert "PORT_CKPT_LOAD_OK n=3" in capsys.readouterr().out


def test_struct_change_and_missing_tag_fail(tmp_path, capsys):
    r1 = _write(tmp_path, "r1.json", {"A": _entry(1), "B": _entry(2)})
    moved = _entry(1)
    moved["params_struct_sha256"] = "other"
    load = _write(tmp_path, "load.json", {"A": moved})
    assert lc.main([load, r1]) == 1
    out = capsys.readouterr().out
    assert "ERROR PORT_CKPT_LOAD_MISMATCH A params_struct_sha256" in out
    assert "ERROR PORT_CKPT_LOAD_MISMATCH B missing_from_load" in out


def test_wrong_ref_count_fails(tmp_path, capsys):
    r1 = _write(tmp_path, "r1.json", {"A": _entry(1)})
    load = _write(tmp_path, "load.json", {"A": _entry(1)})
    assert lc.main([load, r1, "--expect", "10"]) == 1
    assert "ERROR PORT_CKPT_LOAD_COUNT refs=1 expected=10" in capsys.readouterr().out
