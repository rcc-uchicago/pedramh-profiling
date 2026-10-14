"""port_part0_compare.py: the Part 0 verdicts (stdlib only, so this also runs on a login node)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "port_part0_compare", Path(__file__).resolve().parents[2] / "scripts" / "port_part0_compare.py")
pc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pc)


def _trace(tmp, name, *, scale=1.0, clipped=True, steps=20, sha=None):
    t = {"steps": [{"step": i + 1, "loss": float.hex(0.01 * (i + 1) * scale),
                    "grad_norm": float.hex(0.5 * scale), "clipped": clipped} for i in range(steps)],
         "train_loss_epoch": float.hex(0.1), "valid_loss": float.hex(0.2)}
    if sha:
        t["params_sha256_after"] = sha
    p = tmp / (name + ".json")
    p.write_text(json.dumps(t))
    return str(p)


def test_n2_green_writes_null_and_clipped_ref(tmp_path, capsys):
    a1, a2 = _trace(tmp_path, "a1"), _trace(tmp_path, "a2")
    b1, b2 = _trace(tmp_path, "b1", scale=1 + 2e-7), _trace(tmp_path, "b2", scale=1 + 2e-7)
    null, ref = tmp_path / "n2_null.json", tmp_path / "clipped_ref.json"
    rc = pc.main(["n2", a1, a2, b1, b2, "--null-out", str(null), "--clipped-ref-out", str(ref)])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "N2_ARM arm=a clipped=20/20 repeat_bitwise=yes" in out
    assert "N2_PRETEST clipped=20/20 bitwise=no" in out
    assert "GOLDEN_MATCH clipped_in_job" in out
    steps = json.loads(null.read_text())["steps"]
    assert len(steps) == 20 and 0 < steps[0]["loss_rel"] < 1e-6
    assert json.loads(ref.read_text())["steps"][0]["loss"] == float.hex(0.01)


def test_n2_red_when_a_step_did_not_clip(tmp_path, capsys):
    a1, a2 = _trace(tmp_path, "a1", clipped=False), _trace(tmp_path, "a2", clipped=False)
    b1, b2 = _trace(tmp_path, "b1"), _trace(tmp_path, "b2")
    rc = pc.main(["n2", a1, a2, b1, b2, "--null-out", str(tmp_path / "n"), "--clipped-ref-out", str(tmp_path / "c")])
    out = capsys.readouterr().out
    assert rc == 1 and "N2_ARM arm=a clipped=0/20" in out and "ERROR N2_PRETEST_RED" in out


def test_n1_reports_first_difference(tmp_path, capsys):
    ref = _trace(tmp_path, "ref")
    assert pc.main(["n1", _trace(tmp_path, "same"), ref]) == 0
    assert "N1_TRAIN_PRETEST bitwise=yes" in capsys.readouterr().out
    assert pc.main(["n1", _trace(tmp_path, "moved", scale=1 + 1e-7), ref]) == 1
    assert "N1_TRAIN_PRETEST bitwise=no first=1:loss" in capsys.readouterr().out


def test_crps_ref_only_when_bitwise_including_params(tmp_path, capsys):
    r1, r2 = _trace(tmp_path, "r1", steps=1, sha="aa"), _trace(tmp_path, "r2", steps=1, sha="aa")
    ref = tmp_path / "crps_ref.json"
    assert pc.main(["crps", r1, r2, "--ref-out", str(ref)]) == 0
    assert "GOLDEN_MATCH crps_in_job" in capsys.readouterr().out and ref.exists()
    ref.unlink()
    r3 = _trace(tmp_path, "r3", steps=1, sha="bb")
    assert pc.main(["crps", r1, r3, "--ref-out", str(ref)]) == 0
    assert "CRPS_IN_JOB_NOT_BITWISE first=epoch:params_sha256_after" in capsys.readouterr().out
    assert not ref.exists()
