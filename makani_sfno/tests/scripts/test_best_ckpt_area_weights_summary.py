"""Stdlib test for scripts/best_ckpt_area_weights_summary.py (runs on a login node: python3.11 <this file>).

Plants a ranking flip: e243 is best under naive weights, e223 under band weights; e203 has fewer ICs, so only the
common ICs may be compared.
"""
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "best_ckpt_area_weights_summary.py"


def _run(tmp: Path) -> str:
    r = random.Random(1)
    base = {ic: 0.0128 + r.gauss(0, 0.002) for ic in range(200)}
    spec = {243: (0.0, 0.0), 223: (0.0006, -0.0004), 203: (0.0017, 0.0015)}
    for e, (dn, db) in spec.items():
        ics = list(range(150 if e == 203 else 200))
        rec = {"epoch": e, "ics": ics,
               "naive": [base[i] * (1 + dn) + r.gauss(0, 1e-7) for i in ics],
               "band": [base[i] * (1.0007 + db) + r.gauss(0, 1e-7) for i in ics]}
        (tmp / f"epoch_{e:04d}.json").write_text(json.dumps(rec))
    log = tmp / "out.log"
    log.write_text("Epoch 243 summary:\n validation loss: 0.0128394\nEpoch 223 summary:\n validation loss: 0.0128473\n"
                   "Epoch 203 summary:\n validation loss: 0.0128615\n")
    out = subprocess.run([sys.executable, str(SCRIPT), "--dir", str(tmp), "--out-log", str(log)],
                         capture_output=True, text=True, check=True)
    return out.stdout


def test_flip_is_detected_on_common_ics():
    with tempfile.TemporaryDirectory() as d:
        out = _run(Path(d))
    token = [l for l in out.splitlines() if l.startswith("BEST_CKPT_CHECK")][0]
    assert "n_common=150" in token, token
    assert "logged_best=e243" in token and "naive_best=e243" in token, token
    assert "band_best=e223" in token and "naive_matches_logged=yes" in token, token
    assert "band_same_as_logged=no" in token, token


if __name__ == "__main__":
    test_flip_is_detected_on_common_ics()
    print("TEST_BEST_CKPT_SUMMARY_OK")
