"""Renderer test: warmstart_provenance.txt → `### Warm-start provenance`.

Covers the optional sidecar plumbing added by
docs/2026-05-14_v11_clip_warmstart_continuation_plan.md §6.1:

  - When ``--run-dir`` points at a directory containing
    ``warmstart_provenance.txt``, the rendered ``report.md`` includes a
    ``### Warm-start provenance`` block with the sidecar's
    ``key = value`` pairs.
  - When ``--run-dir`` is unset OR the sidecar is missing, the block is
    silently omitted (non-warmstart runs must keep their existing
    report layout).

The renderer's other inputs (scorecard CSV, channel-name resolution) are
mocked with the minimum data required to drive a successful render.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RENDERER = _REPO_ROOT / "scripts" / "render_eval_report.py"


def _write_min_scorecard(out_root: Path) -> None:
    """Write the minimum CSV the renderer parses without crashing.

    score_nwp.py emits 'model,channel,lead_hours,metric,mean,std,n_ics'.
    The gate-render block reads ('emulator', 'tas', 6, 'rmse'),
    ('persistence', 'tas', 6, 'rmse'), and a zg500 ACC at 24h, so write
    at least those three plus a few more so the scorecard table renders.
    """
    scores = out_root / "scores"
    scores.mkdir(parents=True)
    rows = [
        "model,channel,lead_hours,metric,mean,std,n_ics",
        "emulator,tas,6,rmse,0.4,0.01,5",
        "persistence,tas,6,rmse,0.9,0.02,5",
        "emulator,zg500,24,acc,0.95,0.01,5",
        "emulator,zg500,6,rmse,3.5,0.1,5",
        "emulator,pr_6h,6,rmse,1.0e-7,1.0e-8,5",
        "emulator,ua5,6,rmse,0.5,0.01,5",
        "emulator,ta5,6,rmse,0.3,0.01,5",
    ]
    (scores / "nwp_scorecard_summary.csv").write_text("\n".join(rows) + "\n")


def _write_metadata_json(path: Path) -> None:
    """Write a metadata.json that satisfies resolve_channel_names()'s
    override path so the test does not need real NetCDFs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "coords": {
                    "channel": ["tas", "pr_6h", "zg500", "ua5", "ta5"],
                }
            }
        )
    )


def _write_sidecar(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "warmstart_provenance.txt").write_text(
        "pretrained_checkpoint_path = /scratch/.../best_ckpt_mp0.tar\n"
        "pretrained_checkpoint_flavor = best_ckpt_mp0 (raw)\n"
        "pretrained_checkpoint_size_bytes = 1745920000\n"
        "pretrained_checkpoint_sha256 = deadbeefdeadbeef\n"
        "warmstart_load_order = after super().__init__, before EMAModel construction\n"
        "lr_peak = 0.0001\n"
        "max_epochs = 50\n"
        "batch_size_global = 8\n"
        "ema_decay = 0.999\n"
        "optimizer_max_grad_norm = 32.0\n"
        "input_noise_sigma = 0.05\n"
        "channel_weights = constant\n"
    )


def _run_renderer(out_root: Path, metadata_json: Path, *, run_dir: Path | None) -> str:
    cmd = [
        sys.executable,
        str(_RENDERER),
        "--out-root", str(out_root),
        "--run-tag", "test_run",
        "--eval-sha7", "abc1234",
        "--data-sha7", "def5678",
        "--train-sha7", "fed3210",
        "--ckpt-path", "/fake/ckpt.tar",
        "--metadata-json", str(metadata_json),
    ]
    if run_dir is not None:
        cmd += ["--run-dir", str(run_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, (
        f"render_eval_report.py failed: rc={res.returncode}\n"
        f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    )
    return (out_root / "report.md").read_text()


def test_warmstart_block_emitted_when_sidecar_present(tmp_path: Path):
    out_root = tmp_path / "eval_out"
    run_dir = tmp_path / "run_dir"
    metadata_json = tmp_path / "metadata.json"

    _write_min_scorecard(out_root)
    _write_metadata_json(metadata_json)
    _write_sidecar(run_dir)

    body = _run_renderer(out_root, metadata_json, run_dir=run_dir)

    assert "### Warm-start provenance" in body, (
        "Renderer should emit the Warm-start provenance heading when "
        "--run-dir/warmstart_provenance.txt exists"
    )
    # A representative pair must be rendered as a table row.
    assert "`pretrained_checkpoint_flavor`" in body
    assert "best_ckpt_mp0 (raw)" in body
    # Source-sidecar pointer present for forensic traceability.
    assert "warmstart_provenance.txt" in body


def test_warmstart_block_omitted_when_run_dir_unset(tmp_path: Path):
    out_root = tmp_path / "eval_out"
    metadata_json = tmp_path / "metadata.json"

    _write_min_scorecard(out_root)
    _write_metadata_json(metadata_json)

    body = _run_renderer(out_root, metadata_json, run_dir=None)
    assert "### Warm-start provenance" not in body


def test_warmstart_block_omitted_when_sidecar_absent(tmp_path: Path):
    out_root = tmp_path / "eval_out"
    run_dir = tmp_path / "run_dir_no_sidecar"
    run_dir.mkdir()
    metadata_json = tmp_path / "metadata.json"

    _write_min_scorecard(out_root)
    _write_metadata_json(metadata_json)

    body = _run_renderer(out_root, metadata_json, run_dir=run_dir)
    assert "### Warm-start provenance" not in body
