"""Renderer test: `pr_6h` benchmark-row suppression for own-track reports.

Covers the suppression plumbing added by
docs/2026-05-23_pr6h_unit_alignment_plan.md §5:

  - Default (`--pr6h-unit-align suppress --track own`): the
    `(pr_6h, "5410 benchmark")` row is dropped from BOTH the RMSE and
    ACC scorecard tables, and the §4.3 banner appears in the rendered
    report. Other channels' `5410 benchmark` rows remain. Own `pr_6h`
    `emulator` and `persistence` rows remain.
  - `--pr6h-unit-align none`: previous behavior preserved — the
    `(pr_6h, "5410 benchmark")` row is present in both tables.
  - `--track 5410 --pr6h-unit-align suppress`: guard's `track == "own"`
    clause is load-bearing; in 5410-track mode the rows ARE comparable
    and must NOT be suppressed. Asserts the guard does the right thing
    by calling `_render_table` directly.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RENDERER = _REPO_ROOT / "scripts" / "render_eval_report.py"


def _load_renderer_module():
    """Import the renderer script as a module so we can call helpers
    directly (no subprocess) for the load-bearing-guard test."""
    spec = importlib.util.spec_from_file_location("render_eval_report", _RENDERER)
    module = importlib.util.module_from_spec(spec)
    # The renderer inserts its own dir on sys.path; mirror that so the
    # `from _eval_utils import ...` line works when imported as a module.
    sys.path.insert(0, str(_RENDERER.parent))
    spec.loader.exec_module(module)
    return module


_SCORECARD_HEADER = "model,channel,lead_hours,metric,mean,std,n_ics"


def _write_own_scorecard(out_root: Path) -> None:
    """Minimal own scorecard CSV covering `pr_6h` and `tas` (control)
    across enough leads/metrics to render both RMSE and ACC tables."""
    scores = out_root / "scores"
    scores.mkdir(parents=True)
    rows = [_SCORECARD_HEADER]
    # tas: both RMSE and ACC at multiple leads
    rows += [
        "emulator,tas,6,rmse,0.4,0.01,5",
        "persistence,tas,6,rmse,0.9,0.02,5",
        "emulator,tas,24,rmse,0.55,0.02,5",
        "persistence,tas,24,rmse,1.2,0.04,5",
        "emulator,tas,6,acc,0.99,0.001,5",
        "emulator,tas,24,acc,0.96,0.002,5",
    ]
    # pr_6h: both RMSE and ACC at one lead minimum
    rows += [
        "emulator,pr_6h,6,rmse,1.0e-7,1.0e-8,5",
        "persistence,pr_6h,6,rmse,NaN,NaN,5",
        "emulator,pr_6h,6,acc,0.82,0.05,5",
    ]
    # zg500/ua5/ta5 — present in metadata.json's channel list, so the
    # row resolver needs at least one record per channel to not crash.
    rows += [
        "emulator,zg500,6,rmse,3.5,0.1,5",
        "emulator,zg500,24,acc,0.95,0.01,5",
        "emulator,ua5,6,rmse,0.5,0.01,5",
        "emulator,ta5,6,rmse,0.3,0.01,5",
    ]
    (scores / "nwp_scorecard_summary.csv").write_text("\n".join(rows) + "\n")


def _write_benchmark_scorecard(bench_root: Path) -> None:
    """Minimal 5410-benchmark CSV covering the same channels."""
    scores = bench_root / "scores"
    scores.mkdir(parents=True)
    rows = [_SCORECARD_HEADER]
    rows += [
        "emulator,tas,6,rmse,0.45,0.01,5",
        "emulator,tas,24,rmse,0.62,0.02,5",
        "emulator,tas,6,acc,0.985,0.002,5",
        "emulator,tas,24,acc,0.94,0.003,5",
        # The pr_6h benchmark row — suppression must drop these
        # from the rendered tables under suppress+own.
        "emulator,pr_6h,6,rmse,4.2e-4,2.0e-5,5",
        "emulator,pr_6h,6,acc,0.81,0.04,5",
        "emulator,zg500,6,rmse,3.8,0.12,5",
        "emulator,zg500,24,acc,0.93,0.01,5",
        "emulator,ua5,6,rmse,0.6,0.01,5",
        "emulator,ta5,6,rmse,0.35,0.01,5",
    ]
    (scores / "nwp_scorecard_summary.csv").write_text("\n".join(rows) + "\n")


def _write_metadata_json(path: Path) -> None:
    """Write a metadata.json that satisfies resolve_channel_names()'s
    override path so the test does not need real NetCDFs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"coords": {"channel": ["tas", "pr_6h", "zg500", "ua5", "ta5"]}}
        )
    )


def _run_renderer(
    out_root: Path,
    bench_root: Path,
    metadata_json: Path,
    *,
    track: str = "own",
    pr6h_unit_align: str = "suppress",
) -> str:
    cmd = [
        sys.executable,
        str(_RENDERER),
        "--out-root", str(out_root),
        "--run-tag", "test_pr6h_suppress",
        "--eval-sha7", "abc1234",
        "--data-sha7", "def5678",
        "--train-sha7", "fed3210",
        "--ckpt-path", "/fake/ckpt.tar",
        "--benchmark-5410-out-root", str(bench_root),
        "--metadata-json", str(metadata_json),
        "--track", track,
        "--pr6h-unit-align", pr6h_unit_align,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, (
        f"render_eval_report.py failed: rc={res.returncode}\n"
        f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    )
    return (out_root / "report.md").read_text()


def test_pr6h_suppression_default_own_track(tmp_path: Path):
    """Default own-track + suppress: pr_6h benchmark row dropped from
    both tables; banner present; other channels unaffected."""
    out_root = tmp_path / "eval_out"
    bench_root = tmp_path / "bench_out"
    metadata_json = tmp_path / "metadata.json"

    _write_own_scorecard(out_root)
    _write_benchmark_scorecard(bench_root)
    _write_metadata_json(metadata_json)

    body = _run_renderer(out_root, bench_root, metadata_json)

    # The §4.3 banner must appear.
    assert "5410 benchmark row is suppressed for `pr_6h`" in body, (
        "suppress+own should emit the §4.3 banner explaining suppression"
    )
    # Both forward-transform citations should appear in the banner.
    assert "infer_sfno5410_blocking_h100_packed.py:348-349" in body
    assert "infer_sfno5410_byo_ic.py:425-432" in body

    # No pr_6h / 5410 benchmark TABLE row in either RMSE or ACC tables.
    # (The _load_benchmark banner intentionally mentions the suppression;
    # we filter to actual markdown-table rows here.)
    for line in body.splitlines():
        if line.startswith("| pr_6h |") and "5410 benchmark" in line:
            raise AssertionError(
                f"pr_6h 5410-benchmark row should be suppressed but found: {line!r}"
            )

    # Other channels' 5410 benchmark rows remain.
    assert "| tas | 5410 benchmark |" in body, (
        "tas 5410-benchmark row should remain when only pr_6h is suppressed"
    )

    # Own pr_6h emulator + persistence rows remain.
    assert "| pr_6h | emulator |" in body, "own pr_6h emulator row must remain"
    assert "| pr_6h | persistence |" in body, (
        "own pr_6h persistence row must remain"
    )


def test_pr6h_suppression_disabled_with_none(tmp_path: Path):
    """`--pr6h-unit-align none`: prior behavior — pr_6h benchmark row
    is present in both tables, and the old partial disclaimer (not the
    new banner) appears as the caption."""
    out_root = tmp_path / "eval_out"
    bench_root = tmp_path / "bench_out"
    metadata_json = tmp_path / "metadata.json"

    _write_own_scorecard(out_root)
    _write_benchmark_scorecard(bench_root)
    _write_metadata_json(metadata_json)

    body = _run_renderer(
        out_root, bench_root, metadata_json, pr6h_unit_align="none"
    )

    # Suppression banner must NOT appear in none mode.
    assert "5410 benchmark row is suppressed for `pr_6h`" not in body, (
        "none mode should not emit the §4.3 banner"
    )

    # pr_6h / 5410 benchmark row IS present in this mode.
    pr6h_bench_lines = [
        line for line in body.splitlines()
        if "| pr_6h |" in line and "5410 benchmark" in line
    ]
    assert pr6h_bench_lines, (
        "none mode must keep the pr_6h 5410-benchmark row in the table"
    )


def test_pr6h_suppression_skipped_in_5410_track(tmp_path: Path):
    """Load-bearing guard test: `--track 5410 --pr6h-unit-align suppress`
    must NOT drop the pr_6h benchmark row. In 5410-track mode both rows
    are in matching group-native units and ARE directly comparable."""
    out_root = tmp_path / "eval_out"
    bench_root = tmp_path / "bench_out"
    metadata_json = tmp_path / "metadata.json"

    _write_own_scorecard(out_root)
    _write_benchmark_scorecard(bench_root)
    _write_metadata_json(metadata_json)

    body = _run_renderer(
        out_root, bench_root, metadata_json,
        track="5410", pr6h_unit_align="suppress",
    )

    # Suppression banner must NOT appear in 5410-track mode (the 5410
    # caption is the one that fires).
    assert "5410 benchmark row is suppressed for `pr_6h`" not in body, (
        "5410-track mode should not emit the own-track suppression banner"
    )

    # pr_6h / 5410 benchmark row MUST remain — the rows are comparable.
    pr6h_bench_lines = [
        line for line in body.splitlines()
        if "| pr_6h |" in line and "5410 benchmark" in line
    ]
    assert pr6h_bench_lines, (
        "5410-track + suppress must NOT drop the pr_6h benchmark row "
        "(track == 'own' guard is load-bearing)"
    )


def test_render_table_guard_unit_directly():
    """Directly drive `_render_table` to confirm the guard's
    (track == 'own' AND pr6h_unit_align == 'suppress' AND ch == 'pr_6h')
    AND-clause. Faster than the subprocess tests and pins the guard's
    boolean logic explicitly."""
    mod = _load_renderer_module()

    # Synthetic summaries: keyed by (model, channel, lead_hours, metric)
    # → (mean, std, n_ics). Just enough to populate one cell per row.
    summary = {
        ("emulator", "pr_6h", 6, "rmse"): (1e-7, 1e-8, 5),
        ("persistence", "pr_6h", 6, "rmse"): (float("nan"), float("nan"), 5),
        ("emulator", "pr_6h", 6, "acc"): (0.8, 0.05, 5),
        ("emulator", "tas", 6, "rmse"): (0.4, 0.01, 5),
        ("persistence", "tas", 6, "rmse"): (0.9, 0.02, 5),
        ("emulator", "tas", 6, "acc"): (0.99, 0.001, 5),
    }
    bench = {
        ("emulator", "pr_6h", 6, "rmse"): (4e-4, 2e-5, 5),
        ("emulator", "pr_6h", 6, "acc"): (0.81, 0.04, 5),
        ("emulator", "tas", 6, "rmse"): (0.45, 0.01, 5),
        ("emulator", "tas", 6, "acc"): (0.985, 0.002, 5),
    }
    key_channels = ("tas", "pr_6h")

    # own + suppress: pr_6h benchmark row dropped.
    out = mod._render_table(
        summary, key_channels,
        benchmark_summary=bench, track="own", pr6h_unit_align="suppress",
    )
    for line in out.splitlines():
        assert not ("| pr_6h |" in line and "5410 benchmark" in line), (
            f"own+suppress should drop pr_6h benchmark row, got: {line!r}"
        )
    assert "| tas | 5410 benchmark |" in out

    # own + none: pr_6h benchmark row present.
    out = mod._render_table(
        summary, key_channels,
        benchmark_summary=bench, track="own", pr6h_unit_align="none",
    )
    assert any(
        "| pr_6h |" in line and "5410 benchmark" in line
        for line in out.splitlines()
    ), "own+none should keep pr_6h benchmark row"

    # 5410 + suppress: pr_6h benchmark row present (track guard).
    out = mod._render_table(
        summary, key_channels,
        benchmark_summary=bench, track="5410", pr6h_unit_align="suppress",
    )
    assert any(
        "| pr_6h |" in line and "5410 benchmark" in line
        for line in out.splitlines()
    ), (
        "5410+suppress must NOT drop pr_6h benchmark row — the "
        "track == 'own' clause is load-bearing"
    )
