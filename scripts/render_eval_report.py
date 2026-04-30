#!/usr/bin/env python3
"""render_eval_report.py — render the Phase 1 markdown report.

Implements docs/sfno_eval_plan.md §F. Reads the scorecard CSV and the
bias-map .npy files produced by score_nwp.py, then writes a single
``report.md`` summarising:

  - Header (run-tag, P-1 status, all three SHAs, checkpoint provenance).
  - Section 1 — NWP scorecard table (5 key channels × 6 leads, all 3 baselines).
  - Section 2 — Sanity gate verdict.
  - Section 3 — Bias-map figure paths (PNGs are out of scope for Phase 1;
    we emit the .npy paths so the user can plot externally).
  - Section 4 — Climate-mode notes (if climate NetCDFs present).
  - Section 5 — Provenance gaps (e.g. ``train_sha7=unknown``).
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


_KEY_CHANNELS = ("tas", "pr_6h", "zg5", "ua5", "ta5")
_SCORED_LEADS_H = (6, 24, 72, 120, 240, 336)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render the Phase 1 markdown report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--out-root", required=True, type=Path,
                   help="Eval output root containing scores/, inference/, baselines/")
    p.add_argument("--run-tag", required=True, type=str)
    p.add_argument("--eval-sha7", required=True, type=str)
    p.add_argument("--data-sha7", required=True, type=str)
    p.add_argument("--train-sha7", required=True, type=str)
    p.add_argument("--ckpt-path", required=True, type=str)
    p.add_argument("--report-out", type=Path, default=None,
                   help="Override report.md path")
    return p.parse_args()


def _read_summary(scorecard_summary_path: Path) -> dict:
    """Return dict[(model, channel, lead, metric)] → (mean, std, n_ics)."""
    out = {}
    with scorecard_summary_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["model"], row["channel"], int(row["lead_hours"]), row["metric"])
            out[key] = (
                float(row["mean"]) if row["mean"] not in ("", "nan") else float("nan"),
                float(row["std"]) if row["std"] not in ("", "nan") else float("nan"),
                int(row["n_ics"]),
            )
    return out


def _format_value(v: float) -> str:
    if v != v:
        return "NaN"
    return f"{v:.4f}"


def _render_table(summary: dict) -> str:
    """Render the §F NWP scorecard table for the 5 key channels."""
    lines = [
        "## NWP Scorecard (mean ± std over ICs)",
        "",
        "RMSE (lower is better) and ACC (higher is better) per channel × lead × baseline.",
        "Persistence on `pr_6h` is undefined (no IC value, see §C.1) and reported as `NaN`.",
        "",
    ]
    for metric in ("rmse", "acc"):
        lines.append(f"### {metric.upper()}")
        lines.append("")
        header = ["channel", "model"] + [f"{h}h" for h in _SCORED_LEADS_H]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join("---" for _ in header) + "|")
        for ch in _KEY_CHANNELS:
            for model in ("emulator", "persistence"):
                if metric == "acc" and model == "persistence":
                    continue  # ACC for persistence not reported
                cells = [ch, model]
                for h in _SCORED_LEADS_H:
                    rec = summary.get((model, ch, h, metric))
                    if rec is None:
                        cells.append("—")
                    else:
                        mean, std, n = rec
                        cells.append(f"{_format_value(mean)} ± {_format_value(std)} (n={n})")
                lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def _render_gate(summary: dict) -> str:
    """Re-derive the gate verdict from the summary CSV."""
    em_tas_6h = summary.get(("emulator", "tas", 6, "rmse"), (float("nan"),))[0]
    pers_tas_6h = summary.get(("persistence", "tas", 6, "rmse"), (float("nan"),))[0]
    em_zg5_24h = summary.get(("emulator", "zg5", 24, "acc"), (float("nan"),))[0]

    pass1 = em_tas_6h < pers_tas_6h if (em_tas_6h == em_tas_6h) else False
    pass2 = em_zg5_24h > 0.6 if (em_zg5_24h == em_zg5_24h) else False
    overall = "PASS" if (pass1 and pass2) else "FAIL"

    return (
        "## Sanity Gate (§D.6)\n\n"
        f"- Emulator RMSE on `tas` at 6 h = `{_format_value(em_tas_6h)}` "
        f"vs persistence `{_format_value(pers_tas_6h)}` → "
        f"**{'PASS' if pass1 else 'FAIL'}**\n"
        f"- Emulator ACC on `zg5` at 24 h = `{_format_value(em_zg5_24h)}` "
        f"vs threshold `0.6` → **{'PASS' if pass2 else 'FAIL'}**\n\n"
        f"**Overall: {overall}**\n"
    )


def _render_bias_maps(scores_dir: Path) -> str:
    """List bias-map .npy paths (PNG plotting is out of scope for Phase 1)."""
    bms = sorted(scores_dir.glob("bias_maps_*.npy"))
    if not bms:
        return "## Bias Maps\n\nNo bias-map files produced.\n"
    lines = ["## Bias Maps", "",
             "Mean error (`pred - truth`) per (channel, lead) pair, IC-averaged.",
             ""]
    for p in bms:
        lines.append(f"- `{p.relative_to(scores_dir.parent)}`")
    lines.append("")
    return "\n".join(lines)


def _render_climate(out_root: Path) -> str:
    cm_dir = out_root / "inference" / "climate"
    if not cm_dir.exists():
        return ""
    nc_files = sorted(cm_dir.glob("*.nc"))
    if not nc_files:
        return ""
    lines = ["## Climate Rollouts", "",
             f"Found **{len(nc_files)}** climate NetCDF(s) under `{cm_dir.name}/`.",
             "Per §0.B, Phase 1 produces a lightweight diagnostic only — full",
             "climate scoring (KE / variance spectra, drift) is deferred to a v3 plan.",
             "Files:"]
    for p in nc_files:
        lines.append(f"- `{p.name}`")
    lines.append("")
    return "\n".join(lines)


def _render_header(args: argparse.Namespace) -> str:
    return (
        f"# SFNO PlaSim Emulator Evaluation Report\n\n"
        f"**Run tag:** `{args.run_tag}`\n\n"
        f"| field | value |\n"
        f"|---|---|\n"
        f"| Eval code SHA | `{args.eval_sha7}` |\n"
        f"| Data packager SHA | `{args.data_sha7}` |\n"
        f"| Training code SHA | `{args.train_sha7}` |\n"
        f"| Checkpoint | `{args.ckpt_path}` |\n\n"
    )


def _render_provenance(args: argparse.Namespace) -> str:
    lines = ["## Provenance Notes", ""]
    if args.train_sha7 == "unknown":
        lines.append(
            "- ⚠️ `train_sha7` is `unknown`. The training run did not capture a "
            "code SHA at submit time; see §G.5 of the plan for the recommended "
            "`scripts/submit_full.slurm` patch."
        )
    else:
        lines.append(f"- ✅ `train_sha7` recovered: `{args.train_sha7}`.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    scores_dir = args.out_root / "scores"
    summary_path = scores_dir / "nwp_scorecard_summary.csv"
    if not summary_path.is_file():
        raise SystemExit(f"missing {summary_path}; run score_nwp.py first")

    summary = _read_summary(summary_path)

    parts = [
        _render_header(args),
        _render_table(summary),
        _render_gate(summary),
        _render_bias_maps(scores_dir),
        _render_climate(args.out_root),
        _render_provenance(args),
    ]
    body = "\n".join(p for p in parts if p)

    out_path = args.report_out or (args.out_root / "report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body)
    logging.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
