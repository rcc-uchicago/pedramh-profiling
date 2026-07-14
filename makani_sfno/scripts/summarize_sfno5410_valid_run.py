#!/usr/bin/env python3
"""Write a concise completion summary for the valid SFNO-5410 run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import argparse


def read_metric(summary_csv: Path, channel: str, lead: int, metric: str) -> tuple[float, float, int] | None:
    with summary_csv.open() as f:
        for row in csv.DictReader(f):
            if (
                row["model"] == "emulator"
                and row["channel"] == channel
                and int(row["lead_hours"]) == lead
                and row["metric"] == metric
            ):
                return float(row["mean"]), float(row["std"]), int(row["n_ics"])
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root
    raw_dir = run_root / "inference" / "upstream_raw"
    nwp_dir = run_root / "inference" / "nwp"
    scores_dir = run_root / "scores"
    summary_csv = scores_dir / "nwp_scorecard_summary.csv"
    report = run_root / "report.md"
    figures = run_root / "figures"
    inference_meta = run_root / "inference" / "inference_metadata.json"
    sanity = {}
    if inference_meta.exists():
        with inference_meta.open() as f:
            sanity = json.load(f).get("sanity_gate", {})

    lines = [
        "# SFNO-5410 Valid H100 Packed Derecho Run Summary",
        "",
        f"run_root: `{run_root}`",
        f"raw_forecast_netcdf_count: `{len(list(raw_dir.glob('*.nc')))}`",
        f"adapted_score_netcdf_count: `{len(list(nwp_dir.glob('*.nc')))}`",
        f"scorecard_summary: `{summary_csv}`",
        f"report: `{report}`",
        f"figures_dir: `{figures}`",
        f"sanity_gate_passed: `{sanity.get('passed')}`",
        "",
        "## Requested Metrics",
        "",
        "| channel | lead_h | ACC mean | ACC std | RMSE mean | RMSE std | n |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for channel in ("tas", "zg500"):
        for lead in (6, 24, 120, 336):
            acc = read_metric(summary_csv, channel, lead, "acc")
            rmse = read_metric(summary_csv, channel, lead, "rmse")
            if acc is None or rmse is None:
                lines.append(f"| {channel} | {lead} | missing | missing | missing | missing | missing |")
                continue
            lines.append(
                f"| {channel} | {lead} | {acc[0]:.6g} | {acc[1]:.6g} | "
                f"{rmse[0]:.6g} | {rmse[1]:.6g} | {rmse[2]} |"
            )

    out = run_root / "completion_summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(out)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
