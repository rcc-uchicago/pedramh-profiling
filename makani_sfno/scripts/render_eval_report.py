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


sys.path.insert(0, str(Path(__file__).resolve().parent))

from _eval_utils import (  # noqa: E402 -- after sys.path adjustment
    bias_channels,
    detect_z500_channel,
    resolve_channel_names,
)


_SCORED_LEADS_H = (6, 24, 72, 120, 240, 336)


# Rationale for the default-on `pr_6h` benchmark-row suppression
# (`--pr6h-unit-align suppress`, own-track only). Two compounding reasons
# make the on-disk 5410 `pr_6h` row non-comparable to own-track `pr_6h`:
#
# (a) Prediction-side anomaly. The upstream 5410 inference scripts call the
#     *forward* z-score transform on the diagnostic channel at
#       scripts/infer_sfno5410_blocking_h100_packed.py:348-349 (packed path)
#       scripts/infer_sfno5410_byo_ic.py:425-432 (BYO path)
#     asymmetric with the surface/upper-air paths at :343/:346 that use the
#     inverse. The on-disk 5410 `pr_6h` prediction is therefore in a
#     transformed space; RMSE/ACC against the physical-units truth and
#     climatology is not scalar-recoverable.
#
# (b) Truth-side unit convention. 5410 truth follows the group's "6-hour
#     precip proxy" convention `instantaneous_pr_rate(t) × 6h`
#     (docs/2026-05-06_group_sfno_5410_eval_plan.md:127, which explicitly
#     warns *not* to describe it as "accumulated precipitation"); own truth
#     keeps the instantaneous rate (m/s). The nominal factor is 21,600 s/6h
#     but the empirically observed truth-stats ratio is ~3,600-4,400× per
#     docs/2026-05-14_pr_6h_units_mismatch_ticket.md — a ~5× unexplained gap
#     on top of the unit factor.
#
# See docs/2026-05-23_pr6h_unit_alignment_plan.md.
_PR6H_SUPPRESSION_RATIONALE = (
    "5410 benchmark `pr_6h` row suppressed by default (own-track only) due "
    "to (a) upstream forward-z-score transform anomaly at "
    "`infer_sfno5410_blocking_h100_packed.py:348-349` (also "
    "`infer_sfno5410_byo_ic.py:425-432`) and (b) own-vs-5410 truth-unit "
    "convention mismatch (own m/s vs 5410 6-hour proxy, observed truth-stats "
    "ratio ~3,600-4,400× — not the clean 21,600 the unit factor would imply). "
    "Pass `--pr6h-unit-align none` to restore the row."
)


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
    p.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Optional override: read channel_names from this metadata.json "
        "instead of from the inference NetCDFs. Normally not needed.",
    )
    p.add_argument(
        "--benchmark-5410-out-root",
        type=Path,
        default=None,
        help=(
            "Optional 5410 benchmark OUT_ROOT. When set and "
            "<root>/scores/nwp_scorecard_summary.csv exists, the scorecard "
            "table gains a '5410 benchmark' model row per channel and the "
            "report header records the benchmark path. (Note: with "
            "`--track own` the 5410 `pr_6h` row is suppressed by default "
            "per `--pr6h-unit-align suppress`; pass `none` to restore.) "
            "When the file is absent or empty, a loud warning is added at "
            "the top of the report and the table renders without the "
            "benchmark row."
        ),
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Optional path to the training EXP_DIR/run_num/ directory for "
            "this run. When set and <run-dir>/warmstart_provenance.txt "
            "exists, its key=value lines are embedded in a "
            "'### Warm-start provenance' subsection of the report. Silent "
            "no-op when --run-dir is unset or the sidecar is absent."
        ),
    )
    p.add_argument(
        "--track", choices=("own", "5410"), default="own",
        help=(
            "Emulator track for unit-aware caption text. Default 'own' "
            "treats pr_6h as m s^-1 in the scorecard table notes. '5410' "
            "treats pr_6h as kg m^-2 (6h accum.) in the group's native "
            "convention — used for group_clone runs and for direct 5410 "
            "evals. Does NOT alter scorecard values (scoring is in native "
            "units regardless)."
        ),
    )
    p.add_argument(
        "--pr6h-unit-align", choices=("suppress", "none"), default="suppress",
        help=(
            "How to handle the cross-track `pr_6h` row when a 5410 benchmark "
            "overlay is present and `--track own`. Default 'suppress' drops "
            "the 5410-benchmark `pr_6h` row from the RMSE and ACC scorecard "
            "tables and emits a banner citing the upstream forward-z-score "
            "anomaly and truth-side unit-convention mismatch. 'none' "
            "preserves the prior behavior (5410 `pr_6h` row present, with "
            "the older partial disclaimer). Has no effect under `--track "
            "5410` (which is already in matching group-native units)."
        ),
    )
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
    # Use scientific notation for values that would round to 0.0000 with %.4f.
    # Why: pr_6h is stored in m/s (mean ~1.7e-7, std ~2.8e-7), so RMSE lands
    # in the 1e-7 to 1e-6 range — fixed-point %.4f rendered every entry as
    # "0.0000" and made it look like a bug. Threshold 5e-5 is the round-half-up
    # boundary for %.4f. The CSV scorecard always carries full precision.
    if v != 0.0 and abs(v) < 5e-5:
        return f"{v:.3e}"
    return f"{v:.4f}"


def _render_table(
    summary: dict,
    key_channels: tuple[str, ...],
    *,
    benchmark_summary: dict | None = None,
    track: str = "own",
    pr6h_unit_align: str = "suppress",
) -> str:
    """Render the §F NWP scorecard table for the 5 key channels.

    When benchmark_summary is non-None, each channel gets a "5410 benchmark"
    model row (the 5410 emulator's mean ± std for that channel/lead/metric)
    next to the own emulator + persistence rows. Cells fall back to "—" when
    the benchmark lacks the (channel, lead, metric) key (e.g. channel not
    present in the 5410 channel set).
    """
    lines = [
        "## NWP Scorecard (mean ± std over ICs)",
        "",
        "RMSE (lower is better) and ACC (higher is better) per channel × lead × baseline.",
        "Persistence on `pr_6h` is undefined (no IC value, see §C.1) and reported as `NaN`.",
    ]
    if benchmark_summary is not None:
        if track == "5410":
            lines.append(
                "All values (this run + 5410 benchmark) are in the group's "
                "native units (no unit conversion); for `pr_6h` this is "
                "`kg m^-2` per 6h. Rows are directly comparable."
            )
        elif pr6h_unit_align == "suppress":
            lines.append(
                "**Note on `pr_6h` cross-track comparison.** The 5410 "
                "benchmark row is suppressed for `pr_6h` for two compounding "
                "reasons: (a) **Prediction-side anomaly** — the upstream 5410 "
                "inference scripts apply the *forward* z-score transform to "
                "the diagnostic channel at "
                "`infer_sfno5410_blocking_h100_packed.py:348-349` (packed "
                "benchmark path, which is what this report's overlay "
                "consumes) and equivalently at "
                "`infer_sfno5410_byo_ic.py:425-432` (BYO path), asymmetric "
                "with the surface (`:343`) and upper-air (`:346`) paths that "
                "use the inverse. The on-disk 5410 `pr_6h` prediction is "
                "therefore in a transformed space and RMSE/ACC against the "
                "physical-units truth and climatology is not "
                "scalar-recoverable. (b) **Truth-side unit convention** — "
                "5410 truth uses the group's \"6-hour precip proxy\" "
                "`instantaneous_rate × 6h` "
                "(`docs/2026-05-06_group_sfno_5410_eval_plan.md:127`), while "
                "own keeps the instantaneous rate (m/s); the nominal factor "
                "would be 21,600 s/6h but the empirically observed "
                "truth-stats ratio is ~3,600-4,400× "
                "(`docs/2026-05-14_pr_6h_units_mismatch_ticket.md`) — a ~5× "
                "unexplained gap on top, so even fixing (a) would still "
                "leave an unaudited scalar between own and 5410 `pr_6h`. "
                "Own-track `pr_6h` rows remain in their native m/s. See "
                "`docs/2026-05-23_pr6h_unit_alignment_plan.md`. Pass "
                "`--pr6h-unit-align none` to restore the suppressed row."
            )
        else:
            lines.append(
                "5410 benchmark values are in the group's native units (no unit "
                "conversion); for `pr_6h` this is `kg m^-2` per 6h, so the 5410 "
                "row is not directly comparable to own-track `pr_6h` (m s^-1)."
            )
    lines.append("")
    for metric in ("rmse", "acc"):
        lines.append(f"### {metric.upper()}")
        lines.append("")
        header = ["channel", "model"] + [f"{h}h" for h in _SCORED_LEADS_H]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join("---" for _ in header) + "|")
        for ch in key_channels:
            row_specs = [("emulator", "emulator", summary)]
            if metric == "rmse":
                row_specs.append(("persistence", "persistence", summary))
            if benchmark_summary is not None and not (
                pr6h_unit_align == "suppress"
                and track == "own"
                and ch == "pr_6h"
            ):
                row_specs.append(("emulator", "5410 benchmark", benchmark_summary))
            for model_key, model_label, src in row_specs:
                cells = [ch, model_label]
                for h in _SCORED_LEADS_H:
                    rec = src.get((model_key, ch, h, metric))
                    if rec is None:
                        cells.append("—")
                    else:
                        mean, std, n = rec
                        cells.append(f"{_format_value(mean)} ± {_format_value(std)} (n={n})")
                lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def _render_masked_tas(
    summary: dict,
    *,
    benchmark_summary: dict | None = None,
) -> str:
    """Render the sea-ice-masked tas section.

    Mask convention: drops cells where truth sic >= 0.15; land + open
    ocean stay. Soft-skip if no `tas_no_ice` rows are in the summary.
    """
    if not any(k[1] == "tas_no_ice" for k in summary):
        return ""

    lines = [
        "## Sea-ice-masked tas (`tas_no_ice`, sic < 0.15)",
        "",
        "Lat-weighted RMSE/ACC restricted to land + open-ocean cells "
        "(sea-ice cells dropped per the truth `sic` field at each lead).",
        "",
    ]
    leads = (6, 24, 120, 240)
    for metric in ("rmse", "acc"):
        lines.append(f"### {metric.upper()}")
        lines.append("")
        header = ["model"] + [f"{h}h" for h in leads]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join("---" for _ in header) + "|")
        row_specs = [("emulator", "emulator", summary)]
        if metric == "rmse":
            row_specs.append(("persistence", "persistence", summary))
        if benchmark_summary is not None:
            row_specs.append(("emulator", "5410 benchmark", benchmark_summary))
        for model_key, model_label, src in row_specs:
            cells = [model_label]
            for h in leads:
                rec = src.get((model_key, "tas_no_ice", h, metric))
                if rec is None:
                    cells.append("—")
                else:
                    mean, std, n = rec
                    cells.append(f"{_format_value(mean)} ± {_format_value(std)} (n={n})")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("Figures: `tas` is one of the four panels in "
                 "`figures/rmse_vs_lead.png` and `figures/acc_vs_lead.png` "
                 "(the sea-ice-masked `tas_no_ice` single-panel companions are "
                 "`figures/rmse_vs_lead_tas_no_ice.png` and "
                 "`figures/acc_vs_lead_tas_no_ice.png`).")
    lines.append("")
    return "\n".join(lines)


def _render_gate(summary: dict, z500_id: str, z500_label: str) -> str:
    """Re-derive the gate verdict from the summary CSV.

    The Z500 channel id (``z500_id``) and human-readable label
    (``z500_label`` — "Z500 (literal)" for v10, "Z500 (sigma proxy, v9)"
    for v9) come from the channel-adaptive resolver; this matches what
    score_nwp.py prints to stderr.
    """
    em_tas_6h = summary.get(("emulator", "tas", 6, "rmse"), (float("nan"),))[0]
    pers_tas_6h = summary.get(("persistence", "tas", 6, "rmse"), (float("nan"),))[0]
    em_z500_24h = summary.get(("emulator", z500_id, 24, "acc"), (float("nan"),))[0]

    pass1 = em_tas_6h < pers_tas_6h if (em_tas_6h == em_tas_6h) else False
    pass2 = em_z500_24h > 0.6 if (em_z500_24h == em_z500_24h) else False
    overall = "PASS" if (pass1 and pass2) else "FAIL"

    return (
        "## Sanity Gate (§D.6)\n\n"
        f"- Emulator RMSE on `tas` at 6 h = `{_format_value(em_tas_6h)}` "
        f"vs persistence `{_format_value(pers_tas_6h)}` → "
        f"**{'PASS' if pass1 else 'FAIL'}**\n"
        f"- Emulator ACC on `{z500_id}` ({z500_label}) at 24 h = "
        f"`{_format_value(em_z500_24h)}` vs threshold `0.6` → "
        f"**{'PASS' if pass2 else 'FAIL'}**\n\n"
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

    warmstart_block = _render_warmstart_provenance(getattr(args, "run_dir", None))
    if warmstart_block:
        lines.append(warmstart_block)
    return "\n".join(lines)


def _render_warmstart_provenance(run_dir: Path | None) -> str:
    """Render the warm-start provenance subsection from a sidecar, if any.

    The sidecar (``$RUN_DIR/warmstart_provenance.txt``) is written by
    ``train_plasim.py`` for warm-started runs. Non-warmstart runs simply
    omit the file; this function returns "" in that case.
    """
    if run_dir is None:
        return ""
    sidecar = run_dir / "warmstart_provenance.txt"
    if not sidecar.is_file():
        return ""
    pairs: list[tuple[str, str]] = []
    for raw in sidecar.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        pairs.append((key.strip(), value.strip()))
    if not pairs:
        return ""
    lines = ["### Warm-start provenance", "",
             f"Source sidecar: `{sidecar}`", "",
             "| key | value |", "|---|---|"]
    for key, value in pairs:
        lines.append(f"| `{key}` | `{value}` |")
    lines.append("")
    return "\n".join(lines)


def _load_benchmark(
    bench_root: Path | None,
    *,
    track: str = "own",
    pr6h_unit_align: str = "suppress",
) -> tuple[dict | None, str]:
    """Load 5410 benchmark summary if available.

    Returns (summary_or_None, banner_text). banner_text is a markdown
    fragment to insert near the top of the report — empty string when no
    benchmark was requested, a status line when one was loaded, or a loud
    warning when the benchmark was requested but unavailable.
    """
    if bench_root is None:
        return None, ""
    sc = bench_root / "scores" / "nwp_scorecard_summary.csv"
    if not sc.is_file():
        msg = (
            f"⚠️ **5410 benchmark unavailable.** Requested benchmark "
            f"`{bench_root}` has no `scores/nwp_scorecard_summary.csv`. "
            f"Report rendered own-only.\n"
        )
        logging.warning("benchmark scorecard missing at %s — rendering own-only", sc)
        return None, msg
    summary = _read_summary(sc)
    if not summary:
        msg = (
            f"⚠️ **5410 benchmark scorecard is empty.** "
            f"`{sc}` has no rows. Report rendered own-only.\n"
        )
        logging.warning("benchmark scorecard at %s is empty — rendering own-only", sc)
        return None, msg
    if track == "own" and pr6h_unit_align == "suppress":
        banner = (
            f"**5410 benchmark:** `{bench_root}` "
            f"(scorecard: `{sc.relative_to(bench_root)}`). Side-by-side rows "
            f"appear in the scorecard table (**5410 benchmark `pr_6h` row "
            f"suppressed by default** — see the note above the table; "
            f"own-track `pr_6h` rows remain in native m/s); bias maps "
            f"overlay the benchmark in the figures job.\n"
        )
    else:
        banner = (
            f"**5410 benchmark:** `{bench_root}` "
            f"(scorecard: `{sc.relative_to(bench_root)}`). Side-by-side rows "
            f"appear in the scorecard table; line plots and bias maps overlay "
            f"the benchmark in the figures job.\n"
        )
    return summary, banner


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    scores_dir = args.out_root / "scores"
    summary_path = scores_dir / "nwp_scorecard_summary.csv"
    if not summary_path.is_file():
        raise SystemExit(f"missing {summary_path}; run score_nwp.py first")

    summary = _read_summary(summary_path)
    benchmark_summary, benchmark_banner = _load_benchmark(
        args.benchmark_5410_out_root,
        track=args.track,
        pr6h_unit_align=args.pr6h_unit_align,
    )

    # Resolve channel-name list once, from inference NetCDFs (per plan
    # §3.10), then derive the Z500 channel id and the 5-key-channel list
    # for the scorecard table.
    nc_dir = args.out_root / "inference" / "nwp"
    channel_names = resolve_channel_names(
        nc_dir / "*.nc", metadata_json_override=args.metadata_json
    )
    z500_id, z500_label = detect_z500_channel(channel_names)
    key_channels = bias_channels(channel_names)

    parts = [
        _render_header(args),
        benchmark_banner,
        _render_table(
            summary,
            key_channels,
            benchmark_summary=benchmark_summary,
            track=args.track,
            pr6h_unit_align=args.pr6h_unit_align,
        ),
        _render_masked_tas(summary, benchmark_summary=benchmark_summary),
        _render_gate(summary, z500_id, z500_label),
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
