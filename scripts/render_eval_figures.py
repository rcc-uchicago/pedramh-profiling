#!/usr/bin/env python3
"""Render figures for the SFNO eval report.

Reads ``scores/nwp_scorecard_summary.csv`` and ``scores/bias_maps_*.npy``
under ``$OUT_ROOT`` and writes PNGs into ``$OUT_ROOT/figures/``:

  - ``rmse_vs_lead.png`` — 5 panels (one per report channel), emulator vs
    persistence with +/- 1 sigma shading.
  - ``acc_vs_lead.png``  — same layout, emulator only.
  - ``bias_<channel>.png`` (x5) — 1x6 panels of (lat, lon) bias maps for
    the 6 report leads, symmetric colorbar shared per channel.

Channels and leads match the report scorecard in ``render_eval_report.py``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT_CHANNELS = ["tas", "pr_6h", "zg500", "ua5", "ta5"]
# ACC and RMSE on intermittent, heavy-tailed precipitation are misleading:
# zero-inflation drives apparent ACC near 1 from spatial sparsity alone, and
# RMSE is dominated by a few heavy events. Line plots therefore exclude pr_6h;
# bias maps still include it (spatial mean error is well-defined).
LINE_PLOT_CHANNELS = ["tas", "zg500", "ua5", "ta5"]
REPORT_LEADS = [6, 24, 72, 120, 240, 336]  # hours (used for npy/CSV lookup)

CHANNEL_LABELS: dict[str, str] = {
    "tas":        "tas (Near-surface 2-m air temperature)",
    "tas_no_ice": "tas (ice-free cells, sic<0.15)",
    "pr_6h":      "pr_6h (Precipitation flux, 6-h)",
    "zg500":      "zg500 (500 hPa geopotential height, Z500)",
    "ua5":        "ua5 (Zonal wind, sigma level 5)",
    "ta5":        "ta5 (Air temperature, sigma level 5)",
}

# Display units for RMSE / bias-map labels. PlaSim native units are taken from
# the postproc NetCDF metadata; pr_6h is converted from m s^-1 to mm day^-1
# (factor 1000 mm/m * 86400 s/day = 8.64e7) for readable axis numbers.
CHANNEL_UNITS: dict[str, str] = {
    "tas":        "K",
    "tas_no_ice": "K",
    "pr_6h":      "mm day$^{-1}$",
    "zg500":      "m",
    "ua5":        "m s$^{-1}$",
    "ta5":        "K",
}

CHANNEL_UNIT_SCALE: dict[str, float] = {
    "pr_6h": 86400.0 * 1000.0,  # m s^-1 -> mm day^-1
}


# Slide-grade styling table for the three series rendered on the line plots.
# Colorblind-friendly Wong-style palette; linestyle + marker provide a second
# channel of distinction so the figure also survives B/W reproduction.
# Emulator (the protagonist) gets the heaviest stroke; persistence baseline
# is intentionally faded; 5410 benchmark uses a contrasting orange.
SERIES_STYLE: dict[str, dict] = {
    "own": {
        "color": "#0072B2",  # blue
        "ls": "-",
        "lw": 2.8,
        "marker": "o",
        "ms": 7.5,
        "mew": 0.0,
        "alpha": 1.0,
        "fill_alpha": 0.18,
        "zorder": 3,
    },
    "own persistence": {
        "color": "#6E6E6E",  # neutral gray
        "ls": (0, (5, 3)),   # dashed
        "lw": 1.6,
        "marker": "x",
        "ms": 6.0,
        "mew": 1.6,
        "alpha": 0.85,
        "fill_alpha": 0.07,
        "zorder": 2,
    },
    "5410 benchmark": {
        "color": "#D55E00",  # vermillion
        "ls": (0, (4, 2, 1, 2)),  # dash-dot
        "lw": 2.2,
        "marker": "s",
        "ms": 6.5,
        "mew": 0.0,
        "alpha": 1.0,
        "fill_alpha": 0.12,
        "zorder": 2,
    },
}


def _apply_slide_style() -> None:
    """Idempotently set matplotlib rcParams for slide-ready figures."""
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 160,
        "font.size": 12.5,
        "axes.titlesize": 13.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 12.5,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 12,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "lines.solid_capstyle": "round",
    })


def _lead_days(h: int) -> float:
    return h / 24.0


def _lead_label(h: int) -> str:
    d = _lead_days(h)
    return f"{d:g} d"


def load_summary(path: Path) -> dict[tuple[str, str, int, str], tuple[float, float, int]]:
    """Return ``{(model, channel, lead_h, metric): (mean, std, n_ics)}``."""
    out: dict[tuple[str, str, int, str], tuple[float, float, int]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["model"], row["channel"], int(row["lead_hours"]), row["metric"])
            out[key] = (float(row["mean"]), float(row["std"]), int(row["n_ics"]))
    return out


def _load_benchmark_summary(bench_root: Path | None) -> dict | None:
    """Load the 5410 benchmark scorecard summary if available, else None.

    Loud-warns to stderr when the path is set but the scorecard is missing
    or empty, then returns None so the rest of the pipeline continues
    without an overlay.
    """
    import sys
    if bench_root is None:
        return None
    sc = bench_root / "scores" / "nwp_scorecard_summary.csv"
    if not sc.is_file():
        print(f"[render_eval_figures] WARN: benchmark scorecard missing at {sc} — "
              f"rendering own-only", file=sys.stderr)
        return None
    summary = load_summary(sc)
    if not summary:
        print(f"[render_eval_figures] WARN: benchmark scorecard at {sc} is empty — "
              f"rendering own-only", file=sys.stderr)
        return None
    return summary


def plot_lines(
    summary: dict,
    metric: str,
    out_path: Path,
    *,
    include_persistence: bool,
    benchmark_summary: dict | None = None,
) -> None:
    _apply_slide_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.8))
    axes_flat = axes.flatten()
    n_ics_keys = [("emulator", ch, 6, metric) for ch in LINE_PLOT_CHANNELS]
    n_ics = next((summary[k][2] for k in n_ics_keys if k in summary),
                 summary.get(("emulator", "tas", 6, metric), (None, None, 0))[2])
    leads_d = [_lead_days(h) for h in REPORT_LEADS]
    metric_pretty = "RMSE" if metric == "rmse" else "ACC"

    handles_by_label: dict[str, object] = {}
    for ax, ch in zip(axes_flat, LINE_PLOT_CHANNELS):
        scale = CHANNEL_UNIT_SCALE.get(ch, 1.0) if metric == "rmse" else 1.0
        series = [("emulator", "own", summary, scale)]
        if include_persistence:
            series.append(("persistence", "own persistence", summary, scale))
        if benchmark_summary is not None:
            # 5410 benchmark always uses native units (group convention; no
            # m/s -> mm/day scaling). Since pr_6h is excluded from the line
            # plot channels above, scale=1.0 here is correct for all channels.
            series.append(("emulator", "5410 benchmark", benchmark_summary, 1.0))
        for model, label, src, src_scale in series:
            style = SERIES_STYLE[label]
            means: list[float] = []
            stds: list[float] = []
            has_data = False
            for L in REPORT_LEADS:
                key = (model, ch, L, metric)
                if key in src:
                    m, s, _ = src[key]
                    means.append(m)
                    stds.append(s)
                    has_data = True
                else:
                    means.append(float("nan"))
                    stds.append(float("nan"))
            if not has_data:
                continue
            m_arr = np.array(means) * src_scale
            s_arr = np.array(stds) * src_scale
            (line,) = ax.plot(
                leads_d, m_arr,
                color=style["color"],
                linestyle=style["ls"],
                linewidth=style["lw"],
                marker=style["marker"],
                markersize=style["ms"],
                markeredgewidth=style["mew"],
                markerfacecolor=style["color"],
                alpha=style["alpha"],
                zorder=style["zorder"],
                label=label,
            )
            ax.fill_between(
                leads_d, m_arr - s_arr, m_arr + s_arr,
                color=style["color"], alpha=style["fill_alpha"],
                linewidth=0, zorder=style["zorder"] - 1,
            )
            handles_by_label.setdefault(label, line)

        ax.set_title(CHANNEL_LABELS.get(ch, ch))
        if metric == "rmse":
            unit = CHANNEL_UNITS.get(ch)
            ax.set_ylabel(f"RMSE ({unit})" if unit else "RMSE")
            ax.set_ylim(bottom=0)
        else:
            ax.set_ylabel("ACC")
            ax.set_ylim(top=1.03)
            ax.axhline(0.0, color="#999", linewidth=0.8, linestyle=":", zorder=1)
        ax.set_xticks(leads_d)
        ax.set_xticklabels([f"{d:g}" for d in leads_d])
        ax.tick_params(axis="x", labelrotation=0)
        ax.margins(x=0.03)
        ax.grid(True, which="major", axis="both", alpha=0.25, linestyle="--")

    # Only label the x axis on the bottom row to reduce clutter.
    for ax in axes[-1, :]:
        ax.set_xlabel("lead time (days)")
    for ax in axes[0, :]:
        ax.set_xlabel("")

    bench_suffix = "   (with SFNO-5410 benchmark overlay)" if benchmark_summary is not None else ""
    fig.suptitle(
        f"NWP {metric_pretty} vs lead time{bench_suffix}",
        fontsize=16.5, fontweight="bold", y=0.985,
    )

    # Unified legend at the bottom (single source of truth for all 4 panels).
    fig.legend(
        handles=list(handles_by_label.values()),
        labels=list(handles_by_label.keys()),
        loc="lower center",
        ncol=len(handles_by_label),
        bbox_to_anchor=(0.5, 0.045),
        frameon=False,
        handlelength=2.6,
        columnspacing=2.2,
    )

    # Methodology footnote — small, gray, single line.
    fig.text(
        0.5, 0.008,
        f"n = {n_ics} initial conditions   ·   pr_6h omitted "
        f"({metric_pretty} not appropriate for intermittent precipitation)",
        ha="center", va="bottom", fontsize=10, color="#555555", style="italic",
    )

    fig.tight_layout(rect=(0, 0.10, 1, 0.955))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_single_channel_lines(
    summary: dict,
    metric: str,
    out_path: Path,
    *,
    channel: str,
    include_persistence: bool,
    benchmark_summary: dict | None = None,
) -> bool:
    """Render a single-panel line plot for ``channel`` vs lead.

    Returns True if the figure was written, False if no rows for
    ``channel`` exist in the summary (caller skips).
    """
    if not any(k[1] == channel for k in summary):
        return False

    _apply_slide_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.0))
    leads_d = [_lead_days(h) for h in REPORT_LEADS]
    metric_pretty = "RMSE" if metric == "rmse" else "ACC"
    n_ics_rec = summary.get(("emulator", channel, 6, metric))
    n_ics = n_ics_rec[2] if n_ics_rec is not None else 0

    series = [("emulator", "own", summary)]
    if include_persistence:
        series.append(("persistence", "own persistence", summary))
    if benchmark_summary is not None and any(
        k[1] == channel for k in benchmark_summary
    ):
        series.append(("emulator", "5410 benchmark", benchmark_summary))

    handles_by_label: dict[str, object] = {}
    for model, label, src in series:
        style = SERIES_STYLE[label]
        means: list[float] = []
        stds: list[float] = []
        has_data = False
        for L in REPORT_LEADS:
            key = (model, channel, L, metric)
            if key in src:
                m, s, _ = src[key]
                means.append(m)
                stds.append(s)
                has_data = True
            else:
                means.append(float("nan"))
                stds.append(float("nan"))
        if not has_data:
            continue
        m_arr = np.array(means)
        s_arr = np.array(stds)
        (line,) = ax.plot(
            leads_d, m_arr,
            color=style["color"],
            linestyle=style["ls"],
            linewidth=style["lw"],
            marker=style["marker"],
            markersize=style["ms"],
            markeredgewidth=style["mew"],
            markerfacecolor=style["color"],
            alpha=style["alpha"],
            zorder=style["zorder"],
            label=label,
        )
        ax.fill_between(
            leads_d, m_arr - s_arr, m_arr + s_arr,
            color=style["color"], alpha=style["fill_alpha"],
            linewidth=0, zorder=style["zorder"] - 1,
        )
        handles_by_label.setdefault(label, line)

    ax.set_title(CHANNEL_LABELS.get(channel, channel))
    if metric == "rmse":
        unit = CHANNEL_UNITS.get(channel, "")
        ax.set_ylabel(f"RMSE ({unit})" if unit else "RMSE")
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylabel("ACC")
        ax.set_ylim(top=1.03)
        ax.axhline(0.0, color="#999", linewidth=0.8, linestyle=":", zorder=1)
    ax.set_xticks(leads_d)
    ax.set_xticklabels([f"{d:g}" for d in leads_d])
    ax.set_xlabel("lead time (days)")
    ax.margins(x=0.03)
    ax.grid(True, which="major", axis="both", alpha=0.25, linestyle="--")

    bench_suffix = "   (with SFNO-5410 benchmark overlay)" if (
        benchmark_summary is not None and "5410 benchmark" in handles_by_label
    ) else ""
    if channel == "tas_no_ice":
        title_lead = "Sea-ice-masked tas"
        footnote = (f"n = {n_ics} ICs   ·   mask: sic >= 0.15 dropped per "
                    "truth at each lead (land kept; NaN sic treated as land)")
    else:
        title_lead = CHANNEL_LABELS.get(channel, channel).split(" (")[0]
        footnote = f"n = {n_ics} ICs"
    fig.suptitle(
        f"{title_lead} {metric_pretty} vs lead time{bench_suffix}",
        fontsize=14, fontweight="bold",
    )
    fig.legend(
        handles=list(handles_by_label.values()),
        labels=list(handles_by_label.keys()),
        loc="lower center",
        ncol=len(handles_by_label),
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        handlelength=2.6,
        columnspacing=2.2,
    )
    fig.text(
        0.5, -0.04,
        footnote,
        ha="center", va="top", fontsize=9, color="#555555", style="italic",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _load_bias_row(scores_dir: Path, channel: str, scale: float) -> list[np.ndarray] | None:
    """Load 6 bias maps for one channel, scaled. Return None if any are missing."""
    arrs: list[np.ndarray] = []
    for L in REPORT_LEADS:
        p = scores_dir / f"bias_maps_{channel}_{L}h.npy"
        if not p.is_file():
            return None
        arrs.append(np.load(p) * scale)
    return arrs


def _row_vmax(arrs: list[np.ndarray]) -> float:
    vmax = max(float(np.abs(a).max()) for a in arrs)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return vmax


def _draw_bias_row(axes, arrs: list[np.ndarray], vmax: float, *,
                   row_label: str, show_xlabel: bool):
    """Render one row of 6 bias-map panels. Returns the last imshow handle.

    Slide-grade conventions:
      - lon ticks at 0/180/360 with "0°" / "180°" / "0°" wraparound labels
      - lat ticks at -90/-45/0/45/90 with N/S hemisphere suffixes
      - leftmost panel carries a bold row label as the y-axis title;
        non-leftmost panels suppress tick labels for cleanliness
      - bottom row shows lon tick labels; top row (in 2-row layout) hides them
    """
    im = None
    for i, (ax, L, arr) in enumerate(zip(axes, REPORT_LEADS, arrs)):
        im = ax.imshow(
            arr, origin="upper", extent=[0, 360, -90, 90],
            cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
        )
        ax.set_title(_lead_label(L))
        ax.set_xticks([0, 180, 360])
        ax.set_yticks([-90, -45, 0, 45, 90])
        if show_xlabel:
            ax.set_xticklabels(["0°", "180°", "0°"])
        else:
            ax.set_xticklabels([])
        if i == 0:
            ax.set_yticklabels(["90°S", "45°S", "0°", "45°N", "90°N"])
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="both", which="both", length=3)
        ax.grid(False)
    axes[0].set_ylabel(row_label, fontsize=13, fontweight="semibold", labelpad=14)
    return im


def plot_bias(out_root: Path, channel: str, fig_path: Path,
              *, benchmark_root: Path | None = None) -> None:
    """Render the bias-map figure for one channel.

    Default (no benchmark): single 1x6 row of own-track panels.
    With benchmark: 2x6 stacked layout (own on top, 5410 on bottom). Each
    row keeps its own symmetric colorbar so unit differences (e.g. own-track
    pr_6h scaled to mm/day vs 5410 pr_6h native kg/m^2 per 6h) are honest.
    """
    _apply_slide_style()
    own_scale = CHANNEL_UNIT_SCALE.get(channel, 1.0)
    own_unit = CHANNEL_UNITS.get(channel, "")
    own_arrs = _load_bias_row(out_root / "scores", channel, own_scale)
    if own_arrs is None:
        # Match prior behavior: caller doesn't expect FileNotFound silently.
        np.load(out_root / "scores" / f"bias_maps_{channel}_{REPORT_LEADS[0]}h.npy")
        return
    own_vmax = _row_vmax(own_arrs)

    # Benchmark row: 5410 always uses native units (no scaling). For pr_6h
    # the units differ from own (kg/m^2 per 6h vs mm/day) — separate
    # colorbar per row keeps the comparison honest.
    bench_arrs = None
    bench_unit = ""
    if benchmark_root is not None:
        bench_arrs = _load_bias_row(benchmark_root / "scores", channel, 1.0)
        if bench_arrs is not None:
            if channel == "pr_6h":
                bench_unit = "kg m$^{-2}$ (6h accum.)"
            else:
                bench_unit = own_unit

    cbar_base = f"{channel} bias (pred - truth)"
    own_cb_label = cbar_base + (f" [{own_unit}]" if own_unit else "")
    bench_cb_label = cbar_base + (f" [{bench_unit}]" if bench_unit else "")

    own_max_blurb = f"own |max| = {own_vmax:.3g}{(' ' + own_unit) if own_unit else ''}"

    if bench_arrs is None:
        # Single-row layout. constrained_layout natively reserves room for
        # the colorbar attached to all axes (tight_layout warns and misplaces).
        fig, axes = plt.subplots(1, 6, figsize=(22, 4.2),
                                 constrained_layout=True)
        im = _draw_bias_row(axes, own_arrs, own_vmax,
                            row_label="own", show_xlabel=True)
        cbar = fig.colorbar(im, ax=list(axes), shrink=0.92, aspect=30, pad=0.015)
        cbar.set_label(own_cb_label, fontsize=12)
        cbar.ax.tick_params(labelsize=10.5)
        fig.suptitle(
            f"Bias maps — {CHANNEL_LABELS.get(channel, channel)}   ·   "
            f"{own_max_blurb}",
            fontsize=14.5, fontweight="bold",
        )
    else:
        # Two-row layout: own on top, 5410 on bottom, separate colorbars.
        bench_vmax = _row_vmax(bench_arrs)
        bench_max_blurb = (
            f"5410 |max| = {bench_vmax:.3g}"
            f"{(' ' + bench_unit) if bench_unit else ''}"
        )
        fig, axes2d = plt.subplots(2, 6, figsize=(22, 8.4),
                                   constrained_layout=True)
        im_own = _draw_bias_row(axes2d[0], own_arrs, own_vmax,
                                row_label="own", show_xlabel=False)
        im_bn = _draw_bias_row(axes2d[1], bench_arrs, bench_vmax,
                               row_label="5410 benchmark", show_xlabel=True)
        cb_own = fig.colorbar(im_own, ax=list(axes2d[0]), shrink=0.92,
                              aspect=22, pad=0.015)
        cb_bn = fig.colorbar(im_bn, ax=list(axes2d[1]), shrink=0.92,
                             aspect=22, pad=0.015)
        cb_own.set_label(own_cb_label, fontsize=12)
        cb_bn.set_label(bench_cb_label, fontsize=12)
        cb_own.ax.tick_params(labelsize=10.5)
        cb_bn.ax.tick_params(labelsize=10.5)
        fig.suptitle(
            f"Bias maps — {CHANNEL_LABELS.get(channel, channel)}   ·   "
            f"{own_max_blurb}   ·   {bench_max_blurb}",
            fontsize=14.5, fontweight="bold",
        )

    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument(
        "--track", choices=("own", "5410"), default="own",
        help=(
            "Emulator track. Default 'own' uses the existing m s^-1 → mm day^-1 "
            "scaling for pr_6h. '5410' clears CHANNEL_UNIT_SCALE and re-labels "
            "pr_6h as 'kg m^-2 (6h accum.)' because 5410 outputs are already "
            "in rate × 6h kg/m² (group convention; do NOT unit-convert)."
        ),
    )
    ap.add_argument(
        "--benchmark-5410-out-root", type=Path, default=None,
        help=(
            "Optional 5410 benchmark OUT_ROOT. When set and the scorecard at "
            "<root>/scores/nwp_scorecard_summary.csv exists, every line plot "
            "gains a 5410 overlay and every bias map becomes a 2-row layout "
            "(own on top, 5410 on bottom; separate colorbars). When the file "
            "is absent or empty, a loud warning prints and figures render "
            "own-only. Honored for both --track=own (default mixed-units "
            "layout) and --track=5410 (group-clone case: both rows in the "
            "group's native kg m^-2 (6h accum.) units, same colorbar scale)."
        ),
    )
    args = ap.parse_args()

    if args.track == "5410":
        # Codex round-3 fix #4 / round-4 fix #3 (per
        # docs/2026-05-08_sfno_5410_scoring_plan.md v4+):
        # - Disable the m/s → mm/day scaling for pr_6h. 5410's pr_6h
        #   is already a 6h-accumulated mass per area in kg/m².
        # - Relabel the colorbar / axis so the figure is honest.
        CHANNEL_UNIT_SCALE.clear()
        CHANNEL_UNITS["pr_6h"] = r"kg m$^{-2}$ (6h accum.)"

    fig_dir = args.out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(args.out_root / "scores" / "nwp_scorecard_summary.csv")

    bench_summary = None
    bench_root_for_bias: Path | None = None
    if args.benchmark_5410_out_root is not None:
        bench_summary = _load_benchmark_summary(args.benchmark_5410_out_root)
        if bench_summary is not None:
            bench_root_for_bias = args.benchmark_5410_out_root

    plot_lines(summary, "rmse", fig_dir / "rmse_vs_lead.png",
               include_persistence=True, benchmark_summary=bench_summary)
    plot_lines(summary, "acc", fig_dir / "acc_vs_lead.png",
               include_persistence=False, benchmark_summary=bench_summary)
    wrote_masked = 0
    if plot_single_channel_lines(summary, "rmse",
                                 fig_dir / "rmse_vs_lead_tas_no_ice.png",
                                 channel="tas_no_ice",
                                 include_persistence=True,
                                 benchmark_summary=bench_summary):
        wrote_masked += 1
    if plot_single_channel_lines(summary, "acc",
                                 fig_dir / "acc_vs_lead_tas_no_ice.png",
                                 channel="tas_no_ice",
                                 include_persistence=False,
                                 benchmark_summary=bench_summary):
        wrote_masked += 1
    for ch in REPORT_CHANNELS:
        plot_bias(args.out_root, ch, fig_dir / f"bias_{ch}.png",
                  benchmark_root=bench_root_for_bias)

    n_figs = 2 + wrote_masked + len(REPORT_CHANNELS)
    overlay_msg = " (5410 benchmark overlaid)" if bench_summary is not None else ""
    print(f"[render_eval_figures] wrote {n_figs} figures to {fig_dir}{overlay_msg}")


if __name__ == "__main__":
    main()
