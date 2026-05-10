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
    "tas":   "tas (Near-surface 2-m air temperature)",
    "pr_6h": "pr_6h (Precipitation flux, 6-h)",
    "zg500": "zg500 (500 hPa geopotential height, Z500)",
    "ua5":   "ua5 (Zonal wind, sigma level 5)",
    "ta5":   "ta5 (Air temperature, sigma level 5)",
}

# Display units for RMSE / bias-map labels. PlaSim native units are taken from
# the postproc NetCDF metadata; pr_6h is converted from m s^-1 to mm day^-1
# (factor 1000 mm/m * 86400 s/day = 8.64e7) for readable axis numbers.
CHANNEL_UNITS: dict[str, str] = {
    "tas":   "K",
    "pr_6h": "mm day$^{-1}$",
    "zg500": "m",
    "ua5":   "m s$^{-1}$",
    "ta5":   "K",
}

CHANNEL_UNIT_SCALE: dict[str, float] = {
    "pr_6h": 86400.0 * 1000.0,  # m s^-1 -> mm day^-1
}


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
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    n_ics = summary[("emulator", "tas", 6, metric)][2]
    leads_d = [_lead_days(h) for h in REPORT_LEADS]
    for ax, ch in zip(axes, LINE_PLOT_CHANNELS):
        scale = CHANNEL_UNIT_SCALE.get(ch, 1.0) if metric == "rmse" else 1.0
        series = [("emulator", "own", "C0", summary, scale)]
        if include_persistence:
            series.append(("persistence", "own persistence", "C1", summary, scale))
        if benchmark_summary is not None:
            # 5410 benchmark always uses native units (group convention; no
            # m/s -> mm/day scaling). Since pr_6h is excluded from the line
            # plot channels above, scale=1.0 here is correct for all channels.
            series.append(("emulator", "5410 benchmark", "C2", benchmark_summary, 1.0))
        for model, label, color, src, src_scale in series:
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
            ax.plot(leads_d, m_arr, "o-", color=color, label=label)
            ax.fill_between(
                leads_d, m_arr - s_arr, m_arr + s_arr, color=color, alpha=0.2
            )
        ax.set_title(CHANNEL_LABELS.get(ch, ch), fontsize=10)
        ax.set_xlabel("lead time (days)")
        if metric == "rmse":
            unit = CHANNEL_UNITS.get(ch)
            ax.set_ylabel(f"RMSE ({unit})" if unit else "RMSE")
        else:
            ax.set_ylabel(metric.upper())
        ax.set_xticks(leads_d)
        ax.set_xticklabels([f"{d:g}" for d in leads_d])
        ax.tick_params(axis="x", labelrotation=0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    suffix = " — 5410 benchmark overlaid" if benchmark_summary is not None else ""
    fig.suptitle(
        f"NWP {metric.upper()} vs lead time (n={n_ics} ICs){suffix} — pr_6h omitted "
        f"({metric.upper()} not appropriate for intermittent precipitation)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


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
    """Render one row of 6 bias-map panels. Returns the last imshow handle."""
    im = None
    for ax, L, arr in zip(axes, REPORT_LEADS, arrs):
        im = ax.imshow(
            arr, origin="upper", extent=[0, 360, -90, 90],
            cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
        )
        ax.set_title(_lead_label(L))
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_yticks([-90, -45, 0, 45, 90])
        if show_xlabel:
            ax.set_xlabel("lon")
    axes[0].set_ylabel(f"{row_label}\nlat")
    return im


def plot_bias(out_root: Path, channel: str, fig_path: Path,
              *, benchmark_root: Path | None = None) -> None:
    """Render the bias-map figure for one channel.

    Default (no benchmark): single 1x6 row of own-track panels.
    With benchmark: 2x6 stacked layout (own on top, 5410 on bottom). Each
    row keeps its own symmetric colorbar so unit differences (e.g. own-track
    pr_6h scaled to mm/day vs 5410 pr_6h native kg/m^2 per 6h) are honest.
    """
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

    if bench_arrs is None:
        # Single-row layout (original).
        fig, axes = plt.subplots(1, 6, figsize=(22, 3.6))
        im = _draw_bias_row(axes, own_arrs, own_vmax,
                            row_label="own", show_xlabel=True)
        cbar = fig.colorbar(im, ax=list(axes), shrink=0.85, pad=0.02)
        cbar_label = f"{channel} bias (pred - truth)"
        if own_unit:
            cbar_label += f" [{own_unit}]"
        cbar.set_label(cbar_label)
        fig.suptitle(
            f"Bias maps - {CHANNEL_LABELS.get(channel, channel)}  "
            f"(symmetric colorbar, |max|={own_vmax:.3g}{(' ' + own_unit) if own_unit else ''})"
        )
    else:
        # Two-row layout: own on top, 5410 on bottom, separate colorbars.
        bench_vmax = _row_vmax(bench_arrs)
        fig, axes2d = plt.subplots(2, 6, figsize=(22, 7.2))
        im_own = _draw_bias_row(axes2d[0], own_arrs, own_vmax,
                                row_label="own", show_xlabel=False)
        im_bn = _draw_bias_row(axes2d[1], bench_arrs, bench_vmax,
                               row_label="5410 benchmark", show_xlabel=True)
        cb_own = fig.colorbar(im_own, ax=list(axes2d[0]), shrink=0.85, pad=0.02)
        cb_bn = fig.colorbar(im_bn, ax=list(axes2d[1]), shrink=0.85, pad=0.02)
        own_label = f"{channel} bias (pred - truth)"
        if own_unit:
            own_label += f" [{own_unit}]"
        cb_own.set_label(own_label)
        bn_label = f"{channel} bias (pred - truth)"
        if bench_unit:
            bn_label += f" [{bench_unit}]"
        cb_bn.set_label(bn_label)
        fig.suptitle(
            f"Bias maps - {CHANNEL_LABELS.get(channel, channel)}  "
            f"(own |max|={own_vmax:.3g}{(' ' + own_unit) if own_unit else ''}; "
            f"5410 |max|={bench_vmax:.3g}{(' ' + bench_unit) if bench_unit else ''})"
        )

    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
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
            "own-only. Ignored when --track=5410."
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
    if args.track == "own" and args.benchmark_5410_out_root is not None:
        bench_summary = _load_benchmark_summary(args.benchmark_5410_out_root)
        if bench_summary is not None:
            bench_root_for_bias = args.benchmark_5410_out_root

    plot_lines(summary, "rmse", fig_dir / "rmse_vs_lead.png",
               include_persistence=True, benchmark_summary=bench_summary)
    plot_lines(summary, "acc", fig_dir / "acc_vs_lead.png",
               include_persistence=False, benchmark_summary=bench_summary)
    for ch in REPORT_CHANNELS:
        plot_bias(args.out_root, ch, fig_dir / f"bias_{ch}.png",
                  benchmark_root=bench_root_for_bias)

    n_figs = 2 + len(REPORT_CHANNELS)
    overlay_msg = " (5410 benchmark overlaid)" if bench_summary is not None else ""
    print(f"[render_eval_figures] wrote {n_figs} figures to {fig_dir}{overlay_msg}")


if __name__ == "__main__":
    main()
