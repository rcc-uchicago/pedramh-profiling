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


def plot_lines(
    summary: dict,
    metric: str,
    out_path: Path,
    *,
    include_persistence: bool,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    n_ics = summary[("emulator", "tas", 6, metric)][2]
    leads_d = [_lead_days(h) for h in REPORT_LEADS]
    for ax, ch in zip(axes, LINE_PLOT_CHANNELS):
        scale = CHANNEL_UNIT_SCALE.get(ch, 1.0) if metric == "rmse" else 1.0
        for model, color in [("emulator", "C0"), ("persistence", "C1")]:
            if model == "persistence" and not include_persistence:
                continue
            means: list[float] = []
            stds: list[float] = []
            has_data = False
            for L in REPORT_LEADS:
                key = (model, ch, L, metric)
                if key in summary:
                    m, s, _ = summary[key]
                    means.append(m)
                    stds.append(s)
                    has_data = True
                else:
                    means.append(float("nan"))
                    stds.append(float("nan"))
            if not has_data:
                continue
            m_arr = np.array(means) * scale
            s_arr = np.array(stds) * scale
            ax.plot(leads_d, m_arr, "o-", color=color, label=model)
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
    fig.suptitle(
        f"NWP {metric.upper()} vs lead time (n={n_ics} ICs) — pr_6h omitted "
        f"({metric.upper()} not appropriate for intermittent precipitation)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_bias(out_root: Path, channel: str, fig_path: Path) -> None:
    fig, axes = plt.subplots(1, 6, figsize=(22, 3.6))
    scale = CHANNEL_UNIT_SCALE.get(channel, 1.0)
    unit = CHANNEL_UNITS.get(channel, "")
    arrs: list[np.ndarray] = []
    for L in REPORT_LEADS:
        arr = np.load(out_root / "scores" / f"bias_maps_{channel}_{L}h.npy") * scale
        arrs.append(arr)
    vmax = max(float(np.abs(a).max()) for a in arrs)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    im = None
    for ax, L, arr in zip(axes, REPORT_LEADS, arrs):
        im = ax.imshow(
            arr,
            origin="upper",
            extent=[0, 360, -90, 90],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_title(_lead_label(L))
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_yticks([-90, -45, 0, 45, 90])
        ax.set_xlabel("lon")
    axes[0].set_ylabel("lat")
    cbar = fig.colorbar(im, ax=list(axes), shrink=0.85, pad=0.02)
    cbar_label = f"{channel} bias (pred - truth)"
    if unit:
        cbar_label += f" [{unit}]"
    cbar.set_label(cbar_label)
    fig.suptitle(
        f"Bias maps - {CHANNEL_LABELS.get(channel, channel)}  "
        f"(symmetric colorbar, |max|={vmax:.3g}{(' ' + unit) if unit else ''})"
    )
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", required=True, type=Path)
    args = ap.parse_args()

    fig_dir = args.out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(args.out_root / "scores" / "nwp_scorecard_summary.csv")

    plot_lines(summary, "rmse", fig_dir / "rmse_vs_lead.png", include_persistence=True)
    plot_lines(summary, "acc", fig_dir / "acc_vs_lead.png", include_persistence=False)
    for ch in REPORT_CHANNELS:
        plot_bias(args.out_root, ch, fig_dir / f"bias_{ch}.png")

    n_figs = 2 + len(REPORT_CHANNELS)
    print(f"[render_eval_figures] wrote {n_figs} figures to {fig_dir}")


if __name__ == "__main__":
    main()
