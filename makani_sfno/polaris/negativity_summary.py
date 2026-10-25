#!/usr/bin/env python3
"""Summarise negativity_rollout.py outputs against the same statistic in E3SM truth.

Torch-free (numpy, h5py). Two subcommands::

    python polaris/negativity_summary.py truth --pack $PACK --start 2044 1092 \\
        --n-leads 1460 --stride 4 --channels RELHUM_l00 ... --out truth_neg.npz
    python polaris/negativity_summary.py summarize --truth truth_neg.npz \\
        --csv neg.csv --md neg.md  OUT/neg_*.npz

Truth is read every ``stride`` leads (default 4 = once a day): the question is
whether the MODEL goes negative where E3SM does not, and a daily truth sample is
enough to tell "E3SM is never negative" from "E3SM is sometimes negative too".

Per (member, channel), over the leads the member ran:
  ever_neg        any lead with a negative grid cell
  first_neg_lead  first such lead (-1 = never)
  mean_frac_pct   mean over leads of the area-weighted % of cells < 0
  max_frac_pct    worst lead's % of cells < 0
  min_value       most negative value seen (physical units)
The markdown table groups the 18 RELHUM levels (worst level shown) so it fits.
PASS token: NEGATIVITY_SUMMARY_OK n=<members>.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from climate_screen_summary import _channel_index, locate_year_file, valid_time  # noqa: E402
from sfno_ensemble.scores import equiangular_weights  # noqa: E402

FRAMES_PER_YEAR = 1460
COLUMNS = ("member", "channel", "steps", "truncated_at_step", "dry_air_fix", "ever_neg",
           "first_neg_lead", "mean_frac_pct", "max_frac_pct", "min_value",
           "truth_ever_neg", "truth_mean_frac_pct", "truth_min_value")
GROUPS = (("RELHUM (worst level)", r"^RELHUM_l\d\d$"), ("RHREFHT", r"^RHREFHT$"),
          ("PRECT", r"^PRECT$"), ("TMQ", r"^TMQ$"), ("SOILWATER_10CM", r"^SOILWATER_10CM$"))


class NegativityError(RuntimeError):
    pass


def build_truth(pack, start, n_leads, channels, stride=4, chunk=64) -> dict:
    """neg_frac and min of each channel in the pack at leads stride, 2*stride, …"""
    import h5py

    leads = np.arange(stride, n_leads + 1, stride)
    vt = np.array([valid_time(tuple(start), int(k)) for k in leads], dtype=np.int64)
    frac = np.full((leads.size, len(channels)), np.nan)
    mn = np.full((leads.size, len(channels)), np.nan)
    w = None
    for year in np.unique(vt[:, 0]):
        rows = np.flatnonzero(vt[:, 0] == year)
        frames = vt[rows, 1]
        with h5py.File(locate_year_file(Path(pack), int(year)), "r") as f:
            for c, name in enumerate(channels):
                ds, idx = _channel_index(f, name)
                d = f[ds]
                if w is None:
                    w = equiangular_weights(int(d.shape[2]))
                for a in range(0, frames.size, chunk):
                    fr = frames[a:a + chunk]
                    x = np.asarray(d[int(fr[0]):int(fr[-1]) + 1:stride, idx], dtype=np.float64)
                    if x.shape[0] != fr.size:
                        raise NegativityError(f"TRUTH_FRAMES: read {x.shape[0]} frames, want {fr.size}")
                    frac[rows[a:a + chunk], c] = (x < 0).mean(axis=-1) @ w
                    mn[rows[a:a + chunk], c] = x.min(axis=(-2, -1))
    return dict(channels=np.array(list(channels)), leads=leads, neg_frac=frac, min=mn,
                valid_year=vt[:, 0].astype(np.int32), valid_frame=vt[:, 1].astype(np.int32),
                start=np.array(start, dtype=np.int32), stride=np.int32(stride))


def _load(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["channels"] = [str(x) for x in d["channels"]]
    return d


def summarize_member(m: dict, truth: dict) -> list[dict]:
    n = int(m["neg_frac"].shape[0])
    if tuple(int(x) for x in m["start"]) != tuple(int(x) for x in truth["start"]):
        raise NegativityError(f"TRUTH_MISALIGNED: member start {tuple(m['start'])} "
                              f"!= truth {tuple(truth['start'])}")
    # truth rows covering the member's leads, and a valid-time check on them
    tl = truth["leads"]
    keep = tl <= n
    for j in np.flatnonzero(keep):
        k = int(tl[j]) - 1
        if (int(m["valid_year"][k]), int(m["valid_frame"][k])) != (
                int(truth["valid_year"][j]), int(truth["valid_frame"][j])):
            raise NegativityError(f"TRUTH_MISALIGNED at lead {k + 1}")
    tch = [str(x) for x in truth["channels"]]
    rows = []
    for c, name in enumerate(m["channels"]):
        fr = m["neg_frac"][:, c]
        neg = np.flatnonzero(fr > 0)
        row = dict(member=str(m["member_id"]), channel=name, steps=n,
                   truncated_at_step=int(m["truncated_at_step"]),
                   dry_air_fix=int(m.get("dry_air_fix", 0)),
                   ever_neg=int(neg.size > 0), first_neg_lead=int(neg[0] + 1) if neg.size else -1,
                   mean_frac_pct=float(100 * np.nanmean(fr)) if n else float("nan"),
                   max_frac_pct=float(100 * np.nanmax(fr)) if n else float("nan"),
                   min_value=float(np.nanmin(m["min"][:, c])) if n else float("nan"))
        if name in tch and keep.any():
            tf = truth["neg_frac"][keep, tch.index(name)]
            row.update(truth_ever_neg=int((tf > 0).any()),
                       truth_mean_frac_pct=float(100 * tf.mean()),
                       truth_min_value=float(truth["min"][keep, tch.index(name)].min()))
        else:
            row.update(truth_ever_neg=-1, truth_mean_frac_pct=float("nan"),
                       truth_min_value=float("nan"))
        rows.append(row)
    return rows


def group_table(rows: list[dict]) -> str:
    members = list(dict.fromkeys(r["member"] for r in rows))
    head = ["member", "steps", "trunc"] + [g for g, _ in GROUPS]
    out = ["Cell = mean % of area < 0 over leads / worst lead % / min value "
           "[truth: mean % / min]. RELHUM = the worst of its 18 levels by mean %.", "",
           "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for mem in members:
        mr = [r for r in rows if r["member"] == mem]
        cells = [mem, str(mr[0]["steps"]), str(mr[0]["truncated_at_step"])]
        for _, pat in GROUPS:
            g = [r for r in mr if re.match(pat, r["channel"])]
            if not g:
                cells.append("—")
                continue
            r = max(g, key=lambda x: (x["mean_frac_pct"] if np.isfinite(x["mean_frac_pct"]) else -1))
            lvl = f"{r['channel']}: " if len(g) > 1 else ""
            cells.append(f"{lvl}{r['mean_frac_pct']:.3g} / {r['max_frac_pct']:.3g} / "
                         f"{r['min_value']:.3g} [{r['truth_mean_frac_pct']:.3g} / "
                         f"{r['truth_min_value']:.3g}]")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("truth")
    t.add_argument("--pack", required=True, type=Path)
    t.add_argument("--start", required=True, type=int, nargs=2)
    t.add_argument("--n-leads", type=int, default=1460)
    t.add_argument("--stride", type=int, default=4)
    t.add_argument("--channels", nargs="+", required=True)
    t.add_argument("--out", required=True, type=Path)
    s = sub.add_parser("summarize")
    s.add_argument("--truth", required=True, type=Path)
    s.add_argument("--csv", required=True, type=Path)
    s.add_argument("--md", required=True, type=Path)
    s.add_argument("npz", nargs="+", type=Path)
    a = ap.parse_args(argv)

    if a.cmd == "truth":
        tr = build_truth(a.pack, tuple(a.start), a.n_leads, a.channels, a.stride)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.out, **tr)
        ever = [c for c, v in zip(tr["channels"], (tr["neg_frac"] > 0).any(axis=0)) if v]
        print(f"TRUTH negative somewhere: {ever or 'none'}")
        print(f"NEGATIVITY_TRUTH_OK leads={tr['leads'].size} out={a.out}")
        return 0

    truth = _load(a.truth)
    truth["channels"] = [str(x) for x in truth["channels"]]
    rows, errors = [], []
    for p in a.npz:
        try:
            rows += summarize_member(_load(p), truth)
        except NegativityError as exc:
            errors.append(f"{p}: {exc}")
    with open(a.csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(COLUMNS))
        wr.writeheader()
        wr.writerows(rows)
    md = group_table(rows)
    a.md.write_text(md)
    print(md)
    for e in errors:
        print(f"ERROR NEGATIVITY_MEMBER {e}")
    if errors:
        return 1
    print(f"NEGATIVITY_SUMMARY_OK n={len(set(r['member'] for r in rows))} csv={a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
