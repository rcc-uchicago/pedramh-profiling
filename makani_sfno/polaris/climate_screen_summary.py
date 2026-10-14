#!/usr/bin/env python3
"""Stage-0 stability screen: truth series, per-checkpoint metrics, pre-registered rank.

``polaris_makani_finetune_stability_handoff.md`` §2. One fixed rollout per
checkpoint (``polaris_climate_screen.pbs``) writes a climate-driver member
NetCDF. This module turns those into one CSV row per checkpoint and a markdown
table. It is torch-free (numpy, netCDF4, h5py), so it runs anywhere the venv
does -- but never on a login node (CLAUDE.md #3).

Two subcommands::

    # once per start date: truth global means for the screen's valid times
    python polaris/climate_screen_summary.py truth --pack $PACK \\
        --start 2044 1092 --n-leads 1460 --out $MEMBER_ROOT/runs/makani_eval/screen_truth_2044f1092.npz

    # after the rollouts
    python polaris/climate_screen_summary.py summarize --truth screen_truth_2044f1092.npz \\
        --csv screen.csv --md screen.md  OUT/screen_*.nc

Why a truth series: model-minus-lead-1 drift includes the seasonal cycle (B's
-1 to -3 K in ``T`` over Oct->Feb is mostly season). Drift here is **model minus
truth at the same valid time**, both area-weighted with
``sfno_ensemble.scores.equiangular_weights`` -- the weights the driver uses.

The ranking rule is pre-registered in
``docs/2026-09-24_climate_screen_prereg.md``; :func:`rank_key` is its only
implementation. Changing it after a screen has been read invalidates the screen.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sfno_ensemble.scores import equiangular_weights  # noqa: E402  (torch-free)

FRAMES_PER_YEAR = 1460                      # E3SM noleap, 6-hourly
PACK_SPLITS = ("train", "valid", "test")
TRUTH_CHANNELS = ("PS", "T_l17", "Z3_l10", "TREFHT", "TMQ")
GRAVITY = 9.80665                           # m s-2, as sfno_training.models.mass_fix
SIGMA_THRESHOLD = 3.0
# Physical-unit conversion per truth channel for the report (PS is Pa -> hPa).
UNIT = {"PS": ("hpa", 0.01), "T_l17": ("k", 1.0), "Z3_l10": ("m", 1.0), "TREFHT": ("k", 1.0),
        "TMQ": ("kgm2", 1.0)}
# (column, channel, lead); lead None = the member's last finite lead.
DRIFT_COLUMNS = (
    ("ps_drift_hpa@600", "PS", 600),
    ("ps_drift_hpa@1460", "PS", 1460),
    ("t17_drift_k@1460", "T_l17", 1460),
    ("z10_drift_m@1460", "Z3_l10", 1460),
    ("trefht_drift_k@1460", "TREFHT", 1460),
    ("ps_drift_hpa@last", "PS", None),
    ("tmq_drift_kgm2@1460", "TMQ", 1460),
    # dry-air surface pressure PS - g*TMQ: what the dry-air fix holds (diagnostic)
    ("dry_drift_hpa@600", "DRY", 600),
    ("dry_drift_hpa@1460", "DRY", 1460),
)
CSV_COLUMNS = (
    "rank", "label", "ckpt_epoch", "epoch_check", "survived", "truncated_at_step",
    "truncated_channel", "steps_run", "median_cross_3sigma", "n_past_3sigma",
    *[c for c, _, _ in DRIFT_COLUMNS], "first_past_3sigma", "run_dir", "ckpt", "nc",
)


class ScreenError(RuntimeError):
    """A precondition the screen refuses to report past."""


# ---------------------------------------------------------------------------
# calendar
# ---------------------------------------------------------------------------

def valid_time(start: tuple[int, int], lead: int) -> tuple[int, int]:
    """(year, frame) of lead ``lead`` from IC ``start`` (lead 0 = the IC)."""
    a = start[1] + lead
    return start[0] + a // FRAMES_PER_YEAR, a % FRAMES_PER_YEAR


# ---------------------------------------------------------------------------
# truth series
# ---------------------------------------------------------------------------

def locate_year_file(pack: Path, year: int) -> Path:
    hits = [Path(pack) / s / f"{year}.h5" for s in PACK_SPLITS
            if (Path(pack) / s / f"{year}.h5").is_file()]
    if len(hits) != 1:
        raise ScreenError(f"TRUTH_YEAR_FILE: {year}.h5 found {len(hits)} times under {pack}")
    return hits[0]


def _channel_index(f, name: str) -> tuple[str, int]:
    """(dataset, index) of channel ``name``: ``fields_state`` first, then diagnostic."""
    for ds, names in (("fields_state", "channel_state"),
                      ("fields_diagnostic", "channel_diagnostic")):
        if names in f:
            got = [x.decode() if isinstance(x, bytes) else str(x) for x in f[names][...]]
            if name in got:
                return ds, got.index(name)
    raise ScreenError(f"TRUTH_CHANNEL_MISSING: {name} not in {f.filename}")


def build_truth(pack: Path, start: tuple[int, int], n_leads: int,
                channels: Sequence[str] = TRUTH_CHANNELS, chunk: int = 146) -> dict:
    """Area-weighted global mean of ``channels`` at the valid time of leads 1..n_leads.

    Reads the pack's physical fields frame-chunk by frame-chunk (the files are
    chunked by frame), one channel at a time.
    """
    import h5py

    vt = np.array([valid_time(start, k) for k in range(1, n_leads + 1)], dtype=np.int64)
    gm = np.full((n_leads, len(channels)), np.nan, dtype=np.float64)
    nlat = None
    for year in np.unique(vt[:, 0]):
        leads = np.flatnonzero(vt[:, 0] == year)             # contiguous, frame-ordered
        f0, f1 = int(vt[leads[0], 1]), int(vt[leads[-1], 1]) + 1
        if f1 - f0 != leads.size:
            raise ScreenError(f"TRUTH_FRAMES: year {year} leads are not contiguous frames")
        with h5py.File(locate_year_file(pack, int(year)), "r") as f:
            for c, name in enumerate(channels):
                ds, idx = _channel_index(f, name)
                d = f[ds]
                if d.shape[0] != FRAMES_PER_YEAR:
                    raise ScreenError(f"TRUTH_FRAMES: {f.filename}:{ds} has {d.shape[0]} frames")
                if nlat is None:
                    nlat = int(d.shape[2])
                    w = equiangular_weights(nlat)
                for a in range(f0, f1, chunk):
                    b = min(a + chunk, f1)
                    x = np.asarray(d[a:b, idx], dtype=np.float64)     # (n, H, W)
                    gm[leads[a - f0]:leads[b - 1 - f0] + 1, c] = x.mean(axis=-1) @ w
    if not np.all(np.isfinite(gm)):
        raise ScreenError("TRUTH_NONFINITE: truth global mean has non-finite values")
    return dict(channels=np.array(list(channels)), global_mean=gm,
                valid_year=vt[:, 0].astype(np.int32), valid_frame=vt[:, 1].astype(np.int32),
                start=np.array(start, dtype=np.int32), nlat=np.int32(nlat), pack=str(pack))


def save_truth(truth: dict, out: Path) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **truth)


def load_truth(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        t = {k: z[k] for k in z.files}
    t["channels"] = [str(x) for x in t["channels"]]
    t["start"] = (int(t["start"][0]), int(t["start"][1]))
    return t


# ---------------------------------------------------------------------------
# one member
# ---------------------------------------------------------------------------

def load_member(path: Path) -> dict:
    import netCDF4

    with netCDF4.Dataset(path) as f:
        a = {k: f.getncattr(k) for k in f.ncattrs()}
        m = dict(
            names=[str(x) for x in f["channel"][:]],
            metrics=[str(x) for x in f["metric"][:]],
            thresholds=np.asarray(f["threshold"][:], dtype=np.float64),
            global_mean=np.asarray(f["global_mean"][:], dtype=np.float64),     # (T, C)
            first_bad=np.asarray(f["first_bad_step"][:]),                       # (M, Th, C)
            median_cross=np.asarray(f["median_cross_step"][:]),                 # (M, Th)
            valid_year=np.asarray(f["valid_year"][:], dtype=np.int64),
            valid_frame=np.asarray(f["valid_frame"][:], dtype=np.int64),
            nlat=int(f.dimensions["lat"].size),
        )
    m["attrs"] = a
    m["path"] = str(path)
    return m


def check_alignment(member: dict, truth: dict) -> None:
    """Refuse a member whose start or per-lead valid times differ from the truth's,
    or a truth file missing a channel a drift column needs (an older truth npz
    would otherwise turn those columns into NaN silently)."""
    missing = [c for c in TRUTH_CHANNELS if c not in truth["channels"]]
    if missing:
        raise ScreenError(f"TRUTH_MISSING_CHANNEL: truth file lacks {missing}; rebuild it "
                          "(new path -- do not overwrite a truth a committed CSV came from)")
    a = member["attrs"]
    mstart = (int(a["start_year"]), int(a["start_frame"]))
    if mstart != truth["start"]:
        raise ScreenError(f"TRUTH_MISALIGNED: member start {mstart} != truth start {truth['start']}")
    if member["nlat"] != int(truth["nlat"]):
        raise ScreenError(f"TRUTH_GRID: member nlat {member['nlat']} != truth nlat {truth['nlat']}")
    n = min(member["global_mean"].shape[0], truth["global_mean"].shape[0])
    bad = np.flatnonzero((member["valid_year"][:n] != truth["valid_year"][:n])
                         | (member["valid_frame"][:n] != truth["valid_frame"][:n]))
    if bad.size:
        k = int(bad[0])
        raise ScreenError(
            f"TRUTH_MISALIGNED: lead {k + 1} member valid "
            f"{(int(member['valid_year'][k]), int(member['valid_frame'][k]))} != truth "
            f"{(int(truth['valid_year'][k]), int(truth['valid_frame'][k]))}")


def expected_epoch(label: str) -> int | None:
    m = re.search(r"_e(\d+)$", label)
    return int(m.group(1)) if m else None


def summarize_member(member: dict, truth: dict, n_leads: int) -> dict:
    check_alignment(member, truth)
    a = member["attrs"]
    names = member["names"]
    gm = member["global_mean"]
    steps_run = gm.shape[0]
    trunc = int(a["truncated_at_step"])
    if truth["global_mean"].shape[0] < n_leads:
        raise ScreenError(f"TRUTH_SHORT: truth has {truth['global_mean'].shape[0]} leads < {n_leads}")
    if trunc == -1 and steps_run < n_leads:
        raise ScreenError(f"MEMBER_INCOMPLETE: {a.get('member_id')} ran {steps_run} of "
                          f"{n_leads} leads without a non-finite value")
    m = member["metrics"].index("anom_rms_sigma")
    t = int(np.flatnonzero(np.isclose(member["thresholds"], SIGMA_THRESHOLD))[0])
    fb = member["first_bad"][m, t]
    order = [c for c in np.argsort(np.where(fb > 0, fb, 10**9), kind="stable") if fb[c] > 0]

    label = str(a.get("member_id", Path(member["path"]).stem))
    ep = int(a.get("ckpt_epoch", -1))
    exp = expected_epoch(label)
    row = dict(
        label=label, ckpt_epoch=ep,
        epoch_check="n/a" if exp is None else ("ok" if exp == ep else f"MISMATCH(label {exp})"),
        survived=int(trunc == -1),
        truncated_at_step=trunc, truncated_channel=str(a.get("truncated_channel", "")),
        steps_run=steps_run,
        median_cross_3sigma=int(member["median_cross"][m, t]),
        n_past_3sigma=int((fb > 0).sum()),
        first_past_3sigma=" ".join(f"{names[c]}@{fb[c]}" for c in order[:3]),
        run_dir=str(a.get("run_dir", "")), ckpt=str(a.get("ckpt", "")), nc=member["path"],
    )
    tch = truth["channels"]
    def drift(ch, L):
        """(model - truth) global mean of ``ch`` at lead L, physical units; NaN if absent."""
        if ch == "DRY":
            d_ps, d_q = drift("PS", L), drift("TMQ", L)
            return d_ps - GRAVITY * d_q                               # Pa
        if ch not in names or ch not in tch or L > steps_run or L < 1:
            return float("nan")
        return float(gm[L - 1, names.index(ch)] - truth["global_mean"][L - 1, tch.index(ch)])

    for col, ch, lead in DRIFT_COLUMNS:
        L = steps_run if lead is None else lead
        row[col] = drift(ch, L) * (0.01 if ch == "DRY" else UNIT[ch][1])
    return row


# ---------------------------------------------------------------------------
# the pre-registered ranking rule (docs/2026-09-24_climate_screen_prereg.md §2)
# ---------------------------------------------------------------------------

def rank_key(row: dict) -> tuple:
    """Survivors first; then fewest channels past 3σ; then smallest |PS drift @1460|.

    Tie-breaks after the handoff's three (pre-registered too): later truncation
    first, then the label (deterministic). NaN |PS drift| (non-survivors) sorts last.
    """
    ps = abs(row["ps_drift_hpa@1460"])
    return (-row["survived"], row["n_past_3sigma"], ps if math.isfinite(ps) else math.inf,
            -(row["truncated_at_step"] if row["truncated_at_step"] > 0 else 10**9),
            row["label"])


def rank_rows(rows: list[dict]) -> list[dict]:
    out = sorted(rows, key=rank_key)
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if isinstance(v, float):
        return "nan" if not math.isfinite(v) else f"{v:+.3f}"
    return str(v)


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in CSV_COLUMNS})


MD_COLUMNS = ("rank", "label", "ckpt_epoch", "survived", "truncated_at_step",
              "median_cross_3sigma", "n_past_3sigma", "ps_drift_hpa@600", "ps_drift_hpa@1460",
              "t17_drift_k@1460", "z10_drift_m@1460", "trefht_drift_k@1460",
              "ps_drift_hpa@last", "tmq_drift_kgm2@1460", "dry_drift_hpa@1460",
              "first_past_3sigma")


def markdown(rows: list[dict], truth: dict) -> str:
    head = (f"Screen: start {truth['start'][0]} frame {truth['start'][1]}, "
            f"{truth['global_mean'].shape[0]} leads, n=1 per checkpoint. Drift = model − truth "
            "global mean (equiangular weights). -1 = never / not truncated.\n\n")
    lines = ["| " + " | ".join(MD_COLUMNS) + " |", "|" + "---|" * len(MD_COLUMNS)]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r[k]) for k in MD_COLUMNS) + " |")
    return head + "\n".join(lines) + "\n"


def summarize(ncs: Sequence[Path], truth: dict, n_leads: int) -> tuple[list[dict], list[str]]:
    rows, errors = [], []
    for p in ncs:
        try:
            rows.append(summarize_member(load_member(p), truth, n_leads))
        except ScreenError as exc:
            errors.append(f"{p}: {exc}")
    return rank_rows(rows), errors


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("truth")
    t.add_argument("--pack", required=True, type=Path)
    t.add_argument("--start", required=True, type=int, nargs=2, metavar=("YEAR", "FRAME"))
    t.add_argument("--n-leads", type=int, default=1460)
    t.add_argument("--out", required=True, type=Path)
    s = sub.add_parser("summarize")
    s.add_argument("--truth", required=True, type=Path)
    s.add_argument("--n-leads", type=int, default=1460)
    s.add_argument("--csv", required=True, type=Path)
    s.add_argument("--md", required=True, type=Path)
    s.add_argument("nc", nargs="+", type=Path)
    args = p.parse_args(argv)

    if args.cmd == "truth":
        tr = build_truth(args.pack, tuple(args.start), args.n_leads)
        save_truth(tr, args.out)
        gm = tr["global_mean"]
        print("TRUTH " + json.dumps({c: [round(float(gm[0, i]), 3), round(float(gm[-1, i]), 3)]
                                     for i, c in enumerate(tr["channels"])}) + " (lead 1, last)")
        print(f"CLIMATE_SCREEN_TRUTH_OK leads={gm.shape[0]} out={args.out}")
        return 0

    truth = load_truth(args.truth)
    rows, errors = summarize(args.nc, truth, args.n_leads)
    write_csv(rows, args.csv)
    md = markdown(rows, truth)
    args.md.write_text(md)
    print(md)
    for e in errors:
        print(f"ERROR CLIMATE_SCREEN_MEMBER {e}")
    if errors:
        return 1
    print(f"CLIMATE_SCREEN_SUMMARY_OK n={len(rows)} csv={args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
