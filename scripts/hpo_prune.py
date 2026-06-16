#!/usr/bin/env python3
"""hpo_prune.py — distill-then-delete HPO runs for the own-track SFNO emulator.

Design: docs/2026-05-23_hpo_prune_plan.md (user-approved 2026-05-23).

Subcommands:
  inventory   walk both trees, write inventory.csv
  distill     extract training loss curves + eval scorecard rows;
              archive report.md/provenance/scores/figures into docs/hpo_distill/runs/
  summarize   emit INDEX.md + per-group G*.md notes
  manifest    write prune_manifest.csv listing every path to delete
  prune       enforce manifest; default dry-run, --apply to delete
  all-dry     inventory + distill + summarize + manifest (no delete)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------

SCRATCH = Path(os.environ.get("SCRATCH", "/scratch/11114/zhixingliu"))
WORK = Path(os.environ.get("WORK", "/work2/11114/zhixingliu/stampede3"))
REPO_ROOT = Path(__file__).resolve().parent.parent

TRAIN_ROOT = SCRATCH / "SFNO_Climate_Emulator" / "runs"
EVAL_ROOT = WORK / "SFNO_Climate_Emulator" / "results" / "sfno_eval"
DISTILL_ROOT = REPO_ROOT / "docs" / "hpo_distill"

CUTOFF_DATE = dt.date(2026, 5, 16)  # >7 days before 2026-05-23
NOW_ISO = dt.datetime.now().isoformat(timespec="seconds")

# ---------------------------------------------------------------------------
# Protect-list (plan §2)
# ---------------------------------------------------------------------------

PROTECTED_TRAIN_DIRS = {
    "sfno_zgplev_full",
    "sfno_zgplev_full.pre-ema-20260504",
    # Non-HPO legacy
    "sfno_full",
    "sfno_short",
    "sfno_short_ddp",  # listed in plan §2; defensive even if absent today
    "sfno_short_ddp_sweep",  # listed in plan §2; defensive
    "sfno_short_diagnostics",
    "sfno_smoke",
    "sfno_tiny",
    "sfno_zgplev_short",
    "sfno_zgplev_short_ddp",
    "sfno_zgplev_short_ddp_sweep",
    "sfno_zgplev_smoke_proto",
    "sfno_zgplev_tiny_proto",
    "sfno_zgplev_full_microbench",
    "sfno_zgplev_full_smoke_post_i3_20260508",
    # Sister scientific tracks
    "sfno_group_sigma10_full",
    "sfno_group_sigma10_smoke",
}

# Eval-side protect: canonical reference symlinks + sister-track roots
PROTECTED_EVAL_NAMES = {
    "v10_zgplev_full_n96",  # canonical v10 reference per INDEX.md (dangling symlink today; keep entry)
}

# Familyless eval dirs (pre-`_family-<X>_` naming, before 2026-05-15) explicitly
# allowed for age-based PRUNE per plan §3 G0. NEW familyless evals not in this
# set default to UNCLASSIFIED_PROTECT — they must be added here intentionally.
FAMILYLESS_EVAL_PRUNE_ALLOWLIST = {
    # 20260509_gb4_ema EXCLUDED per Codex r2 P1: its scores/ is empty
    # (no report.md and no scorecard csv) → no scientific record to preserve.
    # Defaults to UNCLASSIFIED_PROTECT.
    "20260509_y11valid_gb4_k60",
    "20260510_eval-8b395eb_data-e3c934b",
    "20260511_eval-8b395eb_data-e3c934b",
    "20260512_eval-8b395eb_data-e3c934b",
    "20260512_eval-8b395eb_data-e3c934b_ckpt-best_ckpt_ema_mp0",
    "20260513_v11_clip_on_v11_testset_ema",
    "20260514_v11_noclip_on_v11_testset_ema",
    "tas_no_ice_20260514_1415_group_clone_v10_mp0",
    "tas_no_ice_20260514_1415_v11_clip_ema",
}

# Sister-track eval roots (not under EVAL_ROOT but documented for completeness)
PROTECTED_SISTER_EVAL_ROOTS = {
    WORK / "SFNO_Climate_Emulator" / "results" / "sfno_eval_5410",
    WORK / "SFNO_Climate_Emulator" / "results" / "sfno_eval_group",
}

# ---------------------------------------------------------------------------
# Sweep assignments (plan §3, post-2026-05-23-revert)
# Maps training-run basename → (group_id, verdict, reason)
# verdict ∈ {"KEEP", "PRUNE"}
# ---------------------------------------------------------------------------

SWEEP_ASSIGNMENTS: dict[str, tuple[str, str, str]] = {
    # G1 — Legacy GB16/GB32
    "sfno_zgplev_full_gb16_lr1e4_20260508": (
        "G1", "PRUNE", "old + dominated (GB4 won per project_zgplev_gb_decision)"),
    "sfno_zgplev_full_gb16_lr2e4_20260509": (
        "G1", "PRUNE", "old + dominated (GB4 won)"),
    "sfno_zgplev_full_gb16_lr2e4_20260509_retry1": (
        "G1", "PRUNE", "old + dominated (GB4 won)"),
    "sfno_zgplev_full_gb32_20260508": (
        "G1", "PRUNE", "old + dominated (GB4 won)"),
    "sfno_zgplev_gbhpo40_gb16_lr2_0e-4_20260511": (
        "G1", "PRUNE", "old + dominated (GB4 won)"),
    "sfno_zgplev_gbhpo40_gb16_lr2_83e-4_20260511": (
        "G1", "PRUNE", "old + dominated (GB4 won)"),

    # G2 — Early group_clone exploration
    "sfno_zgplev_group_clone": (
        "G2", "PRUNE", "old; superseded by v11 lineage"),
    "sfno_zgplev_group_clone_smoke": (
        "G2", "PRUNE", "old smoke run; no scientific record needed"),
    "sfno_zgplev_group_clone_nonoise": (
        "G2", "PRUNE", "old; nonoise is known loser per feedback_input_noise_is_load_bearing"),
    "sfno_zgplev_group_clone_gb32": (
        "G2", "PRUNE", "dominated by GB4 (G1) and superseded by v11_gb32 (G4)"),
    "sfno_zgplev_group_clone_v10_warmstart": (
        "G2", "KEEP", "live v10 warm-start line of inquiry; not obsoleted by v11"),

    # G3 — v11 clip A/B (concluded)
    "sfno_zgplev_group_clone_v11": (
        "G3", "PRUNE", "old; v11 baseline obsoleted by v11_clip + v11_gb32 lineage"),
    "sfno_zgplev_group_clone_v11_clip": (
        "G3", "PRUNE", "old; clip A/B concluded"),
    "sfno_zgplev_group_clone_v11_clip_warmstart": (
        "G3", "PRUNE", "old; warmstart variant of obsoleted v11_clip"),

    # G4 — v11_gb32 peak-LR sweep
    "sfno_zgplev_group_clone_v11_gb32": (
        "G4", "PRUNE", "old; baseline (lr=2.83e-4) dominated by lr8e4"),
    "sfno_zgplev_group_clone_v11_gb32_lr2p83e4": (
        "G4", "PRUNE", "old + dominated by lr8e4"),
    "sfno_zgplev_group_clone_v11_gb32_lr4e4": (
        "G4", "PRUNE", "dominated by lr8e4"),
    "sfno_zgplev_group_clone_v11_gb32_lr5p66e4": (
        "G4", "PRUNE", "dominated by lr8e4"),
    "sfno_zgplev_group_clone_v11_gb32_lr8e4": (
        "G4", "KEEP", 'sweep winner. User-verbatim: "For gb32, sweeping peak learning rate '
                       'found 8e-4 best so far; ~1e-3 degraded performance, and 1.6e-3 made the '
                       'loss itself unstable."'),
    "sfno_zgplev_group_clone_v11_gb32_lr1p13e3": (
        "G4", "PRUNE", 'dominated (the "~1e-3 degraded" probe)'),
    "sfno_zgplev_group_clone_v11_gb32_lr1p13e3_noise0p035": (
        "G4", "PRUNE", "dominated (1.13e-3 LR loser; noise=0.035 also a loser per G6)"),
    "sfno_zgplev_group_clone_v11_gb32_lr1p6e3": (
        "G4", "PRUNE", 'dominated (the "1.6e-3 unstable" probe)'),

    # G5 — v11_gb32_lr8e4 min-LR sweep
    "sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e4": (
        "G5", "PRUNE", "dominated by minlr1e5"),
    "sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5": (
        "G5", "KEEP", "min-LR sweep winner; new operating point for downstream HPO"),

    # G6 — v11_gb32_lr8e4 first-round noise sweep
    "sfno_zgplev_group_clone_v11_gb32_lr8e4_noise0p020": (
        "G6", "PRUNE", "dominated by baseline noise=0.05 (per project_v11_noise_sweep_result)"),
    "sfno_zgplev_group_clone_v11_gb32_lr8e4_noise0p035": (
        "G6", "PRUNE", "dominated by baseline noise=0.05"),

    # G7 — v11_gb32_lr8e4_minlr1e5 β₁ sweep (null result)
    "sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_beta1_0p95": (
        "G7", "PRUNE", "dominated by β₁=0.9 baseline (per project_v11_beta1_sweep_null)"),
    "sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_beta1_0p97": (
        "G7", "PRUNE", "dominated by β₁=0.9 baseline"),

    # G8 — v11_gb32_lr8e4_minlr1e5 second-round noise sweep
    "sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_noise0p070": (
        "G8", "PRUNE", "dominated (failed tas 6h persistence gate per project_v11_noise_sweep_result)"),
    "sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_noise0p020_epochs75": (
        "G8", "KEEP", "1d old, insufficient evidence; review at next prune pass"),

    # G9 — v11_gb32_lr8e4_minlr1e5 epochs extension
    "sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_epochs75": (
        "G9", "KEEP", "current candidate operating point; not yet superseded"),
}

GROUP_TITLES = {
    "G1": "Legacy GB16/GB32 (pre-v11 partial-clone era)",
    "G2": "Early group_clone exploration (pre-v11 / non-noise)",
    "G3": "v11 clip A/B (concluded)",
    "G4": "v11_gb32 peak-LR sweep",
    "G5": "v11_gb32_lr8e4 min-LR sweep",
    "G6": "v11_gb32_lr8e4 noise sweep (first round)",
    "G7": "v11_gb32_lr8e4_minlr1e5 β₁ sweep (null result)",
    "G8": "v11_gb32_lr8e4_minlr1e5 noise sweep (second round)",
    "G9": "v11_gb32_lr8e4_minlr1e5 epochs extension",
}

GROUP_HYPOTHESES = {
    "G1": "Whether GB16 or GB32 own-track training could beat GB4 (the eventual production baseline).",
    "G2": "Early baseline runs cloning the group-emulator config; explored input-noise on/off and GB.",
    "G3": "Restore gradient-norm clipping (max_grad_norm=32) on top of v11 to see if it stabilises long-lead loss.",
    "G4": "Peak learning-rate sweep on the v11_gb32 base.",
    "G5": "Min learning-rate (cosine floor) sweep at the v11_gb32_lr8e4 winner.",
    "G6": "Input-noise σ sweep at v11_gb32_lr8e4 (default minlr).",
    "G7": "Adam β₁ sweep at the v11_gb32_lr8e4_minlr1e5 winner.",
    "G8": "Second-round input-noise σ sweep at the v11_gb32_lr8e4_minlr1e5 winner, plus a longer (epochs=75) variant.",
    "G9": "Epochs-extension probe (50 → 75) at the cumulative winner.",
}

GROUP_OUTCOMES = {
    "G1": ("GB4 wins. GB16 and GB32 (standalone) both worse on the own-track scorecard. "
           "Result memorialised in `project_zgplev_gb_decision` (2026-05-09)."),
    "G2": ("Baseline group-clone superseded by v11 lineage. `nonoise` was a clear loser, "
           "confirming input-noise is load-bearing (`feedback_input_noise_is_load_bearing`). "
           "GB32 group-clone dominated by both the GB4 winner and the v11_gb32 LR-sweep winner."),
    "G3": ("v11_clip restored gradient-norm clipping; A/B concluded and the line moved on to "
           "the v11_gb32 LR sweep. Live record in `project_v11_clip_experiment` (2026-05-12)."),
    "G4": ('Sweeping peak LR found 8e-4 best so far; ~1e-3 (1.13e-3) degraded performance; '
           '1.6e-3 made the loss itself unstable. Verbatim user note preserved.'),
    "G5": "minlr=1e-5 (not 1e-4) was the better cosine-floor target at lr8e4.",
    "G6": ("Baseline σ=0.05 is the operating point; σ=0.020 and σ=0.035 are both worse on val. "
           "Confirms `project_v11_noise_sweep_result`."),
    "G7": ("β₁ ∈ {0.9, 0.95, 0.97} produced no meaningful change; baseline 0.9 marginally best. "
           "Null result; see `project_v11_beta1_sweep_null` (2026-05-21)."),
    "G8": ("σ=0.070 fails the tas-6h persistence gate (per `project_v11_noise_sweep_result`). "
           "σ=0.020-with-epochs75 is too fresh to call (kept-for-review)."),
    "G9": ("Epochs=75 extension at the cumulative winner is the current candidate operating point; "
           "not yet superseded."),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def newest_mtime(p: Path) -> dt.datetime | None:
    """Return the newest mtime over all regular files under p (recursively).
    None if p does not exist or has no files."""
    if not p.exists():
        return None
    newest = 0.0
    try:
        for sub in p.rglob("*"):
            try:
                if sub.is_file() and not sub.is_symlink():
                    m = sub.stat().st_mtime
                    if m > newest:
                        newest = m
            except (OSError, PermissionError):
                continue
    except (OSError, PermissionError):
        return None
    if newest == 0.0:
        return None
    return dt.datetime.fromtimestamp(newest)


def du_bytes(p: Path) -> int:
    """Total bytes used by p (du -sb semantics, follows no symlinks)."""
    if not p.exists():
        return 0
    total = 0
    try:
        for sub in p.rglob("*"):
            try:
                if sub.is_symlink():
                    continue
                if sub.is_file():
                    total += sub.stat().st_size
            except (OSError, PermissionError):
                continue
    except (OSError, PermissionError):
        return 0
    return total


def human_bytes(n: int) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}P"


def parse_eval_family(name: str) -> str | None:
    """Extract <family> from an eval dirname like
    '20260516_eval-<sha>_data-<sha>_family-<family>_ckpt-<tag>'.
    Returns None if no family= segment present."""
    m = re.search(r"_family-(?P<fam>.+?)(_ckpt-|$)", name)
    if not m:
        return None
    return m.group("fam")


def is_protected_train(run_name: str) -> str | None:
    if run_name in PROTECTED_TRAIN_DIRS:
        return "protected_train_dir"
    return None


def is_protected_eval(eval_name: str) -> str | None:
    if eval_name in PROTECTED_EVAL_NAMES:
        return "protected_eval_name"
    return None


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

def find_train_inner(run_root: Path) -> Path | None:
    """A training run dir has shape:
        <run_root>/<plasim_*>/0/
    Return that '0' subdir if found, else None.
    """
    for child in run_root.iterdir() if run_root.is_dir() else []:
        if child.is_dir() and child.name.startswith("plasim_"):
            zero = child / "0"
            if zero.is_dir():
                return zero
    return None


def inventory_train_runs() -> list[dict]:
    rows = []
    for run_dir in sorted(TRAIN_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        protected = is_protected_train(name)
        group_verdict = SWEEP_ASSIGNMENTS.get(name)
        if group_verdict is None:
            group, verdict, reason = ("", "", "")
            if protected is None:
                # Unclassified: not in sweep dict, not protected.
                # Default-PROTECT (refuse to touch); surface in INDEX.
                verdict = "UNCLASSIFIED_PROTECT"
                reason = "no sweep-group assignment; defaulting to protect"
        else:
            group, verdict, reason = group_verdict

        inner = find_train_inner(run_dir)
        ckpt_dir = inner / "training_checkpoints" if inner else None
        mtime = newest_mtime(run_dir)
        size = du_bytes(run_dir)
        ckpt_size = du_bytes(ckpt_dir) if ckpt_dir else 0

        rows.append({
            "run_kind": "train",
            "path": str(run_dir),
            "name": name,
            "family": name,
            "sweep_group": group,
            "verdict": verdict,
            "reason": reason,
            "protected_by": protected or "",
            "mtime_iso": mtime.isoformat(timespec="seconds") if mtime else "",
            "size_bytes": size,
            "heavy_bytes": ckpt_size,  # what prune would reclaim
            "ckpt_dir": str(ckpt_dir) if ckpt_dir else "",
        })
    return rows


def inventory_eval_runs() -> list[dict]:
    rows = []
    if not EVAL_ROOT.exists():
        return rows
    for eval_dir in sorted(EVAL_ROOT.iterdir()):
        # Skip top-level files (INDEX.md) and symlinks (canonical refs).
        if eval_dir.is_symlink():
            # symlinks like v10_zgplev_full_n96 — record protected entry, no size
            protected = is_protected_eval(eval_dir.name) or "symlink"
            rows.append({
                "run_kind": "eval",
                "path": str(eval_dir),
                "name": eval_dir.name,
                "family": "",
                "sweep_group": "",
                "verdict": "PROTECT",
                "reason": "symlink (canonical reference or alias)",
                "protected_by": protected,
                "mtime_iso": "",
                "size_bytes": 0,
                "heavy_bytes": 0,
                "inference_dir": "",
                "baselines_dir": "",
            })
            continue
        if not eval_dir.is_dir():
            continue
        name = eval_dir.name
        family = parse_eval_family(name) or ""
        protected = is_protected_eval(name)

        # Determine verdict
        verdict = ""
        reason = ""
        group = ""
        if protected:
            verdict = "PROTECT"
            reason = f"protected by {protected}"
        elif name.startswith("_INVALID_"):
            verdict = "PRUNE"
            reason = "user-deprecated by _INVALID_ name prefix"
        elif family and family in SWEEP_ASSIGNMENTS:
            group, train_verdict, train_reason = SWEEP_ASSIGNMENTS[family]
            verdict = train_verdict
            reason = f"follows training run {family}: {train_reason}"
        else:
            # Familyless: must be explicitly allow-listed in plan §3 G0 to be prune-eligible.
            # NEW familyless evals not in the allow-list default to UNCLASSIFIED_PROTECT
            # (per Codex r1 P1: don't sweep unenumerated familyless evals by age alone).
            mtime = newest_mtime(eval_dir)
            if mtime is None:
                verdict = "UNCLASSIFIED_PROTECT"
                reason = "no mtime available"
            elif name in FAMILYLESS_EVAL_PRUNE_ALLOWLIST and mtime.date() < CUTOFF_DATE:
                verdict = "PRUNE"
                reason = f"plan §3 G0 (familyless legacy eval, mtime {mtime.date().isoformat()})"
            else:
                verdict = "UNCLASSIFIED_PROTECT"
                reason = "familyless eval not in plan §3 G0 allow-list; protect by default"

        mtime = newest_mtime(eval_dir)
        size = du_bytes(eval_dir)
        inf_dir = eval_dir / "inference"
        base_dir = eval_dir / "baselines"
        heavy = du_bytes(inf_dir) + du_bytes(base_dir)

        rows.append({
            "run_kind": "eval",
            "path": str(eval_dir),
            "name": name,
            "family": family,
            "sweep_group": group,
            "verdict": verdict,
            "reason": reason,
            "protected_by": protected or "",
            "mtime_iso": mtime.isoformat(timespec="seconds") if mtime else "",
            "size_bytes": size,
            "heavy_bytes": heavy,
            "inference_dir": str(inf_dir),
            "baselines_dir": str(base_dir),
        })
    return rows


def cmd_inventory(args):
    DISTILL_ROOT.mkdir(parents=True, exist_ok=True)
    rows = inventory_train_runs() + inventory_eval_runs()
    out = DISTILL_ROOT / "inventory.csv"
    fieldnames = [
        "run_kind", "path", "name", "family", "sweep_group", "verdict",
        "reason", "protected_by", "mtime_iso", "size_bytes", "heavy_bytes",
        "ckpt_dir", "inference_dir", "baselines_dir",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"inventory: wrote {len(rows)} rows → {out}")
    return rows


# ---------------------------------------------------------------------------
# Distill — training log
# ---------------------------------------------------------------------------

EPOCH_HEAD_RE = re.compile(r"Epoch (?P<epoch>\d+) summary:")
METRIC_RES = {
    "train_loss": re.compile(r"training loss:\s+([0-9eE.+-]+)"),
    "val_loss": re.compile(r"validation loss:\s+([0-9eE.+-]+)"),
    "val_loss_ema": re.compile(r"validation loss ema:\s+([0-9eE.+-]+)"),
    "ema_best_loss": re.compile(r"ema best loss:\s+([0-9eE.+-]+)"),
    "grad_norm": re.compile(r"gradient norm:\s+([0-9eE.+-]+)"),
    "epoch_time_s": re.compile(r"epoch time \[s\]:\s+([0-9.]+)"),
    "samples_per_sec": re.compile(r"samples/sec:\s+([0-9.]+)"),
}
TOTAL_TIME_RE = re.compile(r"Total training time is ([0-9.]+) sec")


def parse_training_log(log_path: Path) -> tuple[list[dict], dict]:
    """Return (per-epoch rows, summary dict) parsed from out.log.
    The log uses multi-line 'Epoch N summary:' blocks; we collect the next ~30 lines
    of metrics after each header and pull values via regex."""
    epoch_rows: list[dict] = []
    summary = {"final_epoch": None, "best_val_loss": None, "best_val_epoch": None,
               "best_val_loss_ema": None, "best_val_loss_ema_epoch": None,
               "total_wall_time_s": None}
    if not log_path.exists():
        return epoch_rows, summary
    try:
        text = log_path.read_text(errors="replace")
    except (OSError, PermissionError):
        return epoch_rows, summary

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = EPOCH_HEAD_RE.search(lines[i])
        if not m:
            i += 1
            continue
        epoch = int(m.group("epoch"))
        # collect a window of up to 40 following lines for metrics
        window = "\n".join(lines[i + 1: i + 40])
        row = {"epoch": epoch}
        for k, rgx in METRIC_RES.items():
            mm = rgx.search(window)
            row[k] = float(mm.group(1)) if mm else None
        epoch_rows.append(row)
        i += 1

    # final/best
    if epoch_rows:
        summary["final_epoch"] = max(r["epoch"] for r in epoch_rows)
        vals = [(r["epoch"], r["val_loss"]) for r in epoch_rows if r.get("val_loss") is not None]
        if vals:
            best = min(vals, key=lambda x: x[1])
            summary["best_val_epoch"], summary["best_val_loss"] = best
        evals = [(r["epoch"], r["val_loss_ema"]) for r in epoch_rows if r.get("val_loss_ema") is not None]
        if evals:
            best_e = min(evals, key=lambda x: x[1])
            summary["best_val_loss_ema_epoch"], summary["best_val_loss_ema"] = best_e

    tm = TOTAL_TIME_RE.search(text)
    if tm:
        summary["total_wall_time_s"] = float(tm.group(1))

    return epoch_rows, summary


# ---------------------------------------------------------------------------
# Distill — eval report.md scorecard tables
# ---------------------------------------------------------------------------

CELL_RE = re.compile(
    r"^\s*(?P<mean>[-+]?[0-9eE.+-]+|[-+]?[0-9.]+[eE][-+]?[0-9]+)\s*"
    r"±\s*(?P<std>[-+]?[0-9eE.+-]+|[-+]?[0-9.]+[eE][-+]?[0-9]+)"
    r"(?:\s*\(n=(?P<n>\d+)\))?\s*$"
)
DASH_TOKENS = {"—", "-", "--", "NaN", "nan", ""}


def _parse_lead_header(cell: str) -> int | None:
    """Parse a header cell like '6h' or '120h' or '336h' into hours int."""
    cell = cell.strip()
    m = re.match(r"^(\d+)\s*h$", cell)
    if m:
        return int(m.group(1))
    return None


def _parse_cell(cell: str) -> tuple[float | None, float | None, int | None]:
    cell = cell.strip()
    if cell in DASH_TOKENS:
        return None, None, None
    m = CELL_RE.match(cell)
    if not m:
        return None, None, None
    try:
        mean = float(m.group("mean"))
    except ValueError:
        mean = None
    try:
        std = float(m.group("std"))
    except ValueError:
        std = None
    n = int(m.group("n")) if m.group("n") else None
    return mean, std, n


def parse_scorecard_csv(csv_path: Path) -> list[dict]:
    """Fallback parser for legacy evals lacking report.md.
    Reads scores/nwp_scorecard_summary.csv which has columns:
        model, channel, lead_hours, metric, mean, std, n_ics
    Returns rows in the same shape as parse_eval_report."""
    if not csv_path.exists():
        return []
    rows = []
    try:
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                model = r.get("model", "")
                channel = r.get("channel", "")
                # Same unit-mismatch filter as parse_eval_report (Codex r4 P2):
                # apply for symmetry / future-proofing even though current legacy
                # CSVs do not contain 5410 benchmark rows.
                if model == "5410 benchmark" and channel == "pr_6h":
                    continue
                try:
                    rows.append({
                        "section": "NWP Scorecard (legacy csv)",
                        "channel": channel,
                        "model": model,
                        "metric": r.get("metric", ""),
                        "lead_hours": int(r["lead_hours"]) if r.get("lead_hours") else None,
                        "mean": float(r["mean"]) if r.get("mean") else None,
                        "std": float(r["std"]) if r.get("std") else None,
                        "n": int(r["n_ics"]) if r.get("n_ics") else None,
                    })
                except (ValueError, KeyError):
                    continue
    except (OSError, PermissionError):
        return []
    return rows


def parse_eval_report(report_path: Path) -> list[dict]:
    """Walk report.md, find scorecard tables, and emit one dict per
    (section, channel, model, metric, lead_hours, mean, std, n)."""
    if not report_path.exists():
        return []
    try:
        text = report_path.read_text(errors="replace")
    except (OSError, PermissionError):
        return []

    rows = []
    lines = text.splitlines()
    section = None
    metric = None
    in_table = False
    leads: list[int] = []
    has_model_col = False  # whether the 2nd column is 'model'
    # If a section omits the model column (e.g. tas_no_ice), the model name
    # sits in column 0 and there is no channel — handle that.

    for line in lines:
        stripped = line.strip()
        # Section detection
        if stripped.startswith("## "):
            section = stripped.lstrip("# ").strip()
            metric = None
            in_table = False
            continue
        if stripped.startswith("### "):
            metric = stripped.lstrip("# ").strip().lower()
            in_table = False
            leads = []
            continue
        if not stripped.startswith("|"):
            in_table = False
            continue

        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not cells:
            continue

        # Header row?
        if cells[0].lower() in ("channel", "model"):
            has_model_col = (len(cells) >= 2 and cells[1].lower() == "model")
            # leads are the remaining cells (after channel + model, or after model)
            offset = 2 if has_model_col else 1
            leads = []
            for c in cells[offset:]:
                lh = _parse_lead_header(c)
                if lh is not None:
                    leads.append(lh)
            in_table = True
            continue

        # Separator row (|---|---|...) — ignore
        if all(set(c) <= set("-: ") for c in cells if c):
            continue

        if not in_table or not leads or metric is None:
            continue

        if has_model_col:
            if len(cells) < 2:
                continue
            channel = cells[0]
            model = cells[1]
            value_cells = cells[2:2 + len(leads)]
        else:
            channel = "tas_no_ice"  # by inspection; the only model-only table in report.md
            model = cells[0]
            value_cells = cells[1:1 + len(leads)]

        for lead, cell in zip(leads, value_cells):
            mean, std, n = _parse_cell(cell)
            if mean is None and std is None:
                continue
            # Per Codex r3 P1: suppress 5410-benchmark pr_6h rows in own-track
            # distill — own-track pr_6h is m/s but 5410 is rate×6h (per
            # project_5410_eval_track + render_eval_report.py:39,253), so the
            # numbers in those cells are unit-invalid here.
            if model == "5410 benchmark" and channel == "pr_6h":
                continue
            rows.append({
                "section": section or "",
                "channel": channel,
                "model": model,
                "metric": metric,
                "lead_hours": lead,
                "mean": mean,
                "std": std,
                "n": n,
            })

    return rows


# ---------------------------------------------------------------------------
# Distill — drive
# ---------------------------------------------------------------------------

def archive_eval_record(eval_dir: Path, dest_root: Path) -> None:
    """Copy the small-footprint scientific record into dest_root.
    Idempotent."""
    dest = dest_root / eval_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    for fname in ("report.md", "provenance.txt"):
        src = eval_dir / fname
        if src.exists() and src.is_file():
            shutil.copy2(src, dest / fname)
    for sub in ("scores", "figures", "diagnostics"):
        src = eval_dir / sub
        if src.exists() and src.is_dir():
            dst = dest / sub
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


def cmd_distill(args):
    DISTILL_ROOT.mkdir(parents=True, exist_ok=True)
    (DISTILL_ROOT / "runs").mkdir(exist_ok=True)

    train_scores_path = DISTILL_ROOT / "train_scores.csv"
    train_summary_path = DISTILL_ROOT / "train_summary.csv"
    eval_scores_path = DISTILL_ROOT / "eval_scores.csv"

    train_rows = []
    train_summaries = []
    eval_rows = []

    for run_dir in sorted(TRAIN_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        # We distill ALL runs (winners + losers + protected) so the record is uniform.
        inner = find_train_inner(run_dir)
        if inner is None:
            continue
        log_path = inner / "out.log"
        epochs, summary = parse_training_log(log_path)
        for r in epochs:
            r2 = dict(r)
            r2["name"] = run_dir.name
            r2["path"] = str(run_dir)
            train_rows.append(r2)
        train_summaries.append({
            "name": run_dir.name,
            "path": str(run_dir),
            **summary,
        })

    with train_scores_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["name", "path", "epoch",
                        "train_loss", "val_loss", "val_loss_ema",
                        "ema_best_loss", "grad_norm",
                        "epoch_time_s", "samples_per_sec"],
            extrasaction="ignore",
        )
        w.writeheader()
        for r in train_rows:
            w.writerow(r)

    with train_summary_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["name", "path", "final_epoch",
                        "best_val_loss", "best_val_epoch",
                        "best_val_loss_ema", "best_val_loss_ema_epoch",
                        "total_wall_time_s"],
            extrasaction="ignore",
        )
        w.writeheader()
        for s in train_summaries:
            w.writerow(s)

    # Eval side
    runs_archive = DISTILL_ROOT / "runs"
    if EVAL_ROOT.exists():
        for eval_dir in sorted(EVAL_ROOT.iterdir()):
            if not eval_dir.is_dir() or eval_dir.is_symlink():
                continue
            report = eval_dir / "report.md"
            rows = parse_eval_report(report)
            if not rows:
                # Fallback for pre-report.md legacy evals (Codex r2 P2)
                rows = parse_scorecard_csv(eval_dir / "scores" / "nwp_scorecard_summary.csv")
            for r in rows:
                r2 = dict(r)
                r2["eval_name"] = eval_dir.name
                r2["eval_path"] = str(eval_dir)
                eval_rows.append(r2)
            # archive the small record for every eval (winners + losers) so the
            # record outlives any future delete.
            try:
                archive_eval_record(eval_dir, runs_archive)
            except (OSError, PermissionError) as e:
                print(f"  warn: could not archive {eval_dir.name}: {e}", file=sys.stderr)

    with eval_scores_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["eval_name", "eval_path", "section",
                        "channel", "model", "metric",
                        "lead_hours", "mean", "std", "n"],
            extrasaction="ignore",
        )
        w.writeheader()
        for r in eval_rows:
            w.writerow(r)

    print(f"distill: training epochs   → {train_scores_path}  ({len(train_rows)} rows)")
    print(f"distill: training summary  → {train_summary_path} ({len(train_summaries)} rows)")
    print(f"distill: eval scorecards   → {eval_scores_path}   ({len(eval_rows)} rows)")
    print(f"distill: per-eval archives → {runs_archive}")


# ---------------------------------------------------------------------------
# Summarize — INDEX.md + per-group MD
# ---------------------------------------------------------------------------

def _load_train_summary() -> dict[str, dict]:
    p = DISTILL_ROOT / "train_summary.csv"
    if not p.exists():
        return {}
    out = {}
    with p.open() as f:
        for r in csv.DictReader(f):
            out[r["name"]] = r
    return out


def _load_eval_scores() -> dict[str, list[dict]]:
    p = DISTILL_ROOT / "eval_scores.csv"
    if not p.exists():
        return {}
    by_eval: dict[str, list[dict]] = defaultdict(list)
    with p.open() as f:
        for r in csv.DictReader(f):
            by_eval[r["eval_name"]].append(r)
    return by_eval


def _evals_for_family(family: str) -> list[str]:
    """List eval-dir names whose _family-<family>_ matches."""
    if not EVAL_ROOT.exists():
        return []
    out = []
    for eval_dir in sorted(EVAL_ROOT.iterdir()):
        if not eval_dir.is_dir() or eval_dir.is_symlink():
            continue
        if parse_eval_family(eval_dir.name) == family:
            out.append(eval_dir.name)
    return out


def cmd_summarize(args):
    DISTILL_ROOT.mkdir(parents=True, exist_ok=True)
    train_summary = _load_train_summary()
    eval_scores = _load_eval_scores()

    # Group by group_id
    by_group: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for run_name, (gid, verdict, reason) in SWEEP_ASSIGNMENTS.items():
        by_group[gid].append((run_name, verdict, reason))

    # ---- Per-group MD ----
    for gid in sorted(by_group):
        runs = sorted(by_group[gid], key=lambda x: x[0])
        title = GROUP_TITLES.get(gid, gid)
        path = DISTILL_ROOT / f"{gid}_{title.split('—')[0].strip().lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')[:60]}.md"

        lines = []
        lines.append(f"# {gid} — {title}")
        lines.append("")
        lines.append(f"**Hypothesis.** {GROUP_HYPOTHESES.get(gid, '(n/a)')}")
        lines.append("")
        lines.append(f"**Outcome.** {GROUP_OUTCOMES.get(gid, '(n/a)')}")
        lines.append("")
        lines.append("## Runs")
        lines.append("")
        lines.append("| run | verdict | best_val_loss_ema (epoch) | final_epoch | wall_time_h | reason |")
        lines.append("|---|---|---|---|---|---|")
        for run_name, verdict, reason in runs:
            s = train_summary.get(run_name, {})
            bvle = s.get("best_val_loss_ema") or ""
            bvle_ep = s.get("best_val_loss_ema_epoch") or ""
            fe = s.get("final_epoch") or ""
            wt = s.get("total_wall_time_s") or ""
            wt_h = f"{float(wt)/3600:.2f}" if wt else ""
            bvle_str = f"{float(bvle):.3e} (ep {bvle_ep})" if bvle else "—"
            lines.append(
                f"| `{run_name}` | **{verdict}** | {bvle_str} | {fe} | {wt_h} | {reason} |"
            )

        # Eval scorecard headline per run
        lines.append("")
        lines.append("## Eval headline (RMSE @ 24 h / 120 h / 336 h on emulator)")
        lines.append("")
        any_eval = False
        for run_name, _, _ in runs:
            evals = _evals_for_family(run_name)
            for ev in evals:
                rows = eval_scores.get(ev, [])
                # pick emulator-RMSE-NWP-scorecard rows
                lines.append(f"### `{ev}`")
                lines.append("")
                headline_channels = ("tas", "pr_6h", "zg500", "ua5", "ta5")
                headline_leads = (24, 120, 336)
                lines.append("| channel | " + " | ".join(f"{lh}h" for lh in headline_leads) + " |")
                lines.append("|---|" + "|".join(["---"] * len(headline_leads)) + "|")
                for ch in headline_channels:
                    vals = []
                    for lh in headline_leads:
                        match = [
                            r for r in rows
                            if r["section"].startswith("NWP Scorecard")
                            and r["channel"] == ch
                            and r["model"] == "emulator"
                            and r["metric"] == "rmse"
                            and int(r["lead_hours"]) == lh
                        ]
                        if match:
                            mean = float(match[0]["mean"])
                            std = float(match[0]["std"]) if match[0]["std"] else None
                            vals.append(f"{mean:.4g}" + (f" ± {std:.2g}" if std is not None else ""))
                        else:
                            vals.append("—")
                    lines.append(f"| `{ch}` | " + " | ".join(vals) + " |")
                lines.append("")
                any_eval = True
        if not any_eval:
            lines.append("*(no eval rollouts found for this group)*")
            lines.append("")

        path.write_text("\n".join(lines) + "\n")

    # ---- INDEX.md ----
    idx = DISTILL_ROOT / "INDEX.md"
    lines = []
    lines.append("# HPO distill index")
    lines.append("")
    lines.append(f"Generated: {NOW_ISO}")
    lines.append("")
    lines.append("Plan: [docs/2026-05-23_hpo_prune_plan.md](../2026-05-23_hpo_prune_plan.md)")
    lines.append("")
    lines.append("## Per-group notes")
    lines.append("")
    for gid in sorted(by_group):
        title = GROUP_TITLES.get(gid, gid)
        # Find the actual filename we just wrote (must match)
        slug = title.split("—")[0].strip().lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")[:60]
        fname = f"{gid}_{slug}.md"
        keep = sum(1 for (_n, v, _r) in by_group[gid] if v == "KEEP")
        prune = sum(1 for (_n, v, _r) in by_group[gid] if v == "PRUNE")
        lines.append(f"- **[{gid}]({fname})** — {title} ({keep} keep, {prune} prune)")
    lines.append("")
    lines.append("## Distilled tables")
    lines.append("")
    lines.append("- `inventory.csv` — every discovered training + eval dir with verdict and bytes")
    lines.append("- `train_scores.csv` — per-epoch (train_loss, val_loss, val_loss_ema, grad_norm, …)")
    lines.append("- `train_summary.csv` — per-run (best val loss, final epoch, wall time)")
    lines.append("- `eval_scores.csv` — per (eval, section, channel, model, metric, lead) scorecard rows")
    lines.append("- `prune_manifest.csv` — every path the prune subcommand will delete")
    lines.append("- `prune_audit.jsonl` — append-only log of actual deletions (written by `prune --apply`)")
    lines.append("")
    lines.append("## Archived per-eval records")
    lines.append("")
    lines.append("`runs/<eval_name>/` — verbatim copy of `report.md`, `provenance.txt`, `scores/`, `figures/`, `diagnostics/` for every eval (winner + loser). These survive deletion of the eval's `inference/` and `baselines/` NetCDFs.")
    lines.append("")
    idx.write_text("\n".join(lines) + "\n")

    print(f"summarize: wrote INDEX.md and {len(by_group)} per-group notes to {DISTILL_ROOT}")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def cmd_manifest(args):
    inv_path = DISTILL_ROOT / "inventory.csv"
    if not inv_path.exists():
        sys.exit(f"manifest: {inv_path} missing; run `inventory` first")

    manifest_rows = []
    with inv_path.open() as f:
        for row in csv.DictReader(f):
            verdict = row.get("verdict", "")
            if verdict != "PRUNE":
                continue
            if row["run_kind"] == "train":
                ckpt = row.get("ckpt_dir") or ""
                if not ckpt or not Path(ckpt).exists():
                    continue
                manifest_rows.append({
                    "path": ckpt,
                    "kind": "train_ckpts",
                    "bytes": row.get("heavy_bytes", "0"),
                    "run_name": row["name"],
                    "sweep_group": row.get("sweep_group", ""),
                    "reason": row.get("reason", ""),
                    "mtime_iso": row.get("mtime_iso", ""),
                })
            elif row["run_kind"] == "eval":
                inf = row.get("inference_dir") or ""
                base = row.get("baselines_dir") or ""
                for d in (inf, base):
                    if d and Path(d).exists():
                        manifest_rows.append({
                            "path": d,
                            "kind": "eval_heavy",
                            "bytes": str(du_bytes(Path(d))),
                            "run_name": row["name"],
                            "sweep_group": row.get("sweep_group", ""),
                            "reason": row.get("reason", ""),
                            "mtime_iso": row.get("mtime_iso", ""),
                        })

    out = DISTILL_ROOT / "prune_manifest.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["path", "kind", "bytes", "run_name",
                        "sweep_group", "reason", "mtime_iso"],
        )
        w.writeheader()
        for r in manifest_rows:
            w.writerow(r)

    total = sum(int(r["bytes"]) for r in manifest_rows)
    print(f"manifest: {len(manifest_rows)} paths totaling {human_bytes(total)} → {out}")


# ---------------------------------------------------------------------------
# Prune
# ---------------------------------------------------------------------------

def _path_protected_runtime(p: Path) -> str | None:
    """Hard runtime guard: refuse any path under protected training/eval roots."""
    p = p.resolve()
    for name in PROTECTED_TRAIN_DIRS:
        root = (TRAIN_ROOT / name).resolve()
        try:
            p.relative_to(root)
            return f"under protected train dir {name}"
        except ValueError:
            pass
    for name in PROTECTED_EVAL_NAMES:
        root = (EVAL_ROOT / name).resolve()
        try:
            p.relative_to(root)
            return f"under protected eval dir {name}"
        except ValueError:
            pass
    for sister in PROTECTED_SISTER_EVAL_ROOTS:
        try:
            p.relative_to(sister.resolve())
            return f"under sister-track eval root {sister}"
        except (ValueError, FileNotFoundError):
            pass
    return None


def _scientific_record_exists(run_name: str, kind: str) -> bool:
    if kind == "train_ckpts":
        # train_summary.csv must contain this run
        p = DISTILL_ROOT / "train_summary.csv"
        if not p.exists():
            return False
        with p.open() as f:
            for r in csv.DictReader(f):
                if r["name"] == run_name:
                    return True
        return False
    if kind == "eval_heavy":
        # Per Codex r2 P1: require an actual record, not just an empty scores/ dir.
        # Accept: report.md OR scores/nwp_scorecard*.csv OR scores/*.npy bias maps OR scores/*.json
        archive = DISTILL_ROOT / "runs" / run_name
        if (archive / "report.md").exists():
            return True
        scores = archive / "scores"
        if scores.is_dir():
            for pat in ("nwp_scorecard*.csv", "*.csv", "*.json", "*.npy"):
                if any(scores.glob(pat)):
                    return True
        return False
    return False


def _active_slurm_jobs() -> list[str] | None:
    """Return list of active jobs (empty list = verified empty).
    Return None if squeue couldn't be reached at all — caller must treat as
    UNVERIFIED and refuse --apply (fail-closed per Codex r2 P1)."""
    try:
        r = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", "zhixingliu"), "-h", "-o", "%i %j"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode != 0:
            return None  # squeue failed — unverified
        return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None  # missing/timeout — unverified


def _hash_metadata(run_dir: Path) -> str:
    """sha256 over metadata.json+config.json so run identity survives deletion."""
    h = hashlib.sha256()
    for name in ("metadata.json", "config.json"):
        # try a few likely locations
        candidates = [run_dir / name]
        for sub in run_dir.iterdir() if run_dir.is_dir() else []:
            if sub.is_dir():
                candidates.append(sub / "0" / name)
        for c in candidates:
            if c.exists() and c.is_file():
                try:
                    h.update(c.read_bytes())
                    break
                except (OSError, PermissionError):
                    continue
    return h.hexdigest()


def _validate_manifest_row(row: dict) -> tuple[str, str, int, dt.datetime | None]:
    """Return (status, reason, bytes_here, cur_mtime).
    status ∈ {"DELETE", "ALREADY_GONE", "PROTECTED", "MTIME_MOVED", "NO_RECORD"}.
    Pure: no side effects, no deletes."""
    path = Path(row["path"])
    kind = row["kind"]
    run_name = row["run_name"]

    if not path.exists():
        return "ALREADY_GONE", "path no longer exists", 0, None

    protected = _path_protected_runtime(path)
    if protected:
        return "PROTECTED", protected, 0, None

    cur_mtime = newest_mtime(path)
    rec_mtime = row.get("mtime_iso", "")
    if rec_mtime and cur_mtime:
        try:
            rec_dt = dt.datetime.fromisoformat(rec_mtime)
            if cur_mtime > rec_dt + dt.timedelta(minutes=5):
                return ("MTIME_MOVED",
                        f"recorded={rec_dt.isoformat(timespec='seconds')} "
                        f"now={cur_mtime.isoformat(timespec='seconds')}",
                        0, cur_mtime)
        except ValueError:
            pass

    if not _scientific_record_exists(run_name, kind):
        return "NO_RECORD", f"no archived record (run_name={run_name}, kind={kind})", 0, cur_mtime

    return "DELETE", "", du_bytes(path), cur_mtime


def cmd_prune(args):
    """Two-phase prune (per Codex r3 P1):
       Phase A — validate every manifest row WITHOUT any side effect.
       Phase B — if validation clean (or dry-run), perform deletions in row order.
    Audit JSONL only written in --apply mode (per Codex r3 P2)."""
    manifest_path = DISTILL_ROOT / "prune_manifest.csv"
    if not manifest_path.exists():
        sys.exit(f"prune: {manifest_path} missing; run `manifest` first")
    audit_path = DISTILL_ROOT / "prune_audit.jsonl"

    with manifest_path.open() as f:
        manifest_rows = list(csv.DictReader(f))

    apply_mode = bool(args.apply)
    print(f"prune: mode = {'APPLY' if apply_mode else 'DRY-RUN'}")
    print(f"prune: {len(manifest_rows)} manifest rows")

    # Pre-flight: active SLURM jobs (fail-closed per Codex r2 P1)
    if apply_mode:
        active = _active_slurm_jobs()
        if active is None:
            print("prune: REFUSING — could not verify SLURM queue state "
                  "(squeue missing, timed out, or returned non-zero).")
            print("prune: pass --force-active only if you are certain no jobs touch the manifest paths.")
            if not args.force_active:
                sys.exit(2)
        elif active:
            print("prune: ACTIVE SLURM JOBS DETECTED:")
            for j in active:
                print(f"  {j}")
            print("prune: refusing to delete with active jobs in the queue. "
                  "Re-run when queue is empty or pass --force-active to override.")
            if not args.force_active:
                sys.exit(2)

    # Phase A: validation pass over ALL rows (no side effects)
    validated: list[tuple[dict, str, str, int, dt.datetime | None]] = []
    counts = defaultdict(int)
    bytes_planned = 0
    for row in manifest_rows:
        status, reason, bytes_here, cur_mtime = _validate_manifest_row(row)
        validated.append((row, status, reason, bytes_here, cur_mtime))
        counts[status] += 1
        if status == "DELETE":
            bytes_planned += bytes_here

    hard_refusals = counts["PROTECTED"] + counts["MTIME_MOVED"] + counts["NO_RECORD"]
    print(f"prune: validation: DELETE={counts['DELETE']}, "
          f"ALREADY_GONE={counts['ALREADY_GONE']}, "
          f"PROTECTED={counts['PROTECTED']}, "
          f"MTIME_MOVED={counts['MTIME_MOVED']}, "
          f"NO_RECORD={counts['NO_RECORD']}")
    print(f"prune: planned {counts['DELETE']} deletes / {human_bytes(bytes_planned)}")

    # Surface refusals
    for row, status, reason, _bh, _ct in validated:
        if status in ("PROTECTED", "MTIME_MOVED", "NO_RECORD"):
            print(f"  REFUSE ({status}): {row['path']}  [{reason}]")
        elif not apply_mode and status == "DELETE":
            print(f"  WOULD DELETE: {row['path']}  ({human_bytes(_bh)})  [{row['kind']}]")

    if apply_mode and hard_refusals > 0:
        print(f"prune: REFUSING --apply — {hard_refusals} row(s) failed validation. "
              "Fix the underlying issue(s) and re-run.")
        sys.exit(3)

    if not apply_mode:
        # Dry-run ends here. No audit write.
        return

    # Phase B: actually delete (validation clean above)
    bytes_freed = 0
    audit_lines = []
    for row, status, _reason, bytes_here, cur_mtime in validated:
        if status == "ALREADY_GONE":
            audit_lines.append(json.dumps({
                "ts": NOW_ISO, "path": row["path"], "kind": row["kind"],
                "run_name": row["run_name"], "action": "skipped_already_gone",
            }))
            continue
        if status != "DELETE":
            continue  # validation already refused these in --apply path; we shouldn't reach here

        path = Path(row["path"])
        kind = row["kind"]
        run_name = row["run_name"]

        ckpt_sha = ""
        try:
            if kind == "train_ckpts":
                rr = path.parents[2]  # runs/<run>
                ckpt_sha = _hash_metadata(rr)
            else:
                rr = path.parent  # eval dir
                ckpt_sha = _hash_metadata(rr)
        except Exception:
            pass

        try:
            shutil.rmtree(path)
            bytes_freed += bytes_here
            print(f"  DELETED: {path}  ({human_bytes(bytes_here)})")
            audit_lines.append(json.dumps({
                "ts": NOW_ISO,
                "path": str(path),
                "kind": kind,
                "run_name": run_name,
                "sweep_group": row.get("sweep_group", ""),
                "reason": row.get("reason", ""),
                "bytes_freed": bytes_here,
                "mtime_at_delete_iso": cur_mtime.isoformat(timespec="seconds") if cur_mtime else "",
                "ckpt_sha256_pre_delete": ckpt_sha,
                "action": "deleted",
            }))
        except (OSError, PermissionError) as e:
            print(f"  ERROR: {path}: {e}")
            audit_lines.append(json.dumps({
                "ts": NOW_ISO, "path": str(path), "kind": kind, "run_name": run_name,
                "action": "delete_failed", "error": str(e),
            }))

    # Only write audit in --apply mode (Codex r3 P2.7)
    if audit_lines:
        with audit_path.open("a") as f:
            for ln in audit_lines:
                f.write(ln + "\n")

    print()
    print(f"prune: applied {human_bytes(bytes_freed)} freed across {counts['DELETE']} paths")


# ---------------------------------------------------------------------------
# all-dry
# ---------------------------------------------------------------------------

def cmd_all_dry(args):
    cmd_inventory(args)
    cmd_distill(args)
    cmd_summarize(args)
    cmd_manifest(args)
    # Final dry-run prune print
    args.apply = False
    args.force_active = False
    cmd_prune(args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("inventory").set_defaults(func=cmd_inventory)
    sub.add_parser("distill").set_defaults(func=cmd_distill)
    sub.add_parser("summarize").set_defaults(func=cmd_summarize)
    sub.add_parser("manifest").set_defaults(func=cmd_manifest)
    pp = sub.add_parser("prune")
    pp.add_argument("--apply", action="store_true", help="actually delete (default: dry-run)")
    pp.add_argument("--force-active", action="store_true",
                    help="ignore active SLURM jobs (NOT recommended)")
    pp.set_defaults(func=cmd_prune)
    ap = sub.add_parser("all-dry")
    ap.set_defaults(func=cmd_all_dry, apply=False, force_active=False)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
