# HPO prune plan — distill-then-delete (own-track only)

**Date:** 2026-05-23
**Author:** Zhixing Liu (driven via Claude)
**Status:** DRAFT — awaiting user sign-off on §2 protect-list, §3 sweep-group assignments, and §4 dry-run output before any deletion.

---

## 0. Why

The two HPO trees have grown large: training checkpoints at `$SCRATCH/SFNO_Climate_Emulator/runs/` total
~235 GB, eval rollouts at `$WORK/SFNO_Climate_Emulator/results/sfno_eval/` total ~551 GB. Most runs are
HPO losers whose only remaining value is the *lesson* they taught (which knob broke what).
This plan distills the scientific record into compact CSV + Markdown, then hard-deletes
only the heavy artifacts (checkpoints, NetCDF rollouts), leaving every run's small-footprint
metadata (config, log, scorecard, report.md, figures) and the distilled summaries intact.

**Order is irreversible-by-design:** Phase 1 (distill) must finish and be reviewed *before*
Phase 2 (prune) runs with `--apply`. Phase 2 in dry-run mode is part of Phase 1 deliverable.

---

## 1. Resolved scope (from interview)

| Question | Answer |
|---|---|
| Trees in scope | **Both** training (`$SCRATCH/SFNO_Climate_Emulator/runs/`) and eval (`$WORK/SFNO_Climate_Emulator/results/sfno_eval/`) — own-track only. Skip `sfno_eval_5410/` and `sfno_eval_group/`. |
| Loser training-ckpt policy | **Delete `training_checkpoints/` entirely.** Loser becomes un-rerunnable from a ckpt; only `config.json`, `out.log`, `metadata.json`, `global_means/stds.npy`, the rendered YAML, and the distilled CSV/MD survive. |
| Sweep grouping | I propose explicit sweep groups using judgment + memory (§3 below). User signs off before any delete. |
| Loser eval-artifact policy | (Not asked — defaulting to:) delete `inference/` + `baselines/` subdirs (~16 GB/eval); keep `scores/` (5.8 MB), `figures/` (2.7 MB), `report.md` (8 KB), `provenance.txt`, `diagnostics/`. Per-eval reclaim is ~16 GB; scientific record is ~9 MB. |

**Open item for sign-off** (Q4 was skipped):

## 2. Protect-list (PROPOSED — needs user sign-off)

The tool will refuse to touch anything inside these paths, regardless of age or dominance.
I'm proposing the union of all four candidates I raised in the interview:

1. `$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_full/` — v10 own-track production EXP_DIR family; per memory `feedback_protect_prior_runs`.
2. `$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_full.pre-ema-20260504/` — deliberate pre-EMA snapshot.
3. All non-HPO legacy roots: `sfno_full/`, `sfno_short/`, `sfno_short_ddp/`, `sfno_short_ddp_sweep/`, `sfno_short_diagnostics/`, `sfno_smoke/`, `sfno_tiny/`, `sfno_zgplev_short/`, `sfno_zgplev_short_ddp/`, `sfno_zgplev_short_ddp_sweep/`, `sfno_zgplev_smoke_proto/`, `sfno_zgplev_tiny_proto/`, `sfno_zgplev_full_microbench/`, `sfno_zgplev_full_smoke_post_i3_20260508/`. Combined footprint is small (<1.5 GB) and these pre-date the HPO sweep — not what "obsolete HPO runs" means.
4. `sfno_group_sigma10_full/`, `sfno_group_sigma10_smoke/` — different scientific track (5410 / sigma10). Out of scope.
5. `$WORK/SFNO_Climate_Emulator/results/sfno_eval_5410/`, `$WORK/SFNO_Climate_Emulator/results/sfno_eval_group/` — sister-track eval roots.

**Tell me before sign-off if any of these should *not* be protected.** If the user disagrees on any, drop them from the protect-list and the tool will treat them like any other candidate (age + dominance test still applies).

---

## 3. Sweep groups + winner/loser calls (PROPOSED — needs user sign-off)

The cutoff date for the age criterion is **2026-05-16** (>7 days before today, 2026-05-23).
"Pruneable" = old (>=1 week) **OR** dominated within its sweep (union — the user's example
prunes `lr1p6e3` despite it being 6 days old).

Below, `KEEP` rows are the winners (or insufficiently-tested follow-ups); `PRUNE` rows are
the losers/old. Each PRUNE entry lists the reason and the disk reclaim (training ~6.8 GB +
eval ~16 GB if both exist).

### G1 — Legacy GB16/GB32 (pre-v11 partial-clone era)

Per memory `project_zgplev_gb_decision` (resolved 2026-05-09): **GB4 wins**; GB16/GB32 HPO paused.
All GB16/GB32 standalone training runs are dominated by the GB4 own-track production line, which lives
under the protected `sfno_zgplev_full/`.

| Run | mtime | Disk (train+eval) | Verdict |
|---|---|---|---|
| `sfno_zgplev_full_gb16_lr1e4_20260508` | 2026-05-08 | 6.8 G + 0 | PRUNE — old + dominated (GB4 won) |
| `sfno_zgplev_full_gb16_lr2e4_20260509` | 2026-05-09 | 6.8 G + 0 | PRUNE — old + dominated |
| `sfno_zgplev_full_gb16_lr2e4_20260509_retry1` | 2026-05-09 | 6.8 G + 0 | PRUNE — old + dominated |
| `sfno_zgplev_full_gb32_20260508` | 2026-05-08 | 6.8 G + 0 | PRUNE — old + dominated |
| `sfno_zgplev_gbhpo40_gb16_lr2_0e-4_20260511` | 2026-05-12 | 6.8 G + 16 G | PRUNE — old + dominated |
| `sfno_zgplev_gbhpo40_gb16_lr2_83e-4_20260511` | 2026-05-12 | 6.8 G + 16 G | PRUNE — old + dominated |

### G2 — Early group_clone exploration (pre-v11 / non-noise)

| Run | mtime | Disk | Verdict |
|---|---|---|---|
| `sfno_zgplev_group_clone` | 2026-05-09 | 6.8 G + 0 | PRUNE — old; superseded by v11 lineage |
| `sfno_zgplev_group_clone_smoke` | 2026-05-09 | (~3.6 G) | PRUNE — old smoke run, no scientific record needed (move to legacy-protect if user prefers) |
| `sfno_zgplev_group_clone_nonoise` | 2026-05-10 | 6.8 G + 0 | PRUNE — old; per memory `feedback_input_noise_is_load_bearing`, nonoise is a known loser |
| `sfno_zgplev_group_clone_gb32` | 2026-05-15 | 6.8 G + 16 G | PRUNE — dominated by GB4 (G1) and superseded by v11_gb32 (G4) |
| `sfno_zgplev_group_clone_v10_warmstart` | 2026-05-14 | 6.8 G + 16 G | **KEEP** — v10 warm-start is its own line of inquiry, not obsoleted by v11 sweeps |

### G3 — v11 clip A/B (concluded)

Per memory `project_v11_clip_experiment` (2026-05-12). v11_clip restored `optimizer_max_grad_norm=32`
against v11. Conclusion was reached and the line of work moved on to v11_gb32 LR HPO.

| Run | mtime | Disk | Verdict |
|---|---|---|---|
| `sfno_zgplev_group_clone_v11` | 2026-05-11 | 6.8 G + 0 | PRUNE — old; v11 baseline obsoleted by v11_clip + v11_gb32 lineage |
| `sfno_zgplev_group_clone_v11_clip` | 2026-05-12 | 6.8 G + 16 G | PRUNE — old; clip A/B concluded |
| `sfno_zgplev_group_clone_v11_clip_warmstart` | 2026-05-14 | 6.8 G + 16 G | PRUNE — old; warmstart variant of obsoleted v11_clip |

### G4 — v11_gb32 peak-LR sweep

| Run | mtime | Disk | Verdict |
|---|---|---|---|
| `sfno_zgplev_group_clone_v11_gb32` | 2026-05-15 | 6.8 G + 16 G | PRUNE — old; baseline (lr=2.83e-4, the lowest probe) dominated by lr8e4 |
| `sfno_zgplev_group_clone_v11_gb32_lr2p83e4` | 2026-05-15 | 6.8 G + 0 | PRUNE — old + dominated by lr8e4 |
| `sfno_zgplev_group_clone_v11_gb32_lr4e4` | 2026-05-16 | 6.8 G + 16 G | PRUNE — dominated by lr8e4 |
| `sfno_zgplev_group_clone_v11_gb32_lr5p66e4` | 2026-05-16 | 6.8 G + 16 G | PRUNE — dominated by lr8e4 |
| **`sfno_zgplev_group_clone_v11_gb32_lr8e4`** | 2026-05-16 | 6.8 G + 16 G | **KEEP** — sweep winner. *Verbatim user note:* "For gb32, sweeping peak learning rate found 8e-4 best so far; ~1e-3 degraded performance, and 1.6e-3 made the loss itself unstable." |
| `sfno_zgplev_group_clone_v11_gb32_lr1p13e3` | 2026-05-17 | 6.8 G + 16 G | PRUNE — dominated (the "~1e-3 degraded" probe) |
| `sfno_zgplev_group_clone_v11_gb32_lr1p13e3_noise0p035` | 2026-05-17 | 6.8 G + 0 | PRUNE — dominated (1.13e-3 LR is already a loser; noise=0.035 is also a loser per G6) |
| `sfno_zgplev_group_clone_v11_gb32_lr1p6e3` | 2026-05-17 | 6.8 G + 0 | PRUNE — dominated (the "1.6e-3 unstable" probe) |

### G5 — v11_gb32_lr8e4 min-LR sweep

| Run | mtime | Disk | Verdict |
|---|---|---|---|
| `sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e4` | 2026-05-19 | 6.8 G + 16 G | PRUNE — dominated by minlr1e5 |
| **`sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5`** | 2026-05-19 | 6.8 G + 16 G | **KEEP** — min-LR sweep winner; becomes the new operating point for downstream HPO |

### G6 — v11_gb32_lr8e4 noise sweep (first round)

Per memory `project_v11_noise_sweep_result` (2026-05-21): input_noise=0.05 is the operating point;
0.020 and 0.035 are dominated.

| Run | mtime | Disk | Verdict |
|---|---|---|---|
| `sfno_zgplev_group_clone_v11_gb32_lr8e4_noise0p020` | 2026-05-18 | 6.8 G + 16 G + 16 G | PRUNE — dominated by baseline noise=0.05 (both 2026-05-18 data-e3c934b and 2026-05-21 data-8b395eb evals) |
| `sfno_zgplev_group_clone_v11_gb32_lr8e4_noise0p035` | 2026-05-18 | 6.8 G + 16 G + 16 G | PRUNE — dominated by baseline noise=0.05 (both 2026-05-18 data-e3c934b and 2026-05-21 data-8b395eb evals) |

### G7 — v11_gb32_lr8e4_minlr1e5 β₁ sweep (null result)

Per memory `project_v11_beta1_sweep_null` (2026-05-21): β₁ ∈ {0.9, 0.95, 0.97} produced no meaningful
change; baseline 0.9 marginally best.

| Run | mtime | Disk | Verdict |
|---|---|---|---|
| `sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_beta1_0p95` | 2026-05-20 | 6.8 G + 16 G | PRUNE — dominated by β₁=0.9 baseline |
| `sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_beta1_0p97` | 2026-05-20 | 6.8 G + 16 G | PRUNE — dominated by β₁=0.9 baseline |

### G8 — v11_gb32_lr8e4_minlr1e5 noise sweep (second round)

Per memory `project_v11_noise_sweep_result`: noise=0.07 fails persistence gate.

| Run | mtime | Disk | Verdict |
|---|---|---|---|
| `sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_noise0p070` | 2026-05-21 | 6.8 G + 16 G | PRUNE — dominated (failed tas 6h persistence gate) |
| `sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_noise0p020_epochs75` | 2026-05-22 | 6.8 G + 16 G | **KEEP for now** — newest (1d old); insufficient evidence yet. Bundles noise=0.020 with epochs=75 — outcome unclear pending follow-up analysis. Mark for review at next prune pass. |

### G9 — v11_gb32_lr8e4_minlr1e5 epochs extension (in-flight)

| Run | mtime | Disk | Verdict |
|---|---|---|---|
| `sfno_zgplev_group_clone_v11_gb32_lr8e4_minlr1e5_epochs75` | 2026-05-21 | 6.8 G + 16 G | **KEEP** — 2 days old; current candidate operating point (epochs extension at the cumulative winner). Not yet superseded. |

### G-INVALID — User-deprecated eval dirs (`_INVALID_*` prefix)

Any eval dir whose name starts with `_INVALID_` is deprecated by user prefix-rename and is
**always PRUNE regardless of which family it tracks** — including the case where the family
itself is a §3 KEEP (e.g. `_INVALID_v10data_..._lr8e4_minlr1e5_...` whose training family is
the G5 KEEP winner). The prefix is the user's own deprecation marker; per the earlier
interview answer the prune verdict here overrides family-inherited verdicts.

Currently present:
- `_INVALID_v10data_20260520_eval-867fead_data-e3c934b_family-...lr8e4_minlr1e4_...` — PRUNE
- `_INVALID_v10data_20260520_eval-867fead_data-e3c934b_family-...lr8e4_minlr1e5_...` — PRUNE (family is G5 KEEP, but `_INVALID_` overrides)

### G0 — Familyless legacy eval dirs (pre-`_family-<X>_` naming)

Older eval dirs (pre-2026-05-15) lack the `_family-<X>_` tag added by `submit_eval_prelude.sh`,
so they can't be auto-linked to a §3 sweep group. Listing them explicitly here per Codex r1 P1:
each gets a verdict on the same distill-then-delete contract (scores/ + figures/ + report.md
archived to `docs/hpo_distill/runs/<name>/`; inference/ + baselines/ deleted).

| Eval dir | mtime | Verdict |
|---|---|---|
| `20260509_gb4_ema` | 2026-05-09 | **UNCLASSIFIED_PROTECT** (per Codex r2 P1) — its `scores/` dir is empty (no scorecard CSV, no bias-map .npys, no `report.md`). Nothing scalar to distill, so prune would lose the only record. Removed from `FAMILYLESS_EVAL_PRUNE_ALLOWLIST`; needs manual review if reclaim is desired. |
| `20260509_y11valid_gb4_k60` | 2026-05-09 | PRUNE — early v10 y11-valid k=60 re-eval; scores/ archived. |
| `20260510_eval-8b395eb_data-e3c934b` | 2026-05-10 | PRUNE — early v10 own-track re-eval; predates family-tag. |
| `20260511_eval-8b395eb_data-e3c934b` | 2026-05-11 | PRUNE — early v10 own-track re-eval. |
| `20260512_eval-8b395eb_data-e3c934b` | 2026-05-12 | PRUNE — early v10 own-track re-eval. |
| `20260512_eval-8b395eb_data-e3c934b_ckpt-best_ckpt_ema_mp0` | 2026-05-12 | PRUNE — early v10 own-track EMA re-eval. |
| `20260513_v11_clip_on_v11_testset_ema` | 2026-05-13 | PRUNE — v11_clip eval (G3 era); follows G3 PRUNE. |
| `20260514_v11_noclip_on_v11_testset_ema` | 2026-05-14 | PRUNE — v11 (no-clip) eval (G3 era); follows G3 PRUNE. |
| `tas_no_ice_20260514_1415_group_clone_v10_mp0` | 2026-05-14 | PRUNE — one-off tas_no_ice analysis; superseded by built-in `tas_no_ice` section in `report.md`. |
| `tas_no_ice_20260514_1415_v11_clip_ema` | 2026-05-14 | PRUNE — one-off tas_no_ice analysis; superseded. |

Future-proofing: the implementation defaults age-only PRUNE for familyless evals on the same
contract, but ANY new familyless eval that appears after this plan is signed must be added to
this table explicitly before being pruned — otherwise it falls through to UNCLASSIFIED_PROTECT.

### Totals (proposed)

- **Training-side PRUNE candidates:** **26 runs** × ~6.8 GB = **~177 GB reclaim**.
- **Eval-side PRUNE candidates:** ~32 evals × ~16 GB = **~470 GB reclaim** (includes the 10 G0 familyless dirs).
- **Combined reclaim:** **~640 GB** out of ~786 GB total (~81 %).
- **Scientific footprint preserved:** ~150 KB metadata/run × 26 + ~9 MB scores+figures+report.md/eval × 32 ≈ **~290 MB total**, plus the distilled CSV/MD.

*(Note: an interim 2026-05-23 override briefly kept `lr8e4_minlr1e4`, `lr8e4_noise0p020`, `lr8e4_noise0p035` — reverted after the user confirmed the distilled scorecard + report.md + figures is sufficient record for these dominated runs.)*

---

## 4. Phase 1 — distill (no deletion)

### 4.1 Inventory step

`scripts/hpo_prune.py inventory` walks both trees and emits:

- `docs/hpo_distill/inventory.csv` — one row per discovered training-run dir AND per eval-run dir, columns: `run_kind` (`train` or `eval`), `path` (absolute), `name`, `family` (for evals, the substring between `_family-` and `_ckpt-`; for training runs, equal to `name`), `sweep_group` (from §3, e.g. `G4`; blank for protected / non-HPO / familyless not in §3 G0), `verdict` (one of `KEEP`, `PRUNE`, `PROTECT`, `UNCLASSIFIED_PROTECT`), `reason`, `protected_by`, `mtime_iso` (newest mtime over all files in the dir), `size_bytes`, `heavy_bytes` (bytes that prune would reclaim — `training_checkpoints/` size for training, `inference/`+`baselines/` size for eval), and `ckpt_dir` / `inference_dir` / `baselines_dir` (exact delete targets).

### 4.2 Scalar-metric extraction

`scripts/hpo_prune.py distill` reads:

- **Training side** (`out.log`): parse the makani trainer log's multi-line `Epoch N summary:` blocks. Each block is followed by lines like `training loss: X`, `validation loss: Y`, `validation loss ema: Z`, `ema best loss: W`, `gradient norm: G`, `epoch time [s]: T`, `samples/sec: S`. The parser collects a ~40-line window after each `Epoch N summary:` header and pulls each metric via regex. Produces `docs/hpo_distill/train_scores.csv` with columns: `name, path, epoch, train_loss, val_loss, val_loss_ema, ema_best_loss, grad_norm, epoch_time_s, samples_per_sec` (no `lr` column — the trainer doesn't log LR per epoch). Also produces `docs/hpo_distill/train_summary.csv` (one row per run: `name, path, final_epoch, best_val_loss, best_val_epoch, best_val_loss_ema, best_val_loss_ema_epoch, total_wall_time_s`). Both `best_val_loss` (raw) and `best_val_loss_ema` (EMA-best, the canonical eval ckpt per `feedback_ema_is_canonical_ckpt`) are tracked separately.
- **Eval side**: primary parser reads `report.md` markdown tables. **Fallback parser** (per Codex r2 P2) reads `scores/nwp_scorecard_summary.csv` (columns: `model, channel, lead_hours, metric, mean, std, n_ics`) for legacy evals that predate `report.md` generation, e.g. the §3 G0 dirs. Both produce rows in `docs/hpo_distill/eval_scores.csv` with the same shape: `(eval_name, eval_path, section, channel, model, metric, lead_hours, mean, std, n)`. Modern repo scoring schema (per `scripts/score_nwp.py:53,158` and `_eval_utils.py:83`): **leads** = `6, 24, 72, 120, 240, 336` hours; **channels** = `tas, pr_6h, zg500` (or `zg5`), `ua5, ta5`, plus optional `tas_no_ice`; **metrics** = `rmse, acc` (no `mae` is emitted). **Note (Codex r3 P2):** the CSV-fallback parser may surface *additional* legacy leads (e.g. `360h`) and *additional* channels (e.g. `hus1` and other state variables) from older eval pipelines; those rows are preserved as-is in `eval_scores.csv` for completeness — downstream summarize/group MD only reads the modern lead/channel subset. **Per-row filter (Codex r3 P1):** both parsers drop `(model="5410 benchmark", channel="pr_6h")` rows because own-track `pr_6h` is m/s but 5410 is rate×6h, so the cell values are unit-invalid here (per `project_5410_eval_track` + `render_eval_report.py:39,253`). Also copies `report.md`, `provenance.txt`, `scores/`, `figures/`, and `diagnostics/` *verbatim* into `docs/hpo_distill/runs/<eval_dirname>/` so the per-eval record outlives the deletion.

### 4.3 Per-group Markdown notes

`scripts/hpo_prune.py summarize` emits:

- `docs/hpo_distill/INDEX.md` — top-level: per-group winner+losers table, total disk reclaimed, links to per-group notes, links to CSVs.
- `docs/hpo_distill/G1_legacy_gb16_gb32.md` … `docs/hpo_distill/G9_epochs_extension.md` — one per sweep group from §3. Each note has:
  - **Hypothesis** (what was being tested)
  - **Runs table** (mtime, hparams, key scores, verdict)
  - **Outcome** — explicit, with FAILURE modes called out
  - **Verbatim user quotes** preserved where I have them (G4 has the gb32 LR sentence)
  - **Memory cross-refs** — link to `[[project_v11_clip_experiment]]`, `[[project_v11_beta1_sweep_null]]`, etc.

### 4.4 Manifest

`scripts/hpo_prune.py manifest` produces `docs/hpo_distill/prune_manifest.csv`:

| column | meaning |
|---|---|
| `path` | absolute path to delete |
| `kind` | `train_ckpts` (deletes `training_checkpoints/` dir) or `eval_heavy` (deletes `inference/` + `baselines/` dirs) |
| `bytes` | bytes that will be reclaimed (computed from current `du -sb`) |
| `reason` | sweep-group + "old" / "dominated" / "old+dominated" |
| `mtime_iso` | for audit |
| `sweep_group` | from §3 |

This manifest IS the dry-run output. **Phase 2 will refuse to delete anything not in this manifest.**

---

## 5. Phase 2 — prune (`--apply` only after sign-off)

`scripts/hpo_prune.py prune [--apply] [--force-active]` (manifest path is fixed at `docs/hpo_distill/prune_manifest.csv`; per Codex r4 P2, the CLI does not accept a `--manifest` override):

**Two-phase execution** (per Codex r3 P1): Phase A validates every manifest row with **zero side effects**; if any row fails validation in `--apply` mode, the run aborts before any deletion. Phase B performs the actual deletes in row order (only reached when validation is clean, or always reached in dry-run mode for the WOULD-DELETE print).

1. **Pre-flight (only in `--apply` mode):**
   - **Queue must be empty** for `$USER` *and* the check must succeed: `_active_slurm_jobs()` returns `None` if `squeue` is missing/timed out/non-zero; `--apply` aborts in that case (override with `--force-active`, NOT recommended). A non-empty queue also aborts. Per Codex r2 P1, fail-closed: "queue empty" means **successfully checked and empty**, not "couldn't check". Eval jobs share generic names like `sfno_eval_inf` (per `scripts/submit_eval_inference.slurm:2`), so we can't reliably map JOBID → manifest path; queue-empty is the conservative guard.

2. **Phase A — validation (always runs, no side effects):** `_validate_manifest_row` returns one of `DELETE`, `ALREADY_GONE`, `PROTECTED`, `MTIME_MOVED`, `NO_RECORD` for each row:
   - `PROTECTED` if the path is under any protect-list root (runtime guard via `_path_protected_runtime`, not just at manifest-build time).
   - `MTIME_MOVED` if the path's mtime now is later than the recorded mtime by more than 5 minutes (the run was touched since distill, possibly a resume).
   - `NO_RECORD` if the distilled scientific record is missing: for `eval_heavy`, EITHER `docs/hpo_distill/runs/<name>/report.md` OR at least one archived score artifact matching `scores/nwp_scorecard*.csv`, `scores/*.csv`, `scores/*.json`, or `scores/*.npy` (per Codex r3 P1 — an empty `scores/` dir is **not** sufficient); for `train_ckpts`, the run must appear in `docs/hpo_distill/train_summary.csv`.
   - `ALREADY_GONE` if the path no longer exists (idempotent re-run).
   - `DELETE` otherwise (passes all checks).

3. **Without `--apply` (dry-run):** print `WOULD DELETE: <path> (<bytes>) [<kind>]` for each `DELETE` row + `REFUSE (<status>): <path>  [<reason>]` for each refusal. Exit 0. **No audit log is written in dry-run** (per Codex r3 P2 — dry-run must have zero side effects).

4. **With `--apply`:** if Phase A had any `PROTECTED`/`MTIME_MOVED`/`NO_RECORD` row, abort with exit 3 — no deletes happen. Otherwise Phase B walks rows in order:
   - `DELETE`: the manifest stores the exact path; `shutil.rmtree(Path(row["path"]))` is called directly (no further globbing — `ckpt_dir`/`inference_dir`/`baselines_dir` were resolved at inventory time).
   - `ALREADY_GONE`: skip; append a `skipped_already_gone` entry to the audit log.
   - Append one JSONL row per delete to `docs/hpo_distill/prune_audit.jsonl`: `{ts, path, kind, run_name, sweep_group, reason, bytes_freed, mtime_at_delete_iso, ckpt_sha256_pre_delete, action}` (sha256 over `metadata.json` + `config.json` for training runs, or over the eval dir's metadata if present — so the run identity is recoverable from the audit log alone).

5. **Protect-list as a hard guard, not just a filter:** even with a manifest hand-edited to include a protected path, the Phase A `_path_protected_runtime` check refuses.

---

## 6. Script layout (single module)

`scripts/hpo_prune.py` — pure pathlib + json + csv + argparse, no external deps. Subcommands:

```
hpo_prune.py inventory     # writes inventory.csv
hpo_prune.py distill       # writes train_scores.csv, train_summary.csv, eval_scores.csv,
                           #   docs/hpo_distill/runs/<name>/{report.md, provenance.txt, scores/}
hpo_prune.py summarize     # writes INDEX.md + G*.md (per §3 sweep-group assignments hardcoded
                           #   in a Python dict; the assignments live in the script, audited
                           #   by version control; edits go through this script, not by hand)
hpo_prune.py manifest      # writes prune_manifest.csv
hpo_prune.py prune [--apply] [--force-active]
                           # Phase A validate → Phase B delete.
                           # Without --apply: dry-run (validate + print, no side effects).
                           # With --apply: aborts (exit 3) before any delete if any row
                           # fails validation (PROTECTED/MTIME_MOVED/NO_RECORD).
                           # --force-active bypasses the squeue-empty fail-closed check
                           # (NOT recommended; use only when squeue is known unreachable).
hpo_prune.py all-dry       # convenience: inventory + distill + summarize + manifest + dry-run prune
```

The `--dry-run` requirement is satisfied by `all-dry`. Default `prune` invocation (without
`--apply`) is also dry.

Sweep-group dict lives at the top of the script and is the source of truth for §3. If the user
later wants to re-classify a run, edit the dict and re-run `summarize` + `manifest` — the
distilled CSVs don't change.

---

## 7. Safety summary (engineering bar)

- [x] No deletion without `--apply`.
- [x] Dry-run is **truly side-effect-free** — no audit-log writes (per Codex r3 P2).
- [x] Two-phase execution: validate ALL rows first; in `--apply`, abort before any delete if any row fails validation (per Codex r3 P1).
- [x] Protect-list enforced at *runtime* (not just at manifest-build time).
- [x] mtime-stability check between distill and delete (refuses on resume).
- [x] Active-SLURM check is fail-closed: queue must be **successfully verified empty** (per Codex r2 P1).
- [x] Audit log is JSONL, append-only, written only in `--apply`.
- [x] Idempotent re-run via existence check + audit log.
- [x] Sweep-group assignments live in code (version-controlled), not implicit in dirname parsing.
- [x] All scientific records (report.md, scores/, figures/, config.json, out.log) are copied or
      left in place — deletion is strictly limited to `training_checkpoints/`, `inference/`,
      and `baselines/`.
- [x] Plan lives in `docs/` for Codex review per `feedback_plan_to_docs`.

---

## 8. What needs user sign-off before I write any code

1. **§2 protect-list.** Confirm all five entries are protected, or tell me which to drop.
2. **§3 sweep-group assignments + verdicts.** Especially:
   - G2: should `sfno_zgplev_group_clone_smoke` be PRUNE'd outright, or moved to the protect-list?
   - G2: confirm `sfno_zgplev_group_clone_v10_warmstart` is KEEP (still a live line of inquiry, not dominated by v11).
   - G8: confirm `noise0p020_epochs75` should be KEEP-for-now (insufficient evidence).
   - G9: confirm `epochs75` is KEEP (current operating-point candidate).
3. **The unstated default in §1 row 4** (eval-side: delete `inference/` + `baselines/`, keep
   `scores/` + `figures/` + `report.md` + `provenance.txt` + `diagnostics/`). Sound right?

Once those three items are signed off, I'll implement `scripts/hpo_prune.py`, run
`all-dry`, and bring the per-group MD notes + manifest back to you for review *before*
running `prune --apply`.
