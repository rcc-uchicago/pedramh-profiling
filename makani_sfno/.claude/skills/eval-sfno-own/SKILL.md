---
name: eval-sfno-own
description: Evaluate the user's own trained SFNO emulator (v10 zgplev) end-to-end on Stampede3. Orchestrates the 4-job inference → score → report → figures chain via scripts/submit_eval.sh. Supports MODE=nwp (12 ICs/yr × 8 yr × K=56) and MODE=climate (8 ICs × 1 yr). Auto-builds climatology and test_holdout when missing. Resolves default RUN_DIR within the production run family `$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_full/plasim_sim52_zgplev_full/[0-9]+`. Use when the user asks to evaluate the trained emulator/checkpoint, run an NWP scorecard, run climate stats, regenerate `report.md`, render eval figures, or compare own-track trained runs. For the group SFNO-5410 track, defer to the sibling skill `eval-sfno-5410`.
---

# Evaluate the own-track SFNO emulator (v10 zgplev)

Orchestrate the own-track SFNO emulator evaluation pipeline (`docs/sfno_eval_plan.md` v2.8). End-to-end: resolve checkpoint → preflight → submit 4-job chained SLURMs (inference → score → report → figures) → return job IDs → exit.

**You SUBMIT and RETURN.** The chain is multi-hour; do not block, poll, or sleep on it. After printing the four JOB IDs, `report.md` path, and `figures/` path, your job is done. The user re-invokes you to check status. Figure rendering is the 4th job in the chain — it is **not** a manual post-step.

## When to use this skill

Trigger when the user asks any of:

- "evaluate the trained emulator" / "evaluate the latest checkpoint" / "evaluate run /3"
- "run the NWP scorecard" / "run the eval" / "rerun the evaluation"
- "run climate stats" / "run climate-mode rollout"
- "regenerate `report.md`" / "render the eval figures" / "redo the bias maps"
- "evaluate this checkpoint: <path>"

Do NOT trigger for: training jobs, packager runs, dataset migrations, eval-plan editing, or anything mentioning "5410" / "group emulator" — those belong to `eval-sfno-5410`.

## Track scope

Own-track v10 zgplev emulator only. Driver: `scripts/submit_eval.sh` → 4-job chain. Output tree: `$WORK2/SFNO_Climate_Emulator/results/sfno_eval/<run_tag>/`. For 5410, defer to `eval-sfno-5410`.

## Canonical checkpoint: `best_ckpt_ema_mp0` when EMA is enabled

When the training config has `ema.enabled: true` (all current production own-track configs), canonical eval uses **`best_ckpt_ema_mp0.tar`**. The raw `best_ckpt_mp0.tar` is a diagnostic control, not the headline.

**Resolution rule:**
1. EMA-enabled run → default `CKPT=$RUN_DIR/training_checkpoints/best_ckpt_ema_mp0.tar`. `submit_eval.sh` auto-picks it.
2. "Compare raw vs EMA" / "use the raw checkpoint" → explicitly pass `CKPT=.../best_ckpt_mp0.tar`; it lands in a separate `_ckpt-best_ckpt_mp0`-tagged OUT_ROOT.
3. EMA-disabled run → only `best_ckpt_mp0.tar` exists; fall back.
4. Head-to-head: don't mix flavors across runs.

Reference: `docs/2026-05-02_ema_implementation_plan.md` §4.3.

## Dataset-family auto-resolution (v10 vs v11 guard)

**Never pass `TEST_HOLDOUT` / `TRAIN_DIR` / `PACKAGER_TEST_SRC` manually for an own-track eval.** The prelude auto-resolves all three from `RUN_DIR/config.json:train_data_path`:

```
dataset_root      = dirname(config.json:train_data_path)         # .../sim52_zgplev_full[_v11]
dataset_family    = basename(dataset_root)                       # sim52_zgplev_full | sim52_zgplev_full_v11
suffix            = dataset_family - "sim52_zgplev_full"         # "" | "_v11"
TEST_HOLDOUT      = $dataset_root/test_holdout
TRAIN_DIR         = $dataset_root/train
PACKAGER_TEST_SRC = $(dirname $dataset_root)/sim52_astro_64x128_zgplev${suffix}/test
```

If the caller passes a value that *disagrees* with the trained family, `submit_eval_prelude.sh` aborts with `FATAL: <var>=... does not match the dataset family the model was trained on (...)`. This is the v10/v11 confound — a v11-trained model evaluated on v10 data (or vice-versa) produces meaningless scores (the `pl` convention differs; see [v11 partial-clone units](project_v11_partial_clone_units.md)).

The override is reserved for the unusual case of intentionally scoring against a different family (e.g. a sanity comparison) — you must pass **all three** consistent paths together for that to succeed, but in practice no one should do this for own-track eval. Always trust the auto-resolution.

## Inputs (env-var contract from `submit_eval.sh`)

All optional. Anything you don't set falls back per §default-resolution.

| Var | Default | Notes |
|---|---|---|
| `RUN_DIR` | (resolved per §default-resolution) | Training run dir containing `config.json`, `global_means.npy`, `training_checkpoints/`. |
| `CKPT` | `$RUN_DIR/training_checkpoints/best_ckpt_ema_mp0.tar` when present, else `best_ckpt_mp0.tar` | EMA-best is canonical for EMA-enabled runs. Override to compare against the raw best as a diagnostic control. |
| `MODE` | `nwp` | `nwp` = 12 ICs/yr × 8 yr × K=56 step rollouts. `climate` = 8 ICs × 1-yr rollouts (~1454 steps each). |
| `TEST_HOLDOUT` | **auto-resolved** from `RUN_DIR/config.json:train_data_path` — see §Dataset-family auto-resolution. Don't pass this manually unless you know exactly what you're doing. |
| `TRAIN_DIR` | **auto-resolved** (same as TEST_HOLDOUT). | Training pool — climatology source. |
| `PACKAGER_TEST_SRC` | **auto-resolved** (same family as TEST_HOLDOUT, `sim52_astro_64x128_zgplev[_vN]/test`). | Source for `build_test_split.py` and `DATA_SHA7` lookup. |
| `BLOCKER_JOB_ID` | unset | If set, inference job runs `--dependency=afterok:$BLOCKER_JOB_ID`. Use to chain after a still-running training job. |
| `FULL_RUN_TAG` | `0` | `1` forces the legacy 4-SHA + ckpt run-tag template. Provenance always recorded in `provenance.txt` either way. |
| `BENCHMARK_5410_OUT_ROOT` | pinned valid 5410 run (`…/sfno_eval_5410/20260509_blocking_96ic_h100_packed_derecho_env_valid`) | Group SFNO-5410 result to overlay in the scorecard table and figures. Shown next to the own-track result for all channels EXCEPT `pr_6h` — that row is suppressed by default in own-track reports per `PR6H_UNIT_ALIGN=suppress` (see next row). Set to empty string to disable; if the path's `scores/nwp_scorecard_summary.csv` is missing or empty, report.md prints a loud `⚠️` banner and figures fall back to own-only. |
| `PR6H_UNIT_ALIGN` | `suppress` | Controls cross-track `pr_6h` row when `BENCHMARK_5410_OUT_ROOT` is set and `TRACK=own`. Default `suppress` drops the 5410-benchmark `pr_6h` row from the RMSE and ACC scorecard tables and emits a banner citing the upstream forward-z-score anomaly (`infer_sfno5410_blocking_h100_packed.py:348-349`, also `infer_sfno5410_byo_ic.py:425-432`) plus the truth-side unit-convention mismatch. `none` restores the row (but the `pr_6h` numbers will still NOT be directly comparable — the underlying unit gap persists; see `docs/2026-05-23_pr6h_unit_alignment_plan.md`). Has no effect under `TRACK=5410` (rows are already in matching group-native units). |

## Default RUN_DIR resolution

When neither `RUN_DIR` nor `CKPT` is provided, resolve in this order. **Do not** scan globally by mtime across all `sfno_*` runs — that picks up smoke tests / microbench / incomplete runs.

1. **Explicit override.** If the user said `RUN_DIR=...` or `CKPT=...`, use it.
2. **Scan production family only:** `$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_full/plasim_sim52_zgplev_full/[0-9]+`.
3. **Sort numerically by run number, descending** (`/3` before `/2` before `/1` before `/0`).
4. **Pick the first run that is BOTH valid AND complete:**
   - valid: either `$cand/training_checkpoints/best_ckpt_ema_mp0.tar` or `$cand/training_checkpoints/best_ckpt_mp0.tar` exists with size > 0. EMA-best is preferred when present.
   - complete: training reached `max_epochs` — grep `out.log` for `"Training complete"`, `"max_epochs reached"`, or final `"epoch X/X"` matching `max_epochs` from `config.json`. If marker absent, treat as **incomplete** and try the next-lower run number.
5. **Refuse if nothing qualifies.** Print:
   > "No completed run found under `$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_full/plasim_sim52_zgplev_full/[0-9]+`. Pass an explicit `RUN_DIR=...` or `CKPT=...` to override."

   Do NOT fall back to mtime, do NOT pick an incomplete run — confirm with the user instead.

### Reference resolver snippet

```bash
production_glob="$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_full/plasim_sim52_zgplev_full"

if [[ -n "${RUN_DIR:-}" ]]; then
    resolved="$RUN_DIR"; source="explicit RUN_DIR"
elif [[ -n "${CKPT:-}" ]]; then
    resolved="$(dirname "$(dirname "$CKPT")")"; source="explicit CKPT"
else
    resolved=""; source="default production-family scan"
    for d in $(ls -1d "$production_glob"/[0-9]* 2>/dev/null \
               | awk -F/ '{print $NF, $0}' | sort -k1,1 -n -r | awk '{print $2}'); do
        # Prefer EMA-best; fall back to raw-best for EMA-disabled legacy runs.
        ckpt_ema="$d/training_checkpoints/best_ckpt_ema_mp0.tar"
        ckpt_raw="$d/training_checkpoints/best_ckpt_mp0.tar"
        if [[ -s "$ckpt_ema" ]]; then
            ckpt_default="$ckpt_ema"
        elif [[ -s "$ckpt_raw" ]]; then
            ckpt_default="$ckpt_raw"
        else
            continue
        fi
        log="$d/out.log"
        cfg="$d/config.json"
        if [[ -f "$log" && -f "$cfg" ]]; then
            max_ep="$(python -c "import json,sys; print(json.load(open('$cfg')).get('max_epochs',''))" 2>/dev/null)"
            if [[ -n "$max_ep" ]] && grep -qE "(Training complete|max_epochs reached|epoch ${max_ep}/${max_ep})" "$log"; then
                resolved="$d"
                ckpt_resolved_default="$ckpt_default"
                break
            fi
        fi
    done
    [[ -z "$resolved" ]] && { echo "ERROR: no completed run found under $production_glob/[0-9]+" >&2; exit 2; }
fi

# If RUN_DIR was explicit and we haven't already chosen, pick the EMA file
# when present, else the raw file. The user's explicit CKPT= always wins.
if [[ -z "${ckpt_resolved_default:-}" ]]; then
    ema_cand="$resolved/training_checkpoints/best_ckpt_ema_mp0.tar"
    raw_cand="$resolved/training_checkpoints/best_ckpt_mp0.tar"
    if [[ -s "$ema_cand" ]]; then
        ckpt_resolved_default="$ema_cand"
    else
        ckpt_resolved_default="$raw_cand"
    fi
fi
ckpt_resolved="${CKPT:-$ckpt_resolved_default}"
run_num="$(basename "$resolved")"
ckpt_basename="$(basename "$ckpt_resolved" .tar)"
ckpt_mtime="$(stat -c '%y' "$ckpt_resolved" 2>/dev/null || echo unknown)"

cat <<EOF
[eval-sfno-own] resolved checkpoint
  source        : $source
  run number    : $run_num
  RUN_DIR       : $resolved
  CKPT          : $ckpt_resolved
  ckpt basename : $ckpt_basename   (ema = canonical when EMA enabled; raw = diagnostic control)
  ckpt mtime    : $ckpt_mtime
EOF

export RUN_DIR="$resolved"
export CKPT="$ckpt_resolved"
```

**Always print all six lines (source, run number, RUN_DIR, CKPT, ckpt basename, mtime) to the user before submitting** so they can stop you if it picked the wrong run or the wrong raw/EMA flavor.

## Preflight (before submit)

1. **Working tree.** `cd $HOME/projects/SFNO_Climate_Emulator`. The submit script reads `git rev-parse --short=7 HEAD` for `EVAL_SHA7` — uncommitted changes still get evaluated under that SHA, which mis-attributes results. If `git status --porcelain` is non-empty, warn the user and ask whether to proceed.
2. **Venv.** `.venv/bin/python` must exist (the SLURMs source it).
3. **`RUN_DIR` sanity.** `$RUN_DIR/config.json` exists; `global_means.npy`/`global_stds.npy` exist (53 floats each); `$CKPT` exists size > 0. Expected sizes: raw ~1.7 GB (has optimizer state), EMA ~0.4 GB (model state only) — both valid. Print which flavor was selected.
4. **`PACKAGER_TEST_SRC`.** At minimum `MOST.0121.h5` must be present (used for `DATA_SHA7`). Inference SLURM auto-builds `TEST_HOLDOUT` from this if empty.
5. **`TRAIN_DIR` non-empty.** Score SLURM auto-builds climatology from this on first eval; thereafter `$OUT_ROOT/baselines/climatology_proleptic.nc` is reused. First build: ~30–60 min wallclock.
6. **MODE.** Must be `nwp` or `climate`.

## Submit & return

After preflight, submit the chain. Do NOT inline-edit `submit_eval.sh` — pass overrides via env. Show the resolved provenance the script prints, then exit.

```bash
cd $HOME/projects/SFNO_Climate_Emulator
RUN_DIR="$RUN_DIR" CKPT="$CKPT" MODE="$MODE" \
TEST_HOLDOUT="${TEST_HOLDOUT:-}" TRAIN_DIR="${TRAIN_DIR:-}" \
PACKAGER_TEST_SRC="${PACKAGER_TEST_SRC:-}" \
BLOCKER_JOB_ID="${BLOCKER_JOB_ID:-}" \
scripts/submit_eval.sh
```

The script:
1. Composes `RUN_TAG` and `OUT_ROOT=$WORK2/SFNO_Climate_Emulator/results/sfno_eval/$RUN_TAG`.
2. Writes `$OUT_ROOT/provenance.txt` (full SHAs + paths).
3. Submits **4** SLURMs with `afterok` dependencies:
   - `submit_eval_inference.slurm` — h100, 1 GPU, ~30–90 min depending on MODE/limits.
   - `submit_eval_score.slurm` — h100, climatology (if missing) + `score_nwp.py`.
   - `submit_eval_report.slurm` — h100, `render_eval_report.py` → `report.md`.
   - `submit_eval_figures.slurm` — **skx-dev**, CPU-only, `render_eval_figures.py` → `figures/*.png` (~1 min).
4. Prints the four `[submit_eval] ... job: <id>` lines and the final artifact paths.

**Surface to the user (verbatim):**
- `RUN_TAG`, `OUT_ROOT`
- All four job IDs (`JOB_INF`, `JOB_SCO`, `JOB_REP`, `JOB_FIG`)
- Final `report.md` path and `figures/` path
- Reminder: "Re-invoke me with `squeue -u $USER` or 'is the eval done' to check status."

Do NOT poll. Do NOT sleep. Do NOT use `Monitor` to wait for the chain.

## Output tree

After all 4 SLURMs complete:

```
$WORK2/SFNO_Climate_Emulator/results/sfno_eval/<RUN_TAG>/
├── provenance.txt                      # full SHAs + resolved paths
├── inference/                          # NetCDF rollouts (per IC)
│   └── <year>/<ic_idx>/predictions.nc
├── baselines/
│   └── climatology_proleptic.nc        # cached after first eval against this TRAIN_DIR
├── diagnostics/
│   ├── calendar_trace.csv
│   └── climatology_source_files.json
├── scores/
│   ├── nwp_scorecard.csv               # per-(channel, lead) RMSE/ACC + persistence
│   ├── nwp_scorecard_summary.csv       # report-channel × report-lead grid
│   └── bias_maps_<channel>_lead<h>.npy
├── figures/                            # produced by JOB_FIG (4th SLURM, skx-dev)
│   ├── rmse_vs_lead.png
│   ├── acc_vs_lead.png
│   └── bias_<channel>.png              # tas, pr_6h, zg500, ua5, ta5
└── report.md                           # produced by JOB_REP
```

`report.md` is the single artifact to surface to the user.

**Group SFNO-5410 benchmark overlay.** The report scorecard table and the figures (`rmse_vs_lead.png`, `acc_vs_lead.png`, `bias_<channel>.png`) include the group SFNO-5410 result side-by-side with the own-track result for all state channels (`tas`, `zg500`, `ua5`, `ta5`), sourced from `BENCHMARK_5410_OUT_ROOT` (default: pinned valid 96-IC H100+packed-env run). **`pr_6h` is the exception in the own-track scorecard table:** the 5410-benchmark `pr_6h` row is suppressed by default (per `PR6H_UNIT_ALIGN=suppress`) because (a) the upstream 5410 inference scripts apply a *forward* z-score transform to the diagnostic channel at `infer_sfno5410_blocking_h100_packed.py:348-349` (also `infer_sfno5410_byo_ic.py:425-432`), so the on-disk 5410 `pr_6h` prediction is in a transformed space, and (b) the truth-side unit convention also differs (own m/s vs 5410 "6-hour precip proxy", with an unaudited ~5× gap on top of the 21,600 s/6h nominal factor). Pass `PR6H_UNIT_ALIGN=none` to restore the row (but numbers will still not be directly comparable). Bias maps render as a 2-row layout (own on top, 5410 on bottom) with **separate colorbars per row** so unit differences are honest — own-track `pr_6h` is scaled m/s → mm/day for display, while 5410 `pr_6h` stays in native `kg m^-2` per 6h per the group convention. The bias-map overlay sits in the same mixed-units regime as the scorecard `pr_6h` row but harmonizing the figures-side display is a deferred follow-up (see `docs/2026-05-23_pr6h_unit_alignment_plan.md` §10). If the benchmark scorecard is missing/empty, the chain still completes and renders own-only with a loud warning.

## Regenerate figures only (no rerun of inference/scoring)

When the figure-rendering script changes (style tweaks, new overlays) and you want to back-fill the new look onto an already-completed `OUT_ROOT`, just rerun the 4th-job script directly. It reads `scores/nwp_scorecard_summary.csv` + `scores/bias_maps_*.npy` and overwrites `figures/*.png` in place — no SLURM, no GPU.

```bash
OUT=/path/to/results/sfno_eval/<RUN_TAG>
BENCH=/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/results/sfno_eval_5410/20260509_blocking_96ic_h100_packed_derecho_env_valid

source $HOME/projects/SFNO_Climate_Emulator/.venv/bin/activate
cd $HOME/projects/SFNO_Climate_Emulator
python scripts/render_eval_figures.py \
    --out-root "$OUT" \
    --benchmark-5410-out-root "$BENCH"   # omit to render own-only
```

Runs in ~10–30 s on a login node. Default `--track own` is correct for own-track runs; the sibling 5410 skill documents the `--track 5410` form for group-track runs.

## RUN_TAG collision guard

`submit_eval.sh` auto-includes `_family-<train_family>` (the parent-of-parent of `RUN_DIR`) in RUN_TAG, and aborts (exit 3) when `$OUT_ROOT/provenance.txt` already records a different `CKPT=`. Recovery: pass `RUN_TAG=<unique-name>` or move the existing dir. Background: `docs/run_log/2026-05-12_v11_gbhpo40_run_tag_collision.md`.

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ERROR: no completed run found under ...` | All `/N` runs are incomplete or no best ckpt of either flavor. | Pass `RUN_DIR=` explicitly. Don't auto-pick an incomplete run. |
| `[submit_eval] FATAL: ... provenance.txt already records a different CKPT` | Two eval chains submitted against the same RUN_TAG (see §RUN_TAG collision guard above). | Pass `RUN_TAG=<unique-name>` or move the existing `$OUT_ROOT`. |
| `DATA_SHA7=unknown` warning | `PACKAGER_TEST_SRC/MOST.0121.h5` missing. | Set `PACKAGER_TEST_SRC` to the right packager output dir. |
| `JOB_SCO` failed early | Climatology build OOM or wrong `TRAIN_DIR`. | Check `logs/sfno_eval_score_<id>.err`; verify `TRAIN_DIR` matches the dataset version of `RUN_DIR`. |
| `report.md` missing scorecard | Score job didn't write `nwp_scorecard.csv`. | Check `logs/sfno_eval_score_<id>.err` — usually a climatology channel-mismatch. |

## Status check (when user asks "is the eval done?")

```bash
squeue -u $USER -o "%.10i %.20j %.8T %.10M %.20S"  # running/pending
ls -lt $WORK2/SFNO_Climate_Emulator/results/sfno_eval/<RUN_TAG>/  # what's been written
```

- If `JOB_FIG` shows `CD` (complete) and `figures/` is non-empty → done; surface `report.md` and figure paths.
- If `JOB_REP` is `CD` but `JOB_FIG` is still queued/running → say "report ready, figures ETA ~1 min on skx-dev".
- If any job shows `F` / `CA` / `TO` → look in `logs/sfno_eval_<inference|score|report|figures>_<id>.err`.
- If anything earlier than `JOB_FIG` is still running → say "still running, ETA ~X min based on its `t` budget" and exit.

## Bundled training+eval flow (added 2026-05-20)

In addition to the 4-job standalone chain described above, production h100
training submits can run the full eval pipeline **inside the training
allocation** to avoid paying the h100 queue wait twice. See
`docs/2026-05-20_bundled_training_eval_plan.md`.

How to identify a bundled-eval run:

- `$OUT_ROOT/bundled_eval_status.txt` exists with per-stage `*_rc=N` lines.
- `logs/bundled_eval_status_<SLURM_JOB_ID>.txt` exists in the AI-RES repo
  (always present for bundled runs, including SKIP/FAIL paths where
  `$OUT_ROOT` was never created).
- Training's status mail body contains a `TRAIN=… EVAL=…` line.

Per-submit defaults (`BUNDLED_EVAL` env var):

| Submit | Default | Notes |
|---|---|---|
| submit_zgplev_full.slurm | `1` | v10 production |
| submit_zgplev_group_clone.slurm | `1` | v10 production |
| submit_zgplev_group_clone_v11.slurm | `1` | v11 production |
| submit_zgplev_group_clone_v11_clip.slurm | `1` | v11_clip production |
| submit_zgplev_group_clone_v10_warmstart.slurm | `0` | warm-start chunk — set `BUNDLED_EVAL=1` on the **final** chunk only |
| submit_zgplev_group_clone_v11_clip_warmstart.slurm | `0` | same |
| submit_zgplev_baseline.slurm | (untouched) | deferred (gh / GH200) |

Skip + failure cases (read from `$OUT_ROOT/bundled_eval_status.txt` if it
exists, else `logs/bundled_eval_status_<jobid>.txt`):

- `SKIP_DISABLED` — `BUNDLED_EVAL=0`. Standard for warm-start non-final chunks.
- `SKIP_NO_RUN_DIR` — submit script forgot to export `RUN_DIR`. Bug.
- `SKIP_NO_NEW_CKPT` — EMA ckpt mtime not refreshed this run (training
  produced no improvement). Expected for non-improving chunks.
- `FAIL_PRELUDE_3` — collision guard tripped; `$OUT_ROOT` already exists.
  Resolve with `ALLOW_RERUN=1 RUN_TAG=<unique-name>` or by moving the
  prior dir.
- `FAIL_INFERENCE | FAIL_SCORE | FAIL_REPORT | FAIL_FIGURES` — that
  stage's stderr is in `logs/sfno_<jobname>_<jobid>.err`. Resume the
  failed stage standalone with
  `RUN_TAG=<existing> ALLOW_RERUN=1 sbatch scripts/submit_eval_<stage>.slurm`
  (and any downstream stages).
- `OK` — all 4 stages completed inside the training allocation.

**Important behaviour change (2026-05-20):** standalone `submit_eval.sh`
now requires `ALLOW_RERUN=1` to reuse ANY existing `$OUT_ROOT`, not just
when the recorded CKPT path differs. Add `ALLOW_RERUN=1` to any workflow
that legitimately re-evaluates into an existing dir.

## Cross-references

- Plan: `docs/sfno_eval_plan.md` v2.8 (locked, implementation-ready)
- Bundled training+eval plan: `docs/2026-05-20_bundled_training_eval_plan.md` (v3.1, approved)
- Sibling skill (5410 track): `.claude/skills/eval-sfno-5410/SKILL.md`
- Per-run postmortems / surprise write-ups: `docs/run_log/` (one file per investigated run; see `docs/run_log/README.md` for the template and policy).
- Migration context: `docs/plasim_zg_plev_migration_plan.md` (zg sigma → pressure-level)
- Dataset contract: `docs/plasim_makani_packager_plan.md` v9
- Source: `src/sfno_inference/`, `src/sfno_eval/`
- Submit entry: `scripts/submit_eval.sh`
