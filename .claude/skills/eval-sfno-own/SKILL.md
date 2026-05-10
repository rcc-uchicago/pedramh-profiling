---
name: eval-sfno-own
description: Evaluate the user's own trained SFNO emulator (v10 zgplev) end-to-end on Stampede3. Orchestrates the 4-job inference → score → report → figures chain via scripts/submit_eval.sh. Supports MODE=nwp (12 ICs/yr × 8 yr × K=56) and MODE=climate (8 ICs × 1 yr). Auto-builds climatology and test_holdout when missing. Resolves default RUN_DIR within the production run family `$SCRATCH/AI-RES/runs/sfno_zgplev_full/plasim_sim52_zgplev_full/[0-9]+`. Use when the user asks to evaluate the trained emulator/checkpoint, run an NWP scorecard, run climate stats, regenerate `report.md`, render eval figures, or compare own-track trained runs. For the group SFNO-5410 track, defer to the sibling skill `eval-sfno-5410`.
---

# Evaluate the AI-RES own SFNO emulator

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

Own-track v10 zgplev emulator only. Driver: `scripts/submit_eval.sh` → 4-job chain. Output tree: `$WORK2/AI-RES/results/sfno_eval/<run_tag>/`. For 5410, defer to `eval-sfno-5410`.

## Inputs (env-var contract from `submit_eval.sh`)

All optional. Anything you don't set falls back per §default-resolution.

| Var | Default | Notes |
|---|---|---|
| `RUN_DIR` | (resolved per §default-resolution) | Training run dir containing `config.json`, `global_means.npy`, `training_checkpoints/`. |
| `CKPT` | `$RUN_DIR/training_checkpoints/best_ckpt_mp0.tar` | Override to evaluate a different checkpoint. |
| `MODE` | `nwp` | `nwp` = 12 ICs/yr × 8 yr × K=56 step rollouts. `climate` = 8 ICs × 1-yr rollouts (~1454 steps each). |
| `TEST_HOLDOUT` | `$SCRATCH/AI-RES/data/makani/sim52_zgplev_full/test_holdout` | Test h5 dir. Auto-built from `PACKAGER_TEST_SRC` if empty/missing. |
| `TRAIN_DIR` | `$SCRATCH/AI-RES/data/makani/sim52_zgplev_full/train` | Training pool — climatology source. |
| `PACKAGER_TEST_SRC` | `$SCRATCH/AI-RES/data/makani/sim52_astro_64x128_zgplev/test` | Source for `build_test_split.py` and `DATA_SHA7` lookup. |
| `BLOCKER_JOB_ID` | unset | If set, inference job runs `--dependency=afterok:$BLOCKER_JOB_ID`. Use to chain after a still-running training job. |
| `FULL_RUN_TAG` | `0` | `1` forces the legacy 4-SHA + ckpt run-tag template. Provenance always recorded in `provenance.txt` either way. |
| `BENCHMARK_5410_OUT_ROOT` | pinned valid 5410 run (`…/sfno_eval_5410/20260509_blocking_96ic_h100_packed_derecho_env_valid`) | Group SFNO-5410 result to overlay in the scorecard table and figures (always shown next to the own-track result). Set to empty string to disable; if the path's `scores/nwp_scorecard_summary.csv` is missing or empty, report.md prints a loud `⚠️` banner and figures fall back to own-only. |

## Default RUN_DIR resolution

When neither `RUN_DIR` nor `CKPT` is provided, resolve in this order. **Do not** scan globally by mtime across all `sfno_*` runs — that picks up smoke tests / microbench / incomplete runs.

1. **Explicit override.** If the user said `RUN_DIR=...` or `CKPT=...`, use it.
2. **Scan production family only:** `$SCRATCH/AI-RES/runs/sfno_zgplev_full/plasim_sim52_zgplev_full/[0-9]+`.
3. **Sort numerically by run number, descending** (`/3` before `/2` before `/1` before `/0`).
4. **Pick the first run that is BOTH valid AND complete:**
   - valid: `$cand/training_checkpoints/best_ckpt_mp0.tar` exists, size > 0
   - complete: training reached `max_epochs` — grep `out.log` for `"Training complete"`, `"max_epochs reached"`, or final `"epoch X/X"` matching `max_epochs` from `config.json`. If marker absent, treat as **incomplete** and try the next-lower run number.
5. **Refuse if nothing qualifies.** Print:
   > "No completed run found under `$SCRATCH/AI-RES/runs/sfno_zgplev_full/plasim_sim52_zgplev_full/[0-9]+`. Pass an explicit `RUN_DIR=...` or `CKPT=...` to override."

   Do NOT fall back to mtime, do NOT pick an incomplete run — confirm with the user instead.

### Reference resolver snippet

```bash
production_glob="$SCRATCH/AI-RES/runs/sfno_zgplev_full/plasim_sim52_zgplev_full"

if [[ -n "${RUN_DIR:-}" ]]; then
    resolved="$RUN_DIR"; source="explicit RUN_DIR"
elif [[ -n "${CKPT:-}" ]]; then
    resolved="$(dirname "$(dirname "$CKPT")")"; source="explicit CKPT"
else
    resolved=""; source="default production-family scan"
    for d in $(ls -1d "$production_glob"/[0-9]* 2>/dev/null \
               | awk -F/ '{print $NF, $0}' | sort -k1,1 -n -r | awk '{print $2}'); do
        ckpt="$d/training_checkpoints/best_ckpt_mp0.tar"
        log="$d/out.log"
        cfg="$d/config.json"
        [[ -s "$ckpt" ]] || continue
        if [[ -f "$log" && -f "$cfg" ]]; then
            max_ep="$(python -c "import json,sys; print(json.load(open('$cfg')).get('max_epochs',''))" 2>/dev/null)"
            if [[ -n "$max_ep" ]] && grep -qE "(Training complete|max_epochs reached|epoch ${max_ep}/${max_ep})" "$log"; then
                resolved="$d"; break
            fi
        fi
    done
    [[ -z "$resolved" ]] && { echo "ERROR: no completed run found under $production_glob/[0-9]+" >&2; exit 2; }
fi

ckpt_resolved="${CKPT:-$resolved/training_checkpoints/best_ckpt_mp0.tar}"
run_num="$(basename "$resolved")"
ckpt_mtime="$(stat -c '%y' "$ckpt_resolved" 2>/dev/null || echo unknown)"

cat <<EOF
[eval-sfno-own] resolved checkpoint
  source        : $source
  run number    : $run_num
  RUN_DIR       : $resolved
  CKPT          : $ckpt_resolved
  ckpt mtime    : $ckpt_mtime
EOF

export RUN_DIR="$resolved"
export CKPT="$ckpt_resolved"
```

**Always print all 5 lines (source, run number, RUN_DIR, CKPT, mtime) to the user before submitting** so they can stop you if it picked the wrong run.

## Preflight (before submit)

1. **Working tree.** `cd $HOME/AI-RES`. The submit script reads `git rev-parse --short=7 HEAD` for `EVAL_SHA7` — uncommitted changes still get evaluated under that SHA, which mis-attributes results. If `git status --porcelain` is non-empty, warn the user and ask whether to proceed.
2. **Venv.** `.venv/bin/python` must exist (the SLURMs source it).
3. **`RUN_DIR` sanity.** `$RUN_DIR/config.json` exists; `$RUN_DIR/global_means.npy` and `global_stds.npy` exist (53 floats each); `$CKPT` exists, size > 1 GB.
4. **`PACKAGER_TEST_SRC`.** At minimum `MOST.0121.h5` must be present (used for `DATA_SHA7`). Inference SLURM auto-builds `TEST_HOLDOUT` from this if empty.
5. **`TRAIN_DIR` non-empty.** Score SLURM auto-builds climatology from this on first eval; thereafter `$OUT_ROOT/baselines/climatology_proleptic.nc` is reused. First build: ~30–60 min wallclock.
6. **MODE.** Must be `nwp` or `climate`.

## Submit & return

After preflight, submit the chain. Do NOT inline-edit `submit_eval.sh` — pass overrides via env. Show the resolved provenance the script prints, then exit.

```bash
cd $HOME/AI-RES
RUN_DIR="$RUN_DIR" CKPT="$CKPT" MODE="$MODE" \
TEST_HOLDOUT="${TEST_HOLDOUT:-}" TRAIN_DIR="${TRAIN_DIR:-}" \
PACKAGER_TEST_SRC="${PACKAGER_TEST_SRC:-}" \
BLOCKER_JOB_ID="${BLOCKER_JOB_ID:-}" \
scripts/submit_eval.sh
```

The script:
1. Composes `RUN_TAG` and `OUT_ROOT=$WORK2/AI-RES/results/sfno_eval/$RUN_TAG`.
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
$WORK2/AI-RES/results/sfno_eval/<RUN_TAG>/
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

**Group SFNO-5410 benchmark overlay.** The report scorecard table and the figures (`rmse_vs_lead.png`, `acc_vs_lead.png`, `bias_<channel>.png`) always include the group SFNO-5410 result side-by-side with the own-track result, sourced from `BENCHMARK_5410_OUT_ROOT` (default: pinned valid 96-IC H100+packed-env run). Bias maps render as a 2-row layout (own on top, 5410 on bottom) with **separate colorbars per row** so unit differences are honest — own-track `pr_6h` is scaled m/s → mm/day for display, while 5410 `pr_6h` stays in native `kg m^-2` per 6h per the group convention. If the benchmark scorecard is missing/empty, the chain still completes and renders own-only with a loud warning.

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ERROR: no completed run found under ...` | All `/N` runs are incomplete or no `best_ckpt_mp0.tar`. | Pass `RUN_DIR=` explicitly. Don't auto-pick an incomplete run. |
| `DATA_SHA7=unknown` warning | `PACKAGER_TEST_SRC/MOST.0121.h5` missing. | Set `PACKAGER_TEST_SRC` to the right packager output dir. |
| `JOB_SCO` failed early | Climatology build OOM or wrong `TRAIN_DIR`. | Check `logs/sfno_eval_score_<id>.err`; verify `TRAIN_DIR` matches the dataset version of `RUN_DIR`. |
| `report.md` missing scorecard | Score job didn't write `nwp_scorecard.csv`. | Check `logs/sfno_eval_score_<id>.err` — usually a climatology channel-mismatch. |

## Status check (when user asks "is the eval done?")

```bash
squeue -u $USER -o "%.10i %.20j %.8T %.10M %.20S"  # running/pending
ls -lt $WORK2/AI-RES/results/sfno_eval/<RUN_TAG>/  # what's been written
```

- If `JOB_FIG` shows `CD` (complete) and `figures/` is non-empty → done; surface `report.md` and figure paths.
- If `JOB_REP` is `CD` but `JOB_FIG` is still queued/running → say "report ready, figures ETA ~1 min on skx-dev".
- If any job shows `F` / `CA` / `TO` → look in `logs/sfno_eval_<inference|score|report|figures>_<id>.err`.
- If anything earlier than `JOB_FIG` is still running → say "still running, ETA ~X min based on its `t` budget" and exit.

## Cross-references

- Plan: `docs/sfno_eval_plan.md` v2.8 (locked, implementation-ready)
- Sibling skill (5410 track): `.claude/skills/eval-sfno-5410/SKILL.md`
- Migration context: `docs/plasim_zg_plev_migration_plan.md` (zg sigma → pressure-level)
- Dataset contract: `docs/plasim_makani_packager_plan.md` v9
- Source: `src/sfno_inference/`, `src/sfno_eval/`
- Submit entry: `scripts/submit_eval.sh`
