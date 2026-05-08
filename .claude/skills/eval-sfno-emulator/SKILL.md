---
name: eval-sfno-emulator
description: Evaluate a trained SFNO emulator end-to-end on Stampede3. Orchestrates the inference → score → report → figures chain via scripts/submit_eval.sh for the user's own SFNO (v9 sigma or v10 zgplev), and handles the in-progress group SFNO-5410 track in smoke-only mode. Supports MODE=nwp (12 ICs/yr × 8 yr × K=56) and MODE=climate (8 ICs × 1 yr). Auto-builds climatology and test_holdout when missing. Resolves default RUN_DIR within the production run family `$SCRATCH/AI-RES/runs/sfno_zgplev_full/plasim_sim52_zgplev_full/[0-9]+`. Use when the user asks to evaluate a trained emulator/checkpoint, run an NWP scorecard, run climate stats, regenerate `report.md`, render eval figures, or compare trained runs.
---

# Evaluate a trained SFNO emulator

Orchestrate the SFNO emulator evaluation pipeline (`docs/sfno_eval_plan.md` v2.8). End-to-end: resolve checkpoint → preflight → submit 4-job chained SLURMs (inference → score → report → figures) → return job IDs → exit.

**You SUBMIT and RETURN.** The chain is multi-hour; do not block, poll, or sleep on it. After printing the four JOB IDs, `report.md` path, and `figures/` path, your job is done. The user re-invokes you to check status. Figure rendering is the 4th job in the chain — it is **not** a manual post-step under normal use.

## When to use this skill

Trigger when the user asks any of:

- "evaluate the trained emulator" / "evaluate the latest checkpoint" / "evaluate run /3"
- "run the NWP scorecard" / "run the eval" / "rerun the evaluation"
- "run climate stats" / "run climate-mode rollout"
- "regenerate `report.md`" / "render the eval figures" / "redo the bias maps"
- "evaluate this checkpoint: <path>"
- "evaluate the group 5410 emulator" → degraded path, see §5410 below

Do NOT trigger for: training jobs, packager runs, dataset migrations, or eval-plan editing.

## Two evaluation tracks

| Track | Status | Driver | Out tree |
|---|---|---|---|
| Own emulator (v9 sigma / v10 zgplev) | **production** | `scripts/submit_eval.sh` → 4-job chain | `$WORK2/AI-RES/results/sfno_eval/<run_tag>/` |
| Group SFNO-5410 | **in progress** (per `docs/2026-05-06_group_sfno_5410_eval_plan.md`) | `scripts/eval_inference_5410.py` + only `submit_eval_inference_5410_smoke.slurm` | `$WORK2/AI-RES/results/sfno_eval_5410/<run_tag>/` |

If the user asks for the 5410 path, **do not auto-launch a production run** — only the smoke SLURM exists. See §5410.

## Inputs (env-var contract from `submit_eval.sh`)

All optional. Anything you don't set falls back per §default-resolution.

| Var | Default | Notes |
|---|---|---|
| `RUN_DIR` | (resolved per §default-resolution) | Training run dir containing `config.json`, `global_means.npy`, `training_checkpoints/`. |
| `CKPT` | `$RUN_DIR/training_checkpoints/best_ckpt_mp0.tar` | Override to evaluate a different checkpoint. |
| `MODE` | `nwp` | `nwp` = 12 ICs/yr × 8 yr × K=56 step rollouts. `climate` = 8 ICs × 1-yr rollouts (~1454 steps each). |
| `TEST_HOLDOUT` | `$SCRATCH/AI-RES/data/makani/sim52_zgplev_full/test_holdout` (v10) | Test h5 dir. Auto-built from `PACKAGER_TEST_SRC` if empty/missing. |
| `TRAIN_DIR` | `$SCRATCH/AI-RES/data/makani/sim52_zgplev_full/train` (v10) | Training pool — climatology source. |
| `PACKAGER_TEST_SRC` | `$SCRATCH/AI-RES/data/makani/sim52_astro_64x128_zgplev/test` (v10) | Source for `build_test_split.py` and `DATA_SHA7` lookup. |
| `BLOCKER_JOB_ID` | unset | If set, inference job runs `--dependency=afterok:$BLOCKER_JOB_ID`. Use to chain after a still-running training job. |
| `FULL_RUN_TAG` | `0` | `1` forces the legacy 4-SHA + ckpt run-tag template. Provenance always recorded in `provenance.txt` either way. |

### v9 sigma overrides (older checkpoint)

```
RUN_DIR=$SCRATCH/AI-RES/runs/sfno_full/plasim_sim52_full/0
TEST_HOLDOUT=$SCRATCH/AI-RES/data/makani/sim52_full/test_holdout
TRAIN_DIR=$SCRATCH/AI-RES/data/makani/sim52_full/train
PACKAGER_TEST_SRC=$SCRATCH/AI-RES/data/makani/sim52_astro_64x128/test
```

The scoring scripts auto-detect Z500 channel name (`zg5` for v9, `zg500` for v10) from the inference NetCDFs; the gate threshold is the same.

## Default RUN_DIR resolution

When neither `RUN_DIR` nor `CKPT` is provided by the user, resolve in this order. **Do not** scan globally by mtime across all `sfno_*` runs — that picks up smoke tests / microbench / incomplete runs.

1. **Explicit override.** If the user said `RUN_DIR=...` or `CKPT=...`, use it. Skip steps 2–4.
2. **Scan production family only:** `$SCRATCH/AI-RES/runs/sfno_zgplev_full/plasim_sim52_zgplev_full/[0-9]+`.
3. **Sort numerically by run number, descending** (`/3` before `/2` before `/1` before `/0`).
4. **Pick the first run that is BOTH valid AND complete:**
   - valid: `$cand/training_checkpoints/best_ckpt_mp0.tar` exists, size > 0
   - complete: training reached `max_epochs` — grep `out.log` for a clear training-complete marker (`"Training complete"`, `"max_epochs reached"`, or final `"epoch X/X"` line where both sides match `max_epochs` from `config.json`). If the marker isn't found, treat as **incomplete** and try the next-lower run number.
5. **Refuse if nothing qualifies.** Print:
   > "No completed run found under `$SCRATCH/AI-RES/runs/sfno_zgplev_full/plasim_sim52_zgplev_full/[0-9]+`. Pass an explicit `RUN_DIR=...` or `CKPT=...` to override."

   Do NOT fall back to mtime, do NOT fall back to v9, do NOT pick an incomplete run — confirm with the user instead.

### Reference resolver snippet

Run this Bash snippet to do the resolution and print the required provenance. Edit only the `production_glob` to retarget if the user names a different family.

```bash
production_glob="$SCRATCH/AI-RES/runs/sfno_zgplev_full/plasim_sim52_zgplev_full"

if [[ -n "${RUN_DIR:-}" ]]; then
    resolved="$RUN_DIR"; source="explicit RUN_DIR"
elif [[ -n "${CKPT:-}" ]]; then
    resolved="$(dirname "$(dirname "$CKPT")")"; source="explicit CKPT"
else
    resolved=""; source="default production-family scan"
    # Sort numerically, descending, by trailing run number.
    for d in $(ls -1d "$production_glob"/[0-9]* 2>/dev/null \
               | awk -F/ '{print $NF, $0}' | sort -k1,1 -n -r | awk '{print $2}'); do
        ckpt="$d/training_checkpoints/best_ckpt_mp0.tar"
        log="$d/out.log"
        cfg="$d/config.json"
        [[ -s "$ckpt" ]] || continue
        # Completeness check: max_epochs from config matched in out.log.
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
[eval-sfno-emulator] resolved checkpoint
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

Run these checks; halt and surface issues to the user rather than papering over them.

1. **Working tree.** `cd $HOME/AI-RES`. The submit script reads `git rev-parse --short=7 HEAD` for `EVAL_SHA7` — uncommitted changes still get evaluated under that SHA, which mis-attributes results. If `git status --porcelain` is non-empty, warn the user and ask whether to proceed.
2. **Venv.** `.venv/bin/python` must exist (the SLURMs source it).
3. **`RUN_DIR` sanity.**
   - `$RUN_DIR/config.json` exists.
   - `$RUN_DIR/global_means.npy` and `global_stds.npy` exist (53 floats each).
   - `$CKPT` exists, size > 1 GB (sanity).
4. **`PACKAGER_TEST_SRC`.** At minimum `MOST.0121.h5` must be present (used for `DATA_SHA7`). Inference SLURM auto-builds `TEST_HOLDOUT` from this if empty.
5. **`TRAIN_DIR` non-empty.** Score SLURM auto-builds climatology from this on first eval; thereafter `$OUT_ROOT/baselines/climatology_proleptic.nc` is reused. Wall-clock for first build: ~30–60 min.
6. **MODE.** Must be `nwp` or `climate`. Anything else → reject.
7. **`logs/` dir.** `mkdir -p logs` (submit script does this, but harmless).

## Submit & return

After preflight passes, submit the chain. Do NOT inline-edit `submit_eval.sh` — pass overrides via env. Show the resolved provenance the script prints, then exit.

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
4. Prints `[submit_eval] inference job: <id>`, `scoring job: <id>`, `report job: <id>`, `figures job: <id>` and the final artifact paths.

**Surface to the user (verbatim):**
- `RUN_TAG`
- `OUT_ROOT`
- All four job IDs (`JOB_INF`, `JOB_SCO`, `JOB_REP`, `JOB_FIG`)
- Final `report.md` path and `figures/` path
- Reminder: "Re-invoke me with `squeue -u $USER` or 'is the eval done' to check status."

Do NOT poll. Do NOT sleep. Do NOT use `Monitor` to wait for the chain.

## Figure rendering (chained via JOB_FIG)

`render_eval_figures.py` produces PNGs (`rmse_vs_lead.png`, `acc_vs_lead.png`, `bias_<channel>.png × 5`) into `$OUT_ROOT/figures/`. It is the 4th job in the chain (`submit_eval_figures.slurm`, skx-dev, CPU-only), launched automatically with `--dependency=afterok:$JOB_REP` by `submit_eval.sh`. **Do not** invoke it manually under normal use.

When the user asks "render figures" or "redo the figures" for an eval that already exists (i.e. `$OUT_ROOT/scores/nwp_scorecard_summary.csv` is present), prefer the manual fallback below over re-running the full chain — but only when the eval ran already and you only want fresh PNGs (e.g. after editing `render_eval_figures.py`):

```bash
source $HOME/AI-RES/.venv/bin/activate
python $HOME/AI-RES/scripts/render_eval_figures.py \
    --out-root "$OUT_ROOT"
```

Pre-conditions for either path: `$OUT_ROOT/scores/nwp_scorecard_summary.csv` and `$OUT_ROOT/scores/bias_maps_*.npy` must exist. If they don't (score job hasn't finished), surface the missing files rather than running anything.

## Group SFNO-5410 path (in progress)

Per `docs/2026-05-06_group_sfno_5410_eval_plan.md` and `docs/2026-05-06_group_sfno_5410_climatology_prompt_for_derecho.md`. Climatology is built off-platform (Derecho); local production pipeline is **not yet built**.

Available locally:
- `scripts/eval_inference_5410.py` — inference driver
- `scripts/submit_eval_inference_5410_smoke.slurm` — smoke SLURM (limited ICs/files)
- (no full submit chain, no `submit_eval_score_5410`, no `submit_eval_report_5410`)

When the user asks to evaluate the 5410 emulator:

1. **Refuse to auto-launch a production run.** Say: "The group SFNO-5410 production pipeline is in progress per `docs/2026-05-06_group_sfno_5410_eval_plan.md`. Only a smoke SLURM exists locally; full inference→score→report chain is not yet built. I can launch the smoke run, or wait for you to confirm the production path is ready."
2. **Smoke launch (only if user explicitly asks):**
   ```bash
   sbatch $HOME/AI-RES/scripts/submit_eval_inference_5410_smoke.slurm
   ```
3. **Do NOT** invent a production chain. **Do NOT** copy `submit_eval.sh` for 5410 unless the user asks for that as a separate task. Building the missing pieces is a feature request, not a routine eval.

Group conventions to remember (from project memory): `pl = ln(p_s)`, `zg` is in geopotential metres, `pr_6h` is rate × 6h. **Do not unit-convert** these on top of the group's pipeline outputs.

## Output tree (own emulator)

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

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ERROR: no completed run found under ...` | All `/N` runs are incomplete or no `best_ckpt_mp0.tar`. | Pass `RUN_DIR=` explicitly. Don't auto-pick an incomplete run. |
| `DATA_SHA7=unknown` warning | `PACKAGER_TEST_SRC/MOST.0121.h5` missing. | Set `PACKAGER_TEST_SRC` to the right packager output dir. |
| `JOB_SCO` failed early | Climatology build OOM or wrong `TRAIN_DIR`. | Check `logs/sfno_eval_score_<id>.err`; verify `TRAIN_DIR` matches the dataset version of `RUN_DIR`. |
| Z500 ACC gate fails on v9 ckpt | Plan v2.8 §D.6: gate detects channel by name. | Both `zg5` (v9) and `zg500` (v10) are scorable; if it complains, the inference NetCDF is missing zg channels — re-check `RUN_DIR`/`CKPT`. |
| Inference SLURM hits CPU/GPU mismatch | Pre-v2.6 bug; should not occur on current `src/sfno_inference/checkpoint_loader.py`. | If it does, regenerate the checkpoint loader from §B.0 of the plan. |
| `report.md` missing scorecard | Score job didn't write `nwp_scorecard.csv`. | Check `logs/sfno_eval_score_<id>.err` — usually a climatology channel-mismatch (v9 vs v10). |

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
- 5410 sibling plan: `docs/2026-05-06_group_sfno_5410_eval_plan.md`
- 5410 climatology prompt (Derecho): `docs/2026-05-06_group_sfno_5410_climatology_prompt_for_derecho.md`
- Migration context: `docs/plasim_zg_plev_migration_plan.md` (zg sigma → pressure-level)
- Dataset contract: `docs/plasim_makani_packager_plan.md` v9
- Source: `src/sfno_inference/`, `src/sfno_eval/`
- Submit entry: `scripts/submit_eval.sh`
