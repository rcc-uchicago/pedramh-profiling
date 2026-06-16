# zg pressure-level subset: drop 150 hPa, add 1000 hPa (v10.1)

**Status:** v4 — incorporates Codex round-3 bookkeeping (submit.slurm + submit_loop.slurm in §3.1, idempotent scancel, N=109 precheck). Codex round-2 approved modulo these. Awaiting final Zhixing sign-off before code edits.
**Date:** 2026-05-04.
**Predecessors:** docs/plasim_zg_plev_migration_plan.md (v7, the v9→v10 sigma→plev migration that introduced ZG_PLEV_HPA); docs/2026-05-02_ema_implementation_plan.md (EMA, orthogonal — preserved as-is).

## 1. Goal

Switch the 10-channel pressure-level subset from

    OLD: (150, 200, 250, 300, 400, 500, 600, 700, 850, 925)  hPa
to
    NEW: (200, 250, 300, 400, 500, 600, 700, 850, 925, 1000) hPa

The trainer contract (52 state + 1 diagnostic = 53 target channels, 6 forcing) is unchanged in **shape**. **Channel identity at every zg slot 42..51 changes** — slot 42 was `zg150`, becomes `zg200`; slots 43..50 each shift one position down (the channel that lived in slot k now lives in slot k−1); slot 51 was `zg925`, becomes `zg1000`. As a side-effect `zg500` moves from absolute index 47 to absolute index 46. Fresh-start training is mandatory; no checkpoint resume or weight transplant is permitted without explicit per-channel remapping (out of scope).

### Motivation

The user wants channel parity with ACE (which uses 200..1000 hPa). Including 1000 hPa is the only contentious change — over high-terrain cells (Tibet, Andes) PlaSim's postprocessor extrapolates `zg1000` below ground. Decision logged: include 1000 anyway, accept the extrapolation noise, plan to spot-check the zg1000 spatial means over high-terrain cells after first eval.

## 2. Single source of truth — and one drift gap to fix

`src/plasim_makani_packager/channels.py:30-32` defines `ZG_PLEV_HPA`. Most downstream consumers import the tuple directly (packager.py, validate.py, stats.py, packager tests) or name channels by string (`zg500`, `zg1000`). Eval scripts (`scripts/_eval_utils.py`, `scripts/score_nwp.py`, `scripts/render_eval_figures.py`) resolve `zg500` by name — unaffected by the index shift.

**Gap:** the SFNO YAML configs and the packager YAML template hard-code the full 53-element `channel_names` list — they are *not* generated from `TARGET_CHANNELS`, so they can silently drift from the tuple. The v10.1 work introduces a parser test (§3.2 below) that opens every zgplev YAML and asserts `channel_names == TARGET_CHANNELS`, so future plev-list changes can never desync the YAML without a test failure.

## 3. Files touched

### 3.1 Code (must edit)
| File | Change |
| --- | --- |
| `src/plasim_makani_packager/channels.py` | `ZG_PLEV_HPA` tuple → new values; rationale comment (the "drops 1000 (below-ground)" line is now historically inaccurate, replace with "includes 1000 for ACE parity; accepts below-ground extrapolation noise over high terrain"); asserts `STATE_CHANNELS[42] == "zg200"`, `STATE_CHANNELS[46] == "zg500"`, `STATE_CHANNELS[51] == "zg1000"`. |
| `src/plasim_makani_packager/templates/plasim_64x128_zgplev.yaml` | `channel_names` zg block + header comment. |
| `src/sfno_training/config/plasim_sim52_zgplev_full.yaml`     | `channel_names` zg block + header comment. |
| `src/sfno_training/config/plasim_sim52_zgplev_baseline.yaml` | `channel_names` zg block + header comment. |
| `src/sfno_training/config/plasim_sim52_zgplev_tiny.yaml`     | `channel_names` zg block + header comment. |
| `src/sfno_training/config/plasim_sim52_zgplev_short.yaml`    | `channel_names` zg block + header comment. |
| `src/sfno_training/config/plasim_sim52_zgplev_smoke.yaml`    | `channel_names` zg block + header comment. |
| `src/plasim_makani_packager/submit.slurm`      | Add `OVERWRITE_FLAG=()` env-gate so `OVERWRITE=1` opts into `--overwrite` (Codex round-2 finding #3). |
| `src/plasim_makani_packager/submit_loop.slurm` | Same env-gated `OVERWRITE_FLAG` pattern as submit.slurm. |

### 3.2 Tests (must edit / add)
| File | Change |
| --- | --- |
| `tests/plasim_makani_packager/test_channel_flatten.py` | L36-39 asserts (zg200/zg500@46/zg1000 + new tuple); L130-135 lev_2 lookup comments + indices (zg200 at lev_2 index 3 instead of zg150 at 2; zg500 at lev_2 index 7 stays). |
| `tests/plasim_makani_packager/test_metadata.py` | L88 (`zg200`), L89 (`zg1000`). |
| `tests/plasim_makani_packager/test_zg_plev_value_lookup.py` | L98-105: `ZG_PLEV_HPA[5]==500` → `ZG_PLEV_HPA[4]==500`; channel index 47 → 46; index 42 expected value 150 → 200; case-(c) "drop 925" still works (any element from the new tuple is fine). |
| **NEW** `tests/plasim_makani_packager/test_zgplev_yaml_channels.py` | Parse every zgplev YAML (5 SFNO configs + 1 packager template) and assert `yaml.safe_load(...)[block]["channel_names"] == TARGET_CHANNELS`. Drift guard against the §2 gap. |

### 3.3 Docs (must edit — keep in sync)
| File | Change |
| --- | --- |
| `docs/sfno_eval_plan.md:6`   | "zg150..zg925" → "zg200..zg1000" in the migration-summary line. |
| `docs/sfno_eval_plan.md:205` | "zg500 at index 47" → "zg500 at index 46"; channel range "zg150..zg925" → "zg200..zg1000". |

### 3.4 Out of scope (deliberately untouched)
- `docs/plasim_zg_plev_migration_plan.md` — historical record of the v9→v10 migration; do not rewrite history. This v10.1 plan supersedes the L1 level-selection rationale.
- `.pr_body_zgplev.md` — old PR body for the v10 PR; archival.
- `docs/plasim_postprocessor_*.md`, `docs/aires_rad_profile_plan.md`, `skills/plasim-postprocess/SKILL.md` — describe the postprocessor's full 13-level output `[50, 100, ..., 1000]`, which is unchanged. These docs already list 1000 as available.
- `src/plasim_postprocessor/plasim_postprocessor.py` — produces all 13 plevs; only the *subset selection* changes, not the source data.
- All eval scripts (`scripts/score_nwp.py`, `scripts/_eval_utils.py`, `scripts/render_eval_*.py`) — name-based zg500 lookup; index shift transparent to them.

## 4. Index-shift impact (zg500: 47 → 46)

Verified zero hardcoded `[47]` in eval scripts. The only callsites that index by 47 are:
- `channels.py:50` assert (updated above).
- `tests/plasim_makani_packager/test_channel_flatten.py:37`, `:135` (updated above).
- `tests/plasim_makani_packager/test_zg_plev_value_lookup.py:99`, `:101` (updated above).
- `docs/plasim_zg_plev_migration_plan.md` (historical, do not edit).

Trainer code, eval code, stats audit (`stats.py:64` uses `TARGET_CHANNELS.index("zg500")`) — all by-name. Safe.

## 5. Disk / runtime ops (in order)

### 5.1 Cancel queued job — no run-dir park needed

```bash
# scancel must be idempotent: by the time this runs, 3083640 may already
# have started/finished/been cancelled, in which case scancel returns
# non-zero and aborts the rest of the script under set -e.
scancel 3083640 2>/dev/null || true

# runs/sfno_zgplev_full does NOT currently exist on disk (only the
# .pre-ema-20260504 archive from the prior EMA-on cancel does), so no mv
# is normally needed. Verified via `ls $SCRATCH/SFNO_Climate_Emulator/runs/`.
[ -d "$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_full" ] && \
    mv "$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_full" \
       "$SCRATCH/SFNO_Climate_Emulator/runs/sfno_zgplev_full.pre-1000hpa-20260504"
```

### 5.2 Repackage in place — env-gated `--overwrite` via `submit_loop.slurm`

Two issues conflate here:

**Issue A — `--overwrite`.** `packager.py:493` skips existing H5 outputs unless `--overwrite` is passed. `submit.slurm:100` and `submit_loop.slurm` both lack the flag. Hardcoding `--overwrite` would leave packager submits permanently destructive-by-default — wrong. **Fix:** add **env-gated** support to both wrappers, so `OVERWRITE=1 sbatch ...` flips it on for THIS run only.

Edit pattern (apply to `src/plasim_makani_packager/submit.slurm` and `src/plasim_makani_packager/submit_loop.slurm`):

```bash
# Near the other *_FLAG arrays:
OVERWRITE_FLAG=()
if [[ "${OVERWRITE:-}" == "1" ]]; then OVERWRITE_FLAG=(--overwrite); fi
# In the python invocation, add:    "${OVERWRITE_FLAG[@]}" \
```

**Issue B — array cap.** With train 12-111 + valid 11 + test 121-128, N=109. Stampede3's `MaxArraySize=100` rejects `--array=0-108`. `submit_loop.slurm` was purpose-built for this case (its own header comment says "Used when the array size (109 in our v10 zgplev pack) exceeds Stampede3's MaxArraySize=100 limit"). It runs all 109 tasks in a single skx node with a bash-bg parallel pool. Use it.

(Alternative path Codex offered: shard `submit.slurm` into `YEAR_SLICE="11 100" --array=0-89` + `YEAR_SLICE="101 128" --array=0-18`. Note: `packager.py --count-tasks` returns the unsliced enumeration — packager.py:692 prints len(tasks) before any year-slice filter — so the shard sizes (90 and 19) are computed from the year ranges by hand, not from `--count-tasks`. We default to `submit_loop.slurm` since it exists for this purpose and avoids the by-hand shard counting.)

**Failure mode (in-place):** if the loop dies partway, the dataset contains a mix of old (150..925) and new (200..1000) H5s; validator rejects the whole tree and recovery is re-running the loop. Acceptable per user's "in-place" decision (recorded §1). Validator gate `--mode files` (§5.3) will catch it before stats burn 30-60 min.

```bash
# From login node:
cd $HOME/projects/SFNO_Climate_Emulator

# 5.2.1 — apply the OVERWRITE env-gating edit to submit.slurm + submit_loop.slurm.
#          (Done as part of code edits; no user action.)

# 5.2.2 — sanity precheck: enumeration must equal 109 (train:100 + valid:1 + test:8).
#          packager.py --count-tasks ignores --year-slice (packager.py:692 prints
#          the unsliced enumeration), which is exactly what we want here.
N=$(python3 -m plasim_makani_packager.packager \
        --sims 52 \
        --train-years 12 111 \
        --valid-years 11 11 \
        --test-years 121 128 \
        --count-tasks)
test "$N" = "109" || { echo "PRECHECK FAIL: N=$N expected 109"; exit 1; }

# 5.2.3 — submit the parallel-loop packager.
OVERWRITE=1 \
SIMS=52 \
POSTPROC_ROOT=$SCRATCH/SFNO_Climate_Emulator/data/postproc \
BOUNDARY_ROOT=$SCRATCH/SFNO_Climate_Emulator/data/boundary_astro \
OUTPUT_ROOT=$SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev \
TRAIN_YEARS="12 111" \
VALID_YEARS="11 11" \
TEST_YEARS="121 128" \
N_TASKS=109 \
PARALLEL=8 \
sbatch src/plasim_makani_packager/submit_loop.slurm
```

### 5.3 Validate H5s, run stats, validate stats — IN THIS ORDER

`validate --mode files` runs first to fail fast on a bad H5 (wrong `zg_pressure_levels_hpa` attr) before paying the ~30-60 min cost of `stats`.

```bash
python -m plasim_makani_packager.validate \
    --output-root $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev \
    --mode files

python -m plasim_makani_packager.stats \
    --output-root $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev \
    --epsilon 1e-8        # G-stats: pr_6h std is ~3e-7

python -m plasim_makani_packager.validate \
    --output-root $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev \
    --mode stats
```

zg500 audit (5400–5700 m mean) is unaffected by adding zg1000.

### 5.4 Regenerate metadata + rendered config (was missing in v1)

Training reads `metadata/data.json` (consumed by `train_plasim.py:207`). The packager loop does NOT update this file — it only writes per-year H5s. The metadata module renders both `metadata/data.json` and `config/<config_name>.yaml`.

**Critical (Codex round-2 finding #2):** `metadata.py:194-201` defaults to train 3-100 / valid 101-120 / test 121-128 — **not** the 12-111 / 11 / 121-128 split this dataset uses. We must pass year ranges explicitly:

```bash
python -m plasim_makani_packager.metadata \
    --output-root $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev \
    --variant zgplev \
    --train-years 12 111 \
    --valid-years 11 11 \
    --test-years 121 128
```

After this step, `data.json["coords"]["channel_state"][42]` should read `"zg200"`, `[46] == "zg500"`, `[51] == "zg1000"`.

### 5.5 Rebuild subset (training input)

```bash
rm -rf $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full
python scripts/build_subset_dataset.py \
    --src $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev \
    --dst $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full \
    --train-years 12-111 \
    --valid-years 11
```

`build_subset_dataset.py` symlinks `metadata/`, `config/`, `stats/` from src → dst (LINKED_DIRS). It builds train/ + valid/ symlink farms and creates an **empty** `test/`. It does NOT create `test_holdout/` — that is §5.6.

### 5.6 Rebuild test_holdout (NEW STEP — was missing in v1)

The eval workflow (`scripts/submit_eval_inference.slurm:79`, `scripts/submit_eval.sh:38`) reads from `sim52_zgplev_full/test_holdout/`. The previous step's wipe destroyed it; rebuild explicitly:

```bash
python scripts/build_test_split.py \
    --src $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev/test \
    --dst $SCRATCH/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full/test_holdout \
    --years 0121-0128
```

Optional: `test_holdout_only122/` and `test_holdout_only123/` (single-year ad-hoc dirs) were present pre-wipe. If still wanted, recreate manually with `--years 0122` / `--years 0123`.

### 5.7 Resubmit training

```
sbatch src/sfno_training/submit_zgplev_full.slurm
```

EMA-on, fresh start, no checkpoint to resume.

## 6. Validation gates

| Gate | Check |
| --- | --- |
| G-tuple | `ZG_PLEV_HPA == (200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)` and `STATE_CHANNELS[46] == "zg500"`. |
| G-tests | `pytest tests/plasim_makani_packager/` green after edits, **including the new `test_zgplev_yaml_channels.py` drift guard**. |
| G-pack | `validate --mode files` green: every new H5 has `zg_pressure_levels_hpa == new ZG_PLEV_HPA`. |
| G-z500 | mean_tgt[zg500] ∈ [5400, 5700] m (existing inline audit unchanged). |
| G-meta | `metadata/data.json` regenerated; `data.json["coords"]["channel_state"][46] == "zg500"`. |
| G-holdout | `sim52_zgplev_full/test_holdout/MOST.{0121..0128}.h5` symlinks present and resolve. |
| G-z1000-soft | After first eval, spot-check zg1000 fldmean over high-terrain cells (Tibet, Andes); not a hard gate. |
| G-train | First epoch logs show 53 target channels, EMA scalars present (regression guard from EMA work). |

## 7. Rollback

If the run trains poorly or the zg1000 below-ground extrapolation produces visible artifacts:
- `git revert` the channels.py + YAML commits.
- Move the new run dir aside, restore `sfno_zgplev_full.pre-1000hpa-*` to `sfno_zgplev_full`.
- The pre-1000hPa packager output is gone (in-place overwrite); recover by repackaging from postproc with the old tuple. Postproc source is preserved.

The eval-comparison workflow (v9 vs v10 vs v10.1) is supported by the channel-adaptive `_eval_utils.detect_z500_channel`.

## 8. Decisions log (formerly: open questions)

- **Repackage strategy:** in-place (Zhixing's call, recorded §1). `OUTPUT_ROOT`/`EXP_DIR` unchanged in `submit_zgplev_full.slurm`. Archival is via the `.pre-1000hpa-20260504` rename of the run dir (no parallel data-root rename; the dataset gets overwritten).
- **`--overwrite` policy:** env-gated (`OVERWRITE=1 sbatch ...`), not hardcoded — protects future packager submits from being silently destructive (Codex round-2 finding #3).
- **Array sizing path:** `submit_loop.slurm` over array sharding (Codex round-2 finding #1). Purpose-built for N>100 packs; avoids hand-counting shard sizes.
