# Plan: SST sea-ice handling fix (PlaSim-faithful `surface` mode) — rev 5

Date: 2026-05-10 (rev 5, fourth Codex pass). Status: **awaiting review**, no code changes applied.

## Changes vs rev 4

Two new blocking issues verified against the actual code:

- **B-rev5-1 `--epsilon` default fails on `pr_6h`**: `stats.py:40` sets `MIN_STD_EPSILON = 1e-6`, but the comment immediately below acknowledges that `pr_6h` global std is ~2.8e-7 because precipitation is mostly zero in PlaSim. v10 stats confirm `pr_6h = 2.8087172e-07`. With the default, both `stats.py` and `validate --mode stats` would hard-fail. Fix → pass `--epsilon 1e-8` to every `stats.py` and every `validate --mode stats` invocation in this plan.
- **B-rev5-2 Phase B direct packager call has no postprocessor SHA**: `validate.py:280-283` rejects `postprocessor_git_sha == "unknown"` unless `--allow-unknown-postproc-sha`. Production runs are fine because `src/plasim_makani_packager/submit.slurm:104-105` computes the SHA via `git -C "$POSTPROC_SOURCE_DIR" rev-parse HEAD`. Phase B uses a direct `python3 -m plasim_makani_packager.packager` call that doesn't supply the SHA, so `validate --mode files` would reject. Fix → compute the SHA at the start of Phase B and pass `--postprocessor-git-sha "$POSTPROC_SHA"` to the packager call. (Codex preferred passing the SHA over relaxing validation.)

(All rev-3 and rev-4 fixes remain in place; this rev only adds the two CLI hardenings above.)

---

## Context

Our `boundary_astro` adaptor currently collapses every sea-ice ocean cell to exactly 271.35 K via a freezing-point clamp in `compute_sst` (`src/emulator_adaptor/adaptor.py:94`). Numerical evidence (year-11 snapshot): PlaSim's `ts` has a 232–273 K range over 1034 ice-covered ocean cells; our output collapses all of them to 271.35 K, throwing away the 30–40 K cold-ice signal that PlaSim physics encodes and that upstream 5410 preserves.

For an atmospheric emulator trained against PlaSim, this is the wrong call: the atmosphere couples to the actual surface skin temperature (water OR ice). The SST-like boundary channel must carry PlaSim's `ts` over all non-land cells. The clamp was inherited from ERA5/CMIP reanalysis convention, which is *not* the right convention for a PlaSim emulator.

This plan adds a `--sst-mode` flag with default `surface` (PlaSim-faithful: pass `ts` through, no clamp, NaN only over land) and keeps the existing clamped behavior as opt-in `ocean_era5` for back-compat. Output datasets are produced under a new v11 path so v10 stays intact for back-comparison.

---

## 1. Files / functions to edit

### 1.1 `src/emulator_adaptor/adaptor.py`

| Site | Change |
|---|---|
| Module constant (line 67) | `SST_MIN_K: float = 170.0` (covers PlaSim's coldest ice-surface temps ~209 K with margin; was 271.34 = effectively the clamp value). `SST_MAX_K = 310.0` unchanged. |
| Module docstring (lines 19-32) | Document the new `surface` mode and that the `sst` channel is now *surface temperature over non-land* rather than CF `sea_surface_temperature`. Note both modes in the variable description. |
| `compute_sst` (lines 76-101) | Add `sst_mode: str = "surface"` arg. Branch: `surface` → drop the `sic` requirement, drop the icy clamp, drop the freezing floor; emit `sst = xr.where(ds["lsm"] < LAND_EPSILON, ds["ts"], np.nan)`. `ocean_era5` → existing clamp + floor logic preserved bit-for-bit. Update returned `attrs["long_name"]` per mode: `"surface_temperature_over_non_land"` or `"sea_surface_temperature_era5_convention"`. |
| `process_one` (line 351) | `sst = compute_sst(ds, opts.sst_mode)` |
| `process_one` attrs block (lines 366-371) | After the `rsdt_method` stamping, add `out.attrs["sst_mode"] = opts.sst_mode`. |
| `_parse_args` (after line 415) | Add `--sst-mode {surface, ocean_era5}` with `default="surface"`. Help string makes the choice explicit. |
| `_validate` (lines 266-295) | The `sst_nan_frac == lsm_land_frac` assertion still holds under `surface` mode (NaN only over land); no change to the assertion itself. Update the log line `"sst range over ocean: ..."` to `"sst range over non-land: ..."` and include `mode` in the log line for traceability. |

### 1.2 `src/emulator_adaptor/submit.slurm`

| Site | Change |
|---|---|
| Lines 14-18 (var assignment block) | Change to env-preserving pattern matching the packager submit (`SIMS="${SIMS:-}"`, `YEAR_START="${YEAR_START:-}"`, etc.). Comment block updated to point at `sbatch --export=ALL,VAR=...` as the documented invocation form. |
| Line 24 (existing `RSDT_METHOD=`) | Convert to `RSDT_METHOD="${RSDT_METHOD:-}"` for the same env-preservation reason. |
| New optional var block (~line 25) | `SST_MODE="${SST_MODE:-}"` |
| Flag-building block (~line 62) | Add `SST_MODE_FLAG=()` then `if [[ -n "$SST_MODE" ]]; then SST_MODE_FLAG=(--sst-mode "$SST_MODE"); fi`. |
| Python invocation (~line 67) | Splice `"${SST_MODE_FLAG[@]}"` into the python call between `"${RSDT_METHOD_FLAG[@]}"` and `--task-index`. |

This patch is the **smallest** edit that lets the documented `sbatch --export=ALL,SIMS=52,...,RSDT_METHOD=astronomical,SST_MODE=surface submit.slurm` form work without hand-editing the file per run.

### 1.3 `src/plasim_makani_packager/packager.py`

| Site | Change |
|---|---|
| `file_attrs` dict in the per-(sim,year) H5 writer (lines 540-568, the block ending at `_write_h5(...)`) | Insert one line: `"sst_mode": str(boundary_ds.attrs.get("sst_mode", "ocean_era5")),`. The default `"ocean_era5"` is for back-compat with v10 boundary files that lack the attr; never silently writes `surface` unless the boundary file actually says so. The hardcoded `"rsdt_method": "astronomical"` stays (already guarded by `_assert_rsdt_method` upstream). |

No other change to `packager.py`. The land-fill `np.where(np.isnan(sst), sst_land_fill_k, sst)` step at line 181 stays — under `surface` mode it now operates exclusively on land cells (the only NaN cells), which is the desired semantic.

### 1.4 `src/plasim_makani_packager/metadata.py`

| Site | Change |
|---|---|
| `attrs` dict (lines 112-128) | Add `"sst_mode": sst_mode,` to the dict. Parameterize `source_boundary_root` — read it from a CLI arg (next row) instead of hardcoding `/scratch/.../boundary_astro/sim52`. |
| `_parse_args` (around line 204) | Add `--sst-mode` (string, default `ocean_era5` for back-compat; CLI caller passes `surface` explicitly for v11). Add `--source-boundary-root` (str, default = current hardcoded v10 path for back-compat). |
| The function building `metadata` dict (signature change) | Thread the new args through. Pass them into the attrs dict above. |
| **NEW self-consistency check** (per Codex rev-3 minor suggestion, rev-4 back-compat fix) | After parsing args and before writing `data.json`, glob one H5 from `{output_root}/train/`, open it with `h5py`, and assert provenance attrs match the CLI. For `sst_mode`: read `actual = f.attrs.get("sst_mode", "ocean_era5")` (decode bytes if present) and assert it equals `args.sst_mode` — the `.get()` default lets the same metadata.py work against legacy v10 H5 files (which lack the attr) when run with `--sst-mode ocean_era5`. For `rsdt_method`: read the attr directly (always present after the §1.3 packager already hardcodes it) and assert equality. Both failures raise with a clear error message naming the CLI value and the H5 actual value. This makes it structurally impossible for `metadata.json[attrs].sst_mode` to lie about the dataset, while preserving v10 back-compat. |

### 1.5 What does **not** change

- `src/plasim_makani_packager/submit.slurm` — already env-preserving (`${VAR:-}` pattern from line 17 onwards). Just needs the right env vars at submit time.
- `src/plasim_makani_packager/stats.py` — invariant under the SST change. Stats automatically reflect new SST distribution.
- `src/plasim_makani_packager/validate.py` — structural and makani-smoke modes already validate finite-ness etc.; no schema change.
- `scripts/build_subset_dataset.py` — invariant; just point it at the v11 roots.
- `scripts/build_boundary_dir.py` — never invokes the adaptor; reads adaptor output. No change.
- `scripts/package_sim52_astro.sh` — already documents the 5-phase pipeline; we follow it.

---

## 2. CLI / config flags and defaults

| Flag | Where exposed | Default | Choices | Notes |
|---|---|---|---|---|
| `--sst-mode` (adaptor) | `src/emulator_adaptor/adaptor.py::_parse_args` | `surface` | `surface`, `ocean_era5` | New default = PlaSim-faithful. Legacy mode reproduces v10 SST bit-for-bit at the sst-array level. |
| `SST_MODE` env var | `src/emulator_adaptor/submit.slurm` | unset → adaptor uses its own `surface` default | string | New, forwards to `--sst-mode`. |
| `RSDT_METHOD` env var | `src/emulator_adaptor/submit.slurm` (existing) | unset → adaptor uses `arithmetic` (NOT what we want) | `arithmetic`, `astronomical` | **Must be set to `astronomical` for v11** (packager hard-asserts it). |
| `--sst-mode` (metadata) | `src/plasim_makani_packager/metadata.py::_parse_args` | `ocean_era5` (back-compat with v10) | string | For v11 pass `surface` explicitly. Metadata also asserts H5 sample agrees (rev-3 minor). |
| `--source-boundary-root` (metadata) | same | hardcoded v10 path | path | For v11 pass `/scratch/.../boundary_astro_v11/sim52` explicitly. |
| `--packager-version` (metadata, existing) | same | `sim52_astro_64x128_zgplev` (= `DEFAULT_PACKAGER_VERSION`) | string | For v11 pass `sim52_astro_64x128_zgplev_v11_surface_sst` explicitly. The constant in source is unchanged. |

Provenance chain (now self-consistency-checked end-to-end):
1. Adaptor writes `boundary.YYYY.nc[attrs.sst_mode] = "surface"` (or `ocean_era5`).
2. Packager reads `boundary_ds.attrs.get("sst_mode", "ocean_era5")` and writes it into every per-(sim,year) H5 `file_attrs`.
3. `metadata.py` CLI receives `--sst-mode surface` and writes it into `metadata.json[attrs].sst_mode`, **after asserting** a sample H5 file's `attrs.sst_mode` matches.

---

## 3. Dataset version / provenance naming (two-tier layout, matches v10)

v10 uses a **two-tier** layout (verified): an underlying packager root with one H5 per (sim, year) covering train [12, 111] + valid [11, 11] + test [121, 128], plus a `build_subset_dataset.py`-built symlink farm that filters to the train+valid production layout (test/ left empty; test_holdout/ populated by a separate step). v11 replicates the same shape.

| Layer | v10 (existing) | v11 (new) |
|---|---|---|
| Adaptor output | `/scratch/.../AI-RES/data/boundary_astro/sim52/boundary.YYYY.nc` | `/scratch/.../AI-RES/data/boundary_astro_v11/sim52/boundary.YYYY.nc` |
| Adaptor NetCDF attrs | `rsdt_method=astronomical` only | adds `sst_mode=surface` |
| Underlying packager root | `/scratch/.../AI-RES/data/makani/sim52_astro_64x128_zgplev/` (`train/MOST.0012.h5`–`MOST.0111.h5`, `valid/MOST.0011.h5`, `test/MOST.0121.h5`–`MOST.0128.h5`) | `/scratch/.../AI-RES/data/makani/sim52_astro_64x128_zgplev_v11/` (same year ranges) |
| Per-H5 file attrs | `rsdt_method=astronomical`, `sst_land_fill_K=271.35`, ... | adds `sst_mode=surface` |
| Stats files | `…/sim52_astro_64x128_zgplev/stats/` | `…/sim52_astro_64x128_zgplev_v11/stats/` |
| Top-level metadata | `…/sim52_astro_64x128_zgplev/metadata/data.json` (`packager_version=sim52_astro_64x128_zgplev`, `source_boundary_root=…/boundary_astro/sim52`, `train_years=[12,111]`, `valid_years=[11,11]`, `test_years=[121,128]`) | `…/sim52_astro_64x128_zgplev_v11/metadata/data.json` (`packager_version=sim52_astro_64x128_zgplev_v11_surface_sst`, `source_boundary_root=…/boundary_astro_v11/sim52`, **`sst_mode=surface`**, same year ranges) |
| Subset (training) farm | `/scratch/.../AI-RES/data/makani/sim52_zgplev_full/` (`train/` symlinks for years 12-111, `valid/` symlink for year 11, `test/` **empty**, `test_holdout/` populated separately with years 121-128) | `/scratch/.../AI-RES/data/makani/sim52_zgplev_full_v11/` (same shape: subset script handles train+valid only; `test_holdout/` populated by a separate symlink step) |
| Training YAML | `src/sfno_training/config/plasim_sim52_zgplev_full.yaml` (paths point at v10) | **out of scope for this plan** — a follow-up training plan will introduce `plasim_sim52_zgplev_full_v11.yaml` |

Nothing in v10 is touched.

---

## 4. Validation script / checks

Three phases, each strict-precondition for the next.

### Phase A — unit-level (in-process; runs in <1 min)

Read one existing postproc NetCDF (`/scratch/.../postproc/sim52/MOST.0011.nc`) and call the patched `compute_sst(ds, sst_mode=...)` in both modes. Assert:

| # | Assertion |
|---|---|
| A1 | `surface` mode: at least one ocean cell with `sic > 0.5` returns `ts < 270 K` (cold-ice signal present). |
| A2 | `surface` mode: NaN cells exactly equal cells with `lsm >= LAND_EPSILON` (no NaN over ocean). |
| A3 | `surface` mode: fraction of cells == exactly 271.35 K is below 0.1 % (clamp signature gone). |
| A4 | `surface` mode: all finite values satisfy `SST_MIN_K <= v <= SST_MAX_K` (170 / 310). |
| A5 | `ocean_era5` mode: every ocean cell with `sic > 0.5` returns exactly 271.35 K (clamp reproduced). |
| A6 | `ocean_era5` mode: sst-values array is **bit-for-bit equal** to the existing v10 adaptor output on the same input (compare `.values`, not attrs; new mode attrs will differ). |
| A7 | Provenance: `compute_sst`-callers produce output with `attrs.long_name` differing per mode. |
| A8 | LSM is binary on this input (`unique(lsm) == {0., 1.}`) — validates the simplification in M3-rev2. |

### Phase B — smoke regeneration on year 11

0. **Compute the postprocessor git SHA** (Phase B direct-Python path; not auto-resolved as in production submit.slurm):
   `POSTPROC_SHA=$(git -C $HOME/projects/SFNO_Climate_Emulator/src/plasim_postprocessor rev-parse HEAD 2>/dev/null || echo unknown)`
1. **Adaptor** — process year 11 only with `--sst-mode surface --rsdt-method astronomical`, write to `boundary_astro_v11_smoke/sim52/`.
2. **Packager** — process year 11 only via `packager.py` (`--train-years 11 11 --valid-years 11 11 --test-years 11 11 --postprocessor-git-sha "$POSTPROC_SHA"`) with `--boundary-root .../boundary_astro_v11_smoke` and `--output-root .../sim52_astro_64x128_zgplev_v11_smoke`. **No `--packager-version` flag here** — that lives on `metadata.py`. Note: `resolve_split` checks train first (`packager.py:441`), so year 11 lands in `train/` only; `valid/` and `test/` will be empty in the smoke root. This is why step 5 below avoids `--mode smoke-live` / `full` (both need a `valid/` file). The `--postprocessor-git-sha` flag is required because `validate --mode files` (step 5a) rejects `"unknown"` by default (`validate.py:280-283`).
3. **Stats** — `python3 -m plasim_makani_packager.stats --output-root .../sim52_astro_64x128_zgplev_v11_smoke --train-years 11 11 --epsilon 1e-8 -v`. The `--epsilon 1e-8` is required because `pr_6h` global std is ~2.8e-7 (`stats.py:40-44`) and the default `1e-6` would hard-fail.
4. **Metadata** — `python3 -m plasim_makani_packager.metadata --output-root .../sim52_astro_64x128_zgplev_v11_smoke --exp-dir <tmp> --rsdt-method astronomical --sst-land-fill-k 271.35 --sst-mode surface --source-boundary-root .../boundary_astro_v11_smoke/sim52 --packager-version sim52_astro_64x128_zgplev_v11_smoke --train-years 11 11 --valid-years 11 11 --test-years 11 11 -v`.
5. **Validate — explicit modes** (avoid `structural` alias, which `validate.py:600` maps to `full` = files + stats + smoke-synthetic + **smoke-live**; smoke-live needs `valid/` to be non-empty, which it isn't in the 1-year smoke root):
   - `python3 -m plasim_makani_packager.validate --output-root .../sim52_astro_64x128_zgplev_v11_smoke --mode files -v`
   - `python3 -m plasim_makani_packager.validate --output-root .../sim52_astro_64x128_zgplev_v11_smoke --mode stats --epsilon 1e-8 -v`
   - Keep B7 (below) as the custom `PlasimForcingDataset` loader smoke (no `--mode smoke-live`).

Then assert:

| # | Assertion |
|---|---|
| B1 | Adaptor NetCDF: `attrs.sst_mode == "surface"` and `attrs.rsdt_method == "astronomical"`. |
| B2 | Adaptor NetCDF: SST `min < 270 K` (cold-ice signal present), `max < 310 K`. |
| B3 | Packaged H5 file attrs: `sst_mode == "surface"`, `rsdt_method == "astronomical"`. |
| B4 | Packaged H5 forcing tensor: `np.isfinite(forcing).all()` (no NaN, no Inf). |
| B5 | Stats: `forcing_global_means.npy[3]` (sst channel) in `[270, 282]` K and `forcing_global_stds.npy[3]` in `[15, 25]` K (cold-ice broadens the distribution vs v10's 282 / 12). |
| B6 | `metadata.json[attrs]`: `sst_mode == "surface"`, `source_boundary_root` ends in `_v11_smoke/sim52`, `packager_version == "sim52_astro_64x128_zgplev_v11_smoke"`. |
| B7 | Loader smoke (load one batch through `PlasimForcingDataset` pointing at the smoke dataset): all returned tensors finite, all channels in expected post-normalization ranges. |
| B8 | (new, rev-3) `metadata.py`'s self-consistency check passes: it raised no error during step 4, meaning a sample H5's `attrs.sst_mode == "surface"`. |

### Phase C — full-regeneration spot-checks

Run after the full 128-year regeneration completes (commands in §5). Assert across **all** files in the v11 underlying root:

| # | Assertion |
|---|---|
| C1 | Every `boundary_astro_v11/sim52/boundary.YYYY.nc` has `attrs.sst_mode == "surface"` and `attrs.rsdt_method == "astronomical"`. |
| C2 | Every per-(sim, year) H5 in `sim52_astro_64x128_zgplev_v11/{train,valid,test}` has `attrs.sst_mode == "surface"`. |
| C3 | Across all H5 files in train/valid/test: `np.isfinite(state).all()`, `np.isfinite(diagnostic).all()`, `np.isfinite(forcing).all()`. Single-file fail prints sim/year. |
| C4 | Aggregate SST channel statistics (over the train split only) match Phase B5 bands within ±0.5 K. |
| C5 | `sim52_zgplev_full_v11/metadata/data.json` matches `sim52_zgplev_full/metadata/data.json` byte-for-byte **except** for: `source_boundary_root`, `packager_version`, `sst_mode`. No accidental drift in channel order, level lists, train/valid/test year ranges. |
| C6 | `sim52_zgplev_full_v11/train/MOST.YYYY.h5` for each Y in [12, 111] is a symlink (not a copy) into the v11 underlying root; `sim52_zgplev_full_v11/valid/MOST.0011.h5` is a symlink. |
| C7 | `sim52_zgplev_full_v11/test/` is empty (matches v10). `test_holdout/` symlinks for years 121-128 exist (created by the separate post-subset step in §5). |
| C8 | Loader-batch probe (mirroring `analysis_outputs/upstream_loader_nan_probe.py`): `PlasimForcingDataset` on the v11 farm returns zero-NaN tensors for the first 10 batches. |

Validation scripts live under `analysis_outputs/`:
- `validate_v11_sst_handling_phaseA.py` (unit; runs locally)
- `validate_v11_sst_handling_phaseB.py` (smoke; coordinates the 5-phase mini-pipeline)
- `validate_v11_sst_handling_phaseC.py` (post-regeneration sweep; few minutes for 128 years)

---

## 5. Commands for regenerating v11

Documented for execution **after** plan approval. All commands assume `cd $HOME/projects/SFNO_Climate_Emulator` and venv activated.

```bash
# === Phase 0 — adaptor (boundary fields with sst_mode=surface, rsdt=astronomical) ===
# Prerequisite: src/emulator_adaptor/submit.slurm patched to ${VAR:-} pattern per §1.2.
SIMS="52"; YEAR_START=1; YEAR_END=128                   # 128, not 129 (MOST.0129.nc does not exist)
N=$(python3 src/emulator_adaptor/adaptor.py --sims $SIMS --years $YEAR_START $YEAR_END --count-tasks)
sbatch --array=0-$((N-1)) \
    --export=ALL,SIMS="$SIMS",YEAR_START=$YEAR_START,YEAR_END=$YEAR_END,\
INPUT_ROOT=/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/postproc,\
OUTPUT_ROOT=/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/boundary_astro_v11,\
RSDT_METHOD=astronomical,\
SST_MODE=surface \
    src/emulator_adaptor/submit.slurm

# (Wait for completion.)

# === Phase 1 — packager (writes one H5 per (sim, year) under v11 root) ===
SIMS=52
TRAIN_YEARS="12 111"; VALID_YEARS="11 11"; TEST_YEARS="121 128"   # match v10 exactly
N=$(python3 -m plasim_makani_packager.packager \
        --sims $SIMS --train-years $TRAIN_YEARS \
        --valid-years $VALID_YEARS --test-years $TEST_YEARS --count-tasks)
sbatch --array=0-$((N-1)) \
    --export=ALL,SIMS="$SIMS",\
POSTPROC_ROOT=/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/postproc,\
BOUNDARY_ROOT=/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/boundary_astro_v11,\
OUTPUT_ROOT=/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev_v11,\
POSTPROC_SOURCE_DIR=$HOME/projects/SFNO_Climate_Emulator/src/plasim_postprocessor,\
TRAIN_YEARS="$TRAIN_YEARS",VALID_YEARS="$VALID_YEARS",TEST_YEARS="$TEST_YEARS",\
SST_LAND_FILL_K=271.35 \
    src/plasim_makani_packager/submit.slurm
# NOTE: --packager-version belongs to metadata.py only; do not pass it here.

# (Wait for completion.)

# === Phase 2 — stats (train-years MUST be set explicitly; default is [3,100]).
#                --epsilon 1e-8 is REQUIRED: stats.py default is 1e-6 but pr_6h std
#                is ~2.8e-7 (known PlaSim quirk, stats.py:40-44 + v10 stats confirm). ===
python3 -m plasim_makani_packager.stats \
    --output-root /scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev_v11 \
    --train-years 12 111 \
    --epsilon 1e-8 \
    -v

# === Phase 3 — metadata + YAML render (year flags MUST be set; defaults are [3,100] / [101,120]) ===
python3 -m plasim_makani_packager.metadata \
    --output-root /scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev_v11 \
    --exp-dir /scratch/11114/zhixingliu/SFNO_Climate_Emulator/runs/sim52_astro_64x128_zgplev_v11 \
    --rsdt-method astronomical \
    --sst-land-fill-k 271.35 \
    --sst-mode surface \
    --source-boundary-root /scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/boundary_astro_v11/sim52 \
    --packager-version sim52_astro_64x128_zgplev_v11_surface_sst \
    --train-years 12 111 \
    --valid-years 11 11 \
    --test-years 121 128 \
    -v
# metadata.py's new self-consistency check (§1.4) asserts a sample H5's
# attrs.sst_mode matches --sst-mode and rsdt_method matches --rsdt-method.

# === Phase 4a — explicit-mode validation (avoid the deprecated `structural` alias,
#                which maps to `full` and runs smoke-live; we run smoke-live separately
#                only if/when we want the full-pipeline check). --epsilon 1e-8 on the
#                stats sub-mode for the same reason as Phase 2. ===
python3 -m plasim_makani_packager.validate \
    --output-root /scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev_v11 \
    --mode files -v
python3 -m plasim_makani_packager.validate \
    --output-root /scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev_v11 \
    --mode stats --epsilon 1e-8 -v
# Optional once the production root is fully populated (train + valid + test all non-empty):
# python3 -m plasim_makani_packager.validate --output-root .../sim52_astro_64x128_zgplev_v11 --mode full --epsilon 1e-8 -v

# === Phase 5a — subset farm (train + valid; subset script DOES NOT handle test) ===
python3 scripts/build_subset_dataset.py \
    --src /scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev_v11 \
    --dst /scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full_v11 \
    --train-years 12-111 \
    --valid-years 11

# === Phase 5b — test_holdout symlinks (mirror v10; subset script does NOT populate test/) ===
DST_TH=/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/makani/sim52_zgplev_full_v11/test_holdout
SRC_TEST=/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/makani/sim52_astro_64x128_zgplev_v11/test
mkdir -p "$DST_TH"
set -euo pipefail
for Y in 121 122 123 124 125 126 127 128; do
    SRC="$SRC_TEST/MOST.$(printf %04d $Y).h5"
    if [[ ! -f "$SRC" ]]; then
        echo "ERROR: missing source $SRC — packager phase incomplete for year $Y" >&2
        exit 1
    fi
    ln -sf "$SRC" "$DST_TH/MOST.$(printf %04d $Y).h5"
done
# v10 leaves `test/` empty inside the subset farm; v11 does the same.

# === Validation ===
.venv/bin/python analysis_outputs/validate_v11_sst_handling_phaseC.py
```

---

## 6. Risks / edge cases

| Risk | Mitigation |
|---|---|
| Forgetting `--train-years 12 111` on `stats.py` / `metadata.py` → defaults to [3, 100] / [101, 120] → silently writes wrong stats and wrong metadata. | Example commands set them explicitly. Phase B5/B6/C4 catch wrong stats/metadata. |
| Forgetting `RSDT_METHOD=astronomical` at adaptor submit → boundary files emit arithmetic rsdt → packager rejects via `_assert_rsdt_method`. | Phase B1 catches this. The example sbatch commands set it explicitly. The env-preserving slurm patch lets the export actually take effect. |
| Forgetting `SST_MODE=surface` at adaptor submit → adaptor uses its own default (`surface`), still correct. But forgetting `--sst-mode surface` at `metadata.py` → top-level metadata defaults to `ocean_era5` → metadata cannot lie thanks to the rev-3 self-consistency check; the script will raise. | Defense-in-depth from §1.4 self-consistency check + Phase B6/B8. |
| Forgetting `--source-boundary-root` at metadata call → top-level metadata points at v10 boundary path (wrong provenance). | Phase B6 catches this. Example commands set it explicitly. |
| Stats shift: SST mean drops ~282→~278 K, std grows ~12→~20 K. Model trained on v10 stats would mis-normalize if mixed with v11 data. | Datasets and stats live in disjoint output dirs. Phase B5 + C4 assert v11 stats are in the expected v11 band. Training YAML for v11 is created separately as part of a follow-up plan. |
| `SST_MIN_K` widening (271.34 → 170 K) could mask real bugs in future PlaSim source data. | 170 K is 100 K above absolute zero and 30 K below PlaSim's observed minimum — looser than v10 but still a real corruption gate. |
| PlaSim's sub-freezing open-ocean cells (~3 % of cell-timesteps) were previously floored. Under `surface` mode they pass through (~270 K). | Deliberate: PlaSim's internal sic-flip lag is part of its behaviour and the emulator should learn it. Documented in `compute_sst` docstring. |
| Coastal cells with fractional `lsm`. | Not applicable for PlaSim sim52 (LSM is binary {0, 1} — verified). Phase A8 asserts this. |
| Subset script CLI syntax (`--train-years` is a string `"12-111"`, not space-separated). | Example command uses the correct form. Phase C6 verifies the symlink farm has the expected files. |
| Forgetting Phase 5b — `test_holdout/` not populated → training inference path breaks. | Phase C7 explicitly checks `test_holdout/` symlinks exist for years 121-128. |
| Accidentally overwriting v10. | All v11 output roots end in `_v11`. The example commands use disjoint paths. The packager and adaptor never auto-discover v10 paths — they only write where directed. |
| Existing `sim52_zgplev_full/` symlink farm could be repointed by mistake. | New farm directory is `sim52_zgplev_full_v11/`; v10 farm untouched. |
| Stale gates: `analysis_outputs/compare_forcing_pipelines.py:63` uses 271.35 as land-fill; old docs claim SST is `sea_surface_temperature` over ice. | Listed for documentation update in §8. None block training. |

---

## 7. What should NOT be changed

- Default `rsdt` window stays forward `[T, T+6h]`. No backward / `--rsdt-window` flag added in this plan.
- No changes to `compute_rsdt_arithmetic` / `compute_rsdt_astronomical`.
- No changes to `sic` clipping or representation.
- No changes to `stats.py`, `validate.py`, `build_subset_dataset.py`, `build_boundary_dir.py`, the packager `submit.slurm` (already env-preserving), or `scripts/package_sim52_astro.sh`.
- No changes to `lsm`, `sg`, `z0` handling.
- No changes to normalization scheme (z-score with global stats); stats are explicitly recomputed in Phase 2.
- No changes to existing v10 dataset, v10 stats, v10 run dirs, or existing training YAMLs.
- No changes to `data_loader_multifiles.py` (upstream code).
- No retraining launched as part of this plan.
- No changes to `DEFAULT_PACKAGER_VERSION` constant in `metadata.py` (use the existing CLI override).
- `sst_land_fill_k=271.35` stays — it now applies exclusively over land (the only NaN cells under `surface` mode), where the semantics are unchanged.

---

## 8. Stale-gate / documentation audit (informational; not blocking training)

Files mentioning the old freezing-clamp semantics that should be updated in a follow-up doc PR (out of scope here):

| File | Note |
|---|---|
| `docs/plasim_expansion_and_adaptor_plan.md` (multiple lines) | Documents the old freezing-clamp design. Add a note pointing at this plan and the new `sst_mode` flag. |
| `docs/emulator_adaptor_audit.md:56,62` | Same. |
| `docs/plasim_makani_packager_plan.md` (multiple lines) | 271.35 references are about land-fill (still valid) and stats notes (still valid except the description "land-filled 271.35 K" should clarify "land-only under v11"). |
| `skills/plasim-makani-packager/SKILL.md:46,73,94,105` | Update channel description: `sst -- varying, surface temperature over non-land` (was: "from boundary adaptor (NaN-over-land filled with 271.35 K)"). |
| `analysis_outputs/compare_forcing_pipelines.py:63` | Old comparison script. Land-fill use of 271.35 is still correct semantically; no functional change needed. Leave a comment if revising. |
| `docs/2026-05-10_forcing_pipeline_numerical_diff.md` and `docs/2026-05-10_sfno_param_count_forensic.md` | Forensic write-ups that motivated this plan. Already consistent with the new direction. |

Code references that are **still correct** under v11 and need no change:
- `src/emulator_adaptor/adaptor.py::FREEZING_SEAWATER_K = 271.35` (still used in `ocean_era5` mode and as land-fill cross-reference).
- `src/plasim_makani_packager/packager.py:181` `sst_land_fill_k=271.35` (now applies over land only, semantically unchanged).

---

## 9. Verification (end-to-end summary)

Plan is complete when:

1. **Phase A** unit tests all pass (in-process, no SLURM).
2. **Phase B** smoke regeneration on year 11 produces a v11 packaged dataset with cold-ice SST signal, finite forcing, correct provenance attrs at all three layers (adaptor NetCDF, per-H5 attrs, top-level metadata), `metadata.py` self-consistency check passes, and stats in expected v11 ranges.
3. **Phase A6** regression: `ocean_era5`-mode adaptor SST values are bit-for-bit equal to v10 (attrs differ, expected).
4. **Phase C** full-regeneration sweep: every adaptor NetCDF, every packaged H5, the symlink farm (train + valid + test_holdout), and the top-level metadata all carry correct provenance; aggregate stats match Phase B5 bands; loader probe returns NaN-free tensors.

Once all four are green, the v11 dataset is ready for a follow-up training-plan PR (out of scope here) that adds `plasim_sim52_zgplev_full_v11.yaml` pointing at the v11 farm.
