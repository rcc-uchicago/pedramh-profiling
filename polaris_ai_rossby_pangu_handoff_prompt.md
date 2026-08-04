# HANDOFF — PanguPlasim-on-E3SM in the ai-rossby recipe, on Polaris (PanguWeather parity)

You are picking up a bring-up. Goal: train a **PanguWeather-comparable** model — ai-rossby's
**`PanguPlasimLegacy`** on the E3SM archive, with the **exact PanguWeather variable set (108
fields)** — on **Polaris**, logging to **wandb**. This is the group's PanguWeather→PhysicsNeMo
consolidation (`awikner/physicsnemo @ ai-rossby`). Read this whole prompt first. Work on branch
`fix/tsoi-fill-270` in `/eagle/projects/lighthouse-uchicago/members/mehta5/pedramh-profiling`.

## ⚠ STEP 0 — BEFORE ANY CODING: write the variable-parity assertion doc

Write `ai_rossby_panguweather_variable_parity.md` (repo root) that **proves the variables are the
same** as the PanguWeather run jesswan already trained. Do not start any coding until it shows full
parity (or the user signs off on a flagged delta).

It must:
1. **Identify jesswan's actually-trained PanguWeather E3SM run** — the ground truth. Start from
   `PanguWeather/v2.0/config/E3SM_SFNO_H5_POLARIS.yaml` (and `_ALLDATA`, `_STAMPEDE_jsw`); confirm
   which config jesswan actually ran by checking run artifacts (wandb dirs, checkpoints, logs under
   jesswan's dirs / `PanguWeather/v2.0/`). Record the config path + evidence of the run.
2. **Extract that run's variable contract:** the groups (`upper_air_variables`, `surface_variables`,
   `diagnostic_variables`, `land_variables`, `constant_boundary_variables`,
   `varying_boundary_variables`, `ocean_variables`), the level set, and `mask_fill`.
3. **Extract the planned ai-rossby PanguPlasim contract:** the variable groups from the
   `conf/model/pangu_plasim_e3sm.yaml` this handoff creates, plus the fills from `conf/dataset/e3sm.yaml`.
4. **Assert, field-by-field, that they are IDENTICAL** — a table per group with an explicit PASS/FAIL
   on: variable names, group/role membership, level list, and fill values. Flag ANY difference.
   Expected contract (must match jesswan's):
   - upper_air `[T,U,V,Z3,RELHUM]` @ 18 hybrid levels (clouds CLDLIQ/CLDICE/CLOUD excluded)
   - surface `[TREFHT,U10,RHREFHT,PS,PSL,TMQ]`; diagnostic `[FSNTOA,FSNT,PRECT]`
   - land `[SOILWATER_10CM,TSOI_10CM]`; ocean `[]`
   - constant_boundary `[PCT_GLACIER,PFTDATA_MASK,PCT_NATVEG,TOPO]`; varying `[SST,ICE,sol_in]`
   - fills (parity): `SST=270, TSOI_10CM=270`, everything else `0`
5. **Explicitly separate variable-parity from architecture:** jesswan's E3SM runs were **SFNO**
   (`nettype: sfno_plasim`; PanguWeather has no Pangu-on-E3SM config — its PANGU_* configs are all
   PLASIM). We are running the **Pangu** architecture on the **same E3SM variables**. State clearly:
   the *variable set* is asserted identical; the *architecture* (Pangu vs the SFNO jesswan ran on
   E3SM) is the intended difference. Also note the non-variable mechanics that do NOT change which
   variables exist: `sol_in` solar-name patch, land folded into the store's `surface` group.
6. **Conclude** with a one-line verdict: "ai-rossby PanguPlasim trains the [identical / differs-by-X]
   variable set vs jesswan's PanguWeather E3SM run," and the config paths that back it.

## Locked decisions (already made with the user)

- Model **`PanguPlasimLegacy`** (deterministic, no VAE) — exact PanguWeather architecture parity.
- Variables = the PanguWeather groups above (108 fields; **SST/ICE prescribed** = varying_boundary).
- Normalization **parity-first**: reuse the shipped precomputed stats + reference fills
  (`SST=270, TSOI=270`, else 0); a masked-stats recompute + `SST=-1.8` is a documented **fast-follow**.
- Placement: **git subtree** at top-level `physicsnemo_ai_rossby/` on branch `fix/tsoi-fill-270`,
  from **jesswan's local copy**.
- Cluster **Polaris**; ai-rossby needs its **own uv venv** (cannot reuse `$SFNO_VENV`).

## Implementation steps (only after Step 0 passes)

1. **Vendor (git subtree).** `git subtree add --prefix=physicsnemo_ai_rossby <jesswan copy> ai-rossby`
   per `polaris_pbs_notes.md §6b` (unsquashed; `-c safe.directory=<path>` for the ownership guard;
   top-level, NOT nested in `physicsnemo_sfno/` — rule #5). Source =
   `/eagle/.../jesswan/physicsnemo_ai-rossby` (branch `ai-rossby`, HEAD `87002adb`).
2. **Polaris venv (cheap, wheels-only).** `polaris_setup_ai_rossby_venv.sh`: `module load conda` →
   `uv sync --extra cu129 --extra sfno-extras --extra utils-extras` in `physicsnemo_ai_rossby/`
   (`uv` ships with the module; login node has direct outbound; no source builds; torch_harmonics +
   wandb are in those extras). Verify: `import physicsnemo.experimental.datapipes.plasim,
   physicsnemo.experimental.models.pangu_plasim` + `torch.cuda.is_available()`.
3. **Code edits (vendored tree).** (a) **BLOCKER:** add `"sol_in"` to `_solar_names` in
   `pangu_plasim_legacy.py:262` + `pangu_plasim.py:364` (Pangu else `ValueError`s — E3SM's solar
   name isn't `rsdt`). (b) `train.py:636,739`: append `land+ocean` to the surface name lists so the
   split `land_variables` fill/loss don't broadcast-mismatch (parity option; alt = fold into
   `surface_variables`). (c) `tools/data/e3sm/pangu_h5_to_zarr.py` `PANGU_E3SM_CHANNELS`: the
   108-field groups with **land folded last into `surface_variables`** in store order; fix stale docstring.
4. **Parity normalization zarr.** `build_normalization_zarr.py` from
   `mehta5/pangu_polaris_data/data_2015-2050_{mean,std_corr}.nc` (use `std_corr`, fix the `Z`/`Z_2`
   coord) → ai-rossby schema under `AI_ROSSBY_DATA`. Minutes; the store already has every needed var.
5. **Convert the E3SM store** from `$E3SM_ROOT/h5/plev_data` (NOT sigma_data) → per-year
   ai-rossby-schema zarr, folded 108-field surface order, raw NaN preserved. **Layout:**
   `e3sm/train/<year>.zarr` + val + norm store in *separate* subdirs (the multi-year loader globs
   `*.zarr` — co-locating the norm store crashes it). Start 1 train + 1 val year; PBS compute job.
6. **Configs.** `conf/model/pangu_plasim_e3sm.yaml` (base on `pangu_plasim_s2s.yaml`, already
   180×360: `levels`=18 E3SM hPa, `horizontal_resolution:[180,360]`, `window_size:[2,6,10]`,
   `vertical_windowing:True`, `embed_dim:240`, `checkpointing:3`, keep `patch_size/depths/num_heads`;
   `name:PanguPlasimLegacy`; the locked groups). `conf/dataset/e3sm.yaml`: `AI_ROSSBY_DATA` paths,
   `nan_fill_values:{SST:270.0, TSOI_10CM:270.0}`, `nan_fill_default:0.0`,
   `normalize_constant_boundary/diagnostic:True`.
7. **Polaris launcher + wandb.** `physicsnemo_ai_rossby/polaris/polaris_pangu_plasim.pbs`: our PBS
   header (`-A lighthouse-uchicago -q preemptable -l select=1:system=polaris -l filesystems=home:eagle
   -r y`), `source ../polaris_env.sh`, activate the ai-rossby venv, `export AI_ROSSBY_DATA=...`,
   `python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$NPROC train.py
   model=pangu_plasim_e3sm dataset=e3sm training=pangu_plasim_legacy loss=mae training.amp=bf16
   dataset.batch_size=1 hydra.run.dir=<stable eagle run dir> wandb.mode=offline
   wandb.project=pedramh-profiling wandb.name=<run>`. NEVER bare `torchrun`. Offline + `wandb sync`
   from login; `wandb.mode=online` is a one-flag flip once the key is set (`polaris_setup_wandb.sh`).

## Key constraints / gauntlet findings (adversarial + cold reviewed)

- **Channel order is a SILENT failure** — tensors stack in store-attrs order; fills/loss build from
  model-config lists; nothing cross-checks (`torch.cat` can't catch it). Since we build the store,
  write store attrs and model lists in the **identical** order, and add a **preflight assertion**
  `store.attrs[group]==cfg.model[group]` per group.
- **Launch/OOM:** must pass `training=pangu_plasim_legacy loss=mae training.amp=bf16
  dataset.batch_size=1` (defaults compose the wrong curriculum and OOM 40 GB A100s). Pangu
  `checkpointing` only gates recovery heads; bf16 + batch_size are the real levers.
- **DDP:** torch pinned `<2.11` (real regression, `train.py:884-891`); wandb inits on all ranks.
- **Normalization is fill-baked** (SST→270 in °C, TSOI→0) — parity reuses it; the corrected
  masked-stats path is a NEW tool (`build_normalization_zarr.py` only converts, computes nothing).
- Archive is **2015–2049** (35 yr). Full store ≈1.43 TB / ~600k inodes — check quota (16/50 TB now).

## Verification

- Step-0 doc shows full variable parity (PASS on every group).
- Venv imports + CUDA true. Store attrs = 108-field groups in exact model order (land last in
  surface); a few samples bitwise vs `h5/plev_data`; raw NaN preserved; 18 levels match norm store.
- Preflight `store.attrs[group]==cfg.model[group]` passes.
- Smoke (PBS, 1 train + 1 val year): constructor builds (sol_in cleared); no fill/loss broadcast
  error; finite decreasing loss; zero NaN reaches the model (strict fill check); wandb offline run +
  `wandb sync`. PASS = advancing `train/*` epoch metrics.

## Out of scope / untouched

- Corrected normalization (masked stats + `SST=-1.8`) — fast-follow after parity trains.
- Full 35-year conversion + production run; embed_dim tuning.
- Our `physicsnemo_sfno/` SFNO path, the running v2 conversion `7324098` (fallback), jesswan's checkout.
