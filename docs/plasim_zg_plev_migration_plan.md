# PlaSim → Makani packager — zg sigma → zg_plev migration plan (v7)

**Status:** Draft v7 — addresses Codex review of v6 (1 high + 2 low cleanup). Awaiting re-review.
**Parent contract:** `docs/plasim_makani_packager_plan.md` (v9). This document is a *diff-only* plan against v9; v9 remains the authoritative source for everything not explicitly changed here.

**v6 → v7 changelog:**
- §5 P5 third bullet (the full-subset smoke-live) now **renders the trainer-side `plasim_sim52_zgplev_full.yaml` template first** (via the same `sed` substitution `submit_full.slurm:60-62` uses) before passing the *rendered* file to `--yaml-config`. The trainer-side templates contain `{{OUTPUT_ROOT}}` / `{{EXP_DIR}}` placeholders (verified across all 5 v9 yamls); v6 passed the unrendered template directly, which would have made the loader try to open literal `{{OUTPUT_ROOT}}/train` paths (Codex v6 high finding).
- §3.7 `run_smoke_live(...)` gains a placeholder-check helper `_assert_no_yaml_placeholders(yaml_path)` (mirrors `scripts/preflight.py:311-326`) that fails fast on any `{{...}}` left in the loaded yaml. Belt-and-braces: the operator is supposed to render before passing, but if they pass an unrendered template by mistake, smoke-live fails with a clear message instead of with a `FileNotFoundError` deep inside the dataloader (Codex v6 high finding).
- §3.7 `run_smoke_live` docstring corrected: v5 said "P4 renders all 5 zgplev YAMLs" but v6 reverted P4 to baseline-only. Updated to "P4 renders the baseline yaml only; the other four are hand-curated trainer-side templates that must be rendered before being passed to `--yaml-config`" (Codex v6 low cleanup #1).
- §3.13 explicit base-script mapping: `submit_zgplev_baseline.slurm` is based on `submit_train.slurm` (the closest v9 sibling — there is no `submit_baseline.slurm` in the v9 set). The other four v10 submits are based on their like-named v9 siblings (Codex v6 low cleanup #2).

**v5 → v6 changelog:**
- §5 P4 reverted to a **single** `metadata.py` invocation (baseline only). v5 wrongly claimed 5 invocations would produce 5 semantically distinct configs, but `metadata.py` reads one packager-side template (`plasim_64x128_zgplev.yaml`), substitutes `{{OUTPUT_ROOT}}` / `{{EXP_DIR}}`, and only renames the top-level key — so 5 invocations would produce 5 copies of the *same baseline* content with different top-level config-name keys. The 5 semantically-distinct configs already live at `src/sfno_training/config/plasim_sim52_zgplev_*.yaml` (see §3.4) — those are hand-curated for tiny / short / full / smoke / baseline, with the right `train_data_path` / `n_train_samples_per_epoch` / etc. for each (Codex v5 finding).
- §5 P5 second smoke-live now uses `--yaml-config $REPO_ROOT/src/sfno_training/config/plasim_sim52_zgplev_full.yaml` (the trainer-side full-subset config, not the packager-rendered baseline). This was exactly the use-case the §3.7 `--yaml-config` override was designed for; v5 left the override flag unused at the call site and would have re-tested the baseline yaml against the full-subset root.
- Minor cleanup: §5 P0b's "edits in §3.1–§3.12" updated to "§3.1–§3.13" (§3.13 was added in v5); the §3.1 code-comment string updated from "v4" to current version.

**v4 → v5 changelog:**
- §5 P4 now renders **all five** zgplev YAMLs (smoke / tiny / short / baseline / full) into the parent root's `config/`. v4 only rendered baseline; the subset's symlinked `config/` then exposes whatever the parent has, so this fix unblocks the second smoke-live invocation against `_zgplev_full` (Codex v4 finding #1).
- §3.7 `run_smoke_live(...)` gains optional `--yaml-config PATH` and `--config-name NAME` to decouple from the `{output_root}/config/` convention. Mirrors `scripts/preflight.py`. Belt-and-braces: P4's expanded rendering is the primary fix; the override flags are an escape hatch (Codex v4 finding #1, supplementary).
- §3.13 added: 5 new `src/sfno_training/submit_zgplev_*.slurm` scripts mirroring the existing v9 submits, each pinning its v10 yaml + config-name. v9 submits stay untouched, consistent with L8 (Codex v4 finding #2).
- Stale-text cleanup: "four modes" → "five modes" at §5 preamble; "~4 hourly steps/year" → "1460 6-hourly samples/year (4 per day × 365)" in the disk-space note (Codex v4 finding #3).

**v3 → v4 changelog:**
- §3.10 corrected: `score_nwp.py` derives `channel_names` from the first inference NetCDF's `channel` coord (which `src/sfno_inference/nc_writer.py:131` writes on every file), not from `metadata.json` or the eval-summary CSV. The eval `OUT_ROOT` layout is `inference/`, `baselines/`, `scores/`, with no `metadata/` subdir, and `score_nwp.py` runs **before** the summary CSV exists, so the v3 fallback chain was unrunnable. `--metadata-json` becomes an optional override flag (Codex v3 finding #1).
- §5 P5: adds a **second** `smoke-live` invocation against `…/sim52_zgplev_full` with `plasim_sim52_zgplev_full.yaml` to actually exercise the production training root (subset + full yaml), not just the packager root with the baseline yaml (Codex v3 finding #2).
- §3.9: fixture helper now constructs synthetic `zg_plev` so `zg500` (lev_2 = 500 hPa, source-array index 7) has mean ~5550 m, in the [5400, 5700] m audit band. Random fixtures would otherwise trip the new `_audit_zg500_inline` audit gate inside any test that calls `compute_stats()` (Codex v3 finding #3).
- Stale-text cleanup: the "must match the old slice" sentence in §3.1, the §3.7 CLI list missing `smoke-live`, the legacy "subsets added to P3" line at the §5 preamble, and the document footer (was "v1") all corrected (Codex v3 finding #4).

**v2 → v3 changelog:**
- §5 reordered: subset-build moves to **after** metadata, since `build_subset_dataset.py` requires `stats/`, `metadata/`, and `config/` to exist in the source root (Codex v2 finding #1).
- §3.5 reversed defaults: `metadata.py --variant` now defaults to `zgplev`; the `astro64x128` variant raises a clear "v9 codebase is frozen at tag plasim-makani-packager-v9-final" error rather than silently producing v9-named output with v10 channel names (Codex v2 finding #2).
- §3.9 expanded: adds explicit edits to `tests/plasim_makani_packager/test_multifile_loader_smoke.py:88,103,109,115-116` and `tests/sfno_training/helpers.py:82-83,189,204,210` (both files hard-code `plasim_64x128.yaml` / `plasim_sim52_astro_64x128`); previous "no code change" claim was wrong (Codex v2 finding #3).
- §2 + §3.9: stale "slice contiguously" / "wrong lev_2 ordering must raise" language removed; the new `test_zg_plev_value_lookup.py` correctly pins missing-level → raise and reordered-but-complete-lev_2 → pass (Codex v2 finding #4).
- §3.7 + §5 P5: a new `validate --mode smoke-live` runs a real-data preflight against the actual `--output-root` (loads the rendered yaml + metadata.json, instantiates the patched Makani loader, runs a 3-step rollout). The original `--mode smoke` is kept as a wrapper-regression alias for the synthetic-fixture pytest. P5 is the gate for `smoke-live`; `smoke` is for CI (Codex v2 finding #5).
- §3.10 + §3.11: `render_eval_report.py` and `score_nwp.py` now take an explicit `--metadata-json` CLI arg (defaults to `{out_root}/metadata/data.json`) for the channel-name source; with concrete fallback to deriving channel names from the eval-summary CSV column headers (Codex v2 finding #6).

**v1 → v2 changelog (preserved for context):**
- L8 added: codebase becomes v10-only after this PR; v9 freeze is enforced by a git tag, not by variant-aware code.
- §3.2 rewritten to select pressure levels by lev_2 *value*, not by hard-coded slice.
- §3.6 audit moved inline into `compute_stats()` before the .npy save and uses `ValueError`, not `KeyError`.
- §3.7 splits `validate.py` into `--mode files` / `stats` / `smoke` / `smoke-live` / `full` so per-file checks can run before stats exist.
- §3.10 made channel-adaptive (auto-detect `zg500` vs legacy `zg5`) so v9 A/B scoring still works.
- §3.12 added: `scripts/build_subset_dataset.py` coordination + new `sim52_zgplev_full` subset.
- §5 rollout uses the new validate modes in the right order; §6 gates updated to match.
**Scope:** Replace sigma-level `zg1..zg10` in the 52-channel state with pressure-level `zg150, zg200, zg250, zg300, zg400, zg500, zg600, zg700, zg850, zg925`, keeping the same 58-input / 53-output model contract and packaging a parallel `…_zgplev` dataset alongside the existing `sim52_astro_64x128` dataset. Bundles the lockstep eval-script and eval-plan updates that depend on the channel rename.
**Scientific motivation:** The v9 contract's `zg5` is sigma-level zg, not Z500. Primary downstream goal is blocking / Z500 skill, with secondary goals (`tas`, `pr_6h`) preserved. Making `zg500` a literal pressure-level channel — and making 500 hPa a true autoregressive state variable — is the change.

---

## 1. Locked decisions

These are answered, not open. Each item has a one-line *why* so future readers do not re-litigate.

| # | Decision | Why |
|---|---|---|
| L1 | Pressure subset: `[150, 200, 250, 300, 400, 500, 600, 700, 850, 925]` hPa (10 levels). | Drops 50/100 (redundant given PlaSim top sigma ≈ 38 hPa + sigma channels above 200 hPa) and 1000 (below-ground extrapolation over orography is an unaudited noise source); drops 250-spaced gap fillers (250/300 kept for jet sampling). Includes 925 (boundary-layer top) for tas/pr_6h coupling, includes 500 explicitly for blocking. |
| L2 | Channel naming: `zg{P}` where P is the integer hPa value. State channels [42:52] become `zg150, zg200, zg250, zg300, zg400, zg500, zg600, zg700, zg850, zg925`. Order: TOA → surface (150 → 925). | Self-documenting; eliminates the `zg5 ≠ Z500` foot-gun the v9 contract carried into the eval scripts. Order matches v9's sigma convention (lev[0] = TOA). |
| L3 | Sigma `zg` is ignored by the packager. Postprocessor is **not** modified; sigma `zg` continues to live in the postproc NetCDF. | Audit snapshot stays valid; no need to re-run postprocessor on 100+ years; preserves diagnostic optionality. |
| L4 | This plan is a new sibling doc, not an in-place revision of v9. | Keeps v9 frozen as the historical contract for the existing `sim52_astro_64x128` dataset and any checkpoints trained against it. |
| L5 | Output dataset is a parallel directory: `$SCRATCH/AI-RES/data/makani/sim52_astro_64x128_zgplev`, with paired `$SCRATCH/AI-RES/runs/sim52_astro_64x128_zgplev`. The v9 dataset and runs are untouched. | Both contracts coexist for A/B comparison; no destructive overwrite of existing checkpoints. |
| L6 | Eval-side updates (scripts/score_nwp.py, scripts/render_eval_report.py, docs/sfno_eval_plan.md) ship in the same plan / PR. | The `zg5 → zg500` rename is mechanical and the gate threshold meaning changes (it now refers to literal Z500), so coupling them avoids a transient broken-eval window. |
| L7 | Hard constraints: (a) `stats.py` must rerun on the new HDF5 dir before any retrain; (b) Phase 4b Makani smoke must pass before any retrain; (c) the postprocessor's git SHA must be pinned into each new HDF5 file's `file_attrs` (in addition to the existing packager SHA); (d) audit gate — `zg500` global mean over the training split must lie in [5400, 5700] m, matching the audit-snapshot finding. | (a, b) standard v9 gates re-applied to the new contract; (c, d) new gates specific to the contract change. |
| L8 | After this PR merges, the **codebase is v10-only**. The packager / stats / validate / metadata modules read `STATE_CHANNELS` as a single global list and that list becomes the v10 list. The v9 dataset on disk (`sim52_astro_64x128`) keeps working for any *trainer-side* run, because `PlasimTrainer` reads channel names from its yaml, not from `channels.py`. But v9 cannot be **regenerated** from this codebase: re-running `packager.py` / `stats.py` / `metadata.py` / `validate.py` against v9 inputs would now produce v10-named outputs that don't match the v9 H5 files on disk. v9 regeneration requires `git checkout plasim-makani-packager-v9-final` (a tag this PR creates pointing at the last v9-supporting commit). | Variant-aware code (a `get_state_channels(variant)` indirection threaded through every consumer) was considered and rejected as too much surface area for a one-shot contract change. The freeze tag is sufficient because v9 artifacts on disk are immutable and don't need regeneration in normal operation. |

**Out of scope (explicitly):**
- Changing `n_state_channels` away from 52 (locked by `n_state_channels: 52` in 5 yaml configs and by `PlasimPreprocessor` / `PlasimTrainer` hard asserts; this plan keeps it 52, so trainer code is untouched).
- Changing forcing channels (lsm/sg/z0/sst/rsdt/sic) or diagnostic (pr_6h).
- Changing the postprocessor or boundary adaptor.
- Inference-side patches (`src/sfno_inference/`) beyond reading the new channel names from metadata — `nc_writer.py:67` already takes `channel_names` as a parameter, so it is contract-data-driven and needs no code change.

---

## 2. Channel-list diff

**v9 (frozen):**
```
STATE_CHANNELS[0..1]   = ["pl", "tas"]
STATE_CHANNELS[2..11]  = ["ta1", ..., "ta10"]
STATE_CHANNELS[12..21] = ["ua1", ..., "ua10"]
STATE_CHANNELS[22..31] = ["va1", ..., "va10"]
STATE_CHANNELS[32..41] = ["hus1", ..., "hus10"]
STATE_CHANNELS[42..51] = ["zg1", ..., "zg10"]            # SIGMA-LEVEL
```

**v10 (this plan):**
```
STATE_CHANNELS[0..1]   = ["pl", "tas"]
STATE_CHANNELS[2..11]  = ["ta1", ..., "ta10"]
STATE_CHANNELS[12..21] = ["ua1", ..., "ua10"]
STATE_CHANNELS[22..31] = ["va1", ..., "va10"]
STATE_CHANNELS[32..41] = ["hus1", ..., "hus10"]
STATE_CHANNELS[42..51] = ["zg150", "zg200", "zg250", "zg300",
                         "zg400", "zg500", "zg600", "zg700",
                         "zg850", "zg925"]                # PRESSURE-LEVEL
```

`DIAGNOSTIC_CHANNELS = ["pr_6h"]`, `FORCING_CHANNELS = ["lsm", "sg", "z0", "sst", "rsdt", "sic"]` — both unchanged.

`zg500` is at `STATE_CHANNELS[47]` (was `zg5` at `STATE_CHANNELS[46]` under v9 — note the index shift even though both happen to land near the middle of the sigma block; eval-script consumers must update by name not index).

**Index map for source-NetCDF `zg_plev[lev_2]` → packed state:**
The postprocessor's current `lev_2` coordinate is `[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]` (verified by `docs/audit_snapshots/manifest.txt:41`). The 10 levels we want are looked up by *value* against `lev_2`, not by a fixed slice — see §3.2. So a future postprocessor that reorders `lev_2` or adds new levels stays compatible as long as every value in `ZG_PLEV_HPA` appears somewhere in `lev_2`. Today's `lev_2` happens to give a contiguous index span [2..11]; that's an observation, not a contract.

---

## 3. Code changes (file-by-file)

### 3.1 `src/plasim_makani_packager/channels.py`

Replace the existing `_SIGMA_VARS` / `_SIGMA_LEVELS` block + state list with an explicit two-block construction. The 4 sigma vars (ta/ua/va/hus) keep the existing `_sigma_names` helper; zg gets a new explicit list.

```python
# Locked by docs/plasim_zg_plev_migration_plan.md (v7).
_SIGMA_VARS: tuple[str, ...] = ("ta", "ua", "va", "hus")  # zg removed
_SIGMA_LEVELS: int = 10
_ZG_PLEV_HPA: tuple[int, ...] = (
    150, 200, 250, 300, 400, 500, 600, 700, 850, 925,
)  # TOA → surface. Each value must appear in postproc lev_2; positions are
   # resolved by value lookup (see §3.2), not by index slice.


def _sigma_names(var: str) -> list[str]:
    return [f"{var}{i}" for i in range(1, _SIGMA_LEVELS + 1)]


def _zg_plev_names() -> list[str]:
    return [f"zg{p}" for p in _ZG_PLEV_HPA]


STATE_CHANNELS: list[str] = (
    ["pl", "tas"]
    + _sigma_names("ta")
    + _sigma_names("ua")
    + _sigma_names("va")
    + _sigma_names("hus")
    + _zg_plev_names()
)
assert len(STATE_CHANNELS) == 52
```

Export `_ZG_PLEV_HPA` (rename to module-public `ZG_PLEV_HPA`) so packager.py and tests can import the locked tuple instead of duplicating it.

### 3.2 `src/plasim_makani_packager/packager.py`

`_stack_fields_state` (lines 78–98) currently iterates `("ta", "ua", "va", "hus", "zg")` and assumes each is `(T, 10, H, W)` from `ds[var]`. Split the loop into two blocks:

1. The 4 sigma vars (unchanged, still read `ds[var]` shape `(T, 10, H, W)` from `lev`).
2. zg: read `ds["zg_plev"]` and **select by lev_2 value**, not by index slice. The locked tuple `ZG_PLEV_HPA` (exported from `channels.py` per §3.1) is the single source of truth; the packager looks each value up in the source `lev_2` coordinate. If a future change to the level subset reorders or replaces values in `ZG_PLEV_HPA`, no other code needs updating.

```python
from plasim_makani_packager.channels import ZG_PLEV_HPA  # tuple of 10 ints, TOA→surface


def _stack_fields_state(ds: xr.Dataset) -> np.ndarray:
    T = ds.sizes["time"]; H = ds.sizes["lat"]; W = ds.sizes["lon"]
    out = np.empty((T, 52, H, W), dtype=np.float32)
    out[:, 0] = ds["pl"].values
    out[:, 1] = ds["tas"].values
    col = 2
    for var in ("ta", "ua", "va", "hus"):
        arr = ds[var].values
        if arr.shape != (T, 10, H, W):
            raise RuntimeError(f"{var} shape {arr.shape}, expected {(T, 10, H, W)}")
        out[:, col : col + 10] = arr
        col += 10

    zgp = ds["zg_plev"]
    lev2 = zgp.coords["lev_2"].values.astype(int).tolist()
    # Look up each requested hPa value in lev_2; fail loudly if missing.
    try:
        idx = [lev2.index(int(p)) for p in ZG_PLEV_HPA]
    except ValueError as e:
        raise RuntimeError(
            f"zg_plev lev_2 = {lev2} does not contain all of "
            f"ZG_PLEV_HPA = {list(ZG_PLEV_HPA)} ({e})"
        ) from e
    zgp_arr = zgp.values
    if zgp_arr.shape[0] != T or zgp_arr.shape[2:] != (H, W):
        raise RuntimeError(
            f"zg_plev shape {zgp_arr.shape}, expected (T={T}, *, H={H}, W={W})"
        )
    out[:, col : col + 10] = zgp_arr[:, idx]
    col += 10
    assert col == 52
    return out
```

This is robust to (a) future expansions of `lev_2` (postprocessor adds a new pressure level), (b) reordering, and (c) future changes to `ZG_PLEV_HPA`. The only failure mode is "requested hPa value not present in source," which raises with a clear diagnostic.

Add new `file_attrs` entries (constraints L7c + observability for §3.5 metadata cross-check):

- `"postprocessor_git_sha"` — resolved per the option below.
- `"zg_source_var"` = `"zg_plev"` — locks the source-variable name.
- `"zg_pressure_levels_hpa"` = `np.array(ZG_PLEV_HPA, dtype=np.int32)` — locks the level subset directly into each H5 file. Lets a downstream consumer reconstruct the contract from the file alone.

Resolution path for `postprocessor_git_sha`:

- Each postproc NetCDF is produced by `src/plasim_postprocessor/plasim_postprocessor.py`. The current postprocessor does **not** write its own git SHA into the output NetCDF. Two options:
  - **Option A (preferred):** add a `--postprocessor-git-sha` CLI flag to packager.py; the SLURM submit script populates it from `git -C <postproc-source-dir> rev-parse HEAD` once at job start. Falls back to `"unknown"` if the source dir is not a git checkout.
  - **Option B:** patch the postprocessor to write `postprocessor_git_sha` into `most_ds.attrs` and have the packager copy it through. Cleaner but touches the postprocessor (out of scope L3).
- This plan picks **Option A** — keeps the postprocessor untouched per L3, and the packager already resolves its own SHA via subprocess (`packager.py:290–299`); a parallel helper for the postprocessor source is symmetric.

### 3.3 `src/plasim_makani_packager/templates/plasim_64x128_zgplev.yaml` (NEW)

**Do NOT edit `templates/plasim_64x128.yaml` in place** (parallel to §3.4). Codex finding #1 applies here too: the v9 template is consumed by `metadata.py:196` to render config files for v9 dataset packaging; mutating it silently changes what every fresh v9 metadata-render produces. Per L8 we accept that running `metadata.py` against v9 inputs is not supported on the v10 codebase, but the *template file itself* should still document the v9 contract for the freeze-tag history.

Create `src/plasim_makani_packager/templates/plasim_64x128_zgplev.yaml` as a copy of the v9 template with:

- top-level config-name `plasim_sim52_astro_64x128_zgplev:`,
- `channel_names` zg block updated to `zg150..zg925`,
- `forcing_channel_names` unchanged,
- header comment block referencing this plan (v10) and noting the v9 template is preserved at `plasim_64x128.yaml` for historical reference.

`metadata.py` (§3.5) picks the template based on the new `--variant` flag.

### 3.4 `src/sfno_training/config/plasim_sim52_zgplev_*.yaml` (5 NEW files)

**Do NOT edit the existing `plasim_sim52_{smoke,tiny,short,baseline,full}.yaml` files** (Codex finding #1, partial). They remain the source of truth for the v9 contract: a v9 trainer run reads `channel_names` from these yamls and matches them against the v9 H5 files on disk; both stay synchronized. Mutating them in place would silently break v9 retrains.

Instead, create **5 new sibling files**:

- `src/sfno_training/config/plasim_sim52_zgplev_smoke.yaml`
- `src/sfno_training/config/plasim_sim52_zgplev_tiny.yaml`
- `src/sfno_training/config/plasim_sim52_zgplev_short.yaml`
- `src/sfno_training/config/plasim_sim52_zgplev_baseline.yaml`
- `src/sfno_training/config/plasim_sim52_zgplev_full.yaml`

Each is a copy of its v9 counterpart with these changes:

- top-level config name `plasim_sim52_zgplev_*` (was `plasim_sim52_*`),
- `channel_names`: zg block changed to `"zg150","zg200","zg250","zg300","zg400","zg500","zg600","zg700","zg850","zg925"`,
- `train_data_path` / `valid_data_path` / `inf_data_path` redirected per §3.12 (subset roots for tiny/short/full, packager root for smoke/baseline),
- header comment block updated to point at this plan and call out the v10 contract.

`n_state_channels: 52`, `n_diagnostic_channels: 1`, `n_forcing_channels: 6`, and the trainer-side architecture/training keys all stay byte-identical to their v9 counterparts.

### 3.5 `src/plasim_makani_packager/metadata.py`

Per L8, the codebase is v10-only post-merge. To make this concrete in `metadata.py`:

- `DEFAULT_DATASET_NAME` becomes `"plasim-sim52-astro-64x128-zgplev"`, `DEFAULT_CONFIG_NAME` becomes `"plasim_sim52_astro_64x128_zgplev"`, `DEFAULT_TEMPLATE_NAME` becomes `"plasim_64x128_zgplev.yaml"`.
- New CLI flag `--variant {zgplev,astro64x128}` with **`zgplev` as default**. Selecting `astro64x128` raises a hard error pointing at the v9 freeze tag:

```python
if args.variant == "astro64x128":
    sys.exit(
        "error: the v9 (astro64x128, sigma-zg) variant is frozen. The v10 "
        "codebase produces v10-named metadata only. To regenerate v9 metadata, "
        "check out git tag plasim-makani-packager-v9-final and run that codebase."
    )
```

Setting v9 as default would let a stale CLI invocation silently emit v9-named `data.json` while reading v10-named channels from the H5 (which is exactly the divergence Codex v2 finding #2 flagged). Hard-erroring is the safe default.

`build_metadata.attrs.description` reads `"PlaSim sim52 postproc 64x128, astronomical rsdt, three-dataset layout for patched Makani — pressure-level zg (zg150..zg925) variant"`.

### 3.6 `src/plasim_makani_packager/stats.py`

No structural change to shapes — the `(1, 53, ...)` outputs and the per-channel `MIN_STD_EPSILON` hard-fail (`stats.py:170-183`) are unchanged because the channel count is unchanged.

Add the L7d audit **inline** inside `compute_stats()` immediately after Welford reduction finishes and **before** any `.npy` is written, using the in-memory `mean_tgt` array — not by re-loading from disk. A failed audit must abort the stats output rather than produce tainted normalization files. Use `ValueError` for the missing-channel case (which is what `list.index` raises).

```python
ZG500_AUDIT_RANGE_M: tuple[float, float] = (5400.0, 5700.0)
ZG500_CHANNEL_NAME: str = "zg500"


def _audit_zg500_inline(mean_tgt: np.ndarray) -> None:
    """Hard-fail before any stats .npy is written if zg500 mean is implausible.

    `mean_tgt` is the (53,) Welford-reduced per-channel mean in float64 (target =
    state ‖ diagnostic). The slot for zg500 is found by name lookup in
    TARGET_CHANNELS, so a future reordering of the channel list does not silently
    audit the wrong slot.
    """
    try:
        idx = TARGET_CHANNELS.index(ZG500_CHANNEL_NAME)
    except ValueError as e:
        raise RuntimeError(
            f"L7d audit: '{ZG500_CHANNEL_NAME}' not found in TARGET_CHANNELS "
            f"(this is a contract-rotation bug, not a data bug): {e}"
        ) from e
    val = float(mean_tgt[idx])
    lo, hi = ZG500_AUDIT_RANGE_M
    if not (lo <= val <= hi):
        raise RuntimeError(
            f"L7d audit FAIL: zg500 global mean {val:.1f} m outside "
            f"[{lo}, {hi}] m (audit-snapshot range from "
            f"docs/plasim_postprocessor_audit.md:193). "
            f"Aborting before stats .npy write."
        )
```

Call site (insert immediately before the existing `_save("global_means.npy", ...)` block at `stats.py:191`):

```python
_audit_zg500_inline(mean_tgt)  # L7d hard-gate
# ...existing _save() calls follow...
```

Constraint L7a (rerun on new output root) is procedural and lives in §5 P3, not in code.

### 3.7 `src/plasim_makani_packager/validate.py`

**Bug surfaced by Codex review (v1 finding #2):** `run_structural` at `validate.py:342` calls `_validate_stats(...)` unconditionally. So calling validate before stats exist (which the v1 rollout did at P1/P2, before P3 ran stats) would always fail. The v9 plan never tripped this because v9's rollout always ran stats first; v10's parallel-dataset story exposes the gap.

**Fix — split into five explicit modes:**

| `--mode` | Runs | Requires |
|---|---|---|
| `files` | per-file structural checks (`_validate_file`) + cross-file monotonicity. Plus the v10-specific extras: `zg_source_var == "zg_plev"`, `zg_pressure_levels_hpa == ZG_PLEV_HPA`, `postprocessor_git_sha` present and non-empty (G-pps, §6). | only HDF5 files in `{output-root}/{train,valid,test}/`. **Does NOT need stats/.** |
| `stats` | `_validate_stats(...)` only — checks the six `.npy` shapes/dtypes, std epsilon, and the `zg500` mean range (G-z500, §6). | `{output-root}/stats/*.npy` from a completed `stats.py` run. |
| `smoke` | Synthetic-fixture Phase 4b pytest (existing `run_makani_smoke`). Wrapper-regression check: confirms the patched Makani loader / preprocessor / wrappers still work on synthetic data after this PR's edits. **Does NOT touch `--output-root`.** | torch + makani + physicsnemo. CI-runnable. |
| `smoke-live` | **Live-data preflight against the actual `--output-root`.** Loads `{output-root}/metadata/data.json` + `{output-root}/config/{config-name}.yaml`, instantiates the patched Makani loader against the real H5 train split, runs a 3-step rollout, asserts shapes and finiteness. This is the gate that confirms the new dataset will train. | torch + makani + completed stats + completed metadata + completed config. Cannot run on CI; runs on a compute node. |
| `full` | files → stats → smoke → smoke-live, in order. | everything. |

The legacy `structural` mode is retained as a deprecated alias for `full` for one release with a `DeprecationWarning`; the legacy `makani_smoke` mode aliases to the new `smoke` (synthetic) for backward compat. Downstream callers (`submit.slurm`, `scripts/package_sim52_astro.sh`) migrate to the explicit mode they actually want.

`run_smoke_live(output_root, config_name, n_steps=3, *, yaml_config_override=None)` outline:

```python
_PLACEHOLDER_RE = re.compile(r"\{\{[^}]+\}\}")


def _assert_no_yaml_placeholders(yaml_path: Path) -> None:
    """Fail fast if a yaml passed to smoke-live still has `{{...}}` markers.

    Mirrors `scripts/preflight.py:311-326`. Catches the case where an
    operator passes an unrendered trainer-side template via
    `--yaml-config` — which would otherwise surface deep inside the
    dataloader as a FileNotFoundError on a path like
    `{{OUTPUT_ROOT}}/train`.
    """
    text = yaml_path.read_text()
    leftover = sorted(set(_PLACEHOLDER_RE.findall(text)))
    if leftover:
        raise ValidationError(
            f"yaml {yaml_path} still contains placeholder(s) {leftover[:5]}; "
            f"render with `sed -e s|{{{{OUTPUT_ROOT}}}}|...|g -e s|{{{{EXP_DIR}}}}|...|g` "
            f"(see submit_full.slurm:60-62) before passing to --yaml-config."
        )


def run_smoke_live(
    output_root: Path,
    config_name: str,
    n_steps: int = 3,
    *,
    yaml_config_override: Path | None = None,
) -> None:
    """Live-data preflight: actually load the new dataset and roll forward.

    Distinct from run_smoke_synthetic (the pytest synthetic-fixture
    regression). This is the real gate for "the new dataset will train" —
    it touches the real metadata.json, the real yaml, and the real H5
    files, and exercises the full PlasimForcingDataset →
    PlasimSingleStepWrapper path against them.

    Path resolution (Codex v4 finding #1):
      1. If `yaml_config_override` is set, use it directly. The override
         must already be a *rendered* yaml (placeholders substituted) —
         enforced by `_assert_no_yaml_placeholders`. P4 only renders the
         baseline yaml from the packager-side template; trainer-side
         templates (smoke / tiny / short / full) live in
         `src/sfno_training/config/` with `{{OUTPUT_ROOT}}` / `{{EXP_DIR}}`
         placeholders and must be rendered by the caller (e.g. via the
         `sed` invocation `submit_full.slurm:60-62` uses) before being
         passed here (Codex v6 high finding).
      2. Else default to `{output_root}/config/{config_name}.yaml` (the
         packager-rendered baseline; concrete paths already substituted).
    """
    cfg_path = yaml_config_override or (output_root / "config" / f"{config_name}.yaml")
    meta_path = output_root / "metadata" / "data.json"
    if not cfg_path.exists() or not meta_path.exists():
        raise ValidationError(
            f"smoke-live requires {cfg_path} and {meta_path} to exist; "
            f"run --mode files / stats first, plus metadata.py rendering "
            f"the baseline yaml into {output_root}/config/ (P4 in §5)."
        )
    _assert_no_yaml_placeholders(cfg_path)
    # ... build params, dataset, wrapper, run n_steps ...
```

CLI exposes both: `--config-name NAME` (used with the convention) and `--yaml-config PATH` (override for non-conventional layouts; mirrors `scripts/preflight.py`).

CLI (`_parse_args`): `--mode {files,stats,smoke,smoke-live,full,structural,makani_smoke}` with `full` default (`structural` and `makani_smoke` warn-deprecated). Add a `--config-name` flag (used by `smoke-live` and `full`; defaults to `plasim_sim52_zgplev_baseline`).

Code shape (sketch — full implementation in PR):

```python
def run_files(output_root: Path) -> None:
    split_files = _split_files_in_year_order(output_root)
    if sum(len(v) for v in split_files.values()) == 0:
        raise ValidationError(f"no MOST.*.h5 under {output_root}/{{train,valid,test}}")
    for split, files in split_files.items():
        for path in files:
            year = int(path.stem.split(".")[1])
            T = _validate_file(path, year)         # existing checks
            _validate_v10_attrs(path)              # NEW: zg_source_var, zg_pressure_levels_hpa, postprocessor_git_sha
            logger.info("  ok  %s  (T=%d)", path, T)
        if files:
            _assert_cross_file_monotonic(files)


def run_stats(output_root: Path, epsilon: float) -> None:
    _validate_stats(output_root, epsilon)          # existing
    _validate_zg500_saved_mean(output_root)        # NEW: re-checks the saved .npy (defense-in-depth vs §3.6 inline audit)


def run_full(output_root: Path, epsilon: float, config_name: str) -> int:
    run_files(output_root)
    run_stats(output_root, epsilon)
    rc = run_smoke_synthetic()                     # CI-style synthetic fixture
    if rc != 0:
        return rc
    run_smoke_live(output_root, config_name)        # real-data preflight
    return 0
```

`_validate_v10_attrs(path)`:

- `f.attrs["zg_source_var"]` must equal `"zg_plev"`.
- `np.array(f.attrs["zg_pressure_levels_hpa"]).tolist()` must equal `list(ZG_PLEV_HPA)`.
- `f.attrs["postprocessor_git_sha"]` must be a non-empty string. In production, must not equal `"unknown"` (gate raises if it does); in test/dry-run mode an `--allow-unknown-postproc-sha` flag relaxes this.

`_validate_zg500_saved_mean(output_root)`: re-loads `stats/global_means.npy` and re-runs the audit-range check (defense in depth — `_audit_zg500_inline` in §3.6 protects against bad writes; this call protects against bad stats files arriving from elsewhere or being hand-edited).

**(Note:** the canonical CLI spec is on the earlier `_parse_args` line in this section — `--mode {files,stats,smoke,smoke-live,full,structural,makani_smoke}` with `full` default; `structural` and `makani_smoke` are deprecated aliases. The earlier two-mode list is from the v2 draft and is superseded.)

### 3.8 `src/plasim_makani_packager/submit.slurm`

Update the env-injected paths (`POSTPROC_ROOT`, `BOUNDARY_ROOT`, `OUTPUT_ROOT`) so the v10 SLURM run targets `…_zgplev/`. Add the postprocessor SHA resolution from L7c:

```bash
POSTPROC_SHA=$(git -C "$POSTPROC_SOURCE_DIR" rev-parse HEAD 2>/dev/null || echo unknown)
python3 -m plasim_makani_packager.packager \
    --postprocessor-git-sha "$POSTPROC_SHA" \
    ... rest of v9 args ...
```

`POSTPROC_SOURCE_DIR` defaults to `$HOME/AI-RES/src/plasim_postprocessor`.

### 3.9 `tests/plasim_makani_packager/`

| Test | Change |
|---|---|
| `test_channel_flatten.py:33-34` | `assert STATE_CHANNELS[42] == "zg150"`, `assert STATE_CHANNELS[51] == "zg925"`. |
| `test_channel_flatten.py:71` | Loop is `for v in ("ta", "ua", "va", "hus")` (no `zg`); add a separate stanza that synthesizes `zg_plev` shape `(T, 13, H, W)` on the test fixture. |
| `test_channel_flatten.py:114-115` | Replace the `arr[:, 42] == ds["zg"].values[:, 0]` assert with `arr[:, 42] == ds["zg_plev"].values[:, 2]` (i.e. lev_2 index 2 = 150 hPa = `zg150`). Add a parallel assert at index 47 for `zg500` against `ds["zg_plev"].values[:, 7]` (lev_2 index 7 = 500 hPa). |
| `test_hdf5_writer.py:59` | Same fixture-creation change: build `zg_plev` on `lev_2 = 13` instead of `zg` on `lev = 10`. |
| New: `test_zg_plev_value_lookup.py` | Pin §3.2's value-lookup semantics. **Three cases:** (a) `lev_2 == [50, 100, ..., 1000]` (current source order) packs successfully; (b) `lev_2` shuffled to a non-contiguous order containing all `ZG_PLEV_HPA` values still packs successfully and yields zg500 from the row whose lev_2 value is 500; (c) `lev_2` missing one of `ZG_PLEV_HPA` (e.g. drop 925) raises `RuntimeError` mentioning the missing value. |
| New: `test_zg500_mean_audit.py` | Pin the L7d audit: a synthetic `mean_tgt` with `zg500` mean = 4000 m must trigger the inline `_audit_zg500_inline` assertion; with `zg500` mean = 5550 m must pass; defense-in-depth `_validate_zg500_saved_mean` against a fabricated `global_means.npy` parallels both cases. |
| `test_metadata.py` | The channel-name list it validates is read from `channels.py`, so it picks up the new names automatically. Add explicit assertions: `"zg500" in metadata["coords"]["channel"]`, `metadata["coords"]["channel"][47] == "zg500"`, and that `--variant astro64x128` raises with a reference to the v9 freeze tag. |
| `test_multifile_loader_smoke.py:88` | Replace `Path(...) / "templates" / "plasim_64x128.yaml"` with `Path(...) / "templates" / "plasim_64x128_zgplev.yaml"`. |
| `test_multifile_loader_smoke.py:103,109,115-116` | Replace every literal `"plasim_sim52_astro_64x128"` (4 occurrences) with `"plasim_sim52_astro_64x128_zgplev"`. |

Same renames in `tests/sfno_training/helpers.py:82-83,189,204,210` (4 occurrences of the config name + 1 occurrence of the template path). Codex v2 finding #3: the v2 plan's "no code change" claim for the smoke test was wrong because both files hard-code the v9 template path and config name. Once the v9 template + v9 yamls are no longer the default-rendered output, these tests would import a non-existent template.

Test fixture helper: factor a `_make_synthetic_postproc_ds()` helper into `conftest.py` that emits a postproc-shaped Dataset with `zg_plev(time, lev_2=13, lat, lon)` matching the v10 contract. Drop the `zg_mode` parameter — v9 sigma fixtures are not maintained on the v10 codebase (per L8).

**Physically plausible synthetic `zg_plev` (Codex v3 finding #3).** The new `_audit_zg500_inline` audit (§3.6) hard-fails if `zg500` mean is outside [5400, 5700] m. Several existing tests call `compute_stats()` on synthetic data; if the helper writes random or zero `zg_plev`, every such test trips the audit. Fix in the helper:

```python
# Approximate hydrostatic standard-atmosphere geopotential heights at the 13 lev_2 hPa.
# Used as the *mean* per-level value; per-cell noise added on top is small (~50 m).
_ZG_PLEV_REFERENCE_M: dict[int, float] = {
    50:   20500.0, 100:  16100.0, 150:  13500.0, 200:  11700.0,
    250:  10300.0, 300:   9100.0, 400:   7100.0, 500:   5550.0,
    600:   4200.0, 700:   3000.0, 850:   1450.0, 925:    750.0, 1000:    100.0,
}
# 500 hPa = 5550 m → centre of [5400, 5700] audit band.

def _make_synthetic_zg_plev(T: int, H: int, W: int, *, rng: np.random.Generator) -> np.ndarray:
    out = np.empty((T, 13, H, W), dtype=np.float32)
    for k, hpa in enumerate([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]):
        out[:, k] = _ZG_PLEV_REFERENCE_M[hpa] + rng.normal(0.0, 50.0, size=(T, H, W)).astype(np.float32)
    return out
```

This guarantees `mean(zg500) ≈ 5550 m` over any `(T, H, W)` slab, so all tests that call `compute_stats()` pass the audit. Tests that explicitly want to exercise the audit-fail path build the array manually with an out-of-range `zg500` value (already covered by the new `test_zg500_mean_audit.py`).

### 3.10 `scripts/score_nwp.py` and `scripts/render_eval_report.py` — channel-adaptive

**Codex finding #3:** v1 hard-replaced `zg5` with `zg500` in the scoring scripts, breaking v9 A/B scoring (the v9 emulator's H5/metadata has `zg5`, not `zg500`). v10 must support both contracts because the entire reason for L5's parallel-directory layout is to compare a v9 checkpoint against a v10 checkpoint side-by-side. So:

**Make the scoring scripts channel-adaptive.** Detect the Z500-proxy channel from the run config / inference H5 instead of hard-coding it.

```python
# scripts/score_nwp.py (and parallel change in render_eval_report.py)
Z500_PREFERRED: tuple[str, ...] = ("zg500", "zg5")  # try in order


def _detect_z500_channel(channel_names: Sequence[str]) -> tuple[str, str]:
    """Return (channel_id, label). label = 'Z500 (literal)' or 'Z500 (sigma proxy, v9)'."""
    for name in Z500_PREFERRED:
        if name in channel_names:
            label = "Z500 (literal)" if name == "zg500" else "Z500 (sigma proxy, v9)"
            return name, label
    raise RuntimeError(
        f"no Z500 channel found in {list(channel_names)}; "
        f"expected one of {Z500_PREFERRED}"
    )


def _bias_channels(channel_names: Sequence[str]) -> tuple[str, ...]:
    z500, _ = _detect_z500_channel(channel_names)
    return ("tas", "pr_6h", z500, "ua5", "ta5")
```

Replace the module-constant `_BIAS_CHANNELS = ("tas", "pr_6h", "zg5", "ua5", "ta5")` (`scripts/score_nwp.py:42`) with `_bias_channels(channel_names)` resolved per run. Same for `_KEY_CHANNELS` at `scripts/render_eval_report.py:27`.

For the gate log + assert (`scripts/score_nwp.py:209-220`), parameterize on the detected channel name and surface the label in the printed message:

```python
z500_id, z500_label = _detect_z500_channel(channel_names)
em_acc_24h = _mean(("emulator", z500_id, 24, "acc"))
print(f"[gate] emulator ACC {z500_label} ({z500_id}) 24h = {em_acc_24h:.4f}")
if not (em_acc_24h > 0.6):
    print(f"[gate] FAIL: ACC {z500_label} 24h ({em_acc_24h:.4f}) <= 0.6", file=sys.stderr)
```

The 0.6 threshold stays the same numerically. Its **meaning** changes when the channel is `zg500` (literal Z500) vs `zg5` (sigma proxy). The label in the printed message preserves provenance, which is exactly what the eval-plan §3.11 D.6 stub flags as needing a re-evaluation pass after we have v10 data.

`render_eval_report.py:106-117` parallel changes — format `f"Emulator ACC on \`{z500_id}\` ({z500_label}) at 24 h"`.

**Concrete channel-name source (Codex v3 finding #1 supersedes v2 finding #6).**

The v3 plan resolved channel names from `metadata/data.json` with a CSV fallback, but the eval `OUT_ROOT` actually contains `inference/`, `baselines/`, `scores/` — there is no `metadata/` subdir; that lives at the *packager* output root, not the eval one. And `score_nwp.py` runs **before** the summary CSV exists (the summary is its output). The v3 fallback chain was unrunnable on the real eval directory layout.

**Correct source: read from the inference NetCDF files themselves.** `src/sfno_inference/nc_writer.py:131` already writes `channel=("channel", list(channel_names))` on every emitted NetCDF, so `{out_root}/inference/nwp/*.nc` is the authoritative channel-name source at the moment `score_nwp.py` opens its first input.

```python
# scripts/_eval_utils.py (new shared module)
def resolve_channel_names(
    inference_glob: Path,
    *,
    metadata_json_override: Path | None = None,
) -> list[str]:
    """Resolve channel_names for adaptive Z500 detection.

    Priority:
      1. Explicit --metadata-json override (operator escape hatch).
      2. First inference NetCDF's `channel` coord. All other NetCDFs in the
         glob must agree (hard-fail if any disagree — that would mean the
         eval mixed v9 and v10 inference outputs).
    """
    if metadata_json_override is not None:
        return list(json.loads(metadata_json_override.read_text())["coords"]["channel"])

    nc_files = sorted(inference_glob.parent.glob(inference_glob.name))
    if not nc_files:
        raise RuntimeError(f"no inference NetCDFs at {inference_glob}")
    with xr.open_dataset(nc_files[0]) as ds0:
        names0 = list(ds0["channel"].values.astype(str))
    for p in nc_files[1:]:
        with xr.open_dataset(p) as ds:
            names = list(ds["channel"].values.astype(str))
        if names != names0:
            raise RuntimeError(
                f"channel-name disagreement: {nc_files[0].name} has {names0[:3]}…, "
                f"{p.name} has {names[:3]}…; eval cannot mix v9 and v10 outputs."
            )
    return names0
```

Wire-up:

- `score_nwp.py`: `channel_names = resolve_channel_names(args.out_root / "inference/nwp/*.nc", metadata_json_override=args.metadata_json)`. Adaptive `_detect_z500_channel(channel_names)` then keys all downstream lookups (the `_BIAS_CHANNELS` tuple, the `_mean(("emulator", z500_id, 24, "acc"))` call, the printed gate label).
- `render_eval_report.py`: same — by the time the report is rendered, the inference NetCDFs are still on disk; reading from them keeps render and score consistent.
- Both scripts gain `--metadata-json PATH` as an **optional override**, default `None`. Documented as "for advanced use; normally the channel list is resolved from the inference NetCDFs."

The shared `scripts/_eval_utils.py` module hosts `resolve_channel_names` and `_detect_z500_channel` so the resolution rule has a single source of truth.

`scripts/eval_inference.py` is already data-driven via `cfg["channel_names"]` (lines 171, 252) and needs no code change.

`ua5` and `ta5` remain sigma-level — those channels are unchanged across the v9→v10 cut, so no detection needed for them.

### 3.11 `docs/sfno_eval_plan.md`

| Location | v9 | v10 |
|---|---|---|
| `:769` | "Emulator ACC on `zg500` (channel `zg5`) at 24 h **>** 0.6" | "Emulator ACC on `zg500` at 24 h **>** 0.6 (channel literally `zg500`, not the v9 sigma proxy)." |
| `:758` | bias-channel list `tas, pr_6h, zg5, ua5, ta5` | `tas, pr_6h, zg500, ua5, ta5` |
| `:192` | "42..51  zg1..zg10" | "42..51  zg150..zg925 (TOA→surface; zg500 at index 47)" |
| `:892` | (D.6) sanity-gate-threshold open question | Resolution stub: "Now that the gate measures literal Z500, the 0.6 threshold can be re-evaluated against PlaSim's simpler atmosphere; left at 0.6 for first run, revisit after the first emulator scoring on `…_zgplev`." |

Add a new "Migration note" preamble pointing to this plan.

### 3.12 `scripts/build_subset_dataset.py` and the `sim52_full` subset story

**Codex finding #6:** `plasim_sim52_full.yaml` and `plasim_sim52_short.yaml` do **not** point at the packager root directly — they point at a symlink-farm subset built by `scripts/build_subset_dataset.py`, sharing the full-dataset normalization. v1 of this plan referenced `plasim_sim52_zgplev_full.yaml` without specifying how the corresponding subset gets built or what it points at. Closing that gap now:

The subset-builder script itself is **contract-agnostic**: it symlinks per-year `MOST.YYYY.h5` files and the `stats/` / `metadata/` / `config/` dirs from a source root into a destination root. It does not read or write channel lists. **No code change to `build_subset_dataset.py` is required.**

What is required is the matching subset-build invocations and yaml updates:

| New subset | Built by |
|---|---|
| `$SCRATCH/AI-RES/data/makani/sim52_zgplev_tiny`  | `build_subset_dataset.py --src …_zgplev --dst …_zgplev_tiny  --train-years 3   --valid-years 101` |
| `$SCRATCH/AI-RES/data/makani/sim52_zgplev_short` | `build_subset_dataset.py --src …_zgplev --dst …_zgplev_short --train-years 3-7 --valid-years 101-102` |
| `$SCRATCH/AI-RES/data/makani/sim52_zgplev_full`  | `build_subset_dataset.py --src …_zgplev --dst …_zgplev_full  --train-years 12-111 --valid-years 11` |

Yaml updates (per §3.4): the new `plasim_sim52_zgplev_{tiny,short,full}.yaml` files set `train_data_path: …/sim52_zgplev_{tiny,short,full}` (subset roots), not `…/sim52_astro_64x128_zgplev` (packager root). `plasim_sim52_zgplev_{smoke,baseline}.yaml` keep pointing at the packager root, matching the v9 convention for those configs.

Subset builds run in §5 **P4b**, after stats (P3) and metadata+config (P4). They require `stats/`, `metadata/`, and `config/` to exist in the source root because `build_subset_dataset.py` symlinks all three. They finish in seconds (just symlinks) and run before P5 smoke-live so the second smoke-live invocation can exercise the actual production-training dataset shape.

### 3.13 `src/sfno_training/submit_zgplev_*.slurm` (5 NEW files)

**Codex v4 finding #2:** The existing submit scripts hard-code v9 paths — `submit_full.slurm:57-79` references `plasim_sim52_full.yaml` and `plasim_sim52_full` literally; same pattern in `submit_smoke.slurm`, `submit_tiny.slurm`, `submit_short.slurm`. P7 / P8 cannot run without v10 equivalents.

Per L8, do **not** edit the v9 submit scripts in place — they remain functional for v9 retrains on `main`. Instead create 5 new sibling scripts. Base-script mapping (Codex v6 cleanup):

| New v10 script | Base v9 script |
|---|---|
| `src/sfno_training/submit_zgplev_smoke.slurm`    | `src/sfno_training/submit_smoke.slurm`    |
| `src/sfno_training/submit_zgplev_tiny.slurm`     | `src/sfno_training/submit_tiny.slurm`     |
| `src/sfno_training/submit_zgplev_short.slurm`    | `src/sfno_training/submit_short.slurm`    |
| `src/sfno_training/submit_zgplev_baseline.slurm` | `src/sfno_training/submit_train.slurm` *(no `submit_baseline.slurm` exists in the v9 set; `submit_train.slurm` is the closest sibling — same single-node training shape, no subset symlink farm)* |
| `src/sfno_training/submit_zgplev_full.slurm`     | `src/sfno_training/submit_full.slurm`     |

Each is a copy of its base v9 script with these textual replacements:

| v9 string | v10 string |
|---|---|
| `plasim_sim52_{smoke,tiny,short,full}.yaml` | `plasim_sim52_zgplev_{smoke,tiny,short,baseline,full}.yaml` |
| `plasim_sim52_{smoke,tiny,short,full}` (config-name arg) | `plasim_sim52_zgplev_{smoke,tiny,short,baseline,full}` |
| `sim52_{tiny,short,full}` (subset root in `train_data_path` env-var) | `sim52_zgplev_{tiny,short,full}` |
| `sim52_astro_64x128` (packager root for smoke/baseline) | `sim52_astro_64x128_zgplev` |
| run-dir / exp-dir suffix | append `_zgplev` |

Header comment of each script gains a one-liner pointing at this plan and noting the v9 sibling stays in place for the frozen contract.

A future cleanup PR can deduplicate via env-var-parameterized launchers (`CONFIG_TEMPLATE`, `CONFIG_NAME`, `OUTPUT_ROOT`), but doubling the file count today is the lower-risk path: fewer moving parts, and the v9 vs v10 split is fully visible at the file-listing level.

---

## 4. Output dataset strategy (L5 detail)

| Path | v9 (frozen) | v10 (new) |
|---|---|---|
| Postproc NetCDF | `$SCRATCH/AI-RES/data/postproc/sim52/MOST.{YYYY}.nc` | unchanged (same files) |
| Boundary NetCDF | `$SCRATCH/AI-RES/data/boundary_astro/sim52/boundary.{YYYY}.nc` | unchanged |
| Packaged HDF5 | `$SCRATCH/AI-RES/data/makani/sim52_astro_64x128/{train,valid,test}/MOST.{YYYY}.h5` | `$SCRATCH/AI-RES/data/makani/sim52_astro_64x128_zgplev/{train,valid,test}/MOST.{YYYY}.h5` |
| Stats | `…_64x128/stats/*.npy` | `…_zgplev/stats/*.npy` |
| Metadata | `…_64x128/metadata/data.json` | `…_zgplev/metadata/data.json` |
| Trainer YAML | `src/sfno_training/config/plasim_sim52_*.yaml` | `src/sfno_training/config/plasim_sim52_zgplev_*.yaml` |
| Run dir | `$SCRATCH/AI-RES/runs/sim52_astro_64x128/` | `$SCRATCH/AI-RES/runs/sim52_astro_64x128_zgplev/` |
| Eval results | `$SCRATCH/AI-RES/results/sim52_astro_64x128/…` | `$SCRATCH/AI-RES/results/sim52_astro_64x128_zgplev/…` |

Disk-space note: 126 packaged years × 1460 6-hourly samples/year (4/day × 365) × 64 × 128 × 53 channels at f32 ≈ 50 GB per dataset; both contracts coexisting costs ~100 GB on `$SCRATCH`. Acceptable.

---

## 5. Rollout order

Phase boundaries are gates: do not start phase N+1 if phase N's verification step failed. Validate-mode names below refer to the five modes defined in §3.7 (`files`, `stats`, `smoke`, `smoke-live`, `full`).

1. **P0a — v9 freeze tag.**
    - Before any code edit lands: `git tag plasim-makani-packager-v9-final && git push origin plasim-makani-packager-v9-final`.
    - This is the only commit at which v9 packager artifacts can be regenerated post-merge (per L8).
    - **Gate:** tag exists in remote.

2. **P0b — Code edits (local).**
    - All §3.1–§3.13 file edits, all §3.9 test updates.
    - Run `pytest tests/plasim_makani_packager/` (the 24 non-makani tests). Must pass.
    - **Gate:** unit-test pass + a `git diff` that only touches files in §3.1–§3.13.

3. **P1 — Packaging dry run (1 sim-year).**
    - Pick `sim52, year=3` (first non-warmup year).
    - `python -m plasim_makani_packager.packager --sims 52 --task-index 0 --postproc-root … --boundary-root … --output-root $SCRATCH/AI-RES/data/makani/sim52_astro_64x128_zgplev --postprocessor-git-sha $(git -C $HOME/AI-RES/src/plasim_postprocessor rev-parse HEAD)`.
    - `python -m plasim_makani_packager.validate --output-root $SCRATCH/AI-RES/data/makani/sim52_astro_64x128_zgplev --mode files`.
    - **Gate:** `--mode files` passes on the single file (G-pps + G-ASCII + G-zattrs). Stats are not yet present and `--mode stats` is **not** run here — that's the v1 bug Codex caught.

4. **P2 — Full packaging SLURM (sim52, all 126 non-warmup years).**
    - Update `submit.slurm` per §3.8.
    - `sbatch --array=0-125 src/plasim_makani_packager/submit.slurm`.
    - Spot-check a sample of files across train/valid/test splits.
    - `python -m plasim_makani_packager.validate --output-root … --mode files`.
    - **Gate:** all 126 jobs succeed; `--mode files` passes over the full output root.

5. **P3 — Stats + audit (L7a, L7d).**
    - `python -m plasim_makani_packager.stats --output-root $SCRATCH/AI-RES/data/makani/sim52_astro_64x128_zgplev` (inline `_audit_zg500_inline` from §3.6 raises before any `.npy` is written if `mean_tgt[zg500]` is out of range).
    - `python -m plasim_makani_packager.validate --output-root … --mode stats` (defense-in-depth: re-checks the saved `global_means.npy` per §3.7 `_validate_zg500_saved_mean`).
    - **Gate:** stats files present + std epsilon clean + zg500 audit passes.

6. **P4 — Metadata + baseline config render.**
    - `python -m plasim_makani_packager.metadata --output-root $SCRATCH/AI-RES/data/makani/sim52_astro_64x128_zgplev --variant zgplev --exp-dir $SCRATCH/AI-RES/runs/sim52_astro_64x128_zgplev --config-name plasim_sim52_zgplev_baseline` (variant defaults to `zgplev` per §3.5; `astro64x128` errors pointing at the v9 freeze tag).
    - Writes `{output_root}/metadata/data.json` and `{output_root}/config/plasim_sim52_zgplev_baseline.yaml`. Only the *baseline* yaml is rendered from the packager template — `metadata.py` reads one template, substitutes `{{OUTPUT_ROOT}}` / `{{EXP_DIR}}`, and only renames the top-level key, so it cannot semantically differentiate smoke / tiny / short / full configs. **The other four configs already exist as hand-curated trainer-side files at `src/sfno_training/config/plasim_sim52_zgplev_{smoke,tiny,short,full}.yaml` (per §3.4); they do not get re-rendered here.**
    - **Gate:** `data.json` exists with `zg500` in `coords.channel`; `config/plasim_sim52_zgplev_baseline.yaml` exists and passes `yaml.safe_load` round-trip; the yaml's `channel_names[47] == "zg500"`.

7. **P4b — Subsets (depends on P3 + P4).**
    - `build_subset_dataset.py` requires `stats/`, `metadata/`, and `config/` to exist in the source root (`LINKED_DIRS` at line 69 of the subset-builder). So subset construction must come **after** P3 (stats) AND P4 (metadata + config). Codex v2 finding #1: in v2 this was bundled into P3, which would have failed.
    - Build subsets per §3.12: `scripts/build_subset_dataset.py` invocations for `_zgplev_tiny`, `_zgplev_short`, `_zgplev_full`.
    - **Gate:** all three subsets exist with `train/valid/test/stats/metadata/config` symlinks resolving to the parent root.

8. **P5 — Live Makani preflight + synthetic regression (L7b).**
    - Synthetic regression: `python -m plasim_makani_packager.validate --output-root … --mode smoke` (CI-runnable; exercises the patched-Makani plumbing on synthetic fixtures only — does not touch `--output-root`).
    - **Live preflight (baseline yaml, packager root):** `python -m plasim_makani_packager.validate --output-root $SCRATCH/AI-RES/data/makani/sim52_astro_64x128_zgplev --mode smoke-live --config-name plasim_sim52_zgplev_baseline` — exercises the rendered baseline yaml against the full packager root.
    - **Live preflight (full yaml, full subset) — uses `--yaml-config` override per §3.7.** The trainer-side template `plasim_sim52_zgplev_full.yaml` contains `{{OUTPUT_ROOT}}` / `{{EXP_DIR}}` placeholders (same convention as v9), so it must be rendered first (Codex v6 high finding). Mirror what `submit_full.slurm:60-62` already does:
        ```bash
        FULL_OUTPUT_ROOT=$SCRATCH/AI-RES/data/makani/sim52_zgplev_full
        FULL_EXP_DIR=$SCRATCH/AI-RES/runs/sim52_astro_64x128_zgplev_full
        FULL_TPL=$REPO_ROOT/src/sfno_training/config/plasim_sim52_zgplev_full.yaml
        FULL_RENDERED=$FULL_EXP_DIR/plasim_sim52_zgplev_full.rendered.yaml
        mkdir -p "$FULL_EXP_DIR"
        sed -e "s|{{OUTPUT_ROOT}}|$FULL_OUTPUT_ROOT|g" \
            -e "s|{{EXP_DIR}}|$FULL_EXP_DIR|g" \
            "$FULL_TPL" > "$FULL_RENDERED"

        python -m plasim_makani_packager.validate \
            --output-root "$FULL_OUTPUT_ROOT" \
            --mode smoke-live \
            --yaml-config "$FULL_RENDERED" \
            --config-name plasim_sim52_zgplev_full
        ```
      The trainer-side `plasim_sim52_zgplev_full.yaml` (hand-curated per §3.4 with `train_data_path: {{OUTPUT_ROOT}}/train` resolved to `…/sim52_zgplev_full/train`, full-subset training params, etc.) is the actual production-training config — `metadata.py` cannot generate this from its baseline template (Codex v5 finding). The production training path uses the `_zgplev_full` subset (built in P4b) plus this trainer-side yaml; the baseline preflight does not exercise the symlink-farm layout or the subset's `train_data_path`. Without this second preflight, P7 is the first thing that touches the production path — and we'd rather discover a subset-symlink or yaml-wiring bug here, on a 30-second compute-node check, than at the start of a smoke-train. `_assert_no_yaml_placeholders` (§3.7) is the safety net if the operator forgets the `sed` step.
    - **Gate:** all three pass. The two live-preflight passes are the hard gate before any GPU-hours are spent.

9. **P6 — Eval-script + plan updates (§3.10, §3.11).**
    - Same PR; CI runs the relevant `tests/` (any tests referencing the channel-adaptive scoring path).
    - Verify channel-adaptive scoring against **both** a v9 fixture (recorded inference output with `zg5`) and a v10 fixture (with `zg500`).
    - **Gate:** scoring works against v9 fixture (resolves to `zg5`, prints "Z500 (sigma proxy, v9)" label) and v10 fixture (resolves to `zg500`, prints "Z500 (literal)" label).

10. **P7 — Smoke-train (`sbatch src/sfno_training/submit_zgplev_smoke.slurm`, ~1 hour GPU).**
    - First end-to-end run on the new contract via the new submit script (§3.13). The v9 `submit_smoke.slurm` stays untouched.
    - **Gate:** loss decreases monotonically over the first epoch, no NaN, no channel-count assertion fails.

11. **P8 — Production retrain (`sbatch src/sfno_training/submit_zgplev_full.slurm`).**
    - The actual retraining run that gives us a Z500-skill emulator.
    - Eval via channel-adaptive `score_nwp.py` (`ACC zg500 24 h > 0.6` — auto-detected from the checkpoint's `channel_names`).

The v9 dataset (`sim52_astro_64x128`), its checkpoints, and its eval results stay untouched throughout; this plan is purely additive on the storage side. v9 retrains continue to work on `main` because the trainer is data-driven (see L8). v9 *regeneration* via packager / stats / metadata / validate requires `git checkout plasim-makani-packager-v9-final`.

---

## 6. Verification gates (consolidated, per L7)

| Gate | What it checks | Where it runs | Fail mode |
|---|---|---|---|
| G-stats (L7a) | `stats.py` reran on the new output root, all stats `.npy` present, no channel std < 1e-6. | P3 — `stats.py` itself + `validate --mode stats`. | Abort retrain. |
| G-smoke-synth (L7b, CI) | Synthetic-fixture wrapper-regression. | P5 — `validate --mode smoke`. CI-runnable. | Abort retrain (likely a wrapper-patch regression, not a data issue). |
| G-smoke-live (L7b, hard gate) | Live-data preflight: real `--output-root`, real metadata.json, real yaml, real H5; 3-step rollout passes. | P5 — `validate --mode smoke-live`. Compute-node only. | Abort retrain. **This is the actual hard gate before GPU-hours.** |
| G-pps (L7c) | Every new HDF5 file has `postprocessor_git_sha` ∈ file_attrs and (in production) ≠ "unknown". | P1, P2 — `validate --mode files`. | Repackage. |
| G-z500 (L7d) | `mean_tgt[zg500]` ∈ [5400, 5700] m, checked **inline before stats .npy is written**, then again on the saved file. | P3 — inline in `compute_stats()` + `validate --mode stats`. | Abort stats output; investigate (likely wrong slice or wrong lev_2 lookup). |
| G-ASCII | Every new HDF5's `channel_state` ASCII matches the v10 list. | P1, P2 — `validate --mode files`. | Repackage. |
| G-zattrs | Every new HDF5 has `zg_source_var == "zg_plev"` and `zg_pressure_levels_hpa == ZG_PLEV_HPA`. | P1, P2 — `validate --mode files` via `_validate_v10_attrs`. | Repackage. |
| G-adaptive-scoring | Channel-adaptive scoring resolves correctly against both a v9 fixture (`zg5`) and a v10 fixture (`zg500`). | P6 — pytest in `tests/`. | Block merge. |

`validate --mode full` runs G-pps, G-ASCII, G-zattrs (via `files`) → G-stats, G-z500 (via `stats`) → G-smoke-synth (via `smoke`) → G-smoke-live (via `smoke-live`), in that order. Each individual mode runs only its slice — so P1 / P2 can call `--mode files` without tripping on missing stats, and P5 can run `smoke` and `smoke-live` independently.

---

## 7. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Postprocessor changes lev_2 ordering, count, or values without coordination. | Low | §3.2 looks each requested hPa up by value in the source `lev_2`, so reordering is invisible and missing values raise loudly. `postprocessor_git_sha` (L7c) makes the postproc version traceable from each HDF5. |
| `zg_plev` has unphysical values at one of the new levels (especially 925 hPa in regions where surface > 925 hPa is common — high terrain). | Medium-low | G-z500 only audits 500 hPa explicitly; add a soft-warn in stats.py for any zg* level whose global mean falls outside its empirical altitude range (table from `docs/plasim_postprocessor_audit.md`). Not blocking. |
| Trainer code accidentally hard-codes `zg5` somewhere. | Very low | `grep -rn 'zg5\b' src/sfno_training/` finds nothing today (verified during interview). The trainer treats channels as opaque; no code change is required. |
| The 0.6 ACC gate is wrong (too tight or too loose) for literal Z500. | Medium | Acknowledged — keep 0.6 for the first scoring pass; revisit after we have a number. Captured in §3.11 D.6 stub. |
| Stats normalization for `zg500` (~5550 m) vs the previous `zg5` sigma (~5500 m at sigma=0.57 ≈ 580 hPa) is different enough that v9 checkpoints cannot be warm-started on the new dataset. | High (intended) | This is by design — the v9 checkpoint is invalidated for v10, which is why we keep both directories (L5). v10 starts from random init or from a v9 checkpoint with `zg` channel weights re-initialized — to be decided at P8 setup. |
| Eval scripts break for any consumer still feeding v9-trained checkpoints in. | Medium | `scripts/eval_inference.py` reads `channel_names` from the cfg; pointing it at the v9 yaml + v9 checkpoint + v9 dataset still works because all three are in lockstep. The only failure mode is mixing v9 + v10 artifacts, which the `validate.py` channel-list check catches. |

---

## 8. Open questions (for Codex review)

1. **Subset choice (L1):** I argued for the 10-level hybrid (drop 50/100/1000) over the group's bottom-heavy set and Codex's top-heavy set. If Codex disagrees, the change is local (one tuple in `channels.py`) and the rest of this plan is invariant under the choice.
2. **`zg500` audit range (L7d):** [5400, 5700] m comes from `docs/plasim_postprocessor_audit.md:193`. That number is for sim30, year 12, fldmean over time. Is that range conservative enough for sim52's 98-year training mean, or should it widen to e.g. [5350, 5750] to absorb interannual variability?
3. **`zg925` sanity check:** Should we add a soft-warn (not gate) on `zg925` mean to catch the high-terrain extrapolation problem? Mean over global cells should be ~750 m; if it's much lower (e.g. < 500 m) over an overweighted high-terrain region, the postproc has a problem.
4. **Postprocessor SHA resolution mode (§3.2 Option A vs B):** Option A keeps L3 honored. If Codex prefers the cleaner B (write the SHA from the postprocessor itself), we re-open L3.
5. **Retrain initialization (§7 risk row 5):** v10 from random init, or warm-start from v9's first 42 channels (everything except zg)? Out of scope for this plan but flagged for the P8 setup.

---

## 9. Cross-references

- **v9 contract (parent):** `docs/plasim_makani_packager_plan.md`.
- **Postprocessor source:** `src/plasim_postprocessor/plasim_postprocessor.py:79` (`PRESSURE_LEVELS`).
- **Audit snapshot:** `docs/audit_snapshots/manifest.txt:41` (`lev_2` coord).
- **Audit `zg_plev @ 500 hPa` validation:** `docs/plasim_postprocessor_audit.md:193`.
- **Eval plan:** `docs/sfno_eval_plan.md` (lockstep update §3.11).
- **SFNO trainer:** `src/sfno_training/trainer/plasim_trainer.py:111-112, 272-283`, `src/sfno_training/models/preprocessor.py:34-55` — no code change required, called out for completeness.

---

*End of plan v7.*
