# Plan v3: SFNO-5410 external-user inference path on Stampede3

> **Status: pre-implementation, drafted for third-round Codex review.**
> Date: 2026-05-08. Author: zhixingliu (via Claude). Audience: Codex
> reviewer + a group member (the eventual end user).
>
> Supersedes plan-v2 (`..._v2.md`). v2 → v3 changes driven by the
> 2026-05-08 second Codex review and three explicit user policy
> decisions:
>
> | v2 issue | v3 resolution |
> |---|---|
> | "arbitrary days" not actually supported | **Patch upstream** to support partial chunks (user choice) |
> | BYO h5 mode under-specified | **Defer entirely** (user choice); sim52 boundaries only |
> | Year-0001 ensemble relabel changes filenames + Feb 29 trap | **Auto-relabel + manifest**; AND patch removes the year-1 gate so the workaround is unnecessary |
> | Single boundary_template_year too restrictive | **Two flags**: `--leap-template-year`, `--no-leap-template-year` |
> | `ensemble_inference_hours = 8784` blindly | Set to **exact rollout duration** in hours |
> | Fresh-output-dir conflicts with scaffolding order | Freshness checked **before** scaffolding |
> | `requirements.txt` doesn't exist | Use `requirements-stampede3.txt` + `external/PanguWeather_stampede3_env.txt` |
> | v2 inconsistency on `val_year_start = val_year_end` | Fixed in §3 |
>
> The first-cut draft `src/sfno_inference_5410/user_inference.py`
> from v1 is now **stale** and should be rewritten from scratch per
> §3.

---

## 1. Context (unchanged from v2)

A group member on G-819272 will run SFNO-5410 inference on Stampede3
with **her own initial-condition NetCDF**. She does NOT use the sim52
test years (0121–0128) or the 96-IC evaluation pipeline. Use case:
arbitrary IC, optional perturbation ensemble, output written as
NetCDF. Audience for the guide: first-time user of our setup,
comfortable on a Linux cluster.

User decisions locked in (latest first):
- **2026-05-08 round 2**: sub-year horizon → patch upstream;
  BYO h5 → defer; output → manifest + auto-relabel.
- **2026-05-08 round 1**: ensemble flavors = deterministic +
  perturbation; permission scope = group-readable G-819272.

## 2. Upstream invariants

### 2.1 Yearly-chunk save model (now patched)

**Pristine upstream behavior** (verified in
`/work2/.../v2.0/long_inference.py`):
- Line 367: `range(init.year, final.year)` — empty when same year.
- Lines 553-554, 562-563: chunk endpoint hard-coded to `Jan 1 next year`.
- Line 688: save trigger fires only on Jan-1 boundary.
- Lines 826-845: duplicate copy of the same allocator+save logic
  in a second code branch (the non-`use_6h_24h_model` path).

**Decision**: patch upstream to support partial chunks bounded by
`final_datetime`. Patch is small (4 hunks, ~20 lines), tracked in
a new `docs/2026-05-08_panguweather_local_patches.md`.

### 2.2 Boundary year decoupling (kept from v2 §2.2; clarified)

Three independent year fields with single sources of truth:

| Yaml field | Used at | Set from CLI flag |
|---|---|---|
| `val_year_start` | `data_loader_multifiles.py:741` (constant boundary load, **single read** at file `<dir>/<val_year_start>_0000.h5`) | `--no-leap-template-year` (default 121) |
| `val_year_end` | `data_loader_multifiles.py:845-846` (sets dataset start/end for the IC dataset) | `val_year_start + 1` to avoid empty range; no semantic effect |
| `leap_year` | `data_loader_multifiles.py:932-934` (file template for **leap rollout years**) | `--leap-template-year` (default 124) |
| `no_leap_year` | same line, **non-leap rollout years** | `--no-leap-template-year` (default 121) |

The model clock (`init_datetime`, `final_datetime` CLI flags into
upstream) is **independent** of all four yaml year fields. This
matches the upstream branch at
`data_loader_multifiles.py:950-958` which already substitutes
`leap_year`/`no_leap_year` when `start_date.year != val_year_start`.

### 2.3 Ensemble seeding + perturbation gate

`long_inference.py:206`:
```python
seed = self.params.run_iter * self.params.world_size
```

**Pristine gate** at `:558, :830`:
```python
if self.params.epsilon_factor > 0. and self.params.init_datetime.year == 1:
```

The year-1 condition is **also patched out** in the same patch as
§2.1 (atomic upstream change for both v2 blockers at once). After
the patch, the gate is just `epsilon_factor > 0.`, and the user's
IC stays in her actual calendar year. **No Feb 29 trap, no relabel
needed.**

### 2.4 Output filename (kept; relabeling is in our wrapper)

`long_inference.py:1206` (unchanged):
```python
filename = save_basename + f'_member{total_run_iter:03}_y{current_year:04}.nc'
```

Where `current_year` is the data year being saved (in the user's
calendar after the patch). So filenames are now naturally
`*_y<user-year>.nc` and the auto-relabel step in our wrapper
becomes purely a manifest step (no rename needed).

### 2.5 Environment (corrected from v2)

There is no root `requirements.txt`. The actual files are:

| File | Purpose |
|---|---|
| `/home1/11114/zhixingliu/projects/SFNO_Climate_Emulator/requirements-stampede3.txt` | base PyTorch / xarray / h5py / cftime stack for Stampede3 |
| `/home1/11114/zhixingliu/projects/SFNO_Climate_Emulator/external/PanguWeather_stampede3_env.txt` | Pangu/SFNO-specific extras (modulus_makani, etc.) the upstream code imports |

The user guide must point at both.

## 3. Design

### 3.1 Upstream patch: `docs/2026-05-08_panguweather_local_patches.md` + edits to `long_inference.py`

**Patch surface (4 hunks):**

**Hunk A** — `_get_inference_duration` (line 363-369):
```diff
 def _get_inference_duration(self):
     steps_per_year = [
         (self.dataset.datetime_class(year + 1, 1, 1, hour=0, has_year_zero = self.params.has_year_zero) - \
          self.dataset.datetime_class(year, 1, 1, hour=0, has_year_zero = self.params.has_year_zero)).total_seconds() \
          // 3600 / self.params.timedelta_hours for year in range(self.params.init_datetime.year, self.params.final_datetime.year)
     ]
+    # AI-RES-Stampede3 local patch (2026-05-08): support partial-final-year rollouts.
+    # Pristine upstream returns empty list when init.year == final.year (sub-year
+    # rollout); also doesn't account for the partial chunk between the last full
+    # year and final_datetime. We append a final partial step count when needed.
+    init_dt = self.params.init_datetime
+    final_dt = self.params.final_datetime
+    if final_dt > self.dataset.datetime_class(final_dt.year, 1, 1, has_year_zero=self.params.has_year_zero):
+        partial_hours = (final_dt - self.dataset.datetime_class(final_dt.year, 1, 1, has_year_zero=self.params.has_year_zero)).total_seconds() / 3600
+        steps_per_year.append(partial_hours / self.params.timedelta_hours)
+    if init_dt.year == final_dt.year:
+        # Sub-year rollout: replace the (empty) list with the single partial chunk.
+        partial_hours = (final_dt - init_dt).total_seconds() / 3600
+        steps_per_year = [partial_hours / self.params.timedelta_hours]
     return np.array(steps_per_year)
```

**Hunks B + C** — chunk allocator + save trigger, both branches (lines 553-554, 826-829):
```diff
-                    next_output_datetime = self.dataset.datetime_class(current_year+1, 1, 1, hour=next_year_offset_hours,
-                                                                    has_year_zero = self.params.has_year_zero)
+                    # AI-RES-Stampede3 local patch (2026-05-08): bound chunk by final_datetime.
+                    next_year_jan1 = self.dataset.datetime_class(current_year+1, 1, 1, hour=next_year_offset_hours,
+                                                                  has_year_zero = self.params.has_year_zero)
+                    next_output_datetime = min(next_year_jan1, self.params.final_datetime)
```
And same change at the second occurrence inside the inner save loop
(lines 723-724, 836).

**Hunk D** — drop year-1 perturbation gate (lines 558, 830):
```diff
-                    if self.params.epsilon_factor > 0. and self.params.init_datetime.year == 1:
+                    if self.params.epsilon_factor > 0.:
                         print('Perturbing ICs...')
                         input_surface, input_upper_air = self.perturber(input_surface, input_upper_air)
```
Same change at the duplicate site near line 830.

**Patch tracking doc** at
`docs/2026-05-08_panguweather_local_patches.md`:
- Lists each hunk with file:line and rationale.
- Includes a `git diff` snapshot.
- Includes a regression test (Tier 1) that confirms the patches are
  in place: text-grep for `min(next_year_jan1, self.params.final_datetime)`
  and absence of `init_datetime.year == 1` in the file.
- On upstream resync, this test fails loudly so we don't silently
  lose the patch.

### 3.2 Module: `src/sfno_inference_5410/user_inference.py` (rewrite)

The v1 draft is stale. Rewrite per below.

```python
build_user_yaml(
    *,
    out_dir: Path,
    exp_dir: Path,
    leap_template_year: int,                # default 124 (sim52 leap)
    no_leap_template_year: int,             # default 121 (sim52 non-leap)
    rollout_hours: int,                     # exact (final - init) hours
    epsilon_factor: float,
    num_ensemble_members: int,
    save_basename: str,
    perturbation_type: Optional[str] = None,
    boundary_data_dir: Path = STAMPEDE3_DATA_DIR,
    bias_data_dir: Path = STAMPEDE3_BIAS_DIR,
    climatology_file: Path = STAMPEDE3_CLIM_NC,
    src_yaml: Path = UPSTREAM_YAML_PATH,
) -> Path
```

`_override_section_user` sets exactly:
```python
section["data_dir"]                = str(boundary_data_dir)
section["bias_data_dir"]           = str(bias_data_dir)
section["climatology_file"]        = str(climatology_file)
section["load_exp_dir"]            = str(UPSTREAM_LOAD_EXP_DIR)
section["exp_dir"]                 = str(exp_dir)

# Year fields (independent from model clock — see plan §2.2):
section["val_year_start"]          = no_leap_template_year
section["val_year_end"]            = no_leap_template_year + 1   # +1 to avoid empty range at line 845-846; no semantic effect on constant-boundary load
section["leap_year"]                = leap_template_year
section["no_leap_year"]             = no_leap_template_year

# Inference knobs:
section["save_forecasts"]          = True
section["log_to_wandb"]            = False
section["ensemble_inference_hours"] = rollout_hours              # exact (final - init); see §2.1
section["save_basenames"]          = [save_basename]
section["epsilon_factor"]          = epsilon_factor
section["num_ensemble_members"]    = num_ensemble_members
section["ensemble_members_per_pred"] = num_ensemble_members
if epsilon_factor > 0:
    section["perturbation_type"] = perturbation_type
```

Note: `val_year_end = val_year_start + 1` is set explicitly (NOT
`= val_year_start`); this fixes the v2 inconsistency.

### 3.3 Module: `src/sfno_inference_5410/preflight.py`

Strictly **read-only** validator. Runs after CLI parse, BEFORE any
filesystem scaffolding.

```python
def preflight_pre_scaffold(
    *,
    init_nc: Path,
    output_dir: Path,
    leap_template_year: int,
    no_leap_template_year: int,
    boundary_data_dir: Path,
    init_datetime: cftime.DatetimeProlepticGregorian,
    final_datetime: cftime.DatetimeProlepticGregorian,
    epsilon_factor: float,
    num_ensemble_members: int,
    perturbation_type: Optional[str],
) -> None: ...

def preflight_post_scaffold(
    *,
    yaml_path: Path,
    ckpt_shim_path: Path,
    expected_scaffold: set[Path],
) -> None: ...
```

**Pre-scaffold checks** (raise on failure; output dir untouched):
- `output_dir` either does not exist OR exists, is a directory, AND
  is empty. **Defines "fresh"**: nothing inside it.
- IC NC variable schema (per `data_loader_multifiles.py:83-159`):
  required vars `ta, ua, va, hus, zg, pl, tas`, dims, calendar
  `proleptic_gregorian`, time units `hours since YYYY-01-01 00:00:00`.
- IC NC sigma_lev/plev levels match yaml within `level_delta=1e-4`.
- Boundary tree coverage: every 6h step from init to final must
  resolve to an existing h5 file under
  `<boundary_data_dir>/<template>_<idx>.h5`. Per-step check uses
  `cftime.is_leap_year(year, "proleptic_gregorian", has_year_zero=True)`
  to pick `leap_template_year` vs `no_leap_template_year`. **This is
  the §2.2 rule applied per-step**, so multi-year leap+non-leap
  spans are correctly handled.
- mean/std files at `<boundary_data_dir>/<basename>` for each of
  `surface_mean`, `surface_std`, `surface_ff_std`, `upper_air_mean`,
  `upper_air_std`, `upper_air_ff_std`, `boundary_mean`,
  `boundary_std`, `diagnostic_mean`, `diagnostic_std` (basenames
  from upstream yaml).
- Bias dir exists and is non-empty.
- Climatology file exists.
- Checkpoint at upstream path is readable.
- `final.year > init.year` OR `final > init` AND patches are present
  (text-grep `long_inference.py` for the patch marker — see §3.1).
- `num_members > 1` ⟹ `epsilon_factor > 0` (else error).
- `epsilon_factor > 0` ⟹ `perturbation_type` set, in
  `{gaussian_noise, gaussian_noise_n_minus_1, perlin_noise}`.

**Post-scaffold checks** (run between scaffold and launch):
- Generated yaml has no `/glade/` strings.
- Generated yaml has all expected fields (see `_override_section_user`).
- Ckpt shim resolves to `UPSTREAM_CKPT_PATH` via `os.path.realpath`.
- Expected scaffold set matches what's actually under output_dir
  (no rogue extra files).

Exposed as standalone CLI: `python -m sfno_inference_5410.preflight
--init-nc <path> [other flags]` for the user to run before sbatching.

### 3.4 Module: `scripts/run_sfno_5410_inference.py`

Order of operations (precise, fixes v2 freshness conflict):

```
1.  Parse CLI.
2.  Read IC datetime from NC time coord (or --init-datetime override).
3.  Compute final_datetime:
        final_datetime = init_datetime + horizon_days * 24h
        rollout_hours  = horizon_days * 24
4.  Compute leap-template policy:
        rollout_year_set = years touched by [init, final)
        for each year in rollout_year_set:
            if cftime.is_leap_year(year): expects leap_template_year
            else:                          expects no_leap_template_year
5.  Ensemble pre-checks (no relabel needed after upstream patch §3.1).
6.  Run preflight_pre_scaffold(...). Raises on any failure.
7.  Create output_dir (now empty per pre-scaffold check).
8.  Scaffold:
        a. Build yaml via build_user_yaml(...).
        b. Build ckpt symlink shim under <output_dir>/SFNO/5410/checkpoints/.
        c. Write manifest.json with: original CLI args, IC datetime,
           final datetime, ckpt sha, git rev, leap-template policy,
           upstream-patch markers detected.
9.  Run preflight_post_scaffold(...). Raises on yaml drift.
10. Construct upstream argv:
        python -u <upstream>/long_inference.py
            --run_num 5410 --yaml_config <yaml> --config SFNO
            --init_datetime <s> --final_datetime <s>
            --init_nc_filepaths <init-nc>
            --output_dir <output_dir> --save_basename <save_basename>
            --async_save
11. subprocess.run(argv, cwd=upstream_repo, env=os.environ).
12. Postflight:
        a. Count output NCs.
        b. Per Codex round-2 §3 of v2-review (manifest + auto-relabel):
           since the upstream patch keeps init_datetime.year intact,
           output filenames are already in user's calendar — NO rename
           needed. The "auto-relabel" reduces to:
              i.  Validate output `time` coord matches expected init.
              ii. Append output-summary section to manifest.json.
              iii. (No file renames, no time-coord rewrites.)
        c. Print summary: count of NCs, per-file size, manifest path.
```

**Note**: with the upstream patch, the year-0001 relabel is gone.
"Manifest + auto-relabel" reduces to just the manifest, because
output files are already in the user's calendar. This simplifies
both the wrapper and the documentation.

### 3.5 CLI flags

| Flag | Required | Default | Notes |
|---|---|---|---|
| `--init-nc PATH` | yes | — | her IC NetCDF |
| `--horizon-days N` | yes | — | rollout length in days; sub-year supported via §3.1 patch |
| `--output-dir DIR` | yes | — | must be empty or non-existent |
| `--save-basename NAME` | no | `inference` | drives output filename |
| `--epsilon-factor F` | no | 0.0 | 0 = deterministic, >0 = ensemble |
| `--perturbation-type T` | iff ε>0 | — | gaussian_noise / gaussian_noise_n_minus_1 / perlin_noise |
| `--num-members N` | no | 1 | ensemble size; >1 implies ε>0 |
| `--leap-template-year Y` | no | 124 | sim52 leap year template |
| `--no-leap-template-year Y` | no | 121 | sim52 non-leap year template |
| `--init-datetime YYYY-MM-DD_HH:MM:SS` | no | from IC NC | override IC datetime |
| `--dry-run` | no | False | validate + scaffold, print upstream argv, do NOT launch |

**No `--boundary-mode`** — sim52 only (BYO deferred).
**No `--boundary-data-dir`** in v3 — use sim52 default.

### 3.6 SLURM template: `scripts/submit_sfno_5410_user_inference.slurm`

H100, 1 node, 1 GPU. Wallclock default 4h (covers up to ~3-month
rollout comfortably; user adjusts for longer). Top-of-file knobs:
`INIT_NC`, `OUTPUT_DIR`, `HORIZON_DAYS`, optional ensemble flags.

### 3.7 Tests

| File | Scope |
|---|---|
| `tests/sfno_inference_5410/test_user_inference.py` | Tier-1 yaml regression: deterministic + ensemble; pinned values; leap/no_leap independence |
| `tests/sfno_inference_5410/test_preflight.py` | Each helper individually with monkeypatched filesystem |
| `tests/sfno_inference_5410/test_upstream_patch_present.py` | Static text-grep that long_inference.py contains the 4 patch markers; fails loud on upstream resync that loses the patch |

### 3.8 User guide: `docs/2026-05-08_sfno_5410_external_user_guide.md`

Audience: first-time user. Sections (v2 list with v3 changes):

1. **Prerequisites** — TACC account, allocation, group G-819272.
2. **One-time setup** —
   2.1 Clone repo from GitHub: `git clone git@github.com:feynmanliu214/AI-RES-Stampede3.git`.
   2.2 venv: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements-stampede3.txt && pip install -r external/PanguWeather_stampede3_env.txt`.
   2.3 Confirm shared assets: `ls -la /work2/.../v2.0/results/SFNO/5410/checkpoints/ckpt_epoch_50.tar` etc.
3. **Asset map** — explicit absolute paths.
4. **Input NetCDF schema** — variables, dims, units, calendar; xarray + netCDF4 worked example.
5. **Boundary template years** — explanation of `--leap-template-year` (default 124) and `--no-leap-template-year` (default 121); why two flags; user guide example for a multi-year rollout that crosses both.
6. **Rollout horizon** — `--horizon-days` works for any positive
   horizon (sub-year, 1-year, multi-year). Note: this is enabled by
   our local upstream patch; on a clean upstream fork, sub-year
   wouldn't work.
7. **Deterministic walkthrough** — copy-paste example.
8. **Perturbation-ensemble walkthrough** — copy-paste example.
   - Within-run member diversity (single sbatch ⟹ all members differ).
   - Cross-run reseeding via fresh `--output-dir` + (advanced)
     custom `run_iter` if she wants different filename ranges.
9. **Output schema** — `{save_basename}_member{NNN}_y{YYYY}.nc`,
   8 vars, dims, units (group conventions: `pl=ln(p_s)`, `zg=gpm`,
   `pr_6h=rate×6h` — **must NOT be re-converted**).
10. **Manifest** — `<output-dir>/manifest.json` records inputs +
    git rev + ckpt sha + upstream-patch markers; for reproducibility.
11. **Common failures** — table:
    - `IndexError on inference_idxs` / no output → upstream patch missing (run `pytest tests/sfno_inference_5410/test_upstream_patch_present.py`)
    - `FileNotFoundError: <year>_<idx>.h5` → leap-template policy didn't match a rollout year (fix: pass correct `--leap-template-year` for rollout years that are leap)
    - `Members all identical despite epsilon>0` → upstream patch missing
    - `output dir not empty` → use a fresh path
    - `Permission denied` → check `groups | grep G-819272`
    - `cftime: invalid date Feb 29 year X` → IC year X is not a leap year in proleptic_gregorian; check IC datetime
12. **Verification** — short smoke (1-day, deterministic, IC = our smoke IC if available).
13. **Contact / support**.

### 3.9 Permissions

Same as v2 §3.7 (path-specific). Repeated here for completeness:

| Path | Asset | Current | Proposed |
|---|---|---|---|
| `/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator` | parent | 700 | 750 |
| `/work2/.../v2.0/` (recursive) | upstream code + ckpt | mostly 755 | `g+rX -R` (no-op for already-755 files) |
| `/scratch/11114/zhixingliu/SFNO_Climate_Emulator` | parent | 710 | 750 |
| `/scratch/.../sim52/` (recursive) | sim52 boundary tree + mean/std + bias + clim | mixed | `g+rX -R` |

Commands (NOT yet run):
```bash
chmod g+rX  /work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator
chmod -R g+rX /work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0
chmod g+rX  /scratch/11114/zhixingliu/SFNO_Climate_Emulator
chmod -R g+rX /scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52
```

Repo (code) cloned from GitHub; no $HOME chmod.

## 4. File list

| File | Status | Purpose |
|---|---|---|
| `/work2/.../v2.0/long_inference.py` | **PATCH** (4 hunks) | sub-year horizon + drop year-1 gate |
| `docs/2026-05-08_panguweather_local_patches.md` | **NEW** | track upstream patches |
| `src/sfno_inference_5410/user_inference.py` | **REWRITE** (v1 draft is stale) | yaml builder |
| `src/sfno_inference_5410/preflight.py` | **NEW** | pre/post-scaffold validators |
| `scripts/run_sfno_5410_inference.py` | **NEW** | user-facing CLI |
| `scripts/submit_sfno_5410_user_inference.slurm` | **NEW** | SLURM template |
| `tests/sfno_inference_5410/test_user_inference.py` | **NEW** | yaml regression |
| `tests/sfno_inference_5410/test_preflight.py` | **NEW** | preflight units |
| `tests/sfno_inference_5410/test_upstream_patch_present.py` | **NEW** | guard against losing patch on resync |
| `docs/2026-05-08_sfno_5410_external_user_guide.md` | **NEW** | the deliverable |

**Not modified:**
- `src/sfno_inference_5410/stampede3_yaml_override.py` (eval-track)
- `scripts/eval_inference_5410.py`, `build_5410_yaml_override.py`, the eval SLURMs
- Upstream `PanguWeather/v2.0/` files OTHER than `long_inference.py` (only the 4 hunks above)

## 5. Verification plan

1. **Unit tests** — `pytest tests/sfno_inference_5410/ -q`:
   - existing eval-track Tier 1, 2, 3 (still pass — no eval-code change)
   - new test_user_inference, test_preflight (login-node, <30s)
   - new test_upstream_patch_present (login-node, <1s, fails-loud on patch loss)
2. **Dry-run** — `scripts/run_sfno_5410_inference.py --dry-run` for both deterministic + ensemble; confirm argv + scaffold + manifest.
3. **Smoke deterministic on H100** — 1-year rollout from sim52 IC `121_0000.nc`; compare to existing 5410 smoke output (should match bit-exactly except logging differences). Confirms patch hasn't broken the integer-year case.
4. **Smoke sub-year on H100** — 30-day rollout from sim52 IC `121_0000.nc`; confirm:
   - Output NC has 120 timesteps (30 × 4) — confirms patch supports partial chunks.
   - Variable schema unchanged.
   - No `IndexError` on `inference_idxs`.
5. **Smoke ensemble on H100** — 1-year rollout, IC dated 0125 (her real calendar year, NO relabel needed), num_members=4, epsilon=1e-3, gaussian_noise. Confirm:
   - 4 NCs written (`*_member000_y0125.nc` … `*_member003_y0125.nc`) — filenames in user's calendar.
   - Members differ from each other.
   - Member 0 differs from a deterministic baseline — confirms year-1 gate is removed.
6. **Permission verification** — apply chmod, log before/after stat.
7. **Doc walkthrough** — read user guide cold, follow each step.

## 6. Future work (out of scope for this iteration)

- BYO HDF5 mode (deferred per user choice; see plan-v2 §2.3 for the schema spec when revisiting).
- Real `perturbation_seed` yaml key (decouple seed from `run_iter`).
- Multi-IC ensemble (drop single-IC invariant in eval-track).
- Migrate sim52 boundary tree from $SCRATCH to $WORK to avoid purge.

## 7. Risks (revised)

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | Upstream patch breaks on resync | high if resync happens | `test_upstream_patch_present.py` fails loud; `docs/2026-05-08_panguweather_local_patches.md` documents reapplication |
| 2 | Patch's partial-chunk math has off-by-one in datetime arithmetic | medium | Smoke sub-year (§5.4) catches this; assert exact timestep count |
| 3 | Patch removes year-1 gate but breaks something subtle in eval-track ensembling | low (eval-track uses epsilon=0 always) | Eval Tier 1+2+3 still pass; smoke deterministic confirms |
| 4 | $SCRATCH purge | low | Document |
| 5 | Leap/no-leap template mismatch in multi-year rollout | medium | Per-step preflight (§3.3) |
| 6 | Output dir collision | low | Pre-scaffold freshness check |
| 7 | Permission widening exposes more than needed | low | Path-specific; no $HOME chmod |
| 8 | User passes wrong leap-template year for her rollout span | medium | Per-step preflight surfaces this with clear error message |

## 8. Open questions for round-3 Codex review

1. **Upstream patch hunk B/C symmetry** — both branches need the
   same `min(next_year_jan1, final_datetime)` change. Verify both
   sites are caught (lines 553-554 main loop AND the inner save-fired
   reset around 723-724; PLUS the second outer branch around 826-836).
   I want Codex to spot-check the patch covers all sites before I
   apply it.

2. **`val_year_end = val_year_start + 1` semantics** — confirmed this
   only avoids the empty-range edge case at
   `data_loader_multifiles.py:845-846` and has no effect on the
   constant-boundary load (which only reads `val_year_start`)?

3. **Manifest-only post-processing** — with the patch removing the
   year-1 gate, file renames become unnecessary. Does Codex agree
   the wrapper postprocessing reduces to just writing manifest +
   counting outputs?

4. **`ensemble_inference_hours` exact value** — set to
   `(final - init).total_seconds() // 3600`. For multi-year, this
   is larger than 8784 — is that safe? `data_loader_multifiles.py:597,
   831` use it as a `timedelta(hours=...)` end-of-range, so it
   should scale linearly. Codex to confirm.

5. **Should we expose `run_iter`** for advanced cross-run ensembling?
   Same concern as v2: it shifts filename numbering. Recommendation:
   no, keep it hidden; document the post-hoc renumbering trick if
   user asks.

6. **Tier 3 drift detector for user-yaml path** — should the existing
   `test_upstream_attr_drift.py` be extended to check params reads
   on the user-yaml code path? Or is it sufficient that we share
   the same upstream code?

## 9. Sequence of work after Codex sign-off

1. Apply upstream patch per §3.1; commit `long_inference.py` change.
2. Write `docs/2026-05-08_panguweather_local_patches.md`.
3. Implement `test_upstream_patch_present.py`; run.
4. Rewrite `src/sfno_inference_5410/user_inference.py` per §3.2.
5. Implement `src/sfno_inference_5410/preflight.py` per §3.3.
6. Implement `test_user_inference.py`, `test_preflight.py`; run all 5410 tests.
7. Implement `scripts/run_sfno_5410_inference.py` per §3.4-§3.5.
8. Verify §5.2 dry-run.
9. Implement SLURM template §3.6.
10. Apply permissions per §3.9; log.
11. Run §5.3, §5.4, §5.5 H100 smokes.
12. Write user guide §3.8 cold-walkthrough; iterate until clean.
13. Hand to user for delivery.
