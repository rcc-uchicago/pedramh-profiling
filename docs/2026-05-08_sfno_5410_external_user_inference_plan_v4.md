# Plan v4: SFNO-5410 external-user inference path on Stampede3

> **Status: APPROVED by Codex 2026-05-08 (4th-round review).** Two
> small edits applied since the approval comment:
>
> 1. (Required by Codex) 6-hour alignment preflight on
>    `init_datetime` — see §3.3.
> 2. (Optional, accepted) `ensemble_inference_hours = min(8784,
>    horizon_days * 24)` — see §3.2.
>
> Date: 2026-05-08. Author: zhixingliu (via Claude). Audience: Codex
> reviewer + a group member (the eventual end user).
>
> Supersedes plan-v3 (`..._v3.md`). v3 → v4 changes driven by
> 2026-05-08 third Codex review (4 blockers + 1 name-collision):
>
> | v3 issue | v4 resolution |
> |---|---|
> | Patch only fixed 1st chunk endpoint; loop continuation `current_year < final.year` at lines 720, 998 still cuts off late chunks (Dec 15 + 30 days breaks) | **Patch the continuation conditions too** (lines 720, 998) |
> | BCS dataloader's `single_ic` `_get_dates` ends at `year_start + long_rollout_years` for sub-year (== 0) | **Set yaml key `prediction_duration_days = horizon_days`** — explicit upstream override at `data_loader_multifiles.py:821-823`, no patch needed |
> | Patch hunk list missed line 1002 (sync-path year-rollover) | **Patch all four allocator sites**: 554, 724, 834, 1002 |
> | `_get_inference_duration` hunk was dead code (zero callers; only defined at :363) | **Drop hunk entirely** |
> | `ensemble_inference_hours = rollout_hours` blindly causes multi-year IC-dataset boundary preload bloat | **Cap at 8784**; drives only the IC dataset's preload, not the rollout |
> | `preflight.py` name collision with existing `scripts/preflight.py` (training, not 5410) | Use `src/sfno_inference_5410/preflight.py` only; add explicit note in code header |
>
> Drafts on disk (`src/sfno_inference_5410/user_inference.py` from
> v1) remain stale and should be rewritten per §3.

---

## 1. Context (unchanged from v3)

A group member on G-819272 will run SFNO-5410 inference on Stampede3
with **her own initial-condition NetCDF**. Use case: arbitrary IC,
arbitrary horizon in days, optional perturbation ensemble. sim52
boundary mode only (BYO deferred per user decision). Permission
scope: G-819272 group-readable.

## 2. Upstream invariants (corrected with line-by-line evidence)

### 2.1 Yearly-chunk save model — six edit sites, not four

Pristine upstream behavior in
`/work2/.../v2.0/long_inference.py`:

**Allocator sites** (all set `next_output_datetime = Jan 1 next year`):

| Line | Path | Context |
|---|---|---|
| 554 | async path, initial allocation | first chunk before the rollout loop |
| 724 | async path, year-rollover reallocation | inside `if time_step_in_year == output_surface.shape[1]:` save block, when continuation passes |
| 834 | sync path, initial allocation | mirror of 554 |
| 1002 | sync path, year-rollover reallocation | mirror of 724 |

All four use `current_year+1, 1, 1` as the next chunk endpoint —
need to bound by `final_datetime`.

**Continuation conditions** (decide whether to allocate the next
chunk after a save):

| Line | Path | Pristine condition |
|---|---|---|
| 720 | async path | `if current_year < self.params.final_datetime.year:` |
| 998 | sync path  | `if current_year < self.params.final_datetime.year:` |

**Why these matter for partial-final-year rollouts:**
Example `init=Dec 15 0125, final=Jan 14 0126`:
- Chunk 1 allocated for `Dec 15 0125 → Jan 1 0126` (size 68 steps after the line-554 patch).
- After Jan 1 save, `current_year` increments to 126.
- Continuation `126 < 126` is **false** → loop exits before the `Jan 1 → Jan 14` partial chunk runs.

The fix is to change BOTH continuation conditions to
`current_datetime < self.params.final_datetime`.

**Perturbation gate** (drop year-1 condition):

| Line | Path |
|---|---|
| 558 | async path |
| 830 | sync path |

Both have `if self.params.epsilon_factor > 0. and self.params.init_datetime.year == 1:`.

**Total patch surface**: 4 allocator + 2 continuation + 2
perturbation = **8 hunks** in `long_inference.py`.

### 2.2 BCS dataloader: no patch needed (use existing yaml key)

`data_loader_multifiles.py:818-826`:
```python
if self.single_ic:
    start_date = self.datetime_class(self.year_start, 1, 1, ...) + timedelta(hours=self.single_ic_offset)
    if hasattr(self.params, "prediction_duration_days"):
        print(f'Initializing data loader for {self.params.prediction_duration_days} day prediction.')
        end_date = start_date + timedelta(days=self.params.prediction_duration_days)
    else:
        end_date = self.datetime_class(self.year_start + self.long_rollout_years, 1, 1, ...)
```

The yaml key `prediction_duration_days` is an explicit override. By
setting it to `horizon_days` in our yaml override, the BCS
dataloader runs from `start_date` to `start_date + horizon_days`
exactly — covering both sub-year and partial-final-year cases
without any upstream change.

`long_rollout_years` is unrelated for our path. (It's still computed
at `long_inference.py:190` from `final_datetime.year - init_datetime.year`,
which is 0 for sub-year — but with `prediction_duration_days` set,
that code path is skipped.)

### 2.3 `_get_inference_duration` is dead

`grep -rn _get_inference_duration` in upstream returns only the
definition at `long_inference.py:363`. **Zero callers. Drop the v3
Hunk A patch entirely.**

### 2.4 ensemble_inference_hours is the IC dataset's preload knob

Used at `data_loader_multifiles.py:483, 597, 609, 831, 834` — all
inside the **IC dataset** (the ensemble + init_from_nc loader at
`long_inference.py:192-195`). Specifically:
- Line 483: `ensemble_inference_steps = ensemble_inference_hours // timedelta_hours` (sizes a preload counter).
- Lines 597, 609: end-of-range for `_load_varying_boundary_data`'s preload over the IC dataset's span.
- Lines 831, 834: end_date for the IC dataset's `_get_dates` ensemble branch.

The IC dataset's preloaded varying boundary is **not used by the
rollout** — `long_inference.py:582-590` shows the rollout iterates
over `self.data_loader_bcs` (the BCS dataset) for boundary data,
not the IC dataset. So `ensemble_inference_hours` only matters for
construction-time memory/I/O of an unused tensor.

**Decision**: cap at **8784** (1 leap year, max non-pathological
value). Big enough to not break any internal sizing assumption;
small enough to keep multi-year preload from blowing up.

### 2.5 Boundary year decoupling (kept from v3 §2.2)

Three independent year fields, set from two CLI flags:

| Yaml field | Used at | CLI flag |
|---|---|---|
| `val_year_start` | `data_loader_multifiles.py:741` (constant boundary, single read at `<dir>/<val_year_start>_0000.h5`) | `--no-leap-template-year` (default 121) |
| `val_year_end` | `data_loader_multifiles.py:845-846` (sets dataset year span; constant-boundary load only reads `val_year_start`) | `val_year_start + 1` automatically |
| `leap_year` | `data_loader_multifiles.py:932-934` (template for **leap rollout years**) | `--leap-template-year` (default 124) |
| `no_leap_year` | same line, **non-leap rollout years** | `--no-leap-template-year` (default 121) |

Per-step preflight checks the right template for each year crossed
by the rollout (correct for multi-year leap+non-leap spans).

### 2.6 Environment (corrected from v2)

The repo has `requirements-stampede3.txt` (root) plus
`external/PanguWeather_stampede3_env.txt`. **No `requirements.txt`.**
User guide must point at both.

## 3. Design

### 3.1 Upstream patch: 8 hunks in `long_inference.py`

Tracked in new `docs/2026-05-08_panguweather_local_patches.md`
including a `git diff` snapshot and a Tier-1 regression test
(`tests/sfno_inference_5410/test_upstream_patch_present.py`).

**Hunk 1** — line 554 (async path, initial allocator):
```diff
-                    next_output_datetime = self.dataset.datetime_class(current_year+1, 1, 1, hour=next_year_offset_hours,
-                                                                    has_year_zero = self.params.has_year_zero)
+                    # AI-RES local patch: bound chunk endpoint by final_datetime so partial-final-year rollouts save correctly.
+                    next_year_jan1 = self.dataset.datetime_class(current_year+1, 1, 1, hour=next_year_offset_hours,
+                                                                  has_year_zero = self.params.has_year_zero)
+                    next_output_datetime = min(next_year_jan1, self.params.final_datetime)
```

**Hunk 2** — line 558 (async path, perturbation gate):
```diff
-                    if self.params.epsilon_factor > 0. and self.params.init_datetime.year == 1:
+                    # AI-RES local patch: drop year-1 gate so perturbation fires for any IC year.
+                    if self.params.epsilon_factor > 0.:
                         print('Perturbing ICs...')
                         input_surface, input_upper_air = self.perturber(input_surface, input_upper_air)
```

**Hunk 3** — line 720 (async path, continuation):
```diff
-                            current_year += 1
-                            # If this was not the final year
-                            if current_year < self.params.final_datetime.year:
+                            current_year += 1
+                            current_datetime = next_output_datetime
+                            # AI-RES local patch: continue while still inside rollout window (handles partial-final-year).
+                            if current_datetime < self.params.final_datetime:
```
NB: also moves `current_datetime = next_output_datetime` from
inside the body to before the test, so the new condition can use
the updated value. Body of the `if` keeps using `current_datetime`
via line 722 (which can now be removed since it's hoisted).

**Hunk 4** — line 724 (async path, year-rollover allocator):
```diff
-                                next_output_datetime = self.dataset.datetime_class(current_year+1, 1, 1, hour=next_year_offset_hours,
-                                                                                has_year_zero = self.params.has_year_zero)
+                                # AI-RES local patch: same final_datetime cap as Hunk 1.
+                                next_year_jan1 = self.dataset.datetime_class(current_year+1, 1, 1, hour=next_year_offset_hours,
+                                                                              has_year_zero = self.params.has_year_zero)
+                                next_output_datetime = min(next_year_jan1, self.params.final_datetime)
```

**Hunks 5-8**: mirrors of Hunks 1-4 in the **sync path** at lines
834, 830, 998, 1002. Same diff pattern.

**Patch tracking doc** at `docs/2026-05-08_panguweather_local_patches.md`:
- Lists each hunk with file:line + rationale + diff
- Includes verification commands (apply, run pytest, smoke)
- Lists procedure for re-application after upstream resync

### 3.2 Module: `src/sfno_inference_5410/user_inference.py` (rewrite)

Stale v1 draft replaced. New surface:

```python
build_user_yaml(
    *,
    out_dir: Path,
    exp_dir: Path,
    leap_template_year: int,                # default 124
    no_leap_template_year: int,             # default 121
    horizon_days: int,                      # → prediction_duration_days
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
section["data_dir"]                 = str(boundary_data_dir)
section["bias_data_dir"]            = str(bias_data_dir)
section["climatology_file"]         = str(climatology_file)
section["load_exp_dir"]             = str(UPSTREAM_LOAD_EXP_DIR)
section["exp_dir"]                  = str(exp_dir)

section["val_year_start"]           = no_leap_template_year
section["val_year_end"]             = no_leap_template_year + 1   # avoid empty range; no semantic effect
section["leap_year"]                = leap_template_year
section["no_leap_year"]             = no_leap_template_year

section["save_forecasts"]           = True
section["log_to_wandb"]             = False
section["save_basenames"]           = [save_basename]
section["epsilon_factor"]           = epsilon_factor
section["num_ensemble_members"]     = num_ensemble_members
section["ensemble_members_per_pred"] = num_ensemble_members
if epsilon_factor > 0:
    section["perturbation_type"]    = perturbation_type

# v4 fixes:
section["prediction_duration_days"] = horizon_days   # bounds BCS dataset (per §2.2)
# IC dataset preload cap (per §2.4): cap at 8784 (1 leap year) to avoid
# multi-year preload bloat, but for short horizons use the actual span
# so we don't preload a full year for a 30-day run.
section["ensemble_inference_hours"] = min(8784, horizon_days * 24)
```

### 3.3 Module: `src/sfno_inference_5410/preflight.py`

Header makes the name disambiguation explicit (Codex's note):
```python
"""5410 inference preflight — distinct from scripts/preflight.py
(SFNO training data preflight). This module validates inputs and
scaffold for a single-IC long_inference.py invocation against the
group SFNO-5410 emulator on Stampede3.
"""
```

Public surface and check list unchanged from v3 §3.3, with two
additions:

**1. Generated-yaml `prediction_duration_days` check** (driven by §2.2):
the post-scaffold validator confirms the yaml contains
`prediction_duration_days` matching the user's `horizon_days`.

**2. 6-hour alignment check on `init_datetime`** (Codex round-4
required edit). Pre-scaffold, raise if any of:

```python
init_datetime.hour % 6 != 0
init_datetime.minute != 0
init_datetime.second != 0
```

Rationale: SFNO-5410 advances on a 6-hour grid; the boundary h5
index is computed via integer division at
`data_loader_multifiles.py:928-929`:

```python
data_idx = int((data_datetime - Jan-1-of-data-year).total_seconds()
               // 3600 // self.data_timedelta_hours)
```

An IC at e.g. `03:00` would silently floor to the same `data_idx`
as `00:00`, reading the wrong boundary file (or producing an
off-by-one in subsequent steps). With integer `horizon_days`,
`final_datetime` inherits alignment from `init_datetime`, so the
single check on init suffices. Error message points the user at
the nearest valid 6h boundary so the fix is obvious.

### 3.4 Module: `scripts/run_sfno_5410_inference.py`

Order of operations (unchanged from v3 §3.4 except final_datetime
is now used directly without sub-year branching, and
manifest-only postprocessing is the only postprocess step):

```
 1.  Parse CLI.
 2.  Read IC datetime from NC time coord (or --init-datetime override).
 3.  Compute final_datetime = init_datetime + horizon_days × 24h.
 4.  Compute leap-template policy (per §2.5).
 5.  Ensemble pre-checks (no relabel after Hunks 2 + 6).
 6.  Run preflight_pre_scaffold(...).
 7.  Create empty output_dir.
 8.  Scaffold yaml + ckpt shim + manifest.
 9.  Run preflight_post_scaffold(...).
10.  Construct upstream argv (unchanged from v3).
11.  subprocess.run(argv, cwd=upstream_repo, env=os.environ).
12.  Postflight: count NCs, append output summary to manifest.
```

### 3.5 CLI flags (unchanged from v3 §3.5)

`--init-nc, --horizon-days, --output-dir, --save-basename,
--epsilon-factor, --perturbation-type, --num-members,
--leap-template-year, --no-leap-template-year, --init-datetime,
--dry-run`. No `--boundary-mode`.

### 3.6 SLURM template (unchanged from v3 §3.6)

`scripts/submit_sfno_5410_user_inference.slurm`. H100, 1 node, 1 GPU,
4h wallclock default.

### 3.7 Tests

| File | Scope |
|---|---|
| `tests/sfno_inference_5410/test_user_inference.py` | Tier-1 yaml regression (incl. `prediction_duration_days`, `ensemble_inference_hours==8784`) |
| `tests/sfno_inference_5410/test_preflight.py` | Each preflight helper |
| `tests/sfno_inference_5410/test_upstream_patch_present.py` | Static text-grep for **all 8** patch markers across the 8 hunk sites |

### 3.8 User guide (unchanged from v3 §3.8)

`docs/2026-05-08_sfno_5410_external_user_guide.md`. Sections same as v3.

### 3.9 Permissions (unchanged from v3 §3.9)

Path-specific group-readable widening; no $HOME chmod.

## 4. File list

| File | Status | Purpose |
|---|---|---|
| `/work2/.../v2.0/long_inference.py` | **PATCH** (8 hunks) | sub-year horizon + drop year-1 gate |
| `docs/2026-05-08_panguweather_local_patches.md` | **NEW** | track upstream patches |
| `src/sfno_inference_5410/user_inference.py` | **REWRITE** | yaml builder (v1 stale) |
| `src/sfno_inference_5410/preflight.py` | **NEW** | inference preflight (header notes name disambiguation) |
| `scripts/run_sfno_5410_inference.py` | **NEW** | user-facing CLI |
| `scripts/submit_sfno_5410_user_inference.slurm` | **NEW** | SLURM template |
| `tests/sfno_inference_5410/test_user_inference.py` | **NEW** | yaml regression |
| `tests/sfno_inference_5410/test_preflight.py` | **NEW** | preflight units |
| `tests/sfno_inference_5410/test_upstream_patch_present.py` | **NEW** | 8-hunk patch presence |
| `docs/2026-05-08_sfno_5410_external_user_guide.md` | **NEW** | the deliverable |

**Not modified:**
- `src/sfno_inference_5410/stampede3_yaml_override.py` (eval-track)
- `scripts/{eval_inference_5410.py, build_5410_yaml_override.py}`, the eval SLURMs
- `scripts/preflight.py` (existing SFNO training preflight; distinct from 5410)
- Upstream `PanguWeather/v2.0/` files OTHER than `long_inference.py`

## 5. Verification plan

1. **Unit tests** — `pytest tests/sfno_inference_5410/ -q`:
   - existing eval Tier 1, 2, 3 (pass — no eval code touched)
   - new test_user_inference, test_preflight (login-node, <30s)
   - new test_upstream_patch_present (8 markers; <1s; fails loud on resync)
2. **Dry-run** — `scripts/run_sfno_5410_inference.py --dry-run`
   for both deterministic (`horizon_days=30`) and ensemble
   (`horizon_days=365, num_members=4`); confirm yaml + scaffold + manifest.
3. **Smoke deterministic on H100, 1-year** — IC = sim52 `121_0000.nc`,
   `--horizon-days 365`. Expected: `*_y0121.nc` with 1460 timesteps;
   bit-exact match (modulo logging) to existing eval-track smoke.
4. **Smoke deterministic on H100, sub-year** — IC = sim52 `121_0000.nc`,
   `--horizon-days 30`. Expected: `*_y0121.nc` with 120 timesteps;
   no IndexError; confirms patch supports partial chunks.
5. **Smoke deterministic on H100, partial-final-year** — IC dated
   Dec 15 0125 (would need to be built — could synthesize from
   sim52 if needed), `--horizon-days 30`. Expected: 2 NCs
   (`*_y0125.nc` with 68 timesteps, `*_y0126.nc` with 52 timesteps);
   confirms patch handles year-rollover with partial final chunk.
6. **Smoke ensemble on H100** — IC dated 0125-06-15 (real calendar,
   no relabel), `--horizon-days 365, --num-members 4, --epsilon-factor 1e-3,
   --perturbation-type gaussian_noise`. Expected:
   - 4 NCs `*_member000_y0125.nc` … `*_member003_y0125.nc`
   - filenames in user's calendar (confirms Hunks 2 + 6 dropped year-1 gate)
   - members differ from each other
7. **Permission verification** — apply chmod, log before/after stat.
8. **Doc walkthrough** — read user guide cold, follow each step.

## 6. Future work (out of scope for this iteration)

- BYO HDF5 mode (deferred per user choice; v2 §2.3 has the schema spec for revisit).
- Real `perturbation_seed` yaml key (decouple seed from `run_iter`).
- Multi-IC ensemble.
- Migrate sim52 boundary tree from $SCRATCH to $WORK to avoid purge.

## 7. Risks (revised)

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | Upstream patch breaks on resync | high if resync | `test_upstream_patch_present.py` checks all 8 markers |
| 2 | Hunk 3 / Hunk 7 (continuation rewrite) introduces an off-by-one or breaks the `current_datetime = next_output_datetime` hoisting | medium | Smoke partial-final-year (§5.5) catches this; assert exact NC count + per-NC timestep count |
| 3 | Patches break eval-track smoke (job 3097041 baseline) | low (the rewritten branches are semantically equivalent for full-year rollouts) | Smoke deterministic 1-year (§5.3) compares to baseline |
| 4 | `prediction_duration_days` interacts oddly with the IC dataset (which doesn't read this key) | low | Eval Tier 2 covers IC dataset construction |
| 5 | `ensemble_inference_hours = 8784` is too small for some upstream sizing assumption I missed | low | Smoke 1-year already tested with 8784 in eval-track; sub-year smoke confirms |
| 6 | Multi-year leap+non-leap span template mismatch | medium | Per-step preflight |
| 7 | Output dir collision | low | Pre-scaffold freshness |
| 8 | $SCRATCH purge | low | Document |
| 9 | Permission widening exposes more than needed | low | Path-specific |

## 8. Round-4 Codex review answers (incorporated in this revision)

1. **Hunk 3 / Hunk 7 hoisting safe.** Confirmed by Codex.
2. **`prediction_duration_days`** is the right BCS-loader fix (does
   not leak into IC dataset's ensemble branch). Confirmed by Codex.
3. **`ensemble_inference_hours = min(8784, horizon_hours)`**
   accepted as the optional improvement. Applied in §3.2.
4. **Partial-final-year time coords** should work via
   `xr.date_range(..., inclusive="left")`. Confirmed by Codex.
5. **Patch tests should check all 8 hunks structurally.** Already
   in §3.7 / §5.1 design.

## 9. (none — plan approved)

## 10. Sequence of work after Codex sign-off

1. Apply 8-hunk upstream patch per §3.1; commit `long_inference.py` change.
2. Write `docs/2026-05-08_panguweather_local_patches.md`.
3. Implement `test_upstream_patch_present.py`; run.
4. Rewrite `src/sfno_inference_5410/user_inference.py` per §3.2.
5. Implement `src/sfno_inference_5410/preflight.py` per §3.3.
6. Implement `test_user_inference.py`, `test_preflight.py`; run all 5410 tests.
7. Implement `scripts/run_sfno_5410_inference.py` per §3.4-§3.5.
8. Verify §5.2 dry-run.
9. Implement SLURM template §3.6.
10. Apply permissions per §3.9; log.
11. Run §5.3 (1-year), §5.4 (sub-year), §5.5 (partial-final-year), §5.6 (ensemble) smokes.
12. Write user guide §3.8 cold-walkthrough.
13. Hand to user.
