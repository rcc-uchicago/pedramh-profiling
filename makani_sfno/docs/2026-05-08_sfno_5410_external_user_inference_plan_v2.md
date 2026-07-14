# Plan v2: SFNO-5410 external-user inference path on Stampede3

> **Status: pre-implementation, drafted for second-round Codex review.**
> Date: 2026-05-08. Author: zhixingliu (via Claude). Audience: Codex
> reviewer + a group member (the eventual end user).
>
> Supersedes `2026-05-08_sfno_5410_external_user_inference_plan.md`
> (v1) which had four blocking issues caught by Codex review:
>
> 1. Arbitrary `--horizon-days` was not actually supported — upstream
>    saves yearly chunks and the rollout iterator is empty for
>    same-year init/final.
> 2. The year-0001 ensemble workaround set `val_year_start=1`, which
>    would make the constant-boundary loader read from year 1 (no
>    such file in the sim52 tree).
> 3. BYO HDF5 mode was underspecified — `data_dir` is also the
>    mean/std root, so a BYO boundary tree needs staged stats files.
> 4. Ensemble seeding via `run_iter * world_size` couples the seed to
>    the output filename (`total_run_iter` in
>    `long_inference.py:1196`), and the year-0001 relabel breaks for
>    Feb 29 ICs.
>
> All four are addressed below, with file:line citations to the
> upstream code that drove each fix.
>
> The first-cut draft `src/sfno_inference_5410/user_inference.py`
> from v1 still exists on disk but **needs revision** per §3.1 — do
> not treat it as the proposed implementation.

---

## 1. Context (unchanged from v1)

A group member on G-819272 will run SFNO-5410 inference on Stampede3
with **her own initial-condition NetCDF**. She does NOT use the sim52
test years (0121–0128) or the 96-IC evaluation pipeline. Use case:
arbitrary IC, optional perturbation ensemble, output written as
NetCDF. Audience for the guide: first-time user of our setup,
comfortable on a Linux cluster.

User decisions locked in 2026-05-08:
- Boundaries: **both** sim52 default + BYO h5 advanced.
- Ensemble flavors: **deterministic single-IC** + **perturbation
  ensemble around single IC** (multi-IC NOT in scope).
- Permission scope: **G-819272 group-readable**.

## 2. Upstream invariants the design now respects

All citations resolve in
`/work2/.../v2.0/long_inference.py` and
`/work2/.../v2.0/utils/data_loader_multifiles.py`.

### 2.1 Yearly-chunk save model (Blocker 1)

`long_inference.py:553-554, 562-563, 688`:
```python
next_output_datetime = datetime_class(current_year + 1, 1, 1, ...)
output_inference_steps = (next_output_datetime - current_datetime) / 6h
output_surface = np.zeros((batch, output_inference_steps, ...))
...
if time_step_in_year == output_surface.shape[1]:   # save trigger
    save_to_netcdf(...)
```

Each chunk is sized to the next Jan-1 boundary; save fires only at
that boundary. **Output filename is `{save_basename}_member{NNN}_y{YYYY}.nc`,
one file per (member × year) the rollout actually crosses.**

Plus `long_inference.py:367`:
```python
steps_per_year = [... for year in range(init.year, final.year)]
```

Empty when `init.year == final.year`. So `final.year` MUST be at
least `init.year + 1` for any rollout to occur.

**Implication.** Rollout horizon is always to a Jan-1 boundary in
year `≥ init.year + 1`. Sub-year horizons (e.g., 30 days from
mid-March) do NOT work out of the box and are **NOT in this plan's
scope** — see §6 for future-work options.

### 2.2 val_year_start vs init_datetime.year decoupling (Blocker 2)

Three independent year fields drive different lookups:

| Field | Used at | Drives |
|---|---|---|
| `params.val_year_start` | `data_loader_multifiles.py:741` | constant-boundary file (`<dir>/<val_year_start>_0000.h5`, loaded once) |
| `params.leap_year, params.no_leap_year` | `data_loader_multifiles.py:932-934` | varying-boundary file template (`<dir>/<template>_<idx>.h5` per 6h step) |
| `params.init_datetime.year, .final_datetime.year` | `long_inference.py:190, 199-202` | rollout clock + BCS dataset year range |

The BCS dataset (`long_inference.py:199-202`) is built with
`year_start=init.year, year_end=final.year` — independent of
`val_year_start`. The branch at `data_loader_multifiles.py:950-958`
explicitly handles `start_date.year != val_year_start` by
substituting `leap_year`/`no_leap_year` as the boundary-file template
year. So setting all three to a sim52 year (e.g., 121) while
`init_datetime.year` is anything else (1 for ensemble, 125 for
deterministic) is supported by upstream.

**Decoupling rule.** In our wrapper:
- `val_year_start = val_year_end = leap_year = no_leap_year = boundary_template_year`
- `init_datetime`, `final_datetime` track the user's IC + horizon
  independently.

`val_year_end` is set to `val_year_start + 1` rather than
`val_year_start` to avoid the empty-range edge case at
`data_loader_multifiles.py:845-846` for the ensemble (IC) dataset
(`get_data_loader(... year_start=val_year_start, year_end=val_year_end)`
at `long_inference.py:192-195`). The constant-boundary load reads
year `val_year_start` only, so the +1 has no semantic effect.

### 2.3 BYO HDF5 contract (Blocker 3)

`data_loader_multifiles.py:539-560` loads training mean/std files
from `data_dir`:
```python
self.surface_mean, self.surface_std = self.load_mean_std(
    join(data_dir, params.surface_mean), join(data_dir, params.surface_std), ...)
self.upper_air_mean, ... = ... join(data_dir, params.upper_air_mean) ...
self.varying_boundary_mean, ... = ... join(data_dir, params.boundary_mean) ...
```

So `data_dir` is **simultaneously** the boundary-h5 root and the
mean/std root. A real BYO tree needs both, OR a staging step that
symlinks the sim52 mean/std files into the user tree.

`data_loader_multifiles.py:935`: varying-boundary h5 file is read via
`get_data_given_path(file_path, self.varying_boundary_variables)` —
expects `input/<var>` keys for each varying boundary variable. The
constant-boundary read at `:742` uses the same per-file pattern with
`constant_boundary_variables`.

**BYO HDF5 contract** (a first-class section in the user guide):

| Asset | Required path | Required content |
|---|---|---|
| varying-boundary h5 | `<data_dir>/<template_year>_<idx:04d>.h5` for `idx ∈ [0, steps_per_year)` | h5 keys `input/sst`, `input/rsdt`, `input/sic` (or whatever the yaml's `varying_boundary_variables` says); shape `(64, 128)` per key |
| constant-boundary h5 | `<data_dir>/<template_year>_0000.h5` (same file as `idx=0`, also serves constant load) | h5 keys `input/lsm`, `input/sg`, `input/z0` (same shape) |
| training mean/std | `<data_dir>/<surface_mean>`, `<data_dir>/<surface_std>`, `<data_dir>/<upper_air_mean>`, `<data_dir>/<upper_air_std>`, `<data_dir>/<surface_ff_std>`, `<data_dir>/<upper_air_ff_std>`, `<data_dir>/<boundary_mean>`, `<data_dir>/<boundary_std>`, `<data_dir>/<diagnostic_mean>`, `<data_dir>/<diagnostic_std>` | NetCDF files — basenames from upstream yaml; **must match training distribution** |
| bias dir | `<bias_data_dir>/<var>_bias{hour_str}.npy` etc. (`data_loader_multifiles.py:700-717`) | per-variable bias files (rarely user-supplied; usually staged from sim52) |
| climatology | `<climatology_file>` | NetCDF (used only for some scoring paths; not strictly required for raw inference) |

**Recommended BYO usage.** User supplies only the boundary h5 tree;
bias dir, climatology, and mean/std files come from sim52 via
**staging symlinks** under her `data_dir`. The wrapper auto-creates
those symlinks if `--boundary-mode user-h5` is selected and the
sim52 stats files are present at the expected absolute paths.

### 2.4 Ensemble seeding + Feb-29 trap (Blocker 4)

`long_inference.py:206`:
```python
self.perturber = Perturber(self.params, self.dataset, device=self.device,
    device_idx=self.world_rank, seed=self.params.run_iter * self.params.world_size)
```

`utils/perturbation.py:13`:
```python
self.generator.manual_seed(seed + device_idx)
```

`long_inference.py:1196`:
```python
total_run_iter = (self.params.run_iter - 1) * (num_members * len(init_nc_filepaths)) + ...
filename = save_basename + f'_member{total_run_iter:03}_y{current_year:04}.nc'
```

**Within-run member diversity** (single sbatch, `num_ensemble_members=N`):
all members share the Perturber's `torch.Generator`; consecutive
draws are independent → distinct perturbations. Works out of the box.

**Cross-run reseeding** (multiple sbatches, fresh ensembles): only
lever upstream exposes is `run_iter`. Increasing `run_iter` from 1 to
2 changes the seed AND shifts `total_run_iter` so filenames become
`_member{N..2N-1}` instead of `_member{0..N-1}`.

**Wrapper policy.**
- **Don't expose `run_iter`** to the user. Default 1.
- **Per-run unique output dir is mandatory** (preflight checks the
  output dir is empty; refuses to overwrite). This avoids member
  collision and makes filename ranges predictable (`_member000` to
  `_member{N-1}`).
- **For multi-run ensembles** (e.g., 100 members across 10 runs),
  document the run_iter knob with a worked example showing how
  filenames span across runs.
- A real `perturbation_seed` yaml key would be cleaner; add to
  "future work" §6.

**Feb 29 trap.** `cftime.DatetimeProlepticGregorian(1, 2, 29, ...)`
raises because year 0001 is not leap in proleptic_gregorian (no leap
year before 0004). For ensemble mode (forced year 0001) with a Feb
29 IC, the relabel step fails. **Wrapper policy:** raise a clear
error in `relabel_for_ensemble` ("Feb 29 ICs not supported for
ensemble mode; use Feb 28 or Mar 1, or use deterministic mode with
the original IC date"). Deterministic mode is unaffected.

## 3. Design

### 3.1 Revised module: `src/sfno_inference_5410/user_inference.py`

Existing draft on disk needs three changes:

**Change A.** Remove `init_year`, `final_year` parameters from
`build_user_yaml`. Add `boundary_template_year` as the single source
of truth for `val_year_start = val_year_end - 1 = leap_year =
no_leap_year`. The model clock (`init_datetime`, `final_datetime`)
is passed via the upstream CLI, not the yaml — so the yaml builder
no longer needs init_year/final_year at all.

**Change B.** Rename the leap-template validator to validate the
**rollout boundary year(s)** vs `boundary_template_year`'s leap-ness:
- Compute the set of years the rollout will cross (`init.year`,
  `init.year+1`, …, `final.year - 1`).
- For each such year, check `is_leap_proleptic(y)` matches
  `boundary_template_year in {124, 128}` (the sim52 leap years), or
  raise.

**Change C.** Drop `_is_leap_proleptic`'s init_year-only check; new
checker walks the rollout year range.

Public surface (revised):

```python
build_user_yaml(
    *,
    out_dir: Path,
    exp_dir: Path,
    boundary_template_year: int,           # 121..128 if sim52, else BYO
    rollout_years_crossed: tuple[int, ...], # for leap-template validation
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

`_override_section_user` sets:
```python
section["data_dir"] = boundary_data_dir
section["bias_data_dir"] = bias_data_dir
section["climatology_file"] = climatology_file
section["val_year_start"] = boundary_template_year
section["val_year_end"]   = boundary_template_year + 1
section["leap_year"]      = boundary_template_year
section["no_leap_year"]   = boundary_template_year
section["epsilon_factor"] = epsilon_factor
section["num_ensemble_members"] = num_ensemble_members
section["ensemble_members_per_pred"] = num_ensemble_members
section["save_basenames"] = [save_basename]
section["save_forecasts"] = True
section["log_to_wandb"]   = False
if epsilon_factor > 0:
    section["perturbation_type"] = perturbation_type
# ensemble_inference_hours: still needed by data_loader_multifiles.py:483, 597, 831
# Set to a generous overestimate (max steps the chunk allocator could
# request); upstream uses this to size internal arrays, not to bound
# the rollout. Use full sim52 year (8784) — safe because chunks are
# sized per-year separately.
section["ensemble_inference_hours"] = 8784
```

### 3.2 New module: `src/sfno_inference_5410/preflight.py`

Pure-Python validator, no GPU, runs on login or compute node before
launching upstream. Codex called for this in the design-changes
section.

```python
def preflight(
    *,
    init_nc: Path,
    output_dir: Path,
    boundary_mode: Literal["sim52", "user-h5"],
    boundary_template_year: int,
    boundary_data_dir: Path,
    init_datetime: cftime.DatetimeProlepticGregorian,
    final_datetime: cftime.DatetimeProlepticGregorian,
    epsilon_factor: float,
    num_ensemble_members: int,
    perturbation_type: Optional[str],
) -> None:
    """Raise PreflightError on any failure; return None on success."""
```

Checks (each is a separate helper, each cites the upstream contract
it enforces):

| Check | Upstream contract | What we verify |
|---|---|---|
| ckpt shim path resolvable to upstream `ckpt_epoch_50.tar` | `long_inference.py:expDir` | `realpath(shim) == UPSTREAM_CKPT_PATH` |
| IC NC variable schema | `data_loader_multifiles.py:83-159` (`get_data_given_path_nc`) | required vars, dims, calendar, time units |
| IC NC sigma/plev levels match yaml | `data_loader_multifiles.py:138-150` (level_delta=1e-4 tolerance) | levels within tolerance |
| boundary tree coverage | `data_loader_multifiles.py:927-935` | files exist for every 6h step from init to final |
| mean/std files present | `data_loader_multifiles.py:539-560` | each file at `<data_dir>/<basename>` exists |
| bias dir present | `data_loader_multifiles.py:700-717` | `<bias_data_dir>` is a directory |
| no `/glade/` paths in generated yaml | wrapper invariant | grep |
| output dir is fresh | wrapper policy (filename collision) | empty or non-existent |
| ensemble + epsilon=0 contradiction | wrapper policy | num_members > 1 ⟹ epsilon > 0 |
| Feb 29 + ensemble | upstream year-1 gate (§2.4) | refuse |
| leap-template ↔ rollout-year compatibility | §2.2 | every crossed year has same leap-ness as template |
| same-year init/final | `long_inference.py:367` | `final.year > init.year` |

### 3.3 New module: `scripts/run_sfno_5410_inference.py`

User-facing CLI. Flags:

| Flag | Required | Default | Notes |
|---|---|---|---|
| `--init-nc PATH` | yes | — | her IC NetCDF |
| `--end-year YYYY` | yes | — | rollout ends at Jan 1 of this year (must be > init's year) |
| `--output-dir DIR` | yes | — | per-run output root, must be fresh |
| `--save-basename NAME` | no | `inference` | drives output filename |
| `--epsilon-factor F` | no | 0.0 | 0 = deterministic, >0 = ensemble |
| `--perturbation-type T` | iff ε>0 | — | gaussian_noise / gaussian_noise_n_minus_1 / perlin_noise |
| `--num-members N` | no | 1 | ensemble size; >1 implies ε>0 |
| `--boundary-mode M` | no | `sim52` | explicit `{sim52, user-h5}` (Codex-requested) |
| `--boundary-template-year Y` | no | 121 | sim52 boundary year template (or BYO year key) |
| `--boundary-data-dir DIR` | no | sim52 path | BYO h5 tree (option B) |
| `--bias-data-dir DIR` | no | sim52 path | BYO bias dir |
| `--climatology-file PATH` | no | sim52 path | BYO climatology |
| `--init-datetime YYYY-MM-DD_HH:MM:SS` | no | from IC NC | override IC datetime |
| `--dry-run` | no | False | print upstream argv without launching |

Logic, end-to-end:
1. Parse CLI; read IC NC time coord if `--init-datetime` not given.
2. Compute `final_datetime = (end_year)-01-01_00:00:00`.
3. **Same-year guard**: if `final_datetime.year <= init_datetime.year`, error.
4. **Ensemble pre-checks**:
   - `num_members > 1` ⟹ `epsilon_factor > 0` (else error).
   - `epsilon_factor > 0` ⟹ `perturbation_type` set and valid.
   - `epsilon_factor > 0` ⟹ relabel IC datetime to year 0001
     (raise on Feb 29).
   - `epsilon_factor > 0` ⟹ also relabel `final_datetime` accordingly:
     `final_datetime = (year=1 + (orig.final.year - orig.init.year)).Jan-1`
     (preserves rollout duration).
5. **BYO staging** (if `--boundary-mode user-h5`):
   - Verify required mean/std + bias files at expected sim52 paths.
   - Create `<output-dir>/staged_data/` directory.
   - Symlink boundary h5 tree files from `<--boundary-data-dir>` and
     mean/std + bias files from sim52 paths into `staged_data/`.
   - Set `data_dir = <output-dir>/staged_data/`.
6. Build yaml via `build_user_yaml(...)`.
7. Build ckpt shim under `<output-dir>/SFNO/5410/checkpoints/`.
8. Run `preflight(...)`. Raise on any failure.
9. Construct upstream argv; `subprocess.run(argv, cwd=upstream_repo,
   env=os.environ)`.
10. Postflight: count output NCs (expect `num_members × (final.year - init.year)`).

### 3.4 New SLURM template: `scripts/submit_sfno_5410_user_inference.slurm`

H100, 1 node, 1 GPU. The user edits the top-of-file knobs (`INIT_NC`,
`OUTPUT_DIR`, `END_YEAR`, optional ensemble flags). Default
wallclock 4h covers up to 2-year rollouts comfortably.

### 3.5 New tests: `tests/sfno_inference_5410/test_user_inference.py`

Tier-1-style allowlist + pinned-value test for the user yaml builder.
Two scenarios:
- Deterministic (epsilon=0, num_members=1, boundary_mode=sim52,
  boundary_template_year=121).
- Ensemble (epsilon=1e-3, num_members=4, gaussian_noise,
  boundary_mode=sim52, boundary_template_year=121, IC dated 0001-06-15).

Asserts: yaml has correct `data_dir`, `val_year_start`,
`leap_year`, etc.; perturbation_type present iff epsilon>0;
num_ensemble_members and ensemble_members_per_pred match. Login-node-cheap.

Plus a fast preflight unit test exercising each helper independently
with monkeypatched filesystem.

### 3.6 New doc: `docs/2026-05-08_sfno_5410_external_user_guide.md`

Audience: first-time user. Sections:

1. **Prerequisites** — TACC account, allocation, group G-819272 confirmation.
2. **One-time setup** — clone repo from GitHub, venv, verify access.
3. **Asset map** — explicit absolute paths to all shared assets (with `ls -la` showing group readability).
4. **Input NetCDF schema** — variables, dims, units, calendar; worked example xarray + netCDF4 IC NC writer.
5. **Boundary mode selection** —
   - **sim52 (default)**: `--boundary-mode sim52`, picks `--boundary-template-year` (caveats on leap-template + rollout-year compatibility).
   - **user-h5**: `--boundary-mode user-h5`, full schema spec from §2.3 BYO contract, staging-symlink behavior described.
6. **Rollout horizon model** — explicit "final must be Jan 1 of year > init's year"; multi-year examples; sub-year workaround (run full year + post-trim).
7. **Deterministic walkthrough** — copy-pastable example.
8. **Perturbation-ensemble walkthrough** — explanation of:
   - Year-0001 relabel (with citation to upstream year==1 gate).
   - Fresh-output-dir requirement (filename collision avoidance).
   - Within-run vs cross-run seeding (single sbatch ⟹ all members
     differ; cross sbatches ⟹ same seed by default).
   - Feb 29 limitation.
9. **Output schema** — `{save_basename}_member{NNN}_y{YYYY}.nc`, 8 vars, dims, units (group conventions: `pl=ln(p_s)`, `zg=gpm`, `pr_6h=rate×6h`, **must NOT be re-converted**).
10. **Common failures** — exhaustive table:
    - `IndexError on inference_idxs` / no output → init.year == final.year (set `--end-year` larger)
    - `FileNotFoundError: 1_0000.h5` → val_year_start leaked from init.year (shouldn't happen with our wrapper, but flag for upstream resync drift)
    - `FileNotFoundError: <year>_<idx>.h5` for indices beyond tree → leap/non-leap template mismatch with rollout year
    - `cftime: invalid date Feb 29 year 1` → ensemble + Feb 29 IC
    - `Members all identical despite epsilon>0` → init NC retains real-world year (not relabelled); the wrapper handles this but flag in case user bypasses
    - `Permission denied` → `groups | grep G-819272` to verify
11. **Verification** — short smoke (1-year, deterministic, IC = sim52 year 121 IC if she has it).
12. **Contact / support**.

### 3.7 Permissions widening (path-specific per Codex)

| Path | Asset | Current | Proposed |
|---|---|---|---|
| `/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator` | parent | 700 | 750 |
| `/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0` | upstream code (read-only) | 755 already | `g+rX -R` |
| `/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0/long_inference.py` | upstream entry point | 644 already | `g+r` (no-op verify) |
| `/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0/results/SFNO/5410/checkpoints/ckpt_epoch_50.tar` | the 5410 checkpoint | 644 already | `g+r` (no-op verify) |
| `/scratch/11114/zhixingliu/SFNO_Climate_Emulator` | parent | 710 | 750 |
| `/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/h5/sigma_data` | sim52 boundary tree + mean/std + bias | mixed | `g+rX -R` |
| `/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/bias` | bias_data_dir | mixed | `g+rX -R` |
| `/scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52/sigma_data/climatology.nc` | climatology | mixed | `g+r` |

Exact commands (NOT yet run; deferred until plan sign-off):

```bash
chmod g+rX  /work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator
chmod -R g+rX /work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0
chmod g+rX  /scratch/11114/zhixingliu/SFNO_Climate_Emulator
chmod -R g+rX /scratch/11114/zhixingliu/SFNO_Climate_Emulator/data/derecho_glade/sim52
```

`stat -c '%a %U:%G %n'` will be run before and after each command to
log the change.

**Repo (code)** is cloned from GitHub by the user; **no $HOME chmod
is performed** (privacy-preserving; the user's $HOME stays 700).

## 4. File list

| File | Status | Purpose |
|---|---|---|
| `src/sfno_inference_5410/user_inference.py` | **REVISE** existing draft per §3.1 | yaml builder + helpers |
| `src/sfno_inference_5410/preflight.py` | **NEW** | preflight validator (Codex-requested) |
| `scripts/run_sfno_5410_inference.py` | **NEW** | user-facing CLI orchestrator |
| `scripts/submit_sfno_5410_user_inference.slurm` | **NEW** | H100 SLURM template |
| `tests/sfno_inference_5410/test_user_inference.py` | **NEW** | Tier-1 yaml regression (login-node-cheap) |
| `tests/sfno_inference_5410/test_preflight.py` | **NEW** | preflight unit tests |
| `docs/2026-05-08_sfno_5410_external_user_guide.md` | **NEW** | the deliverable .md guide |

**Not modified:**
- `src/sfno_inference_5410/stampede3_yaml_override.py` (eval-track)
- `scripts/eval_inference_5410.py`, `build_5410_yaml_override.py`, the eval SLURMs
- Upstream `PanguWeather/v2.0/` (kept pristine — confirmed no upstream patches needed)

## 5. Verification plan (post-implementation)

1. **Unit tests** — `pytest tests/sfno_inference_5410/ -q` includes:
   - existing Tier 1, 2, 3 (regression for eval track unchanged)
   - new `test_user_inference.py` (yaml builder)
   - new `test_preflight.py` (validator helpers)
   - target wall: <30 s.
2. **Dry-run** — `scripts/run_sfno_5410_inference.py --dry-run` for both:
   - deterministic + sim52 boundaries
   - ensemble (4 members, gaussian_noise, eps=1e-3) + sim52 boundaries
   Assert the printed upstream argv is what we expect.
3. **Smoke deterministic on H100** — 1-year rollout, IC = our smoke
   IC (`121_0000.nc`) but with `--end-year 0122` from external CLI;
   compare to existing 5410 smoke output (should match
   bit-exactly except for any logging differences).
4. **Smoke ensemble on H100** — 1-year rollout, IC relabelled to
   year 0001, num_members=4, gaussian_noise, eps=1e-3. Verify:
   - 4 output NCs written: `*_member000_y0001.nc` …
     `*_member003_y0001.nc`.
   - Members differ from each other (max abs diff > 0 on at least one
     surface var at the last timestep).
   - Member 0 differs from a deterministic baseline rollout
     (confirms the year-1 gate is satisfied — perturbation actually
     fired).
5. **BYO smoke** — symlink-stage sim52 mean/std into a fresh
   `<output-dir>/staged_data/`; point `--boundary-data-dir` at the
   sim52 boundary tree (still the same data); run the same
   1-year deterministic rollout. Confirm output matches sim52-mode
   smoke.
6. **Permission verification** — after applying chmod, run
   `sudo -u <some-other-G-819272-user> stat <each-asset-path>` if
   feasible, OR `setfacl --check`-equivalent via simple read tests
   from a different user account if available.
7. **Doc walkthrough** — read the .md guide cold, follow each step,
   confirm every command works without prior context.

## 6. Future work (out of scope for this iteration)

- **Sub-year horizon support.** Either (a) a small upstream patch
  to upstream's chunk allocator + save trigger to allow partial
  chunks ending at `final_datetime`, or (b) a wrapper that
  monkey-patches the same. Both invasive; defer until concrete user
  need.
- **Real `perturbation_seed` yaml key.** Decouple the seed from
  `run_iter` to allow cross-run reseeding without filename shifts.
  Small upstream patch.
- **Multi-IC ensemble.** Drop the single-IC invariant in
  `eval_inference_5410.py` and pass comma-separated
  `--init_nc_filepaths`. Out of user request.
- **Migrate sim52 boundary tree from $SCRATCH to $WORK** to avoid
  purge. Cost: a few GB transfer. Open question.
- **Drift detector** for new yaml keys read by upstream — already
  covered by Tier 3 of the existing 5410 regression net; just need
  to verify the user-yaml path triggers it on the same path.

## 7. Risks (revised)

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | Year-0001 relabel hits an undocumented upstream code path that breaks for non-sim52 init years | medium | Smoke ensemble verification (§5.4) catches this; `data_loader_multifiles.py:950-958` branch is the explicit substitution path |
| 2 | $SCRATCH purges sim52 tree if untouched 10+ days | low | Document; revisit migration to $WORK |
| 3 | Leap/non-leap template mismatch on multi-year rollout crossing both leap and non-leap years | medium | Preflight validates every crossed year; raises on mismatch |
| 4 | Output dir collision if user re-runs same `--save-basename` to same dir | low | Preflight refuses non-empty output dir |
| 5 | BYO mean/std staging produces silently-incorrect normalization (different training distribution) | high if user is not careful | Doc explicitly warns: BYO mean/std is only safe for finetuned/retrained models; for SFNO-5410 inference, ALWAYS stage from sim52 |
| 6 | User passes `--boundary-mode user-h5` but `--boundary-data-dir` points at incomplete tree | medium | Preflight checks coverage for every 6h step in rollout span |
| 7 | Permission widening exposes more than needed | low | Per-path scope in §3.7 (Codex's request); no $HOME chmod; checkpoints already 644 |

## 8. Open questions (deferred to second-round Codex review)

1. **Should the wrapper auto-stage mean/std for BYO mode**, or
   require the user to do it explicitly? (Auto-stage is more
   beginner-friendly but hides the assumption that her model expects
   sim52-distribution stats.) **Recommendation**: explicit step in
   the doc, no auto-staging — make the assumption visible.
2. **Should `--init-datetime` default to reading from IC NC time
   coord, or require explicit flag?** **Recommendation**: default
   to IC NC, allow override flag.
3. **`--end-year` vs `--horizon-years` as the user-facing name?**
   `--end-year` is more honest about the upstream year-boundary
   constraint; `--horizon-years` is more user-friendly but
   masks the constraint. **Recommendation**: `--end-year` plus a
   `--horizon-years` convenience alias that errors if init isn't
   Jan 1.
4. **Should the wrapper write a manifest** (input args, generated
   yaml, ckpt sha, git rev) into the output dir for reproducibility?
   **Recommendation**: yes, cheap and useful. `<output-dir>/manifest.json`.
5. **Should the user guide include a "schema validator" script**
   she can run on her IC NC before launching to catch schema bugs
   without burning a SLURM job? **Recommendation**: yes — expose
   `preflight.py` as a standalone CLI: `python -m
   sfno_inference_5410.preflight --init-nc <path>`.

## 9. Sequence of work after Codex sign-off

1. Revise `user_inference.py` per §3.1 (decouple yaml-year fields
   from model-clock fields).
2. Implement `preflight.py` with helpers for each check in §3.2.
3. Implement `run_sfno_5410_inference.py` per §3.3.
4. Implement SLURM template per §3.4.
5. Implement tests per §3.5; run `pytest`.
6. Verification step §5.1, §5.2 (login-node-cheap).
7. Apply permissions per §3.7; log before/after stat.
8. Verification step §5.3, §5.4, §5.5 (H100 SLURM jobs).
9. Write user guide per §3.6, walking through it cold.
10. Hand to user for delivery to colleague.
