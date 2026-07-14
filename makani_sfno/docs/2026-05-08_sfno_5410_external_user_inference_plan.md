# Plan: SFNO-5410 external-user inference path on Stampede3

> **Status: pre-implementation, drafted for Codex review.**
> Date: 2026-05-08. Author: zhixingliu (via Claude). Audience: Codex
> reviewer + a group member (the eventual end user).
>
> This plan proposes new user-facing inference code (`scripts/` +
> `src/sfno_inference_5410/`), permission widening on shared assets,
> and a beginner-friendly `.md` guide. It does **not** modify upstream
> PanguWeather code or our existing 5410 eval-track code.
>
> A first-cut draft of the new yaml-override module exists at
> `src/sfno_inference_5410/user_inference.py` and is referenced
> throughout this plan. It is **pre-review** and may change after
> Codex feedback.

---

## 1. Context

A group member will run SFNO-5410 inference on Stampede3 with **her
own initial-condition NetCDF**. She does NOT use the sim52 test years
(0121–0128) and does NOT use our 96-IC evaluation pipeline. Her use
case is general inference: arbitrary IC, arbitrary forecast horizon
in days, with optional perturbation ensemble. Audience for the guide:
"first-time user of our setup, but comfortable on a Linux cluster".

The existing `_override_section` in
`src/sfno_inference_5410/stampede3_yaml_override.py:90-144` pins three
values that block this use case:

| Pinned value | Eval purpose | Blocks |
|---|---|---|
| `epsilon_factor = 0.0` | deterministic 96-IC NWP eval | perturbation ensemble |
| `save_basenames = ["_unused_len1"]` | length-1 placeholder for single-IC invariant | named outputs (`{save_basename}_member{NNN}_y{YYYY}.nc`) |
| `ensemble_inference_hours = 8760/8784` | full sim52 year | arbitrary horizon |

Plus path defaults (`data_dir`, `bias_data_dir`, `climatology_file`,
`leap_year`, `no_leap_year`) point at our sim52 boundary tree, which
the user may want to override (option B below).

## 2. User decisions locked in (2026-05-08 via AskUserQuestion)

| Question | Choice |
|---|---|
| Boundary-forcing strategy | **Both** — sim52 default + BYO h5 tree as advanced option |
| Ensemble flavors to support | **Deterministic single-IC** + **perturbation ensemble around single IC** (multi-IC NOT in scope) |
| Permission scope | **Group-readable on G-819272** (the user's allocation) |

## 3. Upstream audit (key findings, with file:line)

Verified against
`/work2/.../v2.0/long_inference.py` and
`/work2/.../v2.0/utils/{perturbation,data_loader_multifiles}.py`:

1. **CLI surface** (`long_inference.py:1217-1234`): only required flag
   is `--init_nc_filepaths`. Datetime format is
   `"%Y-%m-%d_%H:%M:%S"` parsed with
   `cftime.datetime.strptime(..., calendar=params.calendar,
   has_year_zero=True)` at `:1318`.

2. **Output filename pattern** (`long_inference.py:1206`):
   `save_basename + f'_member{total_run_iter:03}_y{current_year:04}.nc'`
   where
   `total_run_iter = (run_iter - 1) * (num_members * len(init_nc_filepaths)) + num_members * particle_idx + ensemble_member`.
   For single IC, `run_iter=1`, `particle_idx=0`: filename ranges
   `_member000_…` to `_member{num_members-1:03}_…`. **One file per
   ensemble member per data year touched by the rollout.**

3. **Ensemble fan-out** (`long_inference.py:1265-1274`): yaml key
   `num_ensemble_members` (default 1) drives the rollout fan-out.
   `ensemble_members_per_pred` defaults to `num_ensemble_members`.
   `len(save_basenames)` plays no role; it only sizes a date_range
   array at `data_loader_multifiles.py:829`.

4. **Perturbation gate** (`long_inference.py:558, 830`):
   `if self.params.epsilon_factor > 0. and self.params.init_datetime.year == 1:`
   — **perturbation is only actually applied when the IC's absolute
   year is 0001.** `Perturber` is constructed for any
   `epsilon_factor > 0` (`:204`) but the `add_perturbations()` call
   sites at `:560` and `:832` are gated by the year-1 check. Without
   this workaround, a non-year-1 IC + `epsilon_factor > 0` produces
   a fully deterministic rollout silently.

5. **Boundary path construction**
   (`data_loader_multifiles.py:927-935`):
   `data_year_template = leap_year if data_is_leap else no_leap_year`,
   then `<boundary_data_dir>/<template_year>_<idx:04d>.h5`. The
   absolute year of the IC drives `data_idx` via
   `(data_dt - Jan 1 of data_year) / 6h`, but the *file-path year* is
   the template. So an IC dated year 0001 reading from
   sim52 boundary tree (`121_*.h5`) is fine **as long as
   leap_year=121, no_leap_year=121**.

6. **Boundary forcing schema** (yaml `5410.yaml:39-40`):
   - constant vars: `lsm`, `sg`, `z0` → `(3, 64, 128)`
   - varying vars: `sst`, `rsdt`, `sic` → `(N_steps, 3, 64, 128)`
   per-year file count: 1460 (non-leap) / 1464 (leap 124, 128).

7. **Calendar**: hardcoded `proleptic_gregorian` in upstream yaml.
   `has_year_zero=True`. Model treats absolute year as opaque;
   only day-of-year matters for SST seasonality.

8. **No `/glade/` leaks in our wrapper**: only two hits in
   `stampede3_yaml_override.py` (lines 9, 43) — both in docstrings
   describing the remap, not in code. Confirmed via grep.

## 4. Design

### 4.1 New file: `src/sfno_inference_5410/user_inference.py` (DRAFT EXISTS)

Companion to `stampede3_yaml_override.py`, reusing path constants
(`STAMPEDE3_DATA_DIR`, etc.) + checkpoint-shim builder. **Does not
touch existing eval code paths.**

Public surface:

```python
build_user_yaml(
    *,
    out_dir: Path,                        # where the yaml lands
    exp_dir: Path,                        # drives upstream ckpt discovery
    init_year: int,                       # IC absolute year (1 if ensemble)
    final_year: int,                      # init_year for sub-year horizons
    boundary_template_year: int,          # 121..128 if using sim52, else BYO
    horizon_hours: int,                   # rollout length, drives ensemble_inference_hours
    epsilon_factor: float,                # 0 = deterministic, >0 = ensemble
    num_ensemble_members: int,            # >=1
    save_basename: str,                   # used in output filename
    perturbation_type: Optional[str] = None,    # required iff epsilon_factor > 0
    boundary_data_dir: Path = STAMPEDE3_DATA_DIR,
    bias_data_dir: Path = STAMPEDE3_BIAS_DIR,
    climatology_file: Path = STAMPEDE3_CLIM_NC,
    src_yaml: Path = UPSTREAM_YAML_PATH,
) -> Path
```

Internal helper `_override_section_user` mirrors `_override_section`
but **does not pin epsilon/save_basenames/ensemble_inference_hours**.

Validates leap-template vs init-year compatibility (raises if user
picks template 124/128 with non-leap init year, or vice versa over a
sub-year rollout that crosses Feb 29).

Helpers also exposed: `derive_init_final(...)`, `relabel_for_ensemble(init_dt)`,
`datetime_str(init_dt)`, `VALID_PERTURBATION_TYPES`,
`ENSEMBLE_FORCED_INIT_YEAR=1`.

### 4.2 New file: `scripts/run_sfno_5410_inference.py`

User-facing CLI. Flags:

| Flag | Required | Default | Purpose |
|---|---|---|---|
| `--init-nc PATH` | yes | — | her IC NetCDF |
| `--horizon-days N` | yes | — | rollout length |
| `--output-dir DIR` | yes | — | per-run output root |
| `--save-basename NAME` | no | `inference` | drives output filename |
| `--epsilon-factor F` | no | 0.0 | 0 = deterministic, >0 = ensemble |
| `--perturbation-type T` | iff ε>0 | — | gaussian_noise / gaussian_noise_n_minus_1 / perlin_noise |
| `--num-members N` | no | 1 | ensemble size; >1 implies ε>0 |
| `--boundary-template-year Y` | no | 121 | sim52 boundary year template |
| `--boundary-data-dir DIR` | no | sim52 path | BYO h5 tree (option B) |
| `--bias-data-dir DIR` | no | sim52 path | BYO bias dir |
| `--climatology-file PATH` | no | sim52 path | BYO climatology |
| `--seed S` | no | 0 | passed via env var or yaml — TBD during implementation, see §6 |
| `--init-datetime YYYY-MM-DD_HH:MM:SS` | no | read from IC NC time coord | override IC datetime |
| `--dry-run` | no | False | print upstream argv without launching |

Logic:
1. If `--num-members > 1` and `--epsilon-factor == 0`: **error** (no
   point in N=2 deterministic rollouts).
2. If `epsilon_factor > 0`:
   - **Relabel IC to year 0001** (preserves day-of-year + sub-day
     position; writes a temporary copy of the IC NC into
     `<output-dir>/ic_relabelled.nc` rather than mutating the
     user-provided file). Logs the relabel for transparency.
   - Pass `init_datetime` with `year=1` to upstream.
3. Compute `final_datetime = init_datetime + horizon_days × 24h`.
4. Build yaml via `build_user_yaml(...)`.
5. Build ckpt symlink shim under `<output-dir>/SFNO/5410/checkpoints/`.
6. Construct upstream argv (mirrors `eval_inference_5410.py:build_argv_for_ic`):
   ```
   python -u <upstream>/long_inference.py
       --run_num 5410 --yaml_config <yaml> --config SFNO
       --init_datetime <s> --final_datetime <s>
       --init_nc_filepaths <ic_nc>
       --output_dir <output-dir> --save_basename <save-basename>
       --async_save
   ```
7. `subprocess.run(argv, cwd=<upstream>, env=os.environ)` — same
   pattern as the eval orchestrator.
8. Postflight: count expected output files = `num_members × (final_year - init_year + 1)` (or `× 1` for sub-year horizons).

### 4.3 New file: `scripts/submit_sfno_5410_user_inference.slurm`

H100 SLURM template. The user fills in `INIT_NC`, `OUTPUT_DIR`,
`HORIZON_DAYS`, etc. at the top, then `sbatch`s. Sets
`WORLD_SIZE=1, RANK=0, LOCAL_RANK=0, MASTER_ADDR=localhost,
MASTER_PORT=29500`, activates her venv, calls
`run_sfno_5410_inference.py`. Wallclock default: **1h** (covers up
to ~30-day rollout; longer horizons will need more).

### 4.4 New doc: `docs/2026-05-08_sfno_5410_external_user_guide.md`

Audience: **first-time user**. Sections:

1. **Prerequisites** — TACC account on ALCC allocation MTH240094,
   group G-819272 confirmed via `groups | grep G-819272`.
2. **One-time setup** —
   2.1 Clone repo from GitHub (`git clone git@github.com:feynmanliu214/AI-RES-Stampede3.git`).
   2.2 Create venv: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
   2.3 Verify access to shared assets via `ls` on each path.
3. **Asset map** — explicit absolute paths to:
   - upstream PanguWeather repo (`/work2/.../v2.0/`)
   - SFNO-5410 checkpoint
   - sim52 boundary tree (option A)
   - climatology + bias + mean/std files
   - our wrapper (cloned in step 2.1)
4. **Input NetCDF schema** — table of variables + dims + units, time
   coord conventions, `proleptic_gregorian` calendar requirement.
   Worked example: build a 1-timestep IC NC from xarray, written by
   netCDF4 with `calendar="proleptic_gregorian"`.
5. **Boundary strategy** —
   - **Option A (sim52, default):** pick `--boundary-template-year` ∈ {121..128}; mismatched leap behavior caveat.
   - **Option B (BYO h5 tree):** schema spec for h5 keys, file-naming convention `<year>_<idx:04d>.h5`, exact varying/constant variable list.
6. **Deterministic walkthrough** — `epsilon_factor=0`, `num_members=1`, copy-pastable example with her IC dated 0125-06-15.
7. **Perturbation-ensemble walkthrough** — explanation of the
   year-0001-relabel workaround (with citation back to `long_inference.py:558,830`), copy-pastable example with `num_members=10, epsilon=1e-3, perturbation_type=gaussian_noise`.
8. **Output schema** — file naming `{save_basename}_member{NNN}_y{YYYY}.nc`, variables (8 total: pl, tas, pr_6h, ta, ua, va, hus, zg), dims, units (NOTE: `pl=ln(p_s)`, `zg=gpm`, `pr_6h=rate×6h` — group conventions, **must NOT be re-converted**).
9. **Rollout-length knob** — `--horizon-days`; how to extend across multiple years; SLURM wallclock guidance.
10. **Common failures** —
    - `KeyError: WORLD_SIZE` → set env in SLURM
    - `FileNotFoundError: <template>_<idx>.h5` → leap/non-leap mismatch or template year not in tree
    - `AttributeError: 'YParams' object has no attribute X` → upstream resync; recheck Tier-1 regression test
    - Silent deterministic-only rollout despite `epsilon_factor>0` → forgot to relabel IC to year 0001
    - Permission denied on `/work2/...` or `/scratch/...` → check `groups` output
11. **Verification** — smoke walkthrough she can run: 1-day rollout, deterministic, IC = our smoke IC if she has access (or her own).
12. **Contact / support** — mention this is research code; report issues via GitHub.

### 4.5 Permission widening (chmod / setfacl)

Goal: G-819272 group can read all assets she needs without exposing
private $HOME content.

**Repo (code)**: she clones from GitHub. **No chmod needed on $HOME.**

**Upstream + checkpoints + boundary tree:**

| Path | Current | Proposed | Reason |
|---|---|---|---|
| `/work2/.../stampede3/AI-RES` | 700 | 750 | unblock group traversal |
| `/work2/.../stampede3/AI-RES/artifacts` (recursive) | 755 mostly | `g+rX -R` | upstream + checkpoints |
| `/scratch/.../AI-RES` | 710 | 750 | unblock group traversal |
| `/scratch/.../AI-RES/data` (recursive) | mixed | `g+rX -R` | sim52 boundary tree + climatology + bias + mean/std |

Exact commands (proposed; **NOT yet run**):

```bash
chmod g+rX  /work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator
chmod -R g+rX /work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts
chmod g+rX  /scratch/11114/zhixingliu/SFNO_Climate_Emulator
chmod -R g+rX /scratch/11114/zhixingliu/SFNO_Climate_Emulator/data
```

`g+X` (capital) sets execute on directories only (not regular files),
so binary-files don't accidentally become group-executable.

**Risks of permission change:**
- Files/dirs with sensitive intermediate state (e.g., logs, ckpts in
  progress) become group-readable. Verified: `/work2/.../artifacts`
  contains only the upstream PanguWeather repo + checkpoints —
  nothing personal. `/scratch/.../data` contains the sim52 data tree
  — nothing personal.
- $SCRATCH purges files after ~10 days of no-access. Plan does NOT
  migrate boundary forcings off $SCRATCH; she should be aware that
  if she leaves the tree idle for >10 days it may be purged. **Open
  question: should we migrate to $WORK?** (See §7 risk 2.)

## 5. File list

| File | Status | Purpose |
|---|---|---|
| `src/sfno_inference_5410/user_inference.py` | **DRAFT EXISTS** (pre-review) | yaml builder + helpers for arbitrary-IC user inference |
| `scripts/run_sfno_5410_inference.py` | new (not yet drafted) | user-facing CLI orchestrator |
| `scripts/submit_sfno_5410_user_inference.slurm` | new (not yet drafted) | H100 SLURM template |
| `docs/2026-05-08_sfno_5410_external_user_guide.md` | new (not yet drafted) | beginner-friendly walkthrough |
| `tests/sfno_inference_5410/test_user_inference.py` | new (not yet drafted) | yaml builder regression test (mirrors Tier 1 contract for the user override) |
| `src/sfno_inference_5410/__init__.py` | possibly updated | re-export user_inference if needed |

**Not modified:**
- `src/sfno_inference_5410/stampede3_yaml_override.py` (eval-track)
- `scripts/eval_inference_5410.py` (eval-track orchestrator)
- `scripts/build_5410_yaml_override.py` (eval-track CLI)
- `scripts/submit_eval_inference_5410.slurm` / `submit_eval_inference_5410_smoke.slurm`
- Upstream PanguWeather/v2.0 (kept pristine)

## 6. Open questions / decisions deferred to Codex

1. **Seed control for ensemble** — `Perturber.__init__` reads `seed`
   as an explicit param (`utils/perturbation.py:8`). Upstream
   `long_inference.py:204` constructs `Perturber(self.params,
   self.device, ...)` — need to verify whether `seed` flows through
   yaml or env var. **Action**: re-grep `Perturber(...)` in
   upstream to confirm seed plumbing before final implementation.

2. **`--num-members > 1` with `epsilon_factor == 0` policy** — error
   out (proposed) or warn + run? **Recommendation**: error. Codex's
   call.

3. **`--init-datetime` precedence** — read from IC NC `time` coord
   by default vs require explicit flag. **Recommendation**: read
   from IC NC, allow override flag. Reduces user error.

4. **Year-0001 IC relabel — permanent or temporary file?**
   - Option A (proposed): write `<output-dir>/ic_relabelled.nc`
     fresh each run, deleted at end.
   - Option B: require user to provide year-0001-dated IC herself
     for ensemble.
   **Recommendation**: A (less footgun for the user).

5. **`run_iter` flag** — `long_inference.py:1196` uses `run_iter`
   for output filename collision avoidance across multiple sbatch
   re-runs. Default 1. Should the CLI expose this? **Recommendation:
   no** for now; can be added if needed.

6. **Migrate boundary tree to $WORK** to avoid $SCRATCH purge?
   **Open**. Cost: ~few GB transfer. Benefit: stable long-term
   access. **Recommendation**: defer; document the purge risk in
   the user guide; revisit if the user becomes a long-term user.

7. **Test coverage** — should we add a Tier-1-style regression test
   (`tests/sfno_inference_5410/test_user_inference.py`) mirroring
   `test_required_attrs.py` to lock in the contract? **Recommendation**:
   yes; cheap and catches future drift.

## 7. Risks

1. **Year-0001 relabel breaks something subtle.** The boundary loader
   uses `data_datetime.year` to compute Jan-1 reference for `data_idx`.
   For year 0001, that's `cftime.DatetimeProlepticGregorian(1, 1, 1, 0)`
   — needs `has_year_zero=True` to be valid. Verified upstream uses
   `has_year_zero=True` at `:1318`. **Risk: low**, but worth a
   sanity-check rollout in verification.

2. **$SCRATCH purge.** Boundary tree at `/scratch/.../data` may be
   purged if untouched for 10+ days. Mitigation: document; offer
   migration to $WORK as future work.

3. **Leap-year template mismatch.** User picks `boundary-template-year=124`
   (leap, 1464 indices) with init year 0001 (non-leap, 1460-step
   rollout). The loader looks up index 0..1459 — fine. Reverse case
   (template 121, init year 0124 leap): looks up 0..1463 but tree
   only has 0..1459. **Mitigation**: `build_user_yaml` raises on
   mismatch (see §4.1 validation).

4. **Output dir conflicts on re-run.** `total_run_iter` collision if
   user re-runs same `--save-basename` to same `--output-dir`.
   Upstream uses `run_iter` to disambiguate; we don't expose it
   yet. **Mitigation**: doc says "use a fresh `--output-dir` per
   run" or rotate `--save-basename`.

5. **Permission widening exposes more than needed.** Acceptable
   per the user's "group-readable" choice. World-readable was
   explicitly declined.

## 8. Verification plan (post-implementation)

1. **Tier 1 unit test** — `pytest tests/sfno_inference_5410/test_user_inference.py`
   confirms yaml renders correct values for both deterministic and
   ensemble modes; allowlist + pinned-value checks. <1s, login-node.

2. **Dry-run** — `scripts/run_sfno_5410_inference.py --dry-run` with
   her IC + 1-day horizon, confirm argv matches expected pattern.

3. **Smoke (deterministic)** — 1-day deterministic rollout, single
   IC = our existing smoke IC, confirm output schema matches §B.4
   expectations (8 vars, dims, calendar).

4. **Smoke (ensemble)** — 1-day ensemble rollout, num_members=4,
   epsilon=1e-3, gaussian_noise, year-0001-relabelled IC. Confirm:
   - 4 output files written (`*_member000_y0001.nc` … `*_member003_y0001.nc`)
   - members differ from each other (assert max abs diff > 0 on at
     least one variable)
   - member 0 NOT identical to deterministic baseline (sanity check
     that perturbation actually fired — i.e., the year-1 gate is
     properly satisfied)

5. **Permission verification** — `sudo -u <some-other-G-819272-user>
   stat <each-asset-path>` (or `setfacl --check` equivalent) to
   confirm group-read works.

6. **Doc walk-through** — read the .md file as if first-time user,
   confirm every step is concrete (paths absolute, env vars
   defined, no ambiguity).

## 9. Out of scope

- Multi-IC ensemble (user explicitly excluded; would require dropping
  the single-IC invariant in `eval_inference_5410.py` and is not
  needed for this use case).
- Modifying upstream PanguWeather/v2.0 to remove the year-1 gate
  (workaround via IC relabelling avoids this).
- Migrating boundary tree to $WORK (open question §6.6, deferred).
- Cross-emulator scoring/figures (separate track per
  `2026-05-06_group_sfno_5410_eval_plan.md`).
- World-readable permissions (user declined).

## 10. Sequence of work after Codex sign-off

1. Revise `user_inference.py` per Codex feedback.
2. Implement `run_sfno_5410_inference.py`.
3. Implement `submit_sfno_5410_user_inference.slurm`.
4. Implement `tests/sfno_inference_5410/test_user_inference.py`.
5. Run Tier 1 + dry-run verification.
6. Apply permission changes (chmod commands in §4.5).
7. Run smoke deterministic + smoke ensemble verification.
8. Write the user guide.
9. Hand to user for delivery to colleague.
