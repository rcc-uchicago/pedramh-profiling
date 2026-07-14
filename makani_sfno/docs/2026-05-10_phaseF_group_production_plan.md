# Phase F — Group-code SFNO sigma10 production training plan (v5)

Date: 2026-05-10. Builds on `docs/2026-05-09_group_code_training_track_plan.md` (plan v5, Phase 1 = v10 shim, smoke went green).

**Revision v5 highlights** (response to user review of v4):

- **B1 (offset applied at every index, not just index=0):** Confirmed at `data_loader_multifiles.py:954` (index==0) AND `:982` (index>0): both branches add `+ timedelta(hours=self.nc_bc_offset)`. `_get_boundary_data` (`:926`) then remaps the *already-offset* datetime to canonical year 11 or 12. v4's claim that "subsequent rollout boundary reads do NOT add the offset" was wrong.
- **B2 (re-derived enumeration):** With offset every step, the precise read set per init year is:
  - **Year 121 (non-leap)**: year 11 idx **3..1459** (synthetic year 121 reads at offset) ∪ year 11 idx **0..2** (synthetic year 122 wrap) = year 11 idx 0..1459 (full padded).
  - **Year 124 (leap)**: year 12 idx **3..1463** (synthetic year 124, offset) ∪ year 11 idx **0..2** (synthetic year 125 wrap). Year 12 idx 0,1,2 are NEVER read for this init.
- **B3 (Check 18 / T.3 formula):** rewritten as an explicit enumeration mirroring the code path:
  ```
  index=0 state read:  start_time = canonical_year_jan1 + dates[0]*6h + 18h    (= idx 3 of canonical year)
  index>0 reads:       data_dt   = self.start_date + dates[k]*6h + 18h
                       canonical = leap_year if is_leap(data_dt.year) else no_leap_year
                       data_idx  = ((data_dt - Jan1(data_dt.year)).total_seconds() // 3600 // 6)
  ```
  Mirror, don't hand-derive.
- **B4 (Check 20 concrete provenance):** stats/climatology builders gain explicit attrs:
  - `built_before_padding: true|false`
  - `native_timesteps_total: int` (sum across train years)
  - `native_timesteps_by_year: {12: 1455, 13: 1455, ...}`
  - `manifest_skip_padded_used: true|false`
  Check 20 reads these and asserts consistency with manifest.
- **B5 (held over from v4):** stats/clim phase ordering F.B1 → F.C → F.B2; init NC exact-time check; `--max-output-leads 60` default; climatology comment cleanup; F.K2 explicit smoke sequence — all retained.

## Scope (locked, v4)

Same as v3:

| Knob | Value |
|------|-------|
| Train years | 12–111 (100 yrs) |
| Validation year | 11 |
| Test years | 121–128 (used as v10 source for IC NetCDFs + truth ONLY; NOT converted to group format) |
| **Years to convert (group h5)** | **101** (year 11 + 12-111) |
| **Years to PAD** | 2 (year 11 → 1460 frames, year 12 → 1464) |
| Pad donors | year 12 (5 frames → year 11), year 13 (9 frames → year 12) |
| Padding ordered AFTER stats/climatology | NEW in v4 |
| Epochs | 50 |
| Hardware | Stampede3 h100 multi-node DDP, 1 GPU/node, 15h walltime/segment |
| Eval path | `long_inference.py` per-IC (8 jobs) |
| Comparator | zgplev GB4 own-track baseline |
| Sanity gate | Production: fail-stop. F.K2 (smoke ckpt): tiered (Tier 1 schema + finite-output, Tier 2 science) |

## Loader & runtime contract (v4 — all verified by file:line)

### single_ic boundary remap (v3 finding, v4 wording cleanup)

`_get_boundary_data` at `data_loader_multifiles.py:926-934`:

```python
data_year = data_datetime.year
data_idx = ((data_datetime - datetime_class(data_year, 1, 1, 0)).total_seconds()) // 3600 // 6
if cftime.is_leap_year(data_year, ...):
    data_file_path = get_out_path(self.boundary_data_dir, self.leap_year, data_idx)   # year 12
else:
    data_file_path = get_out_path(self.boundary_data_dir, self.no_leap_year, data_idx) # year 11
```

`single_ic.__getitem__(0)` start-time remap (lines 948-960):

```python
if self.start_date.year != self.params.val_year_start:        # any test IC where start.year != 11
    if cftime.is_leap_year(self.start_date.year, ...):
        start_time = self.datetime_class(self.leap_year, ...) ... + timedelta(hours=self.nc_bc_offset)
    else:
        start_time = self.datetime_class(self.no_leap_year, ...) ... + timedelta(hours=self.nc_bc_offset)
```

**Implication (v5 wording):**
- All reads from the **group H5 data_dir** during long_inference single_ic mode are remapped to canonical years 11 (no_leap) or 12 (leap), regardless of init year. Years 121-128 are NOT in the group H5 data_dir read path.
- The IC state is read SEPARATELY from `init_nc_filepaths` (`data_loader_multifiles.py:905`, `_get_data_nc`). Init NCs ARE year-specific (year 121, 122, ..., 128).
- The +18h `nc_bc_offset` is added at **every** index (lines 954/958/960 for index==0; line 982 for index>0). `_get_boundary_data` then remaps the already-offset datetime via its `data_year`. The "+18h once" framing in v4 was wrong; corrected here.

### Init NC time-exactness contract

`long_inference.py:1330-1334`:

```python
for file in params.init_nc_filepaths:
    ds = xr.open_dataset(file, engine='netcdf4')
    index = ds.get_index("time").get_loc(params["init_datetime"])  # KeyError if not exact
    params['init_nc_timestep_offset'].append(index)
```

**Implication:** init NC time coord MUST contain `init_datetime` EXACTLY (e.g., `0121-01-01 00:00:00`). NO offset, NO interpolation. Build init NC with `time = [init_datetime]` (1 entry) per Phase 1's `build_init_nc_from_v10.py`.

### Year-buffer sizing (recap)

`long_inference.py:729`: `output_inference_steps = int((next_output_datetime - current_datetime).total_seconds() // 3600 / timedelta_hours)`. With `init_datetime=0121-01-01_00:00:00`, `final_datetime=0122-01-01_00:00:00`: 1460 (non-leap). With leap init (0124): 1464.

Combined: padding canonical year 11 to 1460 covers all non-leap synthetic years; padding year 12 to 1464 covers all leap synthetic years.

### nc_bc_offset units & application sites

HOURS, not timesteps (`timedelta(hours=self.nc_bc_offset)`). Hardcoded to 18 by `long_inference.py:1267`.

**Applied at every index in `__getitem__` for single_ic mode:**
- Line 954: index==0, remap branch (init year != val_year_start). `start_time = datetime_class(canonical_year, ...) + dates[0]*6h + 18h`. Used for `_get_data` state read.
- Line 958: index==0, leap branch.
- Line 960: index==0, no-remap branch (init year == val_year_start; falls through to use `self.start_date + dates[0]*6h + 18h`).
- Line 982: index > 0 (rollout). `start_time = self.start_date + dates[index]*6h + 18h`. Used for `_get_boundary_data`.

`_get_boundary_data` (line 926) then determines canonical year from `start_time.year` (the already-offset datetime).

**Effect on enumeration** (assuming `dates[k] = k*6h`, `nc_bc_offset = 18h`, `self.start_date.year = init_dt.year`):

```
data_dt(k) = init_dt + (k*6 + 18) hours
data_year(k) = data_dt(k).year
canonical(k) = leap_year (12) if is_leap(data_year(k)) else no_leap_year (11)
data_idx(k) = ((data_dt(k) - Jan1(data_year(k))).total_seconds() // 3600 // 6)
read file: <canonical(k)>_<data_idx(k):04>.h5
```

For `init_dt = 0121-01-01 00:00:00`, `final_dt = 0122-01-01 00:00:00` (1460 steps):
- k=0..1456: data_year=121 (non-leap), canonical=11, data_idx in {3, 4, ..., 1459}
- k=1457..1459: data_year=122 (non-leap), canonical=11, data_idx in {0, 1, 2}
- Required year-11 indices: {0..1459}. **Padding to 1460 covers exactly.**

For `init_dt = 0124-01-01 00:00:00`, `final_dt = 0125-01-01 00:00:00` (1464 steps):
- k=0..1460: data_year=124 (leap), canonical=12, data_idx in {3, 4, ..., 1463}
- k=1461..1463: data_year=125 (non-leap), canonical=11, data_idx in {0, 1, 2}
- Required year-12 indices: {3..1463}. Year 12 idx 0,1,2 NEVER read.
- Required year-11 indices: {0, 1, 2}. **Padding year 12 to 1464 covers exactly.**

### Argparse contract

- `--nc_bc_offset` does NOT exist (hardcoded).
- `ensemble_inference_hours` MUST be SCALAR.
- `--final_datetime` MUST be passed.

### Climatology / ACC contract

`train.py:3771-3776`: when `use_sigma_levels=True`, ACC + GIF + spectra diagnostics are auto-disabled. Production YAML uses `use_sigma_levels=true` so daily climatology is sufficient. Update `build_group_climatology.py:28-29` comment to reflect this.

## Reuse from Phase 1

Unchanged: `tools/_h5_keys.py`, `tools/build_group_stats_netcdf.py` (modified to skip padded indices, see F.0 below), `tools/build_group_climatology.py` (same modification + comment cleanup), `tools/build_init_nc_from_v10.py`, `tools/render_yaml.py`, `tools/preflight_checks.py`, score-function wrapper, env, all 16 Phase 1 tests.

**Modified:** `tools/convert_v10_to_group_h5.py` — adds `--pad-canonical-years` mode (default off).

**Modified (defensive):** `tools/build_group_stats_netcdf.py` and `tools/build_group_climatology.py` — when manifest is present, skip indices `>= n_timesteps_native` to avoid padded-frame contamination. This is BELT-AND-SUSPENDERS: phase ordering already prevents the issue, but the skip-in-builders fixes any future re-ordering accident.

## Phase ordering (v4 critical change)

**v3 (broken):** F.B (convert + pad) → F.C (stats + clim).
**v4:** F.B1 (native convert) → F.C (stats + clim) → F.B2 (pad).

Stats/climatology see ONLY native frames. Padding happens AFTER stats are written and audited.

## NEW components (Phase F, v4)

### F.0 — Converter `--pad-canonical-years` mode (UNCHANGED from v3)

After native conversion (F.B1) and stats/climatology (F.C), F.B2 padding pass:

- Year 11 → 1460 frames: copy year 12 idx [0..5) into year 11 idx [1455..1460).
- Year 12 → 1464 frames: copy year 13 idx [0..9) into year 12 idx [1455..1464).

Manifest fields per padded year: `n_timesteps_native` (1455), `n_timesteps_padded` (1460/1464), `is_leap`, `pad_source: [(dst_idx, src_year, src_idx)]`.

The padded zone (idx 1455+) is invisible to:
- Train (caps at idx 1453 per train_data_sets ranges)
- Val (caps at idx 1394)
- Stats/climatology (skip via manifest, after F.C ordering)

Only single_ic long_inference reads padded frames, and only at the year-boundary tail of each rollout.

### F.0.b — Stats/climatology builders: defensive padded-frame skip + concrete provenance attrs

Both builders gain:

1. **Manifest-aware skip** (defense-in-depth):

   ```python
   if manifest_path.exists():
       manifest = json.load(open(manifest_path))
       year_meta = next(y for y in manifest['years'] if y['year'] == year)
       n_native = year_meta.get('n_timesteps_native', n_timesteps)
       files = files[:n_native]
       skipped[year] = max(0, len(files_full) - n_native)
   ```

2. **Concrete provenance attrs on the output NetCDF** (so Check 20 has something concrete to assert):

   ```python
   ds.attrs['source'] = 'recomputed_from_converted_h5'
   ds.attrs['built_before_padding'] = True if no manifest had n_timesteps_padded set,
                                      else False
   ds.attrs['native_timesteps_total'] = sum(year_n_native for year in train_years)
   ds.attrs['native_timesteps_by_year'] = json.dumps({str(year): n_native_for_year, ...})
   ds.attrs['manifest_skip_padded_used'] = True if any year had files trimmed, else False
   ds.attrs['git_sha'] = <build-time git SHA>
   ds.attrs['build_timestamp_utc'] = <ISO 8601>
   ```

Plan-level guarantee: F.B1 → F.C → F.B2 ordering means stats are built before pad anyway. The skip + attrs protect against ordering accidents AND make the provenance machine-checkable. T.8 verifies, Check 20 enforces.

### F.1 — Production YAML (UNCHANGED from v3)

Same as v3:
- 50 epochs, batch_size 8, lr 2e-6, num_inferences 128
- log_to_wandb true, fresh_start false, num_data_workers 4
- `varying_boundary_variables: ['z0', 'sst', 'rsdt', 'sic']`
- `constant_boundary_variables: ['lsm', 'sg']`
- 100-year `train_data_sets`, 1-year `validation_data_sets`
- `ensemble_inference_hours: 8760` (SCALAR)
- `save_forecasts: true`, `long_rollout_years: 1`
- **NO `prediction_duration_days`**
- `save_basenames: ["dummy"]` (defensive)
- `forecast_lead_times: [1, 12, 20, 40, 60]` (max 60 = 360h)
- Diagnostic knobs ON (auto-disabled by sigma)

### F.2 — Multi-node DDP train slurm + chain (UNCHANGED from v3)

`#SBATCH -N 4` fixed; override via `submit_train_full_loop.sh` wrapper passing `sbatch -N $NODES`. Resume / sentinel / `.done` logic unchanged.

### F.3 — long_inference slurm (UNCHANGED CLI from v3)

Per-year (121..128). `--init_datetime` exact, `--final_datetime` exact, no `--nc_bc_offset`, torchrun `--standalone`.

### F.4 — long_inference output converter (DEFAULT CHANGED in v4)

`scripts/convert_group_long_inference_to_aires_nc.py`:
- **Default `--max-output-leads = 60`** (covers scorecard 360h max). Scorer never sees NaN truth.
- Override via CLI; full-year (1459) available if needed for ad-hoc analysis.
- Truth read from `MOST.0<NNN>.h5` for leads 1..60 — fully covered (truth horizon = 1454).

### F.5 — Preflight extensions (CHECK 19 REWRITTEN, CHECK 20 NEW)

- **Check 16 (pre_train):** wandb auth available IF `log_to_wandb=true`.
- **Check 17 (pre_train):** if resuming, YAML hash matches sentinel.
- **Check 18 (pre_inference_long, REWRITTEN v5):** boundary-window expansion. Mirror the loader code path with offset-everywhere semantics:
  - For step k in [0, year_buffer):
    - `data_dt = init_dt + (k*6 + 18) hours` (offset applied at every k per `data_loader_multifiles.py:954/958/960/982`)
    - `data_year = data_dt.year`
    - `canonical = leap_year (12) if is_leap_year(data_year, 'proleptic_gregorian', has_year_zero=True) else no_leap_year (11)`
    - `data_idx = ((data_dt - cftime.datetime(data_year, 1, 1, 0)).total_seconds() // 3600 // 6)`
    - Assert `<canonical>_<data_idx:04>.h5` exists in data_dir.
  - Specifically tests YEAR=121 (expect: year 11 idx 0..1459, full padded zone) AND YEAR=124 (expect: year 12 idx 3..1463 + year 11 idx 0..2; year 12 idx 0,1,2 NOT in read set).
  - Asserts NO read into years 121-128 in the data_dir.
- **Check 19 (pre_inference_long, REWRITTEN v4):** init NC time coord contains `init_datetime` EXACTLY. Open each `init_nc_filepaths` entry; assert `ds.get_index("time").get_loc(cftime_init_datetime)` succeeds (no KeyError). NO `+18h` requirement.
- **Check 20 (pre_train, REWRITTEN v5 with concrete provenance):** stats + climatology provenance audit. Read `data_train_mean.nc`, `data_train_std.nc`, `climatology.nc` attrs; assert ALL of:
  - `source == "recomputed_from_converted_h5"`
  - `built_before_padding == True` OR (`built_before_padding == False` AND `manifest_skip_padded_used == True`)
  - `native_timesteps_total` matches `sum(manifest.years[*].n_timesteps_native)` for the train_year set
  - `native_timesteps_by_year` (JSON-encoded dict) matches per-year `n_timesteps_native` from manifest
  - `git_sha` non-empty and parseable
  - `build_timestamp_utc` present
  
  Refuse launch if any field missing or mismatched.
- New phase `--phase pre_inference_long` runs 11-15 + 18 + 19.

### F.6 — Production eval chain (UNCHANGED)

`scripts/submit_eval_group_prod.sh`:
1. Per-IC inference (long_inference).
2. Per-IC convert (max-output-leads=60).
3. Per-IC score (fail-stop).
4. Report + figures (GB4 overlay).

For F.K2 (smoke ckpt), use `--informative-mode` flag wrapping `score_nwp.py` to distinguish Tier 1 (runtime) vs Tier 2 (science).

## F.K2 — Explicit smoke gate sequence (v4 — folds in user's suggestion)

Before F.L 50-epoch launch, F.K2 runs this exact sequence:

1. Convert years 11, 12, 13 (NATIVE only, no padding).
2. Build stats over years 12-13 (sees 1455-frame native files only).
3. Build climatology over years 12-13 (same).
4. Run F.B2 padding for years 11 (donor 12) and 12 (donor 13).
5. Build init NCs for year 121 (non-leap) AND year 124 (leap). Init NC time coord = `[0121-01-01 00:00:00]` and `[0124-01-01 00:00:00]` respectively.
6. Render production YAML.
7. Run preflight `--phase pre_train` (checks 1-10 + 16 + 17 + 20).
8. Take existing 1-epoch smoke ckpt at `$SCRATCH/SFNO_Climate_Emulator/runs/sfno_group_sigma10_smoke/SFNO/smoke_*/checkpoints/best_ckpt.tar`.
9. Run preflight `--phase pre_inference_long` (checks 11-15 + 18 + 19) for YEAR=121 AND YEAR=124.
10. Submit `submit_long_inference_full.slurm YEAR=121` AND `YEAR=124` (in parallel).
11. Wait for both. Assert NetCDFs written.
12. Run `convert_group_long_inference_to_aires_nc.py --max-output-leads 60` on both.
13. Run `score_nwp.py --informative-mode` on both.

**Tier 1 acceptance (must pass):**
- All preflight checks green.
- No FileNotFoundError, argparse error, shape error, or unhandled exception in any step.
- Both NetCDFs written, both 53-channel scorer NetCDFs written.
- All values in leads 1..60 of converted NetCDFs are FINITE (no NaN, no Inf).
- score_nwp.py runs to completion (informative gate may report science fail).

**Tier 2 acceptance (allowed to fail on smoke ckpt):**
- Scientific sanity gates (RMSE/ACC thresholds).

## Tests (v4 — total 10 NEW tests)

### T.1 — `test_long_inference_cli_contract.py`

Same as v3. Tests YEAR=121 AND YEAR=124. Asserts no `--nc_bc_offset`, underscore datetime format, `--final_datetime` explicit.

### T.2 — `test_production_yaml_types.py`

Same as v3. Asserts scalar `ensemble_inference_hours`, no `prediction_duration_days`, `save_basenames` present, `long_rollout_years: 1`.

### T.3 — `test_boundary_window_expansion.py` (REWRITTEN v5 — offset every step)

Mirror the actual loader code path with offset applied at every k:

```python
def expand_boundary_reads(init_dt, final_dt, leap_year=12, no_leap_year=11, nc_bc_offset_hours=18):
    """Mirror data_loader_multifiles.py:954/958/960/982 + 926-934.
    Offset applied at index==0 (state read via _get_data) AND every index>0
    (boundary read via _get_boundary_data). Both go through same canonical-year remap.
    """
    pairs = set()
    n_steps = int((final_dt - init_dt).total_seconds() // 3600 // 6)
    for k in range(n_steps):
        # Same formula for k=0 (state read after remap) and k>0 (boundary read).
        # k=0 uses datetime_class(canonical_year, init_dt.month, init_dt.day, ...) + 18h
        # k>0 uses self.start_date + k*6h + 18h, then _get_boundary_data remaps
        # The required (canonical, idx) pair is identical between the two formulations.
        data_dt = init_dt + timedelta(hours=k*6 + nc_bc_offset_hours)
        data_year = data_dt.year
        canonical = leap_year if cftime.is_leap_year(data_year, 'proleptic_gregorian',
                                                      has_year_zero=True) else no_leap_year
        data_idx = int((data_dt - cftime.DatetimeProlepticGregorian(data_year, 1, 1, 0,
                                                                     has_year_zero=True)
                       ).total_seconds() // 3600 // 6)
        pairs.add((canonical, data_idx))
    return sorted(pairs)
```

Hard assertions per test year:

- **YEAR=121** (init `0121-01-01_00:00:00`, final `0122-01-01_00:00:00`, 1460 steps):
  - All pairs have canonical == 11.
  - data_idx set == {0, 1, 2, 3, 4, ..., 1459} (full padded zone).
  - max(data_idx) == 1459 (must be < `n_timesteps_padded` for year 11 = 1460).
- **YEAR=124** (init `0124-01-01_00:00:00`, final `0125-01-01_00:00:00`, 1464 steps):
  - 1461 pairs at canonical 12 with data_idx ∈ {3, 4, ..., 1463}.
  - 3 pairs at canonical 11 with data_idx ∈ {0, 1, 2}.
  - max(data_idx) for year 12 == 1463 (must be < `n_timesteps_padded` for year 12 = 1464).
  - **Assert {0, 1, 2} NOT in year-12 indices** (the leap-init year 12 idx 0,1,2 are never read).
- For all 8 test years (121..128): NO pair where canonical ∈ {121..128}.
- Sentinel files at all required (canonical, idx) exist in fake data_dir; assertion fails loudly if any required file missing.

### T.4 — `test_long_inference_h100_integration` (= F.K2 sequence above)

The slurm-marked integration test. Runs the F.K2 sequence end-to-end. Tier 1 gates F.L.

### T.5 — `test_convert_long_inference_output.py`

Same as v3. Assert 53-channel mapping. Add: assert truth filled for leads 1..60, no NaN/Inf in scorer-relevant slice.

### T.6 — `test_padding_continuity.py`

Same as v3. Assert pad sources bit-identical to donors.

### T.7 — `test_no_year_121_128_in_data_dir.py`

Same as v3. Assert group H5 data_dir contains years 11, 12-111 ONLY.

### T.8 — `test_stats_no_padded_contamination.py` (REVISED v5 — provenance attrs)

- Set up fake data_dir with year 12 native (1455 files) + 9 padded files (idx 1455..1463 stamped with sentinel 9999).
- Run `build_group_stats_netcdf.py` with manifest reporting `n_timesteps_native=1455`.
- Assert stats `count` per key == 1455 × spatial_size (NOT 1464).
- Assert `mean` does NOT include 9999 contribution (within physical range).
- Assert output NetCDF attrs:
  - `built_before_padding` is True (because manifest n_timesteps_padded was absent at build time) OR `manifest_skip_padded_used` is True (if padded already and manifest reports both).
  - `native_timesteps_total == 1455` (single-year fixture).
  - `native_timesteps_by_year == {"12": 1455}`.
- Same test for `build_group_climatology.py`.

### T.9 — `test_init_nc_time_exactness.py` (NEW v4)

- Build init NC for year 121 with `--synthetic-init-dt "0121-01-01 00:00:00"`.
- Open with xarray, get time index.
- Assert `time.get_loc(cftime.DatetimeProlepticGregorian(121, 1, 1, 0, has_year_zero=True))` succeeds (no KeyError).
- Assert NO entry at `init_dt + 18h` (e.g., `0121-01-01 18:00:00`) — that would indicate the +18h offset got baked in incorrectly.
- Repeat for YEAR=124.

### T.10 — `test_converter_K60_finite.py` (NEW v4)

- Build synthetic long_inference NetCDF with shape `[1, 1460, 64, 128]` for surface vars, `[1, 1460, 10, 64, 128]` for upper-air. Fill with smooth physical-range values (e.g., pl ~ 100000 + small noise).
- Run converter with `--max-output-leads 60`.
- Assert output NetCDF has time dim = 60 (or 61 incl. IC).
- Assert no NaN/Inf in `state` or `diagnostic` arrays for leads 1..60.
- Assert truth array filled (read from MOST.0121.h5 leads 1..60), no NaN/Inf.

## Effort estimate (v4)

| Phase | What | Wallclock | Compute |
|-------|------|-----------|---------|
| F.A | Plan v4 + memory note | 30 min | — |
| F.B1 | Convert 101 years NATIVE (slurm array) | 1 h | 101 × 5 min h100 |
| F.C | Stats + climatology over years 12-111 (NATIVE only) | 1 h | 1 h h100 |
| F.B2 | Pad year 11 (5 frames), year 12 (9 frames) | 5 min | local |
| F.D | 8 init NCs (years 121..128) | 30 min | local |
| F.E | Production YAML | 1 h | local |
| F.F | Train slurm + loop wrapper | 3 h | local |
| F.G | long_inference slurm + 8-year orchestrator | 2 h | local |
| F.H | long_inference output converter (incl. --max-output-leads default 60) | 4 h | local |
| F.I | Preflight 16/17/18/19/20 + new phase + builder skip-padded | 3 h | local |
| F.J | Eval chain (prod + F.K2 informative variant) | 3 h | local |
| F.T | T.1–T.10 tests (8 unit + 1 slurm-gated + 1 doc) | 5 h | local; T.4 = 1 h h100 |
| **F.K1** | 1-epoch dry-run train + wrapper smoke (Phase 1 parity) | 4 h wall | 4× h100 1 chain segment |
| **F.K2** | Explicit user-suggested sequence (T.4 with YEAR=121 + YEAR=124, K=60 finite, informative scorer) | 1 h | 1× h100 ~1 h |
| F.L | Full 50-epoch run (gated on F.K2 Tier 1) | 30–40 h wallclock | 4–8× h100 chained |
| F.M | 8 long_inference jobs + scoring | 8 h | 8× h100 1h each |

**Critical path:** F.A → F.T (parallel: F.E–F.J) → F.B1 → F.C → F.B2 → F.D → F.K1 → F.K2 → F.L → F.M.

## Acceptance gates (v4)

- **Pre-launch (must pass before F.L):**
  - All 16 Phase 1 tests pass.
  - All 10 NEW tests T.1-T.10 pass.
  - F.B1 converts 101 native years.
  - **F.C stats + climatology built ONLY on native frames** (verified by T.8 + Check 20).
  - F.B2 pads year 11 to 1460, year 12 to 1464; T.6 verifies bit-identity.
  - F.K1 1-epoch dry-run + clean resume.
  - **F.K2 Tier 1 must pass** for both YEAR=121 (non-leap) AND YEAR=124 (leap):
    - All preflight checks green.
    - No schema/runtime errors.
    - Both NetCDFs written, converter exits 0.
    - All values in leads 1..60 finite.
    - score_nwp.py informative-mode runs without unhandled exception.
  - Preflight 1-10 + 16 + 17 + 20 green; pre_inference_long: 11-15 + 18 + 19 green.

- **Mid-run (during F.L):** unchanged.

- **Post-train:** unchanged.

- **Post-eval (F.M):** unchanged. score_nwp fail-stop must pass on production ckpt.

## Risks (v4 deltas from v3)

- **R-F1 Multi-node DDP rendezvous:** unchanged.
- **R-F2 long_inference correctness:** addressed by F.K2 + T.3 + T.4.
- **R-F3 Production sanity gate:** unchanged.
- **R-F4 wandb auth:** check 16.
- **R-F5 Chain over-submits:** `.done` sentinel.
- **R-F6 fp16 instability:** unchanged.
- **R-F7 h100 queue:** unchanged.
- **R-F8 z0 placement:** unchanged.
- **R-F9 Padding contamination of stats:** ELIMINATED in v4 by phase reordering + manifest-aware skip in builders. T.8 verifies.
- **R-F10 Padded-zone forcing discontinuity:** unchanged from v3 (low). Only 5/9 frames at year-tail, after most scorecard leads scored.
- **R-F11 Truth horizon:** unchanged. With `--max-output-leads 60`, scorer never sees beyond truth horizon.
- **R-F12 single_ic data path quirks:** F.K2 catches.
- **R-F13 Init NC exact-match (low):** if `build_init_nc_from_v10.py` ever returns a time coord that drifts (e.g., 0121-01-01 00:00:01 due to floating-point cftime arithmetic), `get_loc` will KeyError. T.9 verifies bit-exact match.
- **R-F14 Offset wrap correctness (NEW v5, low):** v4's enumeration was off because the offset-everywhere semantics weren't represented. v5 enumeration mirrors `data_loader_multifiles.py:954/958/960/982` directly. Risk reduced to "the loader code itself changes upstream" — pin upstream commit at first production run; T.3 fails loudly if upstream offset semantics change.

## Memory hooks (post F.L green)

- Project: "Phase F group production: years 12-111 train, 11 val, 121-128 test (v10-source-only). 50 epochs h100 multi-node DDP. long_inference single_ic mode reads from group H5 data_dir at CANONICAL years 11/12 (never 121-128); init NCs ARE year-specific. Padding: year 11 → 1460, year 12 → 1464. Phase ordering: F.B1 (native) → F.C (stats) → F.B2 (pad). Conversion scope = 101 years."
- Reference: "long_inference.py contract: NO `--nc_bc_offset`; `ensemble_inference_hours` SCALAR; `--final_datetime` REQUIRED; init/final format `%Y-%m-%d_%H:%M:%S` underscore; `nc_bc_offset` HOURS, applied at EVERY index (k==0 line 954/958/960; k>0 line 982), then `_get_boundary_data` remaps; `prediction_duration_days` BREAKS leap years (use `long_rollout_years: 1`); init NC must contain init_datetime EXACTLY (`get_loc` KeyError otherwise); converter `--max-output-leads 60` default for production."
- Reference: "Boundary read enumeration formula (verified by T.3 against upstream): for k in [0, year_buffer), `data_dt = init_dt + (k*6 + nc_bc_offset_hours) hours`; `canonical = leap_year(12) if is_leap(data_dt.year) else no_leap_year(11)`; `data_idx = ((data_dt - Jan1(data_dt.year)) // 6h)`. For YEAR=121 init: year 11 idx 0..1459. For YEAR=124 init: year 12 idx 3..1463 + year 11 idx 0..2."
- Reference: "single_ic boundary remap at `data_loader_multifiles.py:926-934` always reads `leap_year`/`no_leap_year` for group H5 data_dir; init NC reads at `:905` are year-specific."

## Deferred / open

1. Phase 2 (13-plev repackage).
2. bf16 wrapper.
3. 5410 group-track scoring.
4. EMA decay tuning.
5. z0 const-vs-varying sensitivity study.
6. Hour-quarter binning climatology (only needed if non-sigma model variants added).

## Recommended next steps

1. User reviews v4.
2. F.B1 (101 years native convert) starts on approval.
3. F.E–F.J + F.T (tests) proceed in parallel.
4. F.C runs after F.B1 (NEVER concurrent with F.B2).
5. F.B2 runs after F.C.
6. F.K1 + F.K2 (both YEAR=121 AND YEAR=124, finite K=60 output) must pass before F.L.
7. F.L chains until 50 epochs.
8. F.M after train.
