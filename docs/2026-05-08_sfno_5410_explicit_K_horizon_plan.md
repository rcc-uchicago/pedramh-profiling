# 5410 NWP eval — explicit K-step horizon (production-path fix) — v3

_Plan for Codex review. Date: 2026-05-08. Branch: zgplev-migration-dsi-bootstrap._
_v1 → v2: addressed 2 Codex blockers (`prediction_duration_days`, off-by-one) and 4 gaps._
_v2 → v3: tightened patch-marker counts, explicit production-yaml rebuild, listed existing tests to update, replaced asserts with ValueError, documented K+1 BCS iteration count._

## Context

The group SFNO-5410 NWP eval pipeline currently rolls every IC to **Jan 1 of (Y+1)** (a full-remainder-of-year forecast) and relies on a downstream adapter to slice the first K leads. We want **K as an explicit forecast-leads knob** (default K=60 = 15 days), with the rollout actually stopping at K leads at the inference layer, plus preflight evidence that it does.

Pre-fix sites:

- `scripts/eval_inference_5410.py:67-72` — `final_datetime_for(Y)` hard-codes Jan 1 (Y+1).
- `src/sfno_inference_5410/stampede3_yaml_override.py:57-65` — `_ensemble_inference_hours_for_year(Y)` hard-codes 8760/8784h. Override does **not** set `prediction_duration_days`.
- `scripts/build_5410_yaml_override.py` — has no `--K` argument; calls `build_per_y_yaml(Y, config_dir, exp_dir)` with no horizon parameter.
- `scripts/submit_eval_inference_5410{,_smoke}.slurm` — hard-code `FINAL=Jan 1 (SMOKE_Y+1)`; no K propagation.
- `/work2/.../v2.0/long_inference.py` lines 554, 720, 724, 834, 998, 1002 hard-code `next_output_datetime = (current_year+1, 1, 1)` and gate the rollout loop on `current_year < final_datetime.year` (year-only). Even with a sub-year `--final_datetime`, the loop overshoots.

User-confirmed design (2026-05-08):

- **K=60** forecast leads, exposed as `--K`.
- **Apply 6 of 8 v4 upstream hunks** (4 allocators + 2 continuation conditions; skip the 2 perturbation gates — eval is deterministic at `epsilon_factor=0`).
- **Add K to eval-track files** (do not consolidate onto `user_inference.py`).

## Two key invariants Codex caught (must be honored)

### (A) `prediction_duration_days` is the actual BCS rollout knob

`long_inference.py:202` constructs the **BCS data loader** with `single_ic=True`. In `data_loader_multifiles.py:818-826`, the `single_ic` branch computes:

```python
if hasattr(self.params, "prediction_duration_days"):
    end_date = start_date + timedelta(days=self.params.prediction_duration_days)
else:
    end_date = self.datetime_class(self.year_start + self.long_rollout_years, 1, 1, ...)
```

For sub-year rollouts `long_rollout_years = final.year - init.year = 0`, so the fallback `end_date == start_date` (or earlier) gives an empty/wrong BCS date range. **`prediction_duration_days` MUST be set in the per-Y yaml.** v4 plan §2.2 (lines 100-109) called this out; v1 of this plan dropped it. Restored in v2.

Note: the BCS `single_ic` loader uses `prediction_duration_days`. The IC dataset (constructed at `long_inference.py:195` with `ensemble=True, init_from_nc=True`) uses `ensemble_inference_hours` for its preload (`data_loader_multifiles.py:831`). Both must be set consistently — they're both downstream of the same horizon.

### (B) Off-by-one: K forecast leads needs `raw_steps = K+1`

Upstream sizes the output buffer at `long_inference.py:562, 836`:

```python
output_inference_steps = int((next_output_datetime - current_datetime).total_seconds() // 3600 / timedelta_hours)
output_surface = np.zeros(..., output_inference_steps, ...)
```

The IC is written at `output[time_step_in_year=0]`, forecasts at `time_step+1` guarded by `if time_step_in_year + 1 < output_surface.shape[1]` (lines 638, 665, 912, 939). The chunk flushes at `time_step_in_year == output_surface.shape[1]`.

With `final = init + K*6h`, `output_inference_steps = K`, buffer holds K rows: IC at 0, forecasts at 1..K-1 — only **K-1 saved forecast leads**. To deliver K forecast leads we need K+1 rows → `final = init + (K+1)*6h`.

**Resolution (Codex's preferred option A, smaller patch):** Define `K` as scorer forecast leads. Compute upstream parameters from `raw_steps = K + 1`:

| Param | Value (K=60) | Where |
|---|---|---|
| `final_datetime` | `init + (K+1) * 6h` = `init + 366h` | CLI `--final_datetime` |
| `ensemble_inference_hours` | `(K+1) * 6` = `366` | yaml `SFNO.ensemble_inference_hours` |
| `prediction_duration_days` | `(K+1) * 6 / 24` = `15.25` (float OK; `timedelta(days=15.25)` works) | yaml `SFNO.prediction_duration_days` |
| Output time dim | `K + 1` = `61` (IC + 60 forecast leads) | NetCDF `time` |

Adapter slicing of the output (if any code still does `time[1:K+1]`) continues to extract the K forecast leads cleanly.

## Files to change

### 1. Upstream `long_inference.py` — partial-horizon patch (6 hunks)

`/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0/long_inference.py`

Outside the AI-RES repo. Applied + tracked in `docs/2026-05-04_makani_local_patches.md` (existing local-patch ledger; new section `## 2026-05-08 — partial-horizon (eval-5410 K=60)`).

For each of the 4 allocator sites (lines **554, 724, 834, 1002**):

```python
# BEFORE
next_output_datetime = self.dataset.datetime_class(
    current_year+1, 1, 1, hour=next_year_offset_hours,
    has_year_zero=self.params.has_year_zero
)
# AFTER
next_year_jan1 = self.dataset.datetime_class(
    current_year+1, 1, 1, hour=next_year_offset_hours,
    has_year_zero=self.params.has_year_zero
)
next_output_datetime = min(next_year_jan1, self.params.final_datetime)
```

For each of the 2 continuation sites (lines **720, 998**):

```python
# BEFORE
current_year += 1
if current_year < self.params.final_datetime.year:
    current_datetime = next_output_datetime
    ...
# AFTER
current_year += 1
current_datetime = next_output_datetime  # hoist before the check
if current_datetime < self.params.final_datetime:
    next_year_offset_hours = current_datetime.hour % self.params.timedelta_hours
    ...
```

**Skipped (eval is deterministic):** the two perturbation-gate hunks at lines 558 and 830. `epsilon_factor=0.0` makes those branches unreachable.

### 2. `src/sfno_inference_5410/stampede3_yaml_override.py`

- **Replace** `_ensemble_inference_hours_for_year(Y)` (lines 57-65) with two helpers parameterised by K. **Use explicit `ValueError`, not `assert`** (Codex edit #4 — assertions can be disabled with `python -O`; preflight must fail loud regardless of optimization flags):
  ```python
  def _raw_steps_for_K(K: int) -> int:
      # Reject bool explicitly — isinstance(True, int) is True in Python.
      if isinstance(K, bool) or not isinstance(K, int) or K < 1:
          raise ValueError(f"K must be a positive int (not bool), got {K!r} ({type(K).__name__})")
      return K + 1  # output buffer rows = IC + K forecast leads
  def _horizon_hours_for_K(K: int) -> int:
      h = _raw_steps_for_K(K) * 6
      if h > 8784:
          raise ValueError(f"K={K} → {h}h exceeds one-year cap; multi-year not supported here")
      return h
  ```
- **Thread `K: int` (required, no default) through** `_override_section`, `build_per_y_yaml`, `build_all`. Refuse `K is None` or non-positive.
- **Add new yaml keys** in `_override_section`:
  ```python
  horizon_hours = _horizon_hours_for_K(K)
  section["ensemble_inference_hours"] = horizon_hours              # IC dataset preload
  section["prediction_duration_days"] = horizon_hours / 24.0       # BCS rollout span — REQUIRED (Codex blocker)
  ```
  Both keys MUST be set on every section so SFNO's explicit copy beats PLASIM via merge.
- Keep `val_year_start = Y, val_year_end = Y + 1` unchanged (boundary-loader contract still holds; `xr.cftime_range(..., inclusive='left')` won't read past `init + (K+1)*6h ≤ Jan 1 (Y+1)` for K=60).
- Update module docstring (lines 1-20): "rollout is K forecast leads (raw_steps = K+1, horizon = (K+1)·6h), not full-year".

### 3. `scripts/build_5410_yaml_override.py` (Codex gap — was missing)

- Add `--K` argparse arg, `type=int`, **required** (no default).
- Pass `K=args.K` to `build_per_y_yaml(...)`.
- Print `[build] K={K} horizon_hours={(K+1)*6} prediction_duration_days={(K+1)*6/24}` per Y.
- Update docstring examples to show `--K 60`.

### 4. `scripts/eval_inference_5410.py`

- **Lines 67-72:** rewrite `final_datetime_for(Y)` → `final_datetime_for(init_dt, K) -> cftime.DatetimeProlepticGregorian` returning `init_dt + dt.timedelta(hours=(K + 1) * 6)`. Removes the Y+1 / Jan 1 logic.
- **Lines 80-139 (`build_argv_for_ic`):** add `K: int` parameter (required); call `final_datetime_for(init_dt, K)` at line 116.
- **Add post-build assertion** (preflight #2):
  ```python
  expected_final = init_dt + dt.timedelta(hours=(K + 1) * 6)
  assert final_dt == expected_final, (
      f"final_datetime mismatch for Y={Y} s={s}: got {final_dt}, "
      f"expected init + (K+1)*6h = {expected_final}"
  )
  ```
- **Lines 142-167 (`build_run_plan`):** thread `K` through; **also pass `K` to `nwp_ic_offsets_5410(n_samples, K=K)`** (Codex gap — was using default K=60 in offset validator while passing a different K to inference).
- **Lines 170-181 (`_parse_args`):** add `--K`, `type=int`, **required** (no default — Codex contradiction fix). Add a positive-int validator. Print `[orchestrator] forecast_K={K} raw_steps={K+1} raw_hours={(K+1)*6}` at start of `main()`.
- **In `main()` per-IC log line:** print `[ic Y={Y} s={s}] init={init_dt} final={final_dt} forecast_K={K} raw_steps={K+1}` so logs show the K-step end date, not "Y+1-01-01" (preflight #5).
- **Add yaml-horizon preflight in `main()` before `--launch`:** for each per-Y yaml in the run plan, load it, read `SFNO.ensemble_inference_hours` and `SFNO.prediction_duration_days`, assert:
  - `ensemble_inference_hours == (K + 1) * 6`
  - `prediction_duration_days == (K + 1) * 6 / 24`
  - `ensemble_inference_hours not in {8760, 8784}` (would indicate year-long override leaked through)
- **Add upstream-patch preflight (strict counts — Codex edit #1):** byte-grep `_UPSTREAM_LONG_INFERENCE` and assert **exactly 4** occurrences of `min(next_year_jan1, self.params.final_datetime)` (the 4 allocator hunks at lines 554, 724, 834, 1002) AND **exactly 2** occurrences of `current_datetime < self.params.final_datetime` (the 2 continuation hunks at lines 720, 998). Reject `< 4` or `< 2` with "upstream long_inference.py is partially patched; expected 4+2 markers, got {a}+{c}; reapply hunks per docs/2026-05-04_makani_local_patches.md before launching". Reject `> 4` or `> 2` with the same fail-loud message (guards against accidental duplicate application).

### 5. `scripts/submit_eval_inference_5410.slurm` (production)

- Add `: "${K:=60}"` near top with comment "canonical eval-track forecast horizon = K forecast leads".
- Pass `--K "$K"` to the orchestrator at line 85-88.
- Add preflight loop: for each `Y in YEARS`, run a small inline python that reads the per-Y yaml and asserts both `ensemble_inference_hours == (K+1)*6` and `prediction_duration_days == (K+1)*6/24`.
- Add upstream-patch byte-grep preflight with **strict counts** (Codex edit #1):
  ```bash
  ALLOC=$(grep -c 'min(next_year_jan1, self.params.final_datetime)' "$UPSTREAM_REPO/long_inference.py" || true)
  CONT=$(grep -c 'current_datetime < self.params.final_datetime' "$UPSTREAM_REPO/long_inference.py" || true)
  if [[ "$ALLOC" -ne 4 || "$CONT" -ne 2 ]]; then
      echo "FATAL: upstream long_inference.py patch incomplete: allocator markers=$ALLOC (expected 4), continuation markers=$CONT (expected 2)" >&2
      echo "  reapply hunks per docs/2026-05-04_makani_local_patches.md" >&2
      exit 2
  fi
  echo "[preflight] upstream patch markers OK: allocators=$ALLOC continuations=$CONT"
  ```
- Add `[launch] forecast_K=$K raw_steps=$((K+1)) raw_hours=$(((K+1)*6))` log line before the orchestrator runs.

### 6. `scripts/submit_eval_inference_5410_smoke.slurm`

- Add `: "${K:=60}"`.
- **Line 95:** replace hardcoded `FINAL=$(printf '%04d-01-01_00:00:00' $((SMOKE_Y + 1)))` with:
  ```bash
  FINAL=$(python -c "
  import cftime, datetime as dt
  init = cftime.DatetimeProlepticGregorian($SMOKE_Y, 1, 1, 0, has_year_zero=True) + dt.timedelta(hours=$SMOKE_S * 6)
  final = init + dt.timedelta(hours=($K + 1) * 6)
  print(final.strftime('%Y-%m-%d_%H:%M:%S'))
  ")
  ```
- Add yaml + patch preflights (mirror production).
- **Post-flight assertion (preflight #4, off-by-one-aware):** open the output NetCDF with xarray, assert `ds.sizes["time"] == K + 1`. Fail with non-zero exit on mismatch.
- Update `[smoke]` log line to include `forecast_K=$K raw_steps=$((K+1))`.

### 7. `src/sfno_inference_5410/preflight.py` (NEW)

Thin helpers shared by orchestrator, SLURMs (via `python -c` or a tiny CLI), and tests:

```python
def assert_K_explicit(K) -> None                                           # int, K >= 1
def assert_final_datetime_matches(init, final, K, dt_hours=6) -> None      # final == init + (K+1)*6h
def assert_yaml_horizon(yaml_path, K, *, section="SFNO") -> None           # both keys, both correct
def assert_upstream_patched(upstream_long_inference_path: Path) -> None    # byte-grep marker
def assert_output_time_dim(nc_path: Path, K: int) -> None                  # ds.sizes["time"] == K + 1
```

`assert_yaml_horizon` rejects `ensemble_inference_hours in {8760, 8784}` and asserts both `ensemble_inference_hours == (K+1)*6` and `prediction_duration_days == (K+1)*6/24`.

### 8. Tests

**New tests:**

- `tests/sfno_inference_5410/test_yaml_override_K.py` (NEW): `build_per_y_yaml(Y=121, ..., K=60)` writes `ensemble_inference_hours == 366` and `prediction_duration_days == 15.25`. Also test `K=56 → 342, 14.25`. Assert call without `K` raises `TypeError` (required arg). Assert `K=0`, `K=-1`, `K="60"`, **`K=True`, `K=False`** raise `ValueError` — bool must be rejected explicitly because `isinstance(True, int)` is True in Python.
- `tests/sfno_inference_5410/test_eval_driver_K.py` (NEW): `build_argv_for_ic(Y=121, s=0, ..., K=60)` returns `final_datetime == init + 366h`. For `Y=125, s=1342` (last IC of a year), assert no overrun (`s + (K+1) ≤ 1460`).
- `tests/sfno_inference_5410/test_upstream_patch_present.py` (NEW): byte-grep the live `/work2/.../v2.0/long_inference.py` for **exactly 4 allocator markers + 2 continuation markers**. Skipped (with `pytest.skip`) on machines without the upstream tree (CI nodes).

**Existing tests to update (Codex edit #3 — these currently break under the new signature or pin the old year-long horizon):**

- `tests/sfno_inference_5410/test_required_attrs.py`:
  - **lines 25, 175-180:** docstring + assertion pin `ensemble_inference_hours == 8784 if Y in {124, 128} else 8760`. Replace with `ensemble_inference_hours == (K+1)*6` and assert `prediction_duration_days == (K+1)*6/24`. Add a `K=60` fixture parameter.
  - **line 115, 139:** `build_per_y_yaml(Y, config_dir, exp_dir)` calls — pass `K=60`.
- `tests/sfno_inference_5410/test_get_dates_contract.py`:
  - **line 108:** `build_per_y_yaml(Y, config_dir, exp_dir)` — pass `K=60`.
  - **line 158:** `steps = self.params.ensemble_inference_hours // self.params.timedelta_hours` is dynamic and survives the change; verify it now equals `K+1 = 61`.
- `tests/sfno_inference_5410/test_yaml_override.py`:
  - **lines 51, 58, 63:** `build_all(config_dir, exp_dir)` — pass `K=60`.
  - Audit any pinned `8760/8784` value in this file and update to `(K+1)*6`.
- `tests/sfno_inference_5410/test_runtime_args_5410.py`:
  - Audit references to `ensemble_inference_hours`, `final_datetime_for`, or year-long horizon values; update for the new signature/values. (Codex flagged line 83; the implementer should read this file and adjust whatever specific assertion lives there.)
- Any other test that imports `final_datetime_for(Y)` — pass the new `(init_dt, K)` signature.

### 9. `.claude/skills/eval-sfno-5410/SKILL.md`

Add a sentence: production now stops at **K=60 forecast leads** per IC (raw_steps=61, horizon=366h). Note the upstream-patch dependency and that smoke must succeed before production.

### 10. `docs/2026-05-04_makani_local_patches.md`

Append `## 2026-05-08 — partial-horizon (eval-5410 K=60)` listing the 6 hunks (file, line, before, after) and the **strict-count** verification (matching the SLURM/orchestrator preflights, not `grep -q`):

```bash
ALLOC=$(grep -c 'min(next_year_jan1, self.params.final_datetime)' /work2/.../v2.0/long_inference.py)
CONT=$(grep -c 'current_datetime < self.params.final_datetime' /work2/.../v2.0/long_inference.py)
[[ "$ALLOC" -eq 4 && "$CONT" -eq 2 ]] || echo "patch incomplete: allocators=$ALLOC (expect 4) continuations=$CONT (expect 2)"
```

A single `grep -q` would pass a partially-applied patch (1 of 4 hunks). The ledger entry must use exact counts so re-application after upstream resync is verifiable.

## Five preflight gates (mapping to user's list, post-Codex)

| # | Requirement | Where enforced | Failure mode |
|---|---|---|---|
| 1 | K is explicit | `--K` is **required** (no default) in both `eval_inference_5410.py` and `build_5410_yaml_override.py`; SLURMs set the env-var default; `_override_section` requires `K` arg | argparse exits 2 if `--K` missing; `assert_K_explicit` rejects None/0/negative |
| 2 | `final_datetime = init + (K+1) × 6h` | Assertion inside `build_argv_for_ic` after computing `final_dt` | Orchestrator raises `AssertionError` before any `--launch` work |
| 3 | YAML carries the right horizon: `ensemble_inference_hours = (K+1)·6` AND `prediction_duration_days = (K+1)·6/24`, neither equal to year-long values | `assert_yaml_horizon` called by both SLURMs and orchestrator before `long_inference.py` is invoked | Exit 2 with "yaml horizon == 8760, expected 366 (K=60)" or "prediction_duration_days missing — BCS loader will collapse" |
| 4 | Smoke output has K+1 time dim | `assert_output_time_dim` in smoke SLURM post-flight | Smoke SLURM exits non-zero if `len(ds.time) != K+1` |
| 5 | Logs show requested end date, not Y+1-01-01 | Orchestrator per-IC log + smoke `[smoke] init=... final=... forecast_K=...` line | Visible in `logs/5410_inf96_*.out` and smoke logs |

Plus a sixth (Codex gap):

| 6 | Upstream patch is applied | byte-grep for `min(next_year_jan1, self.params.final_datetime)` in production SLURM, smoke SLURM, and orchestrator preflight | Exit 2 with "upstream long_inference.py is unpatched" |

## Codex review (before production resubmit)

Once smoke passes, Codex review on the partial-horizon production fix. Diff scope:

- 6 upstream `long_inference.py` hunks (paste before/after).
- `eval_inference_5410.py` K threading + required arg + IC-offset validator threading.
- `stampede3_yaml_override.py` `_horizon_hours_for_K` + `prediction_duration_days` addition.
- `build_5410_yaml_override.py` `--K` plumbing.
- Both SLURMs' preflight + post-flight changes.
- New `preflight.py` helpers.
- New tests.
- Patch ledger update.

Specifically ask Codex to validate:
1. `min(next_year_jan1, final_datetime)` does not break the multi-year continuation path (setting `final = Jan 1 (Y+5)` still rolls correctly).
2. The hoisted `current_datetime = next_output_datetime` before line 720/998 doesn't double-step `time_step_in_year`.
3. K+1 raw rows actually deliver K forecast leads end-to-end (trace IC at index 0 → forecasts at 1..K → adapter slicing if any).
4. `prediction_duration_days = (K+1)*6/24` (a float) is consumed safely by `timedelta(days=...)` and produces an integer-multiple-of-6h end_date.
5. K=60 leaves enough margin (`last_s + (K+1) = 1342 + 61 = 1403 < 1460`) — already guarded in `ic_offsets.py:51` once K is threaded through.
6. The `single_ic` BCS loader truly governs the rollout iteration count and there is no other path where `prediction_duration_days` would be ignored for our YAML.

## Verification (end-to-end smoke + production resubmit)

1. **Apply 6-hunk patch** to `/work2/.../v2.0/long_inference.py`. Verify exact counts (Codex edit #1):
   ```bash
   grep -c "min(next_year_jan1, self.params.final_datetime)" /work2/.../long_inference.py   # MUST be 4
   grep -c "current_datetime < self.params.final_datetime" /work2/.../long_inference.py     # MUST be 2
   ```
2. **Rebuild Y=121 smoke yaml:** `python scripts/build_5410_yaml_override.py --year 121 --K 60 --config-dir <config_dir> --exp-dir <exp_dir>`. Verify emitted yaml has `ensemble_inference_hours: 366` and `prediction_duration_days: 15.25` in the SFNO section.
3. **Run unit tests:** `pytest tests/sfno_inference_5410/` — both new and updated tests must pass.
4. **Smoke:** `K=60 sbatch scripts/submit_eval_inference_5410_smoke.slurm` (1 IC, Y=121 s=0).
5. **Verify in smoke log:**
   - `[preflight] upstream patch markers OK: allocators=4 continuations=2`.
   - `[smoke] init=0121-01-01_00:00:00 final=0121-01-16_06:00:00 forecast_K=60 raw_steps=61` (61·6h = 366h after Jan 1 → Jan 16 06:00).
   - Upstream BCS tqdm progresses through **61 short steps** (one per BCS load — see Note below) then exits — **does not** continue to Jan 1 0122.
   - `[post] time dim = 61` (IC + 60 forecast leads).
   - `ncdump -h` shows `time = 61`.
6. **Submit Codex review** on the full diff (orchestrator + yaml override + yaml builder + SLURMs + new helper + new+updated tests + patch ledger entry).
7. **Codex green → rebuild ALL 8 production yamls explicitly** (Codex edit #2 — do not leave this implicit):
   ```bash
   python scripts/build_5410_yaml_override.py --all-years --K 60 \
       --config-dir <PRODUCTION_CONFIG_DIR> --exp-dir <PRODUCTION_EXP_DIR>
   ```
   Confirm all 8 emitted yamls have `ensemble_inference_hours: 366` and `prediction_duration_days: 15.25`.
8. **Submit production:** `K=60 sbatch scripts/submit_eval_inference_5410.slurm` for the 96-IC sweep. Production SLURM's per-Y yaml-horizon preflight + patch-markers preflight will catch any stale yaml or unpatched upstream and fail loud before consuming H100 time.

**Note on BCS iteration count (Codex edit #5):** the BCS data loader will run **K+1 = 61 short steps** to flush K+1 = 61 raw rows; only 60 of those forward passes' outputs land in the saved buffer (the 61st forecast is computed and discarded by the `time_step_in_year + 1 < shape[1]` guard at `long_inference.py:638, 665, 912, 939`). So tqdm will tick 61 times for 60 scored leads — expected, not a bug. Compute cost is ~1.7% above the theoretical minimum.

## Critical files to read before implementing

- `/home1/11114/zhixingliu/projects/SFNO_Climate_Emulator/scripts/eval_inference_5410.py` (full file)
- `/home1/11114/zhixingliu/projects/SFNO_Climate_Emulator/scripts/build_5410_yaml_override.py` (full file)
- `/home1/11114/zhixingliu/projects/SFNO_Climate_Emulator/src/sfno_inference_5410/stampede3_yaml_override.py` (full file)
- `/home1/11114/zhixingliu/projects/SFNO_Climate_Emulator/src/sfno_inference_5410/user_inference.py` lines 168, 215-236 (`ensemble_inference_hours = horizon_hours` pattern; we extend with `prediction_duration_days`)
- `/work2/.../v2.0/long_inference.py` lines 192-205 (the two data-loader constructions: IC dataset = ensemble+init_from_nc, BCS dataset = single_ic), 540-740, 820-1010
- `/work2/.../v2.0/utils/data_loader_multifiles.py` lines 818-836 (single_ic vs ensemble branches in `_get_dates`)
- `docs/2026-05-08_sfno_5410_external_user_inference_plan_v4.md` §2.1, §2.2, §3.2 (the v4 patch design, including the `prediction_duration_days` key — we are reusing the BCS-loader insight here)
- `docs/2026-05-04_makani_local_patches.md` (existing local-patch ledger)

## Out of scope

- The 2 perturbation-gate hunks (`init_datetime.year == 1` checks at lines 558, 830). Eval is deterministic; revisit if a future eval needs ensemble perturbations.
- Adapter-side slicing logic. Once inference stops at K+1 raw rows, the adapter becomes near-passthrough; cleaning up is a separate task.
- Wallclock reduction in production SLURM. Could drop from 4 h → 1.5 h once K=60 lands; keep as follow-up to avoid coupling resource changes with correctness fix.
- Score / report / figures SLURMs (still don't exist; tracked separately in the eval-track plan).

## Revision history

- **v1 (2026-05-08, morning):** initial draft. Used `final = init + K*6h` and only `ensemble_inference_hours = K*6`. Codex review flagged: (a) missing `prediction_duration_days` (BCS loader bound); (b) off-by-one — `K*6h` yields K-1 saved leads; (c) `build_5410_yaml_override.py` not in change list; (d) production preflight didn't check `prediction_duration_days` or upstream patch; (e) "K explicit" contradicted by `--K` default; (f) `nwp_ic_offsets_5410` didn't receive K.
- **v2 (2026-05-08, midday):** define K as forecast leads; `raw_steps = K+1`; YAML sets both `ensemble_inference_hours = (K+1)·6` and `prediction_duration_days = (K+1)·6/24`; `--K` required (no default) in orchestrator and yaml builder; SLURMs supply env-var default; threaded K through `nwp_ic_offsets_5410`; added upstream-patch byte-grep preflight; `build_5410_yaml_override.py` added to change list; smoke assertion is `time == K+1`; logs print both `forecast_K` and `raw_steps`.
- **v3 (2026-05-08, afternoon):** Codex round-2 edits:
  1. Patch-marker preflight requires **exact counts** (4 allocators + 2 continuations), not just "≥1".
  2. Verification runbook now has an explicit step "rebuild all 8 production yamls with `--all-years --K 60`" between smoke success and production submit — no longer implicit.
  3. Existing tests to update enumerated by file: `test_required_attrs.py` (lines 25, 115, 139, 175-180), `test_get_dates_contract.py` (line 108), `test_yaml_override.py` (lines 51, 58, 63), `test_runtime_args_5410.py` (line 83 audit).
  4. Helper sketches use `if not isinstance(K, int) or K < 1: raise ValueError(...)` instead of `assert K >= 1` — survives `python -O`.
  5. Documented that BCS tqdm ticks **K+1 = 61 short steps for 60 scored leads** (61st forecast is computed and discarded by the `< shape[1]` save guard); compute cost ~1.7% above theoretical minimum.
- **v3.1 (this doc, post-approval nits):** two non-blocker tweaks pre-implementation:
  1. `docs/2026-05-04_makani_local_patches.md` ledger entry uses the **same strict-count verification** as the SLURMs (4 allocators + 2 continuations), not `grep -q` — guards against partially-applied patches showing as "OK".
  2. `K`-validation helper + tests reject **`bool`** explicitly (`isinstance(K, bool)` short-circuit), since `isinstance(True, int)` is True in Python.
