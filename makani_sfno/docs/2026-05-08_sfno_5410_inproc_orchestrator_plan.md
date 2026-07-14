# 5410 NWP eval — in-process orchestrator (deep refactor) — v2.1

_Plan for Codex review. Date: 2026-05-08. Branch: zgplev-migration-dsi-bootstrap._
_Follow-up to docs/2026-05-08_sfno_5410_explicit_K_horizon_plan.md (v3.1)._
_v1 → v2: addresses Codex round-1 (3 blockers + 3 majors)._
_v2 → v2.1: Codex round-2 conditional-approval edits — split hydration allowlist (static vs per-IC), replaces raw `assert` with `ValueError` in `_run_one_ic`, gates `assert_output_dir_empty` on `--launch`, makes `reconfigure_for_ic` validate `init_nc_timestep_offset`, expands `assert_output_dir_complete` to also check the variable set._

## Context

The K=60 partial-horizon fix shipped successfully (smoke 3097936 verified
all six preflights green; output time-dim==61 confirmed). However, the
production submit (3098028, cancelled) revealed a structural inefficiency:

**`scripts/eval_inference_5410.py` calls `subprocess.run(["python", "long_inference.py", ...])` 96 times.** Each subprocess pays:

- Python startup: ~5 s
- torch import: ~15 s
- Other module imports (xarray, h5py, upstream utils, custom layers): ~30 s
- Network construction (106 M-param SFNO): ~5–10 s
- Checkpoint load (`torch.load` + `state_dict`): ~5–10 s
- Mean/std + constant-boundary load: ~10 s
- Data loader init (boundary preload, 61 6-h steps): ~8 s
- BCS data loader init: ~5 s

**~95 s/IC of setup overhead × 96 ICs ≈ 2.5 h wasted on cold imports + reload.** Pure inference (60 forward passes × 96 ICs) is only ~48 min. The 4 h budget is ~76 % overhead.

**The own-track (`scripts/eval_inference.py` for the v10 zgplev emulator) does this correctly:** ONE Python process imports torch, builds the model wrapper + dataset ONCE, then loops over all 96 ICs calling `rollout_one_ic(wrapper, dataset, ...)`. SLURM comment confirms 17 016 forward passes (NWP+climate combined) finishes in ~1.5 h on H100. Mirroring that architecture for the 5410 path should bring our 4 h → ~50 min.

User-confirmed design (2026-05-08, after smoke success):
- **Deep refactor.** Extract Stepper-init from `long_inference.py main()`. Keep the model + checkpoint loaded across all 96 ICs. Rebuild only the data loaders per IC.
- **Add LP-004 patch** to upstream `long_inference.py`: `Stepper.reconfigure_for_ic(...)` method.
- **Rewrite `scripts/eval_inference_5410.py`** to import upstream in-process, build a single Stepper, loop over 96 ICs.

## Stepper.__init__ — what's reusable vs per-IC

After reading `long_inference.py:122–360`:

| Phase | Lines | Reusable across ICs? | Notes |
|---|---|---|---|
| Scalar param setup (land/ocean/mask_output, save_level_idxs, ...) | 122–184 | ✅ shared | Architecture-level config, no per-IC dependence |
| `params.long_rollout_years` | 190 | ⚠️ per-IC | Computed from final.year − init.year; for K=60 always 0 (sub-year). |
| IC data loader (`get_data_loader(... ensemble=True, init_from_nc=True)`) | 192–195 | ⚠️ per-IC | Depends on `init_datetime` (via `init_nc_filepaths`) + `val_year_start/end`. |
| `params.single_ic_offset` | 197–198 | ⚠️ per-IC | Hours since Jan 1 of init year. |
| BCS data loader (`get_data_loader(... single_ic=True)`) | 199–202 | ⚠️ per-IC | Depends on `init_datetime`, `final_datetime`, `prediction_duration_days`. |
| Perturber | 204–206 | ❌ skip | `epsilon_factor=0` for eval-track. |
| `constant_boundary_data` | 214–215 | ⚠️ per-Y, not per-IC | Depends on `val_year_start`; same for all 12 ICs of one year. |
| Model build (`SFNO(params, ...)`) | 308–336 | ✅ build once | Architecture invariant across all 96 ICs. |
| `restore_checkpoint(self.model, ...)` | 340 | ✅ load once | Same checkpoint for all ICs. |
| DDP wrap | 348–352 | ✅ once | Single-rank in eval; no-op anyway. |

**Per-IC wallclock estimate (after refactor):**
- IC data loader rebuild: ~5–8 s (boundary preload of 61 entries)
- BCS data loader rebuild: ~3–5 s (boundary preload of 61 entries)
- Constant boundary refresh: ~1 s (only when crossing Y boundary, 8× total)
- 60 forward passes: ~25–30 s
- NetCDF save (async): ~5 s
- **≈ 35–45 s per IC × 96 = 56–72 min**

**One-time setup:** ~120 s (imports + model + ckpt + mean/std).

**Total estimated wallclock: ~70–80 min.** Matches own-track timing (~1.5 h for 3× the work).

## Files to change

### 1. Upstream `long_inference.py` — LP-004 patch (`Stepper.reconfigure_for_ic`)

`/work2/.../v2.0/long_inference.py`

Add a new method on `Stepper` (between `__init__` at line 122–360 and `restore_checkpoint` at line 371):

```python
def reconfigure_for_ic(
    self,
    *,
    init_datetime,
    final_datetime,
    init_nc_filepaths,
    save_basename,
    output_dir=None,
    val_year_start=None,
    val_year_end=None,
    leap_year=None,
    no_leap_year=None,
    val_year_changed: bool,  # REQUIRED, no default — caller computes
):
    """AI-RES local patch (LP-004, 2026-05-08): reconfigure the
    Stepper for a new IC without rebuilding the model or reloading
    the checkpoint.

    The orchestrator passes `val_year_changed` explicitly (Codex
    blocker #1). The patch must NOT infer year-change from
    self.params, because the orchestrator may already have mutated
    val_year_start by the time this method runs (set_per_y_params is
    called BEFORE reconfigure_for_ic so the IC dataset rebuild sees
    the new year). Inferring `val_year_changed` from a comparison
    against self.params would yield False on every cross-year call.

    Mutates self.params and rebuilds:
      * self.data_loader, self.dataset (IC dataset, ensemble + init_from_nc)
      * self.data_loader_bcs, self.dataset_bcs (BCS dataset, single_ic)
      * self.params.single_ic_offset, self.params.long_rollout_years
      * self.constant_boundary_data (only when val_year_changed)

    Does NOT touch:
      * self.model, self.model_24h
      * checkpoint state (loaded once in __init__)
      * scalar param config (land/ocean/mask_output flags, save_level_idxs)
    """
    import torch
    from utils.data_loader_multifiles import get_data_loader

    p = self.params

    if val_year_start is not None:
        p['val_year_start'] = int(val_year_start)
    if val_year_end is not None:
        p['val_year_end'] = int(val_year_end)
    if leap_year is not None:
        p['leap_year'] = int(leap_year)
    if no_leap_year is not None:
        p['no_leap_year'] = int(no_leap_year)

    p['init_datetime'] = init_datetime
    p['final_datetime'] = final_datetime
    p['init_nc_filepaths'] = (
        list(init_nc_filepaths)
        if isinstance(init_nc_filepaths, (list, tuple))
        else [init_nc_filepaths]
    )
    p['save_basename'] = save_basename
    if output_dir is not None:
        p['output_dir'] = str(output_dir)

    p['long_rollout_years'] = p.final_datetime.year - p.init_datetime.year

    # Round-2 fix #4: validate init_nc_timestep_offset matches the IC
    # we're about to bind. set_per_ic_params already recomputed this by
    # opening the IC NC and looking up p.init_datetime in its time index;
    # if a future caller bypasses set_per_ic_params and only updates
    # init_datetime + init_nc_filepaths, the stale offset would
    # silently produce wrong output. Re-derive the expected offset
    # from the current IC NC and fail loud on mismatch.
    import xarray as xr
    expected_offsets = []
    for ic_path in p.init_nc_filepaths:
        with xr.open_dataset(ic_path, engine='netcdf4') as ds:
            expected_offsets.append(int(ds.get_index("time").get_loc(p.init_datetime)))
    if list(p.init_nc_timestep_offset) != expected_offsets:
        raise ValueError(
            f"init_nc_timestep_offset stale: got {list(p.init_nc_timestep_offset)}, "
            f"expected {expected_offsets} from current IC NC files + "
            f"init_datetime={p.init_datetime}. "
            f"Caller must invoke set_per_ic_params(...) before "
            f"reconfigure_for_ic(...) so the offset is recomputed."
        )

    # IC dataset (ensemble + init_from_nc). Re-reads init_nc_filepaths and
    # init_nc_timestep_offset (which the orchestrator updated via
    # set_per_ic_params before calling us).
    self.data_loader, self.dataset = get_data_loader(
        p, p.data_dir, dist.is_initialized(),
        year_start=p.val_year_start, year_end=p.val_year_end, train=False,
        ensemble=True, init_from_nc=True,
    )

    # single_ic_offset is read by the BCS loader's single_ic branch.
    p['single_ic_offset'] = int(
        (p.init_datetime - self.dataset.datetime_class(
            p.init_datetime.year, 1, 1, 0,
            has_year_zero=p.has_year_zero,
        )).total_seconds() // 3600
    )

    # BCS dataset (single_ic=True). Reads prediction_duration_days from
    # params, which the per-Y yaml pinned at (K+1)*6/24 days.
    self.data_loader_bcs, self.dataset_bcs = get_data_loader(
        p, p.data_dir, dist.is_initialized(),
        year_start=p.init_datetime.year, year_end=p.final_datetime.year,
        train=False, single_ic=True,
    )

    # Constant boundary depends on val_year_start; refresh only on Y change.
    if val_year_changed or not hasattr(self, 'constant_boundary_data'):
        self.constant_boundary_data = (
            self.dataset.constant_boundary_data.unsqueeze(0)
            * torch.ones(p.batch_size, 1, 1, 1)
        )
        self.constant_boundary_data = self.constant_boundary_data.to(
            self.device, non_blocking=True,
        )
```

**No changes to `Stepper.__init__`, `Stepper.predict`, or any other method.** The patch is purely additive.

Tracked in `docs/2026-05-04_makani_local_patches.md` as LP-004 with the same strict-marker verification pattern as LP-003 (grep for `reconfigure_for_ic`, expect ≥ 1 occurrence).

### 2. `src/sfno_inference_5410/upstream_hydration.py` — NEW (Codex blocker #2 fix)

The exhaustive helper hydrates the `Stepper(...)` params surface and applies one 5410-specific correction: `nc_bc_offset=0`. Upstream standalone `long_inference.py` had hard-coded `18`, but the model's validation autoregression consumes current-step boundary forcing. The 18-hour offset was later found to corrupt the NWP rollout and invalidate the 2026-05-08 scorecard. Other hydrated fields include `init_nc_timestep_offset` (recomputed per-IC by reading each IC NC's time index), `run_iter`, `has_diagnostic`, `num_ensemble_members`, `ensemble_members_per_pred`, `world_size`, `batch_size`, `local_rank`, `enable_amp`, `experiment_dir`, `checkpoint_dir`, `best_checkpoint_path`, `latest_checkpoint_path`, `checkpoint_path_globstr`, `resuming`, `log_to_wandb`, `log_to_screen`, plus cftime-normalized `init_datetime` / `final_datetime` via `_dt_cls(...)`.

Module surface:

```python
def hydrate_static_params(yaml_path: Path, K: int, *, run_num: str = "5410",
                           config_section: str = "SFNO", world_rank: int = 0,
                           local_rank: int = 0) -> "YParams":
    """Load YParams + apply EVERY one-time mutation upstream main() does
    before Stepper construction. Returns a single hydrated params object.
    
    Mirrors long_inference.py:1252-1445 verbatim except for the per-IC
    fields (init_datetime, final_datetime, init_nc_filepaths,
    init_nc_timestep_offset, save_basename) which set_per_ic_params
    handles. Values pinned: nc_bc_offset=0, run_iter=1, world_size=1,
    batch_size=1, enable_amp=True, log_to_wandb=False, resuming=True,
    has_diagnostic from diagnostic_variables list, num_ensemble_members
    default 1.
    """
    ...

def set_per_ic_params(params, *, init_datetime, final_datetime,
                      init_nc_filepaths, save_basename, output_dir) -> None:
    """Mutate ONLY the per-IC fields. Does not swap params object.
    
    - init_datetime, final_datetime: cftime-normalized via
      datetime_class_from_calendar(params.calendar).
    - init_nc_filepaths: single-element list.
    - init_nc_timestep_offset: RECOMPUTED by opening the IC NC file and
      looking up params.init_datetime in its time index (same as
      long_inference.py:1340-1344).
    - save_basename: str.
    - output_dir: str (constant across 96 ICs but exposed for symmetry).
    """
    ...

def set_per_y_params(params, *, Y: int) -> None:
    """Mutate ONLY the per-Y fields (val_year_start, val_year_end,
    leap_year, no_leap_year). All 8 eval-track yamls share an identical
    architecture/checkpoint/normalization config; only year fields
    differ — see assert_yamls_share_static_arch."""
    ...

def assert_yamls_share_static_arch(yaml_paths: list[Path]) -> None:
    """Cross-year preflight: every architecture / variables / levels /
    checkpoint / normalization / precision field must match across all 8
    yamls. Only val_year_start, val_year_end, leap_year, no_leap_year
    may differ. Fail loud if any other field diverges — proves the
    model can be built once from one params and reused across all Y.
    """
    ...
```

`hydrate_static_params` is unit-tested against an explicit allowlist of every field upstream `main()` sets — see Tests §5.

### 3. `scripts/eval_inference_5410.py` — full rewrite (no params swap)

Per Codex blocker #1: keep ONE hydrated params object across all 96 ICs. **Never** swap `stepper.params`. Compute year-change BEFORE mutation. Mutate only the four per-Y fields when crossing Y, and the five per-IC fields every IC.

```python
def main():
    args = _parse_args()
    K = args.K
    assert_K_explicit(K)
    plan = build_run_plan(args.run_root, args.config_dir, K=K)

    # Preflight: yaml horizon (gate #3), upstream patches (#6 + new #7).
    yaml_paths = sorted({Path(e['yaml']) for e in plan})
    assert_upstream_patched(_UPSTREAM_LONG_INFERENCE)              # LP-003
    assert_upstream_patched_lp004(_UPSTREAM_LONG_INFERENCE)        # LP-004
    for yp in yaml_paths:
        assert_yaml_horizon(yp, K)
    assert_yamls_share_static_arch(yaml_paths)                     # NEW

    out_dir = plan[0]['output_dir']

    if not args.launch:
        # Dry-run prints the plan; does NOT enforce output-dir emptiness
        # (Codex round-2 fix #3 — dry-run is exactly when we may want to
        # inspect a stale run-root).
        for entry in plan:
            print(f"[ic Y={entry['Y']} s={entry['s']:04d}] init={entry['init_datetime']} "
                  f"final={entry['final_datetime']} forecast_K={K}")
        return 0

    # Output-dir hygiene (Codex major #3): empty or refuse, only on launch.
    assert_output_dir_empty(out_dir)

    # === one-time setup (paid ONCE across all 96 ICs) ===
    os.chdir(_UPSTREAM_REPO)
    sys.path.insert(0, str(_UPSTREAM_REPO))

    # Build the SINGLE params object hydrated from yaml_paths[0].
    # Architecture/ckpt fields are static across all 8 yamls (verified
    # by assert_yamls_share_static_arch above), so any one yaml suffices.
    params = hydrate_static_params(yaml_paths[0], K=K)
    set_per_y_params(params, Y=plan[0]['Y'])
    set_per_ic_params(params,
        init_datetime=plan[0]['init_datetime'],
        final_datetime=plan[0]['final_datetime'],
        init_nc_filepaths=[plan[0]['ic_nc']],
        save_basename=plan[0]['save_basename'],
        output_dir=plan[0]['output_dir'],
    )

    # Build Stepper once. Loads model + checkpoint + mean/std.
    # async_save=False on the first equivalence smoke (Codex major #2);
    # production may flip to True once the drain invariant is documented.
    from long_inference import Stepper  # type: ignore
    stepper = Stepper([params], world_rank=0, async_save=args.async_save)

    # First IC's reconfigure_for_ic + predict + post-assertions.
    _run_one_ic(stepper, plan[0], K, val_year_changed=True)

    # === per-IC loop (95 more times) ===
    t0 = time.time()
    for i, entry in enumerate(plan[1:], start=2):
        t_ic = time.time()
        # Compute val_year_changed BEFORE any mutation (Codex blocker #1).
        prev_Y = plan[i - 2]['Y']
        val_year_changed = entry['Y'] != prev_Y
        if val_year_changed:
            set_per_y_params(stepper.params, Y=entry['Y'])
        set_per_ic_params(stepper.params,
            init_datetime=entry['init_datetime'],
            final_datetime=entry['final_datetime'],
            init_nc_filepaths=[entry['ic_nc']],
            save_basename=entry['save_basename'],
            output_dir=entry['output_dir'],
        )
        _run_one_ic(stepper, entry, K, val_year_changed=val_year_changed)
        print(f"[{i:>2}/{len(plan)}] {entry['save_basename']} "
              f"({time.time() - t_ic:.1f}s)")

    elapsed_min = (time.time() - t0) / 60.0
    print(f"[orchestrator] all {len(plan)} ICs done in {elapsed_min:.1f} min")

    # Postflight (Codex major #3): exact-filename + time==K+1 audit.
    assert_output_dir_complete(out_dir, plan, K)
    return 0


def _run_one_ic(stepper, entry, K, *, val_year_changed):
    """Call reconfigure_for_ic + predict + four post-reconfigure assertions
    (Codex blocker #3): proves the rollout itself is K+1 raw steps long,
    not just that the saved file looks right."""
    stepper.reconfigure_for_ic(
        init_datetime=entry['init_datetime'],
        final_datetime=entry['final_datetime'],
        init_nc_filepaths=[entry['ic_nc']],
        save_basename=entry['save_basename'],
        output_dir=entry['output_dir'],
        val_year_start=entry['Y'],
        val_year_end=entry['Y'] + 1,
        leap_year=entry['Y'],
        no_leap_year=entry['Y'],
        val_year_changed=val_year_changed,
    )

    # Codex blocker #3 + round-2 fix #2: verify the loader rebuild
    # actually shortened the rollout. Use explicit `if ... raise` (NOT
    # `assert`) so `python -O` cannot disable these checks. Same pattern
    # as the existing preflight helpers.
    if len(stepper.data_loader_bcs) != K + 1:
        raise ValueError(
            f"BCS loader length {len(stepper.data_loader_bcs)} != K+1={K+1} "
            f"(Y={entry['Y']}, s={entry['s']}); "
            f"prediction_duration_days may not have propagated"
        )
    if len(stepper.data_loader) != 1:
        raise ValueError(
            f"IC loader length {len(stepper.data_loader)} != 1 "
            f"(Y={entry['Y']}, s={entry['s']}); "
            f"single-IC invariant violated"
        )
    if len(stepper.params.init_nc_filepaths) != 1:
        raise ValueError(
            f"init_nc_filepaths len != 1: {stepper.params.init_nc_filepaths}"
        )
    expected_final = entry['init_datetime'] + dt.timedelta(hours=(K + 1) * 6)
    if stepper.params.final_datetime != expected_final:
        raise ValueError(
            f"final_datetime {stepper.params.final_datetime} != "
            f"init + (K+1)*6h = {expected_final} "
            f"(Y={entry['Y']}, s={entry['s']})"
        )

    stepper.predict()
```

`reconfigure_for_ic` takes `val_year_changed` explicitly (Codex blocker #1: pre-computed in the orchestrator from `prev_Y vs entry['Y']`, never inferred from the params object's current state). This makes the Y-crossing semantics auditable.

### 3. `scripts/submit_eval_inference_5410.slurm` — drop subprocess loop, shorten wallclock

- Wallclock: `4:00:00 → 2:00:00` (~1h estimate + 1h margin against I/O variance / cold cache).
- The orchestrator is now a single `python eval_inference_5410.py --K $K --launch` call; no inner subprocess loop. The `cd $UPSTREAM_REPO` happens inside the orchestrator (already does), so SLURM CWD stays at REPO_ROOT.
- All preflights (patch markers, yaml horizon) stay as-is.

### 4. `scripts/submit_eval_inference_5410_smoke.slurm` — repurpose as in-process sanity

- Smoke now exercises the SAME `eval_inference_5410.py` codepath as production, with `--years 121` → only 12 ICs (or `--years 121 --limit-ics 1` if we add that flag) → 1 IC.
- Validates: in-process Stepper builds correctly, reconfigure_for_ic works at least once, output time-dim == K+1.

### 5. Tests (Codex major #1: A/B equivalence is required, mocking is not enough)

**New unit tests:**

- `tests/sfno_inference_5410/test_lp004_patch_present.py` (NEW): byte-grep for `def reconfigure_for_ic` in `/work2/.../v2.0/long_inference.py`. Strict count: exactly 1 occurrence. Skipped on machines without upstream tree.
- `tests/sfno_inference_5410/test_upstream_hydration.py` (NEW): unit tests with **two separate allowlists** matching the helper boundaries (Codex round-2 fix #1). Asserts that every field upstream `main()` mutates between argparse and Stepper construction is present at the right phase:

  ```python
  # Set by hydrate_static_params() alone — i.e., before any IC is bound.
  _STATIC_ATTRS = (
      "run_iter", "has_diagnostic", "num_ensemble_members",
      "ensemble_members_per_pred", "nc_bc_offset",
      "world_size", "batch_size", "local_rank", "enable_amp",
      "experiment_dir", "checkpoint_dir", "best_checkpoint_path",
      "latest_checkpoint_path", "checkpoint_path_globstr", "resuming",
      "log_to_wandb", "log_to_screen",
  )

  # Set by set_per_ic_params() — only present after the FIRST IC is bound.
  _PER_IC_ATTRS = (
      "init_datetime", "final_datetime", "init_nc_filepaths",
      "init_nc_timestep_offset", "save_basename", "output_dir",
  )

  # Set by set_per_y_params() — only present after a Y is bound.
  _PER_Y_ATTRS = (
      "val_year_start", "val_year_end", "leap_year", "no_leap_year",
  )
  ```
  
  Tests:
  - `test_static_attrs_present_after_hydrate`: after `hydrate_static_params(yaml_path, K=60)`, every `_STATIC_ATTRS` field is set; **none** of `_PER_IC_ATTRS` is set (since no IC has been bound yet — the per-IC fields would be carry-over from the yaml at most, and `hydrate_static_params` must clear them or assert they are absent).
  - `test_static_pinned_values`: `nc_bc_offset == 0`, `world_size == 1`, `batch_size == 1`, `enable_amp == True`, `log_to_wandb == False`, `resuming == True`, `run_iter == 1`.
  - `test_per_ic_attrs_present_after_set_per_ic`: after `set_per_ic_params(...)`, every `_PER_IC_ATTRS` field is set, including `init_nc_timestep_offset` recomputed from the IC NC's time index.
  - `test_per_y_attrs_present_after_set_per_y`: after `set_per_y_params(params, Y=121)`, all four `_PER_Y_ATTRS` are set (`val_year_start=121`, `val_year_end=122`, `leap_year=121`, `no_leap_year=121`).
  - `test_full_main_equivalence`: after `hydrate_static_params(...) + set_per_y_params(...) + set_per_ic_params(...)`, every union-attr is present — i.e., the params object matches what upstream `main()` produces just before `Stepper(params_list, ...)` is called.

  This is the explicit guard against the BCS phase bug (`nc_bc_offset=18` shifts first-step boundary forcing to `init+18h`) and the round-2 fix for the helper/test allowlist mismatch.
- `tests/sfno_inference_5410/test_static_arch_invariant.py` (NEW): exercises `assert_yamls_share_static_arch` against the live 8 per-Y yamls produced by `build_all(K=60)`. Asserts every architecture/checkpoint/normalization/precision field matches across all 8; only `val_year_start`, `val_year_end`, `leap_year`, `no_leap_year` may differ.

**Updated unit tests:**

- `tests/sfno_inference_5410/test_runtime_args_5410.py`: `build_run_plan(...)` now returns per-IC kwargs dicts (no argv). Update existing assertions to check the dict shape (`init_datetime`, `final_datetime`, `ic_nc`, `save_basename`, `Y`, `s`, `output_dir`, `yaml`).
- `tests/sfno_inference_5410/test_eval_driver_K.py`: same.

**A/B equivalence smokes (Codex major #1: this is the must-have):**

These run the OLD subprocess path AND the NEW in-process path against the same IC(s) and compare outputs. They live in `tests/sfno_inference_5410/integration/test_ab_equivalence.py` and are gated behind a `RUN_AB_TESTS=1` env var (so `pytest tests/sfno_inference_5410/` doesn't fire them by default). They require GPU + the upstream tree.

**Legacy-launcher fixture (Codex round-2 minor recommendation):** Once `eval_inference_5410.py` is rewritten, the original subprocess-based path no longer exists in the working orchestrator. The A/B tests must keep an independent reference launcher. Captured at `tests/sfno_inference_5410/integration/_legacy_subprocess_launcher.py` — a minimal helper that takes a per-IC entry dict and runs:

```python
def launch_legacy_subprocess(entry, *, upstream_repo, output_dir, K):
    """Reference launcher: invokes long_inference.py via subprocess.run
    for a single IC, identical to the pre-refactor eval_inference_5410.py
    behavior. Lives in the test fixtures dir so it survives the orchestrator
    rewrite and remains the canonical 'old path' for A/B comparison.
    """
    init_dt = entry['init_datetime']
    final_dt = init_dt + dt.timedelta(hours=(K + 1) * 6)
    argv = [
        sys.executable, "-u", str(upstream_repo / "long_inference.py"),
        "--run_num", "5410",
        "--yaml_config", str(entry['yaml']),
        "--config", "SFNO",
        "--init_datetime", init_dt.strftime("%Y-%m-%d_%H:%M:%S"),
        "--final_datetime", final_dt.strftime("%Y-%m-%d_%H:%M:%S"),
        "--init_nc_filepaths", str(entry['ic_nc']),
        "--output_dir", str(output_dir),
        "--save_basename", entry['save_basename'],
    ]
    # NOTE: omits --async_save so the legacy reference is sync (matches
    # v2.1 default for the new path's first smoke).
    return subprocess.run(argv, cwd=str(upstream_repo),
                          env=os.environ.copy(), check=True)
```

The launcher is intentionally minimal — no preflight, no plan-building, no orchestration. Just one IC via subprocess. The A/B tests build the per-IC entries themselves (using the same `build_argv_for_ic` / `build_run_plan` helpers as the new orchestrator) and call this launcher for the "old" path.

```python
# Three test cases. For each, run both paths and assert NetCDF outputs
# match on dims, coords, var names, time range, and numerical values
# at rtol=1e-5.

@pytest.mark.skipif(...)  # require GPU + upstream + RUN_AB_TESTS=1
def test_one_ic_equivalence(tmp_path):
    """Y=121 s=0, K=60. Old subprocess vs new in-process."""
    ...

def test_two_same_year_ics_equivalence(tmp_path):
    """Y=121, s=0 then s=122. Tests reconfigure_for_ic within a year
    (no Y crossing — constant_boundary_data should NOT refresh)."""
    ...

def test_cross_year_pair_equivalence(tmp_path):
    """Y=121 s=0 then Y=122 s=0. Tests reconfigure_for_ic across a Y
    boundary. constant_boundary_data MUST refresh; val_year_start /
    leap_year / no_leap_year all change."""
    ...
```

For each test, the comparison is:
- `ds.dims == ds_ref.dims`
- `set(ds.data_vars) == set(ds_ref.data_vars)` (all of: pl, tas, pr_6h, ta, ua, va, hus, zg)
- `ds.time.values == ds_ref.time.values` (cftime-equal)
- For every var: `np.allclose(ds[var].values, ds_ref[var].values, rtol=1e-5, atol=1e-7)`

If any variable diverges beyond tolerance, the equivalence test fails with a per-var max-error report. This catches: BCS phase errors, stale constant_boundary_data, missing runtime params (e.g., `nc_bc_offset=18` would shift the BCS time index and produce numerically different rollouts).

The cross-year pair (third test) is the critical one. Codex specifically called it out — it exercises the most state in `reconfigure_for_ic` (val_year_changed=True path).

### 6. `docs/2026-05-04_makani_local_patches.md`

Append `## LP-004 — In-process Stepper.reconfigure_for_ic` (eval-5410 in-process orchestrator). Single hunk: a new method added at line ~360 of long_inference.py. Verification: `grep -c "def reconfigure_for_ic" long_inference.py` returns exactly 1.

### 7. `.claude/skills/eval-sfno-5410/SKILL.md`

Update production timing line: "~50–80 min on 1×H100" instead of "~4h". Note the LP-004 patch dependency (alongside LP-003).

## Output directory hygiene (Codex major #3)

The current production SLURM only checks `wc -l upstream_raw/*.nc == 96` post-flight. Codex caught that this passes if old + new files coexist in the same dir (the contamination state we hit on 3098028's previous run). Mixed K=60 and year-long files would have wrong `time` dims and silently mask a partial rerun.

Two new helpers in `src/sfno_inference_5410/preflight.py`:

```python
def assert_output_dir_empty(out_dir: Path) -> None:
    """Pre-launch: refuse if upstream_raw is non-empty.
    
    Operator must explicitly choose: backup, delete, or fresh RUN_ROOT.
    The orchestrator will not silently overwrite or coexist with
    prior outputs. The error message lists the prior files + their
    time-dim sizes so the operator can decide quickly.
    """
    nc_files = list(out_dir.glob("Y*_member*_y*.nc"))
    if nc_files:
        raise ValueError(
            f"output dir {out_dir} is non-empty ({len(nc_files)} prior "
            f"NetCDFs). Backup, delete, or use a fresh RUN_ROOT before "
            f"submitting. First few: {[p.name for p in nc_files[:5]]}"
        )

_EXPECTED_VARS = frozenset({"pl", "tas", "pr_6h", "ta", "ua", "va", "hus", "zg"})

def assert_output_dir_complete(out_dir: Path, plan, K: int,
                                expected_vars: frozenset = _EXPECTED_VARS) -> None:
    """Post-flight: every (Y, s) in plan has a NetCDF with the right
    time dim AND the right variable set.

    Stronger than `wc -l == 96` (Codex major #3 + round-2 minor):
      * exact filename match per (Y, s) tuple in the plan;
      * no extra unexpected files;
      * each NetCDF's time dim equals K+1 (rules out year-long sentinels
        117/118/1460/1464 from a prior run that didn't get cleaned up);
      * each NetCDF has the exact 8-variable set {pl, tas, pr_6h, ta, ua,
        va, hus, zg} — guards against truncated writes or schema drift.
    """
    import xarray as xr
    expected = {f"{e['save_basename']}_member000_y{e['Y']:04d}.nc" for e in plan}
    actual = {p.name for p in out_dir.glob("Y*_member*_y*.nc")}
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise ValueError(f"missing {len(missing)} expected NetCDFs: "
                         f"{sorted(missing)[:5]}...")
    if extra:
        raise ValueError(f"unexpected extra NetCDFs in output dir: "
                         f"{sorted(extra)[:5]}...")
    for fname in expected:
        path = out_dir / fname
        assert_output_time_dim(path, K)
        with xr.open_dataset(path) as ds:
            actual_vars = frozenset(ds.data_vars)
        if actual_vars != expected_vars:
            raise ValueError(
                f"{fname}: data_vars {sorted(actual_vars)} != "
                f"expected {sorted(expected_vars)} "
                f"(missing: {sorted(expected_vars - actual_vars)}, "
                f"extra: {sorted(actual_vars - expected_vars)})"
            )
```

Both gates wired into the orchestrator (pre-launch + post-launch) AND the production SLURM (final summary).

## Async-save invariant (Codex major #2)

Upstream's `save_prediction()` (long_inference.py:1197) reads `self.params.save_basename` and `output_dir` **at save time**, not at queue time. If we mutate `params.save_basename` for the next IC before the previous IC's async save thread drains, filenames collide / overwrite.

**v2.1 default:** `args.async_save=False` for the orchestrator. The first equivalence smoke runs with `async_save=False` so the post-IC assertions and reconfigure can't race a still-pending save. Production may flip to `True` later, but only after we either:

- Add an explicit `stepper.drain_save_queue()` method that the orchestrator calls between ICs, OR
- Document and verify that `predict()` already awaits the save task before returning (the smoke log shows it does — but Codex correctly notes this is undocumented and load-bearing).

The `--async-save` flag stays exposed in the orchestrator argparse. Production SLURM passes `--no-async-save` (or omits the flag, since default is False) on the first production submit.

## Preflight gates (full list, post-Codex)

| # | Requirement | Where enforced |
|---|---|---|
| 1 | K is explicit | `--K` required; `assert_K_explicit` |
| 2 | `final_datetime = init + (K+1)·6h` | `assert_final_datetime_matches` in build_argv_for_ic + post-reconfigure assertion in `_run_one_ic` |
| 3 | YAML horizon: `ensemble_inference_hours = (K+1)·6` AND `prediction_duration_days = (K+1)·6/24` | `assert_yaml_horizon` per per-Y yaml |
| 4 | Output NetCDF time dim == K+1 | `assert_output_time_dim` in postflight |
| 5 | Logs show requested end date, not Y+1-01-01 | per-IC log line in orchestrator + smoke SLURM |
| 6 | LP-003 upstream patch applied (4+2 markers) | `assert_upstream_patched` |
| 7 | LP-004 upstream patch applied (`reconfigure_for_ic` exists) | `assert_upstream_patched_lp004` |
| 8 | All 8 per-Y yamls share static architecture/ckpt/norm config | `assert_yamls_share_static_arch` (NEW) |
| 9 | Output dir empty before launch | `assert_output_dir_empty` (NEW) |
| 10 | Output dir has exactly the 96 expected files post-launch | `assert_output_dir_complete` (NEW) |
| 11 | Per-IC: `len(stepper.data_loader_bcs) == K+1`, `len(stepper.data_loader) == 1`, `len(init_nc_filepaths) == 1`, final = init+(K+1)·6h | `_run_one_ic` post-reconfigure block (NEW — Codex blocker #3) |

## Codex review questions (round 2 — post round-1 revisions)

**Round-1 blockers — addressed in v2, please confirm:**

- **Blocker 1 (params swap):** v2 keeps ONE hydrated params object across all 96 ICs. `set_per_y_params` and `set_per_ic_params` mutate only allowed fields. `val_year_changed` is computed in the orchestrator BEFORE any mutation (from `entry['Y'] != prev_Y`) and passed explicitly to `reconfigure_for_ic`, never inferred from the params object's current state. The static-arch invariant across all 8 yamls is asserted at preflight (`assert_yamls_share_static_arch`).
- **Blocker 2 (runtime params injected by main()):** v2 introduces `hydrate_static_params` with a unit-tested explicit allowlist of 22 fields. As corrected on 2026-05-09, the 5410 NWP eval pins `nc_bc_offset = 0` to match validation boundary phase. See `test_upstream_hydration.py` and `test_boundary_phase_alignment.py`.
- **Blocker 3 (assert K=60 after loader rebuild):** v2 adds four post-reconfigure assertions in `_run_one_ic`: BCS loader length == K+1, IC loader length == 1, init_nc_filepaths length == 1, final == init + (K+1)·6h. Fires every IC.

**Round-1 majors — addressed in v2, please confirm:**

- **Major 1 (test coverage):** v2 adds A/B equivalence integration tests for one-IC, two same-year ICs, and a cross-year pair (Y=121 s=0 → Y=122 s=0). Compares NetCDF coords + var names + time range + numerical values at rtol=1e-5.
- **Major 2 (async save):** v2 defaults `async_save=False`. The first equivalence smoke runs synchronously. Production flip to True is deferred until either an explicit drain method lands or upstream's await-before-return is verified-and-documented.
- **Major 3 (output dir hygiene):** v2 adds `assert_output_dir_empty` (pre-launch) and `assert_output_dir_complete` (post-launch with exact filename match + per-file time-dim==K+1 audit).

**Open round-2 questions:**

1. **Stepper state leakage.** Does `Stepper.predict()` leave instance attributes mutated that need reset between ICs? Specifically: `current_datetime`, `current_year`, `time_step_in_year`, `output_surface`, `output_upper_air`, `output_diagnostic`. These appear inside `predict()`'s body as locals (per long_inference.py:550-590), so they should be re-initialized each call — but please verify by reading the full async/sync paths in upstream.
2. **YParams mutability.** `params['init_datetime'] = ...` mutates the params dict. Does any downstream consumer hold a reference to the OLD params dict that would diverge? Specifically: the Dataset built at line 192 stores `self.params = params` — when we mutate params after rebuilding the Dataset, does the new Dataset see the new value? In v2 the order is: mutate params first, THEN call `get_data_loader(params, ...)` which builds a new Dataset → safe.
3. **`predict()` re-entry safety.** Calling `Stepper.predict()` 96 times in one process — any module-level state (logging handlers, cudnn benchmark, torch.manual_seed) that needs resetting? `torch.manual_seed(0)` is set once in `main()` (line 1314). If `predict()` consumes randomness, we'd want a fresh seed per IC for reproducibility but determinism on a per-IC basis is fine.
4. **`init_nc_timestep_offset` correctness.** `set_per_ic_params` opens each IC NC file (`xr.open_dataset`) to compute the time index. For 96 ICs that's 96 NC opens. Combined with the data loader's preload, total NC opens per orchestrator ≈ 96 + 96 + 8 ≈ 200. Acceptable — these are small files.
5. **Async-save drain.** The smoke log we have shows `predict()` ending with "DONE ---- rank 0" before main() exits. But that's main() exiting; does `predict()` itself await async save tasks before returning? If yes (verify by reading the predict() implementation), `async_save=True` would be safe and we could enable it for production. If unclear, stick with `async_save=False` for the first equivalence run.
6. **CWD behavior across `predict()` calls.** Upstream may write `out.log` or `hyperparams.yaml` relative to CWD. Verify that running in upstream CWD doesn't accumulate per-IC log noise or overwrite a shared file 96 times.
7. **DDP no-op assumption.** Single-rank inference has `dist.is_initialized()` == False. Confirm that no part of `Stepper.__init__` (especially the DDP wrapping at line 348) tries to access NCCL state if dist is uninitialized. (Smoke succeeded in single-rank, so this is empirically OK; please confirm by reading.)
8. **Equivalence tolerance.** rtol=1e-5 atol=1e-7 on float32 outputs. Is this tight enough to catch a genuinely-incorrect IC? Or should we go tighter (e.g., bit-exact)? For deterministic CUDA + fixed seed, bit-exact should be achievable, but cudnn nondeterminism may push us to ~1e-5.

## Verification plan (v2.1 — sequenced gates, each must pass before next)

1. **Apply LP-004 patch** to `/work2/.../v2.0/long_inference.py`. Verify: `grep -c "def reconfigure_for_ic" long_inference.py` == 1.
2. **Implement** `upstream_hydration.py`, the orchestrator rewrite, the new preflight helpers, and the updated tests.
3. **Run unit tests:** `pytest tests/sfno_inference_5410/`. All existing 199 + new unit tests pass. The hydration allowlist test is the load-bearing one.
4. **A/B equivalence — one IC** (Codex major #1, gate A): `RUN_AB_TESTS=1 pytest tests/sfno_inference_5410/integration/test_ab_equivalence.py::test_one_ic_equivalence`. Old subprocess vs new in-process for Y=121 s=0 K=60. NetCDF outputs must match at rtol=1e-5. **If this fails, abort** — the in-process rebuild is incorrect somewhere.
5. **A/B equivalence — two same-year ICs** (gate B): exercises reconfigure_for_ic within Y=121 (no Y crossing). val_year_changed=False on the second call.
6. **A/B equivalence — cross-year pair** (gate C, Codex's must-have): Y=121 s=0 then Y=122 s=0. Exercises the val_year_changed=True path in reconfigure_for_ic. Verifies constant_boundary_data refreshes.
7. **Smoke (1 IC) via repurposed smoke SLURM** to confirm the SLURM-launched path matches the pytest path: `K=60 sbatch scripts/submit_eval_inference_5410_smoke.slurm`. Time-dim==61 on output.
8. **Mini-prod (12 ICs, Y=121 only):** `K=60 sbatch ... --years 121`. Wallclock target: ~10 min. Output: 12 NetCDFs each time-dim==61. assert_output_dir_complete fires post-flight.
9. **Codex round-2 review** of the full diff (LP-004 hunk + upstream_hydration.py + orchestrator + new preflights + new tests + patch ledger).
10. **Production (96 ICs, all Y):** `K=60 sbatch scripts/submit_eval_inference_5410.slurm`. Wallclock target: ~50–80 min. assert_output_dir_complete confirms the 96 expected files + each time-dim==61.

Gates A → B → C cannot be skipped. They are the difference between "we counted 96 calls" and "we proved the new path produces the same NetCDFs as the old path".

## Out of scope

- DDP fan-out across multiple GPUs. Single-rank H100 is fast enough at K=60.
- Multi-IC batched inference (Codex's "depth refactor + multi-IC batch" option). Reconsider only if 1-GPU wallclock exceeds ~1.5 h.
- Score / report SLURM gluing (still missing; tracked separately).

## Revision history

- **v1 (2026-05-08, evening):** initial draft, post-smoke-success, post-prod-cancel. Designed in response to user's observation that 4 h is wildly excessive given the own-track does similar work in 1.5 h via single-process architecture.
- **v2 (2026-05-08, evening):** Codex round-1 revisions. Three blockers and three majors addressed:
  1. **No `stepper.params` swap** — single hydrated params object across all 96 ICs; mutate only allowed per-IC + per-Y fields; `val_year_changed` computed in orchestrator BEFORE mutation, passed explicitly to `reconfigure_for_ic`.
  2. **Exhaustive `hydrate_static_params` helper** unit-tested against a 22-field allowlist mirroring upstream `main()` lines 1252-1445. Corrected on 2026-05-09 to pin `nc_bc_offset = 0` for NWP boundary-phase alignment.
  3. **Post-reconfigure assertions** (Codex blocker #3): `len(data_loader_bcs) == K+1`, `len(data_loader) == 1`, `len(init_nc_filepaths) == 1`, `final == init + (K+1)·6h`. Every IC.
  4. **A/B equivalence integration tests** for one IC, two same-year ICs, and a cross-year pair. NetCDF compared at rtol=1e-5 across coords + var names + time + values.
  5. **`async_save=False` default** for first equivalence smoke (Codex major #2). Production flip to True deferred until drain semantics verified.
  6. **Output dir hygiene** (Codex major #3): `assert_output_dir_empty` pre-launch + `assert_output_dir_complete` post-launch with exact-filename + per-file time-dim audit.
  7. **`assert_yamls_share_static_arch` preflight** asserts all 8 per-Y yamls match on architecture/checkpoint/normalization/precision; only year fields differ. Justifies hydrating params from yaml_paths[0] and reusing the model across all Y.
  8. **`reconfigure_for_ic(val_year_changed: bool, ...)` is now a required keyword arg**, not inferred from `self.params`. Caller (orchestrator) computes from `entry['Y'] != prev_Y`. Eliminates the "compares against the already-replaced params" failure mode Codex identified.
- **v2.1 (this doc):** Codex round-2 conditional-approval edits:
  1. **Hydration test allowlist split** into `_STATIC_ATTRS` (post-`hydrate_static_params`) vs `_PER_IC_ATTRS` (post-`set_per_ic_params`) vs `_PER_Y_ATTRS` (post-`set_per_y_params`), so each helper is tested at its own boundary. Plus a `test_full_main_equivalence` that checks the union after all three run.
  2. **Raw `assert` replaced with `if ... raise ValueError(...)`** in `_run_one_ic`'s four post-reconfigure checks, mirroring the existing preflight helpers — survives `python -O`.
  3. **`assert_output_dir_empty` gated on `args.launch`** — dry-run no longer refuses on stale output dirs (dry-run is exactly the time we want to inspect them).
  4. **`reconfigure_for_ic` validates `init_nc_timestep_offset`** by re-deriving from the current IC NC files and comparing. If a future caller bypasses `set_per_ic_params`, the helper fails loud with a clear "call set_per_ic_params first" message.
  5. **`assert_output_dir_complete` also checks the variable set** (`pl, tas, pr_6h, ta, ua, va, hus, zg`) of each NetCDF, not just filenames + time dim — catches truncated writes or schema drift.
