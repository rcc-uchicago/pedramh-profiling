# 5410 NWP eval — in-process orchestrator — Codex round-3 diff summary

_Date: 2026-05-08 evening. Branch: zgplev-migration-dsi-bootstrap._
_All three A/B equivalence gates have passed; production submit is gated on this round-3 review._

## What you're reviewing

Implementation of v2.1 of `docs/2026-05-08_sfno_5410_inproc_orchestrator_plan.md` (Codex-approved 2026-05-08). Refactors the 96-IC SFNO-5410 NWP eval from "spawn 96 fresh `python long_inference.py` subprocesses" to "build upstream's `Stepper` once + loop calling `Stepper.reconfigure_for_ic + predict` 96 times in one process". Mirrors the own-track architecture (`scripts/eval_inference.py`).

Expected wallclock: 4 h → ~50–80 min. Pure inference time is ~48 min; per-IC subprocess setup overhead (~2.5 h on 96 ICs) is gone.

## Exact files changed

### Upstream (outside the repo)

`/work2/11114/zhixingliu/stampede3/SFNO_Climate_Emulator/artifacts/derecho_glade/PanguWeather/v2.0/long_inference.py`

- Pre-existing **LP-003** (4 allocator + 2 continuation hunks for partial-horizon K=60). Already shipped earlier today; verified by smoke 3097936.
- New **LP-004** (single hunk, ~120 lines): `Stepper.reconfigure_for_ic` method inserted between the end of `Stepper.__init__` and `_get_inference_duration`. Diff vs pre-K-patch backup is 236 lines total (LP-003 + LP-004).
- Backup at `long_inference.py.bak.20260508_pre_K_patch` for byte-comparison if needed.

Strict-count verification:
```bash
grep -c 'min(next_year_jan1, self.params.final_datetime)' long_inference.py  # → 4
grep -c 'current_datetime < self.params.final_datetime'   long_inference.py  # → 2
grep -c 'def reconfigure_for_ic'                           long_inference.py  # → 1
```

### Repo (all under `~/projects/SFNO_Climate_Emulator`, untracked since the 5410 work pre-dates this branch's first commit)

| File | Lines | Role |
|---|---:|---|
| `scripts/eval_inference_5410.py` | 427 | Orchestrator (full rewrite) |
| `scripts/build_5410_yaml_override.py` | 96 | Per-Y yaml builder (`--K` plumbed in) |
| `scripts/submit_eval_inference_5410.slurm` | 144 | Production SLURM (96 ICs, 2 h budget) |
| `scripts/submit_eval_inference_5410_smoke.slurm` | 129 | Smoke (1 IC via `--limit-ics 1`) |
| `scripts/submit_ab_gate_a.slurm` | 55 | A/B gate A wrapper |
| `scripts/submit_ab_gate_b.slurm` | 49 | A/B gate B wrapper |
| `scripts/submit_ab_gate_c.slurm` | 55 | A/B gate C wrapper |
| `scripts/submit_legacy_baseline.slurm` | 50 | Legacy-vs-legacy nondeterminism floor |
| `src/sfno_inference_5410/upstream_hydration.py` | 372 | NEW — mirrors upstream `main()` mutations |
| `src/sfno_inference_5410/preflight.py` | 311 | + 4 new helpers |
| `src/sfno_inference_5410/stampede3_yaml_override.py` | 322 | K threaded through (from prior K=60 work) |
| `tests/sfno_inference_5410/test_lp004_patch_present.py` | 59 | NEW |
| `tests/sfno_inference_5410/test_upstream_hydration.py` | 262 | NEW (3 allowlists × tests + bool reject + ckpt + full-equivalence) |
| `tests/sfno_inference_5410/test_static_arch_invariant.py` | 126 | NEW |
| `tests/sfno_inference_5410/test_runtime_args_5410.py` | 237 | UPDATED (drop argv assertions) |
| `tests/sfno_inference_5410/test_eval_driver_K.py` | (in test file) | UPDATED (drop argv check) |
| `tests/sfno_inference_5410/integration/__init__.py` | 0 | NEW |
| `tests/sfno_inference_5410/integration/_legacy_subprocess_launcher.py` | 105 | NEW — captures pre-refactor subprocess pattern |
| `tests/sfno_inference_5410/integration/test_ab_equivalence.py` | 282 | NEW — gates A/B/C |
| `tests/sfno_inference_5410/integration/test_legacy_baseline.py` | 138 | NEW — nondeterminism floor diagnostic |
| `docs/2026-05-04_makani_local_patches.md` | 315 | + LP-004 entry |
| `docs/2026-05-08_sfno_5410_inproc_orchestrator_plan.md` | 676 | The v2.1 plan (Codex round-2 approved) |
| `.claude/skills/eval-sfno-5410/SKILL.md` | 217 | Updated description + Architecture section |

## The cudnn-globals fix (gate-A debugging)

Gate A's first run failed at `pl max_rel = 4.04e-5`. Legacy-vs-legacy baseline (job 3098325) returned **bit-exact** (`max_abs = max_rel = 0.000e+00` for all 8 vars), proving the legacy path is fully deterministic and the divergence was a real refactor regression.

**Root cause:** Upstream `long_inference.py main()` lines 1463-1465 (post-LP-003-shifted from 1314) sets three torch-level globals before `Stepper(...)`:

```python
torch.manual_seed(world_rank)        # = 0 in single-rank
torch.cuda.set_device(local_rank)    # = 0
torch.backends.cudnn.benchmark = True
```

The in-process orchestrator was skipping all three. The load-bearing one is `cudnn.benchmark = True` — it enables cudnn's autotuner, which under fp16/AMP picks **different kernels** than the cudnn default. Different kernels → different rounding → ~4e-5 numerical drift.

Confirming evidence visible in the gate-A failure log:
- Legacy first iter: `[00:23<23:41, 23.69s/it]` (cudnn autotuning)
- Inproc first iter: `[00:00<00:26, 2.25it/s]` = 0.44 s (no autotuning)

**Fix location:** `scripts/eval_inference_5410.py` lines 376-381, immediately before `Stepper([params], ...)`:

```python
# Mirror upstream main()'s torch-level globals (long_inference.py:1463-1465).
# Critical: cudnn.benchmark=True selects autotuned kernels; without it
# cudnn picks conservative kernels and fp16/AMP output diverges from
# the legacy path (verified by gate-A failure + legacy-vs-legacy
# bit-exact baseline 2026-05-08).
import torch
torch.manual_seed(0)  # world_rank=0 in single-rank inference
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

print(f"[orchestrator] constructing Stepper (one-time model + ckpt load)...")
t_setup = time.time()
from long_inference import Stepper  # type: ignore
stepper = Stepper([params], world_rank=0, async_save=args.async_save)
```

After the fix, gate A passed (job 3098365) and matched the legacy output within rtol=1e-5. Gates B and C also passed.

## `upstream_hydration.py` — three field tiers

Codex blocker #2 in round 1: upstream `main()` injects 22+ fields between argparse and `Stepper(...)` (notably `nc_bc_offset = 18`). The new helper module mirrors all of them. Codex round-2 fix #1: split the allowlist by helper boundary.

| Tier | Helper | Fields |
|---|---|---|
| **Static** (set once, reused across all 96 ICs) | `hydrate_static_params(yaml_path, K, *, upstream_repo, ...) -> YParams` | `run_iter`, `has_diagnostic`, `num_ensemble_members`, `ensemble_members_per_pred`, `nc_bc_offset = 18`, `world_size = 1`, `batch_size = 1`, `local_rank`, `enable_amp = True`, `experiment_dir`, `checkpoint_dir`, `best_checkpoint_path` (resolved best > latest > globstr), `latest_checkpoint_path`, `checkpoint_path_globstr`, `resuming = True`, `log_to_wandb = False`, `log_to_screen` (17 fields) |
| **Per-Y** (re-set on Y boundary) | `set_per_y_params(params, *, Y)` | `val_year_start = Y`, `val_year_end = Y + 1`, `leap_year = Y`, `no_leap_year = Y` (4 fields) |
| **Per-IC** (re-set every IC) | `set_per_ic_params(params, *, init_datetime, final_datetime, init_nc_filepaths, save_basename, output_dir)` | `init_datetime` (cftime-normalized via `_dt_cls`), `final_datetime` (same), `init_nc_filepaths` (single-element list), `init_nc_timestep_offset` (recomputed by reading IC NC's time index), `save_basename`, `output_dir` (6 fields) |
| **Cross-yaml invariant** | `assert_yamls_share_static_arch(yaml_paths)` | Asserts all 8 per-Y yamls match on every non-per-Y field. Justifies hydrating from `yaml_paths[0]` and reusing the model across all Y. |

Tested by `tests/sfno_inference_5410/test_upstream_hydration.py` with one test per tier + a `test_full_main_equivalence` that asserts the union after all three helpers run. Bool inputs to `set_per_y_params` are explicitly rejected (Codex round-1 fix; `isinstance(True, int)` is True in Python).

## Orchestrator behavior (`scripts/eval_inference_5410.py`)

```
build run plan from (Y, s) tuples + K
preflight: LP-003 patch (4+2), LP-004 patch (1), per-Y yaml horizon, static-arch invariant across yamls
print per-IC log line for each entry (init / final / forecast_K / raw_steps)

if --launch:
    assert_output_dir_empty(out_dir)              # gated on --launch (Codex round-2 fix #3)
    cd into upstream repo; sys.path.insert upstream
    params = hydrate_static_params(yaml_paths[0], K=K, ...)        # ONE params object
    set_per_y_params(params, Y=plan[0]['Y'])
    set_per_ic_params(params, init_datetime=..., ...)              # bind to first IC

    torch.manual_seed(0); torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True                          # ← cudnn fix

    stepper = Stepper([params], world_rank=0, async_save=args.async_save)   # ← built ONCE

    _run_one_ic(stepper, plan[0], K, val_year_changed=True)        # first IC
    for i, entry in enumerate(plan[1:], start=2):
        prev_Y = plan[i - 2]['Y']
        val_year_changed = entry['Y'] != prev_Y                    # computed BEFORE any mutation
        if val_year_changed:
            set_per_y_params(stepper.params, Y=entry['Y'])         # MUTATE in place; no swap
        set_per_ic_params(stepper.params, ...)                     # MUTATE in place; no swap
        _run_one_ic(stepper, entry, K, val_year_changed=val_year_changed)

    assert_output_dir_complete(out_dir, plan, K)                   # filenames + time-dim + var set
```

Properties:

- **Single hydrated params object across all 96 ICs.** `stepper.params` is **never** swapped — only mutated in place via `set_per_y_params` / `set_per_ic_params`. The model and checkpoint are bound to this object during `Stepper.__init__` and reuse it for every subsequent `predict()` call.
- **`val_year_changed` is computed before any mutation**, from `entry['Y'] != prev_Y`, and passed explicitly to `reconfigure_for_ic` as a required keyword. Never inferred from `self.params` (Codex round-1 blocker #1).
- **Per-IC: `_run_one_ic` runs four `if … raise ValueError(...)` gates** after `reconfigure_for_ic` returns:
  - `len(stepper.data_loader_bcs) == K + 1`
  - `len(stepper.data_loader) == 1`
  - `len(stepper.params.init_nc_filepaths) == 1`
  - `stepper.params.final_datetime == init + (K + 1) * 6 h`
  These survive `python -O` (Codex round-2 fix #2).
- **`reconfigure_for_ic` itself validates `init_nc_timestep_offset`** by re-deriving from the current IC NC's time index and comparing. If a future caller skips `set_per_ic_params`, the helper raises `ValueError` with an explicit "call set_per_ic_params first" message (Codex round-2 fix #4).

## `async_save = False` proof

1. **Argparse default is False.** `scripts/eval_inference_5410.py`:
   ```python
   p.add_argument("--async-save", action="store_true",
                  help="...v2.1 default is False (synchronous saves) so post-IC "
                       "assertions and reconfigure can't race a still-pending "
                       "save thread.")
   ```
   `action="store_true"` defaults to False.

2. **Production SLURM does not pass `--async-save`** — `scripts/submit_eval_inference_5410.slurm` invokes:
   ```bash
   python -u scripts/eval_inference_5410.py \
       --run-root "$RUN_ROOT" \
       --config-dir "$RUN_ROOT/inference" \
       --K "$K" \
       --launch
   ```
   No `--async-save`. So `args.async_save = False`, `Stepper(...)` is constructed with `async_save=False`.

3. **Smoke SLURM does not pass `--async-save`** — `scripts/submit_eval_inference_5410_smoke.slurm` is identical except for `--years $SMOKE_Y --limit-ics 1`.

4. **A/B gate SLURMs do not pass `--async-save`** — gates A/B/C all exec the same orchestrator path with no `--async-save` flag.

5. **Legacy launcher does not pass `--async_save`** — `tests/sfno_inference_5410/integration/_legacy_subprocess_launcher.py` line 84-94 builds the upstream argv and explicitly **omits** `--async_save`. Comment in source: "NOTE: omits --async_save so the legacy reference is sync (matches v2.1 default for the new path's first smoke)."

Both legacy and inproc paths in all four jobs ran synchronously. The flag stays exposed as a future toggle.

## Confirmation: legacy launcher is independent of the new orchestrator

`tests/sfno_inference_5410/integration/_legacy_subprocess_launcher.py` is a 105-line standalone module. Greps:

```
4-11:   docstring states it "preserves the pre-refactor invocation pattern"
38:     def docstring: "Run upstream long_inference.py for ONE IC via subprocess"
66:     long_inf = upstream_repo / "long_inference.py"
68:     raise FileNotFoundError(f"upstream long_inference.py not found: {long_inf}")
102:    return subprocess.run(argv, cwd=str(upstream_repo), env=env, check=True)
```

Imports:

```python
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping
```

**Zero imports from `eval_inference_5410.py` or any module it touches.** `subprocess.run([..., "long_inference.py", ...])` is invoked with `cwd=upstream_repo`. The new orchestrator is never on the call path.

The A/B test wires both paths in `tests/sfno_inference_5410/integration/test_ab_equivalence.py`:

- `_run_legacy_subprocess(...)` → calls `launch_legacy_subprocess(...)` from `_legacy_subprocess_launcher`. Old path.
- `_run_inproc(...)` → spawns `python eval_inference_5410.py --K 60 ... --launch` as a fresh subprocess. New path.

Different launcher, different process, different code path.

## A/B job log paths

All four jobs ran synchronously, completed cleanly (exit 0). Logs available at:

| Job | Purpose | State | Elapsed | `.out` log |
|---|---|---|---|---|
| 3098325 | Legacy-vs-legacy nondeterminism floor | COMPLETED | 2:53 | `/home1/11114/zhixingliu/projects/SFNO_Climate_Emulator/logs/5410_legacy_floor_3098325.out` |
| 3098261 | Gate A (initial — failed) | FAILED 1:0 | 3:14 | `/home1/11114/zhixingliu/projects/SFNO_Climate_Emulator/logs/5410_ab_A_3098261.out` |
| 3098365 | Gate A (re-run after cudnn fix) | COMPLETED | 2:23 | `/home1/11114/zhixingliu/projects/SFNO_Climate_Emulator/logs/5410_ab_A_3098365.out` |
| 3098372 | Gate B (two same-year ICs) | COMPLETED | 2:52 | `/home1/11114/zhixingliu/projects/SFNO_Climate_Emulator/logs/5410_ab_B_3098372.out` |
| 3098403 | Gate C (cross-year pair Y=121→Y=122) | COMPLETED | 2:53 | `/home1/11114/zhixingliu/projects/SFNO_Climate_Emulator/logs/5410_ab_C_3098403.out` |

Key log lines:

**3098325 (legacy floor):**
```
=== LEGACY-vs-LEGACY nondeterminism floor ===
  IC: Y=121 s=0 K=60
  var      shape                    max_abs      max_rel
  hus      (61, 10, 64, 128)        0.000e+00    0.000e+00
  pl       (61, 64, 128)            0.000e+00    0.000e+00
  pr_6h    (61, 64, 128)            0.000e+00    0.000e+00
  ta       (61, 10, 64, 128)        0.000e+00    0.000e+00
  tas      (61, 64, 128)            0.000e+00    0.000e+00
  ua       (61, 10, 64, 128)        0.000e+00    0.000e+00
  va       (61, 10, 64, 128)        0.000e+00    0.000e+00
  zg       (61, 10, 64, 128)        0.000e+00    0.000e+00
  WORST: None at max_rel=0.000e+00
```

**3098261 (gate A initial):**
```
E    AssertionError: pl numerical mismatch: max_abs=4.358e-04 max_rel=4.042e-05 (rtol=1e-05 atol=1e-07)
FAILED tests/sfno_inference_5410/integration/test_ab_equivalence.py::test_one_ic_equivalence
```

**3098365 (gate A re-run, post-fix):**
```
tests/sfno_inference_5410/integration/test_ab_equivalence.py::test_one_ic_equivalence PASSED [100%]
```

**3098372 (gate B):**
```
tests/sfno_inference_5410/integration/test_ab_equivalence.py::test_two_same_year_ics_equivalence PASSED [100%]
```

**3098403 (gate C):**
```
tests/sfno_inference_5410/integration/test_ab_equivalence.py::test_cross_year_pair_equivalence PASSED [100%]
```

All assertions in `_compare_netcdfs` (dims, data_vars, time coords, per-variable values at rtol=1e-5 atol=1e-7) passed for all eight variables (pl, tas, pr_6h, ta, ua, va, hus, zg) on each test.

## Test commands

**Unit tests (login node, no GPU):**

```bash
source ~/projects/SFNO_Climate_Emulator/.venv/bin/activate
PYTHONPATH=~/projects/SFNO_Climate_Emulator/src python -m pytest tests/sfno_inference_5410/ \
    --ignore=tests/sfno_inference_5410/integration
```

Output (verified 2026-05-08):

```
tests/sfno_inference_5410/test_eval_driver_K.py .......                  [  2%]
tests/sfno_inference_5410/test_get_dates_contract.py ...................
.....                                                                    [ 10%]
tests/sfno_inference_5410/test_ic_nc_compatibility.py ssss...ssss        [ 41%]
tests/sfno_inference_5410/test_ic_offsets.py ........                    [ 43%]
tests/sfno_inference_5410/test_lp004_patch_present.py .....              [ 45%]
tests/sfno_inference_5410/test_required_attrs.py ...................     [ 65%]
tests/sfno_inference_5410/test_runtime_args_5410.py ...........          [ 74%]
tests/sfno_inference_5410/test_static_arch_invariant.py ....             [ 75%]
tests/sfno_inference_5410/test_upstream_attr_drift.py ...                [ 76%]
tests/sfno_inference_5410/test_upstream_hydration.py .......             [ 79%]
tests/sfno_inference_5410/test_upstream_patch_present.py ...             [ 80%]
tests/sfno_inference_5410/test_yaml_override.py ..................       [ 94%]
tests/sfno_inference_5410/test_yaml_override_K.py .................      [100%]

================= 215 passed, 97 skipped, 2 warnings in 8.57s ==================
```

97 skipped tests are RUN_ROOT-gated (require live IC NCs at `$RUN_ROOT/inference/ic_nc/...`); they're sanity-checks for the IC NC schema and are exercised only on the production run-root.

**A/B integration tests (GPU node, gated behind RUN_AB_TESTS=1):**

Each gate is its own SLURM submission; the `submit_ab_gate_{a,b,c}.slurm` wrappers each invoke a single pytest test. Concrete commands inside the wrappers:

```bash
RUN_AB_TESTS=1 python -m pytest \
    tests/sfno_inference_5410/integration/test_ab_equivalence.py::test_one_ic_equivalence \
    -v --tb=long

RUN_AB_TESTS=1 python -m pytest \
    tests/sfno_inference_5410/integration/test_ab_equivalence.py::test_two_same_year_ics_equivalence \
    -v --tb=long

RUN_AB_TESTS=1 python -m pytest \
    tests/sfno_inference_5410/integration/test_ab_equivalence.py::test_cross_year_pair_equivalence \
    -v --tb=long
```

Plus the legacy-vs-legacy diagnostic:

```bash
RUN_AB_TESTS=1 python -m pytest \
    tests/sfno_inference_5410/integration/test_legacy_baseline.py::test_legacy_vs_legacy_baseline \
    -v -s --tb=long
```

## What I'm NOT doing

- **No production submit.** Holding for your round-3 sign-off.
- **No score / report SLURMs.** Still TODO; production output remains 96 NetCDFs in `upstream_raw/`. Downstream scoring is manual.
- **No DDP fan-out.** Single-rank H100 is fast enough at K=60.
- **No async_save flip.** Stays False until either upstream's `predict()`-awaits-save invariant is verified-and-documented or an explicit `drain_save_queue` method is added.

## What I'd like you to look at most carefully

In rough order of risk:

1. **The cudnn-globals fix** (`scripts/eval_inference_5410.py:376-381`). Are there other torch-level globals upstream main() sets that I might still be missing? My grep for `torch\.manual_seed|torch\.cuda\.set_device|torch\.backends|torch\.set_default|torch\.use_deterministic` found only the three I patched, but you might know of others.
2. **`reconfigure_for_ic` body** (the LP-004 hunk in `long_inference.py`, ~120 lines). Specifically the `val_year_changed` semantics and the `init_nc_timestep_offset` validation.
3. **The single-params-mutation discipline** in the orchestrator's per-IC loop. Is there any code path where `stepper.params` could end up referring to a different object than the one the model was built with?
4. **`assert_yamls_share_static_arch`** — does it cover everything that would invalidate the "model built once, reused across all Y" assumption? It checks every non-`(val_year_start, val_year_end, leap_year, no_leap_year)` SFNO-section field; anything outside the SFNO section is ignored, which I think is correct because YParams loads only the selected section.
