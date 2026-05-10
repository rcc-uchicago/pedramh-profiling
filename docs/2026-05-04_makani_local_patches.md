# Local patches against vendored Makani

This file is the canonical record of every modification we make to
`makani-src/` core. It is also the merge gate for any item in the
Phase-1 efficiency plan that touches Makani — review this list when
bumping `makani-src/` and re-apply each patch on top of the new
upstream tree before merging the bump.

**Vendored upstream pin:** `makani-src/` HEAD = `c970430` at
2026-05-04. (Run `cd makani-src && git rev-parse --short HEAD` to
re-confirm. If `makani-src/` is not a checked-in submodule but a
local clone, the date above is the source of truth.)

## Patch index

| ID | File | Lines (pre-patch) | Plan item | Date applied |
|----|------|-------------------|-----------|--------------|
| LP-001 | `makani-src/makani/utils/training/deterministic_trainer.py` | 470-471 (training); 608-609 (validation) | Phase-1 P7 | 2026-05-04 |
| LP-002 | `makani-src/makani/utils/driver.py` | 84-85 | DDP smoke crash on non-rank-0 ranks | 2026-05-04 |
| LP-003 | `/work2/.../v2.0/long_inference.py` | 554, 720, 724, 834, 998, 1002 (6 hunks) | docs/2026-05-08_sfno_5410_explicit_K_horizon_plan.md (eval-5410 partial-horizon K=60) | 2026-05-08 |
| LP-004 | `/work2/.../v2.0/long_inference.py` | inserted after `Stepper.__init__` (1 hunk: new `reconfigure_for_ic` method) | docs/2026-05-08_sfno_5410_inproc_orchestrator_plan.md (in-process orchestrator) | 2026-05-08 |
| LP-005 | `makani-src/makani/models/preprocessor.py` | 257 (perturb-mode index_put) | group-recipe clone with bf16 amp + input_noise mode=perturb | 2026-05-09 |

## LP-001 — `non_blocking=True` on the H2D copy

### Why

With `pin_memory=True` already on the DataLoader
(`src/sfno_training/trainer/plasim_trainer.py`), the host→device copy
should issue asynchronously so the host thread can advance to the
next prefetched batch instead of blocking inside `.to()`. Adding
`non_blocking=True` is the standard way to express this. It does
**not** by itself create a separate copy stream from compute, so
kernel-level overlap on the GPU side is opportunistic — the
guaranteed win is host-thread freeing only.

### Diff (against upstream `c970430`)

Training loop (around `train_one_epoch`, line 471 pre-patch):

```diff
             # map to device
-            gdata = map(lambda x: x.to(self.device), data)
+            # Local Phase-1 patch (P7): non_blocking=True. Frees the host
+            # thread so the prefetched-pinned data path (DataLoader uses
+            # pin_memory=True) is not blocked on the H2D copy. See
+            # docs/2026-05-04_makani_local_patches.md for the full diff
+            # and the upstream commit we are patched against.
+            gdata = map(lambda x: x.to(self.device, non_blocking=True), data)
```

Validation loop (around `validate_one_epoch`, line 609 pre-patch):

```diff
                     # map to gpu
-                    gdata = map(lambda x: x.to(self.device), data)
+                    # Local Phase-1 patch (P7): non_blocking=True (validation).
+                    # See docs/2026-05-04_makani_local_patches.md.
+                    gdata = map(lambda x: x.to(self.device, non_blocking=True), data)
```

### Test / verification

- `pytest tests/sfno_training/` (no regressions; pinned-memory H2D is
  numerically equivalent to the blocking path).
- A 2-epoch short-config run before and after the patch should
  produce loss curves within stochastic noise.
- A post-P6 profiler trace should show the host-side `aten::to` event
  no longer dominating the inter-step gap.

### Re-apply procedure on Makani bump

1. Diff the new upstream `deterministic_trainer.py` against the
   pre-patch baseline above. If the surrounding `# map to device`
   /  `# map to gpu` comments and the bare `.to(self.device)` call
   are still present at functionally-equivalent locations, the
   patch re-applies cleanly: just re-add `non_blocking=True` and the
   comment block.
2. If upstream has refactored the data-movement step (e.g.
   introduced an explicit prefetcher), reconsider whether this patch
   is still necessary — the upstream change may already do the
   right thing.
3. Update the "Vendored upstream pin" line at the top of this file
   and the "Lines (pre-patch)" column in the index after re-applying.

## LP-002 — Always assign `self.logger` in `Driver.__init__`

### Why

Upstream gates `self.logger = logging.getLogger()` on
`log_to_screen` (effectively rank 0 only). But several call sites
in `deterministic_trainer.py` use `self.logger.info(...)`
unconditionally, so non-zero ranks under DDP crash with
`AttributeError: 'PlasimTrainer' object has no attribute 'logger'`
during `Trainer.__init__`. The cascade:

1. Ranks 1, 2, 3 crash at `deterministic_trainer.py:232`
   ("No channels to visualize, skipping visualization.").
2. Rank 0 successfully reaches `dist.barrier(...)` at
   `deterministic_trainer.py:297`.
3. The barrier hangs because the dead ranks never join → 10-minute
   `NCCL_ASYNC_ERROR_HANDLING` timeout → the watchdog terminates
   all ranks.

Caught by `submit_zgplev_short_ddp.slurm` (job 3084328). This patch
makes `self.logger` exist on every rank. On non-rank-0 the root
logger is unconfigured (WARNING level, no handlers), so any
unguarded `self.logger.info(...)` is silently dropped — there is
no observable behaviour change beyond fixing the crash.

### Diff (against upstream `c970430`)

`makani-src/makani/utils/driver.py`, around line 84:

```diff
         # set the logger
         self.log_to_screen = self.params.log_to_screen if (hasattr(params, "log_to_screen") and params.log_to_screen) else False
-        if self.log_to_screen:
-            self.logger = logging.getLogger()
+        # Local Phase-1 patch (LP-002): always assign self.logger.
+        # Upstream gates the assignment on log_to_screen, but several
+        # call sites in deterministic_trainer.py (e.g. line 232,
+        # "No channels to visualize") use self.logger.info(...)
+        # unconditionally — they crash with AttributeError on non-zero
+        # ranks under DDP. On rank > 0 the root logger is unconfigured
+        # (WARNING level, no handlers), so the unguarded INFO calls are
+        # silently dropped. See docs/2026-05-04_makani_local_patches.md.
+        self.logger = logging.getLogger()
```

### Test / verification

- `submit_zgplev_short_ddp.slurm` → ≥ 2 epochs complete on 4 H100
  ranks with no NCCL hang (this was the failure that triggered the
  patch; re-running it post-patch is the merge gate).
- Rank-0 stdout/stderr unchanged (no spurious INFO messages, no
  removed INFO messages).

### Re-apply procedure on Makani bump

1. Locate the `# set the logger` block in the new upstream
   `driver.py` (currently lines 84-85). If still gated on
   `log_to_screen`, drop the `if`-guard and apply the comment
   block.
2. If upstream has restructured the logger setup (e.g. moved to a
   helper, made it always-on), reconsider whether this patch is
   still necessary.
3. Update the "Vendored upstream pin" line at the top of this file
   and the "Lines (pre-patch)" column in the index after re-applying.

## LP-003 — Partial-horizon support in `long_inference.py` (eval-5410 K=60)

### Why

The 5410 NWP eval needs to stop each IC at an explicit `K` forecast
leads (canonical `K=60`, 15 days) instead of rolling to next Jan 1.
Pristine upstream hard-codes `next_output_datetime = (current_year+1,
1, 1)` at four allocator sites and gates the rollout-continuation on
`current_year < final_datetime.year` at two sites — a year-only
comparison that overshoots any sub-year `--final_datetime`. Without
this patch the K limit is enforced only by adapter-side slicing,
which costs ~5–6× compute per IC.

This file lives **outside the AI-RES repo**, at
`/work2/11114/zhixingliu/stampede3/AI-RES/artifacts/derecho_glade/PanguWeather/v2.0/long_inference.py`.
It is the actual entry point invoked by `scripts/eval_inference_5410.py`
(see `_UPSTREAM_LONG_INFERENCE` at the top of that script).

The matching skipped hunks (perturbation gates at lines 558, 830 in
the pristine file: `init_datetime.year == 1` checks) are intentionally
**NOT applied** — eval-track pins `epsilon_factor=0.0` so those
branches are unreachable. They can be added later if a future eval
needs ensemble perturbations.

### Diff

Six hunks: 4 allocators (lines 554, 724, 834, 1002 pre-patch) + 2
continuations (lines 720, 998 pre-patch). The 4 allocator sites all
get the same transformation:

```diff
-                    next_output_datetime = self.dataset.datetime_class(current_year+1, 1, 1, hour=next_year_offset_hours,
-                                                                    has_year_zero = self.params.has_year_zero)
+                    # AI-RES local patch (2026-05-08): bound chunk endpoint by final_datetime so partial-final-year rollouts save correctly.
+                    next_year_jan1 = self.dataset.datetime_class(current_year+1, 1, 1, hour=next_year_offset_hours,
+                                                                    has_year_zero = self.params.has_year_zero)
+                    next_output_datetime = min(next_year_jan1, self.params.final_datetime)
```

The 2 continuation sites both get the same transformation (hoist
`current_datetime = next_output_datetime` before the check, then
compare datetimes instead of years):

```diff
                             current_year += 1
-                            # If this was not the final year
-                            if current_year < self.params.final_datetime.year:
-                                # Get the number of time steps in the next year
-                                current_datetime = next_output_datetime
+                            # AI-RES local patch (2026-05-08): hoist current_datetime before the check so we compare datetimes, not years (handles partial-final-year).
+                            current_datetime = next_output_datetime
+                            if current_datetime < self.params.final_datetime:
+                                # Get the number of time steps in the next year
                                 next_year_offset_hours = current_datetime.hour % self.params.timedelta_hours
-                                next_output_datetime = self.dataset.datetime_class(current_year+1, 1, 1, hour=next_year_offset_hours,
-                                                                                has_year_zero = self.params.has_year_zero)
+                                # AI-RES local patch (2026-05-08): bound chunk endpoint by final_datetime.
+                                next_year_jan1 = self.dataset.datetime_class(current_year+1, 1, 1, hour=next_year_offset_hours,
+                                                                                has_year_zero = self.params.has_year_zero)
+                                next_output_datetime = min(next_year_jan1, self.params.final_datetime)
```

### Verification (strict counts — guards against partial application)

```bash
ALLOC=$(grep -c 'min(next_year_jan1, self.params.final_datetime)' \
    /work2/11114/zhixingliu/stampede3/AI-RES/artifacts/derecho_glade/PanguWeather/v2.0/long_inference.py)
CONT=$(grep -c 'current_datetime < self.params.final_datetime' \
    /work2/11114/zhixingliu/stampede3/AI-RES/artifacts/derecho_glade/PanguWeather/v2.0/long_inference.py)
[[ "$ALLOC" -eq 4 && "$CONT" -eq 2 ]] || \
    echo "patch incomplete: allocators=$ALLOC (expect 4) continuations=$CONT (expect 2)"

# And no stragglers from the pre-patch year-only continuation:
grep -c 'current_year < self.params.final_datetime.year' \
    /work2/11114/zhixingliu/stampede3/AI-RES/artifacts/derecho_glade/PanguWeather/v2.0/long_inference.py
# expected: 0
```

A single `grep -q` would pass a partially-applied patch (1 of 4
allocator hunks); always use exact counts. The same strict-count
preflight is wired into both `scripts/submit_eval_inference_5410.slurm`
and `scripts/submit_eval_inference_5410_smoke.slurm`, plus the
orchestrator (`scripts/eval_inference_5410.py`'s `main()` calls
`assert_upstream_patched(...)`).

### Backup

Pre-patch byte-identical copy at:
```
/work2/11114/zhixingliu/stampede3/AI-RES/artifacts/derecho_glade/PanguWeather/v2.0/long_inference.py.bak.20260508_pre_K_patch
```

### Re-application after upstream resync

1. Re-confirm the 4 allocator sites still appear at (or near) lines
   554, 724, 834, 1002, and the 2 continuation sites at 720, 998.
   The line numbers may drift; the surrounding context (`time_step_in_year = 0`
   above the first allocator; perturbation block above the third
   allocator; `current_year += 1` above each continuation) is the
   stable anchor.
2. Apply the diff above. Run the strict-count verification.
3. Run `pytest tests/sfno_inference_5410/test_upstream_patch_present.py`
   to confirm.
4. Run the smoke (`K=60 sbatch scripts/submit_eval_inference_5410_smoke.slurm`)
   before re-submitting production.

## LP-004 — `Stepper.reconfigure_for_ic` (in-process orchestrator)

### Why

The 5410 NWP eval orchestrator was rewritten on 2026-05-08 from a
96-subprocess loop to a single Python process that builds the upstream
``Stepper`` once and reuses self.model + ckpt across all 96 ICs (mirrors
the own-track architecture). Saves ~95 s of cold-import + ckpt reload
**per IC** (~2.5 h on 96 ICs).

For the in-process loop to work, the Stepper needs a way to re-bind to
a new IC after construction. LP-004 adds a single new method
``Stepper.reconfigure_for_ic`` that mutates per-IC params and rebuilds
both data loaders + the constant_boundary_data tensor (on Y crossings),
without touching self.model or the loaded checkpoint.

The method takes ``val_year_changed: bool`` as a **required** keyword
arg — never inferred from self.params (the caller may already have
mutated val_year_start by the time this runs). This is the Codex
round-1 blocker fix.

The method also **validates** ``init_nc_timestep_offset`` against the
current IC NC files. If a future caller bypasses
``set_per_ic_params``, the helper raises ``ValueError`` with an explicit
"call set_per_ic_params first" message (Codex round-2 fix #4).

### Diff

Single hunk: a new method inserted between ``Stepper.__init__`` (ends
line 361 in the LP-003-patched file) and ``_get_inference_duration``.
The method body is ~120 lines including its docstring; see
docs/2026-05-08_sfno_5410_inproc_orchestrator_plan.md §1 for the
canonical reference.

### Verification (strict count)

```bash
N=$(grep -c 'def reconfigure_for_ic' \
    /work2/11114/zhixingliu/stampede3/AI-RES/artifacts/derecho_glade/PanguWeather/v2.0/long_inference.py)
[[ "$N" -eq 1 ]] || echo "patch count $N != 1"
```

A single ``grep -q`` would pass duplicate application; always use the
exact-count form. Wired into both
``scripts/submit_eval_inference_5410.slurm`` and
``scripts/submit_eval_inference_5410_smoke.slurm`` preflights, plus
``src/sfno_inference_5410/preflight.py:assert_upstream_patched_lp004``,
plus ``tests/sfno_inference_5410/test_lp004_patch_present.py``.

### Re-application after upstream resync

1. Locate the end of ``Stepper.__init__`` in the new upstream
   ``long_inference.py`` (currently line 361 post-LP-003). Insert the
   ``reconfigure_for_ic`` method as a sibling method immediately
   before ``_get_inference_duration``.
2. Verify the strict count above returns 1.
3. Run ``pytest tests/sfno_inference_5410/test_lp004_patch_present.py``.
4. Run the smoke (``K=60 sbatch scripts/submit_eval_inference_5410_smoke.slurm``).
5. Run the A/B equivalence tests (``RUN_AB_TESTS=1 pytest tests/sfno_inference_5410/integration/``)
   before any production submit.
