# DDP throughput fix — implementation plan

Date: 2026-05-05
Status: plan only — no code changes in this commit.
Scope: speed fix for the 4-rank DDP zgplev full run, which is currently
wall-clock-equivalent to the single-GPU baseline because per-rank batch
size collapses to 1.
Source diagnosis: see the 4-rank DDP scaling analysis from 2026-05-05
(no separate doc; numbers reproduced under "Verified facts" below).

**Revision history**

- v1 (2026-05-05): initial draft.
- **v5 (2026-05-05, fifth review): one blocker fix + one wandb-path
  simplification. No structural changes since v4.**
  - **I2 explicit `max_epochs: 2` overlay.** v4's I2 said "2 epochs per
    point" but did not list `max_epochs` among the YAML overlays.
    Inheriting the full YAML unchanged would carry
    `max_epochs: 50` from
    `src/sfno_training/config/plasim_sim52_zgplev_full.yaml:110`, so
    every microbench point would run the full schedule. v5 adds an
    explicit overlay bullet under I2: render `max_epochs: 2` per
    point. `scheduler_T_max` stays at its current value (45); it
    governs the cosine schedule shape, not the run length, and the
    microbench is not a convergence run.
  - **Samples/sec wandb path simplified.** v4 added a separate
    `wandb.log({"samples/sec": ...}, step=self.epoch)` call after
    `super().log_epoch(...)` returned. That call lands **after**
    Makani's `commit=True` at
    `makani-src/makani/utils/training/deterministic_trainer.py:738`,
    which would push it into the next wandb step (or be silently
    dropped, depending on wandb version semantics). v5 instead puts
    `samples/sec` into **both** `timing_logs` (so screen output is
    unchanged in shape) and `train_logs` (so it flows through Makani's
    existing `wandb.log(train_logs, step=self.epoch)` at `:733`,
    before the `commit=True` line). Two dict writes, one destination
    each, no extra `wandb.log` call. The override therefore drops the
    explicit wandb branch entirely.
- **v4 (2026-05-05, fourth review): two blocker fixes + two concern
  fixes + one wording tweak. No structural changes since v3.**
  - **Rollback section reconciled with I1 monotonicity rule.** v3
    softened I1 to a tiered review trigger but left the rollback
    section saying any A→D monotonicity violation halts. v4 rewrites
    the rollback bullet to match I1's tiered rule (note + proceed; or
    halt only on a reproducible ≥15% regression).
  - **Rerun count standardized to one.** v3 revision-history wording
    said "reproducible across two reruns" while I1 said "one rerun".
    v4 standardizes on **one rerun of the affected pair** in both
    places, to control H100 cost (each short-config DDP rerun is ~5
    min, but a flat policy is simpler than two).
  - **Samples/sec wandb path corrected.** v3 claimed adding the new
    key to `timing_logs` would flow through to both screen and wandb.
    Verified at `makani-src/makani/utils/training/deterministic_trainer.py:695-740`:
    `log_epoch` iterates `timing_logs.keys()` dynamically for the
    *screen* logger at `:712-713`, but the wandb branch at `:732-738`
    only emits `train_logs`, `valid_logs["base"]`, and
    `valid_logs["metrics"]` — `timing_logs` is **not** sent to wandb.
    v4 makes the override explicit: keep `samples/sec` in
    `timing_logs` (so screen output is unchanged in shape), and add an
    explicit `wandb.log({"samples/sec": ...}, step=self.epoch)` call
    in the override when `self.log_to_wandb` is True. The current
    production YAML has `log_to_wandb: False`, so the wandb branch is
    a no-op today, but it must be correct for when wandb is flipped on.
  - **T1 helper extraction.** v3 left the divisibility assertion inline
    at `src/sfno_training/train_plasim.py:134-141`. v4 adds an explicit
    refactor to commit 1: extract a pure
    `_resolve_batch_sizes(params, data_parallel_size) -> int` helper
    that returns the per-rank batch and asserts divisibility. T1 then
    tests the helper directly with a mock `params` object, instead of
    trying to exercise the full CLI / distributed wireup path.
  - **I2 grep wording softened.** The bf16 stability check is a smoke
    signal sufficient for a microbench, not a true numerical scan
    (no checkpoint tensor validation). v4 adds a one-line clarifier so
    nobody reads the grep section as a substitute for a future
    `scan_for_nans.py`.
- **v3 (2026-05-05, third review): three blocker fixes + three concern
  fixes + one wording tweak. No structural changes since v2.**
  - **I2 sample budget rewritten as a step target.** v2 said
    `n_train_samples_per_epoch=2048`, which at GB=32 is only 64 train
    steps per epoch (~4 s of actual training at ~60 ms/step) — not the
    "5–10 min" the same paragraph claimed. v3 sets a measured-step
    target (≈1500 steady-state steps per point) and derives
    `n_train_samples_per_epoch = global_batch × target_steps_per_epoch`
    per batch point. Epoch 1 is treated as warmup (JIT, cache,
    DataLoader prefetch fill) and dropped before averaging.
  - **EMA stays enabled in I2.** v2 left EMA disabling as "TBD"; that
    was wrong. There is no `--ema` CLI override in the parser, and
    disabling EMA changes the optimizer-step path because EMA registers
    a `register_step_post_hook` at
    `src/sfno_training/trainer/plasim_trainer.py:309-311`. The
    microbench must measure the production-shaped step. Validation and
    checkpoint writing are still optionally skipped (those do not touch
    the optimizer).
  - **T3 split.** v2 placed T3 in commit 1 but its assertion targets
    were the I2-chosen values, which do not exist until commit 4. v3
    splits into T3a (commit 1: YAML loads cleanly, current values
    present) and T3b (commit 4: asserts the post-I2 values).
  - **Samples/sec implementation pinned.** v2 said "follow-up
    `logger.info` in `train_plasim.py` between epochs" was acceptable;
    that path is not really available because Makani computes
    `timing_logs` and calls `self.log_epoch(...)` from inside the
    trainer loop at
    `makani-src/makani/utils/training/deterministic_trainer.py:421-441`.
    v3 makes the implementation concrete: override
    `PlasimTrainer.log_epoch`, inject `samples/sec` into `timing_logs`,
    then `super().log_epoch(...)`. The awk-parser fallback stays as a
    last resort if the override turns out to be more invasive than
    expected.
  - **`scripts/scan_for_nans.py` reference removed.** Confirmed it does
    not exist (the existing slurm scripts call it conditionally:
    `[ -f scripts/scan_for_nans.py ] && ...`). I2's stability check is
    rewritten as a finite-loss + grad-norm scan over the I2 `out.log`
    using the existing per-step training-progress lines, plus a `grep`
    for `nan|inf|loss=nan` in stdout/stderr. Adding `scan_for_nans.py`
    is out of scope for this plan.
  - **I1 monotonicity softened.** v2 made non-monotonicity an automatic
    halt; that is too strict for a short-config signal test (the short
    architecture is small enough that step-time noise can swamp small
    throughput gains). v3 makes it a review trigger — non-monotone by
    < 15% across consecutive points is logged-and-noted; non-monotone
    by ≥ 15% **and reproducible across one rerun of the affected pair**
    halts. (v4: standardized to "one rerun" in both places.)
  - **F2 wording softened.** Linear LR scaling cannot be decided from
    short-config runs alone (different model + different total samples
    per epoch makes the loss curve only weakly indicative). F2 is
    explicitly framed as a cheap screen feeding a full-config decision,
    not the decision itself.
- **v2 (2026-05-05, post-review): structural changes from review notes.**
  - **Two-stage benchmarking.** The short-config sweep is no longer
    allowed to choose the production batch — its purpose is downgraded
    to *launch / DDP / memory-ceiling signal*. A new full-config
    microbenchmark (I2, "decision-maker") gates the production YAML
    edit. Rationale: the short config is `embed_dim=128, num_layers=4,
    scale_factor=3` (`config/plasim_sim52_zgplev_short.yaml:60-65`)
    while production is `embed_dim=256, num_layers=12, scale_factor=1`
    (`config/plasim_sim52_zgplev_full.yaml:64-83`). Step-time and
    activation-memory shapes will not line up.
  - **Fresh-dir requirement.** All sweep and benchmark items now require
    a unique `EXP_DIR` (or unique `--run_num`) per batch-size point.
    `train_plasim.py:171-176` auto-resumes when checkpoints exist, so
    sharing a directory between sweep points contaminates timing,
    optimizer state, and scheduler state.
  - **GB=8 added to the short sweep.** Mandatory points now {4, 8, 16,
    32}. Adds a low-risk per-rank-batch-2 fallback and helps locate
    the first throughput knee.
  - **First production candidate is GB=32, fallback GB=16.** GB=64 is
    explicitly *not* a first-science-run candidate; only allowed after
    a separate stability follow-up.
  - **Logging call site moved.** Logger configuration runs at
    `train_plasim.py:196-203`; the I0 helper now fires after that block
    (just before the `parse_dataset_metadata` call at `:208`). Output
    lands in both `stdout` and the per-experiment `out.log`.
  - **Samples/sec made explicit.** Either logged as a derived field on
    the per-epoch summary, or produced by a documented one-line awk
    parser; the v1 wording said "do not log it" which contradicted
    decision 4. Both options enumerated below; recommendation is the
    derived field.
  - **`submit_zgplev_baseline.slurm` removed from the I3 edit list.** It
    does not pass `--batch_size` today, so there is nothing to remove.
  - **Argparse default corrected.** `--batch_size` defaults to `-1`
    (`makani-src/makani/utils/argument_parser.py:30`), not 0. Behaviour
    is unchanged because `train_plasim.py:134` checks `> 0`, but the
    plan text said 0.
  - **"Dry run" mention dropped.** `--skip_training --skip_validation`
    still constructs the dataloaders, model, and trainer; it is not a
    cheap echo path. Replaced with a true echo via the I0 helper +
    YAML round-trip test (T2/T3 below).
  - **Memory-headroom wording fixed.** Activation memory scales with
    architecture and batch, not "full-dataset batches". Plan reworded.
  - **F5 (reduce `multistep_count`) removed.** Production already runs
    `--multistep_count 1` (`submit_zgplev_full.slurm:117`); there is
    no further reduction available.
  - **Tests expanded.** Added T4 (full-config sanity in fresh dir), T5
    (resume sanity from that checkpoint — LR/scheduler/EMA restore),
    T6 (expected-train-steps formula given sampler `drop_last=True`).
  - **Commit / merge order restructured** to match the recommended
    breakdown: (1) logging + tests, (2) sweep harness only, (3)
    production YAML+SLURM after the full-config benchmark, (4) docs
    with measured numbers.
  - **Comparability note added.** The first speed-optimized run is *not*
    directly comparable to the GB=4 single-GPU baseline as a science
    result. The change-of-record will note the change in effective batch
    + LR; downstream eval should treat it as a new lineage point.

---

## Decisions captured before drafting (user, 2026-05-05)

1. **Resumption.** Apply the fix to the *next* training run. The current
   `sfno_zgplev_full` run (epoch 20+ at draft time) finishes at the
   current rate. Do **not** stop and restart from the current checkpoint
   under different batch / LR semantics — that would invalidate the
   optimizer / scheduler state.
2. **LR scaling rule.** Sqrt scaling. Linear scaling is mentioned only
   as a follow-up "more aggressive" option after the first
   speed-optimized run is observed to be stable.
3. **Plan scope.** Speed fix only — batch + LR + smoke sweep + logging.
   `ema_validation_period: 5` is **not** bundled; first speed-optimized
   production run keeps EMA behaviour unchanged for observability. Listed
   as an optional follow-up.
4. **YAML vs CLI.** YAML is the single source of truth for production
   configs. The smoke sweep is allowed to override `--batch_size` from
   the CLI because the sweep's purpose is intentionally to test
   multiple batch sizes. Logging must explicitly surface the rendered
   global batch, per-rank batch, world size, steps/epoch, and
   samples/sec.
5. **Smoke H100 budget.** ~2 h is acceptable. Sweep covers global batch
   {4, 8, 16, 32} (v2: 8 added). {64} is a follow-up only, not a
   first-science-run candidate.

---

## Verified facts (used by the items below)

Pulled directly from `out.log` of the two runs and from the code paths
that resolve `batch_size`:

- **`src/sfno_training/train_plasim.py:134-141`** treats `--batch_size`
  as the **global** batch and rewrites `params.batch_size` to
  `global // comm.get_size("data")` before the DataLoader is built.
  So `torchrun --nproc_per_node=4 --batch_size 4` ⇒ per-rank=1.
  The argparse default is `-1`
  (`makani-src/makani/utils/argument_parser.py:30`); `:134` only
  overrides the YAML value when `args.batch_size > 0`.
- **Old single-GPU baseline** (`runs/sfno_zgplev_full.pre-ema-20260504/.../out.log`):
  `world_size=1`, `disable_ddp=True`, global=4, per-rank=4, EMA off,
  training step time **61.0 ms**, training time/epoch **~2219 s**,
  validation **~26.5 s**, memory **2.46 GB**.
- **Current 4-rank DDP run** (`runs/sfno_zgplev_full/.../out.log`):
  `world_size=4`, `disable_ddp=False`, global=4, per-rank=1, EMA on
  (period=1), training step time **60.7 ms**, training time/epoch
  **~2210 s**, validation **~50.4 s** (raw + EMA pass), memory
  **11.0 GB**, minimal IO rate **0.21 GB/s**.
- **Optimizer steps per epoch** are identical in both: 36 398 (same
  dataset size 145 592 samples, same global batch 4). Per-step time is
  identical. DDP all-reduce overhead exactly cancels the per-rank work
  reduction. **No DDP scaling is being captured.**
- **DistributedSampler is correct.** `_make_train_eval_sampler`
  (`src/sfno_training/trainer/plasim_trainer.py:84-106`) uses
  `drop_last=True` and shards train + eval across ranks. Per-rank
  expected train batches = `floor(len(train_dataset) /
  global_batch_size)` (and equivalently `len // (per_rank × world_size)`
  with sampler+loader both `drop_last=True`).
- **Validation overhead is the EMA second pass**, not a sharding bug.
  The doubled validation time is exactly `validate_one_epoch`'s raw +
  EMA pass at `plasim_trainer.py:573-630`, gated by
  `ema_validation_period`.
- **Scheduler / warmup units.** Makani steps the scheduler **once per
  epoch** at `makani-src/makani/utils/training/deterministic_trainer.py:377-380`.
  LinearLR is built at `makani-src/makani/utils/driver.py:707-713` with
  `start_factor=lr_start, end_factor=1.0, total_iters=lr_warmup_steps`,
  then composed via `SequentialLR` with the same `milestones`. So
  `lr_warmup_steps: 5` means **5 epochs** of warmup, not 5 optimizer
  steps. The name is misleading; the unit is epochs because of where
  `scheduler.step()` is called. **No scheduler-side code change is
  needed for this plan.**
- **Logger init point.** `train_plasim.py:196-203` configures the file
  logger on rank 0 (`logging_utils.config_logger()` and
  `logging_utils.log_to_file(...)`). Anything earlier than that line
  has to use `print` to land in the per-experiment `out.log`.
- **Memory headroom.** H100 has ~80 GB. Current footprint at per-rank
  batch=1 is 11 GB. Activation memory at `multistep_count=1`,
  `checkpointing_level=2`, bf16 scales with **architecture and per-rank
  batch** — primarily `embed_dim`, `num_layers`, and image size. Going
  from short (`embed_dim=128, num_layers=4, scale_factor=3`) to full
  (`embed_dim=256, num_layers=12, scale_factor=1`) at the same per-rank
  batch is expected to multiply per-sample activation cost
  substantially. Hence the v2 requirement for a full-config
  microbenchmark before the production batch is locked.

---

## Root cause (one line)

The submit scripts hard-code `--batch_size 4` (the *global* batch) on a
4-rank launch, which yields per-rank batch=1. At 64×128×58 channels with
SFNO-12L-256d on H100, batch=1 puts each step in the kernel-launch /
DDP-allreduce-bound regime; per-rank step time at batch=1 is
indistinguishable from per-rank step time at batch=4, and the optimizer
step count per epoch is unchanged. DDP shards the data correctly but
captures no wall-clock speedup.

---

## Goal

After this plan lands, the next 50-epoch full run should hit a
**per-epoch training time of ~½–⅓ of the current 2210 s** (target:
≤1100 s/epoch), at a per-rank batch chosen by the full-config
microbenchmark, with EMA behaviour unchanged from the current run. The
plan does **not** target any specific quality / convergence metric;
convergence is a follow-up once the speed config is locked.

**Comparability caveat.** The next production run uses a different
effective global batch (16 or 32 vs 4) and a different LR (sqrt-scaled).
It is *not* directly comparable to the GB=4 single-GPU baseline as a
science result; treat it as a new lineage point. Downstream evaluation
should compare runs only at matched (batch, LR) settings, or use a
held-out reference whose lineage is documented in the run README.

---

## Items

### I0 — Per-launch DDP-config logging block

**Why.** The current `out.log` already logs `world_size`, `batch_size`,
`global_batch_size`, `disable_ddp`, but they are scattered across
~40 lines of YAML dump and easy to miss. The user asked for explicit
visibility on rendered global batch, per-rank batch, world size,
steps/epoch, and samples/sec. Putting this in **one labelled block at
launch + a small per-epoch derived field** makes A/B comparison and
incident triage trivial, and is the artefact the smoke sweep records.
This item lands first because (a) it is behaviour-preserving, (b) it
ships in the same commit as its CPU tests, and (c) every downstream
sweep / benchmark relies on it for clean A/B numbers.

**What.**

- **Extract a pure batch-resolution helper.** The current global → per-rank
  computation lives inline at `src/sfno_training/train_plasim.py:134-141`,
  which makes it awkward to unit-test (T1) without exercising the full
  CLI / `comm.init` path. v4 lifts it into a small module-level helper:

      def _resolve_batch_sizes(params, data_parallel_size: int) -> int:
          """Return per-rank batch; raises if global batch is not divisible.

          Reads ``params.batch_size`` (already overwritten by an
          ``args.batch_size > 0`` CLI override at the call site) as the
          *global* batch, asserts divisibility by ``data_parallel_size``,
          stores ``params['global_batch_size']`` and the new per-rank
          ``params['batch_size']``, and returns the per-rank value.
          """

  Call site in `main()` becomes a one-liner that passes
  `comm.get_size("data")` in. Behaviour-preserving — same assert message,
  same params keys.
- Add a single `_log_ddp_launch_summary(params)` helper, called from
  `src/sfno_training/train_plasim.py` **after the rank-0 logger is
  configured** (i.e. immediately after `:203` and before
  `parse_dataset_metadata` at `:208`), guarded by `world_rank == 0`.
  Use `logger.info` so the block lands in both stdout and the
  per-experiment `out.log`. (v1 placed this at `:142`, before logger
  setup; that was wrong.)
- The helper logs **one labelled block** with at minimum:
  - `world_size` (= `comm.get_world_size()`)
  - `data_parallel_size` (= `comm.get_size("data")`)
  - `global_batch_size`, `per_rank_batch_size`
  - `expected_train_steps_per_epoch` (computed from
    `len(train_dataset)` is not yet available at this call site, so
    print the formula instead: `floor(len(train) / global_batch_size)`,
    with a note that the actual count appears on the first epoch's
    `training steps:` line; T6 confirms they match)
  - `num_data_workers`, `prefetch_factor`, `persistent_workers`
  - `multistep_count`, `valid_autoreg_steps`
  - `ema.enabled`, `ema_validation_period`
  - `amp_mode`, `checkpointing_level`
- **Samples/sec** (decision 4). Concrete implementation: override
  `PlasimTrainer.log_epoch(self, train_logs, valid_logs, timing_logs)`
  in `src/sfno_training/trainer/plasim_trainer.py`. Inside the
  override:
  1. Compute
     `samples_per_sec = per_rank_batch * world_size /
     (timing_logs["training step time [ms]"] / 1000.0)`
     (guard against `0` step time → emit `0.0`).
  2. Write the value into **both** dicts before delegating:
     - `timing_logs["samples/sec"] = samples_per_sec`
     - `train_logs["samples/sec"] = samples_per_sec`
  3. Defer to `super().log_epoch(train_logs, valid_logs, timing_logs)`.

  This is a ~10-line change with **no explicit `wandb.log` call** in
  the override. Two destinations, two writes, one delegation:

  - **Screen** picks it up because Makani's stock `log_epoch`
    iterates `timing_logs.keys()` dynamically at
    `makani-src/makani/utils/training/deterministic_trainer.py:712-713`.
    Adding to `train_logs` does *not* affect screen output (the screen
    path only prints specific train_logs keys at `:721-724`), so the
    block keeps its existing shape.
  - **wandb** picks it up because Makani's stock `log_epoch` does
    `wandb.log(train_logs, step=self.epoch)` at `:733`, **before** the
    `commit=True` call at `:738`. The new key flows through with no
    further instrumentation.

  Why not the v4 plan (separate `wandb.log` after `super()`): that
  call would land *after* the `commit=True` at `:738`, which under
  wandb's commit semantics either pushes it into the next step or
  drops it depending on the wandb-SDK version. Putting the value into
  `train_logs` instead is one extra dict write and avoids the
  commit-ordering question entirely.

  The current production YAML has `log_to_wandb: False`, so the wandb
  branch is a no-op today; it must be correct for when wandb is
  flipped on. Anchor for the patched call site: Makani computes
  `timing_logs` and calls `self.log_epoch(...)` at
  `makani-src/makani/utils/training/deterministic_trainer.py:421-441`.
  **First-use verification (when wandb is later turned on):** confirm
  one round-trip in the wandb dashboard that `samples/sec` appears at
  the same `step` as `training loss` (i.e. that step assignment is
  correct).
  **Fallback (last resort, only if the override turns out to be more
  invasive than expected):** a documented awk one-liner that extracts
  `(world_size, per_rank_batch, training step time [ms])` from
  `out.log` and prints `samples/sec`. The fallback path does not
  produce wandb-shaped numbers and is harder to A/B in the smoke
  results table.
- Block format suggestion:

      ===== DDP launch summary =====
      world_size                = 4
      data_parallel_size        = 4
      global_batch_size         = 16
      per_rank_batch_size       = 4
      expected_train_steps_per_epoch  = floor(len(train) / 16)
      num_data_workers          = 4
      prefetch_factor           = 4
      persistent_workers        = True
      multistep_count           = 1
      valid_autoreg_steps       = 3
      ema.enabled               = True
      ema_validation_period     = 1
      amp_mode                  = bf16
      checkpointing_level       = 2
      ==============================

**Pass criteria.**

- Block appears exactly once, on rank 0, before the first epoch line.
- All values are read from `params` and `comm` (not hard-coded).
- Samples/sec line appears once per epoch on rank 0.
- The smoke sweep's `out.log` parser (manual or 5-line awk) can pick
  the block out reliably for A/B/C/D comparison.

---

### I1 — Short-config DDP smoke sweep (signal only, NOT decision)

**Why.** This sweep validates launch / NCCL / DDP behaviour and gives
us the first read on per-rank-batch-vs-step-time scaling. It does
**not** decide the production batch — short and full have very
different architectures (`embed_dim=128, num_layers=4, scale_factor=3`
vs `embed_dim=256, num_layers=12, scale_factor=1`), so step time and
activation memory will not transfer cleanly. Treat results as a
sanity-check on direction and a pre-flight for I2.

**What.**

- Reuse `src/sfno_training/submit_zgplev_short_ddp.slurm` as the sweep
  driver. **Do not** change the YAML for the short config — pass
  `--batch_size` on the CLI to override, exactly as today. This is the
  one place where YAML/CLI mismatch is intentional.
- **Fresh-dir requirement (mandatory).** Each sweep point must use a
  unique `EXP_DIR` *or* a unique `--run_num` so that no point
  auto-resumes off another point's checkpoint. Recommended
  implementation: parameterize `EXP_DIR` per sweep point, e.g.
  `EXP_DIR=$SCRATCH/AI-RES/runs/sfno_zgplev_short_ddp_sweep/gb${GB}`.
  The sweep harness (a small wrapper script under
  `scripts/`, name TBD) iterates over GB values and submits one slurm
  job per value with `EXP_DIR` and `--batch_size` exported. The
  per-batch slurm jobs themselves keep `--run_num 0` because the
  directory is already unique.
- Run **4 mandatory sweep points** + **0 optional in this stage**:
  - A: `--batch_size 4`  → per-rank 1  (reproduces current behaviour)
  - B: `--batch_size 8`  → per-rank 2  (low-risk fallback; throughput-knee probe)
  - C: `--batch_size 16` → per-rank 4  (matches old single-GPU per-rank)
  - D: `--batch_size 32` → per-rank 8
  - GB=64 is **deferred** to a follow-up; not part of this sweep.
- For each run, record from `out.log`:
  - `training step time [ms]` (epoch ≥ 2 to get warm)
  - `memory footprint [GB]`
  - `training time [s]`, `validation time [s]`, `epoch time [s]`
  - `training steps`, `validation steps`
  - the I0 launch block + per-epoch samples/sec line
- **Signal rule (what we expect to learn).** A monotone-improving
  throughput curve from A→D with a knee somewhere; no NaN; no NCCL
  hang; memory below the H100 ceiling at all four points. If GB=8
  already saturates throughput while GB=16 and GB=32 plateau, that is
  itself a useful early-warning signal that the bottleneck is *not*
  per-rank batch and we should re-diagnose before booking I2 time.

**Pass criteria (sweep itself).**

- All 4 points complete ≥ 2 epochs.
- No OOM, no NCCL hang, no NaN.
- I0 launch block appears in every `out.log`.
- **Monotonicity is a review trigger, not an automatic halt.** A→D
  throughput should be non-decreasing in expectation, but the short
  architecture is small enough that step-time noise can swamp small
  gains between adjacent points. Treat results as follows:
  - All four monotone non-decreasing within ±5%: pass, proceed to I2.
  - Non-monotone by < 15% between any adjacent pair: log-and-note in
    the PR description; proceed to I2 with both candidate batches
    flagged for the microbench.
  - Non-monotone by ≥ 15% **and reproducible across one rerun of the
    affected pair**: halt and re-diagnose before scheduling I2.

OOM / NCCL hang / NaN at any point still halts unconditionally.

**Out of scope.**

- Adjusting LR or warmup. Short config uses `ReduceLROnPlateau` with
  no warmup; LR scaling is not exercised here.
- Picking a production batch. That is I2's job.

---

### I2 — Full-config microbenchmark (decision-maker)

**Why.** Production architecture is materially different from the short
config. The production batch must be chosen against the actual
production architecture (`embed_dim=256, num_layers=12,
scale_factor=1`) on a fresh `EXP_DIR`, with measurable step time and
memory at the candidate batch sizes that survived I1.

**What.**

- New slurm script `src/sfno_training/submit_zgplev_full_microbench.slurm`
  (or a `--microbench` flag on the existing full slurm; recommend a new
  file for clean separation). The microbench script:
  - Inherits the full YAML (same model, same dataset path, **EMA
    enabled** as in production — see below).
  - **Overlays `max_epochs: 2` per point.** Without this overlay, the
    inherited YAML carries `max_epochs: 50`
    (`src/sfno_training/config/plasim_sim52_zgplev_full.yaml:110`)
    and each microbench point would run the full schedule.
    `scheduler_T_max` is **not** overridden — it stays at 45 because
    it controls the cosine shape, not the run length, and the
    microbench is not a convergence run. Implementation: a small
    YAML overlay file written next to the rendered YAML (e.g.
    `${EXP_DIR}/microbench_overlay.yaml` containing just
    `plasim_sim52_zgplev_full: { max_epochs: 2,
    n_train_samples_per_epoch: <gb*1500>, save_checkpoint: "none" }`),
    or — if `YParams` does not natively merge overlays — a `sed` /
    `yq` pass that rewrites the rendered YAML in place per point
    before `torchrun` is launched. Pick whichever is least invasive
    in commit 3.
  - **Sample budget by step target, not by raw sample count.** Set a
    target of **≈1500 steady-state training steps per point** and
    derive `n_train_samples_per_epoch = global_batch × 1500` per
    batch point:
    - GB=8  → 12 000 samples / epoch (≈1500 steps / epoch)
    - GB=16 → 24 000 samples / epoch
    - GB=32 → 48 000 samples / epoch
    Run **2 epochs per point**. Treat **epoch 1 as warmup** (JIT,
    cudnn-benchmark autotune, DataLoader prefetch fill, NCCL
    bootstrap) and report timings only from epoch 2. At ~60 ms / step
    this is ~90 s of measured training per point plus ~90 s of
    discarded warmup, so each point runs in ~3–5 min including init,
    well within the 2 h H100 budget for 2–3 points.
  - **`--skip_validation` is allowed.** Validation does not exercise
    the optimizer-step path so skipping it does not change what is
    being measured. Speeds up each point by ~50 s.
  - **EMA stays enabled.** There is no `--ema` CLI override
    (`makani-src/makani/utils/argument_parser.py`) and EMA registers
    `optimizer.register_step_post_hook(...)` at
    `src/sfno_training/trainer/plasim_trainer.py:309-311`, which is
    inside the production-shaped optimizer step. Disabling EMA via a
    YAML overlay would change the step time we are measuring; we want
    production-shaped timings, so we leave EMA on. Validation-time EMA
    cost is moot once `--skip_validation` is set.
  - **Checkpoint writing disabled.** Use `save_checkpoint: "none"` via
    a one-line YAML overlay or `--save_checkpoint none` if the parser
    accepts it. Acceptable because microbench results never feed back
    into the production checkpoint.
  - Uses a unique `EXP_DIR` per batch-size point (same fresh-dir rule
    as I1).
- Run **the top 1–2 candidates from I1**, mandatorily including GB=32
  if it survived I1, plus GB=16 as the fallback candidate. So:
  - F1: `--batch_size 32` → per-rank 8  (first production candidate)
  - F2: `--batch_size 16` → per-rank 4  (fallback)
  - F3: optional, only if both F1 and F2 are clean and there is budget
    left: `--batch_size 8` → per-rank 2
- For each, capture from `out.log`:
  - I0 launch block
  - Per-epoch `training step time [ms]`, `memory footprint [GB]`,
    `samples/sec` (from the I0 derived field), `training steps`
  - **Bf16 stability smoke check** (replaces the v2 reference to
    `scripts/scan_for_nans.py`, which does not exist in this repo —
    the existing slurm scripts call it conditionally with
    `[ -f scripts/scan_for_nans.py ] && ...`). This is a **smoke
    signal**, not a numerical / checkpoint-tensor scan; it catches
    obviously-broken runs but is not a substitute for a future
    `scan_for_nans.py`. Stability is verified by:
    1. `grep -E "loss=nan|loss=inf|grad norm=nan|grad norm=inf" out.log`
       returns no matches. Note that tqdm's progress-bar lines are
       interleaved into `out.log` under SLURM's stdout capture, which
       can make raw line counts noisy — the grep is on substring
       presence, not line geometry, so this is robust to tqdm.
    2. The **per-epoch summary** lines emitted by Makani's
       `log_epoch` (`makani-src/makani/utils/training/deterministic_trainer.py:721-724`)
       show finite, float-formatted `training loss` and `gradient
       norm` for the **final** measured epoch (epoch 2 under the
       step-target budget). This is the authoritative signal — the
       per-epoch summary is structured logging, not tqdm.
    3. `grep -E "RuntimeError|CUDA error|NaN" out.log err.log` returns
       no matches.
    Adding a real `scan_for_nans.py` is out of scope for this plan;
    listed as a follow-up if the repeated `[ -f ... ]` guards become a
    nuisance.

**Decision rule.**

1. **Hard-fail filter:** OOM, NCCL hang, NaN before epoch 2 → drop.
2. **Memory headroom guard:** require `memory footprint ≤ 65 GB` (≥15
   GB safety margin under 80 GB H100). The full job runs longer and
   accumulates more transient buffers than the microbench, so leave
   headroom.
3. **First production candidate is GB=32** if F1 passes (1) and (2).
4. **Fallback to GB=16** (F2) if F1 violates the memory guard or shows
   instability.
5. **GB=8 is a deeper fallback** if both F1 and F2 fail. If GB=8 also
   fails, halt and re-diagnose — at GB=8 the per-rank batch is 2,
   which should be safely inside the regime that GB=16 single-GPU
   baseline already operated in. Failure here points at something
   other than per-rank batch.

**Pass criteria.**

- F1 passes, OR F2 passes if F1 hits the memory guard.
- Samples/sec at the chosen point is ≥ 1.7× the GB=4 baseline measured
  on the same `out.log` from the in-flight `sfno_zgplev_full` job
  (~equivalent to 4-rank 60 ms/step at per-rank=1). This is the
  gating "is this actually faster" check before we book the 47.5 h job.

**Out of scope.**

- Convergence assessment. The microbench is a step-time / memory /
  bf16-stability instrument, not a science run.

---

### I3 — Update the production full-config YAML (batch + LR)

**Why.** YAML is the production source of truth (per decision 4). The
current `--batch_size 4` in `submit_zgplev_full.slurm:116` overrides the
YAML's `batch_size: 4` to the same value, which is the silent-override
trap the user wants to avoid. After I2 picks a number, the YAML must
hold the production batch and LR; the slurm script's `--batch_size`
override should be removed.

**What.**

- Edit `src/sfno_training/config/plasim_sim52_zgplev_full.yaml`:
  - Update `batch_size:` from `4` to the global batch chosen by I2
    (expected: 32, fallback 16).
  - Update `lr:` using **sqrt scaling** from the baseline `(batch=4,
    lr=1e-4)`:
    - global 16 → `lr: 2.0E-4`  (= 1e-4 × √4)
    - global 32 → `lr: 2.83E-4` (= 1e-4 × √8; round to 2.8E-4)
    - global 64 is intentionally not handled in this plan.
  - Update the comment on the `batch_size:` line to record (a) the
    chosen value, (b) the sweep + microbench run-ids that justified
    it, (c) the expected per-rank batch on `nproc_per_node=4`.
  - Update the comment on `lr:` to record sqrt-scaling, the baseline
    pair, and a pointer to this plan file.
- Edit `src/sfno_training/submit_zgplev_full.slurm`:
  - **Remove** `--batch_size 4` from the `torchrun` invocation
    (currently lines 111–119). Without `--batch_size`, `train_plasim.py`
    falls through to `params.batch_size` from YAML
    (`train_plasim.py:134-135` only overwrites when `args.batch_size > 0`,
    and the argparse default is `-1`).
  - Update the comment block at lines 105–110 to reflect the new YAML
    source-of-truth contract.
- **Do not edit** `src/sfno_training/submit_zgplev_baseline.slurm`. It
  does not pass `--batch_size` today (`:60-65`); there is nothing to
  remove. (v1 incorrectly listed it.)
- **Do not** change `lr_warmup_steps`, `scheduler_T_max`,
  `scheduler_min_lr`, `lr_start`, `optimizer_*`, or any EMA settings.
  Rationale spelled out in the warmup section below.

**Warmup note (intentionally conservative).**

The current YAML has `lr_warmup_steps: 5`, which Makani's scheduler
plumbing interprets as **5 epochs** of LinearLR warmup (verified above
under "Scheduler / warmup units"), not 5 optimizer steps. Three options
were considered:

- **(A) Preserve epochs of warmup (chosen).** Keep `lr_warmup_steps: 5`.
  Warmup covers the same wall-clock fraction of training and the same
  number of *samples seen*. Per-epoch optimizer-step count drops by the
  batch-scale factor (4× at global=16, 8× at global=32), so each
  in-warmup epoch's LR-jump is 4–8× larger in absolute terms. Combined
  with sqrt LR scaling (peak only 2–2.83×), the in-warmup ramp is mild
  enough to be a safe first try.
- **(B) Preserve optimizer steps of warmup.** Bump `lr_warmup_steps` to
  20 (global=16) or 40 (global=32). Safer in principle but spends 4–8×
  more epochs at sub-peak LR, which is wasteful when we are
  speed-optimizing. Defer to a follow-up if (A) shows divergence in the
  first ~5 epochs.
- **(C) Move warmup to per-step.** Out of scope — would require touching
  Makani's vendored scheduler stepping. Defer indefinitely.

**Pass criteria (I3 itself, before any production launch).**

- `python -c "from makani.utils.YParams import YParams; ..."` round-trips
  the rendered YAML and reports the new `batch_size`, `lr` (T3 below).
- A truncated full-config sanity run (T4) prints the I0 launch block
  with the expected global / per-rank batch and reaches epoch 2 cleanly.

---

### I4 — Plan-anchored doc updates (with measured numbers)

**Why.** Cross-link the change to the diagnosis so the next reviewer
does not re-derive the result, and bake the chosen numbers into the
record once they are measured.

**What.**

- Update `docs/INDEX.md` to add this plan to the index.
- Add a one-paragraph note to `src/sfno_training/README.md` (the
  "Performance" section if it exists, otherwise a new section) that
  records: chosen global batch, per-rank batch, sqrt-LR rule, the
  pre/post per-epoch wall-clock numbers from I2 and the sanity run,
  and a pointer to this plan plus the I1 / I2 run-ids.
- Append a "Measured results" subsection to **this plan file** with
  the actual sweep + microbench numbers, after they exist. Keep it in
  the same file rather than creating a separate doc — it is the same
  decision record.
- **Do not** edit `src/plasim_makani_packager/templates/...` — packager
  outputs are unrelated to training batch.

---

## Tests

### CPU-only, in-tree (`tests/sfno_training/`)

These are tiny and cheap; they catch silent regressions in batch-size
plumbing, the new logging block, and the train-step formula. **No GPU
required.**

- **T1 — `_resolve_batch_sizes` helper.**
  Test the pure helper extracted in I0 (no CLI, no `comm.init`):
  - **T1a (raises on indivisible):** `params.batch_size=15`,
    `data_parallel_size=4` → `AssertionError`.
  - **T1b (happy path):** `params.batch_size=16`,
    `data_parallel_size=4` → returns `4`,
    `params["global_batch_size"] == 16`, `params["batch_size"] == 4`.
  - **T1c (single-rank pass-through):** `params.batch_size=4`,
    `data_parallel_size=1` → returns `4`,
    `params["global_batch_size"] == 4`, `params["batch_size"] == 4`.
  Pin these so the divisibility contract is not silently removed.
- **T2 — DDP launch summary content.**
  Capture the I0 helper's output for a representative `params` mock
  and assert each labelled key is present and correctly populated.
  Use `caplog`; do not parse rendered text positionally.
- **T3a — YAML loadability + current values (commit 1).**
  Load `plasim_sim52_zgplev_full.yaml` with `YParams` and assert it
  parses cleanly, has the keys `batch_size` and `lr`, and matches the
  *current* (pre-edit) values. This pins YAML structural integrity
  before the production edit lands; it is the test that fails loudly if
  someone accidentally breaks the file in an unrelated commit.
- **T3b — YAML post-edit values (commit 4, alongside the I3 edit).**
  Same loader, but assert `batch_size == <I2-chosen GB>` and
  `lr == <sqrt-scaled value>`. Guards against accidental drift after
  the production numbers are locked in. Add this in the same commit as
  the YAML edit so the test and the value land together.
- **T6 — Expected-train-steps formula.**
  Given a fake `len(train_dataset)`, `global_batch_size`,
  `data_num_shards`, with sampler `drop_last=True`, assert that
  per-rank `floor(len / global_batch_size)` matches the formula
  documented in I0. This is the value the user will diff against
  `training steps:` on the first epoch line.

### GPU-required, off-tree (sweep + microbench artefacts)

- **I1 sweep results.** 4 short-config DDP runs, fresh dirs, GB ∈
  {4, 8, 16, 32}. Per the I1 pass criteria.
- **I2 microbench results.** 1–3 full-config truncated runs, fresh
  dirs, GB ∈ {32, 16, optionally 8}. Per the I2 decision rule.

### Production-launch sanity (full config, fresh dir)

- **T4 — Full-config sanity run.** Submit
  `submit_zgplev_full.slurm` with `SBATCH -t` reduced to **1 h**, a
  unique `EXP_DIR`, validation enabled, EMA enabled, checkpoint write
  enabled. Confirm:
  - I0 launch block reports the YAML's `batch_size` and the expected
    per-rank value.
  - First-epoch `training steps:` matches T6's formula.
  - `training step time [ms]` is in the range from I2.
  - One full checkpoint writes successfully.
- **T5 — Resume sanity.** Resubmit the same script with the same
  `EXP_DIR`. Confirm the run resumes, the optimizer/scheduler/EMA
  state restores (checkpoint key set + `EMA resume:` log line at
  `plasim_trainer.py:482-488`), the LR matches the post-warmup value
  (or the in-warmup ramp value if epoch < 5), and at least one further
  epoch completes.

Only after T4 + T5 pass do we book the real 47.5 h job.

---

## Merge / commit order (recommended breakdown)

1. **Commit 1 — logging helper + batch-resolution refactor + CPU tests.**
   I0 (`_resolve_batch_sizes` extraction, launch-summary block,
   `PlasimTrainer.log_epoch` override for `samples/sec` with explicit
   wandb branch) + T1 (T1a/b/c), T2, **T3a**, T6.
   Behaviour-preserving; safe to land while the in-flight run continues.
   **T3b is deferred to commit 4** because it asserts I2-chosen values
   that do not exist yet.
2. **Commit 2 — sweep harness only.** I1's parameterized
   `submit_zgplev_short_ddp.slurm` (or wrapper script). Production
   defaults unchanged. Run the sweep, attach numbers to the PR.
3. **Commit 3 — full-config microbench harness + measured numbers.**
   `submit_zgplev_full_microbench.slurm`, fresh-dir parameterization,
   step-target sample budget, EMA enabled, validation/checkpoint
   skipped. Run the microbench, attach numbers to the PR.
4. **Commit 4 — production YAML + SLURM update + T3b.** I3's edits to
   `plasim_sim52_zgplev_full.yaml` and `submit_zgplev_full.slurm`,
   anchored to the I2 numbers, plus T3b asserting the new values.
   Includes T4 + T5 transcripts.
5. **Commit 5 — docs + index.** I4's `INDEX.md`, README, and
   "Measured results" appendix to this plan file.

The in-flight `sfno_zgplev_full` job is using a *rendered* YAML at
`$EXP_DIR/plasim_sim52_zgplev_full.rendered.yaml`, so editing the
*template* in commits 4–5 does not affect it. Double-check with `diff`
before commit 4 lands.

---

## Rollback / abort criteria

- **I1 result handling (tiered, matches the I1 pass criteria).**
  - All four points monotone non-decreasing within ±5%: pass, proceed
    to I2 with the planned candidate set (GB=32 first, GB=16
    fallback).
  - Non-monotone by < 15% between any adjacent pair: log-and-note in
    the PR description; proceed to I2 with both candidate batches
    flagged for the microbench.
  - Non-monotone by ≥ 15% between any adjacent pair **and reproducible
    across one rerun of the affected pair**: halt. Re-diagnose; do
    not proceed to I2.
  - OOM / NCCL hang / NaN at any point: halt unconditionally.
- **I2 GB=32 and GB=16 both fail.** Halt. Re-open the diagnosis:
  candidates are (a) NCCL all-reduce bandwidth on h100 single-node,
  (b) HDF5 worker-pool contention at per-rank batch ≥ 4, (c) an
  unexpected serialization in `PlasimForcingDataset.__getitem__`.
- **T4 epoch time > 0.7 × current 2210 s.** Halt before the 47.5 h
  resubmit. Investigate before committing the long run.
- **T5 resume produces a different LR than the saved scheduler
  state.** Halt. The scheduler state must restore exactly; an LR
  mismatch on resume is the canonical symptom of a half-applied YAML
  edit or a stale rendered YAML.
- **First-3-epoch loss curve diverges** in the production run vs the
  pre-EMA baseline at the matched-by-samples checkpoint. Roll the YAML
  back to `batch_size: 4, lr: 1.0E-4`. Reconsider warmup option (B)
  above.

Rollback for any of the above is a single revert of the I3 commit; the
I0 logging block + I1 / I2 harnesses stay.

---

## Non-goals (explicit)

- **No EMA period change.** `ema_validation_period: 1` stays. The first
  speed-optimized production run keeps EMA observability identical to
  the current run so any wall-clock change is attributable to the
  batch / LR change, not to EMA cadence.
- **No linear LR scaling.** Sqrt only.
- **No change to** `lr_warmup_steps`, `scheduler_T_max`, optimizer betas,
  weight decay, max grad norm, or any architectural knob.
- **No change to** `multistep_count`, `valid_autoreg_steps`,
  `n_history`, `n_future`, or `prediction_type`. Production already
  uses `multistep_count=1`; it is at its floor.
- **No change to** the sampler, the dataloader knobs (already P2-tuned),
  or the local Makani `non_blocking=True` patch.
- **No checkpoint-format work** (P3 was deferred and stays deferred).
- **No multi-node DDP.** Single-node 4×H100 only.
- **No tensor / model / spatial parallelism.**
- **No edits to `submit_zgplev_baseline.slurm`** — it does not pass
  `--batch_size`, so there is nothing to remove there.

---

## Optional follow-ups (separate plans, not in this commit)

- **F1 — `ema_validation_period: 5`.** Saves ~20 s/epoch ≈ 17 min over
  50 epochs. Already plumbed; one-line YAML change. Land after the
  first speed-optimized run is observed to be stable.
- **F2 — Linear LR scaling experiment.** Compare sqrt vs linear at the
  chosen production batch. Short-config runs are a **cheap screen
  only** — they use a different architecture (`embed_dim=128,
  num_layers=4, scale_factor=3` vs production's
  `embed_dim=256, num_layers=12, scale_factor=1`) and a much smaller
  per-epoch sample budget, so loss-curve shape is only weakly
  indicative of full-config behaviour. Use the screen to rule out
  obviously-divergent settings; the deciding experiment is a
  full-config A/B at matched checkpoints, scoped in F2's own plan.
- **F3 — Warmup option (B).** Increase `lr_warmup_steps` proportionally
  if (A) shows early-epoch instability.
- **F4 — Per-step warmup (option C).** Touches Makani's vendored
  scheduler. Significantly larger surface area; only if (A) and (B)
  both prove insufficient.
- **F5 — GB=64 stability run.** Single short-config + microbench point
  at GB=64 / per-rank=16, with `checkpointing_level=1` if needed for
  memory headroom. Only after the GB=32 production run is observed to
  be stable for ≥ 5 epochs.
