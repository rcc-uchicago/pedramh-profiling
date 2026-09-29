# Handoff — compare the updated ACE2 codebase (`fme` perf branch) against the vendored one

*Written 2026-09-29 by the ACE2 monitor session. Every fact in §1 was read from a primary on that date;
re-verify before a decision rests on it.*

## 0. The task, in one paragraph

An updated ACE2 codebase arrived as `pedramh-profiling/ace2_updated_codebase.gz` and has been extracted
**beside** the vendored copy, to `ACE2_retrain/ace_exp_updated/` — **not over `ACE2_retrain/ace_exp/`**
(§1.3 explains why that matters). Your job: establish whether the update is safe to adopt on Polaris —
(a) that its **default path computes exactly what the current code computes**, (b) that our harness
still works on it, (c) what its opt-in speedups are worth on A100s — and bring the operator a
recommendation. **You do not swap the code in.** That is the operator's call, and not before the
production run finishes (§2).

## 1. State at handoff — verified

### 1.1 What the update is

| item | value | source |
|---|---|---|
| archive | `ace2_updated_codebase.gz` at the repo root, 61.6 MiB gz, 2,973 members, 83.7 MiB; one top dir `updated_codebase/ace_exp/`; no absolute paths, no symlinks | `tarfile` listing |
| extracted to | `ACE2_retrain/ace_exp_updated/` in the **shared checkout** (untracked), 2,636 files incl. its own `.git` | this session, `tar --strip-components=2` |
| its git state | branch **`perf`**, HEAD **`6ce252e2f`** (Katharine Rucker, 2026-09-28, on Delta): *"Perf: NVTX instrumentation, env-gated torch.compile, non_blocking H2D"*. Working tree == HEAD exactly (no uncommitted edits; checked by blob hash, not `git status`) | `git -C … log`, blob-hash compare |
| our vendored copy | `ACE2_retrain/ace_exp/` == ai2cm `fme` **`1c3ebad80`** code-exact (vendored in repo commit `804389b4`); it lacks 50 non-code files the repo's `.gitignore` blocks (binary `.pt`/`.npy` test fixtures, and `scripts/compute_enso_index/compute_enso_index.py`) | blob-hash compare vs `git ls-tree 1c3ebad80` |
| distance | **exactly one commit**: `1c3ebad80..6ce252e2f`. Both report `fme.__version__ = "2026.5.1"`. Our copy **already contains** the earlier perf commit `67242e348` (TF32, native SHT, spectral no-op copy, foreach EMA, DDP flags) | `git log` in the new repo |

`git diff --stat 1c3ebad80 6ce252e2f` — 18 files, +1,777 −217:
4 new docs (`NVTX_ANNOTATIONS.md`, `OPTIMIZATION_CHANGES.md`, `RUN_OPTIMIZED_TRAINING.md`,
`TRAINING_SPEEDUP_HISTORY.md`), new `fme/core/nvtx.py` + `fme/core/test_nvtx.py`, and 12 modified:
`fme/core/generics/{trainer,validation,inference}.py`, `fme/ace/stepper/single_module.py`,
`fme/core/step/single_module.py`, `fme/core/optimization.py`, `fme/ace/train/train.py`,
`fme/ace/data_loading/{batch_data,dataloader,getters,gridded_data}.py`,
`fme/core/distributed/torch_distributed.py`.

### 1.2 What changes on the DEFAULT path (no env vars set) — first-pass classification, to be confirmed (T1)

| change | default | math? |
|---|---|---|
| NVTX ranges throughout (`fme/core/nvtx.py`, ~100 range names, hierarchical: `train_batch/train_on_batch`, `forward_step/predict`, …) | **ON** whenever a GPU is used (`FME_NVTX` defaults to `"1"`) | no — but it adds ranges to every nsys capture (§T5) |
| `batch_data.py`: `v.to(device)` → `v.to(device, non_blocking=True)` | **ON** | should not — same stream; confirm within run-to-run spread in T3 |
| `torch.compile` of the step module (`step/single_module.py`) | off (`FME_COMPILE=0`) | **yes, bitwise** when on (their doc: "bitwise, not mathematically") |
| `optimize_ddp` override | off unless `FME_COMPILE` (`FME_COMPILE_DDP_OPT`) | no |
| DDP `bucket_cap_mb` | `FME_DDP_BUCKET_MB`, default `25` = torch's own default | no |
| DataLoader `persistent_workers` | unchanged unless `FME_PERSISTENT_WORKERS` is set | no (worker lifetime only) |

Checked by this session: after discarding NVTX lines and pure re-indentation, the remaining trainer /
stepper / validation / inference hunks are context-manager **merges** — e.g.
`with optimizer.autocast(), nvtx_range("stepper_step"):`, `with context, nvtx_range("forward_step"):`,
`with torch.no_grad(), self.validation_context(), nvtx_range("train_evaluation"):`. Occurrence counts of
`torch.no_grad()`, `.autocast()`, `validation_context()`, `ema_context`, `GlobalTimer()`, `.backward(`,
`.detach()` are **identical** old→new in all seven core files; the only count differences are range-name
strings (`train_on_batch/step_weights`, `optimizer_zero_grad`). **No `def`/`class` line was added,
removed or renamed** in the six files our harness patches. This is a line-count screen, not a proof — T1
reads the diff, T3 measures.

### 1.3 Why it was NOT extracted over `ace_exp` — the coupling

- `fme` in the production venv is an **editable install of the shared checkout's `ACE2_retrain/ace_exp`**:
  `$MEMBER_ROOT/conda-envs/fme-venv/…/fme-2026.5.1.dist-info/direct_url.json` →
  `file:///lus/eagle/…/pedramh-profiling/ACE2_retrain/ace_exp`, `"editable": true`.
  Whatever sits in that directory is what every ACE2 job imports **at start**.
- The **ACE2 production resume, job 7664776** (preemptable, 72 h, `EPOCHS=45`) is **queued**. It resumes
  `$MEMBER_ROOT/runs/ace2_polaris/ace2_prod_1n_b8` from epoch 23. Replacing `ace_exp`, or
  `pip install -e ace_exp_updated`, would make the production run span two codebases.
- The launcher guards this. `polaris_ace2_train.pbs:149–156` exits with `ERROR ACE2_WRONG_CHECKOUT`
  unless the `fme` it imports resolves (by `realpath`) to `${ACE2_DIR}/ace_exp/fme`. Its comment
  (`:144–148`) explains why: `ace2_telemetry.py` and `ace2_nvtx.py` patch `fme` **by attribute path**,
  and on a different tree a missing attribute is **skipped silently**, which leaves a run with no
  measurement and no error. Any T3 launch of the new tree must satisfy this check honestly. For example,
  run from a worktree whose `ACE2_retrain/ace_exp` resolves to `ace_exp_updated`. Don't delete the check.
  → `MONITOR_ace2_production_training.md` §6 STOP 3 and §9 list `ace_exp` as protected.
- The venv already has **torch 2.10.0+cu129 and triton 3.6.0**, the versions the update's
  `torch.compile` path needs (on Delta that took a separate `fme_compile` env).

### 1.4 Our harness's coupling to `fme` internals

`ACE2_retrain/ace2_telemetry.py` monkeypatches, at lines 381–509: `Trainer.train_one_epoch`,
`GriddedData.subset_loader`, `GriddedData.alternate_shuffle`, `TrainStepper.train_on_batch`,
`Optimization.step_scheduler(valid_loss=, is_iteration=)`, `WandB.log(data, step, sleep=, commit=)`.
`ACE2_retrain/ace2_nvtx.py` emits only `step_{n}` — **no name collision** with fme's new ranges.

### 1.5 The update's own claims — Delta GH200 aarch64, unverified here

`OPTIMIZATION_CHANGES.md` and `RUN_OPTIMIZED_TRAINING.md` in `ace_exp_updated/`:
- the workload is **GPU-bound**: 86–87% busy, 94% of GPU time in the SFNO forward and backward
  (ours measures `gpu_busy_frac` 0.954–0.970);
- `torch.compile` (inductor) **+8.7%** at 1 GPU and **+7.2%** training throughput at 4 GPUs, but only
  with `FME_COMPILE_DDP_OPT=0` (with DDPOptimizer on, compile is **−4.5%**);
- `non_blocking` gives 0% end-to-end; `num_data_workers` 8→32 gives nothing; the cudagraphs backend was
  −11.6% and was reverted;
- their gate is `FME_FORCE_CPU=1 pytest -q fme/ace/stepper fme/core/step fme/ace/data_loading
  fme/core/test_optimization.py fme/core/generics` → **493 passed, 2 skipped**, "identical on torch
  2.5.1 and 2.10, including all 5 `.pt` numerical regression baselines".
- ⚠ Stale sentence: `OPTIMIZATION_CHANGES.md` says "Nothing here is committed yet". It **is** committed,
  as `6ce252e2f`.

## 2. STOPs — the ways to break the production run

1. **Do not modify `ACE2_retrain/ace_exp/`, and do not `pip install` anything into `fme-venv`,** while
   7664776 is queued or running (`qstat -x 7664776`). This covers `pip install -e ace_exp_updated`,
   which would repoint the editable install.
2. **Do not edit `polaris_ace2_train.pbs`, `polaris_ace2_env.sh`, `ace2_telemetry.py` or
   `config_polaris.yaml` in the shared checkout** while it is queued. The resume imports them at start.
   Do harness work in a worktree.
3. **Do not commit `ace_exp_updated/` as-is.** It has a nested `.git` and `.pt`/`.npy`/`.fits`
   binaries (CLAUDE.md #8). Vendoring, if the operator approves it, follows `804389b4`'s pattern.
4. **Do not adopt `FME_COMPILE`, or anything else that is not bitwise, without the DESIGN §4
   equivalence check** (CLAUDE.md #1, #6). Never loosen a tolerance.
5. **No Python on login nodes** (operator ruling 2026-09-24). Every test below is a `debug` job; 1-node
   debug is pre-authorized. `preemptable` and `capacity` get surfaced to the operator first.
6. The login node is **pid-capped at 256** (it ran at 210–237 during this session). No polling loops
   and no `sleep` watchers. Use `qsub -W depend=` or check once when asked.

## 3. Tasks, in order — each with its PASS

**T1 — Read the whole diff.** `git -C ACE2_retrain/ace_exp_updated diff 1c3ebad80 6ce252e2f -- fme`
(object-only, safe on Lustre; never `git status` in there). Confirm or refute §1.2 hunk by hunk,
especially `trainer.py` (+110/−64) and `stepper/single_module.py` (+112/−82). PASS = a written
classification. Every hunk is NVTX, context merge, env-gated, or `non_blocking`, and any exception is
named with its file:line.

**T2 — fme's own test gate on Polaris (debug job, CPU).** Run from `ace_exp_updated` with it **first
on `PYTHONPATH`**, and print `fme.__file__` so the editable install provably didn't win. Run the §1.5
pytest command. The regression `.pt` fixtures exist only in `ace_exp_updated`, because our vendored
copy dropped them, so run the same suite on the old tree using `ace_exp_updated`'s git to check out
`1c3ebad80` into a scratch dir. PASS = same pass/skip counts old vs new, with the `.pt` regression tests
included. Delta's figure is 493/2; x86 may differ, so the old-vs-new pair on Polaris is what counts.

**T3 — Default-path equivalence on GPU (the DESIGN §4 gate).** Same config, same seed, a short run of a
few hundred steps plus one validation pass. Run it three ways: old tree; new tree with `FME_NVTX=1`;
new tree with `FME_NVTX=0`. The reference behavior: two independent same-config, `seed: 3`, 1-node
runs agreed **bitwise on validation loss** and to **~1 float32 ULP (7.7e-8 relative) on the cross-rank
reduced train loss** (CHANGELOG 2026-09-04 cont. 2, jobs 7591998/7592103). The tolerance that entry
derives is **~1e-7 relative on reduced scalars, 1 node**. It is two log scrapes, not a §4.1 tensor
baseline, so capture a per-step trace. Reuse `ACE2_retrain/polaris/compare_resume_trace.py` for the
compare, and take the short shape from the launcher's resume-gate settings (`FULL_VAL=0`,
`SAMPLES_PER_EPOCH`). The launcher's `ACE2_WRONG_CHECKOUT` check (§1.3) must pass for the tree under
test, so launch from a worktree, not the shared launcher (STOP 2). Run old-vs-old as well, to see the
same-code spread on the day. PASS = new-vs-old spread no larger than old-vs-old, and within ~1e-7
relative. Anything larger means you report the first differing step and quantity, and **stop**. Never
widen the tolerance to pass.

**T4 — Harness compatibility (same jobs as T3).** Every `ace2_telemetry.py` patch still fires, and the
telemetry CSV row has exactly the 21 columns of `epoch_telemetry.COLUMNS` (CLAUDE.md #10). PASS = a
row written for the new tree whose per-step timing fields are populated, not zero or empty.

**T5 — NVTX policy.** The new default emits ~100 range names on every GPU run. Measure samples/s with
`FME_NVTX=1` vs `=0` from T3 (the 1-node noise floor is ±0.1%). Check that `parse_nsys.py` and any ACE2
nsys comparison still read correctly with the extra rows. PASS = a measured overhead number and a
recommendation, either `FME_NVTX=0` in production or keep it on.

**T6 — Opt-in `torch.compile` on A100 (only if the operator wants it).** It's cheap to try, since torch
2.10 and triton 3.6 are already in the venv. Settings: `FME_COMPILE=1`, `FME_COMPILE_DDP_OPT=0`,
`TORCHINDUCTOR_COMPILE_THREADS` capped. Keep `TORCHINDUCTOR_CACHE_DIR` on `/eagle`
(memory `polaris-resource-conventions`). Delta's aarch64 CC/CXX trap should not apply on x86, but
verify. Two measurements:
- (a) Throughput A/B, interleaved, ≥3 reps.
- (b) Equivalence. Compile is **not bitwise**, so this needs the DESIGN §4 tolerance check, and the
  loss curve must match within that tolerance.

PASS = both, with numbers. Our step is 720 ms median at 96.6% GPU-busy, so Delta's +7% is plausible
but unmeasured on A100.

**T7 — Report.** Write a CHANGELOG entry (measured, with job ids), then a recommendation to the operator
covering (1) whether to vendor `6ce252e2f` into `ace_exp`, and when (after 7664776 **and** any further
resume of the 45-epoch run finish, never between segments); (2) the NVTX default; (3) compile yes/no.

## 4. Open for the operator

1. Adopt the update at all, and when. The default is after the 45-epoch production run ends, gated on
   T1–T4.
2. Evaluate `FME_COMPILE` on Polaris (T6)? That is a few debug jobs.
3. Housekeeping at the repo root (untracked). `ace2_updated_codebase.gz` (61.6 MiB) is kept until
   adoption is decided. Two **5.6 GB core dumps** from 2026-09-28 03:19–03:22 (`core.2141696`,
   `core.288365`, plus one in `.claude/worktrees/monitor-ace2/`) date from the login-node trouble. They
   are ignored by `.gitignore`'s `core*` rule, and nothing here deletes them.

## 5. Where to look

`ACE2_retrain/ace_exp_updated/{OPTIMIZATION_CHANGES,RUN_OPTIMIZED_TRAINING,NVTX_ANNOTATIONS,TRAINING_SPEEDUP_HISTORY}.md`
· `MONITOR_ace2_production_training.md` (in `.claude/worktrees/monitor-ace2/`) ·
`ACE2_retrain/polaris/polaris_ace2_train.pbs:105–160` (how `fme` is located and checked) ·
`ACE2_retrain/ace2_telemetry.py:381–509` · memory `ace2-production-run-state`,
`git-hangs-on-polaris-lustre`, `polaris-login-pid-cap`.
