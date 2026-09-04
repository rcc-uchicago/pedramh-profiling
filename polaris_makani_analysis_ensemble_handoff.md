# makani handoff — finish the analysis, then build the ensemble

*Written 2026-09-04, closing the session that completed the 1-node production run.*

Read in this order: **`makani_bench_report.md` §5j-§5m and §7e** (the measurements),
then CHANGELOG `2026-09-04`, then this file. Priorities live in `TODO.md`.
`$MEMBER_ROOT` = `/eagle/projects/lighthouse-uchicago/members/mehta5`.
PASS is always the log token, never `rc` (CLAUDE.md #14).

---

## 0. State at handoff

**The training campaign is finished.** Nothing below requires another long run.

| item | status |
|---|---|
| 1-node production (7585080) | ✅ **complete** — 243/243 epochs, `Exit_status 0`, best validation loss **0.01284**, 46.3 node-hours, 12 snapshot checkpoints |
| learning-rate ceiling | ✅ **characterised** — (2e-3, 3e-3], does **not** move with batch size, 9 of 9 arms collapsed above it (§7e) |
| batch 48 | ✅ **closed** on three axes (§5l) |
| memory model | ✅ **works** — retrodicts the batch-64 out-of-memory failure (§5g) |
| kernel-level profile | ✅ **done** (7591822) — **34.9 % of GPU compute time computes nothing** (§5m) |
| C1 rollout fine-tune (7591605) | 🔵 **queued** on `capacity`, 24 epochs, ~7 h |
| lead-time scorecard (7592575/6/7) | 🔵 **queued** on `preemptable`, `va=3/10/20` |

⚠ **Do not re-derive any of the above.** Every one cost real jobs and several are
corrections of earlier wrong answers — the retired claims are listed in §5 below.

---

## 1. First task — read the lead-time ladder. It decides everything after it.

Jobs **7592575 (va=3)**, **7592576 (va=10)**, **7592577 (va=20)**, scoring
production's `best_ckpt` at three rollout lengths.

```bash
grep "validation loss" $MEMBER_ROOT/runs/makani_mn_scaling/score_prod1n_b32_sgdr_va*.log
```

**Check `va=3` FIRST — it is a control.** It must reproduce **0.01284**. If it does
not, the scoring path is wrong and the other two arms mean nothing. (This is the same
discipline that made the 9-arm factorial readable, and the one time it was skipped the
batch-48 result became uninterpretable.)

Then read the *shape* of loss versus lead time. **It distinguishes two different
diseases, and they need different cures:**

| shape from va=3 → va=20 | diagnosis | what to do |
|---|---|---|
| error grows fast, possibly unstably | **exposure bias** — the model never saw its own imperfect output | C1 is correctly aimed; continue §2 |
| error rises then **flattens** toward a plateau | **blurring / under-dispersion** — MSE training regresses to the mean, which compounds | C1 will underdeliver; the fix is CRPS, §4 |
| error barely moves | **neither** — compounding is not our problem | ⚠ **STOP.** C1 and the whole ensemble direction are misaimed. Find the real cause of "inference worse than expected" first |

That third row is a live possibility and nobody has excluded it. The premise that
`n_future: 0` explains the weak inference was **inferred, never confirmed**.

---

## 2. Second task — score C1 the same way

When 7591605 finishes:

```bash
bash makani_sfno/polaris/submit_rollout_scorecard.sh c1_rollout_full_b16 3 10 20
```

Compare **models at a fixed rollout length**. Do *not* compare across lengths — a
longer rollout needs more trailing frames, so the set of valid start indices differs.

⚠ **Expect little or nothing at va=3.** Four steps is 24 forecast hours; compounding is
negligible there. C1's one-epoch probe already read 0.013169 against the base model's
0.01284 — **2.6 % worse** — and that is the *expected* shape of a rollout fine-tune
trading single-step accuracy for multi-step stability. A flat or slightly worse va=3 is
**not** evidence C1 failed. The signal, if it exists, is at va=10 and va=20.

---

## 3. Third task — build the ensemble

We have **12 snapshot checkpoints** (cycle ends, epochs 23…243 every 20). They are
usable three ways; only the third needs building.

1. **As a warm-start for any later stage** — works today.
2. **Scored individually** — works today, via §2's script.
3. **Combined into an ensemble forecast** — **blocked**, and this is the task.

### Why it is blocked

Ensembling requires combining predictions *before* scoring, and **nothing on the
Polaris path writes predictions**:

* `save_raw_forecasts: True` sits in our config with **no reader** in
  `makani/utils/**` or the top-level entrypoints. It is a dead key.
* The machinery that does save (`utils/inference/rollout_buffer.py`, `Inferencer`) is
  **hard-gated off in our fork**: `assert mode != "inference"` in
  `_plasim_get_dataloader`, because stock `Inferencer` has no slot for our forcing
  channels and would silently emit physically wrong output.

### The good news — the port is small, and it was audited on 2026-09-04

`makani_sfno/src/sfno_inference/` already has the pieces, and **the rollout body is
already channel-generic**:

```python
rollout_driver.py:163-164
n_state = int(getattr(eval_params, "n_state_channels", 52))
n_out   = int(getattr(eval_params, "N_out_channels", 53))
```

52 and 53 are only *defaults*. The PlaSim assumptions are concentrated in four places:

| # | location | change |
|---|---|---|
| 1 | `checkpoint_loader.py:74-82` | five hard asserts on 58/53/52/1/6 → a **self-consistency check driven by config** (inputs = state + forcing; outputs = state + diagnostic). PlaSim satisfies it as 52+6/52+1, E3SM as 100+7/100+1 |
| 2 | `checkpoint_loader.py:237` | `assert wrapper.model.out_chans == 53` → compare against `N_out_channels` |
| 3 | `rollout_driver.py:83` | `_load_run_norm_stats(..., n_out: int = 53)` → read from `eval_params` |
| 4 | `rollout_driver.py:281-302` | 🐛 **`_extract_truth_sic` — a silent-wrong-physics bug for our data.** See below |

🐛 **Item 4 in full, because it will not announce itself.** The function recovers sea-ice
concentration from the forcing tensor assuming the PlaSim order
`['lsm','sg','z0','sst','rsdt','sic']` and reading **channel 5**. It guards on forcing
**length** (`if fb.shape[0] < 6: disable`), not identity. Our E3SM pack has **7**
forcings, so `7 < 6` is false, the guard passes, it reads channel 5, and returns a
confidently-labelled `truth_sic` that is **a different variable entirely** — no error, no
warning. **Gate it on the forcing-channel names, not the count**, and disable it for
non-PlaSim contracts.

### Then the driver itself

Loop **12 members × N initial conditions**, accumulate **ensemble mean**, **spread**, and
**CRPS** (makani already ships `utils/metrics/functions.py:422 GeometricCRPS`).

```python
from sfno_inference.checkpoint_loader import load_eval_params, build_wrapper_from_checkpoint
from sfno_inference.rollout_driver import rollout_one_ic
eval_params = load_eval_params(RUN, K=56)          # K = 6-hour steps; 56 = 14 days
wrapper     = build_wrapper_from_checkpoint(eval_params, ckpt_path, device)
result      = rollout_one_ic(wrapper=wrapper, dataset=ds, ic_global_idx=0,
                             eval_params=eval_params, device=device)
```

⚠ **`sfno_inference` is shared with the Stampede3 `eval-sfno-own` path and lives in a
`git subtree`.** Every change must **generalise**, never "make it work for E3SM"
(CLAUDE.md #5). The four fixes above all qualify — they make the code read its contract
from config instead of assuming one. Keep edits minimal and contiguous (subtree pulls).

### What a snapshot ensemble can and cannot give you

It gives a **better mean forecast** and a **rough spread**. It **cannot** give a
*calibrated* spread: the disagreement between members reflects where the optimizer
stopped, not atmospheric uncertainty. That is inherent to how they were made and no
tooling fixes it — it is precisely why FCN3 *trains* its ensemble rather than harvesting
one. Do not present snapshot spread as an uncertainty estimate without that caveat.

⚠ Whether the late members are redundant is **unmeasured**. Members 183-243 differ by
≤4e-5 in loss, but similar loss does **not** imply correlated errors — an earlier claim
that they were redundant was retracted for exactly that reason. If you want diversity,
include the early cycles (23, 43, 63).

---

## 4. If C1 underdelivers — the CRPS route, and what it really costs

FCN3's pretrain-2 is **`ensemble 2 × batch 32 × h2w4`** on 512 A100 and runs makani's
**stochastic** trainer (`makani_bench_report.md:278, :938-939`). It fixes *two* things:
exposure bias (via rollout) **and** blurring (via ensemble + CRPS). **Our C1 does only the
first.** If the ladder shows the blurring signature, this is the gap.

**No retraining from scratch is needed.** Production is pretrain-1 and is banked; any
later stage warm-starts from it — the knobs already exist (`PRETRAINED`,
`PRETRAINED_CKPT`, `LOAD_OPTIMIZER/SCHEDULER/COUNTERS/LOSS`, `OVERRIDE_LR`, `MULTISTEP`).

Three real obstacles, in order:

1. **The fork extends the wrong trainer.** `plasim_trainer.py:63,296` —
   `class PlasimTrainer(Trainer)` where `Trainer` is the **deterministic** one. The four
   patches that make our 107→101 contract and forcing feedback work do not exist on
   `ensemble_trainer.py`. Porting them is the main engineering cost — moderate, not a
   config flip.
2. **Memory.** `ensemble_size 2` on top of `n_future 1` is ≈ **42 gibibytes against a
   39.49 gibibyte card** (from the measured fit). Needs ≤2 samples per GPU; probe first.
3. **It changes what the model computes** ⇒ jesswan's sign-off (CLAUDE.md, division of
   labour). Not a config decision.

Also: FCN3's `h2w4` decomposition is **unusable here** — we measured that `w=4` fails
outright on our grid (§5b), and spatial parallelism costs 40-80 % throughput.

---

## 5. Retired claims — do not resurrect these

Each was believed, then refuted by measurement.

| retired claim | what is true |
|---|---|
| "β₂ 0.95 → 0.999 will help" (was handoff recommendation #3) | **Backwards.** 5 of 5 arms at 0.999 collapsed at epoch 2, the fastest failure of any configuration. **Keep 0.95** |
| "sharding beats pure DDP 1.71×" | a confound; overturned by 7580297 (+43.6 % tax) |
| "memory ≈ 0.99 GB per sample-step, linear" | refuted by the batch-64 out-of-memory failure |
| "memory ≈ 2.5× per doubling, superlinear" | refuted by the batch-48 measurement |
| "the memory cliff" | an artefact — the metric was never a peak |
| "1-node is 29.8 % better than 128-node" | **not converged-versus-converged** — that run stopped while still improving. The defensible claim is cost: **11.3× more samples per node-hour** |
| "the late ensemble members are redundant" | unfounded — loss similarity does not imply correlated errors |
| "makani has never been profiled" | true only of *kernel-level*; the system-level study is extensive |

---

## 6. Traps that will cost you a job each

All measured in this session, all silent.

1. **`ckpt_mp0_v0.tar` or nothing.** makani gates `resuming` on **exactly** that filename
   (`train.py:101-105`). Seed a scratch directory with any other name and the job
   **trains from scratch with no error**. Always assert `resuming True` in the log.
2. **A seeded/forked directory needs `WANDB=0`.** With wandb on *and* resuming,
   `Driver._init_wandb` reads `<expDir>/wandb/makani_restart.yaml`
   (`driver.py:237-248`); a seeded directory has none and **every rank dies at
   construction**, presenting as NCCL teardown noise and `rank 0 exited with code 1`.
   Do **not** fix it by copying that file in — the job would write into the training
   run's wandb history.
3. **`LOAD_COUNTERS=0` for any validation-only run.** The restored epoch counter (243)
   against `EPOCHS=1` makes the loop `range(243, 1)` — empty. No epoch runs, and since
   validation lives *inside* the epoch loop, **no validation runs either**: the job
   restores, logs `resuming True`, exits 0, and writes no loss. Cost 7592332/3/6.
4. **`LOAD_LOSS=0` when `n_future` changes.** `LossHandler`'s running statistics are
   shape-dependent on `n_future`; restoring them raises a size mismatch at construction
   (7590350).
5. **A config-side `n_future` does nothing.** `train.py:119` overwrites it from
   `--multistep_count`. `MULTISTEP` is the only handle.
6. **`pretrained` and `resuming` are mutually exclusive** (`deterministic_trainer.py:237`).
   A fine-tune needs a **new** `RUN_NUM`, or resuming wins silently.
7. **`qalter` is refused outright on Polaris** — every attribute, rc=32. Walltime and
   dependencies are immutable after submission; size with margin, and stagger queued
   jobs with `qhold`/`qrls`.
8. **`capacity` cannot hold a queued successor** — the cap counts running + queued
   together, so a dependent job cannot be pre-staged. Chain via `preemptable`.
9. **Never re-run a stuck job before diagnosing.** `comment` on the queued job tells you
   whether it is the queue or you (CLAUDE.md #12).

---

## 7. After the ensemble — the standing queue

1. **The DESIGN §4.1 equivalence baseline for makani (TODO item 9).** This is now the
   blocker on acting on the kernel profile: 34.9 % of compute time is data movement, but
   **no hot-path change may be adopted without a captured baseline**, and makani has
   none. Nothing else unlocks that 34.9 %.
2. **NVTX phase attribution** — name *which* copies. Harness exists at
   `ACE2_retrain/nvtx_phase_attribution.py`.
3. **Migrate `makani_scaling*.csv` to carry peak memory** (TODO item 13) — a deliberate
   header change; the parser refuses to append on drift, by design.
4. **Hand the per-channel lwrmse panels to the science owner.**

## 8. Assets

| what | where |
|---|---|
| best checkpoint | `$MEMBER_ROOT/runs/makani_mn_scaling/e3sm_mn_scaling/prod1n_b32_sgdr/training_checkpoints/best_ckpt_mp0.tar` (epoch 243, 0.01284) |
| 12 ensemble members | same directory, `ckpt_mp0_v{22,42,62,82,102,122,142,162,182,202,222,242}.tar` |
| how to load and run it | `makani_sfno/docs/2026-09-03_prod1n_b32_sgdr_checkpoint_usage.md` |
| all measurements | `makani_bench_report.md` §5g-§5m, §7c-§7e |
| kernel profile | `$MEMBER_ROOT/bench/nsys_makani_mn_nsys_prod_b32/rank_{0..3}.nsys-rep` |
| launchers | `makani_sfno/polaris/submit_{c1_rollout_finetune,rollout_scorecard,makani_nsys_profile}.sh` |
