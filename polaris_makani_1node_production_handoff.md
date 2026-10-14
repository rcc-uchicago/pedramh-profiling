# makani-SFNO on Polaris — 1-node production handoff

*Written 2026-09-01.* Read **`makani_bench_report.md` §5** first (the measurements this
rests on), then CHANGELOG `2026-09-01 (cont.)`. Priorities live in `TODO.md`.

**Definitions.** `$MEMBER_ROOT` = `/eagle/projects/lighthouse-uchicago/members/mehta5`.
*sample-equivalent* = one full sample through the whole model, **per GPU per step** =
`global_batch ÷ ranks`; a GPU holding 8 samples on a quarter domain carries 2. Units for every
report column: `makani_bench_report.md` §0a. PASS is always the log token, never `rc` (CLAUDE.md #14).

---

## 0. The one result that drives everything

**At a fixed global batch, more nodes make this model slower, not faster.** Measured, batch 32,
ALLDATA, warmup-free:

| nodes | GPUs | samples/GPU (per step) | step_ms | samples/s (total) |
|---|---|---|---|---|
| **1** | **4** | **8** | **365.4** | **87.6** |
| 2 | 8 | 4 | 627.1 | 51.0 |
| 4 | 16 | 2 | 409.5 | 78.1 |
| 8 | 32 | 1 | 492.7 | 64.9 |

Fixed per-step cost is ~**234 ms** the moment the fabric is touched (derived from two runs that
share a node count; ≈4.4 GB/s effective busbw, matching this stack's independently measured
4-6 GB/s). One node never pays it. The 128-node production run scaled *throughput* fine
(888 samples/s) but bought it with batch 512, which is why it took only **8,500 optimizer
updates** in 100 epochs and finished undertrained — validation minimum sitting at the last epoch.

⇒ **Production belongs on one node.** Everything below follows from that.

## 1. The locked configuration

```
1 node · 4× A100-40GB · pure DDP (h1w1) · global batch 32 · LOCAL_BATCH=8
GPU_ORDER=default        # reverse is +0.88% SLOWER here; node-matched pair confirms it
ALLDATA 101-ch pack      # $MEMBER_ROOT/data/e3sm_makani_alldata_production
plugin v1.21.1 + OFI_NCCL_PROGRESS_MODEL=AUTO + NCCL_PROTO=Simple
365.4 ms/step (3 reps, 0.2% spread)   10.33 GB / 40 GB   87.6 samples/s
```

⚠ **`OFI_NCCL_PROGRESS_MODEL=AUTO` lives in no script.** Omit it and the job dies with ENOSYS.
`polaris/submit_when_slot_frees.sh` bakes it in; a hand-written `qsub` must not forget it.


> ⚠ **Authoritative source: `polaris_pbs_notes.md` §1b.** This copy is a convenience and
> drifts — when the two disagree, **the notes win**. (2026-09-02: a stale `preemptable`
> claim had to be corrected in five files at once; that is what this line exists to stop.)

## 2. Queue reality

A ~16-24 h single-node job has exactly one home:

| queue | nodes | walltime | verdict |
|---|---|---|---|
| `debug` | ≤2 | 1 h | tests only |
| `prod`→small/medium/large | ≥16 | 6-24 h | won't take 1 node |
| `preemptable` | 1-10 | ≤72 h | **10 concurrent/project**, but start latency is load-dependent; preemption now costs ~1 epoch since resume is proven, though our launchers are `-r n` |
| **`capacity`** | **1-4** | **≤168 h** | **the only option — fits unchained** |

⚠ `capacity` is **`max_run 1` per PROJECT**. Taking it blocks every other
`lighthouse-uchicago` member for the duration, and cancelling to hand it back destroys accrued
`eligible_time`. **Coordinate before submitting, and run the stages below sequentially** — they
cannot overlap.

## 3. Capacity sequence

> ### ✅ STATUS 2026-09-04 — C3 IS COMPLETE, C1 IS RUNNING
> **C3 (job 7585080) finished all 243 epochs**, `Exit_status 0`, 46 hours 20 minutes of a
> 48-hour allocation. Best validation loss **0.01284** at epoch 243 — **29.8 percent below**
> the 128-node run's 0.018297, at **46.3 node-hours against 216**. Twelve snapshot-ensemble
> members on disk. Measured rate 683.6 seconds per epoch (5.27 epochs per hour).
> → `makani_bench_report.md` §5k, and `makani_sfno/docs/2026-09-03_prod1n_b32_sgdr_checkpoint_usage.md`.
>
> **C1 (job 7591605) is running on `capacity`**, 24 epochs, warm-started from that checkpoint.
> Its machinery was proven first by a one-epoch probe (job 7590355): pretrained path taken,
> `n_future` 1 applied, **26.21 of 39.49 gibibytes** peak memory at 4 samples per GPU.
>
> ⚠ C3 was run *before* C1/C2 because the slot was free; the ordering in this section is the
> original plan, not the order executed.

### C1 — rollout fine-tune FIRST, from the checkpoint we already have

The reported problem is that **inference is worse than expected while training looked fine**.
The leading explanation is `n_future: 0` — the model was trained purely single-step and never on
its own output, so autoregressive error compounds. Upstream's own recipe says so: FCN3
pretrain-2 exists *"to get good autoregressive rollouts"* and is a **fine-tune from pretrain-1's
checkpoint**, not a run from scratch.

**Launched 2026-09-04 as job 7591605** via
`makani_sfno/polaris/submit_c1_rollout_finetune.sh full`, which encodes everything below. Do
not hand-write the `qsub`.

**Warm-starts from `prod1n_b32_sgdr`, NOT prod128.** This document originally named
prod128's checkpoint and warned that *"a rollout fine-tune cannot repair a base model that
never converged"* — prod128 had 8,500 weight updates. C3 has **332,424** and finished with
flattened cycle minima, so that caveat no longer applies and C1 is far better positioned than
when this was written.

```
pretrained: true
pretrained_checkpoint_path: $MEMBER_ROOT/runs/makani_mn_scaling/e3sm_mn_scaling/
                            prod1n_b32_sgdr/training_checkpoints/best_ckpt_mp0.tar
load_optimizer: false        # upstream resets these at the stage boundary
load_scheduler: false
load_counters: false
load_loss:      false        # REQUIRED -- see trap 1 below
override_lr: true
lr: 4.0E-4                   # upstream's pretrain-2 value
max_epochs: 24               # a fine-tune, not a pretrain
LOCAL_BATCH: 4               # 4 samples per GPU; 8 would need ~44 gibibytes and will not fit
MULTISTEP: 2                 # => n_future 1. The ONLY place n_future can be set -- trap 2
```

**Three traps, all measured, all silent if violated:**

1. **`load_loss: false` is required.** `LossHandler` carries running statistics whose *shape*
   depends on `n_future`; the checkpoint's were accumulated at `n_future` 0, so restoring them
   into an `n_future` 1 model raises a size-mismatch `RuntimeError` at trainer construction
   (job 7590350 died there). Do **not** work around it with `strict_restore: false` — that
   would silently skip real mismatches in the model weights too.
2. **A config-side `n_future` does nothing.** `makani/train.py:119` sets
   `params["n_future"] = args.multistep_count - 1`, overwriting the config. Only
   `--multistep_count` has effect.
3. **`pretrained` and `resuming` are mutually exclusive**
   (`deterministic_trainer.py:237` gates on `pretrained and not resuming`), so C1 needs a
   **new** `RUN_NUM`. If the target experiment directory already holds checkpoints, resuming
   wins and `pretrained_checkpoint_path` is silently ignored.

**Verified by a one-epoch probe first (job 7590355, `debug`):** pretrained path taken,
`resuming False`, `multistep_count 2` / `n_future 1` applied, and **26.21 of 39.49 gibibytes**
peak memory (18.21 PyTorch + 8.00 non-PyTorch) at 4 samples per GPU — 66 percent of the card.
Rate 1053.8 seconds per epoch, so 24 epochs is about **7 hours**.

### C2 — scale `n_future`, and mind the memory wall

Activations scale with `samples/GPU × (n_future + 1)`. From the one measurement we have
(10.33 GB at 8 samples/GPU, `n_future=0`; ~2.4 GB is weights + grads + AdamW moments):

✅ **RESOLVED 2026-09-04 — there is now a working memory model, measured with real peaks.**
Units: gibibytes per GPU. The card is 39.49 gibibytes.

| configuration | samples per GPU | global batch size (samples) | peak PyTorch memory (gibibytes) | non-PyTorch memory (gibibytes) | total (gibibytes) | fraction of card | job |
|---|---|---|---|---|---|---|---|
| single-step (`n_future` 0) | 8 | 32 | 19.23 | 8.01 | **27.24** | 69 percent | 7588118 |
| single-step (`n_future` 0) | 12 | 48 | 27.69 | 8.04 | **35.73** | 90 percent | 7588120 |
| single-step (`n_future` 0) | 16 | 64 | not measured | not measured | **~44 (predicted)** | **out of memory** | 7580362 |
| **rollout (`n_future` 1)** | **4** | **16** | **18.21** | **8.00** | **26.21** | **66 percent** | **7590355** |

**peak PyTorch memory (gibibytes) ≈ 2.31 + 2.12 × (samples per GPU)** at `n_future` 0, plus a
fixed ~8 gibibyte non-PyTorch overhead. It **retrodicts the batch-64 failure** (16 samples per
GPU → 44.2 gibibytes), which is what the three earlier models could not do.

`n_future` 1 roughly **doubles the per-sample term**: predicted 19.3 against 18.21 measured.
⇒ **sizing rule for C2:** total gibibytes ≈ 8 + (2.31 + 2.12 × (`n_future` + 1) × samples per
GPU), and it must stay under 39.49. At 4 samples per GPU that allows `n_future` up to about 3;
at 8 samples per GPU even `n_future` 1 does not fit.

⚠ Every "memory" number *elsewhere* in this document predates the instrumentation and is an
epoch-end snapshot — a **lower bound**, understating by 11-16 gibibytes. Size from the table
above only.

⚠⚠ **`memory footprint [GB]` is `total − free` sampled at EPOCH END**
(`deterministic_trainer.py:703`), after the step's transients are freed. The real high-water
mark (`max_memory_allocated`) is computed on the same line and **discarded**. The batch-64 OOM
(7580362) died in the **loss** (`makani/utils/loss.py:402`) at the transient peak, with
**276 MiB** reserved-but-unallocated (so *not* fragmentation) and **~7.7 GB of non-PyTorch
overhead** (CUDA context + cuFFT/cuDNN/NCCL workspaces) inside the total.
⇒ **Log the peak before sizing anything. DONE 2026-09-02** — `plasim_trainer.log_epoch` now
emits `peak torch memory [GB]` (`max_memory_reserved`, the part that scales) and
`non-torch memory [GB]` (the ~7.7 GB fixed tax), with `reset_peak_memory_stats` per epoch.
Compare their **sum** against **39.49 GiB**. → `makani_bench_report.md` §5g.

**C2 can now be sized from measurement rather than guessed.** Use the table and rule above.
Values are in each job's `.o` log and in wandb, **not** in `makani_scaling*.csv` — the parser's
header guard is deliberately not migrated (TODO item 13).

The table below is retained only to show what was predicted and how it failed:

**≈ 0.99 GB per sample-step (REFUTED — see above)**, giving `samples/GPU × (n_future+1) ≲ 34`.

| `n_future` | samples/GPU at batch 32, 1 node | est. memory | fits 1 node? |
|---|---|---|---|
| 0 | 8 | 10.3 GB (measured) | ✅ |
| 1 | 8 | ~18 GB | ✅ |
| 2 | 8 | ~26 GB | ✅ |
| 3 | 8 | ~34 GB | ⚠ at the edge |
| **4** (upstream's value) | 8 | **~42 GB** | ❌ **OOM** — needs ≤4 samples/GPU ⇒ 8 GPUs ⇒ **2 nodes** |

⚠ **This is where the 1-node result collides with upstream's recipe.** `n_future=4` forces 2
nodes, which is the *worst* configuration on this machine (§0). Options, in preference order:
(a) stop at `n_future=2-3` on one node; (b) drop the batch to 16 and keep `n_future=4` on one
node; (c) accept 2 nodes and its ~234 ms toll. **Measure (a) before assuming (c).**
⚠ **Every row of this table is untrustworthy** — it is built on the first refuted model.
`n_future=1` doubles the retained graph, and the one doubling we have measured (8→16
samples/GPU) went from 16.04 GB to an OOM. **So `n_future=1` at 8 samples/GPU may not fit
at all.** C2 should open with `n_future=1` at a REDUCED `LOCAL_BATCH` (4 or 6) and measure,
rather than assuming the production batch survives.
⚠ **Do not substitute the 16.03 GB production figure into this table.** 10.33 GB is a
*training*-dominated snapshot (`EVAL_SAMPLES=8`); 16.03 GB is a *validation*-dominated snapshot
(`EVAL_SAMPLES=512`, 3-step rollout). `n_future` scales the training term only, so the rollout
arithmetic uses 10.33 — but the card must also clear the fixed ~16 GB validation ceiling, which
becomes non-binding once training exceeds it at `n_future ≥ 1`.

### C3 — the batch-32 retrain, with the hyperparameter fixes

Only after C1/C2 have said whether rollouts fix the symptom. This is the long one.

```
global batch 32, LOCAL_BATCH=8, 1 node
scheduler: CosineAnnealingWarmRestarts     # T_0/T_mult/min_lr all plumbed
scheduler_T_0: 20        # EPOCHS (see §5 trap 1)
scheduler_T_mult: 1      # equal cycles ⇒ equal-quality snapshots
scheduler_min_lr: 1.0E-6
lr_warmup_steps: 3       # EPOCHS, not steps
CKPT_VERSIONS: 200       # ~253 GB — REQUIRED, see §5 trap 2
```
Budget **143 epochs = 3 warmup + 7 × 20** ⇒ 195,624 updates (94% of upstream pretrain-1's
208,320), ~23 h, 7 snapshots at epochs 23/43/63/83/103/123/143.

## 4. Hyperparameter recommendations, ranked by expected value

From an adversarial review of the shipped config. **LR is fourth, not first.**

| # | change | from → to | rationale |
|---|---|---|---|
| 1 | **`n_future`** | 0 → 1-3 | C1/C2. Targets the actual symptom |
| 2 | **use the idle 74% of the GPU** | 10.33 / 40 GB | `scale_factor: 3 → 2` (trunk 60×120 → 90×180, stop discarding resolution before the spectral layers) **or** `embed_dim 384→512` / `num_layers 8→12`. Capacity usually beats schedule tuning |
| ~~3~~ | 🔴 **`optimizer_beta2` — RETIRED 2026-09-04, DO NOT APPLY** | ~~0.95 → 0.999~~ | **Measured backwards.** 5 of 5 arms at β₂ 0.999 collapsed at epoch 2 — the fastest failure of any configuration — with gradient excursions up to 7.86e12 against 8.78e7 at β₂ 0.95 (§7e). The steady-state-noise argument below is real but does not govern: for a *spike*, an outlier gradient enters Adam's second-moment accumulator with weight 1 − β₂, which is 0.05 at 0.95 against 0.001 at 0.999, so the long-memory setting barely reacts and damps the next step ~50× less. **Keep β₂ at 0.95.** Original rationale, kept only to show what was refuted: 0.95 is a large-batch setting (short second-moment memory). At batch 32 the gradient is 16× noisier and 0.95 amplifies it. **Probably actively wrong at the new batch** |
| 3 | **`weight_decay`** | 0.0 → **1e-5 … 1e-4** | AdamW at zero decay is just Adam — no regularisation, while taking 16× more updates |
| 4 | ✅ **`lr`** — **CONFIRMED and bounded** | 1e-3 → **2.0e-3** | Swept (§5h) *and* bounded above (§5j/§7e): the ceiling is **(2e-3, 3e-3] at both batch 32 and batch 48**, and 3.0e-3 collapses irreversibly. 2.0e-3 is one rung below a hard limit — do not raise it. 3 arms x 3 full epochs: 2e-3 won on both validation loss and grad norm; **4e-4, the upstream value and my prior, came last**. 3 arms × 30 epochs ≈ 41,000 updates each (5× the *entire* 128-node run) |
| 5 | ✅ **`optimizer_max_grad_norm`** — **CONFIRMED, apply it** | 32 → **1.0** | Measured (§7e): a clip of 1.0 delayed collapse from epoch 2 to 6 at batch 32 and from 4 to 6 at batch 48, and produced the best loss of each batch. It does **not** prevent collapse — nothing tested does. makani's own default is 1.0 (`train.py:74-75`). Across 243 production epochs the maximum gradient norm was **0.30**, so a clip of 32 never engaged: never engages (observed grad norms 0.43 → 0.012, three orders below). Effectively unclipped at a noisier batch. ⚠ verify makani clips **before** `optimizer.step()` — ai-rossby had exactly that bug |
| 6 | **stale schedule keys** | delete | `scheduler_T_max/factor/patience/step_size/gamma` are dead under warm restarts and will mislead the next reader |
| 7 | **EMA** | leave **off** | measured in this codebase: **+0.16%** at convergence, **0.9-10.5% worse** mid-descent, and it doubles validation cost. `docs/2026-05-12_v11_clip_restore_plan.md` |

## 5. Traps — all verified in the source, all silent if violated

1. **`scheduler_T_0` and `lr_warmup_steps` are in EPOCHS, not steps.** makani steps the
   scheduler once per epoch (`deterministic_trainer.py:377-380`). `lr_warmup_steps: 500` would
   request 500 epochs of warmup.
2. **`checkpoint_num_versions` was hardcoded to 3.** Snapshot ensemble members are the cycle
   *ends*, `T_0` epochs apart; a 3-deep rolling window deletes every one. Use `-v CKPT_VERSIONS=`.
3. **An unrecognised `SCHED` string silently disables the scheduler.** `driver.py` falls through
   to `else: scheduler = None` — a typo gives a **constant LR for the whole run, with no error**.
   Echo the rendered `scheduler:` line and eyeball it before a long job.
4. **Warmup is illegal under `ReduceLROnPlateau`** (`driver.py:702` raises). The scheduler switch
   is what unlocks it; the batch-512 run had none and could not have had any.
5. **`SequentialLR` + `CosineAnnealingWarmRestarts` is an awkward torch pairing** and is
   **unproven here** — job **7582170** tests it. Do not launch C3 before reading it.
6. **Resume is untested for this schedule.** `SequentialLR.state_dict()` nesting on a requeue
   could silently restart the cycle. A 23 h `capacity` job fits unchained, but a node failure
   would exercise it.
7. **makani's `--batch_size` is GLOBAL**, split across the data group. `GLOBAL_BATCH =
   LOCAL_BATCH × NRANKS/(HPAR·WPAR)`.

## 6. Corrections carried in — do not re-derive these wrong

| claim made earlier | corrected by | truth |
|---|---|---|
| "sharding beats pure DDP 1.71× per GPU" | 7580297 | **a 40-44% tax**; the 4-node win was 2× the per-GPU work, not the sharding |
| "the sharding tax is a fabric effect" | 7580404 | **+80.8% at 1 node with no fabric at all** — it's the decomposition |
| "`GPU_ORDER=reverse` is a free 7%" | 3+3 reps | **+0.88% worse** at 1 node; node-matched pair (7580482/7580507) confirms. Config-dependent, not free |
| "`h=4` on a 60-row trunk is the problem" | 7580122 | `h4w1` **trains**. The broken axis is **`w=4`** |
| node-hours quoted as 13.9 | this doc | **16.3** — the column excluded the per-epoch overhead the wall-clock column included. At 1 node they must be equal |
| "makani has no EMA" | `src/sfno_training/trainer/ema.py` | the **fork has it**; it defaults off. Upstream lacks it |

## 7. Assets

| what | where | PASS |
|---|---|---|
| launcher | `makani_sfno/polaris/polaris_makani_multinode_scaling.pbs` | `MAKANI_MN_SCALING_OK` + a CSV row **with a timing** |
| schedule knobs | `-v LR / SCHED / SCHED_T0 / SCHED_TMULT / SCHED_MIN_LR / WARMUP_EPOCHS / LR_START / CKPT_VERSIONS` | rendered config echoed at startup |
| queue helper | `polaris/submit_when_slot_frees.sh <tag> "<vars>" [select] [walltime] [csv] [queue]` | `ARM_QUEUED` |
| results | `$MEMBER_ROOT/bench/makani_{occupancy,spatial,production,multinode_scaling}.csv` | — |
| evidence | `makani_bench_report.md` §5 | — |
| eval chain (needs porting) | `makani_sfno/scripts/` + `submit_eval.sh` | SLURM/Stampede3/PLASIM — **port to PBS/Polaris/101-ch E3SM** |

## 8. Jobs referenced by this document — all complete unless noted

| job | what | outcome |
|---|---|---|
| **7585080** | C3, the 1-node production run, 243 epochs | ✅ complete, `Exit_status 0`, best validation loss **0.01284** |
| **7591605** | C1, rollout fine-tune, 24 epochs | 🔵 running on `capacity` |
| **7590355** | C1 one-epoch probe | ✅ complete — machinery verified, 26.21 gibibytes peak |
| **7587738-42, 7587776-79** | the 9-arm learning-rate-ceiling factorial | ✅ complete — **all collapsed** (§7e) |
| **7588118 / 7588120** | batch 32 vs 48 fork A/B from epoch 74 | ✅ complete — batch 32 retained (§5l) |
| **7582088** | 3 full-pass epochs at production settings, 1 node | gives the **measured** per-epoch wall, replacing the assumed +17% overhead. Every wall-clock and node-hour figure here depends on it |
| **7582170** | 8 short epochs, `T_0=2`, warm restarts + warmup | proves the schedule constructs and restarts fire (trap 5), and that `CKPT_VERSIONS` retains the snapshots. **Read before C3** |

## 9. Remaining items, after C1-C3

1. **Port the eval chain** to PBS/Polaris/101-channel E3SM — nothing has scored any makani
   checkpoint yet (`TODO.md` P0-3).
2. **File the ALCF ticket** — ENOSYS root cause + the `OFI_NCCL_PROGRESS_MODEL=AUTO` fix, the
   tree all-reduce **silent corruption** above ~1 GB, `module load conda` broken since
   2026-08-20, three zombie-GPU nodes.
3. **`w=4` defect** — ranks `% 4 == 3` diverge into a barrier while peers broadcast
   (`makani_bench_report.md` §5b). Deprioritised: we no longer need sharding.
4. **Revive the seven dead launchers** (report §9) — both E3SM packers among them.
5. **Snapshot / multi-seed ensemble** from C3's cycle checkpoints.
6. **`omp_threads=64`** on every row to date — 8× CPU oversubscription on the cores the progress
   engine uses. Comparability intact, absolute numbers not clean.
