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

> **STATUS 2026-09-02 — C3 IS RUNNING as job 7585080** (`capacity`, 1 node, 243 epochs,
> LR **2.0e-3**, warm restarts `T_0=20 T_mult=1`, ~45 h). It was launched *before* C1/C2
> because the LR sweep and resume test completed first and the slot was free. C1 (rollout
> fine-tune) and C2 (`n_future` scaling) follow when it finishes or is stopped at a cycle end.
> Measured: **665 s/epoch**, valid 0.0209 by epoch 5 — already closing on the 128-node run's
> *final* 0.018297 at ~1% of its cost.

### C1 — rollout fine-tune FIRST, from the checkpoint we already have

The reported problem is that **inference is worse than expected while training looked fine**.
The leading explanation is `n_future: 0` — the model was trained purely single-step and never on
its own output, so autoregressive error compounds. Upstream's own recipe says so: FCN3
pretrain-2 exists *"to get good autoregressive rollouts"* and is a **fine-tune from pretrain-1's
checkpoint**, not a run from scratch.

We already have a stage-1 model: `best_ckpt_mp0.tar` from 7566145. So C1 buys the most for the
least and does not wait on a retrain.

```
pretrained: true
pretrained_checkpoint_path: $MEMBER_ROOT/runs/makani_mn_scaling/e3sm_mn_scaling/
                            prod128_alldata_v2/training_checkpoints/best_ckpt_mp0.tar
load_optimizer: false        # upstream resets all three at the stage boundary
load_scheduler: false
load_counters: false
n_future: 1                  # multistep_count = n_future + 1
lr: 4.0E-4                   # upstream's pretrain-2 value, with override_lr
max_epochs: ~24              # a fine-tune, not a pretrain
```

⚠ **The stage-1 model is undertrained** (8,500 updates). C1 tests the machinery and may improve
rollout skill, but a rollout fine-tune cannot repair a base model that never converged. If C1's
scorecard is still poor, that is evidence for C3, not against rollouts.

### C2 — scale `n_future`, and mind the memory wall

Activations scale with `samples/GPU × (n_future + 1)`. From the one measurement we have
(10.33 GB at 8 samples/GPU, `n_future=0`; ~2.4 GB is weights + grads + AdamW moments):

⚠⚠ **CORRECTED TWICE — there is no working memory model. Three measured points, a CLIFF, and
nothing to extrapolate with:**

| samples/GPU | global batch | measured memory | job |
|---|---|---|---|
| 8 | 32 | **16.04 GB** | 7585080 / 7582088 |
| **12** | **48** | **18.97 GB — fits with ~20 GB spare** | **7585983** |
| 16 | 64 | **≥39.5 GB — OOM** | 7580362 |

8→12 adds **0.73 GB per sample**; 12→16 would predict ~22 GB and instead blows past 39.5.
**Both models fitted here have been refuted by the next measurement** — first "~0.99 GB per
sample-step, linear" (predicted 18.2 at 16 samples), then "≈2.5× per doubling, superlinear"
(predicted ~21 at 12; truth is 18.97, and it cannot produce a cliff at all).
⇒ **Measure one arm per configuration. Do not fit a third model.**

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
*training*-dominated peak (`EVAL_SAMPLES=8`); 16.03 GB is a *validation*-dominated peak
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
| 3 | **`optimizer_beta2`** | 0.95 → **0.999** | 0.95 is a large-batch setting (short second-moment memory). At batch 32 the gradient is 16× noisier and 0.95 amplifies it. **Probably actively wrong at the new batch** |
| 3 | **`weight_decay`** | 0.0 → **1e-5 … 1e-4** | AdamW at zero decay is just Adam — no regularisation, while taking 16× more updates |
| 4 | **`lr`** | 1e-3 → **2e-3 — MEASURED, sweep §5h** | 3 arms x 3 full epochs: 2e-3 won on both validation loss and grad norm; **4e-4, the upstream value and my prior, came last**. 3 arms × 30 epochs ≈ 41,000 updates each (5× the *entire* 128-node run) |
| 5 | **`optimizer_max_grad_norm`** | 32 → **~1.0** | never engages (observed grad norms 0.43 → 0.012, three orders below). Effectively unclipped at a noisier batch. ⚠ verify makani clips **before** `optimizer.step()` — ai-rossby had exactly that bug |
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

## 8. In flight at handoff — read these first

| job | what | why it matters |
|---|---|---|
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
