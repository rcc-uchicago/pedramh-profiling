# Pre-registration — does `n_future = 3/4` actually buy stability? An adversarial design

**Written 2026-09-10, BEFORE any arm is run.** Scope: makani only.

Goal: reach `n_future = 3` or `4` (FCN3's stage-2 depth) and know — not hope —
whether it helps. Every number in this track today is **n = 1**, including the
C1 verdict measured this morning. This document exists so that a positive result
cannot be an artifact of that.

---

## 0. The claim and its adversary

**H1 (what we want to be true):** fine-tuning at `n_future = 3` or `4` improves
**long-rollout stability** over `n_future = 1`.

**H0 (the adversary, and it is currently favoured):**
- **Measured:** `n_future` 0 -> 1 improved accuracy 3-4.7 % at every lead but
  changed *no* stability statistic — ceiling crossing 94 vs 95 steps, late/early
  slope ratio 0.774 vs 0.779 (`2026-09-10_longroll_blowup_analysis.md` §3).
- **Existence proof:** ACE2 achieves **7300-step (5-year)** stable rollouts at
  `n_forward_steps: 2`, which is exactly `n_future = 1`
  (`2026-09-10_ace2_comparison_the_corrector.md` §2).

⇒ **H0 must be the default.** The burden is on H1, and a weak positive should be
read as noise.

## 0a. 🔴 AMENDED 2026-09-10, BEFORE ANY SCIENCE ARM RAN — "3 seeds" is not implementable

**makani has no global training seed, and asking for one would be a silent
no-op.** Verified:

- Every `seed=333` in the makani tree belongs to a noise module, a DALI loader,
  or drop-path. **This fork uses none of them** — it has its own
  `_plasim_get_dataloader`.
- Our sampler is `DistributedSampler(..., shuffle=True)`
  (`plasim_trainer.py:104-110`) with **no `seed` argument**, so torch defaults to
  seed 0; and **nothing calls `set_epoch()`**, so the shuffle is identical every
  epoch.
- The PBS renderer has **no `seed` key** (`_bools` at `:451-459` plus the
  explicit key list), so `-v SEED=...` is accepted by `qsub` and then dropped.
- **Every arm restores the same pretrained checkpoint**, so there is no random
  weight initialisation to vary either.

⇒ Between identical starting weights and a deterministic data order, repeated
training runs of one configuration differ **only by GPU nondeterminism** (cuDNN
algorithm choice, atomic reduction order). "3 seeds" as originally written would
have produced three near-identical runs and a fake `sigma_0` near zero — which
would have made *any* difference look significant. **That would have inverted
the conclusion.**

**Revised replication strategy — and it is stronger, not weaker:**

| what | replicate over | why |
|---|---|---|
| noise floor | **initial conditions** (3) and **snapshot checkpoints** (3) | neither needs a seed; both are the variation that actually limits ranking |
| each `n_future` arm | trained **once**, evaluated on the **same 3 ICs** as every other arm | a *paired* comparison across ICs, which is statistically stronger than unpaired seeds |
| training nondeterminism | 2 `REPS` of one configuration | **measures** it rather than assuming it |

`REPS` in `submit_nfuture_ladder.sh` therefore sets **no seed** — it only gives
an arm a distinct `expDir`. Two REPS of one config measure GPU nondeterminism
directly.

## 1. Threats to validity — what would make us WRONGLY accept H1

| # | threat | control |
|---|---|---|
| T1 | **Seed noise exceeds the effect** | 3 seeds per arm; noise floor measured *first* (§2) |
| T2 | **IC-to-IC variation** in the stability metric | 3 ICs, held identical across arms |
| T3 | **Batch confound** — `n_future=3` forces global batch <=12, `n_future=4` <=8, against C1's 16 | **Hold global batch = 8 for every arm**, including the `n_future=1` control |
| T4 | **Update-count confound** — smaller batch = more updates for the same data | Hold *samples consumed* fixed, not epochs |
| T5 | **Proxy invalidity** — a short fine-tune may not predict a long one | **Validate the proxy against C1's known answer first** (§3) |
| T6 | **Censored metric** — if a rollout is capped at 500 steps, a surviving arm reads as ">500" and cannot be ranked | Use a **continuous** metric (§4), not divergence step |
| T7 | **Metric cherry-picking** across 101 channels | Metric and channels fixed **here, in advance** |
| T8 | **Memory model is extrapolation** above `n_future=1` and has never been tested there | Probe job before any science arm (§5) |

And what would make us wrongly **reject** H1: a proxy too short to show anything
(T5 again), or global batch 8 degrading *all* arms so much that a real
difference is buried. Both are addressed by §3.

## 2. Phase 0 — measure the noise floor FIRST. Costs no training.

**Nothing else in this document is interpretable until this is done.**

| run | what it measures |
|---|---|
| base checkpoint x **3 different ICs** | IC-to-IC spread (T2) |
| snapshot checkpoints **e203, e223, e243** x 1 fixed IC | model-to-model spread — a **free 3-"seed" population from one training run** |

Six forward-only rollouts. No training at all.

⇒ If the stability metric varies across these by more than the effect we hope
to see, **the experiment cannot work as designed** and must be redesigned before
anything is spent on fine-tuning. That verdict is itself worth the two debug
jobs.

⚠ The snapshot arm is the sharpest control available and it is free: 12
checkpoints from one run already sit on disk, so model-to-model variation can be
quantified without training a thing.

## 3. Phase 1 — validate the proxy against a KNOWN answer

The proxy is a **short fine-tune** (see §6). It is only usable if it reproduces
something we already know.

**We have exactly one known answer, measured today:** C1 (`n_future=1`, 24
epochs) is **3-4.7 % better than base on RMSE at every lead beyond the first,
crossover at lead 2** (`2026-09-10_c1_verdict.md`).

**Proxy validation run:** 3 seeds x `n_future=1`, short fine-tune, batch 8.

- ✅ **Proxy VALID** if all 3 seeds reproduce the sign and roughly the magnitude
  of C1's accuracy effect (RMSE ratio vs base below 1.0 from lead 2 onward,
  best-lead improvement in the 2-6 % band).
- ❌ **Proxy INVALID** if it does not. Then the short fine-tune tells us nothing
  about `n_future=3/4` either, and **Phase 2 must not be run** in this form.

This is the step that stops us fooling ourselves. It also delivers, as a
by-product, the **first multi-seed replication of the C1 result**, which is
currently n=1.

## 4. The metric — continuous, pre-registered, no censoring

**Do NOT use "step at which it diverges."** It is censored (T6), it is a single
noisy event, and it requires running to failure.

**Primary metric — reservoir drift rate.** From a fixed-length rollout
(56 steps), the linear trend per step of the global mean of:

- `SOILWATER_10CM` — measured at **-0.00958 / step** on the base model, the
  largest drift of any sampled channel (`2026-09-10_rollout_spectra_and_drift.md` §2)
- `TSOI_10CM`, `TMQ`, `PRECT`

Continuous, low-variance, cheap, and **already implemented**
(`$CLAUDE_JOB_DIR/tmp/drift.py`).

**Secondary metric — spectral slope.** Trend of the small/large power ratio,
same rollout. Base: `TREFHT` **x0.50 over 14 days** (blurring).

**Tertiary — per-lead RMSE at fixed depth**, the accuracy axis, via the existing
scorecard path. This is what Phase 1 validates against.

⚠ **Channels and metrics are fixed here.** No post-hoc substitution.

⚠ **The four channels sampled so far include none that are dynamically active.**
Every rollout in this design must additionally sample **`U10`** — the
fastest-degrading *fed-back* channel by measurement (crossing at ~36 steps) —
plus one level temperature and one level humidity.

## 5. Feasibility — and the memory model is extrapolating

From the measured model `GiB ~= 8 + 2.31 + 2.12 x (n_future+1) x samples/GPU`
against a 39.49 GiB card:

| `n_future` | max samples/GPU | max global batch | predicted GiB | full epoch |
|---|---|---|---|---|
| 1 | 6 | 24 | 35.75 | 20 min |
| 2 | 4 | 16 | 35.75 | ~30 min |
| **3** | **3** | **12** | 35.75 | **40 min** |
| **4** | **2** | **8** | 31.51 | **50 min** |

Model validation: `n_future=0` at 8 and 12 samples/GPU predicted within
**0.03 GiB**; the `n_future=0` s=16 OOM was retrodicted. But at `n_future=1` it
over-predicts by **1.06 GiB**, and **it has never been evaluated above
`n_future=1`.**

⇒ **Phase -1: a 1-epoch memory probe at `n_future=3` and `n_future=4`** before
any science arm. Cheap, and it is the difference between a clean ladder and six
OOM'd jobs. Same discipline that caught the batch-64 OOM.

## 6. The proxy, and why this size

C1's effect was visible **in epoch 1** (validation 0.013117 at e1 against base
0.012839) and had essentially converged by **e3** (0.013048 vs e24's 0.013038).

⇒ **Proxy = one full epoch of samples**, held constant across arms as a **fixed
sample count** (T4), not as "1 epoch."

At global batch 8 this is ~40 min at `n_future=3` and ~50 min at `n_future=4`,
plus ~5 min startup. `n_future=3` fits the 1-hour `debug` limit;
**`n_future=4` does not, and needs a 2-hour `preemptable` slot** — which I will
surface before submitting, per standing policy.

## 7. Decision rules — written before the data

Let `sigma_0` = the larger of the IC-to-IC and model-to-model spreads of the
primary metric, from Phase 0.

| outcome | rule |
|---|---|
| **H1 supported** | `n_future=3` improves the drift rate over the `n_future=1` control by **> 2 sigma_0**, with **all 3 seeds agreeing in sign** |
| **Ambiguous** | improvement between 1 and 2 `sigma_0`, or seeds disagree in sign -> report as null, do not escalate to `n_future=4` |
| **H0 confirmed** | improvement < 1 `sigma_0` -> **stop**, `n_future` is not the stability lever, and the corrector/normalization route takes priority |

**Escalation gate:** `n_future=4` is run **only** if `n_future=3` clears the H1
bar. Cheapest possible refutation first.

**Pre-committed:** a positive accuracy result with a null stability result is
**H0, not H1.** That is exactly what C1 produced, and it must not be re-read as
success a second time.

## 8. Cost

| phase | jobs | queue |
|---|---|---|
| -1 memory probe (`n_future` 3, 4) | 2 | `debug` |
| 0 noise floor (3 IC + 3 snapshots) | 2 | `debug` |
| 1 proxy validation (`n_future=1` x 3 seeds) | 3 | `debug` |
| 2a ladder (`n_future=3` x 3 seeds) | 3 | `debug` |
| 2b ladder (`n_future=4` x 3 seeds) — **gated** | 3 | `preemptable` (2 h) |

~13 jobs, all but 2b on `debug` (1 node, <=1 h — pre-authorized). `max_run=1`
serializes them, so ~6-8 h wall-clock for phases -1 through 2a with no decision
cost.

⚠ Phase 2b needs `preemptable` and will be surfaced before submission.

## 9. What would falsify this whole design

If Phase 0 shows the drift rate varies as much across ICs or snapshot
checkpoints as the effect we are chasing, **the metric is too noisy and the
design fails** — at which point the honest move is to say so and either add
replicates or abandon the `n_future` question in favour of the corrector route,
which has an existence proof behind it (§0) and needs no training at all.

That outcome is a success of the design, not a failure of it.
