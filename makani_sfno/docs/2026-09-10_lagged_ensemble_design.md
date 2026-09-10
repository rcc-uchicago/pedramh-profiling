# Lagged ensemble vs. many-IC evaluation — which one we are building, and why

**Decision, 2026-09-10: we are building the lagged ensemble (Option A).**
This document records what that is, what the alternative was, why they are not
interchangeable, and what each one costs.

Written in machine-learning terms rather than climate terms on request. Nothing
here is domain-specific — it is an ensembling question about an autoregressive
sequence model.

---

## 0. The model, stated as an ML object

An autoregressive dynamics model. Input `(107, 180, 360)`, output
`(101, 180, 360)`. One forward pass advances one timestep. The channels split
three ways and the split drives everything downstream:

| group | count | role |
|---|---|---|
| state | 100 | fed back autoregressively |
| diagnostic (`PRECT`) | 1 | supervised auxiliary head, **never fed back** |
| exogenous forcing | 7 | *known future covariates* — read from the dataset every step, never predicted |

So the recurrence is `x_{t+1} = f(x_t ⊕ u_t)` with `u_t` supplied externally at
every step. Verified against the run's own `config.json`:
`n_state_channels=100`, `n_diagnostic_channels=1`, `n_forcing_channels=7`,
`N_in_channels=107`, `N_out_channels=101`.

**Training used teacher forcing with a single-step loss.** Test-time use is
free-running rollout. That gap is exposure bias, and it is what the C1/C2
fine-tunes target. → `polaris_makani_1node_production_handoff.md` §C1/§C2.

---

## 1. The two options

Both run the *same* rollout code. They differ only in **which start indices are
chosen** and **what is done with the outputs**.

### Option B — many-IC evaluation (already implemented, `nwp_ic_offsets`)

Start indices are spread far apart so rollouts do not overlap:

```
test sequence, index --->
 40          280         520         760         1000        1240
 |            |           |           |            |           |
 |-K steps->  |           |           |            |           |
              |-K steps-> |           |            |           |
                          |-K steps-> |            |           |
                                      |-K steps->  |           |
                                                   |-K steps-> |

each rollout scored independently  ->  err_1(k), err_2(k), ... err_N(k)
                  average over rollouts  ->  err(k)        ONE CURVE
```

At any target index there is **exactly one prediction**. Nothing is combined.
The output is a lower-variance estimate of the model's error as a function of
rollout depth, because it averages over N cases instead of one.

### Option A — true lagged ensemble (CHOSEN)

Start indices are staggered by `d` so they all land on the same target `T`:

```
                                        target T = 1000
                                              |
   start 984 |-------- 16 steps ------------->|   member D  (k=16)
   start 988     |----- 12 steps ------------>|   member C  (k=12)
   start 992         |---- 8 steps ---------->|   member B  (k=8)
   start 996             |-- 4 steps -------->|   member A  (k=4)
                                              |
             combine:  x_hat_T = sum_k w_k * x_hat_T^(k),  w_k ~ 1/sigma(k)^2
                       spread_T = std over {A, B, C, D}
```

Now there are **M predictions of the same state**, so they can be combined into
one better prediction, and their disagreement is an uncertainty estimate. Slide
`T` forward and repeat.

"Lagged" = members are lagged in *when they were initialized* relative to `T`.

---

## 2. The structural difference, in one line

> **Option B averages across targets. Option A averages across members at a
> single target.**

| | B — many-IC eval | A — lagged ensemble |
|---|---|---|
| start indices | far apart, non-overlapping | staggered by `d`, overlapping |
| predictions per target | 1 | M (one per member) |
| what is averaged | the *errors*, across rollouts | the *predictions*, across members |
| output | one skill curve, err vs depth | per-target prediction **+ spread** |
| ML analogy | evaluating on more test examples | test-time ensembling / deep-ensemble forward pass |
| question answered | "how good is this model at depth k?" | "what is my best guess for T, and how sure am I?" |
| what it changes | what you *know about* the model | what the model *outputs* |
| status | implemented (`nwp_ic_offsets`) | **to build** |

They are not competitors — B is evaluation, A is inference. A's outputs can be
scored with B's machinery.

---

## 3. Why lagged, and not noise injection

The alternative route to an ensemble is perturbing the model or its inputs
(ACE2's `input_noise`, MC-dropout, a deep ensemble of N independently trained
models, or a distributional training objective such as CRPS).

The lagged ensemble was chosen because it is **free**: no retraining, no noise,
no second model, no probabilistic objective. The rollouts have to be run for
evaluation anyway; staggering their start indices makes the *same* rollouts
serve as ensemble members.

⚠ For contrast, the CRPS route requires porting the fork's four contract
patches onto `ensemble_trainer.py` (a different trainer class), which is the
main engineering cost recorded in
`polaris_makani_analysis_ensemble_handoff.md` §4.

---

## 4. Two properties of a lagged ensemble that must be designed around

Neither is a reason not to build it. Both change the implementation.

### 4a. Members are not exchangeable

The `k=4` member is systematically far more accurate than the `k=16` member —
their error distributions differ **by construction**, not by chance. Uniform
weighting is therefore the wrong estimator.

The fix is `w_k ~ 1/sigma(k)^2`, which needs `sigma(k)`: error as a function of
rollout depth. **That is exactly what the per-lead metric patch produces**
(commit `652e9505`; `<expDir>/scores/metrics_epNNNN.h5`, an array of shape
`(leads, channels)`). So the metric work is not a detour before the ensemble —
it generates the ensemble's weights.

### 4b. The measured spread is largely a consequence of the stagger `d`

Pick `d=1` and members are nearly identical; pick `d=20` and they are far
apart. The "uncertainty" can be dialled to almost any value by choosing `d`,
which is a sign that raw member variance is **not** a calibrated predictive
distribution.

⚠ **What this ensemble's spread actually measures is sensitivity to the initial
condition and to accumulated rollout error — not epistemic/model uncertainty.**
Every member shares one set of weights. In particular, if the model is
mode-averaging (the MSE-optimal hedge, which is why MSE-trained video models go
blurry), then every member is mode-averaged the same way: the ensemble mean
stays blurry and the spread is systematically under-dispersed. **A lagged
ensemble cannot repair a mode-averaging failure.**

Whether that failure is present is readable from the shape of `err(k)`:

| shape of err vs depth | diagnosis | consequence for this design |
|---|---|---|
| grows fast, possibly super-linearly | exposure bias — compounding is real | C1/C2 correctly aimed; lagged ensemble is meaningful |
| grows then **saturates** on a plateau | mode-averaging / blurring | rollout training will not fix it; lagged ensemble will be under-dispersed |
| barely moves | compounding is not the problem | the whole rollout/ensemble direction is misaimed |

The third row has never been excluded. Until `err(k)` exists, treat this design
as provisional.

Once `sigma(k)` is known, both 4a and 4b become checkable rather than
assumed: the weights follow from it, and the spread the ensemble *ought* to
show can be predicted and compared against the spread it *does* show. That
comparison is the calibration test.

---

## 5. What Option A needs that Option B does not

1. **A stagger-`d` offset generator.** `nwp_ic_offsets(n_samples, K, n_ic)`
   deliberately spaces starts far apart — the opposite of what is wanted here.
2. **Member alignment.** Each rollout's outputs must be re-indexed by
   *absolute target index* so members can be grouped by `T`. Rollout outputs
   are currently written per-IC with lead-relative indexing only.
3. **A combination step.** Weighted mean + spread per target, with the weights
   from §4a.
4. **`sigma(k)`**, from the per-lead metrics (already built).

Items 1-3 are new. None of them is large; the dependency on item 4 is the
reason for the ordering below.

---

## 6. Order of work

1. ✅ Per-lead metrics — commit `652e9505`, verified `PERLEAD_METRICS_OK`
   (job 7602739).
2. 🔵 Re-score base and C1 at fixed depth `K=20` — jobs **7603089** /
   **7603090**. Yields `sigma(k)` *and* the C1 verdict.
3. Read the shape of `err(k)` against §4b's table **before** building the
   ensemble. If it saturates, revisit.
4. The E3SM inference port —
   → `2026-09-10_e3sm_inference_port_scope.md`.
5. The lagged-ensemble driver (§5 items 1-3).

---

## 7. Cross-references

- Port scope and its justification → `2026-09-10_e3sm_inference_port_scope.md`
- C1/C2/C3 definitions → `polaris_makani_1node_production_handoff.md`
- Checkpoint provenance → `2026-09-03_prod1n_b32_sgdr_checkpoint_usage.md`
- Why `validation loss` is a single-step number → CHANGELOG `2026-09-10`
