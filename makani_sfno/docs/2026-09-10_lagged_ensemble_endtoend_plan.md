# Lagged ensemble — end-to-end plan: data loading, training, inference, uncertainty

Written 2026-09-10. Companion to `2026-09-10_lagged_ensemble_design.md` (what
the ensemble is and why it was chosen over many-IC evaluation) and
`2026-09-10_e3sm_inference_port_scope.md` (what blocks inference today).

This document answers one question end to end: **what has to exist for the
lagged ensemble to work, and what does each stage actually do?**

---

## 0. State at time of writing

| piece | state |
|---|---|
| Trained checkpoint | ✅ `prod1n_b32_sgdr/training_checkpoints/best_ckpt_mp0.tar`, 243 epochs |
| C1 rollout fine-tune | ✅ complete (7593272); per-lead comparison pending (7603119) |
| Per-lead metrics on our channels | ✅ commit `652e9505`, verified job 7602739 |
| First error-vs-depth curve | ✅ job 7603089 — RMSE grows 4.65x from 6 h to 126 h, **linearly, no saturation** |
| Inference on E3SM | ❌ blocked — port scope doc |
| Lagged ensemble | ❌ to build — this document |

---

## 1. TRAINING — nothing to do, and that is the point

**The lagged ensemble requires no training of any kind.** The checkpoint is
frozen. There is no second model, no noise injection, no distributional
objective, no fine-tune. That is precisely why this route was chosen over the
CRPS/`ensemble_trainer` route, which would require porting the fork's contract
patches onto a different trainer class.

For the record, what training already produced:

- Objective: teacher-forced single-step MSE-family loss, `n_future=0`
- 243 epochs, 332,424 optimizer updates, batch 32 on 1 node, 46.3 node-hours
- `CosineAnnealingWarmRestarts`, peak LR 2.0e-3, beta2 0.95
- 12 snapshot checkpoints at cycle ends (epochs 23…243 every 20)

**Training only re-enters under one condition:** if the error-vs-depth curve
turns out to have the mode-averaging signature (§4.5), in which case no amount
of ensembling deterministic rollouts helps and the fix is a distributional
objective. Read the curve before spending anything on training.

⚠ Note the snapshot checkpoints make a **second, independent** ensemble
possible (a snapshot ensemble — different weights, same data). That one *does*
sample model uncertainty, which the lagged ensemble does not (§4.5). The two
are complementary and can be combined multiplicatively (M lagged × N snapshot).
Not in scope here, but the checkpoints are on disk, so the option stays open.

---

## 2. DATA LOADING — how a rollout is actually fed

This is where the exogenous covariates enter, and it is the reason stock
makani's inference path cannot be used.

### 2.1 What one dataset sample contains

`PlasimForcingDataset.__getitem__` returns **four** tensors. With
`n_history=0` and `n_future = K-1` (set from `valid_autoreg_steps`):

| tensor | shape | normalization |
|---|---|---|
| `inp_state` | `(1, 100, 180, 360)` | z-scored by `global_means/stds` |
| `tar` | `(K, 101, 180, 360)` | z-scored — state ‖ diagnostic |
| `inp_forcing` | `(1, 7, 180, 360)` | z-scored by `forcing_global_means/stds` |
| `tar_forcing` | `(K, 7, 180, 360)` | same |

⚠ **Memory scales linearly in K.** One frame of `(101, 180, 360)` float32 is
**26.2 MB**, so a K=56 rollout carries **1.47 GB of target frames per sample**
before prefetch. Use `batch_size=1` for long rollouts. This is the same trap
`submit_rollout_scorecard.sh` documents at its head.

### 2.2 The rollout recurrence

```
step 0:  inpt = inp_state[0]                       (1, 100, H, W)   state only
         forcing for this step is CACHED in the preprocessor
         pred = wrapper(inpt)                      (1, 101, H, W)   100 state + PRECT
                └─ wrapper concatenates the cached 7 forcing channels internally
                   => the model sees 107 channels

step k:  inpt = append_history(inpt, pred, k-1)    (1, 100, H, W)
                └─ drops PRECT (101 -> 100)
                └─ splices tar_forcing[k] as the new exogenous block
         pred = wrapper(inpt)
```

Two properties that matter:

1. **The diagnostic channel is dropped before feedback.** 101 predicted, 100
   fed back. `PRECT` is an auxiliary head.
2. **The exogenous covariates come from the DATA at every step, never from the
   model.** So a rollout is only defined over a time range where the covariates
   exist. For our purposes that is fine — we roll over the test split, where
   they do.

Verified against `rollout_driver.py:200-245` (the shape asserts at `:203`,
`:225`, `:239` pin exactly this contract).

---

## 3. INFERENCE — the five stages

### 3.1 The key efficiency insight

**One rollout contributes a member to many targets.** A rollout from start `s`
of length `L` produces predictions for absolute indices `s+1 … s+L`, at depths
`1 … L`. So for any target `T`, every rollout with `s < T <= s+L` contributes
the member at depth `k = T - s`.

⇒ We do **not** run separate rollouts per target. We run **one sweep** of
rollouts from starts spaced `d` apart, then regroup the outputs by absolute
target index.

```
starts spaced d apart, each of length L:

  s0     |----------------- L -----------------|
  s0+d        |----------------- L -----------------|
  s0+2d            |----------------- L -----------------|
  s0+3d                 |----------------- L -----------------|
                                    ^
                                 target T  <- receives one member from each
                                              rollout that spans it, at depths
                                              T-s0, T-s0-d, T-s0-2d, ...

members per interior target  M = floor(L / d)
```

With `L=56` and `d=4`: **14 members per target**, depths 4, 8, …, 56 steps
(1 to 14 days). The sweep is the same set of rollouts an evaluation would run
anyway.

### 3.2 The stages

| # | stage | what it does | new? |
|---|---|---|---|
| 1 | **Start-index generation** | Emit starts spaced `d` apart over the test range. `nwp_ic_offsets` deliberately does the opposite (far-apart starts), so this is a new generator | **new** |
| 2 | **Rollout sweep** | For each start, K-step autoregressive rollout, de-normalized to physical units, written to NetCDF | exists (needs port) |
| 3 | **Member alignment** | Re-index every rollout's outputs by **absolute target index** `s+k`; build the table `(target, depth) -> prediction`; group by target | **new** |
| 4 | **Combination** | Per target, per channel, per grid cell: weighted mean over members, `w_k ∝ 1/σ(k)²`, plus the weighted spread | **new** |
| 5 | **Uncertainty evaluation** | §4 | **new driver, existing metrics** |

Stage 2 is the port (A–D + F in the port scope doc). Stages 1, 3, 4 are new and
small. Stage 5 reuses code we already have — see next.

### 3.3 Where the weights come from

`w_k ∝ 1/σ(k)²` needs `σ(k)`: error as a function of rollout depth, per
channel. That is exactly the array the per-lead metric patch writes to
`<expDir>/scores/metrics_epNNNN.h5`, shape `(leads, channels)`. Already
measured for the base checkpoint (job 7603089).

⚠ Uniform weighting is **measurably wrong** here: across 101 channels the
median RMSE at depth 21 is **4.65×** the depth-1 value, so a naive mean gives a
14-day-old member the same say as a 1-day-old one.

---

## 4. UNCERTAINTY EVALUATION — reuse, do not rewrite

### 4.1 makani already ships every metric we need

`makani/utils/metrics/functions.py` contains, verified:

| class | line | what it gives us |
|---|---|---|
| `GeometricSpread` | 221 | ensemble spread |
| `GeometricSSR` | 320 | spread-skill ratio |
| `GeometricCRPS` | 433 | continuous ranked probability score |
| `GeometricRankHistogram` | 532 | Talagrand rank histogram |

All four are already wired into `MetricsHandler` through `spread_var_names`,
`ssr_var_names`, `crps_var_names`, `rh_var_names`. The **deterministic** trainer
sets all of them to `[]` (`deterministic_trainer.py:173-175`) for one reason:
it has no ensemble dimension.

### 4.2 And the handler already accepts an ensemble axis

`MetricsHandler.update` (`metric.py:626-640`):

```python
if prediction.dim() == 5:                       # (B, E, C, H, W)
    prediction_mean = torch.mean(prediction, dim=1)
else:
    prediction_mean = prediction
    prediction = prediction.unsqueeze(1)

for handle in self.metric_handles:
    if handle.type == LossType.Deterministic:
        handle.update(prediction_mean, target, idt, weight)   # metrics on the MEAN
    elif handle.type == LossType.Probabilistic:
        handle.update(prediction, target, idt, weight)        # metrics on the ENSEMBLE
```

⇒ **Stack the lagged members onto an ensemble axis and feed the 5-D tensor
straight in.** Deterministic metrics (RMSE/L1/ACC) are then computed on the
ensemble *mean*; probabilistic metrics (CRPS/spread/SSR/rank histogram) on the
*full ensemble*. Same code path, same **equiangular** quadrature that produced
today's curve.

This is a significant saving: no bespoke CRPS or rank-histogram implementation,
and no risk of our scoring disagreeing with the training-side scoring.

⚠ Note this also sidesteps the Gauss-Legendre weighting defect in
`src/sfno_eval/metrics.py`, which is wrong for our equiangular grid
(port scope doc §2.1). Prefer makani's handler over the group scorer for E3SM.

### 4.3 What "does it work?" means, concretely

Four checks, in order of how much they tell us:

1. **Does the ensemble mean beat the best single member?**
   The basic value test. If `RMSE(mean of members) >= RMSE(shallowest member)`,
   the ensemble bought nothing and the weighting is wrong.
2. **Spread-skill ratio ≈ 1?**
   `SSR = spread / RMSE(ensemble mean)`. 1.0 is calibrated; **< 1 is
   under-dispersed**, i.e. over-confident. We *expect* < 1 (§4.5) — the number
   quantifies how badly.
3. **Rank histogram flat?**
   Flat = calibrated; U-shaped = under-dispersed; dome = over-dispersed; sloped
   = biased. ⚠ **Partly invalid here** — see §4.4.
4. **CRPS beats the deterministic baseline?**
   A proper scoring rule. If the ensemble's CRPS does not beat the single
   best-member RMSE-equivalent, the ensemble is not worth its cost.

### 4.4 The honest caveat about the statistics

CRPS and the rank histogram both assume members are **exchangeable** — draws
from one predictive distribution. Our members are **not**: member at depth 4 is
systematically better than member at depth 56.

⇒ A rank histogram of an unweighted lagged ensemble will be U-shaped or sloped
**by construction**, and reading that as "under-dispersed" would be a mistake.
Two mitigations, both to be stated in any write-up:

- Use the **weighted** ensemble (`w_k ∝ 1/σ(k)²`) for CRPS/spread/SSR, not the
  raw one.
- Report the rank histogram as diagnostic only, or restrict it to a
  **fixed-depth sub-ensemble** where exchangeability is closer to true.

### 4.5 The failure mode this ensemble cannot fix

The lagged ensemble's spread measures **sensitivity to initial condition and
accumulated rollout error**. Every member shares one set of weights, so it does
**not** sample model/epistemic uncertainty.

If the model is mode-averaging — the MSE-optimal hedge, and the reason
MSE-trained video models blur — every member is blurred the same way: the mean
stays blurred and SSR is systematically < 1 no matter how the members are
chosen. **No ensembling of deterministic rollouts repairs this.**

Whether that is happening is read off the error-vs-depth curve. Current status,
from job 7603089 (base checkpoint, 21 leads, 101 channels):

- error grows **4.65×** from 6 h to 126 h, **linearly after the first day**
  (≈0.17 per 6 h step), with **no saturation**; ACC falls 1.000 → 0.878
- ⇒ the "error barely moves" outcome is **decisively excluded** — compounding
  is real, so the rollout/ensemble direction is not misaimed
- ⚠ but exposure bias and mode-averaging are **not yet separable**: they
  diverge only as error approaches the climatological level, and at ACC 0.878
  we are far from it. **126 h is too short.**

⇒ **Run the discriminating measurement at K=56 (14 days), which the port
delivers anyway.** Do not build the combination and calibration layers (stages
4-5) until that curve is read.

---

## 5. Ordered task list

**ALL TASKS CLOSED as of 2026-09-21.** Result → `2026-09-21_lagged_ensemble_result.md`.

| # | task | depends on | status |
|---|---|---|---|
| 1 | Per-lead metrics on our channel names | — | ✅ `652e9505`, job 7602739 |
| 2 | Re-score base at K=20 | 1 | ✅ 7603089 |
| 3 | Re-score C1 at K=20, compare at fixed depth | 1 | ✅ 7603119 (−3.00 % at 126 h) |
| 4 | Port A — derive channel contract from config | — | ✅ `f857040b` |
| 5 | Port B — `out_chans` assert | — | ✅ `f857040b` |
| 6 | Port C — `_load_run_norm_stats` n_out | — | ✅ `f857040b` (was silent) |
| 7 | Port D — `_extract_truth_sic` name lookup (bug) | — | ✅ `f857040b` (was silent — returned `solin`) |
| 8 | Port F — Polaris PBS sibling for the eval chain | 4-7 | ✅ `26192b3c` + `65f5e405`, verified 7632679 |
| 9 | Decision E — step-index vs calendar labelling | — | ✅ step-index (pack is noleap) |
| 10 | **K=56 sweep, read the curve to 14 days** | 4-9 | ✅ 7633207 + 7639537 → `2026-09-20_k56_readout_prereg.md` |
| 11 | Stage 1 — stagger-`d` start generator | 10 | ✅ `d7a49780` (`sfno_ensemble.starts`) |
| 12 | Stage 3 — member alignment by absolute target index | 11 | ✅ `d7a49780`, verified on real output 7643103 |
| 13 | Stage 4 — weighted combination, `w_k ∝ 1/σ(k)²` | 12, 2 | ✅ `sfno_ensemble.combine` |
| 14 | Stage 5 — CRPS / spread / SSR / RH | 13 | ✅ `sfno_ensemble.scores` — ⚠ **not** `MetricsHandler`; see below |
| 15 | Fix Gauss-Legendre weights in `sfno_eval/metrics.py` for equiangular grids | — | ✅ change G — ⚠ but `scripts/score_nwp.py` still calls `legendre_gauss_lat_weights` directly and is a live silent defect for E3SM |

⚠ **§4.2's "reuse `MetricsHandler`, do not rewrite" did not survive contact.** Two
reasons, both from reading makani rather than its docs: (a) `_crps_skillspread_kernel`
— the `skillspread` type this repo's CRPS config selects — **ignores its
`ensemble_weights` argument**, reducing over the ensemble with a plain `torch.mean`,
so for the weighted ensemble §4.4 requires there was nothing to reuse and the obvious
constructor would have silently returned an unweighted number; (b) the `Geometric*`
wrappers call `comm.get_size("ensemble")` and so need makani's process groups. The
*definition* is preserved instead: `test_crps_matches_makani_unweighted` pins our
uniform-weight CRPS against makani's own kernel, and ran green in job 7643249.

---

## 6. Open risks

> **Resolved 2026-09-21 — see `2026-09-21_lagged_ensemble_result.md`.** Risk 1 did not
> materialise in the form written (the 14-day curve did not saturate, and it ruled
> mode-averaging *out*), but the ensemble failed for a different reason: its mean is
> **31.7 % worse** than the single freshest member, monotonically in ensemble size, and
> 101 of 101 channels prefer one member to any prefix. Risk 2 proved to understate the
> problem — the members are non-exchangeable *and* the resulting ensemble is
> **over**-dispersed (SSR 1.66), the opposite of §4.5's prediction. Risk 3 stands: n = 1,
> 8 targets, one year.

1. **Task 10 may kill tasks 11-14.** If the 14-day curve saturates, the lagged
   ensemble is treating a symptom. Read it first.
2. **Non-exchangeable members** limit what CRPS and the rank histogram mean
   (§4.4). State it in any write-up rather than reporting the numbers bare.
3. **Every number in this track is n=1.** No repeats, one validation split.
4. `src/sfno_inference/` is inside a `git subtree` **and** shared with the
   Stampede3 `eval-sfno-own` path — every fix must generalize (CLAUDE.md #5).
